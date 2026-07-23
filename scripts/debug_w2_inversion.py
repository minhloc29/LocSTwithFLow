#!/usr/bin/env python3
"""
debug_w2_inversion.py — Instrument the OT-sketch loop to log per-round W2
against a FIXED reference set, and compare final W2 across all three
sketchers using the SAME reference.

This lets us distinguish three possible causes of the inversion:
  A) The NN-displacement proxy doesn't correlate with W2
  B) wasserstein_diag.py uses different reference samples per method
  C) The swaps genuinely increase W2 (real but counterintuitive effect)
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import scanpy as sc

from sketchflow.sketching import UniformSketcher, LeverageSketcher, OTSketcher
from sketchflow.ot_utils.sinkhorn import sinkhorn_cost

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ---- Load data ----
adata = sc.read("data/processed/pbmc3k_train.h5ad")
X = adata.obsm["X_pca"].astype(np.float32)
X_t = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
n, d = X.shape
k = 500

# ---- CREATE ONE FIXED REFERENCE SET (used for ALL evaluations) ----
np.random.seed(42)
ref_size = 5000
ref_idx_fixed = np.random.choice(n, size=min(ref_size, n), replace=False)
ref_pts_fixed = X_t[ref_idx_fixed]

def w2_vs_fixed(idx):
    """W2 between sketch (given by indices) and the fixed reference."""
    return sinkhorn_cost(X_t[idx], ref_pts_fixed, blur=0.05, backend="tensorized").item()

# =====================================================================
# 1. Uniform — baseline
# =====================================================================
print("\n" + "=" * 60)
print("  UNIFORM")
print("=" * 60)
u = UniformSketcher()
u.fit(X, k)
idx_u = u.get_indices()
w2_u = w2_vs_fixed(idx_u)
print(f"  W2(uniform, fixed_ref) = {w2_u:.4f}")

# =====================================================================
# 2. Leverage — warm-start
# =====================================================================
print("\n" + "=" * 60)
print("  LEVERAGE (warm-start for OT)")
print("=" * 60)
lev = LeverageSketcher()
lev.fit(X, k)
idx_lev = lev.get_indices().copy()
w2_lev = w2_vs_fixed(idx_lev)
print(f"  W2(leverage, fixed_ref) = {w2_lev:.4f}")

# =====================================================================
# 3. OT-sketch — instrumented loop
# =====================================================================
print("\n" + "=" * 60)
print("  OT-SKETCH (instrumented)")
print("=" * 60)

n_rounds = 30
candidate_batch_size = min(10_000, n)
blur = 0.05

# Warm-start from leverage
sketch_idx = idx_lev.copy()

# Log per-round W2
w2_history = [w2_vs_fixed(sketch_idx)]
print(f"  round   -1 (warm-start): W2 = {w2_history[0]:.4f}")

for round_i in range(n_rounds):
    sketch_pts = X_t[sketch_idx]
    batch_size = min(candidate_batch_size, n)
    batch_idx = np.random.choice(n, size=batch_size, replace=False)
    batch_pts = X_t[batch_idx]

    # --- Swap criterion (NN displacement proxy) ---
    dists = torch.cdist(batch_pts, sketch_pts)          # (batch, k)
    nearest_dist, _ = dists.min(dim=1)                   # (batch,)
    worst_candidate = batch_idx[nearest_dist.argmax().item()]

    sketch_dists = torch.cdist(sketch_pts, sketch_pts)  # (k, k)
    redundancy = sketch_dists.sum(dim=1) / (k - 1)
    most_redundant_pos = redundancy.argmin().item()

    # --- Measure W2 BEFORE the swap ---
    w2_before = w2_vs_fixed(sketch_idx)

    # --- Perform swap ---
    sketch_idx[most_redundant_pos] = worst_candidate

    # --- Measure W2 AFTER the swap ---
    w2_after = w2_vs_fixed(sketch_idx)
    w2_history.append(w2_after)

    # Also compute: does the proxy agree with W2 movement?
    # NN displacement = nearest_dist.max() (the worst_candidate's distance to sketch)
    # Redundancy score of the replaced point
    proxy_candidate_dist = nearest_dist.max().item()
    proxy_redundancy_val = redundancy[most_redundant_pos].item()

    if round_i == 0 or round_i % 5 == 0 or round_i == n_rounds - 1:
        delta = w2_after - w2_before
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"  round {round_i:3d}: W2_before={w2_before:.4f}  "
              f"W2_after={w2_after:.4f}  Δ={delta:+.4f} {arrow}  "
              f"proxy_cand_dist={proxy_candidate_dist:.4f}  "
              f"proxy_redundancy={proxy_redundancy_val:.4f}")

# =====================================================================
# 4. Final comparison against FIXED reference
# =====================================================================
print("\n" + "=" * 60)
print("  FINAL COMPARISON (fixed reference)")
print("=" * 60)

idx_ot = sketch_idx
w2_ot = w2_vs_fixed(idx_ot)

# Also re-evaluate uniform and leverage with the SAME call
print(f"  Uniform:  W2 = {w2_u:.4f}")
print(f"  Leverage: W2 = {w2_lev:.4f}")
print(f"  OT-sketch final round {n_rounds}: W2 = {w2_ot:.4f}")

# Check ordering
if w2_u <= w2_lev <= w2_ot:
    print("\n  ⚠️  W2 ordering still inverted (uniform < leverage < OT)")
elif w2_ot < w2_lev:
    print("\n  ✓ OT-sketch improved over leverage warm-start")
else:
    print("\n  Mixed — need to examine the per-round trace")

# Did OT-sketch W2 ever decrease from warm-start?
initial_w2 = w2_history[0]
final_w2 = w2_history[-1]
best_w2 = min(w2_history)
print(f"\n  W2 trace: start={initial_w2:.4f}  best={best_w2:.4f}  final={final_w2:.4f}")
print(f"  Net change from start: {final_w2 - initial_w2:+.4f}")
print(f"  Best improvement: {initial_w2 - best_w2:.4f}")

# Is the proxy correlated with W2 deltas?
# We can't fully test this in this run without logging every round,
# but we have every-5-round data. Let's check round 0-1 delta.
print("\n  Diagnostics:")
print(f"  The W2 diagnostic in wasserstein_diag.py uses a RANDOM reference")
print(f"  sample each call — different across the three sketcher eval calls.")
print(f"  This alone can explain the inversion.")
print(f"  The logged per-round W2 against a FIXED reference tells the real story.")

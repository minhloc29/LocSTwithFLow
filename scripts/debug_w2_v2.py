#!/usr/bin/env python3
"""
debug_w2_v2.py — Fix the swap criterion using RESPONSIBILITY instead of
mean-distance redundancy, and verify it drives W2 down.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import scanpy as sc

from sketchflow.sketching import UniformSketcher, LeverageSketcher
from sketchflow.ot_utils.sinkhorn import sinkhorn_cost

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

adata = sc.read("data/processed/pbmc3k_train.h5ad")
X = adata.obsm["X_pca"].astype(np.float32)
X_t = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
n, d = X.shape
k = 500

# Fixed reference for consistent W2 comparison
np.random.seed(42)
ref_idx = np.random.choice(n, size=min(5000, n), replace=False)
ref_pts = X_t[ref_idx]

def w2_vs_ref(idx):
    return sinkhorn_cost(X_t[idx], ref_pts, blur=0.05, backend="tensorized").item()

# =====================================================================
#  Warm-start from leverage
# =====================================================================
lev = LeverageSketcher()
lev.fit(X, k)
sketch_idx = lev.get_indices().copy()
warm_w2 = w2_vs_ref(sketch_idx)
print(f"\nWarm-start (leverage) W2 = {warm_w2:.4f}")

# =====================================================================
#  Responsibility-based OT sketching
# =====================================================================
n_rounds = 30
candidate_batch_size = min(10_000, n)

# Precompute responsibilities using a fixed subsample of the full data
# to make it O(n_ref * k) per round — cheap.
responsibility_ref_size = min(5000, n)
resp_ref_idx = np.random.choice(n, size=responsibility_ref_size, replace=False)
resp_ref_pts = X_t[resp_ref_idx]

w2_history = [warm_w2]

for round_i in range(n_rounds):
    sketch_pts = X_t[sketch_idx]  # (k, d)

    # --- 1. Pick worst-covered candidate (same as before) ---
    batch_size = min(candidate_batch_size, n)
    batch_idx = np.random.choice(n, size=batch_size, replace=False)
    batch_pts = X_t[batch_idx]

    dists = torch.cdist(batch_pts, sketch_pts)          # (batch, k)
    nearest_dist, nearest_sketch = dists.min(dim=1)      # (batch,)
    worst_candidate = batch_idx[nearest_dist.argmax().item()]

    # --- 2. Compute responsibility for each sketch point ---
    # How many reference points have this sketch point as nearest neighbor?
    resp_dists = torch.cdist(resp_ref_pts, sketch_pts)   # (n_ref, k)
    resp_nn = resp_dists.argmin(dim=1)                   # (n_ref,)
    responsibility = torch.zeros(k, device=DEVICE)
    responsibility.scatter_add_(0, resp_nn, torch.ones_like(resp_nn, dtype=torch.float))
    # responsibility[i] = how many reference points are "represented" by sketch point i
    # Lower = more replaceable

    # Pick the sketch point with LOWEST responsibility (fewest dependents)
    most_redundant_pos = responsibility.argmin().item()

    # --- 3. Measure W2 before ---
    w2_before = w2_vs_ref(sketch_idx)

    # --- 4. Swap ---
    sketch_idx[most_redundant_pos] = worst_candidate

    # --- 5. Measure W2 after ---
    w2_after = w2_vs_ref(sketch_idx)
    w2_history.append(w2_after)

    delta = w2_after - w2_before
    if round_i == 0 or round_i % 5 == 0 or round_i == n_rounds - 1:
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"  round {round_i:3d}: W2={w2_after:.4f}  Δ={delta:+.4f} {arrow}  "
              f"removed_resp={int(responsibility[most_redundant_pos].item())} "
              f"(of {responsibility_ref_size})")

# =====================================================================
#  Summary
# =====================================================================
print(f"\n  W2: start={w2_history[0]:.4f}  final={w2_history[-1]:.4f}  "
      f"Δ={w2_history[-1] - w2_history[0]:+.4f}")

# Compare with uniform baseline
u = UniformSketcher()
u.fit(X, k)
w2_u = w2_vs_ref(u.get_indices())
print(f"\n  Uniform:  W2 = {w2_u:.4f}")
print(f"  Leverage: W2 = {warm_w2:.4f}  (warm-start)")
print(f"  OT (resp):W2 = {w2_history[-1]:.4f}  (final)")

if w2_history[-1] < warm_w2:
    print("\n  ✓ Responsibility-based swap improved over leverage warm-start")
else:
    print("\n  ⚠️  Still not improving — need to investigate further")

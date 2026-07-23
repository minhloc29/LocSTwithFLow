#!/usr/bin/env python3
"""
debug_w2_v3.py — Fix the swap criterion.

Root cause analysis of the W2 inversion:
  The NN-displacement proxy selects the FURTHEST point from the sketch
  as the candidate to ADD. At high k/n (~24%), these "furthest" points
  are all periphery/outliers. Adding outliers EXPANDS the sketch's
  convex hull, which INCREASES W2 to the full data distribution.

  The "mean-redundancy" removal selects the DENSEST-REGION point for
  removal, compounding the problem (lose interior → stretch boundary).

Fix: change the ADDITION criterion to pick the candidate whose addition
     REDUCES W2 the most. Since we can't afford full Sinkhorn per
     candidate, we use a cheap proxy:
       For each candidate, compute its distance to the FULL dataset
       (not to the sketch). Add candidates close to the data center.
     This is the OPPOSITE of "furthest from sketch" — we want points
     representative of the full data, not outliers.

  Actually even simpler: the right proxy is to add the point that is
  CLOSEST to its nearest sketch point (already well-covered) — no,
  that does nothing. We need to add points that are representative
  of under-covered regions, but not outliers.

  ACTUAL FIX: Instead of adding the furthest-out candidate, we find
  a batch of candidates, and for each compute how much W2 would
  change if we swapped it in. We pick the ONE swap that reduces W2.
  This is O(candidate_batch * Sinkhorn) per round — manageable if
  candidate_batch is small (e.g., 100).
"""

import os, sys, warnings, time
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

# Fixed reference
np.random.seed(42)
ref = np.random.choice(n, size=min(5000, n), replace=False)
ref_pts = X_t[ref]

def w2_vs_ref(idx):
    return sinkhorn_cost(X_t[idx], ref_pts, blur=0.05, backend="tensorized").item()

# Warm-start from leverage
lev = LeverageSketcher()
lev.fit(X, k)
sketch_idx = lev.get_indices().copy()
warm_w2 = w2_vs_ref(sketch_idx)
print(f"Warm-start (leverage) W2 = {warm_w2:.4f}")

# Uniform baseline
u = UniformSketcher()
u.fit(X, k)
w2_u = w2_vs_ref(u.get_indices())
print(f"Uniform W2 = {w2_u:.4f}")

# =====================================================================
#  OT-sketch with explicit swap-evaluation
# =====================================================================
n_rounds = 30
eval_candidates = 100  # candidates to try per round — small enough for Sinkhorn

w2_history = [warm_w2]

for round_i in range(n_rounds):
    sketch_pts = X_t[sketch_idx]

    # Sample a batch of candidates
    batch_size = min(eval_candidates + 50, n)
    batch_idx = np.random.choice(n, size=batch_size, replace=False)
    batch_pts = X_t[batch_idx]

    # Compute NN displacement proxy to rank candidates
    dists = torch.cdist(batch_pts, sketch_pts)           # (batch, k)
    nearest_dist, _ = dists.min(dim=1)                    # (batch,)
    nearest_dist_np = nearest_dist.cpu().numpy()

    # Find the top candidates by displacement (worst-covered)
    # These are the ones most likely to improve coverage
    top_k_candidates = batch_idx[np.argsort(-nearest_dist_np)[:eval_candidates]]

    # For each candidate, try swapping it for the most redundant sketch point
    # (by responsibility), measure W2, keep the best swap.

    # Compute sketch point responsibilities
    resp_dists = torch.cdist(ref_pts, sketch_pts)        # (n_ref, k)
    resp_nn = resp_dists.argmin(dim=1)
    resp = torch.zeros(k, device=DEVICE)
    resp.scatter_add_(0, resp_nn, torch.ones_like(resp_nn, dtype=torch.float))

    # We'll try replacing the LEAST-responsible sketch point
    # (responsibility argmin = most replaceable)
    # AND the MOST-responsible point (argmax = most central, potentially redundant for W2)

    best_w2 = w2_vs_ref(sketch_idx)
    best_i = -1
    best_j = -1

    # Try replacing each of the top 20% least-responsible points
    replace_candidates = torch.argsort(resp)[:k // 5].cpu().numpy()
    # Also try the single most-responsible
    replace_candidates = np.unique(np.concatenate([replace_candidates, [resp.argmax().item()]]))

    for j in replace_candidates:
        for i in top_k_candidates:
            test_idx = sketch_idx.copy()
            test_idx[j] = i
            w = w2_vs_ref(test_idx)
            if w < best_w2:
                best_w2 = w
                best_i = i
                best_j = j
                # Early exit: if this is good enough, stop
                if best_w2 < warm_w2 * 0.99:
                    break
        if best_w2 < warm_w2 * 0.99:
            break

    if best_i >= 0:
        sketch_idx[best_j] = best_i
        delta = best_w2 - w2_history[-1]
        print(f"  round {round_i:3d}: W2={best_w2:.4f}  Δ={delta:+.4f}  "
              f"replaced_pos={best_j}  (evaluated {len(top_k_candidates)} x {len(replace_candidates)})")
    else:
        # No improving swap found — try replacing the MOST-responsible point
        # (removing a highly-covered-internal point and adding the furthest-out candidate)
        # Actually skip — we already tried
        print(f"  round {round_i:3d}: No improving swap found — stopping")
        break

    w2_history.append(best_w2)

# Summary
print(f"\n  W2: start={w2_history[0]:.4f}  final={w2_history[-1]:.4f}")
print(f"  Uniform:  {w2_u:.4f}")
print(f"  Leverage: {warm_w2:.4f}")
print(f"  OT-sketch:{w2_history[-1]:.4f}")

if w2_history[-1] < warm_w2:
    print("  ✓ OT improved over leverage warm-start")
else:
    print("  ⚠️  OT did not improve")

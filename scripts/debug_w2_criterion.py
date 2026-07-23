#!/usr/bin/env python3
"""
debug_w2_criterion.py — Test alternative swap criteria for W2 improvement.

Current (broken): remove most "redundant" (closest internally), add furthest.
  → Removes interior, adds outlier. W2 ↑ every round.

Fix attempts:
  A) remove most ISOLATED (furthest intra-sketch), add furthest.
     → Swaps one boundary point for another. Sketch boundary shifts.
  B) remove most redundant, add from RANDOM subset of candidates (not furthest).
     → Controls outlier seeking.
  C) Best-of-k: try several candidates for each position, keep best by W2.
     → Ground-truth check.
"""

import os
import sys
import warnings
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

np.random.seed(42)
ref_pts = X_t[np.random.choice(n, size=min(5000, n), replace=False)]
def w2(idx): return sinkhorn_cost(X_t[idx], ref_pts, blur=0.05, backend="tensorized").item()

# Warm-start
lev = LeverageSketcher()
lev.fit(X, k)
idx_lev = lev.get_indices().copy()
w2_lev = w2(idx_lev)
print(f"Leverage W2 = {w2_lev:.4f}")

# Baseline
u = UniformSketcher()
u.fit(X, k)
print(f"Uniform  W2 = {w2(u.get_indices()):.4f}")

# Precompute: for each of the n data points, estimate local density
# as the distance to its 10th nearest neighbor (smaller = denser)
print("Precomputing local density estimates...")
all_dists = torch.cdist(X_t, X_t)
sorted_dists, _ = all_dists.sort(dim=1)
knn_10_dist = sorted_dists[:, 10]  # (n,) — distance to 10th NN, smaller = denser
print(f"  Density range: {knn_10_dist.min().item():.4f} to {knn_10_dist.max().item():.4f}")


def run_ot(name, make_candidate_fn, make_remove_fn, n_rounds=30):
    """Generic OT sketcher with pluggable candidate and removal policies."""
    sketch = idx_lev.copy()
    w2s = [w2_lev]
    for r in range(n_rounds):
        sketch_pts = X_t[sketch]
        batch_size = min(2000, n)
        batch_idx = np.random.choice(n, size=batch_size, replace=False)
        batch_pts = X_t[batch_idx]

        # Pick candidate to ADD
        cand_idx = make_candidate_fn(sketch, sketch_pts, batch_idx, batch_pts)

        # Pick position to REMOVE
        remove_pos = make_remove_fn(sketch, sketch_pts)

        # Swap
        sketch[remove_pos] = cand_idx
        w2s.append(w2(sketch))

        delta = w2s[-1] - w2s[-2]
        if r == 0 or r == n_rounds - 1 or r % 10 == 0:
            print(f"  {name} round {r:3d}: W2={w2s[-1]:.4f}  Δ={delta:+.4f}")
    return w2s


# ---- A) Original (baseline) ----
def farthest_sketch_remove(sketch, sketch_pts):
    dists = torch.cdist(sketch_pts, sketch_pts)
    mean_d = dists.sum(dim=1) / (k - 1)
    return mean_d.argmin().item()  # most redundant (internally closest)

def max_nn_candidate(sketch, sketch_pts, batch_idx, batch_pts):
    dists = torch.cdist(batch_pts, sketch_pts)
    nn_dist, _ = dists.min(dim=1)
    return batch_idx[nn_dist.argmax().item()]  # furthest from sketch

print("\n--- A) ORIGINAL: remove most redundant, add furthest ---")
w2s_A = run_ot("A", max_nn_candidate, farthest_sketch_remove)

# ---- B) Density-weighted candidate: nn_dist / local_density ---
def density_weighted_candidate(sketch, sketch_pts, batch_idx, batch_pts):
    dists = torch.cdist(batch_pts, sketch_pts)
    nn_dist, _ = dists.min(dim=1)  # (batch,) — distance from each candidate to nearest sketch point
    # Score = nn_dist / local_density
    # Points in sparse regions get divided by large knn_10_dist (low density) → moderate score
    # Points in dense regions get divided by small knn_10_dist (high density) → amplified score
    # This prefers, among candidates equidistant from the sketch, the one in a DENSER region

    # To avoid div by zero, use knn_10_dist as local density proxy
    batch_density = knn_10_dist[batch_idx]  # (batch,)
    score = nn_dist / (batch_density + 0.001)
    return batch_idx[score.argmax().item()]

print("\n--- B) Density-weighted candidate ---")
w2s_B = run_ot("B", density_weighted_candidate, farthest_sketch_remove)

# ---- C) Original candidate, remove most ISOLATED (not redundant) ----
def most_isolated_remove(sketch, sketch_pts):
    dists = torch.cdist(sketch_pts, sketch_pts)
    mean_d = dists.sum(dim=1) / (k - 1)
    return mean_d.argmax().item()  # most isolated (internally furthest)

print("\n--- C) Remove most isolated, add furthest ---")
w2s_C = run_ot("C", max_nn_candidate, most_isolated_remove)

# ---- D) Density-weighted + remove most isolated ---
print("\n--- D) Density-weighted candidate + remove most isolated ---")
w2s_D = run_ot("D", density_weighted_candidate, most_isolated_remove)

print("\n" + "=" * 60)
print("  Summary")
print("=" * 60)
print(f"  Uniform baseline:     {w2(u.get_indices()):.4f}")
print(f"  Leverage warm-start:  {w2_lev:.4f}")
print(f"  A (orig, red+far):    {w2s_A[-1]:.4f}  Δ={w2s_A[-1]-w2_lev:+.4f}")
print(f"  B (density+red):      {w2s_B[-1]:.4f}  Δ={w2s_B[-1]-w2_lev:+.4f}")
print(f"  C (orig+iso):         {w2s_C[-1]:.4f}  Δ={w2s_C[-1]-w2_lev:+.4f}")
print(f"  D (density+iso):      {w2s_D[-1]:.4f}  Δ={w2s_D[-1]-w2_lev:+.4f}")

#!/usr/bin/env python3
"""
diagnose_leverage_vs_fulldata.py — Investigate why leverage-sketch models
outperform full-data models on rare-population recall.

Hypotheses tested:
  1. Sketch composition bias — leverage over-samples rare cells, so the
     training set has more rare points per batch.
  2. Easier optimization — smaller, more balanced dataset converges
     faster for minority modes.
  3. Majority overfitting — full-data model overfits to majority classes
     at the cost of minority recall.
  4. Gradient noise — larger batch size in full-data changes dynamics.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import scanpy as sc
import torch

from sketchflow.sketching import UniformSketcher, LeverageSketcher, OTSketcher
from sketchflow.models.flow_matching import OTCFMTrainer
from sketchflow.evaluation.mmd import mmd_rbf
from sketchflow.evaluation.rare_population import rare_population_recall

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Load data
train_adata = sc.read("data/processed/pbmc3k_train.h5ad")
test_adata = sc.read("data/processed/pbmc3k_test.h5ad")
X_train = train_adata.obsm["X_pca"].astype(np.float32)
X_test = test_adata.obsm["X_pca"].astype(np.float32)

# Get labels
label_key = "louvain" if "louvain" in test_adata.obs else None
train_labels = train_adata.obs[label_key].values if label_key else None
test_labels = test_adata.obs[label_key].values if label_key else None

# Identify rare label
unique, counts = np.unique(test_labels, return_counts=True)
rare_label = unique[np.argmin(counts)]
print(f"\nRare label: '{rare_label}' ({counts.min()}/{X_test.shape[0]} test cells)")

# =====================================================================
# 1. Sketch composition analysis
# =====================================================================
print("\n" + "=" * 60)
print("  1. SKETCH COMPOSITION — rare-cell representation")
print("=" * 60)

k = 500

# Get train labels for composition analysis
train_unique, train_counts = np.unique(train_labels, return_counts=True)
print(f"\n  Full training set: {len(train_labels)} cells")
print(f"  Rare cells in train: {sum(train_labels == rare_label)} / {len(train_labels)} "
      f"({sum(train_labels == rare_label)/len(train_labels)*100:.1f}%)")

for name, SketcherClass in [("uniform", UniformSketcher),
                              ("leverage", LeverageSketcher),
                              ("ot_sketch", OTSketcher)]:
    s = SketcherClass() if name != "ot_sketch" else OTSketcher(use_density_weighting=False)
    s.fit(X_train, k)
    idx = s.get_indices()
    sketch_labels = train_labels[idx]
    rare_in_sketch = sum(sketch_labels == rare_label)
    print(f"\n  {name:12s}: k={k}")
    print(f"    Rare cells in sketch: {rare_in_sketch} / {k} "
          f"({rare_in_sketch/k*100:.1f}%)")
    print(f"    Enrichment vs full data: {rare_in_sketch/k / (sum(train_labels==rare_label)/len(train_labels)):.1f}x")

# =====================================================================
# 2. Load trained models and measure per-class generation
# =====================================================================
print("\n" + "=" * 60)
print("  2. PER-CLASS GENERATION FIDELITY")
print("=" * 60)

device = DEVICE
n_gen = 500

def load_trainer(sketcher_name, seed):
    ckpt = torch.load(f"checkpoints/{sketcher_name}_seed{seed}.pt",
                      map_location=device, weights_only=False)
    n_genes = train_adata.raw.shape[1] if train_adata.raw is not None else train_adata.n_vars
    dim = X_train.shape[1]
    trainer = OTCFMTrainer(dim=dim, n_genes=n_genes, device=device)
    trainer.vector_field.load_state_dict(ckpt["vector_field"])
    trainer.zinb_head.load_state_dict(ckpt["zinb_head"])
    return trainer

# For each sketcher, generate samples across seeds
X_test_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)

results = {}
for name in ["full_data", "uniform", "leverage", "ot_sketch"]:
    all_gen = []
    for seed in [0, 1, 2]:
        trainer = load_trainer(name, seed)
        gen = trainer.generate(n_gen, X_train.shape[1], n_steps=50).cpu().numpy()
        all_gen.append(gen)

    # MMD to test set
    gen_all = np.concatenate(all_gen, axis=0)
    gen_t = torch.as_tensor(gen_all, dtype=torch.float32, device=device)
    mmd = mmd_rbf(gen_t, X_test_t)

    # Rare recall (average across seeds)
    recalls = []
    for gen in all_gen:
        r = rare_population_recall(gen, X_test, test_labels, rare_label)
        recalls.append(r)

    # Per-class: NN label transfer to see what each model generates
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(X_test)
    _, indices = nn.kneighbors(gen_all)
    gen_labels = test_labels[indices.flatten()]
    gen_unique, gen_counts = np.unique(gen_labels, return_counts=True)
    gen_frac = dict(zip(gen_unique, gen_counts / len(gen_labels)))

    results[name] = {
        "mmd": mmd,
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls),
        "gen_label_distribution": gen_frac,
        "gen_labels": gen_labels,
    }

    print(f"\n  {name:12s}: MMD={mmd:.4f}, "
          f"rare_recall={np.mean(recalls):.3f}±{np.std(recalls):.3f}")
    print(f"    Generated label distribution (via 1-NN):")
    for lbl, frac in sorted(gen_frac.items(), key=lambda x: -x[1]):
        test_frac = sum(test_labels == lbl) / len(test_labels)
        marker = " ← RARE" if lbl == rare_label else ""
        print(f"      {str(lbl):20s}: {frac*100:5.1f}% (test: {test_frac*100:.1f}%){marker}")

# =====================================================================
# 3. Chi-squared divergence of generated distribution from test distribution
# =====================================================================
print("\n" + "=" * 60)
print("  3. DISTRIBUTION DIVERGENCE — generated vs test labels")
print("=" * 60)

test_fracs = dict(zip(*np.unique(test_labels, return_counts=True)))
total_test = sum(test_fracs.values())
test_fracs = {k: v/total_test for k, v in test_fracs.items()}

for name, res in results.items():
    gen_frac = res["gen_label_distribution"]
    all_labels = set(list(test_fracs.keys()) + list(gen_frac.keys()))

    # KL divergence (test || gen) — measure how much gen distribution deviates
    kl = 0
    chi2 = 0
    for lbl in all_labels:
        p = test_fracs.get(lbl, 1e-10)
        q = gen_frac.get(lbl, 1e-10)
        kl += p * np.log2(max(p / q, 1e-10))
        chi2 += (p - q) ** 2 / max(p + q, 1e-10)

    print(f"  {name:12s}: KL(test||gen)={kl:.4f} bits, chi2={chi2:.4f}")

    # Which classes are over/under-generated?
    diffs = []
    for lbl in all_labels:
        p = test_fracs.get(lbl, 0)
        q = gen_frac.get(lbl, 0)
        diffs.append((q - p, lbl))

    for diff, lbl in sorted(diffs, key=lambda x: -abs(x[0]))[:4]:
        marker = " ← UNDER" if diff < -0.025 else (" ← OVER" if diff > 0.025 else "")
        print(f"      {str(lbl):20s}: test={test_fracs.get(lbl,0)*100:.1f}% "
              f"gen={gen_frac.get(lbl,0)*100:.1f}% "
              f"Δ={diff*100:+.1f}pp{marker}")

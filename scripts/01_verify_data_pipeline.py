#!/usr/bin/env python3
"""
Phase 1 Verification Script

Runs the full data pipeline end-to-end:
    1. Load a dataset
    2. Preprocess (normalize → HVG → PCA)
    3. Train/validation split
    4. Visualize PCA + UMAP to confirm single-cell structure

Usage:
    python scripts/01_verify_data_pipeline.py [--dataset pbmc3k] [--seed 42]

This is the Phase 1 checkpoint — everything should run cleanly before
moving on to Phase 2 (sketching).
"""

import argparse
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from sketchflow.data.loading import load_dataset, list_available_datasets
from sketchflow.data.preprocessing import preprocess_adata
from sketchflow.data.splits import train_val_split
from sketchflow.utils.seed import set_random_seed


def main(args):
    set_random_seed(args.seed)
    print("=" * 60)
    print("SketchFlow — Phase 1 Data Pipeline Verification")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"\n[1/4] Loading dataset: '{args.dataset}'")
    try:
        adata = load_dataset(args.dataset)
    except ValueError as e:
        print(f"  Available datasets: {list(list_available_datasets().keys())}")
        raise

    print(f"  Shape: {adata.shape}")
    print(f"  Cell types available: {'louvain' in adata.obs or 'cell_type' in adata.obs}")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print(f"\n[2/4] Preprocessing (n_hvgs={args.n_hvgs}, n_pcs={args.n_pcs})")
    adata = preprocess_adata(
        adata,
        n_hvgs=args.n_hvgs,
        n_pcs=args.n_pcs,
        compute_umap=True,
        random_state=args.seed,
    )

    pca_key = "X_pca"
    print(f"  PCA shape: {adata.obsm[pca_key].shape}")
    print(f"  Raw counts in .raw: {adata.raw is not None}")
    print(f"  UMAP computed: {'X_umap' in adata.obsm}")

    # ------------------------------------------------------------------
    # 3. Train/validation split
    # ------------------------------------------------------------------
    print(f"\n[3/4] Train/validation split (val_frac={args.val_frac})")
    train_adata, val_adata = train_val_split(
        adata,
        val_frac=args.val_frac,
        stratify_key=args.stratify_key,
        random_state=args.seed,
    )
    print(f"  Train: {train_adata.n_obs} cells")
    print(f"  Val:   {val_adata.n_obs} cells")
    print(f"  Val fraction: {val_adata.n_obs / adata.n_obs:.3f}")

    # ------------------------------------------------------------------
    # 4. Visualize
    # ------------------------------------------------------------------
    print(f"\n[4/4] Generating diagnostic plots ...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"SketchFlow — {args.dataset} Data Pipeline", fontsize=14, fontweight="bold")

    # Compute cluster labels for coloring (if not already present)
    if "leiden" not in adata.obs:
        import scanpy as sc
        sc.tl.leiden(adata, resolution=0.5, random_state=args.seed)
    cluster_key = "leiden"

    # 4a. PCA — full dataset
    ax = axes[0, 0]
    scatter = ax.scatter(
        adata.obsm["X_pca"][:, 0],
        adata.obsm["X_pca"][:, 1],
        c=adata.obs[cluster_key].cat.codes if cluster_key in adata.obs else "steelblue",
        s=3,
        cmap="viridis",
        alpha=0.6,
        rasterized=True,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA — Full Dataset (colored by cluster)")
    ax.legend(*scatter.legend_elements(), loc="upper right", fontsize=5)

    # 4b. PCA explained variance
    ax = axes[0, 1]
    var = adata.uns.get("pca_variance_ratio", np.ones(args.n_pcs))
    ax.bar(range(1, min(len(var) + 1, 21)), var[:20])
    ax.set_xlabel("PC")
    ax.set_ylabel("Variance ratio")
    ax.set_title("Top 20 PCs — Explained Variance")

    # 4c. Cumulative variance
    ax = axes[0, 2]
    cum_var = np.cumsum(var)
    ax.plot(range(1, len(var) + 1), cum_var, "b-", linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="80%")
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative variance")
    ax.set_title("Cumulative Explained Variance")
    ax.legend()

    # 4d. UMAP — full dataset
    ax = axes[1, 0]
    scatter = ax.scatter(
        adata.obsm["X_umap"][:, 0],
        adata.obsm["X_umap"][:, 1],
        c=adata.obs[cluster_key].cat.codes if cluster_key in adata.obs else "steelblue",
        s=3,
        cmap="viridis",
        alpha=0.7,
        rasterized=True,
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP — Full Dataset (colored by cluster)")
    ax.legend(*scatter.legend_elements(), loc="upper right", fontsize=5)

    # 4e. UMAP — train vs val overlay
    ax = axes[1, 1]
    train_mask = np.zeros(adata.n_obs, dtype=bool)
    train_idx = np.where(np.isin(adata.obs_names, train_adata.obs_names))[0]
    val_idx = np.where(np.isin(adata.obs_names, val_adata.obs_names))[0]
    ax.scatter(
        adata.obsm["X_umap"][train_idx, 0],
        adata.obsm["X_umap"][train_idx, 1],
        c="steelblue", s=3, alpha=0.4, label=f"Train ({len(train_idx)})", rasterized=True,
    )
    ax.scatter(
        adata.obsm["X_umap"][val_idx, 0],
        adata.obsm["X_umap"][val_idx, 1],
        c="coral", s=8, alpha=0.7, label=f"Val ({len(val_idx)})", rasterized=True,
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("Train/Val Split Overlay")
    ax.legend(fontsize=8)

    # 4f. Cluster size distribution
    ax = axes[1, 2]
    cluster_counts = adata.obs[cluster_key].value_counts().sort_index()
    ax.bar(range(len(cluster_counts)), cluster_counts.values)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cell count")
    ax.set_title(f"Cluster Size Distribution ({len(cluster_counts)} clusters)")
    ax.set_xticks(range(len(cluster_counts)))
    ax.set_xticklabels(cluster_counts.index, fontsize=6)

    plt.tight_layout()

    # Save figure
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, f"{args.dataset}_pipeline_validation.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved to: {plot_path}")

    # Also save a small stats summary
    summary_path = os.path.join(args.output_dir, f"{args.dataset}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"N_cells: {adata.n_obs}\n")
        f.write(f"N_genes (after HVG): {adata.n_vars}\n")
        f.write(f"N_PCs: {adata.obsm['X_pca'].shape[1]}\n")
        f.write(f"Train cells: {train_adata.n_obs}\n")
        f.write(f"Val cells: {val_adata.n_obs}\n")
        f.write(f"PC1 variance ratio: {var[0]:.4f}\n")
        f.write(f"Top-5 cumulative variance: {np.sum(var[:5]):.4f}\n")
        f.write(f"Total explained variance ({len(var)} PCs): {cum_var[-1]:.4f}\n")
        f.write(f"N_clusters: {len(cluster_counts)}\n")
        for cluster, count in cluster_counts.items():
            f.write(f"  Cluster {cluster}: {count} cells\n")
    print(f"  Summary saved to: {summary_path}")

    print("\n" + "=" * 60)
    print("Phase 1 verification complete! ✓")
    print("=" * 60)
    print(f"\nDataset: {adata.n_obs} cells, {adata.n_vars} genes, "
          f"{adata.obsm['X_pca'].shape[1]} PCs")
    print(f"Split: {train_adata.n_obs} train / {val_adata.n_obs} val")
    print(f"Plot: {plot_path}")
    print("\nReady for Phase 2 (sketching).")

    return adata, train_adata, val_adata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SketchFlow Phase 1 — Verify the data pipeline"
    )
    parser.add_argument(
        "--dataset", type=str, default="pbmc3k",
        help="Dataset name (pbmc3k, pbmc68k, pbmc_multimodal)"
    )
    parser.add_argument("--n_hvgs", type=int, default=2000, help="Number of HVGs")
    parser.add_argument("--n_pcs", type=int, default=50, help="Number of PCs")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--stratify_key", type=str, default=None,
                        help="Obs key for stratification (default: auto Leiden)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="sketchflow/outputs",
                        help="Output directory for plots and summaries")
    args = parser.parse_args()

    main(args)

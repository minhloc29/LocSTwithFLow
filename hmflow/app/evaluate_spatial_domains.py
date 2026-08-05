"""
evaluate_spatial_domains.py

Spatial domain recovery evaluation: does clustering PREDICTED gene expression
recover meaningful tissue organization, and does hierarchy (HFlow-ST) do this
better than a flat model (STFlow)?

Two reference-label modes, auto-selected per slide based on what's available
(see check_annotations.py to determine this ahead of time):

  MODE 1 (STRONG, primary-metric-worthy): if the slide's .h5ad has a real
      pathologist/region annotation column, cluster predicted expression and
      compute ARI/NMI directly against those labels. Independent ground truth
      -> this is the version worth leading with.

  MODE 2 (WEAKER, secondary/supporting): if no annotation column exists,
      cluster GROUND-TRUTH expression as a proxy reference, then compute
      ARI/NMI between predicted-expression clusters and ground-truth-expression
      clusters. Self-referential (correlated with PCC), so treat results from
      this mode as supporting evidence, not a standalone strong claim.

For BOTH modes: also cluster ground-truth expression against the SAME
annotation reference (mode 1) as a ceiling reference, matching the Nature
Communications finding that predicted-SGE clustering sometimes outperforms
ground-truth-SGE clustering — you want that exact comparison available.

K-SWEEP: results are reported across multiple cluster counts (not a single
K), since ARI/NMI are sensitive to this choice and a robust finding should
hold across a reasonable range.

Wiring note: model/config/sampling construction mirrors hmflow/app/flow/../../
tree's local_corruption.py (the proven-working inference harness in this repo),
including strict=False checkpoint loading (hierarchy-gate keys may be absent in
older checkpoints) and the Interpolant/normalize construction.

Usage
-----
    python hmflow/app/evaluate_spatial_domains.py \
        --source_dataroot <root> --dataset LUNG --split_id 0 \
        --checkpoint <root>/results_dir/LUNG/split0/checkpoints/100.pth \
        --representation slide_region_patch \
        --annotation_column <col> \
        --k_values 4 6 8 10 \
        --save_dir results/spatial_domains/LUNG_HFlowST_split0
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm
import torch

from hmflow.utils import set_random_seed
from hmflow.model.denoiser import HFlowDenoiser as Denoiser
from hmflow.model.hflow_config import HFlowConfig
from hmflow.model.config import ModelConfig
from hmflow.flow.interpolant import Interpolant
from hmflow.data.dataset import HESTDataset, HESTDatasetPath, padding_batcher
from hmflow.data.normalize_utils import get_normalize_method


# ── Reuse the exact sampling loop from local_corruption.py's run_sampling ──

@torch.no_grad()
def run_sampling(model, diffusier, img_features, coords, labels, args) -> np.ndarray:
    """Identical to local_corruption.py's run_sampling — kept in sync
    deliberately (including sample_from_prior(labels.shape, labels.device))."""
    model.eval()
    B = img_features.shape[0]
    assert B == 1, "Batch size must be 1 for inference"

    exp_t1 = diffusier.sample_from_prior(labels.shape, labels.device)
    ts = torch.linspace(0.01, 1.0, args.n_sample_steps)[:, None] \
            .expand(args.n_sample_steps, B).to(args.device)

    hierarchy_state = None
    pred = None
    for step_i, (t1, t2) in enumerate(zip(ts[:-1], ts[1:])):
        pred, hierarchy_state = model.inference(
            exp_t1, img_features, coords, t1, hierarchy_state=hierarchy_state,
        )
        if step_i == args.n_sample_steps - 2:
            break
        d_t = t2 - t1
        exp_t1 = diffusier.denoise(pred, exp_t1, t1, d_t)

    return pred.squeeze(0).cpu().numpy()


# ── Reference-label extraction ──────────────────────────────────────────────

def get_annotation_labels(h5ad_path: str, annotation_column: str) -> np.ndarray:
    """
    Loads the specified obs column as reference labels. Returns None if the
    column doesn't exist or is entirely null for this slide (some columns
    are only populated for a subset of slides — check_annotations.py's
    'coverage_pct' output tells you this per-slide ahead of time).
    """
    adata = ad.read_h5ad(h5ad_path, backed="r")
    if annotation_column not in adata.obs.columns:
        return None
    labels = adata.obs[annotation_column].values
    if pd.isna(labels).all():
        return None
    return labels


def cluster_expression(expression: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """
    KMeans clustering of an [N_spots, n_genes] expression matrix.

    NOTE: KMeans is used here for simplicity/speed and to avoid a scanpy/
    Leiden dependency in this evaluation script. If your paper's main text
    wants to match more standard ST-analysis practice, consider swapping
    this for scanpy.tl.leiden (graph-based, doesn't require pre-specifying
    K) as a robustness check — report both if you have time, since a result
    holding under both clustering algorithms is more convincing than one
    that only holds for KMeans specifically.
    """
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(expression)


# ── Per-slide evaluation ─────────────────────────────────────────────────────

def evaluate_one_slide(
    pred_expression: np.ndarray,
    gt_expression: np.ndarray,
    annotation_labels: np.ndarray,  # or None
    k_values: list,
    seed: int,
) -> list:
    """
    Returns a list of result rows, one per k value, containing:
        - mode ('annotation' if labels available, else 'gt_proxy')
        - ARI/NMI of predicted-expression clusters vs reference
        - ARI/NMI of ground-truth-expression clusters vs reference (ceiling
          comparison — replicates the Nature Comms "predicted sometimes
          beats ground-truth clustering" check)
    """
    rows = []
    has_annotation = annotation_labels is not None

    for k in k_values:
        pred_clusters = cluster_expression(pred_expression, k, seed)
        gt_clusters = cluster_expression(gt_expression, k, seed)

        if has_annotation:
            # MODE 1: independent ground truth
            reference = annotation_labels
            mode = "annotation"
        else:
            # MODE 2: ground-truth expression clustering as proxy reference
            reference = gt_clusters
            mode = "gt_proxy"

        ari_pred = adjusted_rand_score(reference, pred_clusters)
        nmi_pred = normalized_mutual_info_score(reference, pred_clusters)

        row = {
            "k": k, "mode": mode,
            "ari_predicted": ari_pred, "nmi_predicted": nmi_pred,
        }

        if has_annotation:
            # Also report the ceiling: how well does GT expression itself
            # cluster against the real annotation? This is the direct
            # replication of the Nature Comms "predicted sometimes beats
            # ground truth" check -- compare ari_predicted vs ari_gt_ceiling.
            row["ari_gt_ceiling"] = adjusted_rand_score(reference, gt_clusters)
            row["nmi_gt_ceiling"] = normalized_mutual_info_score(reference, gt_clusters)

        rows.append(row)

    return rows


# ── Main evaluation loop (mirrors local_corruption.py's structure) ─────────

def run_spatial_domain_evaluation(args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    set_random_seed(args.seed)

    # ── Build + load model (same pattern as local_corruption.py) ─────────
    model_config = ModelConfig(
        d_model=args.hidden_dim, n_layers=args.n_layers, n_genes=args.n_genes,
        dropout=args.dropout, attn_dropout=args.attn_dropout,
        n_neighbors=args.n_neighbors, n_heads=args.n_heads, dim=2,
        feature_dim=args.feature_dim, pairwise_hidden_dim=args.pairwise_hidden_dim,
        activation=args.activation, mlp_ratio=args.mlp_ratio,
    )
    model_config.d_edge_model = args.pairwise_hidden_dim
    model_config.act = args.activation

    hflow_config = HFlowConfig(
        n_region_queries=args.n_region_queries,
        region_hidden_dim=args.hidden_dim, slide_hidden_dim=args.hidden_dim,
        hflow_representation=args.representation,
        hflow_cross_scale=args.cross_scale,
        hflow_region_discovery=args.region_discovery,
        hflow_assignment_temperature=args.assignment_temperature,
        hflow_assignment_entropy_weight=args.assignment_entropy_weight,
    )
    model = Denoiser(model_config, hflow_config=hflow_config).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    # strict=False for backward-compat: hierarchy-gate keys may be absent in
    # older checkpoints (they stay at neutral init) — matches local_corruption.py.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[*] Missing keys (neutral/zero init, likely hierarchy gate): {missing}")
    if unexpected:
        print(f"[!] Unexpected keys ignored: {len(unexpected)}")
    model.eval()
    print(f"[*] Loaded checkpoint: {args.checkpoint} (representation={args.representation}, "
          f"region_discovery={args.region_discovery})")

    diffusier = Interpolant(
        args.prior_sampler, total_count=torch.tensor([args.zinb_total_count]),
        logits=torch.tensor([args.zinb_logits]), zi_logits=args.zinb_zi_logits,
        normalize=args.prior_sampler != "gaussian",
    )

    # ── Test slides ───────────────────────────────────────────────────
    normalize_method = get_normalize_method(args.normalize_method)
    split_dir = os.path.join(args.source_dataroot, args.dataset, "splits")
    split_df = pd.read_csv(os.path.join(split_dir, f"test_{args.split_id}.csv"))
    test_sample_ids = split_df["sample_id"].tolist()
    print(f"[*] Evaluating {len(test_sample_ids)} test slides: {test_sample_ids}")

    all_rows = []
    for sample_id in tqdm(test_sample_ids, desc=f"{args.dataset}/split{args.split_id}"):
        h5ad_path = os.path.join(args.source_dataroot, args.dataset,
                                  "adata", f"{sample_id}.h5ad")
        sample_id_path = HESTDatasetPath(
            name=sample_id,
            h5_path=os.path.join(args.embed_dataroot, args.dataset,
                                  args.feature_encoder, f"fp32/{sample_id}.h5"),
            h5ad_path=h5ad_path,
            gene_list_path=os.path.join(args.source_dataroot, args.dataset, args.gene_list),
        )
        dataset_obj = HESTDataset(sample_id_path, distribution="constant_1.0",
                                   normalize_method=normalize_method, sample_times=1)
        loader = torch.utils.data.DataLoader(dataset_obj, batch_size=1, collate_fn=padding_batcher())

        annotation_labels = None
        if args.annotation_column:
            annotation_labels = get_annotation_labels(h5ad_path, args.annotation_column)
            if annotation_labels is None:
                print(f"[!] {sample_id}: no '{args.annotation_column}' labels found — "
                      f"falling back to gt_proxy mode for this slide")

        for batch in loader:
            batch = [x.to(device) for x in batch]
            img_features, coords, gene_exp = batch
            pad_mask = (img_features.sum(dim=-1) == 0).squeeze(0).cpu().numpy()

            pred_np = run_sampling(model, diffusier, img_features, coords, gene_exp, args)
            gt_np = gene_exp.squeeze(0).cpu().numpy()

            # Drop padding rows before clustering — clustering padded zero
            # rows would inject a spurious "all-zero" cluster.
            pred_valid = pred_np[~pad_mask]
            gt_valid = gt_np[~pad_mask]
            labels_valid = None
            if annotation_labels is not None:
                labels_valid = annotation_labels[~pad_mask] if len(annotation_labels) == len(pad_mask) \
                    else annotation_labels  # already spot-aligned in some HEST layouts; verify per-dataset

            rows = evaluate_one_slide(
                pred_valid, gt_valid, labels_valid, args.k_values, args.seed,
            )
            for r in rows:
                r["slide"] = sample_id
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # ── Aggregate: mean/std across slides, per k, per mode ──────────────
    group_cols = ["mode", "k"]
    metric_cols = [c for c in df.columns if c.startswith("ari_") or c.startswith("nmi_")]
    summary_df = df.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    summary_df.columns = ["_".join(c).strip("_") for c in summary_df.columns]

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_dir / "per_slide.csv", index=False)
    summary_df.to_csv(save_dir / "summary.csv", index=False)

    print(f"\n[*] Saved per-slide results to {save_dir / 'per_slide.csv'}")
    print(f"[*] Saved summary to {save_dir / 'summary.csv'}")
    print("\n" + summary_df.to_string(index=False))

    if "annotation" in df["mode"].values:
        print("\n[*] MODE 1 (annotation) results found — this is your PRIMARY-metric-worthy "
              "evidence. Compare 'ari_predicted' vs 'ari_gt_ceiling': if ari_predicted is "
              "close to or exceeds ari_gt_ceiling, that directly replicates and extends the "
              "Nature Comms finding for your architecture specifically.")
    else:
        print("\n[!] Only MODE 2 (gt_proxy) results — treat as SECONDARY evidence, "
              "not a standalone primary claim (see earlier discussion).")

    return summary_df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spatial domain recovery evaluation (ARI/NMI).")

    g = p.add_argument_group("Data")
    g.add_argument("--source_dataroot", required=True,
                   help="root containing <dataset>/adata/, <dataset>/splits/, and var_50genes.json")
    g.add_argument("--embed_dataroot", default=None,
                   help="root containing <dataset>/<feature_encoder>/fp32/*.h5. "
                        "Defaults to source_dataroot if omitted.")
    g.add_argument("--dataset", required=True)
    g.add_argument("--split_id", type=int, default=0)
    g.add_argument("--gene_list", default="var_50genes.json")
    g.add_argument("--normalize_method", default="log1p")
    g.add_argument("--feature_encoder", default="uni_v1_official")
    g.add_argument("--annotation_column", default=None,
                   help="obs column name for pathologist annotations, found via "
                        "check_annotations.py. Omit to force gt_proxy mode everywhere.")

    g = p.add_argument_group("Model / checkpoint")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--representation", default="slide_region_patch",
                   choices=["flat", "slide_patch", "slide_region_patch"])
    g.add_argument("--dynamic_update", action="store_true", default=False)
    g.add_argument("--cross_scale", default="bidirectional",
                   choices=["none", "bottom_up", "top_down", "bidirectional"])
    g.add_argument("--region_discovery", default="learnable",
                   choices=["learnable", "grid", "kmeans", "assignment"])
    g.add_argument("--n_region_queries", type=int, default=32)
    g.add_argument("--assignment_temperature", type=float, default=1.0)
    g.add_argument("--assignment_entropy_weight", type=float, default=0.1)

    g = p.add_argument_group("Architecture (must match training)")
    g.add_argument("--hidden_dim", type=int, default=128)
    g.add_argument("--pairwise_hidden_dim", type=int, default=128)
    g.add_argument("--mlp_ratio", type=float, default=4.0)
    g.add_argument("--n_layers", type=int, default=4)
    g.add_argument("--dropout", type=float, default=0.2)
    g.add_argument("--attn_dropout", type=float, default=0.2)
    g.add_argument("--n_neighbors", type=int, default=8)
    g.add_argument("--n_heads", type=int, default=4)
    g.add_argument("--feature_dim", type=int, default=1024)
    g.add_argument("--activation", default="swiglu", choices=["relu", "gelu", "swiglu"])

    g = p.add_argument_group("Inference")
    g.add_argument("--n_sample_steps", type=int, default=3)
    g.add_argument("--prior_sampler", default="zinb")
    g.add_argument("--zinb_logits", type=float, default=0.1)
    g.add_argument("--zinb_total_count", type=float, default=1)
    g.add_argument("--zinb_zi_logits", type=float, default=0.0)
    g.add_argument("--n_genes", type=int, default=50)

    g = p.add_argument_group("Clustering")
    g.add_argument("--k_values", type=int, nargs="+", default=[4, 6, 8, 10],
                   help="Cluster counts to sweep — report all, not just one, "
                        "since ARI/NMI are sensitive to this choice.")

    g = p.add_argument_group("Misc")
    g.add_argument("--save_dir", default="results/spatial_domains")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--device", default="cuda:0")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.embed_dataroot is None:
        args.embed_dataroot = args.source_dataroot
    if not torch.cuda.is_available():
        args.device = "cpu"
    run_spatial_domain_evaluation(args)

#!/usr/bin/env python3
"""
evaluate_local_corruption.py  —  Robustness study for HFlow-ST vs STFlow.

This script evaluates a *trained* model (flat / slide-patch / slide-region-patch)
under progressively stronger local morphology corruption, answering:

    Can hierarchical reasoning compensate when local histology is corrupted?

Corruption is applied only to *image features* at evaluation time –
coordinates, gene labels, and the graph are untouched.

Usage
-----
    # Evaluate a single checkpoint (flat mode  =  STFlow)
    python evaluate_local_corruption.py \\
        --checkpoint /path/to/flat_model.pth \\
        --representation flat \\
        --save_dir results/corruption_flat

    # Evaluate hierarchical model (HFlow-ST)
python local_corruption.py --checkpoint results_dir/test_uni_v1_official_spatial_transformer_26-08-01-13-12-56/LUNG/split1/checkpoints/100.pth --representation slide_region_patch --save_dir results/corruption_hflow

    # Evaluate all corruption types (paper robustness table)
    python evaluate_local_corruption.py \\
        --checkpoint /path/to/model.pth \\
        --representation slide_region_patch \\
        --corruption_types zero gaussian dropout blur \\
        --save_dir results/corruption_all

Output
------
    results.json                – per-(radius, mask_ratio, corruption_type) metrics
    summary.csv                 – table of PCC / MSE / MAE for every setting
"""

import os
import json
import csv
import argparse
import warnings
from copy import deepcopy
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from hmflow.utils import set_random_seed
from hmflow.model.denoiser import HFlowDenoiser as Denoiser
from hmflow.model.hflow_config import HFlowConfig
from hmflow.model.config import ModelConfig
from hmflow.flow.interpolant import Interpolant
from hmflow.data.dataset import (
    HESTDataset,
    HESTDatasetPath,
    padding_batcher,
)
from hmflow.data.normalize_utils import get_normalize_method
from hmflow.hest_utils.utils import save_pkl

from corruption import (
    apply_local_artifact,
    compute_metrics,
    DEFAULT_RADII,
    DEFAULT_MASK_RATIOS,
    DEFAULT_CORRUPTION_TYPES,
    CorruptionType,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ── Evaluation helpers ─────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_slide(
    model: torch.nn.Module,
    diffusier: Interpolant,
    img_features: torch.Tensor,
    coords: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> np.ndarray:
    """
    Run the full Euler sampling loop for one slide *without* corruption.

    Returns  prediction   [N, n_genes]  numpy array.
    """
    model.eval()
    B = img_features.shape[0]
    assert B == 1, "Batch size must be 1 for inference"

    exp_t1 = diffusier.sample_from_prior(labels.shape).to(args.device)
    ts = torch.linspace(0.01, 1.0, args.n_sample_steps)[:, None] \
            .expand(args.n_sample_steps, B).to(args.device)

    hierarchy_state = None
    pred = None

    for step_i, (t1, t2) in enumerate(zip(ts[:-1], ts[1:])):
        pred, hierarchy_state = model.inference(
            exp_t1, img_features, coords, t1,
            hierarchy_state=hierarchy_state,
        )
        if step_i == args.n_sample_steps - 2:
            break
        d_t = t2 - t1
        exp_t1 = diffusier.denoise(pred, exp_t1, t1, d_t)

    return pred.squeeze(0).cpu().numpy()


@torch.no_grad()
def evaluate_slide_corrupted(
    model: torch.nn.Module,
    diffusier: Interpolant,
    img_features: torch.Tensor,
    coords: torch.Tensor,
    labels: torch.Tensor,
    radius: float,
    mask_ratio: float,
    corruption_type: CorruptionType,
    args: argparse.Namespace,
    corrupt_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Evaluate one slide under local corruption.

    The corruption is applied to the *clean* image features before every
    Euler step (so the model must rely on global context at every step).
    """
    model.eval()
    B = img_features.shape[0]
    assert B == 1, "Batch size must be 1 for inference"

    # Corrupt once (same corruption for all Euler steps for consistency)
    corrupted_features = apply_local_artifact(
        img_features, coords,
        radius=radius,
        mask_ratio=mask_ratio,
        corruption_type=corruption_type,
        sigma=args.corruption_sigma,
        dropout_p=args.corruption_dropout_p,
        seed=corrupt_seed,
    )

    exp_t1 = diffusier.sample_from_prior(labels.shape).to(args.device)
    ts = torch.linspace(0.01, 1.0, args.n_sample_steps)[:, None] \
            .expand(args.n_sample_steps, B).to(args.device)

    hierarchy_state = None
    pred = None

    for step_i, (t1, t2) in enumerate(zip(ts[:-1], ts[1:])):
        pred, hierarchy_state = model.inference(
            exp_t1, corrupted_features, coords, t1,
            hierarchy_state=hierarchy_state,
        )
        if step_i == args.n_sample_steps - 2:
            break
        d_t = t2 - t1
        exp_t1 = diffusier.denoise(pred, exp_t1, t1, d_t)

    return pred.squeeze(0).cpu().numpy()


# ── Main evaluation loop ───────────────────────────────────────────────────


def run_corruption_evaluation(args: argparse.Namespace) -> dict:
    """
    Run the full corruption robustness evaluation.

    Returns
    -------
    results : dict
        Nested dict keyed by (corruption_type, radius, mask_ratio) holding
        per-slide metrics.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    set_random_seed(args.seed)

    # ── 1. Build model ────────────────────────────────────────────────────
    model_config = ModelConfig(
        d_model=args.hidden_dim,
        n_layers=args.n_layers,
        n_genes=args.n_genes,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        n_neighbors=args.n_neighbors,
        n_heads=args.n_heads,
        dim=2,
        feature_dim=args.feature_dim,
        pairwise_hidden_dim=args.pairwise_hidden_dim,
        activation=args.activation,
        mlp_ratio=args.mlp_ratio,
    )
    # Map aliases — mirrors train.py lines 258-261
    model_config.d_edge_model = args.pairwise_hidden_dim
    model_config.act = args.activation

    hflow_config = HFlowConfig(
        n_region_queries=args.n_region_queries,
        region_hidden_dim=args.hidden_dim,
        slide_hidden_dim=args.hidden_dim,
        hflow_representation=args.representation,
        hflow_dynamic_update=args.dynamic_update,
        hflow_cross_scale=args.cross_scale,
        hflow_region_discovery=args.region_discovery,
        hflow_assignment_temperature=args.assignment_temperature,
        hflow_assignment_entropy_weight=args.assignment_entropy_weight,
    )
    model = Denoiser(model_config, hflow_config=hflow_config).to(device)
    print(f"[*] Model representation: {args.representation}")

    # ── 2. Load checkpoint ────────────────────────────────────────────────
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        # Strip "module." prefix from DDP checkpoints
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
        print(f"[*] Loaded checkpoint: {args.checkpoint}")
    else:
        print("[!] No checkpoint provided — using randomly initialized model")

    model.eval()

    # ── 3. Build interpolant (flow matching noise scheduler) ──────────────
    diffusier = Interpolant(
        args.prior_sampler,
        total_count=torch.tensor([args.zinb_total_count]),
        logits=torch.tensor([args.zinb_logits]),
        zi_logits=args.zinb_zi_logits,
        normalize=args.prior_sampler != "gaussian",
    )

    # ── 4. Prepare data ───────────────────────────────────────────────────
    normalize_method = get_normalize_method(args.normalize_method)
    sample_id_path = HESTDatasetPath(
        name=args.slide_name or "eval_slide",
        h5_path=args.h5_path,
        h5ad_path=args.h5ad_path,
        gene_list_path=args.gene_list_path,
    )
    dataset = HESTDataset(
        sample_id_path,
        distribution="constant_1.0",
        normalize_method=normalize_method,
        sample_times=1,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=padding_batcher()
    )
    gene_list = dataset.gene_list
    print(f"[*] Slide: {args.slide_name or args.h5_path}")
    print(f"[*] Genes: {len(gene_list)}")

    # ── 5. Uncorrupted baseline ───────────────────────────────────────────
    batch = next(iter(loader))
    batch = [x.to(device) for x in batch]
    img_features_clean, coords, labels = batch
    pred_clean = evaluate_slide(model, diffusier, img_features_clean, coords, labels, args)
    pcc_clean, mse_clean, mae_clean = compute_metrics(pred_clean, labels.squeeze(0).cpu().numpy())
    print(f"\n[*] Baseline (no corruption):  PCC={pcc_clean:.4f}  MSE={mse_clean:.4f}  MAE={mae_clean:.4f}")

    # ── 6. Corruption grid ────────────────────────────────────────────────
    results: dict = {"baseline": {"PCC": pcc_clean, "MSE": mse_clean, "MAE": mae_clean}}
    radii = args.radii or DEFAULT_RADII
    mask_ratios = args.mask_ratios or DEFAULT_MASK_RATIOS
    corruption_types = args.corruption_types or ["zero"]

    rows = []
    for ctype in corruption_types:
        for r in radii:
            for mr in mask_ratios:
                corrupt_seed = args.seed + int(r) + int(mr * 100)
                pred_corr = evaluate_slide_corrupted(
                    model, diffusier,
                    img_features_clean, coords, labels,
                    radius=r,
                    mask_ratio=mr,
                    corruption_type=ctype,
                    args=args,
                    corrupt_seed=corrupt_seed,
                )
                gt = labels.squeeze(0).cpu().numpy()
                pcc, mse, mae = compute_metrics(pred_corr, gt)

                key = f"{ctype}/r={r}/mr={mr}"
                results[key] = {"radius": r, "mask_ratio": mr,
                                "corruption_type": ctype,
                                "PCC": pcc, "MSE": mse, "MAE": mae}
                rows.append({
                    "corruption_type": ctype,
                    "radius": r,
                    "mask_ratio": mr,
                    "PCC": round(pcc, 5),
                    "MSE": round(mse, 5),
                    "MAE": round(mae, 5),
                })
                print(f"  [{ctype}]  r={r:3d}  mr={mr:.2f}  →  PCC={pcc:.4f}  MSE={mse:.4f}  MAE={mae:.4f}")

    # ── 7. Save ────────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        print(f"\n[*] Saved results to {save_dir / 'results.json'}")

    # CSV summary
    csv_path = save_dir / "summary.csv"
    fieldnames = ["corruption_type", "radius", "mask_ratio", "PCC", "MSE", "MAE"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[*] Saved summary to {csv_path}")

    return results


# ── CLI ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate model robustness under local morphology corruption.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--h5_path", required=True,
                   help="Path to .h5 file with image features, coords, barcodes")
    g.add_argument("--h5ad_path", required=True,
                   help="Path to .h5ad file with expression data")
    g.add_argument("--gene_list_path", default=None,
                   help="Path to gene list JSON (default: from h5ad parent / gene_list.json)")
    g.add_argument("--slide_name", type=str, default=None,
                   help="Slide name for logging (default: inferred from h5_path)")
    g.add_argument("--normalize_method", type=str, default="log1p",
                   help="Normalisation: log1p | raw | etc.")

    # Model checkpoint
    g = p.add_argument_group("Model")
    g.add_argument("--checkpoint", type=str, default=None,
                   help="Path to trained model checkpoint (.pth)")
    g.add_argument("--representation", type=str, default="slide_region_patch",
                   choices=["flat", "slide_patch", "slide_region_patch"],
                   help="'flat' = STFlow, 'slide_region_patch' = HFlow-ST")
    g.add_argument("--dynamic_update", action="store_true", default=True)
    g.add_argument("--no_dynamic_update", dest="dynamic_update", action="store_false")
    g.add_argument("--cross_scale", type=str, default="bidirectional",
                   choices=["none", "bottom_up", "top_down", "bidirectional"])
    g.add_argument("--region_discovery", type=str, default="learnable",
                   choices=["learnable", "grid", "kmeans", "assignment"])
    g.add_argument("--n_region_queries", type=int, default=32)
    g.add_argument("--assignment_temperature", type=float, default=1.0)
    g.add_argument("--assignment_entropy_weight", type=float, default=0.1)

    # Model architecture (must match training setup)
    g = p.add_argument_group("Architecture")
    g.add_argument("--hidden_dim", type=int, default=128)
    g.add_argument("--pairwise_hidden_dim", type=int, default=128)
    g.add_argument("--mlp_ratio", type=float, default=4.0)
    g.add_argument("--n_layers", type=int, default=4)
    g.add_argument("--dropout", type=float, default=0.2)
    g.add_argument("--attn_dropout", type=float, default=0.2)
    g.add_argument("--n_neighbors", type=int, default=8)
    g.add_argument("--n_heads", type=int, default=4)
    g.add_argument("--feature_dim", type=int, default=1024,
                   help="uni:1024, ciga:512, gigapath:1536")
    g.add_argument("--activation", type=str, default="swiglu",
                   choices=["relu", "gelu", "swiglu"])

    # Inference
    g = p.add_argument_group("Inference")
    g.add_argument("--n_sample_steps", type=int, default=3)
    g.add_argument("--prior_sampler", type=str, default="zinb",
                   help="gaussian | uniform | zero | zinb")
    g.add_argument("--zinb_logits", type=float, default=0.1)
    g.add_argument("--zinb_total_count", type=float, default=1)
    g.add_argument("--zinb_zi_logits", type=float, default=0.0)
    g.add_argument("--n_genes", type=int, default=50)

    # Corruption grid
    g = p.add_argument_group("Corruption")
    g.add_argument("--radii", type=int, nargs="+", default=None,
                   help="List of radii (default: 0 32 64 96 128)")
    g.add_argument("--mask_ratios", type=float, nargs="+", default=None,
                   help="List of mask ratios (default: 0.1 0.25 0.5)")
    g.add_argument("--corruption_types", type=str, nargs="+",
                   default=["zero"],
                   choices=["zero", "gaussian", "dropout", "blur"],
                   help="Corruption types to evaluate")
    g.add_argument("--corruption_sigma", type=float, default=0.5,
                   help="Noise std for 'gaussian' corruption")
    g.add_argument("--corruption_dropout_p", type=float, default=0.5,
                   help="Dropout probability for 'dropout' corruption")

    # Misc
    g = p.add_argument_group("Misc")
    g.add_argument("--save_dir", type=str, default="results/corruption_eval",
                   help="Output directory for results")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--device", type=str, default="cuda:0")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # Implicit default for gene_list_path
    if args.gene_list_path is None:
        args.gene_list_path = os.path.join(
            os.path.dirname(args.h5ad_path), "..", args.gene_list or "var_50genes.json"
        )

    # SLURM / distributed: silently map to cpu if no cuda
    if not torch.cuda.is_available():
        args.device = "cpu"

    run_corruption_evaluation(args)

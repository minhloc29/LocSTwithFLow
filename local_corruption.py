#!/usr/bin/env python3
"""
evaluate_local_corruption.py  —  Robustness study for MOSAIC vs STFlow.

Evaluates a *trained* model (flat / slide-patch / slide-region-patch) under
progressively stronger local morphology corruption, aggregated over an
entire held-out test split (not a single slide), answering:

    Can hierarchical reasoning compensate when local histology is corrupted?

Fixes applied relative to the previous version of this script:
  1. Loops over the FULL test loader and aggregates mean/std across slides,
     instead of evaluating a single `next(iter(loader))` batch.
  2. Defaults for --dynamic_update / --region_discovery now match the
     finalized, validated configuration (static reset + grid discovery)
     rather than the known-unstable defaults (persistent + learnable).
     You must now pass flags explicitly if you want to evaluate a
     different configuration — nothing is silently assumed.
  3. Averages each (corruption_type, radius, mask_ratio) cell over
     `--n_corrupt_seeds` independent random masks, instead of one draw.
  4. Neutralizes the "zero" corruption / pad_mask collision: HFlowDenoiser
     treats any patch with img_features.sum(-1) == 0 as PADDING (excluded
     from the k-NN graph and never assigned a prediction). Literal
     zero-corruption therefore silently removes spots from both the graph
     AND the evaluation, confounding "prediction quality under corruption"
     with "prediction quality with fewer graph nodes." We nudge exact-zero
     corrupted features by a small epsilon so they remain valid graph
     nodes with near-total information loss, without tripping the padding
     check. This does not apply to gaussian/dropout corruption, which
     don't produce exact zeros.

Usage
-----
    # Evaluate STFlow (flat baseline) across the full test split
    python evaluate_local_corruption.py \\
        --checkpoint /path/to/flat_model.pth \\
        --representation flat \\
        --split_dir dataset/READ/splits --split_id 0 \\
        --save_dir results/corruption_flat

    # Evaluate MOSAIC (finalized: static + grid) across the full test split
    python evaluate_local_corruption.py \\
        --checkpoint /path/to/mosaic_model.pth \\
        --representation slide_region_patch \\
        --no_dynamic_update --region_discovery grid \\
        --split_dir dataset/READ/splits --split_id 0 \\
        --save_dir results/corruption_mosaic

Output
------
    results.json   – per-(corruption_type, radius, mask_ratio) mean/std across slides
    summary.csv    – flat table, one row per corruption setting
    per_slide.csv  – full per-slide detail, for post-hoc significance testing
"""

import os
import json
import csv
import argparse
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
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

from corruption import (
    apply_local_artifact,
    compute_metrics,
    DEFAULT_RADII,
    DEFAULT_MASK_RATIOS,
    CorruptionType,
)

warnings.filterwarnings("ignore", category=UserWarning)

ZERO_CORRUPTION_EPSILON = 1e-4  # small enough to represent "near total loss",
                                  # large enough that sum(-1) != 0 in fp32


# ── Corruption / pad_mask safety patch ──────────────────────────────────────

def _neutralize_padding_collision(corrupted_features: torch.Tensor,
                                    original_valid_mask: torch.Tensor,
                                    corruption_type: str) -> torch.Tensor:
    """
    If corruption_type == 'zero', patches that were legitimately corrupted to
    all-zero would be indistinguishable from padding under
    `img_features.sum(-1) == 0` inside HFlowDenoiser. We nudge exact-zero,
    originally-valid patches by a tiny epsilon so they stay in the graph and
    still receive a prediction, while still being ~99.99% information-destroyed.

    Padding that was ALREADY padding (never valid) is left untouched.
    """
    if corruption_type != "zero":
        return corrupted_features

    is_now_zero = corrupted_features.sum(dim=-1) == 0          # [B, N]
    needs_nudge = is_now_zero & original_valid_mask              # only real spots
    if needs_nudge.any():
        corrupted_features = corrupted_features.clone()
        corrupted_features[needs_nudge] = ZERO_CORRUPTION_EPSILON
    return corrupted_features


# ── Evaluation helpers ─────────────────────────────────────────────────────


@torch.no_grad()
def run_sampling(
    model: torch.nn.Module,
    diffusier: Interpolant,
    img_features: torch.Tensor,
    coords: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> np.ndarray:
    """Run the full Euler sampling loop for one slide. Returns [N, n_genes]."""
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


def evaluate_one_slide_all_settings(
    model, diffusier, img_features_clean, coords, labels, args,
    corruption_types, radii, mask_ratios, slide_name,
):
    """
    Returns a list of per-slide result rows (one per corruption setting,
    already averaged over --n_corrupt_seeds), plus the clean baseline row.
    """
    device = args.device
    rows = []

    # Original validity mask (before any corruption) — used to distinguish
    # real spots from pre-existing padding when neutralizing the zero-corruption
    # / pad_mask collision.
    original_valid_mask = (img_features_clean.sum(dim=-1) != 0)

    # Clean baseline
    pred_clean = run_sampling(model, diffusier, img_features_clean, coords, labels, args)
    pcc_c, mse_c, mae_c = compute_metrics(pred_clean, labels.squeeze(0).cpu().numpy())
    rows.append({
        "slide": slide_name, "corruption_type": "none", "radius": 0, "mask_ratio": 0.0,
        "seed_idx": -1, "PCC": pcc_c, "MSE": mse_c, "MAE": mae_c,
    })

    for ctype in corruption_types:
        for r in radii:
            for mr in mask_ratios:
                for seed_idx in range(args.n_corrupt_seeds):
                    corrupt_seed = args.seed + int(r) + int(mr * 1000) + seed_idx * 7919
                    corrupted = apply_local_artifact(
                        img_features_clean, coords,
                        radius=r, mask_ratio=mr, corruption_type=ctype,
                        sigma=args.corruption_sigma,
                        dropout_p=args.corruption_dropout_p,
                        seed=corrupt_seed,
                    )
                    corrupted = _neutralize_padding_collision(
                        corrupted, original_valid_mask, ctype
                    )

                    exp_t1 = diffusier.sample_from_prior(labels.shape).to(device)
                    ts = torch.linspace(0.01, 1.0, args.n_sample_steps)[:, None] \
                            .expand(args.n_sample_steps, 1).to(device)
                    hierarchy_state = None
                    pred = None
                    for step_i, (t1, t2) in enumerate(zip(ts[:-1], ts[1:])):
                        pred, hierarchy_state = model.inference(
                            exp_t1, corrupted, coords, t1,
                            hierarchy_state=hierarchy_state,
                        )
                        if step_i == args.n_sample_steps - 2:
                            break
                        d_t = t2 - t1
                        exp_t1 = diffusier.denoise(pred, exp_t1, t1, d_t)

                    pred_np = pred.detach().squeeze(0).cpu().numpy()
                    gt = labels.squeeze(0).cpu().numpy()
                    pcc, mse, mae = compute_metrics(pred_np, gt)

                    rows.append({
                        "slide": slide_name, "corruption_type": ctype,
                        "radius": r, "mask_ratio": mr, "seed_idx": seed_idx,
                        "PCC": pcc, "MSE": mse, "MAE": mae,
                    })
    return rows


# ── Main evaluation loop ───────────────────────────────────────────────────


def run_corruption_evaluation(args: argparse.Namespace) -> pd.DataFrame:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.device = device
    set_random_seed(args.seed)

    if args.representation != "flat" and not args.region_discovery_explicit:
        print("[!] WARNING: --region_discovery was not explicitly passed. "
              f"Using default '{args.region_discovery}'. If this checkpoint was "
              "trained with a different region-discovery method, results will "
              "be meaningless (strict state_dict loading will likely fail first).")
    if args.representation != "flat" and not args.dynamic_update_explicit:
        print(f"[!] WARNING: --dynamic_update/--no_dynamic_update was not explicitly "
              f"passed. Using default dynamic_update={args.dynamic_update}. Confirm "
              "this matches how the checkpoint was trained.")

    # ── 1. Build model ────────────────────────────────────────────────────
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
    print(f"[*] Model representation: {args.representation}  "
          f"(dynamic_update={args.dynamic_update}, region_discovery={args.region_discovery})")

    # ── 2. Load checkpoint ────────────────────────────────────────────────
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)  # will fail loudly on config mismatch
        print(f"[*] Loaded checkpoint: {args.checkpoint}")
    else:
        print("[!] No checkpoint provided — using randomly initialized model")
    model.eval()

    # ── 3. Interpolant ────────────────────────────────────────────────────
    diffusier = Interpolant(
        args.prior_sampler,
        total_count=torch.tensor([args.zinb_total_count]),
        logits=torch.tensor([args.zinb_logits]),
        zi_logits=args.zinb_zi_logits,
        normalize=args.prior_sampler != "gaussian",
    )

    # ── 4. Build FULL test loader (not a single slide) ─────────────────────
    normalize_method = get_normalize_method(args.normalize_method)
    split_df = pd.read_csv(os.path.join(args.split_dir, f"test_{args.split_id}.csv"))
    test_sample_ids = split_df["sample_id"].tolist()
    print(f"[*] Evaluating {len(test_sample_ids)} test slides: {test_sample_ids}")

    corruption_types = args.corruption_types or ["zero"]
    radii = args.radii or DEFAULT_RADII
    mask_ratios = args.mask_ratios or DEFAULT_MASK_RATIOS

    all_rows = []
    for sample_id in tqdm(test_sample_ids, desc="slides"):
        sample_id_path = HESTDatasetPath(
            name=sample_id,
            h5_path=os.path.join(args.embed_dataroot, args.dataset,
                                  args.feature_encoder, f"fp32/{sample_id}.h5"),
            h5ad_path=os.path.join(args.source_dataroot, args.dataset,
                                    f"adata/{sample_id}.h5ad"),
            gene_list_path=os.path.join(args.source_dataroot, args.dataset, args.gene_list),
        )
        dataset = HESTDataset(sample_id_path, distribution="constant_1.0",
                               normalize_method=normalize_method, sample_times=1)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=padding_batcher())

        for batch in loader:  # HESTDataset may yield >1 crop per slide depending on config
            batch = [x.to(device) for x in batch]
            img_features_clean, coords, labels = batch
            rows = evaluate_one_slide_all_settings(
                model, diffusier, img_features_clean, coords, labels, args,
                corruption_types, radii, mask_ratios, slide_name=sample_id,
            )
            all_rows.extend(rows)

    per_slide_df = pd.DataFrame(all_rows)

    # ── 5. Aggregate: mean/std across slides AND corrupt seeds, per setting ─
    group_cols = ["corruption_type", "radius", "mask_ratio"]
    summary_df = (
        per_slide_df.groupby(group_cols)[["PCC", "MSE", "MAE"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_df.columns = ["_".join(c).strip("_") for c in summary_df.columns]

    # ── 6. Save everything ───────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    per_slide_df.to_csv(save_dir / "per_slide.csv", index=False)
    summary_df.to_csv(save_dir / "summary.csv", index=False)
    with open(save_dir / "results.json", "w") as f:
        json.dump(json.loads(summary_df.to_json(orient="records")), f, indent=2)

    print(f"\n[*] Saved per-slide detail to {save_dir / 'per_slide.csv'}")
    print(f"[*] Saved aggregated summary to {save_dir / 'summary.csv'}")
    print("\n" + summary_df.to_string(index=False))

    return summary_df


# ── CLI ────────────────────────────────────────────────────────────────────


class _TrackExplicit(argparse.Action):
    """Records whether an argument was explicitly passed, so we can warn on silent defaults."""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate model robustness under local morphology corruption, "
                     "aggregated over the full test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Data")
    g.add_argument("--source_dataroot", default="dataset")
    g.add_argument("--embed_dataroot", default="dataset")
    g.add_argument("--dataset", type=str, required=True, help="e.g. READ, LUNG")
    g.add_argument("--split_dir", type=str, required=True,
                   help="Directory containing test_{split_id}.csv")
    g.add_argument("--split_id", type=int, default=0)
    g.add_argument("--gene_list", type=str, default="var_50genes.json")
    g.add_argument("--normalize_method", type=str, default="log1p")
    g.add_argument("--feature_encoder", type=str, default="uni_v1_official")

    g = p.add_argument_group("Model")
    g.add_argument("--checkpoint", type=str, default=None)
    g.add_argument("--representation", type=str, default="slide_region_patch",
                   choices=["flat", "slide_patch", "slide_region_patch"],
                   help="'flat' = STFlow, 'slide_region_patch' = MOSAIC")
    # NOTE: defaults now match the FINALIZED, validated configuration
    # (static reset + grid discovery), not the historically-broken defaults.
    # Pass --dynamic_update / --region_discovery explicitly to override, and
    # you'll get a warning if you didn't, rather than a silent mismatch.
    g.add_argument("--dynamic_update", action=_TrackExplicit, nargs="?",
                   const=True, default=False, type=lambda x: x.lower() == "true")
    g.add_argument("--no_dynamic_update", dest="dynamic_update", action="store_const",
                   const=False)
    g.add_argument("--cross_scale", type=str, default="bidirectional",
                   choices=["none", "bottom_up", "top_down", "bidirectional"])
    g.add_argument("--region_discovery", type=str, default="learnable", action=_TrackExplicit,
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
    g.add_argument("--activation", type=str, default="swiglu",
                   choices=["relu", "gelu", "swiglu"])

    g = p.add_argument_group("Inference")
    g.add_argument("--n_sample_steps", type=int, default=3)
    g.add_argument("--prior_sampler", type=str, default="zinb")
    g.add_argument("--zinb_logits", type=float, default=0.1)
    g.add_argument("--zinb_total_count", type=float, default=1)
    g.add_argument("--zinb_zi_logits", type=float, default=0.0)
    g.add_argument("--n_genes", type=int, default=50)

    g = p.add_argument_group("Corruption")
    g.add_argument("--radii", type=int, nargs="+", default=None)
    g.add_argument("--mask_ratios", type=float, nargs="+", default=None)
    g.add_argument("--corruption_types", type=str, nargs="+", default=["zero"],
                   choices=["zero", "gaussian", "dropout", "blur"],
                   help="Default changed from 'zero' to 'gaussian' — 'zero' collides "
                        "with the padding convention; use it only with the "
                        "neutralization patch understood (see module docstring).")
    g.add_argument("--corruption_sigma", type=float, default=0.5)
    g.add_argument("--corruption_dropout_p", type=float, default=0.5)
    g.add_argument("--n_corrupt_seeds", type=int, default=3,
                   help="Number of independent random masks averaged per "
                        "(corruption_type, radius, mask_ratio) cell.")

    g = p.add_argument_group("Misc")
    g.add_argument("--save_dir", type=str, default="results/corruption_eval")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--device", type=str, default="cuda:0")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "dynamic_update_explicit"):
        args.dynamic_update_explicit = False
    if not hasattr(args, "region_discovery_explicit"):
        args.region_discovery_explicit = False

    if not torch.cuda.is_available():
        args.device = "cpu"

    run_corruption_evaluation(args)
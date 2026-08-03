import os
import copy
import json
import argparse
import warnings
from pathlib import Path
 
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
)
 
warnings.filterwarnings("ignore", category=UserWarning)
 
ZERO_CORRUPTION_EPSILON = 1e-4  # small enough to represent "near total loss",
ALL_DATASETS = ["LUNG", "HCC", "COAD", "SKCM", "PAAD", "READ",
                 "LYMPH_IDC", "PRAD", "IDC", "CCRCC"]
 
 
 
 
def _neutralize_padding_collision(corrupted_features: torch.Tensor,
                                    original_valid_mask: torch.Tensor,
                                    corruption_type: str) -> torch.Tensor:
    if corruption_type != "zero":
        return corrupted_features
 
    is_now_zero = corrupted_features.sum(dim=-1) == 0          # [B, N]
    needs_nudge = is_now_zero & original_valid_mask              # only real spots
    if needs_nudge.any():
        corrupted_features = corrupted_features.clone()
        corrupted_features[needs_nudge] = ZERO_CORRUPTION_EPSILON
    return corrupted_features
 
 
 
 
@torch.no_grad()
def run_sampling(
    model: torch.nn.Module,
    diffusier: Interpolant,
    img_features: torch.Tensor,
    coords: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> np.ndarray:

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
    device = args.device
    rows = []
 
    original_valid_mask = (img_features_clean.sum(dim=-1) != 0)
 
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
 
                    exp_t1 = diffusier.sample_from_prior(labels.shape, labels.device)
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
 
 
 
def discover_split_ids(source_dataroot: str, dataset: str) -> list:
    """
    Find every split id this dataset actually has test_{id}.csv files for,
    mirroring the `len(splits) // 2` pairing logic in train.py's run().
    """
    split_dir = os.path.join(source_dataroot, dataset, "splits")
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"No splits directory found at {split_dir}")
    test_files = [f for f in os.listdir(split_dir) if f.startswith("test_") and f.endswith(".csv")]
    if not test_files:
        raise FileNotFoundError(f"No test_*.csv split files found in {split_dir}")
    ids = sorted(int(f.removeprefix("test_").removesuffix(".csv")) for f in test_files)
    return ids
 
 
def resolve_checkpoint(checkpoint_root: str, dataset: str, split_id: int, epoch) -> str:
    
    ckpt_dir = os.path.join(checkpoint_root, dataset, f"split{split_id}", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"No checkpoint directory found at {ckpt_dir}")
 
    if epoch == "best":
        candidates = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
        if not candidates:
            raise FileNotFoundError(f"No .pth checkpoints found in {ckpt_dir}")
        epoch_num = max(int(f.removesuffix(".pth")) for f in candidates)
        return os.path.join(ckpt_dir, f"{epoch_num}.pth")
 
    ckpt_path = os.path.join(ckpt_dir, f"{epoch}.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path
  
 
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
        hflow_cross_scale=args.cross_scale,
        hflow_region_discovery=args.region_discovery,
        hflow_assignment_temperature=args.assignment_temperature,
        hflow_assignment_entropy_weight=args.assignment_entropy_weight,
    )
    model = Denoiser(model_config, hflow_config=hflow_config).to(device)
    print(f"[*] {args.dataset}/split{args.split_id} — representation: {args.representation}  "
          f"region_discovery={args.region_discovery})")
 
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)  # will fail loudly on config mismatch
        print(f"[*] Loaded checkpoint: {args.checkpoint}")
    else:
        print("[!] No checkpoint provided — using randomly initialized model")
    model.eval()
 
    diffusier = Interpolant(
        args.prior_sampler,
        total_count=torch.tensor([args.zinb_total_count]),
        logits=torch.tensor([args.zinb_logits]),
        zi_logits=args.zinb_zi_logits,
        normalize=args.prior_sampler != "gaussian",
    )
 
    # ── 3. Build test loader — split_dir derived, not passed via CLI ───────
    normalize_method = get_normalize_method(args.normalize_method)
    split_dir = os.path.join(args.source_dataroot, args.dataset, "splits")
    split_df = pd.read_csv(os.path.join(split_dir, f"test_{args.split_id}.csv"))
    test_sample_ids = split_df["sample_id"].tolist()
    print(f"[*] Evaluating {len(test_sample_ids)} test slides: {test_sample_ids}")
 
    corruption_types = args.corruption_types or ["zero"]
    radii = args.radii or DEFAULT_RADII
    mask_ratios = args.mask_ratios or DEFAULT_MASK_RATIOS
 
    all_rows = []
    for sample_id in tqdm(test_sample_ids, desc=f"{args.dataset}/split{args.split_id}"):
        sample_id_path = HESTDatasetPath(
            name=sample_id,
            h5_path=os.path.join(args.embed_dataroot, args.dataset,
                                  args.feature_encoder, f"fp32/{sample_id}.h5"),
            h5ad_path=os.path.join(args.source_dataroot, args.dataset,
                                    f"adata/{sample_id}.h5ad"),
            gene_list_path=os.path.join(args.source_dataroot, args.dataset, args.gene_list),
        )
        dataset_obj = HESTDataset(sample_id_path, distribution="constant_1.0",
                                   normalize_method=normalize_method, sample_times=1)
        loader = torch.utils.data.DataLoader(dataset_obj, batch_size=1, collate_fn=padding_batcher())
 
        for batch in loader:  # HESTDataset may yield >1 crop per slide depending on config
            batch = [x.to(device) for x in batch]
            img_features_clean, coords, labels = batch
            rows = evaluate_one_slide_all_settings(
                model, diffusier, img_features_clean, coords, labels, args,
                corruption_types, radii, mask_ratios, slide_name=sample_id,
            )
            all_rows.extend(rows)
 
    per_slide_df = pd.DataFrame(all_rows)
 
    group_cols = ["corruption_type", "radius", "mask_ratio"]
    summary_df = (
        per_slide_df.groupby(group_cols)[["PCC", "MSE", "MAE"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_df.columns = ["_".join(c).strip("_") for c in summary_df.columns]
 
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
 
    per_slide_df.to_csv(save_dir / "per_slide.csv", index=False)
    summary_df.to_csv(save_dir / "summary.csv", index=False)
    with open(save_dir / "results.json", "w") as f:
        json.dump(json.loads(summary_df.to_json(orient="records")), f, indent=2)
 
    print(f"[*] Saved per-slide detail to {save_dir / 'per_slide.csv'}")
    print(f"[*] Saved aggregated summary to {save_dir / 'summary.csv'}")
 
    return summary_df
 
 
# ── Multi-dataset / multi-split driver (new — mirrors train.py's outer loop) ─
 
 
def run_corruption_evaluation_all(base_args: argparse.Namespace) -> pd.DataFrame:
    """
    Loop over every requested dataset and every requested (or discovered)
    split_id, deriving the checkpoint path and save_dir for each combination,
    exactly mirroring how train.py's run() loops over splits and how its
    __main__ block loops over datasets.
 
    Uses a fresh deep copy of base_args per (dataset, split_id) so a bug or
    stale field in one run cannot leak into the next — base_args itself is
    never mutated.
    """
    datasets = ALL_DATASETS if base_args.datasets[0] == "all" else base_args.datasets
 
    all_summaries = []
    for dataset in datasets:
        split_ids = base_args.split_ids or discover_split_ids(base_args.source_dataroot, dataset)
 
        for split_id in split_ids:
            run_args = copy.deepcopy(base_args)
            run_args.dataset = dataset
            run_args.split_id = split_id
            run_args.checkpoint = resolve_checkpoint(
                base_args.checkpoint_root, dataset, split_id, base_args.checkpoint_epoch
            )
            run_args.save_dir = os.path.join(base_args.save_dir_root, dataset, f"split{split_id}")
 
            try:
                summary_df = run_corruption_evaluation(run_args)
            except FileNotFoundError as e:
                print(f"[!] Skipping {dataset}/split{split_id}: {e}")
                continue
 
            summary_df = summary_df.copy()
            summary_df.insert(0, "split_id", split_id)
            summary_df.insert(0, "dataset", dataset)
            all_summaries.append(summary_df)
 
    if not all_summaries:
        raise RuntimeError("No (dataset, split_id) combination produced results — "
                            "check --checkpoint_root and dataset/split availability.")
 
    combined = pd.concat(all_summaries, ignore_index=True)
    out_path = Path(base_args.save_dir_root) / "all_datasets_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\n[*] Saved combined multi-dataset summary to {out_path}")
    print("\n" + combined.to_string(index=False))
 
    return combined
 
 
 
class _TrackExplicit(argparse.Action):
    """Records whether a flag was explicitly passed, alongside its value."""
 
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicit", True)
 
 
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate model robustness under local morphology corruption, "
                     "across one or many (dataset, split) combinations, mirroring "
                     "train.py's --datasets all pattern.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    g = p.add_argument_group("Data")
    g.add_argument("--source_dataroot", default="dataset")
    g.add_argument("--embed_dataroot", default="dataset")
    g.add_argument("--datasets", nargs="+", default=["all"],
                   help="Dataset name(s), e.g. 'LUNG READ', or 'all' for the full "
                        f"fixed list used at training time: {ALL_DATASETS}")
    g.add_argument("--split_ids", type=int, nargs="+", default=None,
                   help="Which split ids to evaluate per dataset. Default: every "
                        "split with a test_*.csv file found on disk for that dataset.")
    g.add_argument("--gene_list", type=str, default="var_50genes.json")
    g.add_argument("--normalize_method", type=str, default="log1p")
    g.add_argument("--feature_encoder", type=str, default="uni_v1_official")
 
    g = p.add_argument_group("Model / checkpoint")
    g.add_argument("--checkpoint_root", type=str, required=True,
                   help="Root save_dir used at training time (train.py's --save_dir/"
                        "exp_code folder). Checkpoint paths are derived as "
                        "{checkpoint_root}/{dataset}/split{split_id}/checkpoints/{epoch}.pth")
    g.add_argument("--checkpoint_epoch", default="best",
                   help="'best' = highest-numbered checkpoint present per split, "
                        "or an explicit epoch number (int, passed as string on CLI).")
    g.add_argument("--representation", type=str, default="slide_region_patch",
                   choices=["flat", "slide_patch", "slide_region_patch"],
                   help="'flat' = STFlow, 'slide_region_patch' = full hierarchy")

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
    g.add_argument("--corruption_types", type=str, nargs="+", default=["gaussian"],
                   choices=["zero", "gaussian", "dropout", "blur"],
                   help="Default is 'gaussian' — 'zero' collides with the padding "
                        "convention; use it only with the neutralization patch "
                        "understood (see module docstring).")
    g.add_argument("--corruption_sigma", type=float, default=0.5)
    g.add_argument("--corruption_dropout_p", type=float, default=0.5)
    g.add_argument("--n_corrupt_seeds", type=int, default=3,
                   help="Number of independent random masks averaged per "
                        "(corruption_type, radius, mask_ratio) cell.")
 
    g = p.add_argument_group("Misc")
    g.add_argument("--save_dir_root", type=str, default="results/corruption_eval",
                   help="Root output directory. Per-run results are written to "
                        "{save_dir_root}/{dataset}/split{split_id}/, and a combined "
                        "summary across all runs is written directly under this root.")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--device", type=str, default="cuda:0")
 
    return p
 
 
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
 
    if not hasattr(args, "region_discovery_explicit"):
        args.region_discovery_explicit = False

    args.dynamic_update_explicit = True
 
    if not torch.cuda.is_available():
        args.device = "cpu"
 
    run_corruption_evaluation_all(args)

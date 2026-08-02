# """
# plot_corruption_retention.py

# Reproduces the "Performance Retention under Corruption" figure from a list
# of one or more corruption-evaluation summary CSVs (the format produced by
# the corruption evaluation script's `summary.csv` output: columns
# corruption_type, radius, mask_ratio, PCC_mean, PCC_std, PCC_count, ...).

# Two plotting modes:

#   --mode severity  (PRIMARY / main-text figure)
#       x-axis = mask_ratio (corruption severity), one line per model,
#       POOLED across all radii (since radius has minimal effect — see the
#       'radius' mode below for the evidence that justifies pooling). This is
#       the figure that should carry your headline claim: retention should
#       drop, and the model comparison should widen, as severity increases.

#   --mode radius    (SUPPLEMENTARY / appendix figure)
#       x-axis = radius, one panel per mask_ratio, one line per model.
#       Included specifically to demonstrate that pooling over radius in the
#       severity view is justified (i.e. radius doesn't materially change the
#       result at fixed severity).

#   --mode both (default) produces both figures.

# Retention is computed per model as:
#     retention = PCC_mean(corrupted) / PCC_mean(clean baseline, i.e. the
#                 'none' row for that model)

# Std is propagated via standard ratio error propagation. When pooling across
# radii (severity mode), corrupted mean/std/count are first combined via the
# standard pooled-sample formula (accounts for both within-group and
# between-group variance), then propagated against the clean baseline.

# If the clean baseline has no std (e.g. count == 1, single run), its
# contribution to the propagated variance is treated as 0 — this
# UNDERESTIMATES the true retention uncertainty. For a submission-quality
# figure, run the clean baseline over multiple seeds too (matching
# --n_corrupt_seeds) so PCC_std for the 'none' row is non-trivial.

# Usage
# -----
#     python plot_corruption_retention.py \\
#         --csv model_a.csv --label "MOSAIC (ours)" --color "#2255aa" \\
#         --csv model_b.csv --label "STFlow (baseline)" --color "#cc3333" \\
#         --corruption_type zero \\
#         --mode both \\
#         --out retention_plot
# """

# import argparse
# import math
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches


# DEFAULT_COLORS = ["#2255aa", "#cc3333", "#2c9e5b", "#a05bbf", "#e08214"]
# DEFAULT_MARKERS = ["o", "s", "^", "D", "v"]


# def pooled_mean_std_count(means: np.ndarray, stds: np.ndarray, counts: np.ndarray):
#     """
#     Combine several (mean, std, count) groups into one pooled (mean, std, count),
#     accounting for both within-group and between-group variance:

#         pooled_mean = sum(n_i * mean_i) / sum(n_i)
#         pooled_var  = [ sum(n_i * std_i^2) + sum(n_i * (mean_i - pooled_mean)^2) ]
#                       / sum(n_i)
#     """
#     counts = counts.astype(float)
#     total_n = counts.sum()
#     pooled_mean = (counts * means).sum() / total_n
#     within = (counts * stds ** 2).sum()
#     between = (counts * (means - pooled_mean) ** 2).sum()
#     pooled_var = (within + between) / total_n
#     return pooled_mean, math.sqrt(max(pooled_var, 0.0)), total_n


# def load_model_csv(csv_path: str, corruption_type: str) -> dict:
#     """
#     Returns a dict:
#         {
#           'clean_mean': float, 'clean_std': float,
#           'by_mask_ratio': {
#               mask_ratio: {'radii': [...], 'retention_mean': [...], 'retention_std': [...]}
#           }
#         }
#     """
#     df = pd.read_csv(csv_path)

#     clean_rows = df[df["corruption_type"] == "none"]
#     if clean_rows.empty:
#         raise ValueError(
#             f"{csv_path}: no 'none' (clean baseline) row found — "
#             "cannot compute retention without a clean reference."
#         )
#     clean_mean = float(clean_rows["PCC_mean"].iloc[0])
#     clean_std_raw = clean_rows["PCC_std"].iloc[0]
#     clean_std = float(clean_std_raw) if pd.notna(clean_std_raw) else 0.0
#     if clean_std == 0.0:
#         print(f"[!] {csv_path}: clean baseline has no std (single run). "
#               "Retention error bars will UNDERESTIMATE true uncertainty. "
#               "Run the clean baseline over multiple seeds for a rigorous figure.")

#     corrupted = df[(df["corruption_type"] == corruption_type) & (df["radius"] > 0)]

#     by_mask_ratio = {}
#     for mr, group in corrupted.groupby("mask_ratio"):
#         group = group.sort_values("radius")
#         radii = group["radius"].to_numpy(dtype=float)
#         corr_mean = group["PCC_mean"].to_numpy(dtype=float)
#         corr_std = group["PCC_std"].fillna(0.0).to_numpy(dtype=float)

#         retention_mean = corr_mean / clean_mean
#         # ratio error propagation; clean_std contribution folded in identically
#         # across all points for this model (same clean baseline throughout)
#         rel_var = (corr_std / np.where(corr_mean != 0, corr_mean, 1e-12)) ** 2 + \
#                   (clean_std / clean_mean) ** 2
#         retention_std = retention_mean * np.sqrt(rel_var)

#         # Prepend the synthetic baseline point: radius=0, retention=1.0 exactly
#         radii = np.concatenate([[0.0], radii])
#         retention_mean = np.concatenate([[1.0], retention_mean])
#         retention_std = np.concatenate([[0.0], retention_std])

#         by_mask_ratio[mr] = {
#             "radii": radii,
#             "retention_mean": retention_mean,
#             "retention_std": retention_std,
#         }

#     return {
#         "clean_mean": clean_mean,
#         "clean_std": clean_std,
#         "by_mask_ratio": by_mask_ratio,
#         "corrupted_df": corrupted,  # raw filtered rows, used by severity-view pooling
#     }


# def build_severity_view(data: dict) -> dict:
#     """
#     Pools each mask_ratio's per-radius cells into a single (mean, std, count),
#     then computes retention against the clean baseline. Returns:
#         {
#           'mask_ratios': [...],
#           'retention_mean': [...],
#           'retention_std': [...],
#         }
#     (with a synthetic mask_ratio=0, retention=1.0 point prepended)
#     """
#     clean_mean, clean_std = data["clean_mean"], data["clean_std"]
#     df = data["corrupted_df"]

#     mask_ratios, ret_means, ret_stds = [], [], []
#     for mr, group in df.groupby("mask_ratio"):
#         means = group["PCC_mean"].to_numpy(dtype=float)
#         stds = group["PCC_std"].fillna(0.0).to_numpy(dtype=float)
#         counts = group["PCC_count"].to_numpy(dtype=float)

#         pooled_mean, pooled_std, _ = pooled_mean_std_count(means, stds, counts)

#         retention_mean = pooled_mean / clean_mean
#         rel_var = (pooled_std / max(pooled_mean, 1e-12)) ** 2 + (clean_std / clean_mean) ** 2
#         retention_std = retention_mean * math.sqrt(rel_var)

#         mask_ratios.append(mr)
#         ret_means.append(retention_mean)
#         ret_stds.append(retention_std)

#     order = np.argsort(mask_ratios)
#     mask_ratios = np.array(mask_ratios)[order]
#     ret_means = np.array(ret_means)[order]
#     ret_stds = np.array(ret_stds)[order]

#     # prepend synthetic (mask_ratio=0, retention=1.0) baseline point
#     mask_ratios = np.concatenate([[0.0], mask_ratios])
#     ret_means = np.concatenate([[1.0], ret_means])
#     ret_stds = np.concatenate([[0.0], ret_stds])

#     return {"mask_ratios": mask_ratios, "retention_mean": ret_means, "retention_std": ret_stds}


# def plot_severity_view(models: list, corruption_type: str, out_path: str, title: str = None):
#     """
#     PRIMARY figure: single panel, x = mask_ratio (severity), pooled over radius.
#     """
#     corruption_label = {
#         "zero": "Zero (Missing Morphology)",
#         "gaussian": "Gaussian Noise",
#         "dropout": "Feature Dropout",
#         "blur": "Blur",
#     }.get(corruption_type, corruption_type.capitalize())

#     fig, ax = plt.subplots(figsize=(7.5, 6))
#     fig.suptitle(title or f"Performance Retention vs. Corruption Severity ({corruption_label})",
#                  fontsize=15, fontweight="bold", y=1.04)
#     fig.text(0.5, 0.965,
#               r"Retention = PCC$_{corrupted}$ / PCC$_{clean}$   (Higher is better).  "
#               "Pooled across all neighborhood radii.",
#               ha="center", fontsize=10, color="#444444")

#     clean_box_lines = ["Clean PCC (mask_ratio=0)"]
#     summary_rows = []

#     for m in models:
#         sv = build_severity_view(m["data"])
#         x, y, s = sv["mask_ratios"], sv["retention_mean"], sv["retention_std"]

#         ax.plot(x, y, color=m["color"], marker=m["marker"], markersize=9,
#                 linewidth=2.4, label=m["label"])
#         ax.fill_between(x, y - s, y + s, color=m["color"], alpha=0.15)

#         for xi, yi in zip(x, y):
#             ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
#                         xytext=(0, 11), ha="center", fontsize=9.5,
#                         color=m["color"], fontweight="bold")

#         clean_box_lines.append(f"{m['label']}: {m['data']['clean_mean']:.3f} ± {m['data']['clean_std']:.3f}")
#         summary_rows.append((m["label"], x, y, s))

#     ax.set_xlabel("Mask Ratio (fraction of neighborhood corrupted)", fontsize=11)
#     ax.set_ylabel("Performance Retention (PCC)", fontsize=12)
#     ax.set_xticks(sorted(set(np.concatenate([sv_x for _, sv_x, _, _ in summary_rows]))))
#     ax.grid(axis="y", linestyle="--", alpha=0.4)
#     ax.set_ylim(0.3, 1.1)
#     ax.legend(loc="lower left", fontsize=11, frameon=True)

#     ax.text(0.97, 0.97, "\n".join(clean_box_lines), transform=ax.transAxes,
#             fontsize=9.5, va="top", ha="right",
#             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#888888"))

#     fig.text(0.5, -0.02,
#               "Shaded band = pooled mean ± 1 std (across radii and corruption seeds).",
#               ha="center", fontsize=9, color="#666666")

#     print("\n=== [severity view] Retention by mask_ratio (pooled across radii) ===")
#     header = f"{'Model':<22}" + "".join(f"mr={mr:<10.2f}" for mr in summary_rows[0][1][1:])
#     print(header)
#     for label, x, y, s in summary_rows:
#         row = f"{label:<22}"
#         for yi, si in zip(y[1:], s[1:]):
#             row += f"{yi:.3f}±{si:.3f}  "
#         print(row)

#     fig.tight_layout(rect=[0, 0.02, 1, 0.93])
#     fig.savefig(out_path, dpi=200, bbox_inches="tight")
#     print(f"[*] Saved severity-view figure to {out_path}")
#     return fig


# def plot_retention(models: list, corruption_type: str, out_path: str,
#                     title: str = None, y_min: float = 0.5, y_max: float = 1.1):
#     """
#     models: list of dicts, each:
#         {'label': str, 'color': str, 'marker': str, 'data': <load_model_csv output>}
#     """
#     all_mask_ratios = sorted(set().union(*[m["data"]["by_mask_ratio"].keys() for m in models]))
#     n_panels = len(all_mask_ratios)

#     fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 5.6), sharey=True)
#     if n_panels == 1:
#         axes = [axes]

#     corruption_label = {
#         "zero": "Zero (Missing Morphology)",
#         "gaussian": "Gaussian Noise",
#         "dropout": "Feature Dropout",
#         "blur": "Blur",
#     }.get(corruption_type, corruption_type.capitalize())

#     fig.suptitle(title or f"Performance Retention under {corruption_label} Corruption",
#                  fontsize=18, fontweight="bold", y=1.06)
#     fig.text(0.5, 1.00,
#               r"Retention = PCC$_{corrupted}$ / PCC$_{clean}$   (Higher is better)",
#               ha="center", fontsize=12, color="#444444")

#     legend_handles = []
#     for m in models:
#         h = plt.Line2D([0], [0], color=m["color"], marker=m["marker"], markersize=9,
#                         linewidth=2.2, label=m["label"])
#         legend_handles.append(h)
#     fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
#                ncol=len(models), frameon=False, fontsize=12)

#     for ax, mr in zip(axes, all_mask_ratios):
#         ax.set_title(f"Mask ratio = {int(mr * 100)}%", fontsize=13, fontweight="bold",
#                      bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8e8f5", edgecolor="none"))
#         ax.set_xlabel("Neighborhood Radius", fontsize=11)
#         ax.grid(axis="y", linestyle="--", alpha=0.4)
#         ax.set_ylim(y_min, y_max)

#         clean_box_lines = ["Clean PCC (radius=0)"]

#         for m in models:
#             data = m["data"]
#             if mr not in data["by_mask_ratio"]:
#                 continue
#             d = data["by_mask_ratio"][mr]
#             radii, ret_mean, ret_std = d["radii"], d["retention_mean"], d["retention_std"]

#             ax.plot(radii, ret_mean, color=m["color"], marker=m["marker"],
#                      markersize=8, linewidth=2.2, label=m["label"])
#             ax.fill_between(radii, ret_mean - ret_std, ret_mean + ret_std,
#                              color=m["color"], alpha=0.15)

#             for x, y in zip(radii, ret_mean):
#                 ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
#                             xytext=(0, 10 if y >= ret_mean.mean() else -16),
#                             ha="center", fontsize=9, color=m["color"], fontweight="bold")

#             clean_box_lines.append(
#                 f"{m['label']}: {data['clean_mean']:.3f} ± {data['clean_std']:.3f}"
#             )

#         ax.text(0.03, 0.03, "\n".join(clean_box_lines), transform=ax.transAxes,
#                  fontsize=9, va="bottom", ha="left",
#                  bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#888888"))

#     axes[0].set_ylabel("Performance Retention (PCC)", fontsize=12)

#     fig.text(0.5, -0.02,
#               "Shaded band = mean ± 1 std across corruption seeds. "
#               "All models are evaluated on the exact same corrupted inputs for each seed.",
#               ha="center", fontsize=9, color="#666666")

#     # ── Summary table beneath the plot ──
#     print("\n=== Average retention across radii (mean ± std) ===")
#     header = f"{'Model':<22}" + "".join(f"mask={mr:<12.2f}" for mr in all_mask_ratios)
#     print(header)
#     for m in models:
#         row = f"{m['label']:<22}"
#         for mr in all_mask_ratios:
#             d = m["data"]["by_mask_ratio"].get(mr)
#             if d is None:
#                 row += f"{'--':<17}"
#                 continue
#             vals = d["retention_mean"][1:]  # exclude synthetic radius=0 point
#             row += f"{vals.mean():.3f} ± {vals.std():.3f}  "
#         print(row)

#     fig.tight_layout(rect=[0, 0.02, 1, 0.94])
#     fig.savefig(out_path, dpi=200, bbox_inches="tight")
#     print(f"\n[*] Saved figure to {out_path}")
#     return fig


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--csv", action="append", required=True,
#                    help="Path to a summary.csv (repeatable — one per model)")
#     p.add_argument("--label", action="append", required=True,
#                    help="Legend label for the corresponding --csv (repeatable, same order)")
#     p.add_argument("--color", action="append", default=None,
#                    help="Line color for the corresponding --csv (repeatable, same order). "
#                         "If omitted, colors are auto-assigned.")
#     p.add_argument("--corruption_type", type=str, default="zero",
#                    choices=["zero", "gaussian", "dropout", "blur"])
#     p.add_argument("--mode", type=str, default="severity",
#                    choices=["severity", "radius", "both"],
#                    help="'severity' = primary figure (mask_ratio on x-axis, pooled over "
#                         "radius). 'radius' = supplementary figure (radius on x-axis, one "
#                         "panel per mask_ratio) — use this to justify pooling in the "
#                         "severity view. 'both' produces both (default).")
#     p.add_argument("--out", type=str, default="retention_plot",
#                    help="Output path WITHOUT extension when --mode both (suffixes "
#                         "'_severity.png' / '_radius_supp.png' are appended); used as-is "
#                         "for a single mode.")
#     p.add_argument("--ext", type=str, default="pdf", help="Output file extension")
#     p.add_argument("--title", type=str, default=None)
#     args = p.parse_args()

#     assert len(args.csv) == len(args.label), "--csv and --label counts must match"
#     if args.color is not None:
#         assert len(args.color) == len(args.csv), "--color count must match --csv count"

#     plt.rcParams.update({"font.family": "DejaVu Sans"})

#     models = []
#     for i, (csv_path, label) in enumerate(zip(args.csv, args.label)):
#         color = args.color[i] if args.color else DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
#         marker = DEFAULT_MARKERS[i % len(DEFAULT_MARKERS)]
#         data = load_model_csv(csv_path, args.corruption_type)
#         models.append({"label": label, "color": color, "marker": marker, "data": data})

#     if args.mode in ("severity", "both"):
#         out = f"{args.out}_severity.{args.ext}" if args.mode == "both" else f"{args.out}.{args.ext}"
#         plot_severity_view(models, args.corruption_type, out, title=args.title)

#     if args.mode in ("radius", "both"):
#         out = f"{args.out}_radius_supp.{args.ext}" if args.mode == "both" else f"{args.out}.{args.ext}"
#         plot_retention(models, args.corruption_type, out, title=args.title)


# if __name__ == "__main__":
#     main()

"""
plot_corruption_retention.py

Reproduces the "Performance Retention under Corruption" figure from one or
more corruption-evaluation results, where each model's results can be given
either as:

  (a) a single summary.csv (the original behavior), or
  (b) a FOLDER — e.g. results/corruption_eval/LUNG — containing per-split
      subfolders (split0/summary.csv, split1/summary.csv, ...), matching the
      output layout of evaluate.py's run_corruption_evaluation_all(). All
      split summaries found under the folder are POOLED into one combined
      summary (mean/std/count combined via pooled_mean_std_count, accounting
      for both within-split and between-split variance) before computing
      retention — so a single k-fold result for a dataset is one line on
      the plot, not one line per split.

Which mode is used is auto-detected from whether --path points at a file or
a directory; both can be mixed freely across --path invocations in the same
plot (e.g. one model given as a folder of splits, another given as a single
pre-aggregated csv).

Two plotting modes:

  --mode severity  (PRIMARY / main-text figure)
      x-axis = mask_ratio (corruption severity), one line per model,
      POOLED across all radii (since radius has minimal effect — see the
      'radius' mode below for the evidence that justifies pooling). This is
      the figure that should carry your headline claim: retention should
      drop, and the model comparison should widen, as severity increases.

  --mode radius    (SUPPLEMENTARY / appendix figure)
      x-axis = radius, one panel per mask_ratio, one line per model.
      Included specifically to demonstrate that pooling over radius in the
      severity view is justified (i.e. radius doesn't materially change the
      result at fixed severity).

  --mode both (default) produces both figures.

Retention is computed per model as:
    retention = PCC_mean(corrupted) / PCC_mean(clean baseline, i.e. the
                'none' row for that model)

Std is propagated via standard ratio error propagation. When pooling across
splits (folder mode) and/or across radii (severity mode), corrupted
mean/std/count are first combined via the standard pooled-sample formula
(accounts for both within-group and between-group variance), then
propagated against the clean baseline (which is itself pooled across splits
the same way, if given as a folder).

If the resulting clean baseline has no std (e.g. a single split, single
run), its contribution to the propagated variance is treated as 0 — this
UNDERESTIMATES the true retention uncertainty. For a submission-quality
figure, use the folder mode with multiple splits (and multiple
--n_corrupt_seeds at evaluation time) so PCC_std for the 'none' row is
non-trivial at both levels of pooling.

Usage
-----
    # Single pre-aggregated csv per model (original behavior):
    python plot_corruption_retention.py \\
        --path model_a_summary.csv --label "MOSAIC (ours)" --color "#2255aa" \\
        --path model_b_summary.csv --label "STFlow (baseline)" --color "#cc3333" \\
        --corruption_type zero --mode both --out retention_plot

    # Folder mode — pools every split*/summary.csv found under each dataset
    # folder (matches evaluate.py's save_dir_root/{dataset}/ layout):
    python plot_corruption_retention.py \\
        --path results/corruption_eval/LUNG --label "MOSAIC (ours)" --color "#2255aa" \\
        --path results/corruption_eval_stflow/LUNG --label "STFlow (baseline)" --color "#cc3333" \\
        --corruption_type zero --mode both --out retention_plot_LUNG
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


DEFAULT_COLORS = ["#2255aa", "#cc3333", "#2c9e5b", "#a05bbf", "#e08214"]
DEFAULT_MARKERS = ["o", "s", "^", "D", "v"]

# Every numeric metric column that comes out of evaluate.py's
# `.agg(["mean", "std", "count"])` + flattened column naming.
POOLED_METRIC_PREFIXES = ["PCC", "MSE", "MAE"]


def pooled_mean_std_count(means: np.ndarray, stds: np.ndarray, counts: np.ndarray):
    """
    Combine several (mean, std, count) groups into one pooled (mean, std, count),
    accounting for both within-group and between-group variance:

        pooled_mean = sum(n_i * mean_i) / sum(n_i)
        pooled_var  = [ sum(n_i * std_i^2) + sum(n_i * (mean_i - pooled_mean)^2) ]
                      / sum(n_i)

    Used both to pool across radii (severity view) and, now, to pool across
    k-fold splits (folder mode) — same statistical operation either way:
    "combine several summarized sub-samples into one summarized sample."
    """
    counts = counts.astype(float)
    total_n = counts.sum()
    if total_n == 0:
        return float("nan"), float("nan"), 0.0
    pooled_mean = (counts * means).sum() / total_n
    within = (counts * stds ** 2).sum()
    between = (counts * (means - pooled_mean) ** 2).sum()
    pooled_var = (within + between) / total_n
    return pooled_mean, math.sqrt(max(pooled_var, 0.0)), total_n


# ── Folder discovery + cross-split pooling (new) ────────────────────────────


def discover_split_summaries(folder: Path) -> list:
    """
    Find every summary.csv beneath `folder`, matching evaluate.py's output
    layout: {folder}/split{id}/summary.csv. Falls back to a recursive search
    so nonstandard layouts (e.g. a folder that directly contains one
    summary.csv with no split subfolder) still work.
    """
    folder = Path(folder)
    direct = sorted(folder.glob("split*/summary.csv"))
    if direct:
        return direct

    # fallback: a single summary.csv directly in the folder, or any nested layout
    recursive = sorted(folder.rglob("summary.csv"))
    if recursive:
        return recursive

    raise FileNotFoundError(
        f"No summary.csv found under {folder} "
        f"(looked for {folder}/split*/summary.csv and {folder}/**/summary.csv)"
    )


def pool_split_summaries(csv_paths: list) -> pd.DataFrame:
    """
    Load several summary.csv files (one per split) and pool them into a
    single summary DataFrame with the same schema — grouped by
    (corruption_type, radius, mask_ratio), each metric's mean/std/count
    combined via pooled_mean_std_count across splits.

    Assumes each input csv has columns produced by evaluate.py:
        corruption_type, radius, mask_ratio,
        PCC_mean, PCC_std, PCC_count,
        MSE_mean, MSE_std, MSE_count,
        MAE_mean, MAE_std, MAE_count
    Missing metric columns are skipped gracefully (only PCC is required
    downstream by the retention plot).
    """
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df["_split_source"] = str(p)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    group_cols = ["corruption_type", "radius", "mask_ratio"]
    present_metrics = [m for m in POOLED_METRIC_PREFIXES if f"{m}_mean" in combined.columns]
    if not present_metrics:
        raise ValueError(
            f"None of the expected metric columns ({POOLED_METRIC_PREFIXES}) "
            f"found in {csv_paths[0]} — got columns: {list(combined.columns)}"
        )

    rows = []
    for key, group in combined.groupby(group_cols):
        row = dict(zip(group_cols, key))
        for metric in present_metrics:
            means = group[f"{metric}_mean"].to_numpy(dtype=float)
            stds = group[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
            counts = group[f"{metric}_count"].to_numpy(dtype=float)
            p_mean, p_std, p_count = pooled_mean_std_count(means, stds, counts)
            row[f"{metric}_mean"] = p_mean
            row[f"{metric}_std"] = p_std
            row[f"{metric}_count"] = p_count
        rows.append(row)

    pooled_df = pd.DataFrame(rows)
    print(f"[*] Pooled {len(csv_paths)} split summaries "
          f"({', '.join(p.parent.name for p in csv_paths)}) "
          f"into {len(pooled_df)} (corruption_type, radius, mask_ratio) rows")
    return pooled_df


def load_summary_dataframe(path: str, corruption_type_for_check: str = None) -> pd.DataFrame:
    """
    Resolve a --path argument (file or folder) into a single summary
    DataFrame, transparently pooling across splits if given a folder.
    """
    p = Path(path)
    if p.is_dir():
        split_csvs = discover_split_summaries(p)
        return pool_split_summaries(split_csvs)
    if p.is_file():
        return pd.read_csv(p)
    raise FileNotFoundError(f"--path {path} is neither an existing file nor a directory")


# ── Per-model retention loading (mostly unchanged, now takes a DataFrame) ──


def load_model_data(path: str, corruption_type: str) -> dict:
    """
    Returns a dict:
        {
          'clean_mean': float, 'clean_std': float,
          'by_mask_ratio': {
              mask_ratio: {'radii': [...], 'retention_mean': [...], 'retention_std': [...]}
          }
        }

    `path` may be a single summary.csv OR a folder of split*/summary.csv —
    see load_summary_dataframe.
    """
    df = load_summary_dataframe(path)

    clean_rows = df[df["corruption_type"] == "none"]
    if clean_rows.empty:
        raise ValueError(
            f"{path}: no 'none' (clean baseline) row found — "
            "cannot compute retention without a clean reference."
        )
    clean_mean = float(clean_rows["PCC_mean"].iloc[0])
    clean_std_raw = clean_rows["PCC_std"].iloc[0]
    clean_std = float(clean_std_raw) if pd.notna(clean_std_raw) else 0.0
    if clean_std == 0.0:
        print(f"[!] {path}: clean baseline has no std (single split/run). "
              "Retention error bars will UNDERESTIMATE true uncertainty. "
              "Use folder mode with multiple splits (and multiple "
              "--n_corrupt_seeds at eval time) for a rigorous figure.")

    corrupted = df[(df["corruption_type"] == corruption_type) & (df["radius"] > 0)]

    by_mask_ratio = {}
    for mr, group in corrupted.groupby("mask_ratio"):
        group = group.sort_values("radius")
        radii = group["radius"].to_numpy(dtype=float)
        corr_mean = group["PCC_mean"].to_numpy(dtype=float)
        corr_std = group["PCC_std"].fillna(0.0).to_numpy(dtype=float)

        retention_mean = corr_mean / clean_mean
        rel_var = (corr_std / np.where(corr_mean != 0, corr_mean, 1e-12)) ** 2 + \
                  (clean_std / clean_mean) ** 2
        retention_std = retention_mean * np.sqrt(rel_var)

        radii = np.concatenate([[0.0], radii])
        retention_mean = np.concatenate([[1.0], retention_mean])
        retention_std = np.concatenate([[0.0], retention_std])

        by_mask_ratio[mr] = {
            "radii": radii,
            "retention_mean": retention_mean,
            "retention_std": retention_std,
        }

    return {
        "clean_mean": clean_mean,
        "clean_std": clean_std,
        "by_mask_ratio": by_mask_ratio,
        "corrupted_df": corrupted,
    }


def build_severity_view(data: dict) -> dict:
    """
    Pools each mask_ratio's per-radius cells into a single (mean, std, count),
    then computes retention against the clean baseline. Returns:
        {
          'mask_ratios': [...],
          'retention_mean': [...],
          'retention_std': [...],
        }
    (with a synthetic mask_ratio=0, retention=1.0 point prepended)
    """
    clean_mean, clean_std = data["clean_mean"], data["clean_std"]
    df = data["corrupted_df"]

    mask_ratios, ret_means, ret_stds = [], [], []
    for mr, group in df.groupby("mask_ratio"):
        means = group["PCC_mean"].to_numpy(dtype=float)
        stds = group["PCC_std"].fillna(0.0).to_numpy(dtype=float)
        counts = group["PCC_count"].to_numpy(dtype=float)

        pooled_mean, pooled_std, _ = pooled_mean_std_count(means, stds, counts)

        retention_mean = pooled_mean / clean_mean
        rel_var = (pooled_std / max(pooled_mean, 1e-12)) ** 2 + (clean_std / clean_mean) ** 2
        retention_std = retention_mean * math.sqrt(rel_var)

        mask_ratios.append(mr)
        ret_means.append(retention_mean)
        ret_stds.append(retention_std)

    order = np.argsort(mask_ratios)
    mask_ratios = np.array(mask_ratios)[order]
    ret_means = np.array(ret_means)[order]
    ret_stds = np.array(ret_stds)[order]

    mask_ratios = np.concatenate([[0.0], mask_ratios])
    ret_means = np.concatenate([[1.0], ret_means])
    ret_stds = np.concatenate([[0.0], ret_stds])

    return {"mask_ratios": mask_ratios, "retention_mean": ret_means, "retention_std": ret_stds}


def plot_severity_view(models: list, corruption_type: str, out_path: str, title: str = None):
    """
    PRIMARY figure: single panel, x = mask_ratio (severity), pooled over radius
    (and, for folder-mode models, pooled over splits too).
    """
    corruption_label = {
        "zero": "Zero (Missing Morphology)",
        "gaussian": "Gaussian Noise",
        "dropout": "Feature Dropout",
        "blur": "Blur",
    }.get(corruption_type, corruption_type.capitalize())

    fig, ax = plt.subplots(figsize=(7.5, 6))
    fig.suptitle(title or f"Performance Retention vs. Corruption Severity ({corruption_label})",
                 fontsize=15, fontweight="bold", y=1.04)
    fig.text(0.5, 0.965,
              r"Retention = PCC$_{corrupted}$ / PCC$_{clean}$   (Higher is better).  "
              "Pooled across all neighborhood radii (and splits, if applicable).",
              ha="center", fontsize=10, color="#444444")

    clean_box_lines = ["Clean PCC (mask_ratio=0)"]
    summary_rows = []

    for m in models:
        sv = build_severity_view(m["data"])
        x, y, s = sv["mask_ratios"], sv["retention_mean"], sv["retention_std"]

        ax.plot(x, y, color=m["color"], marker=m["marker"], markersize=9,
                linewidth=2.4, label=m["label"])
        ax.fill_between(x, y - s, y + s, color=m["color"], alpha=0.15)

        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=9.5,
                        color=m["color"], fontweight="bold")

        clean_box_lines.append(f"{m['label']}: {m['data']['clean_mean']:.3f} ± {m['data']['clean_std']:.3f}")
        summary_rows.append((m["label"], x, y, s))

    ax.set_xlabel("Mask Ratio (fraction of neighborhood corrupted)", fontsize=11)
    ax.set_ylabel("Performance Retention (PCC)", fontsize=12)
    ax.set_xticks(sorted(set(np.concatenate([sv_x for _, sv_x, _, _ in summary_rows]))))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0.3, 1.1)
    ax.legend(loc="lower left", fontsize=11, frameon=True)

    ax.text(0.97, 0.97, "\n".join(clean_box_lines), transform=ax.transAxes,
            fontsize=9.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#888888"))

    fig.text(0.5, -0.02,
              "Shaded band = pooled mean ± 1 std (across radii, corruption seeds, "
              "and splits where applicable).",
              ha="center", fontsize=9, color="#666666")

    print("\n=== [severity view] Retention by mask_ratio (pooled across radii/splits) ===")
    header = f"{'Model':<22}" + "".join(f"mr={mr:<10.2f}" for mr in summary_rows[0][1][1:])
    print(header)
    for label, x, y, s in summary_rows:
        row = f"{label:<22}"
        for yi, si in zip(y[1:], s[1:]):
            row += f"{yi:.3f}±{si:.3f}  "
        print(row)

    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[*] Saved severity-view figure to {out_path}")
    return fig


def plot_retention(models: list, corruption_type: str, out_path: str,
                    title: str = None, y_min: float = 0.5, y_max: float = 1.1):
    """
    models: list of dicts, each:
        {'label': str, 'color': str, 'marker': str, 'data': <load_model_data output>}
    """
    all_mask_ratios = sorted(set().union(*[m["data"]["by_mask_ratio"].keys() for m in models]))
    n_panels = len(all_mask_ratios)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 5.6), sharey=True)
    if n_panels == 1:
        axes = [axes]

    corruption_label = {
        "zero": "Zero (Missing Morphology)",
        "gaussian": "Gaussian Noise",
        "dropout": "Feature Dropout",
        "blur": "Blur",
    }.get(corruption_type, corruption_type.capitalize())

    fig.suptitle(title or f"Performance Retention under {corruption_label} Corruption",
                 fontsize=18, fontweight="bold", y=1.06)
    fig.text(0.5, 1.00,
              r"Retention = PCC$_{corrupted}$ / PCC$_{clean}$   (Higher is better)",
              ha="center", fontsize=12, color="#444444")

    legend_handles = []
    for m in models:
        h = plt.Line2D([0], [0], color=m["color"], marker=m["marker"], markersize=9,
                        linewidth=2.2, label=m["label"])
        legend_handles.append(h)
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=len(models), frameon=False, fontsize=12)

    for ax, mr in zip(axes, all_mask_ratios):
        ax.set_title(f"Mask ratio = {int(mr * 100)}%", fontsize=13, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8e8f5", edgecolor="none"))
        ax.set_xlabel("Neighborhood Radius", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(y_min, y_max)

        clean_box_lines = ["Clean PCC (radius=0)"]

        for m in models:
            data = m["data"]
            if mr not in data["by_mask_ratio"]:
                continue
            d = data["by_mask_ratio"][mr]
            radii, ret_mean, ret_std = d["radii"], d["retention_mean"], d["retention_std"]

            ax.plot(radii, ret_mean, color=m["color"], marker=m["marker"],
                     markersize=8, linewidth=2.2, label=m["label"])
            ax.fill_between(radii, ret_mean - ret_std, ret_mean + ret_std,
                             color=m["color"], alpha=0.15)

            for x, y in zip(radii, ret_mean):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                            xytext=(0, 10 if y >= ret_mean.mean() else -16),
                            ha="center", fontsize=9, color=m["color"], fontweight="bold")

            clean_box_lines.append(
                f"{m['label']}: {data['clean_mean']:.3f} ± {data['clean_std']:.3f}"
            )

        ax.text(0.03, 0.03, "\n".join(clean_box_lines), transform=ax.transAxes,
                 fontsize=9, va="bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#888888"))

    axes[0].set_ylabel("Performance Retention (PCC)", fontsize=12)

    fig.text(0.5, -0.02,
              "Shaded band = mean ± 1 std across corruption seeds (and splits, where "
              "applicable). All models are evaluated on the exact same corrupted "
              "inputs for each seed.",
              ha="center", fontsize=9, color="#666666")

    print("\n=== Average retention across radii (mean ± std) ===")
    header = f"{'Model':<22}" + "".join(f"mask={mr:<12.2f}" for mr in all_mask_ratios)
    print(header)
    for m in models:
        row = f"{m['label']:<22}"
        for mr in all_mask_ratios:
            d = m["data"]["by_mask_ratio"].get(mr)
            if d is None:
                row += f"{'--':<17}"
                continue
            vals = d["retention_mean"][1:]  # exclude synthetic radius=0 point
            row += f"{vals.mean():.3f} ± {vals.std():.3f}  "
        print(row)

    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n[*] Saved figure to {out_path}")
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", action="append", required=True,
                   help="Path to EITHER a single summary.csv, OR a folder "
                        "(e.g. results/corruption_eval/LUNG) containing "
                        "split*/summary.csv subfolders to pool across "
                        "(repeatable — one --path per model).")
    p.add_argument("--label", action="append", required=True,
                   help="Legend label for the corresponding --path (repeatable, same order)")
    p.add_argument("--color", action="append", default=None,
                   help="Line color for the corresponding --path (repeatable, same order). "
                        "If omitted, colors are auto-assigned.")
    p.add_argument("--corruption_type", type=str, default="zero",
                   choices=["zero", "gaussian", "dropout", "blur"])
    p.add_argument("--mode", type=str, default="severity",
                   choices=["severity", "radius", "both"],
                   help="'severity' = primary figure (mask_ratio on x-axis, pooled over "
                        "radius). 'radius' = supplementary figure (radius on x-axis, one "
                        "panel per mask_ratio) — use this to justify pooling in the "
                        "severity view. 'both' produces both (default).")
    p.add_argument("--out", type=str, default="retention_plot",
                   help="Output path WITHOUT extension when --mode both (suffixes "
                        "'_severity.png' / '_radius_supp.png' are appended); used as-is "
                        "for a single mode.")
    p.add_argument("--ext", type=str, default="pdf", help="Output file extension")
    p.add_argument("--title", type=str, default=None)
    args = p.parse_args()

    assert len(args.path) == len(args.label), "--path and --label counts must match"
    if args.color is not None:
        assert len(args.color) == len(args.path), "--color count must match --path count"

    plt.rcParams.update({"font.family": "DejaVu Sans"})

    models = []
    for i, (path, label) in enumerate(zip(args.path, args.label)):
        color = args.color[i] if args.color else DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        marker = DEFAULT_MARKERS[i % len(DEFAULT_MARKERS)]
        data = load_model_data(path, args.corruption_type)
        models.append({"label": label, "color": color, "marker": marker, "data": data})

    if args.mode in ("severity", "both"):
        out = f"{args.out}_severity.{args.ext}" if args.mode == "both" else f"{args.out}.{args.ext}"
        plot_severity_view(models, args.corruption_type, out, title=args.title)

    if args.mode in ("radius", "both"):
        out = f"{args.out}_radius_supp.{args.ext}" if args.mode == "both" else f"{args.out}.{args.ext}"
        plot_retention(models, args.corruption_type, out, title=args.title)


if __name__ == "__main__":
    main()
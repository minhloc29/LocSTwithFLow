"""
check_annotations.py

Run this FIRST, before building the full spatial-domain-recovery pipeline.

Checks every slide's .h5ad file for columns in `adata.obs` that look like
pathologist / tissue-region annotations (as opposed to just raw expression
+ coordinates). This determines which ground truth you actually have:

  - If real annotation columns exist -> ARI/NMI against them is a STRONG,
    independent primary metric (Option 1 from our discussion).
  - If nothing exists -> you only have Option 2 (cluster ground-truth
    expression itself as a proxy target) -> weaker, keep as secondary,
    not primary.

Common column name patterns to look for in HEST-1k / Visium-derived data:
    'annotation', 'region', 'domain', 'tissue_type', 'pathologist_annotation',
    'histology_type', 'cluster', 'layer', 'zone'
(exact naming varies a lot by dataset/source — this script flags anything
that looks categorical and isn't obviously a technical/QC column, so you
can eyeball the candidates rather than guessing exact names blindly.)

Usage
-----
    python hmflow/app/check_annotations.py \
        --source_dataroot <path> \
        --datasets LUNG HCC COAD \
        --n_samples_per_dataset 3
"""

import os
import argparse
import anndata as ad
import pandas as pd


# Columns that are almost always technical/QC, not biological annotation —
# used to de-prioritize obvious noise from the candidate list (not a hard
# filter, since dataset-specific naming varies).
LIKELY_NOT_ANNOTATION = {
    "in_tissue", "array_row", "array_col", "n_genes", "n_counts",
    "total_counts", "pct_counts_mt", "barcode", "sample", "sample_id",
    "batch", "n_genes_by_counts", "log1p_total_counts", "log1p_n_genes_by_counts",
}


def inspect_h5ad(h5ad_path: str) -> dict:
    """
    Returns a dict summarizing candidate annotation columns found in this
    slide's adata.obs, without loading the full expression matrix into
    memory unnecessarily (backed mode where possible).
    """
    try:
        adata = ad.read_h5ad(h5ad_path, backed="r")
    except Exception as e:
        return {"error": str(e)}

    obs = adata.obs
    candidates = []
    for col in obs.columns:
        if col in LIKELY_NOT_ANNOTATION:
            continue
        series = obs[col]
        # Heuristic: categorical/object dtype with a small-ish number of
        # unique values looks like a region/domain label, not a continuous
        # QC metric or a near-unique identifier.
        n_unique = series.nunique(dropna=True)
        is_categorical_like = (
            series.dtype.name in ("category", "object", "bool")
            or (pd.api.types.is_integer_dtype(series) and n_unique <= 30)
        )
        if is_categorical_like and 1 < n_unique <= 30:
            candidates.append({
                "column": col,
                "n_unique": n_unique,
                "example_values": series.dropna().unique()[:5].tolist(),
                "n_non_null": series.notna().sum(),
                "n_total": len(series),
            })

    return {
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "all_obs_columns": list(obs.columns),
        "annotation_candidates": candidates,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_dataroot", required=True,
                   help="root containing <dataset>/adata/<slide>.h5ad, matching "
                        "the repo's train.py source_dataroot layout")
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--n_samples_per_dataset", type=int, default=3,
                   help="How many slides to check per dataset (checking all "
                        "of them is usually unnecessary and slow).")
    args = p.parse_args()

    report_rows = []

    for dataset in args.datasets:
        adata_dir = os.path.join(args.source_dataroot, dataset, "adata")
        if not os.path.isdir(adata_dir):
            print(f"[!] {dataset}: no adata directory found at {adata_dir}, skipping")
            continue

        h5ad_files = sorted(f for f in os.listdir(adata_dir) if f.endswith(".h5ad"))
        sample_files = h5ad_files[:args.n_samples_per_dataset]

        if not sample_files:
            print(f"[!] {dataset}: adata dir exists at {adata_dir} but contains no .h5ad files")
            continue

        print(f"\n{'='*70}\n{dataset}  ({len(h5ad_files)} total slides, checking {len(sample_files)})\n{'='*70}")

        for fname in sample_files:
            path = os.path.join(adata_dir, fname)
            result = inspect_h5ad(path)

            if "error" in result:
                print(f"  [!] {fname}: ERROR reading file — {result['error']}")
                continue

            print(f"\n  {fname}  (n_obs={result['n_obs']}, n_vars={result['n_vars']})")
            print(f"    All obs columns: {result['all_obs_columns']}")

            if result["annotation_candidates"]:
                print(f"    >>> CANDIDATE ANNOTATION COLUMNS FOUND:")
                for c in result["annotation_candidates"]:
                    coverage = c["n_non_null"] / c["n_total"] * 100
                    print(f"        - '{c['column']}': {c['n_unique']} unique values, "
                          f"{coverage:.0f}% coverage, examples={c['example_values']}")
                    report_rows.append({
                        "dataset": dataset, "file": fname,
                        "column": c["column"], "n_unique": c["n_unique"],
                        "coverage_pct": coverage,
                    })
            else:
                print(f"    >>> NO annotation-like columns found — only expression/coords available.")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    if report_rows:
        df = pd.DataFrame(report_rows)
        print(df.to_string(index=False))
        print(f"\n[*] Found candidate annotation columns in {df['dataset'].nunique()} "
              f"dataset(s): {sorted(df['dataset'].unique())}")
        print("[*] NEXT STEP: manually inspect these columns' actual values "
              "(open a slide's h5ad directly) to confirm they really are "
              "pathologist-style region labels, not just technical metadata "
              "that happened to look categorical.")
    else:
        print("[!] No annotation-like columns found in ANY sampled slide.")
        print("[!] You likely only have Option 2 available: cluster ground-truth")
        print("    expression itself as the reference target for ARI/NMI, which")
        print("    is a weaker, more self-referential result — recommend keeping")
        print("    this as a SECONDARY metric, not primary, unless you can source")
        print("    external region annotations for these tissue types elsewhere.")


if __name__ == "__main__":
    main()

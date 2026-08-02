"""
Corruption helpers for local-morphology robustness experiments.

Provides `apply_local_artifact()` and multiple corruption types that
simulate real histopathology artifacts (tissue folds, staining noise,
missing tissue, blur).  All operations are deterministic given a seed.
"""

import torch
import numpy as np
from typing import Optional, Literal, Tuple


CorruptionType = Literal["zero", "gaussian", "dropout", "blur"]


@torch.no_grad()
def apply_local_artifact(
    img_features: torch.Tensor,
    coords: torch.Tensor,
    radius: float,
    mask_ratio: float,
    corruption_type: CorruptionType = "zero",
    sigma: float = 0.5,
    dropout_p: float = 0.5,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Corrupt image features locally by simulating pathology artifacts.

    For each item in the batch:
      1. Sample K = ceil(mask_ratio * N) target spots uniformly at random.
      2. For each target spot, find all neighbours within Euclidean distance
         *radius* (using *coords*).
      3. Corrupt the image features of those neighbours according to
         *corruption_type*.

    Parameters
    ----------
    img_features :  torch.Tensor  [B, N, feature_dim]
        Original image features (e.g. from a foundation model).
    coords :        torch.Tensor  [B, N, 2]
        Spatial coordinates (decentred or raw).
    radius :        float
        Neighbourhood radius in coordinate units.  0 = single-spot masking.
    mask_ratio :    float
        Fraction of spots to use as artifact centres (0 < mask_ratio ≤ 1).
    corruption_type : str
        ``"zero"``     — set features to 0            (tissue fold / tear)
        ``"gaussian"`` — add Gaussian noise           (staining failure)
        ``"dropout"``  — random feature dropout        (missing tissue)
        ``"blur"``     — neighbour-feature averaging   (scanner blur)
    sigma :         float
        Standard deviation of Gaussian noise (only for ``"gaussian"``).
    dropout_p :     float
        Per-feature dropout probability (only for ``"dropout"``).
    seed :          int, optional
        Random seed for reproducibility.

    Returns
    -------
    torch.Tensor  [B, N, feature_dim]
        Corrupted image features (modification in-place on a copy).
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    B, N, feat_dim = img_features.shape
    device = img_features.device
    out = img_features.clone()

    if mask_ratio <= 0.0 or radius < 0:
        return out  # no corruption

    for b in range(B):
        n_artifacts = max(1, int(np.ceil(mask_ratio * N)))

        # 1. Sample target spots (the centres of "artifacts")
        target_idx = torch.randperm(N, device=device)[:n_artifacts]  # [K]

        # 2. Build pairwise distance matrix for this batch element
        #    coords[b]: [N, 2]
        diff = coords[b].unsqueeze(0) - coords[b].unsqueeze(1)       # [N, N, 2]
        dist = diff.norm(dim=-1)                                      # [N, N]

        # 3. For every target, find neighbours within radius
        k = min(int(radius), N - 1)

        nearest_idx = dist.topk(k + 1, largest=False).indices

        mask = torch.zeros(N, dtype=torch.bool, device=device)

        perm = torch.randperm(N, device=device)

        target_corrupted = int(mask_ratio * N)

        for center in perm:

            if radius == 0:
                mask[center] = True
            else:
                mask[nearest_idx[center]] = True

            if mask.sum() >= target_corrupted:
                break

        idx = mask  # boolean index of spots to corrupt

        if idx.sum() == 0:
            continue

        if corruption_type == "zero":
            out[b, idx] = 0.0

        elif corruption_type == "gaussian":
            noise = torch.randn_like(out[b, idx]) * sigma
            out[b, idx] = out[b, idx] + noise

        elif corruption_type == "dropout":
            keep = torch.rand_like(out[b, idx]) > dropout_p
            out[b, idx] = out[b, idx] * keep

        elif corruption_type == "blur":
            # For each corrupted spot, average features of its uncorrupted neighbours
            for i in torch.where(idx)[0]:
                # Find neighbours of spot i that are NOT masked
                nbr_mask = (dist[i] <= radius) & ~idx
                nbr_mask[i] = False  # exclude self
                nbr_indices = torch.where(nbr_mask)[0]
                if nbr_indices.numel() > 0:
                    out[b, i] = img_features[b, nbr_indices].mean(dim=0)
                # else: no clean neighbours — leave as-is

        else:
            raise ValueError(f"Unknown corruption_type: {corruption_type}")
        # 1. Coordinate scale
        print(
            "Distance stats:",
            dist.min().item(),
            dist.mean().item(),
            dist.max().item(),
        )

        # 2. Average neighbors within each radius
        for r in [0, 32, 64, 96, 128]:
            avg_neighbors = (dist <= r).float().sum(-1).mean().item()
            print(f"Radius {r}: {avg_neighbors:.1f} neighbors")

        # 3. Feature statistics
        print(
            "Feature mean/std:",
            img_features.mean().item(),
            img_features.std().item(),
        )
    return out


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute Pearson correlation coefficient, MSE, and MAE.

    Parameters
    ----------
    pred :   np.ndarray  [N, n_genes]
    target : np.ndarray  [N, n_genes]

    Returns
    -------
    pcc :  float   mean Pearson r across genes
    mse :  float   mean squared error
    mae :  float   mean absolute error
    """
    from scipy.stats import pearsonr

    n_genes = target.shape[1]
    pcc_vals = []
    for g in range(n_genes):
        r, _ = pearsonr(target[:, g], pred[:, g])
        pcc_vals.append(r if not np.isnan(r) else 0.0)

    pcc = float(np.mean(pcc_vals))
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    return pcc, mse, mae


# ── convenience: grid of (radius, mask_ratio) ──────────────────────────────

DEFAULT_RADII = [0, 32, 64, 96, 128]
DEFAULT_MASK_RATIOS = [0.1, 0.25, 0.5]
DEFAULT_CORRUPTION_TYPES: list[CorruptionType] = ["zero", "gaussian", "dropout", "blur"]

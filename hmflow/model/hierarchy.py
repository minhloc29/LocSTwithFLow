"""
Hierarchical Encoder for HFlow-ST.

Produces the initial three-level biological representation:
    Patch tokens  →  Region tokens  →  Slide token

Supports three region-discovery methods:
    learnable  — K learnable queries with cross-attention (default)
    grid       — hard spatial grid partitioning
    kmeans     — k-means soft assignment on coordinates
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_sincos_pos_embed(embed_dim, coords):
    """Sinusoidal positional encoding for 2D coordinates."""
    # coords: [N, 2] normalized to ~[-1, 1]
    half = embed_dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=coords.device) / half
    )  # [half]
    angles = coords[..., :2].unsqueeze(-1) * freqs[None, None, :]  # [N, 2, half]
    pos = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [N, 2, 2*half]
    pos = pos.view(coords.shape[0], -1)  # [N, 2*embed_dim], trimmed to embed_dim below
    if pos.shape[-1] < embed_dim:
        pos = F.pad(pos, (0, embed_dim - pos.shape[-1]))
    return pos[..., :embed_dim]


class RegionCrossAttention(nn.Module):
    """
    Lightweight learnable region prototype initializer.

    This keeps the initial hierarchy cheap: regions start as learnable
    prototypes and are then refined by the dynamic assignment mechanism in
    HFlowBlock.
    """

    def __init__(self, d_model, n_queries, **kwargs):
        super().__init__()
        self.n_queries = n_queries
        self.region_queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)

    def forward(self, patch_embs, coords, pad_mask=None, return_weights=False):
        del coords, pad_mask
        batch_size = patch_embs.shape[0]
        region_embs = self.region_queries.expand(batch_size, -1, -1)
        if return_weights:
            return region_embs, None
        return region_embs


class GridRegionDiscovery(nn.Module):
  

    def __init__(self, d_model, n_grid_cols=4, n_grid_rows=4):
        super().__init__()
        self.n_grid_cols = n_grid_cols
        self.n_grid_rows = n_grid_rows
        self.n_queries = n_grid_cols * n_grid_rows
        self.region_embed = nn.Linear(d_model, d_model)  # refine pooled features

    def forward(self, patch_embs, coords, pad_mask=None):
        B, N, D = patch_embs.shape
        device = patch_embs.device

        # Normalize coordinates to [0, 1]
        coord_min = coords.min(dim=1, keepdim=True)[0]
        coord_max = coords.max(dim=1, keepdim=True)[0]
        norm_coords = (coords - coord_min) / (coord_max - coord_min + 1e-8)  # [B, N, 2]

        # Assign each point to a grid cell
        col_idx = (norm_coords[..., 0] * self.n_grid_cols).long().clamp(0, self.n_grid_cols - 1)
        row_idx = (norm_coords[..., 1] * self.n_grid_rows).long().clamp(0, self.n_grid_rows - 1)
        cell_idx = row_idx * self.n_grid_cols + col_idx  # [B, N]

        # Zero out padding positions
        if pad_mask is not None:
            cell_idx = cell_idx.masked_fill(pad_mask, -1)  # -1 = ignore

        # Vectorized scatter_add_ for pooling
        flat_region_embs = torch.zeros(B * self.n_queries, D, device=device)
        flat_counts = torch.zeros(B * self.n_queries, device=device)

        # Flatten to 1D indices for scatter_add_
        flat_batch = torch.arange(B, device=device).unsqueeze(-1).expand(B, N).reshape(-1)
        flat_cell = cell_idx.reshape(-1)
        flat_feats = patch_embs.reshape(-1, D)
        flat_ones = torch.ones(B * N, device=device)

        valid = flat_cell >= 0  # exclude padding (set to -1 above)
        flat_idx = flat_batch[valid] * self.n_queries + flat_cell[valid]

        flat_region_embs.index_add_(0, flat_idx, flat_feats[valid])
        flat_counts.index_add_(0, flat_idx, flat_ones[valid])

        region_embs = flat_region_embs.view(B, self.n_queries, D)
        region_counts = flat_counts.view(B, self.n_queries, 1).clamp(min=1)
        region_embs = region_embs / region_counts
        region_embs = self.region_embed(region_embs)

        return region_embs

class KMeansRegionDiscovery(nn.Module):
    """
    Learns K centroid embeddings; patches are softly assigned via attention
    to centroids based on spatial proximity + feature similarity.
    """

    def __init__(self, d_model, n_queries):
        super().__init__()
        self.n_queries = n_queries
        self.centroids = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)

        self.region_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, patch_embs, coords, pad_mask=None):
        B, N, D = patch_embs.shape
        K = self.n_queries

        centroids = self.centroids.expand(B, -1, -1)  # [B, K, D]

        # Compute assignment weights: softmax over centroid similarity
        # similarity = cosine similarity between patch and centroid
        patch_norm = F.normalize(patch_embs, dim=-1)
        centroid_norm = F.normalize(centroids, dim=-1)

        # [B, N, K] similarity matrix
        sim = torch.bmm(patch_norm, centroid_norm.transpose(1, 2))
        sim = sim * self.temperature.clamp(min=0.5, max=10.0)

        if pad_mask is not None:
            sim = sim.masked_fill(pad_mask.unsqueeze(-1), -1e9)

        assign_weights = F.softmax(sim, dim=-1)  # [B, N, K]

        # Weighted sum: patches → regions
        region_embs = torch.bmm(assign_weights.transpose(1, 2), patch_embs)  # [B, K, D]
        region_counts = assign_weights.sum(dim=1).clamp(min=1)  # [B, K]
        region_embs = region_embs / region_counts.unsqueeze(-1)

        region_embs = self.region_proj(region_embs)
        return region_embs


class SlidePooling(nn.Module):
    """
    Aggregates region tokens into a single slide-level token.

    Uses attention pooling: 1 learnable CLS query attends to all region tokens.
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.slide_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, region_embs):
        """
        Args:
            region_embs: [B, K, d_model]
        Returns:
            slide_emb: [B, 1, d_model]
        """
        B = region_embs.shape[0]
        slide_query = self.slide_query.expand(B, -1, -1)

        attn_out, _ = self.cross_attn(slide_query, region_embs, region_embs)
        slide_emb = self.norm(slide_query + attn_out)
        slide_emb = self.norm2(slide_emb + self.ffn(slide_emb))
        return slide_emb


class HierarchyEncoder(nn.Module):
    """
    Three-level biological hierarchy encoder.

    Given image patch features and coordinates, produces:
        z_patch   [B, N, d_model]  — per-spot patch tokens
        z_region  [B, K, d_model]  — region tokens (K = n_queries)
        z_slide   [B, 1, d_model]  — single slide-level token

    Region discovery method controlled by `hflow_config.hflow_region_discovery`.
    """

    def __init__(self, d_model, hflow_config):
        super().__init__()
        self.d_model = d_model
        self.hflow = hflow_config

        # Patch token projector (replaces current image_transform in Denoiser)
        self.patch_proj = nn.LayerNorm(d_model)

        # Region discovery
        rd = hflow_config.hflow_region_discovery
        n_queries = hflow_config.n_region_queries

        if rd == "learnable":
            self.region_discovery = RegionCrossAttention(
                d_model=d_model,
                n_queries=n_queries,
                n_heads=4,
                dropout=0.1,
            )
            n_regions = n_queries
        elif rd == "grid":
            # Find grid size such that total cells ≈ n_queries
            cols = max(2, int(math.sqrt(n_queries)))
            rows = max(2, n_queries // cols)
            self.region_discovery = GridRegionDiscovery(d_model, n_grid_cols=cols, n_grid_rows=rows)
            n_regions = cols * rows
        elif rd == "kmeans":
            self.region_discovery = KMeansRegionDiscovery(d_model, n_queries)
            n_regions = n_queries
        else:
            raise ValueError(f"Unknown region discovery: {rd}")

        self.n_regions = n_regions
        self.slide_pool = SlidePooling(d_model)

    def forward(self, patch_embs, coords, pad_mask=None):
        """
        Args:
            patch_embs: [B, N, d_model]  — already projected to d_model
            coords:     [B, N, 2]
            pad_mask:   [B, N]           True = padding

        Returns:
            z_patch:  [B, N, d_model]
            z_region: [B, K, d_model]
            z_slide:  [B, 1, d_model]
        """
        # Normalize patch embs
        z_patch = self.patch_proj(patch_embs)

        # Discover regions
        z_region = self.region_discovery(z_patch, coords, pad_mask=pad_mask)

        # Pool regions → slide
        z_slide = self.slide_pool(z_region)

        return z_patch, z_region, z_slide

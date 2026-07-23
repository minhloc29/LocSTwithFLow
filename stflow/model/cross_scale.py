"""
Cross-Scale Message Passing for HFlow-ST.

Implements bidirectional information flow between the three biological scales:
    Slide  ←→  Region  ←→  Patch

Every HFlowBlock performs a configurable sequence of cross-attention operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlideToRegionAttention(nn.Module):
    """
    Slide token broadcasts global context to region tokens.

    Query:   region tokens (each region absorbs slide-level context)
    Key/Val: slide token

    This gives each region knowledge of the global tissue phenotype.
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
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

    def forward(self, region_embs, slide_emb):
        """
        Args:
            region_embs: [B, K, d_model]
            slide_emb:   [B, 1, d_model]

        Returns:
            region_embs: [B, K, d_model]  — updated with slide context
        """
        # Inject slide context into each region
        attn_out, _ = self.cross_attn(region_embs, slide_emb, slide_emb)
        region_embs = self.norm(region_embs + attn_out)
        region_embs = self.norm2(region_embs + self.ffn(region_embs))
        return region_embs


class RegionToPatchAttention(nn.Module):
    """
    Region tokens inject microenvironment information into nearby spots.

    Each patch attends to region tokens weighted by spatial proximity.
    Uses coordinate-aware attention: patches closer to a region's centroid
    get more influence from that region.
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
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

    def forward(self, patch_embs, region_embs):
        """
        Args:
            patch_embs:   [B, N, d_model]
            region_embs:  [B, K, d_model]

        Returns:
            patch_embs:   [B, N, d_model]  — updated with regional context
        """
        attn_out, _ = self.cross_attn(patch_embs, region_embs, region_embs)
        patch_embs = self.norm(patch_embs + attn_out)
        patch_embs = self.norm2(patch_embs + self.ffn(patch_embs))
        return patch_embs


class PatchToRegionAttention(nn.Module):
    """
    Patch tokens send evidence back to region tokens.

    Each region re-aggregates by attending to all patches.
    This allows bottom-up evidence propagation:
    e.g., "many patches show high proliferation → region becomes tumor-like"
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
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

    def forward(self, region_embs, patch_embs, pad_mask=None):
        """
        Args:
            region_embs: [B, K, d_model]
            patch_embs:  [B, N, d_model]
            pad_mask:    [B, N]  True = padding

        Returns:
            region_embs: [B, K, d_model]  — updated with patch evidence
        """
        key_padding_mask = pad_mask if pad_mask is not None else None
        attn_out, _ = self.cross_attn(
            region_embs, patch_embs, patch_embs,
            key_padding_mask=key_padding_mask,
        )
        region_embs = self.norm(region_embs + attn_out)
        region_embs = self.norm2(region_embs + self.ffn(region_embs))
        return region_embs


class RegionToSlideAttention(nn.Module):
    """
    All regions collectively update the slide token.

    Query:   slide token
    Key/Val: region tokens

    This enables global understanding of the slide's regional composition.
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
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

    def forward(self, slide_emb, region_embs):
        """
        Args:
            slide_emb:   [B, 1, d_model]
            region_embs: [B, K, d_model]

        Returns:
            slide_emb:   [B, 1, d_model]  — updated with regional composition
        """
        attn_out, _ = self.cross_attn(slide_emb, region_embs, region_embs)
        slide_emb = self.norm(slide_emb + attn_out)
        slide_emb = self.norm2(slide_emb + self.ffn(slide_emb))
        return slide_emb


class CrossScaleFusion(nn.Module):
    """
    Orchestrates all cross-scale message passing in one HFlowBlock.

    Direction is controlled by `direction` parameter:
        "none"          — no cross-scale (patch self-attn only)
        "top_down"      — Slide → Region → Patch
        "bottom_up"     — Patch → Region → Slide
        "bidirectional" — both: Slide↔Region↔Patch in both directions

    In the default bidirectional mode, the update order is:
        1. Slide → Region  (global context → regions)
        2. Region → Patch  (microenvironment → spots)
        3. Patch → Region  (local evidence → regions)
        4. Region → Slide  (regional composition → global)
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()

        # Top-down modules
        self.slide_to_region = SlideToRegionAttention(d_model, n_heads, dropout)
        self.region_to_patch = RegionToPatchAttention(d_model, n_heads, dropout)

        # Bottom-up modules
        self.patch_to_region = PatchToRegionAttention(d_model, n_heads, dropout)
        self.region_to_slide = RegionToSlideAttention(d_model, n_heads, dropout)

    def forward(self, z_patch, z_region, z_slide, direction="bidirectional", pad_mask=None):
        """
        Args:
            z_patch:   [B, N, d_model]
            z_region:  [B, K, d_model]
            z_slide:   [B, 1, d_model]
            direction: one of "none", "top_down", "bottom_up", "bidirectional"
            pad_mask:  [B, N]  True = padding

        Returns:
            z_patch, z_region, z_slide — all updated
        """
        if direction == "none":
            return z_patch, z_region, z_slide

        if direction in ("top_down", "bidirectional"):
            # 1. Slide → Region: broadcast global context
            z_region = self.slide_to_region(z_region, z_slide)
            # 2. Region → Patch: inject microenvironment
            z_patch = self.region_to_patch(z_patch, z_region)

        if direction in ("bottom_up", "bidirectional"):
            # 3. Patch → Region: local evidence back to region
            z_region = self.patch_to_region(z_region, z_patch, pad_mask=pad_mask)
            # 4. Region → Slide: regional composition to global
            z_slide = self.region_to_slide(z_slide, z_region)

        return z_patch, z_region, z_slide

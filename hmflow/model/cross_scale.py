import torch
import torch.nn as nn
import torch.nn.functional as F


# def _masked_mean(x, pad_mask=None, dim=1):
#     if pad_mask is None:
#         return x.mean(dim=dim, keepdim=True)

#     valid_mask = (~pad_mask).unsqueeze(-1).to(dtype=x.dtype)
#     denom = valid_mask.sum(dim=dim, keepdim=True).clamp_min(1.0)
#     return (x * valid_mask).sum(dim=dim, keepdim=True) / denom


# class SlideToRegionAttention(nn.Module):
    

#     def __init__(self, d_model, n_heads=4, dropout=0.1):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(
#             d_model, n_heads, dropout=dropout, batch_first=True
#         )
#         self.gene_q = nn.Linear(d_model, d_model)
#         self.gene_k = nn.Linear(d_model, d_model)
#         self.gene_v = nn.Linear(d_model, d_model)
#         self.norm = nn.LayerNorm(d_model)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * 4, d_model),
#             nn.Dropout(dropout),
#         )
#         self.norm2 = nn.LayerNorm(d_model)

#     def forward(self, region_embs, slide_emb, gene_context=None, pad_mask=None):
       
#         if gene_context is not None:
#             gene_summary = _masked_mean(gene_context, pad_mask=pad_mask)
#             region_query = region_embs + self.gene_q(gene_summary).expand_as(region_embs)
#             slide_key = slide_emb + self.gene_k(gene_summary)
#             slide_value = slide_emb + self.gene_v(gene_summary)
#         else:
#             region_query = region_embs
#             slide_key = slide_emb
#             slide_value = slide_emb

#         attn_out, _ = self.cross_attn(region_query, slide_key, slide_value)
#         region_embs = self.norm(region_embs + attn_out)
#         region_embs = self.norm2(region_embs + self.ffn(region_embs))
#         return region_embs


# class RegionToPatchAttention(nn.Module):

#     def __init__(self, d_model, n_heads=4, dropout=0.1):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(
#             d_model, n_heads, dropout=dropout, batch_first=True
#         )
#         self.gene_q = nn.Linear(d_model, d_model)
#         self.gene_k = nn.Linear(d_model, d_model)
#         self.gene_v = nn.Linear(d_model, d_model)
#         self.norm = nn.LayerNorm(d_model)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * 4, d_model),
#             nn.Dropout(dropout),
#         )
#         self.norm2 = nn.LayerNorm(d_model)

#     def forward(self, patch_embs, region_embs, gene_context=None, pad_mask=None):
       
#         if gene_context is not None:
#             gene_summary = _masked_mean(gene_context, pad_mask=pad_mask)
#             patch_query = patch_embs + self.gene_q(gene_context)
#             region_key = region_embs + self.gene_k(gene_summary).expand_as(region_embs)
#             region_value = region_embs + self.gene_v(gene_summary).expand_as(region_embs)
#         else:
#             patch_query = patch_embs
#             region_key = region_embs
#             region_value = region_embs

#         attn_out, _ = self.cross_attn(patch_query, region_key, region_value)
#         patch_embs = self.norm(patch_embs + attn_out)
#         patch_embs = self.norm2(patch_embs + self.ffn(patch_embs))
#         return patch_embs


# class PatchToRegionAttention(nn.Module):
   

#     def __init__(self, d_model, n_heads=4, dropout=0.1):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(
#             d_model, n_heads, dropout=dropout, batch_first=True
#         )
#         self.gene_q = nn.Linear(d_model, d_model)
#         self.gene_k = nn.Linear(d_model, d_model)
#         self.gene_v = nn.Linear(d_model, d_model)
#         self.norm = nn.LayerNorm(d_model)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * 4, d_model),
#             nn.Dropout(dropout),
#         )
#         self.norm2 = nn.LayerNorm(d_model)

#     def forward(self, region_embs, patch_embs, pad_mask=None, gene_context=None):
       
#         key_padding_mask = pad_mask if pad_mask is not None else None
#         if gene_context is not None:
#             gene_summary = _masked_mean(gene_context, pad_mask=pad_mask)
#             region_query = region_embs + self.gene_q(gene_summary).expand_as(region_embs)
#             patch_key = patch_embs + self.gene_k(gene_context)
#             patch_value = patch_embs + self.gene_v(gene_context)
#         else:
#             region_query = region_embs
#             patch_key = patch_embs
#             patch_value = patch_embs

#         attn_out, _ = self.cross_attn(
#             region_query, patch_key, patch_value,
#             key_padding_mask=key_padding_mask,
#         )
#         region_embs = self.norm(region_embs + attn_out)
#         region_embs = self.norm2(region_embs + self.ffn(region_embs))
#         return region_embs


# class RegionToSlideAttention(nn.Module):
#     """
#     All regions collectively update the slide token.

#     Query:   slide token
#     Key/Val: region tokens

#     This enables global understanding of the slide's regional composition.
#     """

#     def __init__(self, d_model, n_heads=4, dropout=0.1):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(
#             d_model, n_heads, dropout=dropout, batch_first=True
#         )
#         self.gene_q = nn.Linear(d_model, d_model)
#         self.gene_k = nn.Linear(d_model, d_model)
#         self.gene_v = nn.Linear(d_model, d_model)
#         self.norm = nn.LayerNorm(d_model)
#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model * 4, d_model),
#             nn.Dropout(dropout),
#         )
#         self.norm2 = nn.LayerNorm(d_model)

#     def forward(self, slide_emb, region_embs, gene_context=None, pad_mask=None):
#         """
#         Args:
#             slide_emb:   [B, 1, d_model]
#             region_embs: [B, K, d_model]
#             gene_context: [B, N, d_model] or [B, 1, d_model]
#             pad_mask:    [B, N]  True = padding

#         Returns:
#             slide_emb:   [B, 1, d_model]  — updated with regional composition
#         """
#         if gene_context is not None:
#             gene_summary = _masked_mean(gene_context, pad_mask=pad_mask)
#             slide_query = slide_emb + self.gene_q(gene_summary)
#             region_key = region_embs + self.gene_k(gene_summary).expand_as(region_embs)
#             region_value = region_embs + self.gene_v(gene_summary).expand_as(region_embs)
#         else:
#             slide_query = slide_emb
#             region_key = region_embs
#             region_value = region_embs

#         attn_out, _ = self.cross_attn(slide_query, region_key, region_value)
#         slide_emb = self.norm(slide_emb + attn_out)
#         slide_emb = self.norm2(slide_emb + self.ffn(slide_emb))
#         return slide_emb


# class CrossScaleFusion(nn.Module):
#     """
#     Orchestrates all cross-scale message passing in one HFlowBlock.

#     Direction is controlled by `direction` parameter:
#         "none"          — no cross-scale (patch self-attn only)
#         "top_down"      — Slide → Region → Patch
#         "bottom_up"     — Patch → Region → Slide
#         "bidirectional" — both: Slide↔Region↔Patch in both directions

#     In the default bidirectional mode, the update order is:
#         1. Slide → Region  (global context → regions)
#         2. Region → Patch  (microenvironment → spots)
#         3. Patch → Region  (local evidence → regions)
#         4. Region → Slide  (regional composition → global)
#     """

#     def __init__(self, d_model, n_heads=4, dropout=0.1):
#         super().__init__()

#         # Top-down modules
#         self.slide_to_region = SlideToRegionAttention(d_model, n_heads, dropout)
#         self.region_to_patch = RegionToPatchAttention(d_model, n_heads, dropout)

#         # Bottom-up modules
#         self.patch_to_region = PatchToRegionAttention(d_model, n_heads, dropout)
#         self.region_to_slide = RegionToSlideAttention(d_model, n_heads, dropout)

#     def forward(self, z_patch, z_region, z_slide, direction="bidirectional", pad_mask=None, gene_context=None):
#         """
#         Args:
#             z_patch:   [B, N, d_model]
#             z_region:  [B, K, d_model]
#             z_slide:   [B, 1, d_model]
#             direction: one of "none", "top_down", "bottom_up", "bidirectional"
#             pad_mask:  [B, N]  True = padding
#             gene_context: [B, N, d_model] or [B, 1, d_model]

#         Returns:
#             z_patch, z_region, z_slide — all updated
#         """
#         if direction == "none":
#             return z_patch, z_region, z_slide

#         if direction in ("top_down", "bidirectional"):
#             # 1. Slide → Region: broadcast global context
#             z_region = self.slide_to_region(z_region, z_slide, gene_context=gene_context, pad_mask=pad_mask)
#             # 2. Region → Patch: inject microenvironment
#             z_patch = self.region_to_patch(z_patch, z_region, gene_context=gene_context, pad_mask=pad_mask)

#         if direction in ("bottom_up", "bidirectional"):
#             # 3. Patch → Region: local evidence back to region
#             z_region = self.patch_to_region(z_region, z_patch, pad_mask=pad_mask, gene_context=gene_context)
#             # 4. Region → Slide: regional composition to global
#             z_slide = self.region_to_slide(z_slide, z_region, gene_context=gene_context, pad_mask=pad_mask)

#         return z_patch, z_region, z_slide

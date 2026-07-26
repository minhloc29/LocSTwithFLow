"""
HFlowBlock — The core building block of the Hierarchical Biological Flow Transformer.

[PATCHED] Adds capture of region_assignment / pad_mask for offline diagnostics
(Moran's I subset analysis, region-tissue correspondence ARI/NMI). See the two
blocks marked "# ── DIAGNOSTIC CAPTURE ──" below.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from stflow.model.fa import FrameAveraging


def rearrange(x, pattern, **kwargs):
    """Minimal einops.rearrange replacement for 'n (h d) -> n h d' and variants."""
    if '->' in pattern:
        in_pat, out_pat = pattern.split('->')
        n = x.shape[0]
        if ' (h d)' in in_pat and 'n h d' in out_pat:
            h = kwargs.get('h', 1)
            return x.reshape(n, h, -1)
        if 'n h d' in in_pat and ' (h d)' in out_pat:
            return x.reshape(n, -1)
    return x


def einsum(*operands_and_equation):
    """Minimal einops.einsum: last arg is equation string, rest are tensors."""
    *operands, equation = operands_and_equation
    return torch.einsum(equation, operands)


def gru_update(cell, hidden, candidate):
    """Apply a GRUCell over batched sequence/state tensors."""
    hidden_shape = hidden.shape
    updated = cell(candidate.reshape(-1, candidate.shape[-1]), hidden.reshape(-1, hidden.shape[-1]))
    return updated.reshape(hidden_shape)


class DynamicRegionAssignment(nn.Module):
    """
    Recompute patch-to-region assignments from the current patch state and
    current gene context, then update region/slide memory recurrently.
    """

    def __init__(self, d_model, gene_dim, drop=0.0):
        super().__init__()
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_dim, d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_model, d_model),
        )
        self.region_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_model, d_model),
        )
        self.slide_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_model, d_model),
        )
        
        self.region_gru = nn.GRUCell(d_model, d_model)
        self.slide_gru = nn.GRUCell(d_model, d_model)

    def forward(self, patch_batch, region_state, slide_state, gene_context_batch, pad_mask=None):
        gene_state = self.gene_proj(gene_context_batch)
        if pad_mask is None:
            gene_summary = gene_state.mean(dim=1, keepdim=True)
        else:
            valid_mask = (~pad_mask).unsqueeze(-1).to(dtype=gene_state.dtype)
            gene_summary = (gene_state * valid_mask).sum(dim=1, keepdim=True) / valid_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        patch_query = patch_batch + gene_state
        region_key = region_state + gene_summary.expand_as(region_state)
        scores = torch.matmul(patch_query, region_key.transpose(-1, -2)) / math.sqrt(patch_batch.shape[-1])

        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(-1), -1e9)

        assignment = F.softmax(scores, dim=-1)
        if pad_mask is not None:
            assignment = assignment * (~pad_mask).unsqueeze(-1).to(dtype=assignment.dtype)

        region_mass = assignment.sum(dim=1).clamp_min(1e-6)
        region_candidate = torch.einsum("bnk,bnd->bkd", assignment, patch_batch)
        region_candidate = region_candidate / region_mass.unsqueeze(-1)
        region_candidate = region_candidate + self.region_gate(
            torch.cat([region_candidate, region_state, gene_summary.expand_as(region_candidate)], dim=-1)
        )

        region_state = gru_update(self.region_gru, region_state, region_candidate)
        slide_candidate = region_state.mean(dim=1, keepdim=True)
        slide_candidate = slide_candidate + self.slide_gate(torch.cat([slide_candidate, gene_summary], dim=-1))
        slide_state = gru_update(self.slide_gru, slide_state, slide_candidate)

        return region_state, slide_state, assignment


def to_dense_batch(x, batch, fill_value=0, max_num_nodes=None):
    """
    Vectorized replacement for torch_geometric.utils.to_dense_batch.

    Converts a flat tensor [N_total, D] with batch indices to dense batched form.

    Returns:
        out:        [B, max_num_nodes, D]  — dense tensor (pad positions = fill_value)
        valid_mask: [B, max_num_nodes]     — True = valid (non-padding) position
    """
    N_total, D = x.shape
    B = int(batch.max().item()) + 1

    if max_num_nodes is None:
        counts = torch.zeros(B, device=batch.device, dtype=torch.long).scatter_add_(
            0, batch, torch.ones(N_total, dtype=torch.long, device=batch.device)
        )
        max_num_nodes = int(counts.max().item())

    out = torch.full((B, max_num_nodes, D), fill_value, dtype=x.dtype, device=x.device)
    valid_mask = torch.zeros(B, max_num_nodes, dtype=torch.bool, device=x.device)

    order = torch.argsort(batch, stable=True)
    batch_sorted = batch[order]
    counts = torch.bincount(batch_sorted, minlength=B)
    starts = torch.cumsum(counts, dim=0) - counts
    pos_sorted = torch.arange(N_total, device=batch.device) - starts[batch_sorted]
    pos = torch.empty_like(pos_sorted)
    pos[order] = pos_sorted

    out[batch, pos] = x
    valid_mask[batch, pos] = True

    return out, valid_mask


class SimpleMlp(nn.Module):
    """Simple MLP without timm dependency."""
    def __init__(self, in_features, hidden_features, out_features, drop=0., act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SpatialEdgeAggregation(FrameAveraging):
    """
    k-NN graph spatial attention with frame averaging.
    Drop-in replacement for MLPAttnEdgeAggregation that doesn't need timm.
    """

    def __init__(
            self,
            d_model,
            d_edge_model,
            n_genes,
            n_heads=1,
            proj_drop=0.,
            attn_drop=0.,
            gene_exp_non_negative=True,
        ):
        super().__init__(dim=2)

        self.d_head = d_model // n_heads
        self.d_edge_head = d_edge_model // n_heads
        self.n_heads = n_heads
        self.n_genes = n_genes

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3),
        )

        self.mlp_attn = SimpleMlp(
            in_features=self.d_head*2+self.d_edge_head+self.n_genes,
            hidden_features=d_model,
            out_features=1,
            drop=proj_drop,
        )
        self.edge_trans = SimpleMlp(
            in_features=self.dim+1,
            hidden_features=d_edge_model,
            out_features=d_edge_model,
            drop=proj_drop,
        )
        self.W_output = SimpleMlp(
            in_features=d_model+d_edge_model,
            hidden_features=d_model,
            out_features=d_model,
            drop=proj_drop,
        )

        self.attn_dropout = nn.Dropout(attn_drop)

    def forward(self, gene_exp, token_embs, coords, neighbor_indices, neighbor_masks=None):
        n_tokens, n_neighbors = token_embs.size(0), neighbor_indices.size(1)
        n_heads, d_head, d_edge_head = self.n_heads, self.d_head, self.d_edge_head

        q_s, k_s, v_s = self.layernorm_qkv(token_embs).chunk(3, dim=-1)
        q_s, k_s, v_s = map(lambda x: rearrange(x, 'n (h d) -> n h d', h=n_heads), (q_s, k_s, v_s))

        """build pairwise representation with FA"""
        radial_coords = coords[neighbor_indices] - coords.unsqueeze(dim=1)
        radial_coord_norm = radial_coords.norm(dim=-1).unsqueeze(-1)

        frame_feats, _, _ = self.create_frame(radial_coords, neighbor_masks)
        frame_feats = frame_feats.view(n_tokens, self.n_frames, n_neighbors, -1)

        radial_coord_norm = radial_coord_norm.unsqueeze(dim=1).expand(n_tokens, self.n_frames, n_neighbors, -1)
        frame_feats = self.edge_trans(torch.cat([frame_feats, radial_coord_norm], dim=-1)).mean(dim=1)

        """gene expression features"""
        gene_exp_diff = gene_exp[neighbor_indices] - gene_exp.unsqueeze(dim=1)
        gene_exp_feats_expand = gene_exp_diff[..., None, :].expand(n_tokens, n_neighbors, n_heads, -1)

        """attention map"""
        q_s = q_s.unsqueeze(dim=1).expand(n_tokens, n_neighbors, n_heads, d_head)
        frame_feats = frame_feats.view(n_tokens, n_neighbors, n_heads, d_edge_head)
        message = torch.cat([q_s, k_s[neighbor_indices], frame_feats, gene_exp_feats_expand], dim=-1)

        attn_map = self.mlp_attn(message).squeeze(-1)
        if neighbor_masks is not None:
            attn_map.masked_fill_(neighbor_masks.unsqueeze(dim=-1), -1e9)
        attn_map = self.attn_dropout(nn.Softmax(dim=-1)(attn_map.transpose(1, 2)))

        """context aggregation"""
        v_s_neighs = v_s[neighbor_indices].view(n_tokens, -1, n_heads, d_head)
        scalar_context = einsum(attn_map, v_s_neighs, 'n h m, n m h d -> n h d').view(n_tokens, -1)
        edge_context = einsum(attn_map, frame_feats, 'n h m, n m h d -> n h d').view(n_tokens, -1)
        return self.W_output(torch.cat([scalar_context, edge_context], dim=-1))


class HFlowBlock(nn.Module):
    """
    A single hierarchical flow block.

    Architecture per block:
        Input: x_t, z_patch, z_region, z_slide, coords, neighbor_indices, batch_idx, pad_mask
          │
          ├─ 1. Spatial Patch Self-Attn (k-NN graph, MLP attention)
          ├─ 2. Cross-scale message passing (direction-dependent)
          ├─ 3. Gene velocity prediction from patch tokens
          │
          └─ Output: velocity_flat, updated z_patch, z_region, z_slide
    """

    def __init__(
        self,
        d_model,
        d_edge_model,
        n_genes,
        n_heads=4,
        activation="gelu",
        attn_drop=0.0,
        proj_drop=0.0,
        hflow_cross_scale="bidirectional",
    ):
        super().__init__()

        # ── 1. Spatial patch self-attention (k-NN graph, FA-aware) ──
        self.spatial_attn = SpatialEdgeAggregation(
            d_model=d_model,
            d_edge_model=d_edge_model,
            n_genes=n_genes,
            n_heads=n_heads,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
        )
        self.spatial_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(proj_drop),
        )
        self.spatial_norm = nn.LayerNorm(d_model)
        self.spatial_norm2 = nn.LayerNorm(d_model)

        self.flow_context_proj = nn.Sequential(
            nn.Linear(n_genes, d_model),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(d_model, d_model),
        )

        self.flow_context_gate = nn.Parameter(torch.tensor(-2.1972246))
        self.velocity_gate = nn.Parameter(torch.tensor(-2.1972246))
        self.region_norm = nn.LayerNorm(d_model)
        self.slide_norm = nn.LayerNorm(d_model)
        self.region_gru = nn.GRUCell(d_model, d_model)
        self.slide_gru = nn.GRUCell(d_model, d_model)
        self.dynamic_assignment = DynamicRegionAssignment(d_model, d_model, drop=proj_drop)

        # ── 2. Gene velocity head (last block's prediction is used) ──
        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(proj_drop),
            nn.Linear(d_model, n_genes),
            nn.Dropout(proj_drop),
        )

        self.cross_scale_direction = hflow_cross_scale

        # ── DIAGNOSTIC CAPTURE (init) ──
        # Populated during eval-mode forward passes; read externally by
        # test_with_diagnostics.py via `model.blocks[-1]._last_region_assignment`
        # (or `model._last_region_assignment`, set on the denoiser after the loop).
        self._last_region_assignment = None
        self._last_pad_mask = None

    def forward(
        self,
        noisy_exp_flat,
        z_patch_flat,
        z_region,
        z_slide,
        coords_flat,
        neighbor_indices,
        batch_idx,
        pad_mask=None,
        max_n_cells=None,
    ):
        """
        Args:
            noisy_exp_flat:   [N_total, n_genes]
            z_patch_flat:     [N_total, d_model]
            z_region:         [B, K, d_model]
            z_slide:          [B, 1, d_model]
            coords_flat:      [N_total, 2]
            neighbor_indices: [N_total, n_neighbors]
            batch_idx:        [N_total]
            pad_mask:         [B, N_max]  True = padding
            max_n_cells:      int

        Returns:
            velocity_flat:  [N_total, n_genes]
            z_patch_flat:   [N_total, d_model]
            z_region:       [B, K, d_model]
            z_slide:        [B, 1, d_model]
        """
        B = z_region.shape[0]
        N_total = z_patch_flat.shape[0]
        direction = self.cross_scale_direction
        initial_patch = z_patch_flat
        gene_context = self.flow_context_proj(noisy_exp_flat)

        # ── Step 1: Spatial patch self-attention (k-NN graph) ──
        attn_out = self.spatial_attn(
            noisy_exp_flat, z_patch_flat, coords_flat, neighbor_indices
        )
        z_patch_flat = self.spatial_norm(z_patch_flat + attn_out)
        z_patch_flat = self.spatial_norm2(z_patch_flat + self.spatial_mlp(z_patch_flat))

        # Keep the current noisy expression visible to the hierarchy.
        flow_alpha = torch.sigmoid(self.flow_context_gate)
        z_patch_flat = z_patch_flat + flow_alpha * gene_context

        # Recompute the hierarchy from the current post-attention patch state.
        if pad_mask is None:
            z_patch_batch, valid_mask = to_dense_batch(
                z_patch_flat,
                batch=batch_idx,
                fill_value=0.0,
                max_num_nodes=max_n_cells,
            )
            gene_context_batch, _ = to_dense_batch(
                gene_context,
                batch=batch_idx,
                fill_value=0.0,
                max_num_nodes=max_n_cells,
            )
            pad_mask_for_attn = ~valid_mask
        else:
            pad_mask_for_attn = pad_mask
            if max_n_cells is not None:
                z_patch_batch = z_patch_flat.new_zeros(B, max_n_cells, z_patch_flat.shape[-1])
                z_patch_batch[~pad_mask] = z_patch_flat
                gene_context_batch = gene_context.new_zeros(B, max_n_cells, gene_context.shape[-1])
                gene_context_batch[~pad_mask] = gene_context
            else:
                z_patch_batch = z_patch_flat.reshape(B, -1, z_patch_flat.shape[-1])
                gene_context_batch = gene_context.reshape(B, -1, gene_context.shape[-1])

        z_region, z_slide, region_assignment = self.dynamic_assignment(
            z_patch_batch,
            z_region,
            z_slide,
            gene_context_batch,
            pad_mask=pad_mask_for_attn,
        )

        # ── DIAGNOSTIC CAPTURE (forward) ──
        # Only stash during eval to avoid retaining graph / memory overhead during training.
        if not self.training:
            self._last_region_assignment = region_assignment.detach()
            self._last_pad_mask = pad_mask_for_attn.detach() if pad_mask_for_attn is not None else None

        # ── Cheap patch refinement from current region memory ──
        if direction != "none":
            patch_region = torch.einsum("bnk,bkd->bnd", region_assignment, z_region)
            z_patch_batch = z_patch_batch + patch_region

            # Convert back to flat
            if pad_mask is not None:
                z_patch_flat = z_patch_batch[~pad_mask]
            else:
                z_patch_flat = z_patch_batch[valid_mask]

        # Preserve the original patch semantics before the readout head.
        z_patch_flat = z_patch_flat + torch.sigmoid(self.velocity_gate) * initial_patch

        # ── Velocity prediction from patch tokens ──
        velocity_flat = self.velocity_head(z_patch_flat)

        return velocity_flat, z_patch_flat, z_region, z_slide
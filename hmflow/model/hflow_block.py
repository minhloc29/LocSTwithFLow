import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hmflow.model.fa import FrameAveraging
from hmflow.model.hierarchy_time_gate import HierarchyTimeGate
from hmflow.model.hierarchy_adaln import HierarchyAdaLN


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
    d_edge: dimension of edge among nodes
    
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
        """
        k-NN spatial attention with frame-averaged edge geometry.

        For every token we compute its message by attending over its k spatial
        neighbours. Each edge carries: (1) the query token's state (q), (2) the
        neighbour's key/value states (k/v), (3) frame-averaged relative geometry
        (rotation + distance), and (4) the gene-expression difference. An MLP
        scores the concatenation to get neighbour attention weights, which then
        aggregate both value states and edge geometry in parallel.

        Args:
            gene_exp:        [n_tokens, n_genes]      current (noisy) gene expression
            token_embs:      [n_tokens, d_model]      patch/token hidden states (query/key/value source)
            coords:          [n_tokens, 2]            spatial (x, y) positions of tokens
            neighbor_indices:[n_tokens, n_neighbors]  per-token indices of nearest neighbours
            neighbor_masks:  [n_tokens, n_neighbors]  True where a neighbour slot is padding

        Returns:
            [n_tokens, d_model] aggregated message per token.
        """
        n_tokens, n_neighbors = token_embs.size(0), neighbor_indices.size(1)
        n_heads, d_head, d_edge_head = self.n_heads, self.d_head, self.d_edge_head

        # ---- 1. Q/K/V projection -------------------------------------------
        # layernorm_qkv: LayerNorm + Linear(d_model -> 3*d_model), then chunk
        # q_s, k_s, v_s: [n_tokens, d_model] each, then per head-reshape:
        # reshaped q_s/k_s/v_s: [n_tokens, n_heads, d_head]
        # .chunk(3, dim = -1) split the tensor into 3 chunk, input : [B, N, 3d] -> 3 x [B, N, D]
        q_s, k_s, v_s = self.layernorm_qkv(token_embs).chunk(3, dim=-1) 
        q_s, k_s, v_s = map(lambda x: rearrange(x, 'n (h d) -> n h d', h=n_heads), (q_s, k_s, v_s))

        # ---- 2. Pairwise geometric (edge) features with Frame Averaging -----
        # radial_coords: [n_tokens, n_neighbors, 2]  relative vector from token to each neighbour
        radial_coords = coords[neighbor_indices] - coords.unsqueeze(dim=1)
        # radial_coord_norm: [n_tokens, n_neighbors, 1]  euclidean distance to each neighbour
        radial_coord_norm = radial_coords.norm(dim=-1).unsqueeze(-1)

        # create_frame returns rotations of radial_coords through n_frames frames:
        # frame_feats: [n_tokens, n_frames(=4), n_neighbors, 2]
        frame_feats, _, _ = self.create_frame(radial_coords, neighbor_masks)
        frame_feats = frame_feats.view(n_tokens, self.n_frames, n_neighbors, -1)

        # broadcast distance across frames and concatenate with rotated vectors:
        # [n_tokens, n_frames, n_neighbors, 2+1] -> edge_trans MLP -> d_edge_model,
        # then average over frames: [n_tokens, n_neighbors, d_edge_model]
        radial_coord_norm = radial_coord_norm.unsqueeze(dim=1).expand(n_tokens, self.n_frames, n_neighbors, -1)
        frame_feats = self.edge_trans(torch.cat([frame_feats, radial_coord_norm], dim=-1)).mean(dim=1)

        # ---- 3. Gene-expression (edge) features ------------------------------
        # gene_exp_diff: [n_tokens, n_neighbors, n_genes] expression diff token vs neighbour
        gene_exp_diff = gene_exp[neighbor_indices] - gene_exp.unsqueeze(dim=1)
        # gene_exp_feats_expand: [n_tokens, n_neighbors, n_heads, n_genes] (copied per head)
        gene_exp_feats_expand = gene_exp_diff[..., None, :].expand(n_tokens, n_neighbors, n_heads, -1)

        # ---- 4. Attention map (MLP-scored edges) ----------------------------
        # q_s broadcast over neighbours: [n_tokens, n_neighbors, n_heads, d_head]
        q_s = q_s.unsqueeze(dim=1).expand(n_tokens, n_neighbors, n_heads, d_head)
        # frame_feats as per head: [n_tokens, n_neighbors, n_heads, d_edge_head]
        frame_feats = frame_feats.view(n_tokens, n_neighbors, n_heads, d_edge_head)
        # message per edge per head: [n_tokens, n_neighbors, n_heads,
        #                             d_head( q) + d_head(k) + d_edge_head + n_genes]
        message = torch.cat([q_s, k_s[neighbor_indices], frame_feats, gene_exp_feats_expand], dim=-1)

        # mlp_attn scores each edge -> [n_tokens, n_neighbors, n_heads, 1] -> squeeze
        # attn_map (pre-softmax): [n_tokens, n_neighbors, n_heads]
        attn_map = self.mlp_attn(message).squeeze(-1)
        if neighbor_masks is not None:
            # blank out padding neighbours (neighbor_masks unsq -> [n, k, 1])
            attn_map.masked_fill_(neighbor_masks.unsqueeze(dim=-1), -1e9)
        # transpose to [n_tokens, n_heads, n_neighbors], softmax over neighbours, dropout
        attn_map = self.attn_dropout(nn.Softmax(dim=-1)(attn_map.transpose(1, 2)))
        # attn_map now: [n_tokens, n_heads, n_neighbors] (weights sum to 1 over neighbours)

        # ---- 5. Context aggregation (content + geometry) --------------------
        # v_s gathered to neighbours: [n_tokens, n_neighbors, n_heads, d_head]
        v_s_neighs = v_s[neighbor_indices].view(n_tokens, -1, n_heads, d_head)
        # scalar_context: attn-weighted sum of values -> [n_tokens, n_heads, d_head] -> [n_tokens, d_model]
        scalar_context = einsum(attn_map, v_s_neighs, 'n h m, n m h d -> n h d').view(n_tokens, -1)
        # edge_context:   attn-weighted sum of frame features -> [n_tokens, d_edge_model]
        edge_context = einsum(attn_map, frame_feats, 'n h m, n m h d -> n h d').view(n_tokens, -1)
        # concatenate content and geometry, project back to d_model
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
        use_time_hierarchy_gate=True,
        modulation_mode="adaln",
        gate_mode="learnable",
        gate_hidden=128,
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
        self.slide_proj = nn.Linear(d_model, d_model)              # ← add this
        self.slide_gate_inject = nn.Parameter(torch.tensor(-2.1972246))
        self.dynamic_assignment = DynamicRegionAssignment(d_model, d_model, drop=proj_drop)

        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(proj_drop),
            nn.Linear(d_model, n_genes),
            nn.Dropout(proj_drop),
        )

        self.cross_scale_direction = hflow_cross_scale

        # ── Time-dependent hierarchical modulation (scalar gate / AdaLN) ──
        self.use_time_hierarchy_gate = use_time_hierarchy_gate
        self.gate_mode = gate_mode  # legacy scalar-gate mode ('static'/'fixed'/'learnable')
        self.modulation_mode = modulation_mode if use_time_hierarchy_gate else "static"
        if self.modulation_mode == "adaln":
            self.hierarchy_adaln = HierarchyAdaLN(d_model, time_dim=d_model)
            self.hierarchy_gate = None
        elif self.modulation_mode == "scalar":
            self.hierarchy_adaln = None
            self.hierarchy_gate = HierarchyTimeGate(gate_hidden)
        else:  # static
            self.hierarchy_adaln = None
            self.hierarchy_gate = None

        # Per-level LayerNorms used by AdaLN (modulate before cross-scale fusion).
        self.patch_adaln_norm = nn.LayerNorm(d_model)
        self.region_adaln_norm = nn.LayerNorm(d_model)
        self.slide_adaln_norm = nn.LayerNorm(d_model)

        self._last_region_assignment = None
        self._last_pad_mask = None

    def _gate_weights(self, t):
        """Return (w_patch, w_region, w_slide) each [B] for the given timestep t.

        Only used by the 'static'/'fixed' scalar paths; the AdaLN path computes
        per-feature scale/shift/gate instead and does not call this.
        """
        B = t.shape[0] if t.ndim else 1
        dev, dt = t.device, t.dtype
        if self.modulation_mode == "static":
            ones = torch.ones(B, device=dev, dtype=dt)
            return ones, ones, ones
        if self.gate_mode == "fixed":
            tf = t.float()
            slide = tf
            patch = 1.0 - tf
            region = 4.0 * tf * (1.0 - tf)
            s = slide + patch + region
            return patch / s, region / s, slide / s
        # scalar learnable gate
        return self.hierarchy_gate(t)

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
        t=None,
        t_emb=None,
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
            t:                [B]  flow timestep (scalar gate / fixed schedule)
            t_emb:            [B, d_model]  fourier time embedding (AdaLN)

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

        attn_out = self.spatial_attn(
            noisy_exp_flat, z_patch_flat, coords_flat, neighbor_indices
        )
        z_patch_flat = self.spatial_norm(z_patch_flat + attn_out)
        z_patch_flat = self.spatial_norm2(z_patch_flat + self.spatial_mlp(z_patch_flat))

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

        if not self.training:
            self._last_region_assignment = region_assignment.detach()
            self._last_pad_mask = pad_mask_for_attn.detach() if pad_mask_for_attn is not None else None

        # ── Time-dependent hierarchical modulation ──
        # AdaLN (proposed): per-level scale/shift/gate conditioned on t_emb.
        #   Region/Slide are LN+scale+shift modulated before use as context.
        #   Patch is the residual stream: LN+scale+shift, then gates scale the
        #   two cross-scale additions (slide->patch, region->patch).
        # AdaLN-Zero init means g=0, 1+scale=1, shift=0 -> identity at start.
        if self.modulation_mode == "adaln" and t_emb is not None:
            p = self.hierarchy_adaln(t_emb)  # {lvl: (scale[B,d], shift[B,d], gate[B,1])}
            (s_p, sh_p, g_p), (s_r, sh_r, g_r), (s_s, sh_s, g_s) = \
                p["patch"], p["region"], p["slide"]

            # Modulate region/slide context before cross-scale use.
            z_region_mod = self.region_adaln_norm(z_region) * (1 + s_r[:, None, :]) + sh_r[:, None, :]
            z_slide_mod = self.slide_adaln_norm(z_slide) * (1 + s_s[:, None, :]) + sh_s[:, None, :]

            # slide -> patch (AdaLN gate on the sigmoid-controlled injection)
            slide_signal = self.slide_proj(z_slide_mod)
            z_patch_batch = z_patch_batch + g_s[:, :, None] * torch.sigmoid(self.slide_gate_inject) * slide_signal

            # Patch residual stream: LN + scale/shift before accumulating context.
            z_patch_batch = self.patch_adaln_norm(z_patch_batch) * (1 + s_p[:, None, :]) + sh_p[:, None, :]

            if direction != "none":
                patch_region = torch.einsum("bnk,bkd->bnd", region_assignment, z_region_mod)
                z_patch_batch = z_patch_batch + g_r[:, :, None] * patch_region

                if pad_mask is not None:
                    z_patch_flat = z_patch_batch[~pad_mask]
                else:
                    z_patch_flat = z_patch_batch[valid_mask]
        else:
            # Scalar gate (previous behavior): one softmax weight per level.
            slide_signal = self.slide_proj(z_slide)
            w_patch_b, w_region_b, w_slide_b = self._gate_weights(t)
            w_slide_ = w_slide_b[:, None, None]
            w_region_ = w_region_b[:, None, None]

            if not self.training:
                self._last_gate_weights = (
                    w_patch_b.detach(), w_region_b.detach(), w_slide_b.detach())
                self._last_gate_t = t.detach()

            z_patch_batch = z_patch_batch + w_slide_ * torch.sigmoid(self.slide_gate_inject) * slide_signal

            if direction != "none":
                patch_region = torch.einsum("bnk,bkd->bnd", region_assignment, z_region)
                z_patch_batch = z_patch_batch + w_region_ * patch_region

                if pad_mask is not None:
                    z_patch_flat = z_patch_batch[~pad_mask]
                else:
                    z_patch_flat = z_patch_batch[valid_mask]

        z_patch_flat = z_patch_flat + torch.sigmoid(self.velocity_gate) * initial_patch

        velocity_flat = self.velocity_head(z_patch_flat)

        return velocity_flat, z_patch_flat, z_region, z_slide
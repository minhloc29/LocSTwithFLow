"""
HFlowDenoiser — Hierarchical Biological Flow Denoiser for STFlow.

Central novelty: hierarchy state (z_slide, z_region, z_patch) persists across
flow timesteps, not just across internal transformer layers.  This means the
biological representation of the slide evolves jointly with the generative
process, exactly as described in the HFlow-ST proposal.

[PATCHED] After the HFlowBlock loop, exposes `_last_region_assignment` and
`_all_block_region_assignments` on the denoiser (only during eval), read
externally by test_with_diagnostics.py.
"""

import math
import torch
import torch.nn as nn

from hmflow.model.hflow_config import HFlowConfig
from hmflow.model.hierarchy import HierarchyEncoder
from hmflow.model.hflow_block import HFlowBlock


class TimestepEmbedder(nn.Module):
    # We want the neural network to know "what time it is" in the diffusion process.
    # Time = 73 not informative -> [0.1, 0.15,...] better
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        # freqs[none] create new dims at dim = 0
        args = t[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class HFlowDenoiser(nn.Module):
    # my work: input: image features, noisy exp x_t, coord (x, y), timestep t -> output: enpoint x1
    # how to predict genes: output is patch updated -> input to predict genes
    """
    Hierarchical Flow Denoiser.

    The conditioning hierarchy (slide, region, patch tokens) is computed
    once from the image features and then evolved through the flow trajectory
    by the HFlowBlocks.  Each call to inference() both accepts and returns
    the current hierarchy state, allowing the sampling loop to thread it
    across Euler steps.

    Ablation modes:
        flat                — original STFlow (no hierarchy)
        slide_patch         — slide + patch tokens, no region level
        slide_region_patch  — full hierarchy (default)
        dynamic_update      — hierarchy resets between layers vs. persists
    """

    def __init__(self, model_config, hflow_config=None):
        super().__init__()

        if hflow_config is None:
            hflow_config = HFlowConfig()

        self.mcfg = model_config
        self.hcfg = hflow_config
        self.d_model = model_config.d_model
        self.representation = hflow_config.hflow_representation
        self.dynamic_update = hflow_config.hflow_dynamic_update

        self.fourier_proj = TimestepEmbedder(self.d_model)

        self.image_transform = nn.Linear(model_config.feature_dim, self.d_model)

        if self.representation in ("slide_patch", "slide_region_patch"):
            self.hierarchy_encoder = HierarchyEncoder(self.d_model, hflow_config)

        if self.representation == "flat":
            from hmflow.model.transformer import SpatialTransformer
            self.backbone = SpatialTransformer(model_config)
        else:
            self.blocks = nn.ModuleList([
                HFlowBlock(
                    d_model=self.d_model,
                    d_edge_model=model_config.pairwise_hidden_dim,
                    n_genes=model_config.n_genes,
                    n_heads=model_config.n_heads,
                    activation=model_config.activation,
                    attn_drop=model_config.attn_dropout,
                    proj_drop=model_config.dropout,
                    hflow_cross_scale=hflow_config.hflow_cross_scale,
                )
                for _ in range(model_config.n_layers)
            ])

        self.loss_func = nn.MSELoss()

        self._last_region_assignment = None
        self._all_block_region_assignments = None


    def _build_graph(self, coords_flat, batch_idx, n_neighbors, exclude_self=True):
        N = coords_flat.shape[0]
        device = coords_flat.device
        exclude_self_mask = torch.eye(N, dtype=torch.bool, device=device)
        batch_mask = batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)
        rel_pos = coords_flat.unsqueeze(1) - coords_flat.unsqueeze(0)
        rel_dist = rel_pos.norm(dim=-1).detach()
        if exclude_self:
            rel_dist.masked_fill_(exclude_self_mask | ~batch_mask, 1e9)
        else:
            rel_dist.masked_fill_(~batch_mask, 1e9)
        effective_n = min(n_neighbors, N)
        _, nearest_indices = rel_dist.topk(effective_n, dim=-1, largest=False)
        return nearest_indices

    def _build_hierarchy(self, img_features, coords, pad_mask):
        # output [patch, region, slide] cung luc
        if self.representation in ("slide_patch", "slide_region_patch"):
            patch_embs = self.image_transform(img_features)          # [B, N, d_model]
            z_patch, z_region, z_slide = self.hierarchy_encoder(
                patch_embs,
                coords,
                pad_mask=pad_mask,
            )
            # slide_patch mode: no real regions; use slide as a single region
            if self.representation == "slide_patch":
                z_region = z_slide.expand(-1, 1, -1)
            return z_patch, z_region, z_slide
        return None, None, None

    def _flatten_patches(self, z_patch_batch, pad_mask):
        return z_patch_batch[~pad_mask]

    def _prepare_flat_inputs(self, noisy_exp, img_features, coords):
        """Common flat-to-batched bookkeeping shared by all modes."""
        B, N_cells, _ = noisy_exp.shape
        device = noisy_exp.device
        pad_mask = img_features.sum(dim=-1) == 0
        batch_idx = torch.arange(B, device=device).unsqueeze(-1).repeat(1, N_cells)
        batch_idx = batch_idx[~pad_mask]
        noisy_exp_flat = noisy_exp[~pad_mask]
        coords_flat = coords[~pad_mask]
        return pad_mask, batch_idx, noisy_exp_flat, coords_flat

    # ── flat mode (original STFlow) ────────────────────────────────────

    def _inference_flat(self, noisy_exp, img_features, coords, t_steps):
        B, N_cells, _ = noisy_exp.shape
        t_emb = self.fourier_proj(t_steps)
        img_proj = self.image_transform(img_features)
        t_emb_expanded = t_emb[:, None, :].expand(B, N_cells, -1)
        features = img_proj + t_emb_expanded
        return self.backbone(gene_exp=noisy_exp, features=features, coords=coords)

    # ── hierarchical mode ─────────────────────────────────────────────

    def _inference_hierarchical(
        self,
        noisy_exp,
        img_features,
        coords,
        t_steps,
        hierarchy_state=None,
    ):
        """
        Returns (prediction, new_hierarchy_state).

        hierarchy_state = (z_patch_flat, z_region, z_slide) to resume,
        or None to encode from scratch.
        """
        B, N_cells, _ = noisy_exp.shape
        device = noisy_exp.device
        self._last_assignment_entropy = None

        pad_mask, batch_idx, noisy_exp_flat, coords_flat = self._prepare_flat_inputs(
            noisy_exp, img_features, coords
        )
        t_emb = self.fourier_proj(t_steps)  # [B, d_model]

        # ── Step A: Build or resume hierarchy ──────────────────────────
        if hierarchy_state is None:
            # First call at this flow step: encode from image
            z_patch, z_region, z_slide = self._build_hierarchy(
                img_features,
                coords,
                pad_mask,
            )
            # Add time embedding — only once (it becomes part of the state)
            t_emb_batch = t_emb[:, None, :].expand(B, N_cells, -1)
            z_patch = z_patch + t_emb_batch
        else:
            z_patch, z_region, z_slide = hierarchy_state
            # The hierarchy carries its own time context from previous steps

        z_patch_flat = self._flatten_patches(z_patch, pad_mask)

        nearest_indices = self._build_graph(
            coords_flat, batch_idx,
            min(self.mcfg.n_neighbors, N_cells),
            exclude_self=True,
        )

        curr_z_patch = z_patch_flat
        curr_z_region = z_region
        curr_z_slide = z_slide
        block_velocities = []

        for block in self.blocks:
            if self.dynamic_update:
                pass
            else:
                # Static: reset to the initial hierarchy before each block
                curr_z_patch = z_patch_flat
                curr_z_region = z_region
                curr_z_slide = z_slide

            velocity_flat, curr_z_patch, curr_z_region, curr_z_slide = block(
                noisy_exp_flat=noisy_exp_flat,
                z_patch_flat=curr_z_patch,
                z_region=curr_z_region,
                z_slide=curr_z_slide,
                coords_flat=coords_flat,
                neighbor_indices=nearest_indices,
                batch_idx=batch_idx,
                pad_mask=pad_mask,
                max_n_cells=N_cells,
            )
            block_velocities.append(velocity_flat)

        prediction_flat = torch.stack(block_velocities).mean(0)
      
        prediction = prediction_flat.new_zeros(B, N_cells, prediction_flat.shape[-1])
        prediction[~pad_mask] = prediction_flat

    
        z_patch_updated = z_patch.new_zeros(B, N_cells, self.d_model)
        z_patch_updated[~pad_mask] = curr_z_patch
        new_hierarchy_state = (z_patch_updated, curr_z_region, curr_z_slide)

        if not self.training:
            self._last_region_assignment = self.blocks[-1]._last_region_assignment
            self._all_block_region_assignments = [
                b._last_region_assignment for b in self.blocks
            ]

        return prediction, new_hierarchy_state


    def inference(self, noisy_exp, img_features, coords, t_steps,
                  hierarchy_state=None):
        """
        Single-step flow inference.

        Args:
            noisy_exp:      [B, N, n_genes]   noisy expression at time t
            img_features:   [B, N, feature_dim]
            coords:         [B, N, 2]
            t_steps:        [B]               current flow time
            hierarchy_state: tuple | None      previous step's (z_p, z_r, z_s)

        Returns:
            prediction:     [B, N, n_genes]
            hierarchy_state: tuple | None      updated (z_p, z_r, z_s)
        """
        if self.representation == "flat":
            return self._inference_flat(noisy_exp, img_features, coords, t_steps), None

        return self._inference_hierarchical(
            noisy_exp, img_features, coords, t_steps,
            hierarchy_state=hierarchy_state,
        )

    def forward(self, exp, img_features, coords, labels, t_steps):
      
        prediction, _ = self.inference(exp, img_features, coords, t_steps,
                                       hierarchy_state=None)
        pad_mask = img_features.sum(-1) == 0
        loss = self.loss_func(prediction[~pad_mask], labels[~pad_mask])
        return prediction, loss
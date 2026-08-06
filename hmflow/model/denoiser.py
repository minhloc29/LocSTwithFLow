import math
import torch
import torch.nn as nn

from hmflow.model.hflow_config import HFlowConfig
from hmflow.model.hierarchy import HierarchyEncoder
from hmflow.model.hflow_block import HFlowBlock, ProgramCommunityHead


def pcc_loss(pred, target, eps=1e-8):
    """Negative Pearson correlation over the gene axis, averaged over spots.

    Equivalent to ``1 - corr(pred, target)`` up to the minus sign; minimizing it
    drives the predicted gene-expression vector to be linearly correlated with the
    ground truth (matches the ``pearson_mean`` validation metric).

    Uses ``linalg.vector_norm`` (not the deprecated ``Tensor.norm(dim)``, which
    returns 0 for leading-dimension-1 inputs on torch >= 2.0).
    """
    pred = pred - pred.mean(-1, keepdim=True)
    target = target - target.mean(-1, keepdim=True)
    pn = pred / (torch.linalg.vector_norm(pred, dim=-1, keepdim=True) + eps)
    tn = target / (torch.linalg.vector_norm(target, dim=-1, keepdim=True) + eps)
    return -(pn * tn).sum(-1).mean()


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

    def __init__(self, model_config, hflow_config=None):
        super().__init__()

        if hflow_config is None:
            hflow_config = HFlowConfig()

        self.mcfg = model_config
        self.hcfg = hflow_config
        self.d_model = model_config.d_model
        self.representation = hflow_config.hflow_representation

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
                    use_time_hierarchy_gate=hflow_config.use_time_hierarchy_gate,
                    modulation_mode=hflow_config.modulation_mode,
                    gate_mode=hflow_config.gate_mode,
                    gate_hidden=hflow_config.gate_hidden,
                    use_program_latent=hflow_config.use_program_latent,
                    n_programs=hflow_config.n_programs,
                    n_communities=hflow_config.n_communities,
                    lambda_program=hflow_config.lambda_program,
                    lambda_community=hflow_config.lambda_community,
                    lambda_niche=hflow_config.lambda_niche,
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

    def _inference_flat(self, noisy_exp, img_features, coords, t_steps):
        B, N_cells, _ = noisy_exp.shape
        t_emb = self.fourier_proj(t_steps)
        img_proj = self.image_transform(img_features)
        t_emb_expanded = t_emb[:, None, :].expand(B, N_cells, -1)
        features = img_proj + t_emb_expanded
        return self.backbone(gene_exp=noisy_exp, features=features, coords=coords)


    def _inference_hierarchical(
        self,
        noisy_exp,
        img_features,
        coords,
        t_steps
    ):
       
        B, N_cells, _ = noisy_exp.shape
        self._last_assignment_entropy = None

        pad_mask, batch_idx, noisy_exp_flat, coords_flat = self._prepare_flat_inputs(
            noisy_exp, img_features, coords
        )
        t_emb = self.fourier_proj(t_steps)  # [B, d_model]

        z_patch, z_region, z_slide = self._build_hierarchy(
            img_features,
            coords,
            pad_mask,
        )

        t_emb_batch = t_emb[:, None, :].expand(B, N_cells, -1)
        z_patch = z_patch + t_emb_batch

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
        latent_returns = []  # (program, community, gene_recon) per block, if enabled

        for block in self.blocks:

            curr_z_patch = z_patch_flat
            curr_z_region = z_region
            curr_z_slide = z_slide

            out = block(
                noisy_exp_flat=noisy_exp_flat,
                z_patch_flat=curr_z_patch,
                z_region=curr_z_region,
                z_slide=curr_z_slide,
                coords_flat=coords_flat,
                neighbor_indices=nearest_indices,
                batch_idx=batch_idx,
                pad_mask=pad_mask,
                max_n_cells=N_cells,
                t=t_steps,
                t_emb=t_emb,
            )
            (velocity_flat, curr_z_patch, curr_z_region, curr_z_slide,
             program_scores, community_probs, gene_recon) = out
            block_velocities.append(velocity_flat)
            if getattr(block, "use_program_latent", False):
                latent_returns.append(
                    (program_scores, community_probs, gene_recon, nearest_indices))

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

        self._last_latent_returns = latent_returns

        return prediction, new_hierarchy_state


    def inference(self, noisy_exp, img_features, coords, t_steps):
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

        return self._inference_hierarchical(noisy_exp, img_features, coords, t_steps)

    def _multi_level_loss(self, labels_flat, latent_returns):
        """Sum the auxiliary program/community/niche objectives across blocks.

        ``latent_returns`` is a list of (program_scores[N,K], community_probs[N,C],
        gene_recon[N,G], nearest_indices[N,k]) — one entry per latent-enabled block.

        Losses (each averaged over blocks):
          L_program   = MSE(low-rank gene recon, true expression) — makes programs
                        carry real gene meaning.
          L_community = neighbor smoothness of community logits (softmax) — adjacent
                        spots in the same niche tend to share community membership.
          L_niche     = neighbor smoothness of program activations — adjacent spots
                        in the same niche tend to have similar program profiles.
        """
        if not latent_returns:
            return 0.0
        w_p = self.hcfg.lambda_program
        w_c = self.hcfg.lambda_community
        w_n = self.hcfg.lambda_niche
        device = labels_flat.device
        l_p = torch.tensor(0.0, device=device)
        l_c = torch.tensor(0.0, device=device)
        l_n = torch.tensor(0.0, device=device)
        for program_scores, community_probs, gene_recon, neighbor_indices in latent_returns:
            l_p = l_p + nn.functional.mse_loss(gene_recon, labels_flat)
            l_c = l_c + ProgramCommunityHead.neighbor_smoothness(community_probs, neighbor_indices)
            l_n = l_n + ProgramCommunityHead.neighbor_smoothness(program_scores, neighbor_indices)
        n = len(latent_returns)
        return w_p * l_p / n + w_c * l_c / n + w_n * l_n / n

    def forward(self, exp, img_features, coords, labels, t_steps):

        prediction, _ = self.inference(exp, img_features, coords, t_steps)
        pad_mask = img_features.sum(-1) == 0
        loss = self.loss_func(prediction[~pad_mask], labels[~pad_mask])
        if self.hcfg.use_program_latent and getattr(self, "_last_latent_returns", None):
            loss = loss + self._multi_level_loss(
                labels[~pad_mask], self._last_latent_returns)
        return prediction, loss
"""Latent flow matching over the biological hierarchy (Program → Community → Niche → Genes).

This is the spatial-transcriptomics analogue of VAE-latent diffusion: instead of flowing
gene vectors ``[N,G]`` through the ODE, a ``LatentFlowDenoiser`` transports PROGRAM states
``[N,K]``. The biological hierarchy is part of the evolved generative state — reconstructed
(Program → Community → Niche → Genes) at the current program state each step — rather than a
post-hoc decoder applied once at the end.

Components (reused from the gene-space baseline where possible):
  * ``encoder E``        : genes[G] -> program[K]   (learned AE bottleneck)
  * flow backbone        : ``HierarchyEncoder`` + ``HFlowBlock`` with ``n_genes=n_programs``,
                           so the whole block operates natively in program space.
  * ``decoder D``        : ``LatentConstructiveDecoder`` — L2C community -> niche -> genes
                           cascade seeded by the program code (construct-and-refine).
  * losses               : flow-in-program-space + gene recon + AE cycle + community/niche
                           neighbor smoothness.

The flow ODE invariant is preserved: the network's predicted endpoint lives in the same
``[N,K]`` space as the (program-space) sampled state ``p_t``, so ``Interpolant.denoise``
mixes them correctly (see hmflow/app/flow/test.py sampling loop).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from hmflow.model.hflow_config import HFlowConfig
from hmflow.model.hierarchy import HierarchyEncoder
from hmflow.model.hflow_block import HFlowBlock
from hmflow.model.denoiser import ConstructiveStage, ConstructiveDecoder


class LatentConstructiveDecoder(nn.Module):
    """L2C cascade seeded by the program code: Community -> Niche -> Genes.

    Given the clean/predicted program code ``sp [N,K]``:

        r = Linear(K -> d)                # program -> representation seed
        sc, r = community_stage(r)        # construct community [N,C], refine r
        sn, r = niche_stage(r)            # construct niche  [N,D], refine r
        genes = Linear(d -> G)(r)         # [N,G]

    Each construct-and-refine stage re-writes ``r`` (gated residual), so the next
    stage sees the previous decision — the sequential L2C construct loop.
    """

    def __init__(self, d_model, n_genes, n_programs, n_communities, n_niches):
        super().__init__()
        self.seed = nn.Sequential(
            nn.Linear(n_programs, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.community = ConstructiveStage(d_model, n_communities)
        self.niche = ConstructiveStage(d_model, n_niches)
        self.gene_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_genes),
        )

    def forward(self, sp):
        r = self.seed(sp)                       # [N, d]
        sc, r = self.community(r)               # [N, C]  (community latent)
        sn, r = self.niche(r)                   # [N, D]  (niche latent)
        genes = self.gene_decoder(r)            # [N, G]
        return genes, sc, sn


class LatentFlowDenoiser(nn.Module):
    """Program-space latent flow denoiser with a learned gene AE bottleneck.

    Training (``forward``):
        1. program_0 = E(gene_gt)                         # [N, K]  (AE code)
        2. p_t, t    = diffusier.corrupt_exp(program_0)   # corrupt in program space
        3. endpoint  = flow_blocks(p_t, img, coords, t)   # [N, K]  program endpoint
        4. genes, sc, sn = D(endpoint)                    # decode for supervision
        5. L = w_f*MSE(endpoint, program_0)               # flow in program space
             + w_g*L_gene(genes, gene_gt)                 # D inverts E
             + w_ae*MSE(D(E(gene_gt)), gene_gt)           # AE cycle consistency
             + lf_c*smooth(sc) + lf_n*smooth(sn)          # hierarchy neighbor consistency

    Inference (``inference``): returns ``(endpoint, (genes, sc, sn))`` so the sampling
    loop uses the program-space endpoint for the ODE and the genes for metrics.
    """

    def __init__(self, model_config, hflow_config=None):
        super().__init__()
        if hflow_config is None:
            hflow_config = HFlowConfig()
        self.mcfg = model_config
        self.hcfg = hflow_config
        self.d_model = getattr(hflow_config, "n_latent_d_model", model_config.d_model)
        self.n_genes = model_config.n_genes
        self.n_programs = hflow_config.n_programs
        self.n_communities = hflow_config.n_communities
        self.n_niches = hflow_config.n_niches

        # ── Learned AE bottleneck: E: genes -> program code ──
        self.encoder = nn.Sequential(
            nn.Linear(self.n_genes, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.n_programs),
        )

        # ── Flow backbone operating natively in program space ──
        # HFlowBlock uses n_genes only for (a) edge gene-diff dim, (b) flow_context_proj,
        # (c) velocity_head output — passing n_genes=n_programs makes it read/write [N,K].
        self.image_transform = nn.Linear(model_config.feature_dim, self.d_model)
        self.fourier_proj = nn.Sequential(
            nn.Linear(256, self.d_model), nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        if hflow_config.hflow_representation in ("slide_patch", "slide_region_patch"):
            self.hierarchy_encoder = HierarchyEncoder(self.d_model, hflow_config)
        self.blocks = nn.ModuleList([
            HFlowBlock(
                d_model=self.d_model,
                d_edge_model=model_config.pairwise_hidden_dim,
                n_genes=self.n_programs,          # program-space state
                n_heads=model_config.n_heads,
                activation=model_config.activation,
                attn_drop=model_config.attn_dropout,
                proj_drop=model_config.dropout,
                hflow_cross_scale=hflow_config.hflow_cross_scale,
                use_time_hierarchy_gate=hflow_config.use_time_hierarchy_gate,
                modulation_mode=hflow_config.modulation_mode,
                gate_mode=hflow_config.gate_mode,
                gate_hidden=hflow_config.gate_hidden,
            )
            for _ in range(model_config.n_layers)
        ])

        # ── L2C decoder: program -> community -> niche -> genes ──
        self.decoder = LatentConstructiveDecoder(
            d_model=self.d_model,
            n_genes=self.n_genes,
            n_programs=self.n_programs,
            n_communities=self.n_communities,
            n_niches=self.n_niches,
        )

        self.loss_func = nn.MSELoss()
        self.diffusier = None   # set after construction (train.py) for program-space corruption

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
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

    def _timestep_embed(self, t):
        """Sine-cosine timestep embedding -> [B, 256] then through fourier_proj [B, d]."""
        half = 128
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(0, half, dtype=torch.float32).to(t.device) / half
        )
        args = t[..., None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, 256]
        return self.fourier_proj(emb)

    def encode(self, gene_exp):
        """Map genes -> program code [N, K]. Accepts [*, G]."""
        return self.encoder(gene_exp)

    # ------------------------------------------------------------------ #
    # forward / inference
    # ------------------------------------------------------------------ #
    def forward(self, exp, img_features, coords, labels, t_steps, diffusier=None):
        """Training step. ``labels`` is gene-space; corruption happens in program space
        using ``diffusier`` (falling back to ``self.diffusier``). ``exp`` is ignored
        (the generic gene-space training call passes a noisy gene batch, but program
        corruption is recomputed here). Returns (endpoint_pred, total_loss)."""
        B, N_cells, _ = labels.shape
        pad_mask = img_features.sum(-1) == 0

        # 1. Clean program code from ground-truth genes (AE bottleneck).
        program_0 = self.encode(labels)                     # [B, N, K]
        program_0_flat = program_0[~pad_mask]

        # 2. Corrupt in program space.
        if diffusier is None:
            diffusier = self.diffusier
        if diffusier is None:
            raise ValueError("LatentFlowDenoiser.forward requires the Interpolant (diffusier).")
        p_t, t_steps = diffusier.corrupt_exp(program_0)     # [B, N, K]

        # 3. Flow backbone -> program endpoint.
        endpoint_pred, sc, sn, graph_idx = self._run_blocks(
            p_t, img_features, coords, t_steps, pad_mask)

        # 4. Reconstruct community/niche/genes from the *predicted* endpoint (gene + hierarchy loss).
        genes_endpoint, sc_end, sn_end = self.decoder(endpoint_pred)
        # AE cycle-consistency on the clean code.
        genes_cycle, _, _ = self.decoder(program_0_flat)

        gt_flat = labels[~pad_mask]
        w_f = self.hcfg.lambda_lf_flow
        w_g = self.hcfg.lambda_lf_gene
        w_ae = self.hcfg.lambda_lf_ae
        w_c = self.hcfg.lambda_lf_community
        w_n = self.hcfg.lambda_lf_niche

        l_flow = self.loss_func(endpoint_pred, program_0_flat)          # program-space flow
        l_gene = self.loss_func(genes_endpoint, gt_flat)                # D inverts E
        l_ae = self.loss_func(genes_cycle, gt_flat.detach())            # AE cycle consistency
        # neighbor smoothness on constructed community / niche
        l_com = ConstructiveDecoder.neighbor_smoothness(torch.softmax(sc_end, -1), graph_idx)
        l_nic = ConstructiveDecoder.neighbor_smoothness(sn_end, graph_idx)

        loss = (w_f * l_flow + w_g * l_gene + w_ae * l_ae
                + w_c * l_com + w_n * l_nic)
        return endpoint_pred, loss

    def _run_blocks(self, p_t, img_features, coords, t_steps, pad_mask):
        """Run the flows blocks on a program-space state; returns
        (endpoint_flat[N,K], sc[N,C], sn[N,D], nearest_indices)."""
        B, N_cells, _ = p_t.shape
        device = p_t.device
        batch_idx = torch.arange(B, device=device).unsqueeze(-1).repeat(1, N_cells)
        batch_idx = batch_idx[~pad_mask]
        p_t_flat = p_t[~pad_mask]                       # [N, K]
        coords_flat = coords[~pad_mask]                 # [N, 2]

        t_emb = self._timestep_embed(t_steps)           # [B, d]

        # Image hierarchy (region/slide context), program-agnostic.
        if hasattr(self, "hierarchy_encoder"):
            patch_embs = self.image_transform(img_features)
            z_patch, z_region, z_slide = self.hierarchy_encoder(
                patch_embs, coords, pad_mask=pad_mask)
            if self.hcfg.hflow_representation == "slide_patch":
                z_region = z_slide.expand(-1, 1, -1)
        else:
            z_patch = self.image_transform(img_features)
            z_region = None
            z_slide = None

        t_emb_batch = t_emb[:, None, :].expand(B, N_cells, -1)
        z_patch = z_patch + t_emb_batch
        z_patch_flat = z_patch[~pad_mask]

        nearest_indices = self._build_graph(
            coords_flat, batch_idx, min(self.mcfg.n_neighbors, N_cells), exclude_self=True)

        curr_z = z_patch_flat
        z_region_c, z_slide_c = z_region, z_slide
        endpoints = []
        for block in self.blocks:
            vel, curr_z, z_region_c, z_slide_c = block(
                noisy_exp_flat=p_t_flat,
                z_patch_flat=curr_z,
                z_region=z_region_c,
                z_slide=z_slide_c,
                coords_flat=coords_flat,
                neighbor_indices=nearest_indices,
                batch_idx=batch_idx,
                pad_mask=pad_mask,
                max_n_cells=N_cells,
                t=t_steps,
                t_emb=t_emb,
            )
            endpoints.append(vel)
        endpoint_flat = torch.stack(endpoints).mean(0)  # [N, K]
        sc = sn = None
        if self.decoder is not None:
            genes, sc, sn = self.decoder(endpoint_flat)
        return endpoint_flat, sc, sn, nearest_indices

    def inference(self, exp, img_features, coords, t_steps):
        """Single ODE step. ``exp`` is the program-space state [B, N, K].
        Returns (endpoint_pred [B,N,K], (genes [B,N,G], sc, sn)) — endpoint used by
        the ODE, genes for metrics."""
        B, N_cells, _ = exp.shape
        pad_mask = img_features.sum(-1) == 0
        endpoint_flat, sc, sn, _ = self._run_blocks(exp, img_features, coords, t_steps, pad_mask)

        endpoint = endpoint_flat.new_zeros(B, N_cells, self.n_programs)
        endpoint[~pad_mask] = endpoint_flat

        genes = sc = sn = None
        if self.decoder is not None:
            genes_flat, sc_flat, sn_flat = self.decoder(endpoint_flat)
            genes = genes_flat.new_zeros(B, N_cells, self.n_genes)
            genes[~pad_mask] = genes_flat
            sc = sc_flat.new_zeros(B, N_cells, self.n_communities)
            sc[~pad_mask] = sc_flat
            sn = sn_flat.new_zeros(B, N_cells, self.n_niches)
            sn[~pad_mask] = sn_flat

        return endpoint, (genes, sc, sn)

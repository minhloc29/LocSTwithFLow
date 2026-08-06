from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HFlowConfig:

    n_region_queries: int = 16       # number of learnable region tokens (K)
    region_hidden_dim: int = 128     # region token dimension (default = d_model)
    slide_hidden_dim: int = 128      # slide token dimension (default = d_model)
  
    hflow_representation: str = "slide_region_patch"
    hflow_cross_scale: str = "bidirectional"
    hflow_region_discovery: str = "learnable"

    hflow_assignment_temperature: float = 1.0
    hflow_assignment_entropy_weight: float = 0.0

    # Time-dependent hierarchical fusion gate
    use_time_hierarchy_gate: bool = True
    modulation_mode: str = "adaln"   # "static" | "scalar" | "adaln"
    gate_mode: str = "learnable"     # legacy scalar-gate mode (scalar ablation)
    gate_hidden: int = 128

    # ── Learned gene-program latent (Program → Community → Niche) ──
    # Off by default so existing flow-matching behavior is unchanged. When enabled,
    # a standalone L2C-style ConstructiveDecoder runs ONCE over the final spot
    # embedding: each stage constructs a latent (program -> community -> niche) then
    # fuses it back to refine the representation, ending with a gene decode. Losses:
    # low-rank gene recon through programs + graph-smoothness on community/niche.
    use_program_latent: bool = False
    n_programs: int = 32            # K latent program activations per spot
    n_communities: int = 8          # C soft communities per spot
    n_niches: int = 16              # D latent niche embedding dim per spot
    lambda_program: float = 0.1     # weight on low-rank gene reconstruction thru programs
    lambda_community: float = 0.05  # weight on neighbor-consistency of community logits
    lambda_niche: float = 0.05      # weight on neighbor-consistency of niche embeddings

    # ── Latent flow matching over the biological hierarchy ──
    # Off by default (gene-space HFlowDenoiser baseline preserved). When enabled,
    # a LatentFlowDenoiser transports PROGRAM states [N,K] through the flow ODE —
    # the hierarchy (Program -> Community -> Niche -> Genes) is part of the evolved
    # state, not a post-hoc decoder. Encoder E: genes->program; decoder D: the L2C
    # community->niche->genes cascade.
    use_latent_flow: bool = False
    n_latent_d_model: int = 128     # hidden dim of the latent-flow backbone/encoder/decoder
    lambda_lf_flow: float = 1.0     # flow matching loss in program space (MSE)
    lambda_lf_gene: float = 1.0     # gene reconstruction of decoded endpoint vs gene_gt
    lambda_lf_ae: float = 0.1       # AE cycle consistency: D(E(gene_gt)) ~ gene_gt
    lambda_lf_community: float = 0.05  # neighbor smoothness of constructed community
    lambda_lf_niche: float = 0.05      # neighbor smoothness of constructed niche

    def __post_init__(self):
        valid_repr = {"flat", "slide_patch", "slide_region_patch"}
        assert self.hflow_representation in valid_repr, \
            f"hflow_representation must be one of {valid_repr}, got '{self.hflow_representation}'"

        valid_cs = {"none", "bottom_up", "top_down", "bidirectional"}
        assert self.hflow_cross_scale in valid_cs, \
            f"hflow_cross_scale must be one of {valid_cs}, got '{self.hflow_cross_scale}'"

        valid_rd = {"learnable", "grid", "kmeans", "assignment"}
        assert self.hflow_region_discovery in valid_rd, \
            f"hflow_region_discovery must be one of {valid_rd}, got '{self.hflow_region_discovery}'"

        valid_gm = {"static", "fixed", "learnable"}
        assert self.gate_mode in valid_gm, \
            f"gate_mode must be one of {valid_gm}, got '{self.gate_mode}'"

        valid_mm = {"static", "scalar", "adaln"}
        assert self.modulation_mode in valid_mm, \
            f"modulation_mode must be one of {valid_mm}, got '{self.modulation_mode}'"

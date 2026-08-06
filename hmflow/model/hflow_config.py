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
    # each block predicts a K-dim program activation, a C-dim community mixing with
    # a low-rank program→gene decoder, and a graph-smoothness regularizer on the
    # community output over the k-NN spatial graph.
    use_program_latent: bool = False
    n_programs: int = 32            # K latent program activations per spot
    n_communities: int = 8          # C soft communities per spot
    lambda_program: float = 0.1     # weight on low-rank gene reconstruction thru programs
    lambda_community: float = 0.05  # weight on neighbor-consistency of community logits
    lambda_niche: float = 0.05      # weight on neighbor-consistency of program/niche emb

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

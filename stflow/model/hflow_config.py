"""
HFlow-ST Configuration — extends ModelConfig with hierarchical flow params.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HFlowConfig:
    """Hierarchical flow matching configuration."""

    # ── Hierarchy structure ──────────────────────────────────────────────
    n_region_queries: int = 16       # number of learnable region tokens (K)
    region_hidden_dim: int = 128     # region token dimension (default = d_model)
    slide_hidden_dim: int = 128      # slide token dimension (default = d_model)

    # ── Ablation knobs ───────────────────────────────────────────────────
    # Representation hierarchy level
    #   "flat"               — no hierarchy, equivalent to STFlow
    #   "slide_patch"        — slide token + patch tokens, no regions
    #   "slide_region_patch" — full hierarchy (default)
    hflow_representation: str = "slide_region_patch"

    # Whether to update the hierarchy at every flow step (dynamic)
    # If False, hierarchy is computed once and kept fixed.
    hflow_dynamic_update: bool = True

    # Cross-scale message-passing direction
    #   "none"         — no cross-scale, patch self-attention only
    #   "bottom_up"    — patch → region → slide only
    #   "top_down"     — slide → region → patch only
    #   "bidirectional" — both directions (default)
    hflow_cross_scale: str = "bidirectional"

    # Region discovery method
    #   "learnable" — learnable region queries with cross-attention (default)
    #   "grid"      — hard spatial grid partitioning
    #   "kmeans"    — k-means soft assignment
    #   "assignment" — dynamic patch-to-region soft assignment
    hflow_region_discovery: str = "learnable"

    # Dynamic region-assignment controls
    hflow_assignment_temperature: float = 1.0
    hflow_assignment_entropy_weight: float = 0.0

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

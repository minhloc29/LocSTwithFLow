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

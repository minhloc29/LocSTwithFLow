import torch
import torch.nn as nn


class HierarchyAdaLN(nn.Module):
    """
    Per-hierarchy-level AdaLN (scale/shift) + gate, conditioned on t_emb.

    scale/shift are per-feature (d_model); gate is a scalar per level.
    The final projection is zero-initialized (AdaLN-Zero) so the module
    starts as an identity (gate=0, 1+scale=1, shift=0) and the network
    learns modulation from there.

    Returns a dict: {level: (scale[B, d], shift[B, d], gate[B, 1])}.
    """

    def __init__(self, d_model, time_dim, levels=("patch", "region", "slide")):
        super().__init__()
        self.levels = list(levels)
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, len(self.levels) * (d_model * 2 + 1)),
        )
        # AdaLN-Zero: zero-init the final projection -> identity at init.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t_emb):
        # t_emb: [B, time_dim] -> raw [B, L*(2d+1)]
        raw = self.net(t_emb)
        params = {}
        i = 0
        for lvl in self.levels:
            scale = raw[:, i:i + self.d_model]; i += self.d_model
            shift = raw[:, i:i + self.d_model]; i += self.d_model
            gate = raw[:, i:i + 1]; i += 1
            params[lvl] = (scale, shift, gate)
        return params  # {level: (scale, shift, gate)}

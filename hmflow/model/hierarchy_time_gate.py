import torch
import torch.nn as nn


class HierarchyTimeGate(nn.Module):
    """
    Predict timestep-dependent Patch / Region / Slide fusion weights.

    A tiny MLP maps the raw flow timestep t to a 3-way softmax over
    (patch, region, slide). Softmax (not sigmoid) guarantees
    alpha_patch + alpha_region + alpha_slide = 1.
    """

    def __init__(self, hidden_dim, t_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(t_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        # AdaLN-Zero-style init: start neutral (weights = 1/3 each).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t):
        # t: [B] scalar -> [B, 3] logits
        t_in = t.unsqueeze(-1).float()
        logits = self.net(t_in)
        weights = torch.softmax(logits, dim=-1)  # [B, 3], sums to 1
        return weights[:, 0], weights[:, 1], weights[:, 2]  # patch, region, slide

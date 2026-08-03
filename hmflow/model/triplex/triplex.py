"""Adapted TRIPLEX model for STFlow's per-spot-embedding pipeline.

TRIPLEX originally consumes raw H&E patches (ResNet18/CIGAR CNN), a per-spot
neighbourhood of patch embeddings, and a whole-slide embedding. STFlow provides
per-spot patch *embeddings* (``features [B,N,D]``), ``coords [B,N,2]`` and gene
``labels [B,N,G]`` — no raw patches and no separate neighbour/global H5s.

This is a faithful re-scoring of the same building blocks and loss onto
embeddings:
  * target branch   -> Linear projection of the provided patch embeddings
  * neighbour branch-> k-NN neighbourhood built from coords + NeighborEncoder
  * global branch   -> per-spot GlobalEncoder with positional encoding
  * fusion + loss   -> identical to TRIPLEX (fusion MSE + (1-a) aux MSE + a distillation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from hmflow.model.triplex.modules import (
    GlobalEncoder,
    NeighborEncoder,
    CrossEncoder,
)


class FusionEncoder(nn.Module):
    """Per-spot fusion, faithful to TRIPLEX: the global spot token attends over
    the target tokens and over the neighbour tokens (two cross-attentions), then
    sums the two fused representations."""

    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.fusion_layer = CrossEncoder(emb_dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x_t=None, x_n=None, x_g=None, mask=None):
        # x_t / x_n / x_g: [N, emb]; mask [N, k] (kept for API symmetry)
        fus1 = self.fusion_layer(x_g.unsqueeze(1), x_t.unsqueeze(1))     # [N,1,emb]
        fus2 = self.fusion_layer(x_g.unsqueeze(1), x_n.unsqueeze(1))     # [N,1,emb]
        fusion = (fus1 + fus2).squeeze(1)
        fusion = self.norm(fusion)
        return fusion


class TriplexModel(nn.Module):
    def __init__(self,
                 feature_dim=1024,
                 emb_dim=512,
                 num_genes=250,
                 depth1=1,
                 depth2=3,
                 depth3=3,
                 num_heads1=4,
                 num_heads2=16,
                 num_heads3=16,
                 mlp_ratio1=4,
                 mlp_ratio2=4,
                 mlp_ratio3=1,
                 dropout1=0.2,
                 dropout2=0.1,
                 dropout3=0.3,
                 kernel_size=3,
                 pos_layer='APEG',
                 n_neighbors=25,
                 res_neighbor=None,
                 alpha=0.3):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_neighbors = n_neighbors
        self.alpha = alpha

        # The NeighborEncoder uses a fixed-grid attention bias (res_neighbor),
        # so the neighbourhood size must equal the grid area. Default 25 = 5x5.
        if res_neighbor is None:
            side = int(round(n_neighbors ** 0.5))
            assert side * side == n_neighbors, \
                f"triplex_n_neighbors must be a perfect square (25 for a 5x5 block), got {n_neighbors}"
            res_neighbor = (side, side)
        self.res_neighbor = res_neighbor

        # Multi-resolution projections (replace ResNet18/CIGAR + raw-image path)
        self.target_proj = nn.Linear(feature_dim, emb_dim)
        self.neighbor_proj = nn.Linear(feature_dim, emb_dim)
        self.global_proj = nn.Linear(feature_dim, emb_dim)

        # Neighbour branch
        self.neighbor_encoder = NeighborEncoder(
            emb_dim, depth3, num_heads3, int(emb_dim * mlp_ratio3),
            dropout=dropout3, resolution=res_neighbor)

        # Global branch
        self.global_encoder = GlobalEncoder(
            emb_dim, depth2, num_heads2, int(emb_dim * mlp_ratio2),
            dropout2, kernel_size, pos_layer)

        # Fusion branch
        self.fusion_encoder = FusionEncoder(
            emb_dim, depth1, num_heads1, int(emb_dim * mlp_ratio1), dropout1)

        # Heads
        self.fc = nn.Linear(emb_dim, num_genes)
        self.fc_target = nn.Linear(emb_dim, num_genes)
        self.fc_neighbor = nn.Linear(emb_dim, num_genes)
        self.fc_global = nn.Linear(emb_dim, num_genes)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _knn(coords, k):
        """Return per-spot k nearest-neighbour indices (self included). [N, k]"""
        N = coords.shape[0]
        k = min(k, N)
        d = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)  # [N, N]
        return d.topk(k, dim=-1, largest=False)[1]

    def _forward_sample(self, features, coords, labels):
        """features [N,D], coords [N,2], labels [N,G] -> dict of per-spot tensors."""
        N = features.shape[0]
        device = features.device

        # Neighbour branch: k-NN neighbourhood of embeddings.
        # Each spot is its own batch element; the k neighbours are the sequence.
        # The neighbourhood is always dense (k clamped to N), so no mask is passed.
        nb_idx = self._knn(coords, self.n_neighbors)              # [N,k]
        nb_emb = self.neighbor_proj(features[nb_idx])             # [N,k,emb]
        neighbor_token = self.neighbor_encoder(nb_emb)            # [N,k,emb]
        neighbor_token = neighbor_token.mean(1)                    # [N,emb]

        # Global branch: per-spot token over projected features with coords
        g = self.global_proj(features)                             # [N,emb]
        # Normalise coords to [-1,1]-ish for positional encoding stability
        pos = coords - coords.mean(0)
        pos = pos / (pos.max(0).values - pos.min(0).values + 1e-5)
        global_token = self.global_encoder(g.unsqueeze(0), pos).squeeze(0)  # [N,emb]

        # Target branch
        target_token = self.target_proj(features)                  # [N,emb]

        # Fusion
        fusion_token = self.fusion_encoder(target_token, neighbor_token, global_token)

        # Heads
        logits = torch.clamp(self.fc(fusion_token), 0)             # [N,G]
        out_target = torch.clamp(self.fc_target(target_token), 0)
        out_neighbor = torch.clamp(self.fc_neighbor(neighbor_token), 0)
        out_global = torch.clamp(self.fc_global(global_token), 0)

        result = {
            'logits': logits,
            'aux': (out_target, out_neighbor, out_global),
        }
        if labels is not None:
            result['loss'] = self.calculate_loss(
                (logits, out_target, out_neighbor, out_global), labels)
        return result

    def calculate_loss(self, preds, label):
        """Identical loss to TRIPLEX: fusion MSE + (1-a) aux MSE + a distillation."""
        loss = F.mse_loss(preds[0], label)
        for i in range(1, len(preds)):
            loss += F.mse_loss(preds[i], label) * (1 - self.alpha)
            loss += F.mse_loss(preds[0], preds[i]) * self.alpha
        return loss

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def forward(self, features, coords, labels=None):
        """features [B,N,D], coords [B,N,2], labels [B,N,G] -> dict.

        Each sample in the batch is an independent slide; padding spots (all-zero
        feature rows, produced by ``padding_batcher``) are masked out.
        """
        B, N, _ = features.shape
        device = features.device

        pad = (features.sum(-1) == 0)      # [B,N] True where padded
        logits_list, aux_list = [], []
        losses = []

        for b in range(B):
            f = features[b][~pad[b]]
            c = coords[b][~pad[b]]
            lb = labels[b][~pad[b]] if labels is not None else None

            if f.shape[0] == 0:
                continue

            out = self._forward_sample(f, c, lb)
            # Build dense [N,G] back (pad rows -> 0)
            dense_logits = torch.zeros(N, out['logits'].shape[-1], device=device)
            dense_logits[~pad[b]] = out['logits']
            logits_list.append(dense_logits)

            dense_aux = []
            for a in out['aux']:
                da = torch.zeros(N, a.shape[-1], device=device)
                da[~pad[b]] = a
                dense_aux.append(da)
            aux_list.append(dense_aux)

            if labels is not None:
                losses.append(out['loss'])

        if not logits_list:
            # all-padding edge case
            logits = torch.zeros(B, N, self.fc.out_features, device=device)
        else:
            logits = torch.stack(logits_list, dim=0)

        result = {'logits': logits}
        if labels is not None:
            result['loss'] = torch.stack(losses).mean()
        return result

    @torch.no_grad()
    def inference(self, features, coords):
        return self.forward(features, coords)['logits']

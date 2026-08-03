"""TRIPLEX building blocks, ported to STFlow.

Self-contained port of ``TRIPLEX/src/model/TRIPLEX/module.py`` with no heavy
dependencies: it uses plain (non-flash) attention and the dense APEG path, and
replaces ``einops.rearrange`` with standard tensor ops. FlashAttention and
MinkowskiEngine remain optional and are only used if importing them succeeds.
"""

import math
import itertools

import torch
from torch import nn

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func  # noqa: F401
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    import MinkowskiEngine as ME  # noqa: F401
    HAS_MINKOWSKI = True
except ImportError:
    HAS_MINKOWSKI = False


def _qkv_to_heads(qkv, h):
    """[B,N,h*dh,3] chunked -> ([B,h,N,dh], [B,h,N,dh], [B,h,N,dh])."""
    d = qkv.shape[-1] // 3
    q, k, v = qkv.chunk(3, dim=-1)
    f = lambda t: t.view(*t.shape[:-1], h, -1).transpose(-2, -3)
    return f(q), f(k), f(v)


def _heads_to_seq(x):
    """[B,h,N,dh] -> [B,N,h*dh]."""
    t = x.transpose(-2, -3)          # [B,N,h,dh]
    return t.contiguous().view(*t.shape[:-2], -1)


class PreNorm(nn.Module):
    def __init__(self, emb_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if 'x_kv' in kwargs.keys():
            kwargs['x_kv'] = self.norm(kwargs['x_kv'])
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, heads=4, dropout=0., attn_bias=False,
                 resolution=(5, 5), flash_attn=False):
        super().__init__()

        assert emb_dim % heads == 0, 'The dimension size must be a multiple of the number of heads.'

        self.flash_attn = flash_attn and HAS_FLASH_ATTN

        self.emb_dim = emb_dim
        dim_head = emb_dim // heads
        project_out = not (heads == 1)

        self.heads = heads
        self.drop_p = dropout
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(emb_dim, emb_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.attn_bias = attn_bias
        if attn_bias:
            points = list(itertools.product(
                range(resolution[0]), range(resolution[1])))
            N = len(points)
            attention_offsets = {}
            idxs = []
            for p1 in points:
                for p2 in points:
                    offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                    if offset not in attention_offsets:
                        attention_offsets[offset] = len(attention_offsets)
                    idxs.append(attention_offsets[offset])
            self.attention_biases = torch.nn.Parameter(
                torch.zeros(heads, len(attention_offsets)))
            self.register_buffer('attention_bias_idxs',
                                 torch.LongTensor(idxs).view(N, N),
                                 persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        # Always propagate the mode. The original TRIPLEX override only called
        # super().train(mode) when attn_bias was set, which left dropout enabled
        # in eval mode for plain attention — we fix that here.
        super().train(mode)
        if self.attn_bias:
            if mode and hasattr(self, 'ab'):
                del self.ab
            else:
                self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x, mask=None, return_attn=False):
        qkv = self.to_qkv(x)  # b x n x d*3
        dim_head = self.emb_dim // self.heads

        if self.flash_attn:
            qkv_h = qkv.view(*qkv.shape[:-1], self.heads, dim_head, 3)\
                        .permute(0, 1, 4, 2, 3).contiguous()
            out = flash_attn_qkvpacked_func(qkv_h, self.drop_p, softmax_scale=None, causal=False)
            out = out.permute(0, 1, 3, 2).contiguous().view(*qkv.shape[:-1], -1)
        else:
            q, k, v = _qkv_to_heads(qkv, self.heads)  # [B,h,N,dh]

            qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            if self.attn_bias:
                qk += (self.attention_biases[:, self.attention_bias_idxs]
                       if self.training else self.ab)

            if mask is not None:
                fill_value = torch.finfo(torch.float16).min
                ind_mask = mask.shape[-1]
                qk[:, :, -ind_mask:, -ind_mask:] = qk[:, :, -ind_mask:, -ind_mask:].masked_fill(mask == 0, fill_value)

            attn_weights = self.attend(qk)  # b h n n
            if return_attn:
                attn_weights_averaged = attn_weights.mean(dim=1)

            out = torch.matmul(attn_weights, v)
            out = _heads_to_seq(out)

            if return_attn:
                return self.to_out(out), attn_weights_averaged[:, 0]

        return self.to_out(out)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, emb_dim, heads=4, dropout=0., flash_attn=False):
        super().__init__()

        assert emb_dim % heads == 0, 'The dimension size must be a multiple of the number of heads.'

        self.flash_attn = flash_attn and HAS_FLASH_ATTN

        self.emb_dim = emb_dim
        dim_head = emb_dim // heads
        project_out = not (heads == 1)

        self.heads = heads
        self.drop_p = dropout
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.to_kv = nn.Linear(emb_dim, emb_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_q, x_kv, mask=None, return_attn=False):
        q = _qkv_to_heads(torch.cat([self.to_q(x_q),
                                     torch.zeros_like(self.to_q(x_q)),
                                     torch.zeros_like(self.to_q(x_q))], -1), self.heads)[0]
        kv = self.to_kv(x_kv).chunk(2, dim=-1)
        k, v = map(lambda t: t.view(*t.shape[:-1], self.heads, -1).transpose(-2, -3), kv)

        qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            fill_value = torch.finfo(torch.float16).min
            ind_mask = mask.shape[-1]
            qk[:, :, -ind_mask:, -ind_mask:] = qk[:, :, -ind_mask:, -ind_mask:].masked_fill(mask == 0, fill_value)

        attn_weights = self.attend(qk)  # b h n n
        if return_attn:
            attn_weights_averaged = attn_weights.mean(dim=1)

        out = torch.matmul(attn_weights, v)
        out = _heads_to_seq(out)

        if return_attn:
            return self.to_out(out), attn_weights_averaged[:, 0]

        return self.to_out(out)


class PosMLP(nn.Module):
    """MLP-based 2D positional encoding (optional pos_layer='MLP')."""

    def __init__(self, input_dim=2, embed_dim=1024, hidden_dim=512, grid_size=None):
        super().__init__()
        self.grid_size = grid_size
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )

    @staticmethod
    def infer_grid_size(pos, rounding_factor=20):
        pos_rounded = (pos / rounding_factor).round() * rounding_factor
        unique_x = torch.unique(pos_rounded[:, 0])
        unique_y = torch.unique(pos_rounded[:, 1])
        return (unique_x.numel(), unique_y.numel())

    def forward(self, x, pos):
        B = x.shape[0]
        device = x.device

        if self.grid_size is None:
            self.grid_size = self.infer_grid_size(pos, rounding_factor=20)

        W, H = self.grid_size

        pos_min = pos.min(dim=0, keepdim=True)[0]
        pos_max = pos.max(dim=0, keepdim=True)[0]
        pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-5)
        grid_pos = pos_norm * torch.tensor([W - 1, H - 1], device=device)
        grid_pos = grid_pos.round()

        pos_emb = self.mlp(grid_pos)  # (N, embed_dim)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, N, embed_dim)

        return x + pos_emb


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0., attn_bias=False,
                 resolution=(5, 5), flash_attn=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(emb_dim, MultiHeadAttention(emb_dim, heads=heads,
                                                    dropout=dropout, attn_bias=attn_bias,
                                                    resolution=resolution, flash_attn=flash_attn)),
                PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, mask=None, return_attn=False):
        for attn, ff in self.layers:
            if return_attn:
                attn_out, attn_weights = attn(x, mask=mask, return_attn=return_attn)
                x += attn_out
                x = ff(x) + x
            else:
                x = attn(x, mask=mask) + x
                x = ff(x) + x

        if return_attn:
            return x, attn_weights
        else:
            return x


class CrossEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(emb_dim, MultiHeadCrossAttention(emb_dim, heads=heads, dropout=dropout)),
                PreNorm(emb_dim, FeedForward(emb_dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x_q, x_kv, mask=None, return_attn=False):
        for attn, ff in self.layers:
            if return_attn:
                attn_out, attn_weights = attn(x_q, x_kv=x_kv, mask=mask, return_attn=return_attn)
                x_q += attn_out
                x_q = ff(x_q) + x_q
            else:
                x_q = attn(x_q, x_kv=x_kv, mask=mask) + x_q
                x_q = ff(x_q) + x_q

        if return_attn:
            return x_q, attn_weights
        else:
            return x_q


class APEG(nn.Module):
    """Anchor Position Encoding Generator (dense Conv2d version)."""

    def __init__(self, dim=512, kernel_size=3, grid_size=None, use_sparse=False, sparse_resolution=128):
        super().__init__()
        self.use_sparse = use_sparse
        self.grid_size = grid_size
        self.sparse_resolution = sparse_resolution

        if self.use_sparse:
            if not HAS_MINKOWSKI:
                raise ImportError("MinkowskiEngine is not installed.")
            self.proj = ME.MinkowskiConvolution(
                in_channels=dim, out_channels=dim, kernel_size=kernel_size,
                stride=1, dimension=2, bias=True)
        else:
            self.proj = nn.Conv2d(
                dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=True, groups=dim)

    @staticmethod
    def dynamic_rounding_factor(pos, base=1, scale_ref=1000.0):
        pos_range = (pos.max(0)[0] - pos.min(0)[0]).max().item()
        scale_ratio = pos_range / scale_ref
        return base * scale_ratio

    @staticmethod
    def infer_grid_size(pos, rounding_factor=None):
        if rounding_factor is None:
            rounding_factor = 20.0
        pos_rounded = (pos / rounding_factor).round() * rounding_factor
        unique_x = torch.unique(pos_rounded[:, 0])
        unique_y = torch.unique(pos_rounded[:, 1])
        return (unique_x.numel(), unique_y.numel())

    def forward(self, x, pos):
        """x: (B, N, dim), pos: (N, 2) -> (B, N, dim)"""
        B, N, C = x.shape
        device = x.device

        if self.use_sparse:
            pos_min = pos.min(dim=0, keepdim=True)[0]
            pos_max = pos.max(dim=0, keepdim=True)[0]
            pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-5)
            discrete_coords = (pos_norm * (self.sparse_resolution - 1)).round().int()
            batch_indices = torch.zeros((N, 1), dtype=torch.int, device=device)
            coords = torch.cat([batch_indices, discrete_coords], dim=1)
            sparse_input = ME.SparseTensor(features=x.squeeze(0), coordinates=coords)
            sparse_output = self.proj(sparse_input)
            out_features = sparse_output.features
            out_coords = sparse_output.coordinates[:, 1:]
            match_idx = []
            for i in range(N):
                match = ((out_coords == discrete_coords[i]).all(dim=1)).nonzero(as_tuple=True)[0]
                match_idx.append(match.item())
            match_idx = torch.tensor(match_idx, device=device, dtype=torch.long)
            return out_features[match_idx].unsqueeze(0)

        if self.grid_size is None:
            self.grid_size = self.infer_grid_size(pos, rounding_factor=20)

        W, H = self.grid_size

        pos_min = pos.min(dim=0, keepdim=True)[0]
        pos_max = pos.max(dim=0, keepdim=True)[0]
        pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-5)
        grid_pos = pos_norm * torch.tensor([W - 1, H - 1], device=device)
        grid_pos = grid_pos.round().long()

        gx, gy = grid_pos[:, 0], grid_pos[:, 1]
        idx_1d = gy * W + gx

        dense_x = torch.zeros((B, C, H * W), device=device)
        dense_mask = torch.zeros((B, 1, H * W), device=device)

        for b in range(B):
            dense_x[b] = dense_x[b].scatter_add(1, idx_1d.unsqueeze(0).expand(C, -1), x[b].transpose(0, 1))
            dense_mask[b] = dense_mask[b].scatter_add(1, idx_1d.unsqueeze(0), torch.ones(1, N, device=device))

        dense_x = dense_x.view(B, C, H, W)
        dense_mask = dense_mask.view(B, 1, H, W).clamp(min=1.0)
        dense_x = dense_x / dense_mask

        x_pos = self.proj(dense_x)

        mask = (dense_mask > 0)
        x_pos = x_pos * mask + dense_x * (~mask)

        x_out = []
        for b in range(B):
            sampled_feat = x_pos[b, :, grid_pos[:, 1], grid_pos[:, 0]].transpose(0, 1)  # (N, C)
            x_out.append(sampled_feat)
        return torch.stack(x_out, dim=0)


class FourierPositionalEncoding(nn.Module):
    def __init__(self, coord_dim=2, embed_dim=128, scale=10.0):
        super().__init__()
        self.coord_dim = coord_dim
        self.embed_dim = embed_dim
        self.scale = scale
        assert embed_dim % (2 * coord_dim) == 0
        self.freqs = nn.Parameter(torch.randn(embed_dim // 2, coord_dim) * scale)

    def forward(self, coord):
        coord = coord.unsqueeze(-2)
        freqs = self.freqs.view(1, 1, -1, self.coord_dim)
        x_proj = 2 * math.pi * torch.sum(coord * freqs, dim=-1)
        pe = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return pe


class SpatialFormerBlock(nn.Module):
    def __init__(self, dim, n_heads=8, mlp_ratio=4, dropout=0.1, flash_attn=False):
        super().__init__()
        self.cross_attn_s2i = MultiHeadCrossAttention(emb_dim=dim, heads=n_heads, dropout=dropout, flash_attn=flash_attn)
        self.cross_attn_i2s = MultiHeadCrossAttention(emb_dim=dim, heads=n_heads, dropout=dropout, flash_attn=flash_attn)
        self.self_attn_s = MultiHeadAttention(emb_dim=dim, heads=n_heads, dropout=dropout, flash_attn=flash_attn)

        self.norm_s1 = nn.LayerNorm(dim)
        self.norm_s2 = nn.LayerNorm(dim)
        self.norm_s3 = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)

        self.ffn_s = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.ReLU(), nn.Linear(dim * mlp_ratio, dim))
        self.ffn_i = nn.Sequential(nn.Linear(dim, dim * mlp_ratio), nn.ReLU(), nn.Linear(dim * mlp_ratio, dim))

    def forward(self, image_tokens, spatial_tokens):
        s_attn = self.cross_attn_s2i(spatial_tokens, image_tokens)
        spatial_tokens = spatial_tokens + s_attn
        spatial_tokens = self.norm_s1(spatial_tokens)

        sa = self.self_attn_s(spatial_tokens)
        spatial_tokens = spatial_tokens + sa
        spatial_tokens = self.norm_s2(spatial_tokens)
        spatial_tokens = spatial_tokens + self.ffn_s(self.norm_s3(spatial_tokens))

        i_attn = self.cross_attn_i2s(image_tokens, spatial_tokens)
        image_tokens = image_tokens + i_attn
        image_tokens = image_tokens + self.ffn_i(self.norm_i(image_tokens))

        return image_tokens, spatial_tokens


class GlobalEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0., kernel_size=3,
                 pos_method='APEG', flash_attn=False):
        super().__init__()

        self.pos_method = pos_method
        if pos_method == 'MLP':
            self.pos_layer = PosMLP(input_dim=2, embed_dim=emb_dim, hidden_dim=emb_dim // 2)
        elif pos_method == 'APEG':
            self.pos_layer = APEG(dim=emb_dim, kernel_size=kernel_size)
        elif pos_method == 'None':
            self.pos_layer = None
        elif pos_method == 'spatialformer':
            self.pos_layer = FourierPositionalEncoding(coord_dim=2, embed_dim=emb_dim)
            self.spatial_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        else:
            raise ValueError(f"Unknown pos_layer: {pos_method}. Choose 'MLP' or 'APEG' or 'None'.")

        if pos_method == 'APEG':
            assert depth > 1, "APEG requires depth > 1."
            self.layer1 = TransformerEncoder(emb_dim, 1, heads, mlp_dim, dropout, flash_attn=flash_attn)
            self.layer2 = TransformerEncoder(emb_dim, depth - 1, heads, mlp_dim, dropout, flash_attn=flash_attn)
        elif pos_method == 'spatialformer':
            self.layers = nn.ModuleList([SpatialFormerBlock(emb_dim, heads, mlp_dim // emb_dim, dropout,
                                                            flash_attn=flash_attn) for _ in range(depth)])
        else:
            self.layer = TransformerEncoder(emb_dim, depth, heads, mlp_dim, dropout, flash_attn=flash_attn)

        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, position):
        if self.pos_method == 'APEG':
            x = self.layer1(x)
            x = self.pos_layer(x, position)
            x = self.layer2(x)
        elif self.pos_method == 'spatialformer':
            spatial_pos = self.pos_layer(position)
            spatial_tokens = spatial_pos + self.spatial_token
            for layer in self.layers:
                x, spatial_tokens = layer(x, spatial_tokens)
        else:
            if self.pos_method == 'MLP':
                x = self.pos_layer(x, position)
            x = self.layer(x)

        x = self.norm(x)
        return x


class NeighborEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout=0., resolution=(5, 5)):
        super().__init__()
        self.layer = TransformerEncoder(emb_dim, depth, heads, mlp_dim, dropout,
                                        attn_bias=True, resolution=resolution)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        x = self.layer(x, mask=mask)
        x = self.norm(x)
        return x


class FusionEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.fusion_layer = CrossEncoder(emb_dim, depth, heads, mlp_dim, dropout)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x_t=None, x_n=None, x_g=None, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)

        fus1 = self.fusion_layer(x_g.unsqueeze(1), x_t)
        fus2 = self.fusion_layer(x_g.unsqueeze(1), x_n, mask=mask)

        fusion = (fus1 + fus2).squeeze(1)
        fusion = self.norm(fusion)
        return fusion

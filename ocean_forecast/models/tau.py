from __future__ import annotations

import math

import torch
from torch import nn

from .base import BaseForecaster


def _group_count(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ConvResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = _group_count(int(out_channels))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act2 = nn.GELU()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        return self.act2(y + self.skip(x))


class LightTAUBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TAUForecaster(BaseForecaster):
    """Temporal Attention Unit forecaster with conditional future queries."""

    def __init__(
        self,
        input_channels: int = 4,
        output_channels: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        max_pred_len: int = 256,
        default_pred_len: int = 72,
        tau_block_layers: int = 0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.default_pred_len = int(default_pred_len)
        self.max_pred_len = int(max_pred_len)
        self.tau_block_layers = int(tau_block_layers)
        if self.tau_block_layers < 0 or self.tau_block_layers > 2:
            raise ValueError(f"tau_block_layers must be in [0,2], got {self.tau_block_layers}")

        self.encoder = nn.Sequential(
            ConvResBlock(input_channels, self.hidden_dim),
            ConvResBlock(self.hidden_dim, self.hidden_dim),
        )
        block_heads = 4 if (self.hidden_dim % 4 == 0) else 1
        self.tau_blocks = nn.ModuleList(
            [LightTAUBlock(self.hidden_dim, num_heads=block_heads, dropout=dropout) for _ in range(self.tau_block_layers)]
        )

        self.token_norm = nn.LayerNorm(self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.query_pos = nn.Parameter(torch.randn(self.max_pred_len, self.hidden_dim) * 0.02)
        self.query_cond = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.decoder_block = ConvResBlock(self.hidden_dim, self.hidden_dim)
        self.decoder_head = nn.Conv2d(self.hidden_dim, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected x shape [B,T,C,H,W], got {tuple(x.shape)}")

        b, t_in, c_in, h, w = x.shape
        t_out = int(pred_len) if pred_len is not None else self.default_pred_len
        if t_out > self.max_pred_len:
            raise ValueError(f"pred_len={t_out} exceeds max_pred_len={self.max_pred_len}")

        enc = self.encoder(x.reshape(b * t_in, c_in, h, w))
        feats = enc.reshape(b, t_in, self.hidden_dim, h, w)

        tokens = self.token_norm(feats.mean(dim=(-1, -2)))
        for block in self.tau_blocks:
            tokens = block(tokens)

        keys = self.key_proj(tokens)
        history_cond = self.query_cond(tokens.mean(dim=1)).unsqueeze(1)
        queries = self.query_pos[:t_out].unsqueeze(0).expand(b, -1, -1) + history_cond
        queries = self.query_norm(queries)

        attn_scores = torch.einsum("bsd,btd->bst", queries, keys) / math.sqrt(float(self.hidden_dim))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.einsum("bst,btdhw->bsdhw", attn_weights, feats)
        context = context + queries.unsqueeze(-1).unsqueeze(-1)

        decoded = self.decoder_block(self.dropout(context.reshape(b * t_out, self.hidden_dim, h, w)))
        decoded = self.decoder_head(decoded)
        return decoded.reshape(b, t_out, -1, h, w)

from __future__ import annotations

import math

import torch
from torch import nn

from .base import BaseForecaster


class TAUForecaster(BaseForecaster):
    """Temporal Attention Unit forecaster."""

    def __init__(
        self,
        input_channels: int = 4,
        output_channels: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        max_pred_len: int = 256,
        default_pred_len: int = 72,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.default_pred_len = int(default_pred_len)
        self.max_pred_len = int(max_pred_len)

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.future_queries = nn.Parameter(torch.randn(self.max_pred_len, self.hidden_dim) * 0.02)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, output_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected x shape [B,T,C,H,W], got {tuple(x.shape)}")

        b, t_in, c_in, h, w = x.shape
        t_out = int(pred_len) if pred_len is not None else self.default_pred_len
        if t_out > self.max_pred_len:
            raise ValueError(f"pred_len={t_out} exceeds max_pred_len={self.max_pred_len}")

        enc = self.encoder(x.reshape(b * t_in, c_in, h, w))
        feats = enc.reshape(b, t_in, self.hidden_dim, h, w)

        tokens = feats.mean(dim=(-1, -2))
        keys = self.key_proj(tokens)
        queries = self.future_queries[:t_out].unsqueeze(0).expand(b, -1, -1)

        attn_scores = torch.einsum("bsd,btd->bst", queries, keys) / math.sqrt(float(self.hidden_dim))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.einsum("bst,btdhw->bsdhw", attn_weights, feats)

        decoded = self.decoder(self.dropout(context.reshape(b * t_out, self.hidden_dim, h, w)))
        return decoded.reshape(b, t_out, -1, h, w)

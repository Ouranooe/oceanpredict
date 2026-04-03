from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .base import BaseForecaster


def _build_2d_sincos_pos_embed(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4 for 2D sin-cos position embedding, got {dim}.")

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    freq_dim = dim // 4
    freq_idx = torch.arange(freq_dim, device=device, dtype=torch.float32)
    freq = torch.exp(-math.log(10000.0) * freq_idx / max(freq_dim - 1, 1))

    y = yy.reshape(-1, 1) * freq.reshape(1, -1)
    x = xx.reshape(-1, 1) * freq.reshape(1, -1)
    pos = torch.cat([torch.sin(y), torch.cos(y), torch.sin(x), torch.cos(x)], dim=1)
    return pos.to(dtype=dtype)


class SpatialTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, D]
        b, t, n, d = x.shape
        z = x.reshape(b * t, n, d)
        attn_in = self.norm1(z)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        z = z + self.dropout(attn_out)
        z = z + self.dropout(self.ffn(self.norm2(z)))
        return z.reshape(b, t, n, d)


class GatedTemporalBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=float(attn_dropout),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, D]
        b, t, n, d = x.shape
        z = x.permute(0, 2, 1, 3).reshape(b * n, t, d)

        attn_in = self.norm1(z)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, attn_mask=attn_mask, need_weights=False)
        z = z + self.dropout(attn_out)

        ffn_in = self.norm2(z)
        gated = torch.sigmoid(self.gate(ffn_in)) * self.ffn(ffn_in)
        z = z + self.dropout(gated)
        return z.reshape(b, n, t, d).permute(0, 2, 1, 3)


class PredFormer(BaseForecaster):
    """Recurrent-free spatiotemporal transformer with gated temporal blocks."""

    def __init__(
        self,
        input_channels: int = 4,
        output_channels: int = 3,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        patch_size: int = 4,
        temporal_kernel: str = "binary_ts",
        max_input_len: int = 256,
        max_pred_len: int = 256,
        default_pred_len: int = 72,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.mlp_ratio = float(mlp_ratio)
        self.patch_size = int(patch_size)
        self.max_input_len = int(max_input_len)
        self.max_pred_len = int(max_pred_len)
        self.default_pred_len = int(default_pred_len)
        self.temporal_kernel = str(temporal_kernel).lower()

        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {self.embed_dim}.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim={self.embed_dim} must be divisible by num_heads={self.num_heads}.")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}.")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}.")
        if self.max_input_len <= 0 or self.max_pred_len <= 0:
            raise ValueError("max_input_len and max_pred_len must be > 0.")
        if self.default_pred_len <= 0:
            raise ValueError(f"default_pred_len must be > 0, got {self.default_pred_len}.")
        if self.temporal_kernel not in {"binary_ts", "full"}:
            raise ValueError(
                f"Unsupported temporal_kernel='{temporal_kernel}'. Expected one of: binary_ts, full."
            )

        self.patch_embed = nn.Conv2d(
            self.input_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.patch_decode = nn.ConvTranspose2d(
            self.embed_dim,
            self.output_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.history_cond = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.future_queries = nn.Parameter(torch.randn(self.max_pred_len, self.embed_dim) * 0.02)
        self.temporal_pos = nn.Parameter(torch.randn(self.max_input_len + self.max_pred_len, self.embed_dim) * 0.02)

        self.spatial_blocks = nn.ModuleList(
            [
                SpatialTransformerBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=float(dropout),
                    attn_dropout=float(attn_dropout),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.temporal_blocks = nn.ModuleList(
            [
                GatedTemporalBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=float(dropout),
                    attn_dropout=float(attn_dropout),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.final_dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        self._mask_cache: Dict[Tuple[int, str, str], torch.Tensor] = {}

    def _build_temporal_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(seq_len), str(device), str(dtype))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
        if self.temporal_kernel == "full":
            disallowed = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
            mask = mask.masked_fill(disallowed, float("-inf"))
        else:
            allow = torch.zeros((seq_len, seq_len), device=device, dtype=torch.bool)
            for i in range(seq_len):
                allow[i, i] = True
                step = 1
                while i - step >= 0:
                    allow[i, i - step] = True
                    step *= 2
            mask = mask.masked_fill(~allow, float("-inf"))

        self._mask_cache[key] = mask
        return mask

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected x shape [B,T,C,H,W], got {tuple(x.shape)}")

        b, t_in, c_in, h, w = x.shape
        if c_in != self.input_channels:
            raise ValueError(f"input_channels mismatch: expected {self.input_channels}, got {c_in}.")

        t_out = int(pred_len) if pred_len is not None else self.default_pred_len
        if t_in > self.max_input_len:
            raise ValueError(f"input_len={t_in} exceeds max_input_len={self.max_input_len}.")
        if t_out <= 0:
            raise ValueError(f"pred_len must be > 0, got {t_out}.")
        if t_out > self.max_pred_len:
            raise ValueError(f"pred_len={t_out} exceeds max_pred_len={self.max_pred_len}.")
        if (t_in + t_out) > self.temporal_pos.shape[0]:
            raise ValueError(
                f"input_len + pred_len = {t_in + t_out} exceeds temporal position table "
                f"size {self.temporal_pos.shape[0]}."
            )

        x_btchw = x.reshape(b * t_in, c_in, h, w)
        pad_h = (-h) % self.patch_size
        pad_w = (-w) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            # Pad to patch-aligned spatial size, then crop prediction back to original H/W.
            x_btchw = F.pad(x_btchw, (0, pad_w, 0, pad_h))
        h_pad = h + pad_h
        w_pad = w + pad_w

        patch_tokens = self.patch_embed(x_btchw)
        hp, wp = patch_tokens.shape[-2:]
        n_tokens = hp * wp
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2).reshape(b, t_in, n_tokens, self.embed_dim)

        spatial_pos = _build_2d_sincos_pos_embed(
            height=hp,
            width=wp,
            dim=self.embed_dim,
            device=x.device,
            dtype=patch_tokens.dtype,
        )
        spatial_pos = spatial_pos.view(1, 1, n_tokens, self.embed_dim)
        patch_tokens = patch_tokens + spatial_pos

        history_global = patch_tokens.mean(dim=(1, 2))
        future_bias = self.history_cond(history_global).view(b, 1, 1, self.embed_dim)
        future_tokens = self.future_queries[:t_out].view(1, t_out, 1, self.embed_dim)
        future_tokens = future_tokens + future_bias
        future_tokens = future_tokens.expand(-1, -1, n_tokens, -1) + spatial_pos

        tokens = torch.cat([patch_tokens, future_tokens], dim=1)
        total_t = t_in + t_out
        tokens = tokens + self.temporal_pos[:total_t].view(1, total_t, 1, self.embed_dim)

        temporal_mask = self._build_temporal_mask(total_t, device=tokens.device, dtype=tokens.dtype)
        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            tokens = spatial_block(tokens)
            tokens = temporal_block(tokens, attn_mask=temporal_mask)

        future = self.final_norm(tokens[:, t_in:, :, :])
        future = self.final_dropout(future)
        future = future.reshape(b * t_out, n_tokens, self.embed_dim).transpose(1, 2).reshape(
            b * t_out, self.embed_dim, hp, wp
        )
        pred = self.patch_decode(future)
        pred = pred.reshape(b, t_out, self.output_channels, h_pad, w_pad)
        return pred[:, :, :, :h, :w]


PredFormerForecaster = PredFormer

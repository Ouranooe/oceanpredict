from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence

import torch
from torch import nn

from .base import BaseForecaster


class _TemporalProbSparseBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=int(hidden_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(int(hidden_dim), int(ff_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(int(ff_dim), int(hidden_dim)),
        )
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

    def _full_attention(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, float]:
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        return attn_out, 1.0

    def _probsparse_attention(
        self,
        x_norm: torch.Tensor,
        probsparse_factor: float,
        probsparse_min_k: int,
    ) -> tuple[torch.Tensor, float]:
        # x_norm: [B, T, D]
        bsz, seq_len, hidden_dim = x_norm.shape
        min_queries = max(1, int(probsparse_min_k))
        if seq_len <= max(8, 2 * min_queries):
            return self._full_attention(x_norm)

        top_queries = int(math.ceil(float(probsparse_factor) * math.log(max(seq_len, 2))))
        top_queries = max(min_queries, top_queries)
        top_queries = min(seq_len, top_queries)
        if top_queries >= seq_len:
            return self._full_attention(x_norm)

        # Informer-like query sparsity score: max(QK) - mean(QK).
        score = torch.matmul(x_norm, x_norm.transpose(1, 2)) / math.sqrt(float(hidden_dim))
        sparsity = score.max(dim=-1).values - score.mean(dim=-1)
        top_idx = torch.topk(sparsity, k=top_queries, dim=1, largest=True, sorted=False).indices

        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, hidden_dim)
        sparse_query = torch.gather(x_norm, dim=1, index=gather_idx)
        sparse_context, _ = self.attn(sparse_query, x_norm, x_norm, need_weights=False)

        base_context = x_norm.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        context = base_context.scatter(1, gather_idx, sparse_context)
        ratio = float(top_queries) / float(seq_len)
        return context, ratio

    def forward(
        self,
        x: torch.Tensor,
        temporal_attention: str,
        probsparse_factor: float,
        probsparse_min_k: int,
    ) -> tuple[torch.Tensor, float]:
        x_norm = self.norm1(x)
        if temporal_attention == "probsparse":
            attn_out, ratio = self._probsparse_attention(
                x_norm=x_norm,
                probsparse_factor=probsparse_factor,
                probsparse_min_k=probsparse_min_k,
            )
        else:
            attn_out, ratio = self._full_attention(x_norm)

        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, ratio


class _LocalMultiScaleExtractor(nn.Module):
    def __init__(self, hidden_dim: int, dilations: Sequence[int]):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        int(hidden_dim),
                        int(hidden_dim),
                        kernel_size=3,
                        padding=int(d),
                        dilation=int(d),
                        groups=int(hidden_dim),
                    ),
                    nn.Conv2d(int(hidden_dim), int(hidden_dim), kernel_size=1),
                    nn.GELU(),
                )
                for d in dilations
            ]
        )
        self.out_proj = nn.Conv2d(int(hidden_dim), int(hidden_dim), kernel_size=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B, T, D, H, W]
        bsz, t_in, hidden_dim, h, w = feats.shape
        x = feats.reshape(bsz * t_in, hidden_dim, h, w)
        outs = [branch(x) for branch in self.branches]
        merged = torch.stack(outs, dim=0).mean(dim=0)
        merged = self.out_proj(merged)
        return merged.reshape(bsz, t_in, hidden_dim, h, w)


class CNNTransformerForecaster(BaseForecaster):
    """CNN encoder + transformer backbone + light cross-attn conditioned decoder."""

    def __init__(
        self,
        input_channels: int = 4,
        output_channels: int = 3,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_input_len: int = 256,
        max_pred_len: int = 256,
        default_pred_len: int = 72,
        temporal_attention: str = "full",
        probsparse_factor: float = 5.0,
        probsparse_min_k: int = 8,
        local_fusion_enabled: bool = False,
        local_dilations: Iterable[int] = (1, 2),
        local_fusion_type: str = "gated",
        front_aux_enabled: bool = False,
        front_aux_hidden_dim: int = 16,
        front_aux_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.default_pred_len = int(default_pred_len)
        self.max_input_len = int(max_input_len)
        self.max_pred_len = int(max_pred_len)
        self.temporal_attention_mode = str(temporal_attention).lower()
        self.probsparse_factor = float(probsparse_factor)
        self.probsparse_min_k = int(probsparse_min_k)
        self.local_fusion_enabled = bool(local_fusion_enabled)
        self.local_fusion_type = str(local_fusion_type).lower()
        self.front_aux_enabled = bool(front_aux_enabled)
        self.front_aux_hidden_dim = int(front_aux_hidden_dim)
        self.front_aux_dropout = float(front_aux_dropout)
        self._last_sparse_query_ratio = 1.0

        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by num_heads={num_heads}"
            )
        if self.temporal_attention_mode not in {"full", "probsparse"}:
            raise ValueError(
                f"Unsupported temporal_attention='{temporal_attention}'. Expected 'full' or 'probsparse'."
            )
        if self.probsparse_factor <= 0:
            raise ValueError(f"probsparse_factor must be > 0, got {self.probsparse_factor}.")
        if self.probsparse_min_k <= 0:
            raise ValueError(f"probsparse_min_k must be > 0, got {self.probsparse_min_k}.")
        if self.local_fusion_type not in {"gated", "film"}:
            raise ValueError(
                f"Unsupported local_fusion_type='{local_fusion_type}'. Expected 'gated' or 'film'."
            )
        if self.front_aux_hidden_dim <= 0:
            raise ValueError(f"front_aux_hidden_dim must be > 0, got {self.front_aux_hidden_dim}.")
        if self.front_aux_dropout < 0:
            raise ValueError(f"front_aux_dropout must be >= 0, got {self.front_aux_dropout}.")

        try:
            dilations = [int(d) for d in local_dilations]
        except TypeError as exc:
            raise ValueError("local_dilations must be an iterable of positive integers.") from exc
        if not dilations:
            raise ValueError("local_dilations must not be empty.")
        if any(d <= 0 for d in dilations):
            raise ValueError(f"local_dilations must be positive integers, got {dilations}.")
        self.local_dilations = tuple(sorted(set(dilations)))

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ff_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.backbone_norm = nn.LayerNorm(self.hidden_dim)
        self.probsparse_blocks = nn.ModuleList(
            [
                _TemporalProbSparseBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=int(num_heads),
                    ff_dim=int(ff_dim),
                    dropout=float(dropout),
                )
                for _ in range(int(num_layers))
            ]
            if self.temporal_attention_mode == "probsparse"
            else []
        )

        self.input_pos = nn.Parameter(torch.randn(self.max_input_len, self.hidden_dim) * 0.02)
        self.future_queries = nn.Parameter(torch.randn(self.max_pred_len, self.hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.future_norm = nn.LayerNorm(self.hidden_dim)
        self.future_to_film = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
        )
        self.local_branch = (
            _LocalMultiScaleExtractor(hidden_dim=self.hidden_dim, dilations=self.local_dilations)
            if self.local_fusion_enabled
            else None
        )
        self.local_gate = (
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Sigmoid(),
            )
            if self.local_fusion_enabled and self.local_fusion_type == "gated"
            else None
        )
        self.local_to_film = (
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            )
            if self.local_fusion_enabled and self.local_fusion_type == "film"
            else None
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.decoder = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, output_channels, kernel_size=1),
        )
        self.front_head = (
            nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.front_aux_hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(self.front_aux_dropout) if self.front_aux_dropout > 0 else nn.Identity(),
                nn.Conv2d(self.front_aux_hidden_dim, 1, kernel_size=1),
            )
            if self.front_aux_enabled
            else None
        )

    @property
    def last_sparse_query_ratio(self) -> float:
        return float(self._last_sparse_query_ratio)

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor | Dict[str, torch.Tensor]:
        if x.dim() != 5:
            raise ValueError(f"Expected x shape [B,T,C,H,W], got {tuple(x.shape)}")

        b, t_in, c_in, h, w = x.shape
        t_out = int(pred_len) if pred_len is not None else self.default_pred_len
        if t_in > self.max_input_len:
            raise ValueError(f"input_len={t_in} exceeds max_input_len={self.max_input_len}")
        if t_out > self.max_pred_len:
            raise ValueError(f"pred_len={t_out} exceeds max_pred_len={self.max_pred_len}")

        enc = self.encoder(x.reshape(b * t_in, c_in, h, w))
        feats = enc.reshape(b, t_in, self.hidden_dim, h, w)

        history_tokens = feats.mean(dim=(-1, -2))
        history_tokens = history_tokens + self.input_pos[:t_in].unsqueeze(0)
        if self.temporal_attention_mode == "probsparse":
            sparse_ratios: list[float] = []
            for block in self.probsparse_blocks:
                history_tokens, ratio = block(
                    history_tokens,
                    temporal_attention=self.temporal_attention_mode,
                    probsparse_factor=self.probsparse_factor,
                    probsparse_min_k=self.probsparse_min_k,
                )
                sparse_ratios.append(float(ratio))
            self._last_sparse_query_ratio = sum(sparse_ratios) / max(len(sparse_ratios), 1)
            history_tokens = self.backbone_norm(history_tokens)
        else:
            self._last_sparse_query_ratio = 1.0
            history_tokens = self.backbone_norm(self.backbone(history_tokens))

        query_tokens = self.future_queries[:t_out].unsqueeze(0).expand(b, -1, -1)
        cross_tokens, attn_weights = self.cross_attn(
            query=query_tokens,
            key=history_tokens,
            value=history_tokens,
            need_weights=True,
        )
        future_tokens = self.future_norm(query_tokens + cross_tokens)

        context = torch.einsum("bst,btdhw->bsdhw", attn_weights, feats)
        if self.local_branch is not None:
            local_feats = self.local_branch(feats)
            local_context = torch.einsum("bst,btdhw->bsdhw", attn_weights, local_feats)
            if self.local_gate is not None:
                gate = self.local_gate(self.dropout(future_tokens))
                context = context + gate.unsqueeze(-1).unsqueeze(-1) * local_context
            elif self.local_to_film is not None:
                local_film = self.local_to_film(self.dropout(future_tokens))
                local_scale, local_bias = torch.chunk(local_film, chunks=2, dim=-1)
                local_context = (
                    local_context * (1.0 + local_scale.unsqueeze(-1).unsqueeze(-1))
                    + local_bias.unsqueeze(-1).unsqueeze(-1)
                )
                context = context + local_context

        film = self.future_to_film(self.dropout(future_tokens))
        scale, bias = torch.chunk(film, chunks=2, dim=-1)
        context = context * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + bias.unsqueeze(-1).unsqueeze(-1)

        decoded_in = self.dropout(context.reshape(b * t_out, self.hidden_dim, h, w))
        decoded = self.decoder(decoded_in)
        field = decoded.reshape(b, t_out, -1, h, w)
        if self.front_head is None:
            return field
        front_logits = self.front_head(decoded_in).reshape(b, t_out, 1, h, w)
        return {
            "field": field,
            "front_mask_logits": front_logits,
        }

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn

from .base import BaseForecaster


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.gate_conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        gates = self.gate_conv(torch.cat([x, h_prev], dim=1))
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(BaseForecaster):
    def __init__(
        self,
        input_channels: int = 4,
        output_channels: int = 3,
        hidden_dims: Sequence[int] = (32, 32),
        kernel_size: int = 3,
        dropout: float = 0.0,
        default_pred_len: int = 72,
    ):
        super().__init__()
        if len(hidden_dims) != 2:
            raise ValueError("hidden_dims must have exactly 2 elements for this baseline.")

        h1, h2 = int(hidden_dims[0]), int(hidden_dims[1])
        self.default_pred_len = int(default_pred_len)

        self.encoder_cells = nn.ModuleList(
            [
                ConvLSTMCell(input_channels, h1, kernel_size),
                ConvLSTMCell(h1, h2, kernel_size),
            ]
        )
        self.decoder_cells = nn.ModuleList(
            [
                ConvLSTMCell(1, h1, kernel_size),
                ConvLSTMCell(h1, h2, kernel_size),
            ]
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(h2, output_channels, kernel_size=1)

    @staticmethod
    def _zeros_state(
        batch_size: int,
        channels: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, channels, height, width, device=device, dtype=dtype)
        c = torch.zeros(batch_size, channels, height, width, device=device, dtype=dtype)
        return h, c

    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected x shape [B,T,C,H,W], got {tuple(x.shape)}")
        b, t_in, _, h, w = x.shape
        t_out = int(pred_len) if pred_len is not None else self.default_pred_len
        device = x.device
        dtype = x.dtype

        enc_states = [
            self._zeros_state(b, self.encoder_cells[0].hidden_dim, h, w, device, dtype),
            self._zeros_state(b, self.encoder_cells[1].hidden_dim, h, w, device, dtype),
        ]

        for t in range(t_in):
            z = x[:, t]
            for l, cell in enumerate(self.encoder_cells):
                h_l, c_l = cell(z, enc_states[l])
                enc_states[l] = (h_l, c_l)
                z = h_l

        dec_states = [enc_states[0], enc_states[1]]
        dec_input = torch.zeros(b, 1, h, w, device=device, dtype=dtype)
        outputs = []

        for _ in range(t_out):
            z = dec_input
            for l, cell in enumerate(self.decoder_cells):
                h_l, c_l = cell(z, dec_states[l])
                dec_states[l] = (h_l, c_l)
                z = h_l
            pred = self.head(self.dropout(z))
            outputs.append(pred)

        return torch.stack(outputs, dim=1)


from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseForecaster(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, pred_len: int | None = None) -> torch.Tensor:
        """Forward.

        Args:
            x: [B, Tin, Cin, H, W]
            pred_len: number of output steps. If None, use model default.

        Returns:
            y_hat: [B, Tout, Cout, H, W]
        """
        raise NotImplementedError


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Union

import torch
from torch import nn


class BaseForecaster(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, x: torch.Tensor, pred_len: int | None = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward.

        Args:
            x: [B, Tin, Cin, H, W]
            pred_len: number of output steps. If None, use model default.

        Returns:
            y_hat: [B, Tout, Cout, H, W]
            or dual-head dict:
                field: [B, Tout, Cout, H, W]
                front_mask_logits: [B, Tout, 1, H, W]
        """
        raise NotImplementedError


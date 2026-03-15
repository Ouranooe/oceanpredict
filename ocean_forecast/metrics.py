from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class MaskedChannelRMSE:
    n_channels: int

    def __post_init__(self):
        self.sum_sq = torch.zeros(self.n_channels, dtype=torch.float64)
        self.count = torch.zeros(self.n_channels, dtype=torch.float64)

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        """
        pred/target: [B, T, C, H, W] in raw units
        mask: [B, H, W] or [H, W]
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = mask.to(dtype=pred.dtype, device=pred.device)
        valid = mask.unsqueeze(1).unsqueeze(2).expand(-1, pred.size(1), pred.size(2), -1, -1)

        err2 = torch.square(pred - target) * valid
        self.sum_sq += err2.sum(dim=(0, 1, 3, 4)).detach().cpu().to(torch.float64)
        self.count += valid.sum(dim=(0, 1, 3, 4)).detach().cpu().to(torch.float64)

    def compute_rmse(self) -> np.ndarray:
        rmse = torch.sqrt(self.sum_sq / torch.clamp(self.count, min=1.0))
        return rmse.numpy().astype(np.float64)


def rmse_to_nrmse(rmse: np.ndarray, nrmse_denom: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.asarray(nrmse_denom, dtype=np.float64), 1e-12)
    return np.asarray(rmse, dtype=np.float64) / denom


def summarize_channel_metrics(rmse: np.ndarray, nrmse: np.ndarray) -> Dict[str, float]:
    return {
        "rmse_sst": float(rmse[0]),
        "rmse_sss": float(rmse[1]),
        "rmse_speed": float(rmse[2]),
        "nrmse_sst": float(nrmse[0]),
        "nrmse_sss": float(nrmse[1]),
        "nrmse_speed": float(nrmse[2]),
        "nrmse_mean": float(np.mean(nrmse)),
    }


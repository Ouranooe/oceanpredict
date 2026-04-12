from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

EVAL_CHANNEL_NAMES = ("sst", "sss", "speed")


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


def to_eval_channels_torch(raw: torch.Tensor) -> torch.Tensor:
    """
    Convert raw prediction/target channels to evaluation channels [sst, sss, speed].
    Supports raw channels:
      - C=3: [sst, sss, speed] (legacy)
      - C=4: [sst, sss, ssu, ssv] (uv mode)
    """
    if raw.dim() != 5:
        raise ValueError(f"raw must be [B,T,C,H,W], got {tuple(raw.shape)}")
    c = int(raw.size(2))
    if c == 3:
        return raw
    if c == 4:
        speed = torch.sqrt(torch.square(raw[:, :, 2]) + torch.square(raw[:, :, 3]))
        return torch.stack([raw[:, :, 0], raw[:, :, 1], speed], dim=2)
    raise ValueError(f"Unsupported raw channel count for eval conversion: C={c}. Expected 3 or 4.")


def to_eval_channels_numpy(raw: np.ndarray) -> np.ndarray:
    """
    Convert raw prediction/target channels to evaluation channels [sst, sss, speed].
    Supports shapes:
      - [T,C,H,W]
      - [B,T,C,H,W]
    Supports raw channels:
      - C=3: [sst, sss, speed] (legacy)
      - C=4: [sst, sss, ssu, ssv] (uv mode)
    """
    if raw.ndim == 4:
        c = int(raw.shape[1])
        if c == 3:
            return raw
        if c == 4:
            speed = np.sqrt(np.square(raw[:, 2]) + np.square(raw[:, 3])).astype(raw.dtype, copy=False)
            return np.stack([raw[:, 0], raw[:, 1], speed], axis=1).astype(raw.dtype, copy=False)
        raise ValueError(f"Unsupported raw channel count for eval conversion: C={c}. Expected 3 or 4.")
    if raw.ndim == 5:
        c = int(raw.shape[2])
        if c == 3:
            return raw
        if c == 4:
            speed = np.sqrt(np.square(raw[:, :, 2]) + np.square(raw[:, :, 3])).astype(raw.dtype, copy=False)
            return np.stack([raw[:, :, 0], raw[:, :, 1], speed], axis=2).astype(raw.dtype, copy=False)
        raise ValueError(f"Unsupported raw channel count for eval conversion: C={c}. Expected 3 or 4.")
    raise ValueError(f"raw must be [T,C,H,W] or [B,T,C,H,W], got {tuple(raw.shape)}")


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


def _as_front_5d(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.dim() == 4:
        t = t.unsqueeze(2)
    if t.dim() != 5:
        raise ValueError(f"{name} must be [B,T,1,H,W] or [B,T,H,W], got {tuple(t.shape)}")
    if t.size(2) != 1:
        raise ValueError(f"{name} channel must be 1, got C={int(t.size(2))}.")
    return t


def masked_front_iou_from_logits(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    ocean_mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}.")
    logits = _as_front_5d(logits, "logits")
    target_mask = _as_front_5d(target_mask, "target_mask")
    if logits.shape != target_mask.shape:
        raise ValueError(
            f"logits and target_mask shape mismatch: {tuple(logits.shape)} vs {tuple(target_mask.shape)}"
        )

    if ocean_mask.dim() == 2:
        ocean_mask = ocean_mask.unsqueeze(0)
    if ocean_mask.dim() != 3:
        raise ValueError(f"ocean_mask must be [H,W] or [B,H,W], got {tuple(ocean_mask.shape)}")

    valid = ocean_mask.to(device=logits.device, dtype=torch.bool).unsqueeze(1).unsqueeze(2)
    valid = valid.expand(-1, logits.size(1), 1, -1, -1)
    pred = torch.sigmoid(logits) >= float(threshold)
    target = target_mask >= 0.5
    inter = torch.sum(pred & target & valid)
    union = torch.sum((pred | target) & valid)
    if int(union.item()) == 0:
        return 1.0
    return float((inter.to(torch.float32) / union.to(torch.float32)).item())


def masked_front_acc_from_logits(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    ocean_mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    if not (0.0 <= float(threshold) <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}.")
    logits = _as_front_5d(logits, "logits")
    target_mask = _as_front_5d(target_mask, "target_mask")
    if logits.shape != target_mask.shape:
        raise ValueError(
            f"logits and target_mask shape mismatch: {tuple(logits.shape)} vs {tuple(target_mask.shape)}"
        )

    if ocean_mask.dim() == 2:
        ocean_mask = ocean_mask.unsqueeze(0)
    if ocean_mask.dim() != 3:
        raise ValueError(f"ocean_mask must be [H,W] or [B,H,W], got {tuple(ocean_mask.shape)}")

    valid = ocean_mask.to(device=logits.device, dtype=torch.bool).unsqueeze(1).unsqueeze(2)
    valid = valid.expand(-1, logits.size(1), 1, -1, -1)
    pred = torch.sigmoid(logits) >= float(threshold)
    target = target_mask >= 0.5
    correct = torch.sum((pred == target) & valid)
    total = torch.sum(valid)
    if int(total.item()) == 0:
        return 1.0
    return float((correct.to(torch.float32) / total.to(torch.float32)).item())

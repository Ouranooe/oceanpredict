from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
import torch

HOURS_PER_DAY = 24.0
HOURS_PER_YEAR = 24.0 * 366.0
BASE_INPUT_CHANNELS = 4


def parse_input_feature_config(data_cfg: Dict[str, Any]) -> Dict[str, bool]:
    raw = data_cfg.get("input_features", {}) or {}
    return {
        "add_mask": bool(raw.get("add_mask", False)),
        "add_time_hour": bool(raw.get("add_time_hour", False)),
        "add_time_year": bool(raw.get("add_time_year", False)),
    }


def compute_model_input_channels(base_channels: int, feature_cfg: Dict[str, bool]) -> int:
    extra = 0
    if feature_cfg.get("add_mask", False):
        extra += 1
    if feature_cfg.get("add_time_hour", False):
        extra += 2
    if feature_cfg.get("add_time_year", False):
        extra += 2
    return int(base_channels) + extra


def augment_input_tensor(
    x: torch.Tensor,
    t_in: torch.Tensor,
    mask: torch.Tensor,
    feature_cfg: Dict[str, bool],
) -> torch.Tensor:
    if x.dim() != 5:
        raise ValueError(f"x must be [B,T,C,H,W], got {tuple(x.shape)}")
    if t_in.dim() != 2:
        raise ValueError(f"t_in must be [B,T], got {tuple(t_in.shape)}")
    if t_in.shape[0] != x.shape[0] or t_in.shape[1] != x.shape[1]:
        raise ValueError(f"t_in shape {tuple(t_in.shape)} not compatible with x shape {tuple(x.shape)}")

    b, t, _, h, w = x.shape
    features = [x]

    if feature_cfg.get("add_mask", False):
        if mask.dim() == 2:
            mask_bt = mask.unsqueeze(0).expand(b, -1, -1)
        elif mask.dim() == 3:
            mask_bt = mask
        else:
            raise ValueError(f"mask must be [H,W] or [B,H,W], got {tuple(mask.shape)}")
        mask_5d = mask_bt.to(dtype=x.dtype).unsqueeze(1).unsqueeze(2).expand(-1, t, -1, -1, -1)
        features.append(mask_5d)

    t_f = t_in.to(dtype=x.dtype)
    if feature_cfg.get("add_time_hour", False):
        hour_phase = (2.0 * math.pi) * (torch.remainder(t_f, HOURS_PER_DAY) / HOURS_PER_DAY)
        hour_sin = torch.sin(hour_phase).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, h, w)
        hour_cos = torch.cos(hour_phase).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, h, w)
        features.extend([hour_sin, hour_cos])

    if feature_cfg.get("add_time_year", False):
        year_phase = (2.0 * math.pi) * (torch.remainder(t_f, HOURS_PER_YEAR) / HOURS_PER_YEAR)
        year_sin = torch.sin(year_phase).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, h, w)
        year_cos = torch.cos(year_phase).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, h, w)
        features.extend([year_sin, year_cos])

    return torch.cat(features, dim=2)


def augment_input_array(
    x: np.ndarray,
    t_in: np.ndarray,
    mask: np.ndarray,
    feature_cfg: Dict[str, bool],
) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"x must be [T,C,H,W], got {x.shape}")
    if t_in.ndim != 1 or t_in.shape[0] != x.shape[0]:
        raise ValueError(f"t_in must be [T] and match x time dim, got {t_in.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be [H,W], got {mask.shape}")

    t, _, h, w = x.shape
    features = [x.astype(np.float32)]

    if feature_cfg.get("add_mask", False):
        mask_ch = mask.astype(np.float32)[None, None, :, :]
        features.append(np.repeat(mask_ch, repeats=t, axis=0))

    t_f = t_in.astype(np.float32)
    if feature_cfg.get("add_time_hour", False):
        hour_phase = (2.0 * np.pi) * (np.mod(t_f, HOURS_PER_DAY) / HOURS_PER_DAY)
        hour_sin = np.sin(hour_phase).astype(np.float32)[:, None, None, None]
        hour_cos = np.cos(hour_phase).astype(np.float32)[:, None, None, None]
        features.append(np.broadcast_to(hour_sin, (t, 1, h, w)))
        features.append(np.broadcast_to(hour_cos, (t, 1, h, w)))

    if feature_cfg.get("add_time_year", False):
        year_phase = (2.0 * np.pi) * (np.mod(t_f, HOURS_PER_YEAR) / HOURS_PER_YEAR)
        year_sin = np.sin(year_phase).astype(np.float32)[:, None, None, None]
        year_cos = np.cos(year_phase).astype(np.float32)[:, None, None, None]
        features.append(np.broadcast_to(year_sin, (t, 1, h, w)))
        features.append(np.broadcast_to(year_cos, (t, 1, h, w)))

    return np.concatenate(features, axis=1).astype(np.float32)

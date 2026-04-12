from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE over [B, T, C, H, W] with mask [B, H, W] or [H, W]."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError(f"mask must be [H,W] or [B,H,W], got {tuple(mask.shape)}")

    mask_5d = mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,H,W]
    mask_5d = mask_5d.expand(-1, pred.size(1), pred.size(2), -1, -1)

    sq_err = torch.square(pred - target) * mask_5d
    denom = torch.clamp(mask_5d.sum(), min=1.0)
    return sq_err.sum() / denom


def masked_front_bce_with_logits_loss(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    ocean_mask: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Masked BCEWithLogits for front-like segmentation.

    Args:
        logits: [B,T,1,H,W] or [B,T,H,W]
        target_mask: same spatial/time shape as logits, values in {0,1}
        ocean_mask: [H,W] or [B,H,W]
        pos_weight: positive class weight for BCEWithLogits
    """
    if logits.dim() == 4:
        logits = logits.unsqueeze(2)
    if target_mask.dim() == 4:
        target_mask = target_mask.unsqueeze(2)
    if logits.dim() != 5 or target_mask.dim() != 5:
        raise ValueError(
            f"logits/target_mask must be [B,T,1,H,W] or [B,T,H,W], got {tuple(logits.shape)} / {tuple(target_mask.shape)}"
        )
    if logits.shape != target_mask.shape:
        raise ValueError(
            f"logits and target_mask shape mismatch: {tuple(logits.shape)} vs {tuple(target_mask.shape)}"
        )
    if logits.size(2) != 1:
        raise ValueError(f"front logits channel must be 1, got C={int(logits.size(2))}.")
    if float(pos_weight) <= 0:
        raise ValueError(f"pos_weight must be > 0, got {pos_weight}.")

    if ocean_mask.dim() == 2:
        ocean_mask = ocean_mask.unsqueeze(0)
    if ocean_mask.dim() != 3:
        raise ValueError(f"ocean_mask must be [H,W] or [B,H,W], got {tuple(ocean_mask.shape)}")

    valid = ocean_mask.to(dtype=logits.dtype, device=logits.device).unsqueeze(1).unsqueeze(2)
    valid = valid.expand(-1, logits.size(1), 1, -1, -1)
    target = target_mask.to(dtype=logits.dtype, device=logits.device).clamp(0.0, 1.0)

    pw = torch.tensor([float(pos_weight)], dtype=logits.dtype, device=logits.device)
    loss_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pw)
    denom = torch.clamp(valid.sum(), min=1.0)
    return (loss_map * valid).sum() / denom


def masked_physics_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    spatial_weight: float = 1.0,
    temporal_weight: float = 1.0,
) -> torch.Tensor:
    """Legacy smoothness regularization on [B, T, C, H, W]."""
    if pred.dim() != 5:
        raise ValueError(f"pred must be [B,T,C,H,W], got {tuple(pred.shape)}")
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError(f"mask must be [H,W] or [B,H,W], got {tuple(mask.shape)}")

    mask_5d = mask.unsqueeze(1).unsqueeze(2)
    mask_5d = mask_5d.expand(-1, pred.size(1), pred.size(2), -1, -1)

    grad_x = pred[..., :, 1:] - pred[..., :, :-1]
    mask_x = mask_5d[..., :, 1:] * mask_5d[..., :, :-1]
    loss_x = (torch.square(grad_x) * mask_x).sum() / torch.clamp(mask_x.sum(), min=1.0)

    grad_y = pred[..., 1:, :] - pred[..., :-1, :]
    mask_y = mask_5d[..., 1:, :] * mask_5d[..., :-1, :]
    loss_y = (torch.square(grad_y) * mask_y).sum() / torch.clamp(mask_y.sum(), min=1.0)

    if pred.size(1) > 1:
        grad_t = pred[:, 1:, ...] - pred[:, :-1, ...]
        mask_t = mask_5d[:, 1:, ...] * mask_5d[:, :-1, ...]
        loss_t = (torch.square(grad_t) * mask_t).sum() / torch.clamp(mask_t.sum(), min=1.0)
    else:
        loss_t = pred.new_zeros(())

    spatial = 0.5 * (loss_x + loss_y)
    return float(spatial_weight) * spatial + float(temporal_weight) * loss_t


def masked_smoothness_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    spatial_weight: float = 1.0,
    temporal_weight: float = 1.0,
) -> torch.Tensor:
    """Recommended name for the legacy masked_physics_loss smoothness regularizer."""
    return masked_physics_loss(
        pred=pred,
        mask=mask,
        spatial_weight=spatial_weight,
        temporal_weight=temporal_weight,
    )


def _as_mask_5d(mask: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError(f"mask must be [H,W] or [B,H,W], got {tuple(mask.shape)}")
    return mask.unsqueeze(1).unsqueeze(2).expand(-1, pred.size(1), pred.size(2), -1, -1)


def masked_density_consistency_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    target_mean: torch.Tensor | Iterable[float],
    target_std: torch.Tensor | Iterable[float],
    rho0: float = 1027.0,
    alpha: float = 2.0e-4,
    beta: float = 7.5e-4,
    t0: float = 0.0,
    s0: float = 35.0,
    sst_channel: int = 0,
    sss_channel: int = 1,
) -> torch.Tensor:
    """Density temporal consistency from denormalized SST/SSS predictions."""
    if pred.dim() != 5:
        raise ValueError(f"pred must be [B,T,C,H,W], got {tuple(pred.shape)}")
    if pred.size(1) < 2:
        return pred.new_zeros(())
    if sst_channel < 0 or sst_channel >= pred.size(2):
        raise ValueError(f"Invalid sst_channel={sst_channel} for pred C={pred.size(2)}")
    if sss_channel < 0 or sss_channel >= pred.size(2):
        raise ValueError(f"Invalid sss_channel={sss_channel} for pred C={pred.size(2)}")

    mean_t = torch.as_tensor(target_mean, dtype=pred.dtype, device=pred.device).view(1, 1, -1, 1, 1)
    std_t = torch.as_tensor(target_std, dtype=pred.dtype, device=pred.device).view(1, 1, -1, 1, 1)
    pred_raw = pred * std_t + mean_t
    sst = pred_raw[:, :, sst_channel : sst_channel + 1, :, :]
    sss = pred_raw[:, :, sss_channel : sss_channel + 1, :, :]

    density = float(rho0) * (
        1.0
        - float(alpha) * (sst - float(t0))
        + float(beta) * (sss - float(s0))
    )

    d_rho = density[:, 1:, ...] - density[:, :-1, ...]
    mask_5d = _as_mask_5d(mask, pred)[:, :, :1, :, :]
    mask_t = mask_5d[:, 1:, ...] * mask_5d[:, :-1, ...]
    denom = torch.clamp(mask_t.sum(), min=1.0)
    return (torch.square(d_rho) * mask_t).sum() / denom


def _channel_weights(
    pred: torch.Tensor,
    channels: Iterable[int] | None,
    default_channels: Iterable[int],
    speed_weight: float = 0.0,
    apply_speed_weight: bool = False,
) -> torch.Tensor:
    c = int(pred.size(2))
    weights = pred.new_zeros((c,))
    selected = list(default_channels if channels is None else channels)
    for idx in selected:
        i = int(idx)
        if i < 0 or i >= c:
            continue
        weights[i] = 1.0
    if apply_speed_weight and c > 2 and float(speed_weight) > 0:
        weights[2] = float(speed_weight)
    return weights


def masked_temporal_diffdiv_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    channels: Iterable[int] | None = (0, 1),
    speed_weight: float = 0.0,
) -> torch.Tensor:
    """Temporal TAU-style diff-div regularization via second-order time differences."""
    if pred.dim() != 5:
        raise ValueError(f"pred must be [B,T,C,H,W], got {tuple(pred.shape)}")
    if pred.size(1) < 3:
        return pred.new_zeros(())

    mask_5d = _as_mask_5d(mask, pred)
    d2_t = pred[:, 2:, ...] - 2.0 * pred[:, 1:-1, ...] + pred[:, :-2, ...]
    mask_t = mask_5d[:, 2:, ...] * mask_5d[:, 1:-1, ...] * mask_5d[:, :-2, ...]

    ch_w = _channel_weights(
        pred=pred,
        channels=channels,
        default_channels=(0, 1),
        speed_weight=speed_weight,
        apply_speed_weight=True,
    ).view(1, 1, -1, 1, 1)
    weighted_mask = mask_t * ch_w
    denom = torch.clamp(weighted_mask.sum(), min=1.0)
    return (torch.square(d2_t) * weighted_mask).sum() / denom


def masked_laplacian_smoothness_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    channels: Iterable[int] | None = None,
) -> torch.Tensor:
    """Spatial Laplacian smoothness loss; independent from temporal diff-div."""
    if pred.dim() != 5:
        raise ValueError(f"pred must be [B,T,C,H,W], got {tuple(pred.shape)}")
    if pred.size(-2) < 3 or pred.size(-1) < 3:
        return pred.new_zeros(())

    mask_5d = _as_mask_5d(mask, pred)
    center = pred[..., 1:-1, 1:-1]
    up = pred[..., :-2, 1:-1]
    down = pred[..., 2:, 1:-1]
    left = pred[..., 1:-1, :-2]
    right = pred[..., 1:-1, 2:]
    lap = up + down + left + right - 4.0 * center

    m_center = mask_5d[..., 1:-1, 1:-1]
    m_up = mask_5d[..., :-2, 1:-1]
    m_down = mask_5d[..., 2:, 1:-1]
    m_left = mask_5d[..., 1:-1, :-2]
    m_right = mask_5d[..., 1:-1, 2:]
    mask_lap = m_center * m_up * m_down * m_left * m_right

    if channels is None:
        ch_w = pred.new_ones((pred.size(2),))
    else:
        ch_w = _channel_weights(
            pred=pred,
            channels=channels,
            default_channels=(),
            speed_weight=0.0,
            apply_speed_weight=False,
        )
    ch_w = ch_w.view(1, 1, -1, 1, 1)
    weighted_mask = mask_lap * ch_w
    denom = torch.clamp(weighted_mask.sum(), min=1.0)
    return (torch.square(lap) * weighted_mask).sum() / denom

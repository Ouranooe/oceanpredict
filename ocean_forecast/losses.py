from __future__ import annotations

import torch


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


def masked_physics_loss(
    pred: torch.Tensor,
    mask: torch.Tensor,
    spatial_weight: float = 1.0,
    temporal_weight: float = 1.0,
) -> torch.Tensor:
    """Physics-inspired regularization on [B, T, C, H, W]."""
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

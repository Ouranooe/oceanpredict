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


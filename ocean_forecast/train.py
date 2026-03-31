from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import load_config, save_config
from .data.dataset import (
    PreparedSeqDataset,
    build_window_starts,
    compute_normalization_stats,
    save_stats,
    split_refs,
)
from .data.npy_reader import NpyYearReader
from .data.prepared import PreparedReader
from .data.zip_reader import FrameRef, hour_index_to_datetime
from .input_features import (
    BASE_INPUT_CHANNELS,
    augment_input_tensor,
    compute_model_input_channels,
    parse_input_feature_config,
)
from .losses import (
    masked_density_consistency_loss,
    masked_mse_loss,
    masked_smoothness_loss,
)
from .metrics import (
    MaskedChannelRMSE,
    rmse_to_nrmse,
    summarize_channel_metrics,
    to_eval_channels_torch,
)
from .models.registry import build_model
from .utils import resolve_device, set_seed

def _log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Yellow-Bohai Sea ConvLSTM baseline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output_dir", default=None, help="Override output directory.")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Debug: cap train batches per epoch.")
    parser.add_argument("--max_eval_batches", type=int, default=None, help="Debug: cap eval batches.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience.")
    return parser.parse_args()


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    nrmse_denom: np.ndarray,
    input_feature_cfg: Dict[str, bool],
    speed_bucket_edges: np.ndarray | None = None,
    max_batches: int | None = None,
) -> Dict[str, float]:
    metrics, _ = evaluate_with_profile(
        model=model,
        loader=loader,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        nrmse_denom=nrmse_denom,
        input_feature_cfg=input_feature_cfg,
        speed_bucket_edges=speed_bucket_edges,
        max_batches=max_batches,
    )
    return metrics


def evaluate_with_profile(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    nrmse_denom: np.ndarray,
    input_feature_cfg: Dict[str, bool],
    speed_bucket_edges: np.ndarray | None = None,
    max_batches: int | None = None,
) -> tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    meter = MaskedChannelRMSE(n_channels=3)
    total_loss = 0.0
    n_batches = 0
    n_samples = 0

    mean_t = torch.tensor(target_mean, dtype=torch.float32, device=device).view(1, 1, -1, 1, 1)
    std_t = torch.tensor(target_std, dtype=torch.float32, device=device).view(1, 1, -1, 1, 1)
    bucket_edges = None
    bucket_sum_sq = None
    bucket_count = None
    bucket_samples = None
    if speed_bucket_edges is not None:
        bucket_edges = _validate_speed_bucket_edges(
            speed_bucket_edges,
            cfg_key="evaluate.speed_bucket_edges",
        ).astype(np.float64)
        n_bucket = int(bucket_edges.size) + 1
        bucket_sum_sq = np.zeros((n_bucket,), dtype=np.float64)
        bucket_count = np.zeros((n_bucket,), dtype=np.float64)
        bucket_samples = np.zeros((n_bucket,), dtype=np.float64)

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    eval_start = time.perf_counter()
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            batch = _to_device(batch, device)
            x_in = augment_input_tensor(
                x=batch["x"],
                t_in=batch["t_in"],
                mask=batch["mask"],
                feature_cfg=input_feature_cfg,
            )
            pred_norm = model(x_in)
            loss = masked_mse_loss(pred_norm, batch["y"], batch["mask"])
            total_loss += float(loss.item())
            n_batches += 1
            n_samples += int(batch["y"].shape[0])

            pred_raw = pred_norm * std_t + mean_t
            y_raw = batch["y"] * std_t + mean_t
            pred_eval = to_eval_channels_torch(pred_raw)
            y_eval = to_eval_channels_torch(y_raw)
            meter.update(pred_eval, y_eval, batch["mask"])
            if bucket_edges is not None and bucket_sum_sq is not None and bucket_count is not None and bucket_samples is not None:
                speed_gt_means = _compute_batch_speed_gt_means(y_eval=y_eval, mask=batch["mask"])
                speed_bucket_idx = _assign_speed_bins(speed_gt=speed_gt_means, bin_edges=bucket_edges)
                batch_bucket_counts = np.bincount(speed_bucket_idx, minlength=bucket_sum_sq.size)
                bucket_samples += batch_bucket_counts.astype(np.float64, copy=False)

                mask = batch["mask"]
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                speed_err2 = torch.square(pred_eval[:, :, 2, :, :] - y_eval[:, :, 2, :, :])
                valid = mask.to(dtype=speed_err2.dtype, device=speed_err2.device).unsqueeze(1).expand_as(speed_err2)
                for bucket_i in range(bucket_sum_sq.size):
                    sample_sel_np = speed_bucket_idx == bucket_i
                    if not np.any(sample_sel_np):
                        continue
                    sample_sel = torch.as_tensor(sample_sel_np, device=speed_err2.device, dtype=speed_err2.dtype)
                    sample_sel = sample_sel.view(-1, 1, 1, 1)
                    valid_bucket = valid * sample_sel
                    bucket_sum_sq[bucket_i] += float((speed_err2 * valid_bucket).sum().item())
                    bucket_count[bucket_i] += float(valid_bucket.sum().item())
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    total_eval_seconds = time.perf_counter() - eval_start

    rmse = meter.compute_rmse()
    nrmse = rmse_to_nrmse(rmse, nrmse_denom)
    metrics = summarize_channel_metrics(rmse, nrmse)
    metrics["loss"] = total_loss / max(n_batches, 1)
    if bucket_edges is not None and bucket_sum_sq is not None and bucket_count is not None and bucket_samples is not None:
        _append_speed_bucket_metrics(
            metrics=metrics,
            bucket_sum_sq=bucket_sum_sq,
            bucket_count=bucket_count,
            bucket_samples=bucket_samples,
            nrmse_denom=nrmse_denom,
        )
    avg_batch_seconds = total_eval_seconds / n_batches if n_batches > 0 else 0.0
    avg_sample_seconds = total_eval_seconds / n_samples if n_samples > 0 else 0.0
    samples_per_second = n_samples / total_eval_seconds if total_eval_seconds > 0 and n_samples > 0 else 0.0
    profile = {
        "total_eval_seconds": float(total_eval_seconds),
        "num_eval_batches": int(n_batches),
        "num_eval_samples": int(n_samples),
        "avg_batch_seconds": float(avg_batch_seconds),
        "avg_sample_seconds": float(avg_sample_seconds),
        "samples_per_second": float(samples_per_second),
        "windows_per_second": float(samples_per_second),
    }
    return metrics, profile


def _build_model_kwargs(
    model_cfg: Dict,
    pred_len: int,
    input_channels: int,
    output_channels: int,
) -> Dict:
    kwargs = {k: v for k, v in model_cfg.items() if k != "name"}
    kwargs["input_channels"] = int(input_channels)
    kwargs["output_channels"] = int(output_channels)
    kwargs["default_pred_len"] = int(pred_len)
    return kwargs


def _param_groups_lrs(optimizer: torch.optim.Optimizer) -> List[float]:
    return [float(pg["lr"]) for pg in optimizer.param_groups]


def _set_param_groups_lrs(optimizer: torch.optim.Optimizer, lrs: List[float]) -> None:
    if len(lrs) != len(optimizer.param_groups):
        raise ValueError("Length mismatch between lrs and optimizer param groups.")
    for pg, lr in zip(optimizer.param_groups, lrs):
        pg["lr"] = float(lr)


def _decay_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    factor: float,
    min_lr: float,
) -> tuple[List[float], List[float]]:
    old_lrs = _param_groups_lrs(optimizer)
    new_lrs = [max(float(min_lr), float(lr) * float(factor)) for lr in old_lrs]
    _set_param_groups_lrs(optimizer, new_lrs)
    return old_lrs, new_lrs


def _all_grads_finite(params: List[torch.nn.Parameter]) -> bool:
    for p in params:
        if p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return False
    return True


def _compute_grad_norm(params: List[torch.nn.Parameter], norm_type: float = 2.0) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    norm_t = torch.norm(
        torch.stack([torch.norm(g.detach(), p=float(norm_type)) for g in grads]),
        p=float(norm_type),
    )
    return float(norm_t.item())


def _get_monitor_value(dev_metrics: Dict[str, float], monitor: str) -> float:
    metric_key = monitor[4:] if monitor.startswith("dev_") else monitor
    if metric_key not in dev_metrics:
        raise KeyError(
            f"stability.scheduler.monitor='{monitor}' not found in dev metrics keys={list(dev_metrics.keys())}"
        )
    return float(dev_metrics[metric_key])


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float, update_after_step: int = 0):
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.num_updates = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    def update(self, model: torch.nn.Module, step: int) -> None:
        if int(step) < self.update_after_step:
            return
        self.num_updates += 1
        decay = self.decay
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad or name not in self.shadow:
                    continue
                self.shadow[name].mul_(decay).add_(p.detach(), alpha=(1.0 - decay))

    def apply_to(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad or name not in self.shadow:
                    continue
                backup[name] = p.detach().clone()
                p.copy_(self.shadow[name])
        return backup

    def restore(self, model: torch.nn.Module, backup: Dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in backup:
                    p.copy_(backup[name])

    def state_dict(self) -> Dict[str, object]:
        return {
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "num_updates": self.num_updates,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.decay = float(state["decay"])
        self.update_after_step = int(state["update_after_step"])
        self.num_updates = int(state["num_updates"])
        raw_shadow = state["shadow"]
        assert isinstance(raw_shadow, dict)
        self.shadow = {str(k): v.clone() for k, v in raw_shadow.items()}


def _load_global_refs(base_root: Path) -> List[FrameRef]:
    refs_path = base_root / "refs_global.npz"
    if not refs_path.exists():
        raise FileNotFoundError(
            f"Missing base refs file: {refs_path}. Run scripts/prepare_base_dataset.py first."
        )
    raw = np.load(refs_path)
    hour_index = np.asarray(raw["hour_index"], dtype=np.int64)
    years = np.asarray(raw["year"], dtype=np.int64)
    time_idx = np.asarray(raw["time_idx"], dtype=np.int64)

    refs: List[FrameRef] = []
    for h, y, t in zip(hour_index, years, time_idx):
        year_i = int(y)
        refs.append(
            FrameRef(
                timestamp=hour_index_to_datetime(int(h)),
                hour_index=int(h),
                zip_path=str(base_root / f"{year_i}_data.npy"),
                member_name=str(year_i),
                time_idx=int(t),
            )
        )
    refs.sort(key=lambda r: r.hour_index)
    return refs


def _artifact_hash(data_cfg: Dict) -> str:
    payload = {
        "version": "exp_artifact_v2_uv_target",
        "train_start": str(data_cfg["train_start"]),
        "train_main_end": str(data_cfg["train_main_end"]),
        "dev_start": str(data_cfg["dev_start"]),
        "dev_end": str(data_cfg["dev_end"]),
        "test_start": str(data_cfg["test_start"]),
        "test_end": str(data_cfg["test_end"]),
        "input_len": int(data_cfg["input_len"]),
        "pred_len": int(data_cfg["pred_len"]),
        "stride": int(data_cfg.get("stride", 1)),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _build_ref_index_map(refs: List[FrameRef]) -> Dict[int, int]:
    return {int(r.hour_index): idx for idx, r in enumerate(refs)}


def _refs_to_indices(refs: List[FrameRef], ref_map: Dict[int, int]) -> np.ndarray:
    return np.asarray([ref_map[int(r.hour_index)] for r in refs], dtype=np.int64)


def _validate_speed_bucket_edges(
    bin_edges_raw: Any,
    cfg_key: str = "train.speed_rebalance.bin_edges",
) -> np.ndarray:
    bin_edges = np.asarray(bin_edges_raw, dtype=np.float64)
    if bin_edges.ndim != 1 or bin_edges.size == 0:
        raise ValueError(f"{cfg_key} must be a non-empty 1D array.")
    if not np.isfinite(bin_edges).all():
        raise ValueError(f"{cfg_key} must contain finite values.")
    if not np.all(np.diff(bin_edges) > 0):
        raise ValueError(f"{cfg_key} must be strictly increasing.")
    return bin_edges.astype(np.float32)


def _validate_speed_rebalance_config(speed_rebalance_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, str]:
    stat = str(speed_rebalance_cfg.get("stat", "future_ocean_mean"))
    if stat != "future_ocean_mean":
        raise ValueError(
            f"Unsupported train.speed_rebalance.stat='{stat}'. "
            "Only 'future_ocean_mean' is supported."
        )

    bin_edges = _validate_speed_bucket_edges(
        speed_rebalance_cfg.get("bin_edges", [0.1, 0.2, 0.4]),
        cfg_key="train.speed_rebalance.bin_edges",
    )

    bin_weights = np.asarray(
        speed_rebalance_cfg.get("bin_weights", [1.0, 1.5, 2.5, 4.0]),
        dtype=np.float64,
    )
    expected_weights = int(bin_edges.size) + 1
    if bin_weights.ndim != 1 or int(bin_weights.size) != expected_weights:
        raise ValueError(
            "train.speed_rebalance.bin_weights size mismatch: "
            f"expected {expected_weights}, got {int(bin_weights.size)}."
        )
    if not np.isfinite(bin_weights).all():
        raise ValueError("train.speed_rebalance.bin_weights must contain finite values.")
    if np.any(bin_weights <= 0):
        raise ValueError("train.speed_rebalance.bin_weights must be > 0.")

    return bin_edges, bin_weights.astype(np.float32), stat


def _assign_speed_bins(speed_gt: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    speed_arr = np.asarray(speed_gt, dtype=np.float64)
    edges_arr = np.asarray(bin_edges, dtype=np.float64)
    return np.digitize(speed_arr, bins=edges_arr, right=False).astype(np.int64)


def _compute_ref_ocean_mean_speed(reader: Any, refs: List[FrameRef]) -> np.ndarray:
    ref_speed = np.zeros((len(refs),), dtype=np.float32)
    for idx, ref in enumerate(refs):
        frame, mask = reader.read_frame(ref)
        if int(frame.shape[0]) < 4:
            raise ValueError(f"Expected frame with >=4 channels for speed calculation, got shape={frame.shape}.")
        speed = np.sqrt(np.square(frame[2]) + np.square(frame[3]))
        valid = mask & np.isfinite(speed)
        if np.any(valid):
            ref_speed[idx] = float(np.mean(speed[valid], dtype=np.float64))
    return ref_speed


def _compute_future_window_mean_speed(
    ref_speed: np.ndarray,
    window_starts: np.ndarray,
    input_len: int,
    pred_len: int,
) -> np.ndarray:
    if int(pred_len) <= 0:
        raise ValueError(f"pred_len must be > 0, got {pred_len}.")
    speed_arr = np.asarray(ref_speed, dtype=np.float64)
    starts = np.asarray(window_starts, dtype=np.int64)

    future_starts = starts + int(input_len)
    future_ends = future_starts + int(pred_len)
    if future_starts.size == 0:
        return np.zeros((0,), dtype=np.float32)
    if int(np.min(future_starts)) < 0 or int(np.max(future_ends)) > int(speed_arr.size):
        raise ValueError(
            "Window index out of range while building speed rebalance weights: "
            f"future range [{int(np.min(future_starts))}, {int(np.max(future_ends))}) "
            f"for ref_speed size={int(speed_arr.size)}."
        )

    prefix = np.concatenate(([0.0], np.cumsum(speed_arr, dtype=np.float64)))
    sums = prefix[future_ends] - prefix[future_starts]
    means = sums / float(pred_len)
    return means.astype(np.float32)


def _speed_bucket_labels(bin_edges: np.ndarray) -> List[str]:
    edges = np.asarray(bin_edges, dtype=np.float64)
    labels = [f"<{edges[0]:g}"]
    labels.extend(f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(int(edges.size) - 1))
    labels.append(f">={edges[-1]:g}")
    return labels


def _compute_batch_speed_gt_means(
    y_eval: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    if y_eval.dim() != 5:
        raise ValueError(f"y_eval must be [B,T,C,H,W], got {tuple(y_eval.shape)}")
    if y_eval.size(2) < 3:
        raise ValueError(f"y_eval must include speed channel at C=2, got C={int(y_eval.size(2))}.")
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError(f"mask must be [H,W] or [B,H,W], got {tuple(mask.shape)}")

    speed = y_eval[:, :, 2, :, :]
    valid = mask.to(dtype=speed.dtype, device=speed.device).unsqueeze(1).expand_as(speed)
    sum_speed = (speed * valid).sum(dim=(1, 2, 3))
    valid_count = torch.clamp(valid.sum(dim=(1, 2, 3)), min=1.0)
    mean_speed = sum_speed / valid_count
    return mean_speed.detach().cpu().numpy().astype(np.float64)


def _masked_speed_aux_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    mask: torch.Tensor,
    target_mean: torch.Tensor | np.ndarray,
    target_std: torch.Tensor | np.ndarray,
    loss_type: str = "huber",
    huber_delta: float = 0.05,
) -> torch.Tensor:
    if pred_norm.dim() != 5 or target_norm.dim() != 5:
        raise ValueError(
            f"pred_norm/target_norm must be [B,T,C,H,W], got {tuple(pred_norm.shape)} and {tuple(target_norm.shape)}"
        )
    if pred_norm.shape != target_norm.shape:
        raise ValueError(f"pred_norm and target_norm shape mismatch: {pred_norm.shape} vs {target_norm.shape}")
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError(f"mask must be [H,W] or [B,H,W], got {tuple(mask.shape)}")

    mean_t = torch.as_tensor(target_mean, dtype=pred_norm.dtype, device=pred_norm.device).view(1, 1, -1, 1, 1)
    std_t = torch.as_tensor(target_std, dtype=pred_norm.dtype, device=pred_norm.device).view(1, 1, -1, 1, 1)
    pred_raw = pred_norm * std_t + mean_t
    target_raw = target_norm * std_t + mean_t
    channels = int(pred_raw.size(2))
    if channels == 3:
        pred_speed = pred_raw[:, :, 2, :, :]
        target_speed = target_raw[:, :, 2, :, :]
    elif channels >= 4:
        pred_speed = torch.sqrt(torch.square(pred_raw[:, :, 2, :, :]) + torch.square(pred_raw[:, :, 3, :, :]))
        target_speed = torch.sqrt(torch.square(target_raw[:, :, 2, :, :]) + torch.square(target_raw[:, :, 3, :, :]))
    else:
        raise ValueError(f"Unsupported channel count for speed aux loss: C={channels}. Expected 3 or 4+.")

    diff = pred_speed - target_speed
    loss_type_norm = str(loss_type).lower()
    if loss_type_norm == "mse":
        loss_map = torch.square(diff)
    elif loss_type_norm == "huber":
        beta = float(huber_delta)
        if beta <= 0:
            raise ValueError(f"speed_aux_loss.huber_delta must be > 0 for huber loss, got {huber_delta}.")
        loss_map = F.smooth_l1_loss(pred_speed, target_speed, reduction="none", beta=beta)
    else:
        raise ValueError(f"Unsupported speed_aux_loss.type='{loss_type}'. Expected 'huber' or 'mse'.")

    valid = mask.to(dtype=loss_map.dtype, device=loss_map.device).unsqueeze(1).expand_as(loss_map)
    denom = torch.clamp(valid.sum(), min=1.0)
    return (loss_map * valid).sum() / denom


def _append_speed_bucket_metrics(
    metrics: Dict[str, float],
    bucket_sum_sq: np.ndarray,
    bucket_count: np.ndarray,
    bucket_samples: np.ndarray,
    nrmse_denom: np.ndarray,
) -> None:
    speed_denom = float(np.maximum(np.asarray(nrmse_denom, dtype=np.float64)[2], 1e-12))
    for idx in range(int(bucket_sum_sq.size)):
        sum_sq = float(bucket_sum_sq[idx])
        cnt = float(bucket_count[idx])
        samples = float(bucket_samples[idx])
        if cnt > 0.0:
            rmse = float(np.sqrt(sum_sq / cnt))
            nrmse = float(rmse / speed_denom)
        else:
            rmse = float("nan")
            nrmse = float("nan")
        metrics[f"speed_bucket_{idx}_samples"] = samples
        metrics[f"speed_bucket_{idx}_rmse"] = rmse
        metrics[f"speed_bucket_{idx}_nrmse"] = nrmse


def _log_bucket_eval_metrics(metrics: Dict[str, float], bucket_edges: np.ndarray, prefix: str) -> None:
    labels = _speed_bucket_labels(bucket_edges)
    for idx, label in enumerate(labels):
        samples = int(round(float(metrics.get(f"speed_bucket_{idx}_samples", 0.0))))
        rmse = float(metrics.get(f"speed_bucket_{idx}_rmse", float("nan")))
        nrmse = float(metrics.get(f"speed_bucket_{idx}_nrmse", float("nan")))
        if np.isfinite(rmse) and np.isfinite(nrmse):
            _log(
                f"{prefix} speed bucket {label}: samples={samples}, "
                f"rmse_speed={rmse:.6f}, nrmse_speed={nrmse:.6f}"
            )
        else:
            _log(
                f"{prefix} speed bucket {label}: samples={samples}, "
                "rmse_speed=nan, nrmse_speed=nan"
            )


def _build_train_speed_rebalance_sampler(
    reader: Any,
    train_refs: List[FrameRef],
    train_windows: np.ndarray,
    input_len: int,
    pred_len: int,
    speed_rebalance_cfg: Dict[str, Any],
) -> tuple[WeightedRandomSampler | None, Dict[str, Any]]:
    enabled = bool(speed_rebalance_cfg.get("enabled", True))
    if not enabled:
        return None, {"enabled": False}

    bin_edges, bin_weights, stat = _validate_speed_rebalance_config(speed_rebalance_cfg)
    ref_speed = _compute_ref_ocean_mean_speed(reader=reader, refs=train_refs)
    speed_gt = _compute_future_window_mean_speed(
        ref_speed=ref_speed,
        window_starts=np.asarray(train_windows, dtype=np.int64),
        input_len=input_len,
        pred_len=pred_len,
    )
    bucket_idx = _assign_speed_bins(speed_gt=speed_gt, bin_edges=bin_edges)
    sample_weights = bin_weights[bucket_idx].astype(np.float64, copy=False)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=int(sample_weights.size),
        replacement=True,
    )
    bucket_counts = np.bincount(bucket_idx, minlength=int(bin_weights.size)).astype(np.int64, copy=False)
    return sampler, {
        "enabled": True,
        "stat": stat,
        "bin_edges": bin_edges,
        "bin_weights": bin_weights,
        "speed_gt": speed_gt,
        "bucket_idx": bucket_idx,
        "bucket_counts": bucket_counts,
    }


def _resolve_or_build_experiment_artifacts(base_root: Path, data_cfg: Dict) -> Path:
    exp_hash = _artifact_hash(data_cfg)
    exp_root = base_root / ".exp_cache" / exp_hash
    required = [
        exp_root / "manifest_exp.json",
        exp_root / "split_refs.npz",
        exp_root / "train_windows.npy",
        exp_root / "dev_windows.npy",
        exp_root / "test_windows.npy",
        exp_root / "stats.npz",
    ]
    if all(p.exists() for p in required):
        _log(f"Using cached prepared artifacts: {exp_root}")
        return exp_root

    exp_root.mkdir(parents=True, exist_ok=True)
    _log(f"Building experiment artifacts: {exp_root}")

    refs = _load_global_refs(base_root)
    if not refs:
        raise RuntimeError(f"No refs in base_root={base_root}.")
    split = split_refs(refs, data_cfg)
    input_len = int(data_cfg["input_len"])
    pred_len = int(data_cfg["pred_len"])
    stride = int(data_cfg.get("stride", 1))

    train_windows = np.asarray(
        build_window_starts(split.train_main, input_len, pred_len, stride=stride),
        dtype=np.int64,
    )
    dev_windows = np.asarray(
        build_window_starts(split.dev, input_len, pred_len, stride=stride),
        dtype=np.int64,
    )
    test_windows = np.asarray(
        build_window_starts(split.test, input_len, pred_len, stride=stride),
        dtype=np.int64,
    )
    if train_windows.size == 0:
        raise RuntimeError("Empty train windows for current config.")
    if dev_windows.size == 0:
        raise RuntimeError("Empty dev windows for current config.")
    if test_windows.size == 0:
        raise RuntimeError("Empty test windows for current config.")
    np.save(exp_root / "train_windows.npy", train_windows)
    np.save(exp_root / "dev_windows.npy", dev_windows)
    np.save(exp_root / "test_windows.npy", test_windows)

    ref_map = _build_ref_index_map(refs)
    np.savez_compressed(
        exp_root / "split_refs.npz",
        train_main_idx=_refs_to_indices(split.train_main, ref_map),
        dev_idx=_refs_to_indices(split.dev, ref_map),
        test_idx=_refs_to_indices(split.test, ref_map),
    )

    years = sorted({int(r.member_name) for r in refs})
    reader = NpyYearReader(root_dir=base_root, years=years)
    stats = compute_normalization_stats(reader, split.train_main)
    save_stats(stats, exp_root / "stats.npz")
    reader.close()

    manifest = {
        "version": 1,
        "format": "exp_index_v1",
        "base_root": str(base_root.resolve()),
        "years": years,
        "input_len": input_len,
        "pred_len": pred_len,
        "stride": stride,
        "train_start": str(data_cfg["train_start"]),
        "train_main_end": str(data_cfg["train_main_end"]),
        "dev_start": str(data_cfg["dev_start"]),
        "dev_end": str(data_cfg["dev_end"]),
        "test_start": str(data_cfg["test_start"]),
        "test_end": str(data_cfg["test_end"]),
        "n_train_refs": int(len(split.train_main)),
        "n_dev_refs": int(len(split.dev)),
        "n_test_refs": int(len(split.test)),
        "n_train_windows": int(train_windows.size),
        "n_dev_windows": int(dev_windows.size),
        "n_test_windows": int(test_windows.size),
        "exp_hash": exp_hash,
    }
    with (exp_root / "manifest_exp.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    _log(
        "Built experiment artifacts: "
        f"refs(train/dev/test)={len(split.train_main)}/{len(split.dev)}/{len(split.test)}, "
        f"windows={train_windows.size}/{dev_windows.size}/{test_windows.size}"
    )
    return exp_root


def main() -> None:
    t0_all = time.perf_counter()
    args = parse_args()
    cfg = load_config(args.config)
    _log(f"Loaded config from {args.config}")

    if args.output_dir is not None:
        cfg["train"]["output_dir"] = args.output_dir
    if args.max_train_batches is not None:
        cfg["train"]["max_train_batches"] = args.max_train_batches
    if args.max_eval_batches is not None:
        cfg["train"]["max_eval_batches"] = args.max_eval_batches
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.patience is not None:
        cfg["train"]["patience"] = args.patience

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    input_feature_cfg = parse_input_feature_config(data_cfg)
    model_input_channels = compute_model_input_channels(BASE_INPUT_CHANNELS, input_feature_cfg)

    set_seed(int(train_cfg["seed"]))
    device = resolve_device(train_cfg["device"])
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "resolved_config.yaml")
    _log(
        "Runtime setup: "
        f"device={device}, amp={use_amp}, seed={train_cfg['seed']}, output_dir={output_dir}"
    )
    _log(
        "Input features: "
        f"add_mask={input_feature_cfg['add_mask']}, "
        f"add_time_hour={input_feature_cfg['add_time_hour']}, "
        f"add_time_year={input_feature_cfg['add_time_year']}, "
        f"model_input_channels={model_input_channels}"
    )

    base_root_text = data_cfg.get("base_root")
    if not base_root_text:
        raise RuntimeError(
            "Training now requires base dataset path `data.base_root`. "
            "Run `python scripts/prepare_base_dataset.py --src_root <raw_root> --out_root <base_root>` first."
        )
    base_root = Path(str(base_root_text))
    prepared_root = _resolve_or_build_experiment_artifacts(base_root=base_root, data_cfg=data_cfg)
    reader = PreparedReader(prepared_root=prepared_root)
    manifest = reader.manifest
    split = reader.load_split_refs()
    train_windows = reader.load_windows("train")
    dev_windows = reader.load_windows("dev")
    test_windows = reader.load_windows("test")
    stats = reader.load_stats()
    target_channels = int(np.asarray(stats["target_mean"]).shape[0])
    if target_channels not in (3, 4):
        raise ValueError(
            f"Unsupported target_channels={target_channels} from stats. Expected 3 (legacy) or 4 (uv)."
        )
    target_mode = "uv" if target_channels == 4 else "legacy_speed"
    input_len = int(manifest.input_len)
    pred_len = int(manifest.pred_len)
    years_text = (
        f"{manifest.years[0]}-{manifest.years[-1]}"
        if manifest.years
        else "unknown"
    )
    _log(
        "Loaded prepared artifacts: "
        f"prepared_root={prepared_root}, years={years_text}, "
        f"input_len={input_len}, pred_len={pred_len}, stride={manifest.stride}, "
        f"target_channels={target_channels} ({target_mode})"
    )
    _log(
        "Prepared split refs/windows: "
        f"train_main={len(split.train_main)}/{len(train_windows)}, "
        f"dev={len(split.dev)}/{len(dev_windows)}, test={len(split.test)}/{len(test_windows)}"
    )
    if train_windows.size == 0:
        raise RuntimeError("No training windows found in prepared artifacts.")
    if dev_windows.size == 0:
        raise RuntimeError("No dev windows found in prepared artifacts.")
    if test_windows.size == 0:
        raise RuntimeError("No test windows found in prepared artifacts.")

    train_ds = PreparedSeqDataset(reader, split.train_main, train_windows, input_len, pred_len, stats)
    dev_ds = PreparedSeqDataset(reader, split.dev, dev_windows, input_len, pred_len, stats)
    test_ds = PreparedSeqDataset(reader, split.test, test_windows, input_len, pred_len, stats)
    _log(
        "Datasets ready: "
        f"train={len(train_ds)}, dev={len(dev_ds)}, test={len(test_ds)}"
    )

    speed_rebalance_cfg = train_cfg.get("speed_rebalance", {}) or {}
    speed_bucket_edges = _validate_speed_bucket_edges(
        speed_rebalance_cfg.get("bin_edges", [0.1, 0.2, 0.4]),
        cfg_key="train.speed_rebalance.bin_edges",
    )
    train_sampler, speed_rebalance_meta = _build_train_speed_rebalance_sampler(
        reader=reader,
        train_refs=split.train_main,
        train_windows=train_windows,
        input_len=input_len,
        pred_len=pred_len,
        speed_rebalance_cfg=speed_rebalance_cfg,
    )
    shuffle_train = bool(train_cfg.get("shuffle_train", False)) if train_sampler is None else False
    if bool(speed_rebalance_meta.get("enabled", False)):
        bin_edges = np.asarray(speed_rebalance_meta["bin_edges"], dtype=np.float64)
        bin_weights = np.asarray(speed_rebalance_meta["bin_weights"], dtype=np.float64)
        bucket_counts = np.asarray(speed_rebalance_meta["bucket_counts"], dtype=np.int64)
        bucket_labels = _speed_bucket_labels(bin_edges)
        total_windows = int(bucket_counts.sum())
        _log(
            "Speed rebalance enabled: "
            f"stat={speed_rebalance_meta['stat']}, bin_edges={bin_edges.tolist()}, "
            f"bin_weights={bin_weights.tolist()}, replacement=True, num_samples={len(train_ds)}"
        )
        for idx, label in enumerate(bucket_labels):
            count = int(bucket_counts[idx])
            ratio = (100.0 * float(count) / max(total_windows, 1))
            _log(
                f"Speed bucket {label}: count={count}, ratio={ratio:.2f}%, "
                f"sample_weight={float(bin_weights[idx]):.4g}"
            )
    else:
        _log(f"Speed rebalance disabled: using shuffle_train={shuffle_train}")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )
    _log(
        "DataLoaders ready: "
        f"batch_size={int(train_cfg['batch_size'])}, num_workers={int(train_cfg.get('num_workers', 0))}, "
        f"pin_memory={device.type == 'cuda'}"
    )

    model = build_model(
        model_name=model_cfg["name"],
        **_build_model_kwargs(
            model_cfg,
            pred_len,
            model_input_channels,
            output_channels=target_channels,
        ),
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    _log(
        "Model ready: "
        f"name={model_cfg['name']}, params={param_count:,}, pred_len={pred_len}"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    params_for_grad = [p for p in model.parameters() if p.requires_grad]
    stability_cfg = train_cfg.get("stability", {}) or {}
    grad_clip_cfg = stability_cfg.get("grad_clip", {}) or {}
    grad_clip_enabled = bool(grad_clip_cfg.get("enabled", True))
    grad_clip_max_norm = float(grad_clip_cfg.get("max_norm", 1.0))
    grad_clip_norm_type = float(grad_clip_cfg.get("norm_type", 2.0))

    scheduler_cfg = stability_cfg.get("scheduler", {}) or {}
    scheduler_enabled = bool(scheduler_cfg.get("enabled", True))
    scheduler_type = str(scheduler_cfg.get("type", "plateau")).lower()
    scheduler_monitor = str(scheduler_cfg.get("monitor", "dev_loss"))
    scheduler_factor = float(scheduler_cfg.get("factor", 0.5))
    scheduler_patience = int(scheduler_cfg.get("patience", 2))
    scheduler_threshold = float(scheduler_cfg.get("threshold", 1e-4))
    scheduler_min_lr = float(scheduler_cfg.get("min_lr", 1e-6))
    scheduler_cooldown = int(scheduler_cfg.get("cooldown", 0))
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if scheduler_enabled:
        if scheduler_type != "plateau":
            raise ValueError(
                f"Unsupported stability.scheduler.type='{scheduler_type}'. Only 'plateau' is supported."
            )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=scheduler_threshold,
            cooldown=scheduler_cooldown,
            min_lr=scheduler_min_lr,
        )

    warmup_cfg = stability_cfg.get("warmup", {}) or {}
    warmup_enabled = bool(warmup_cfg.get("enabled", True))
    warmup_epochs = max(0, int(warmup_cfg.get("epochs", 1)))
    warmup_start_factor = float(warmup_cfg.get("start_factor", 0.1))
    if not (0.0 < warmup_start_factor <= 1.0):
        raise ValueError("train.stability.warmup.start_factor must be in (0, 1].")

    max_train_batches = train_cfg.get("max_train_batches", None)
    if max_train_batches is not None:
        max_train_batches = int(max_train_batches)
    max_eval_batches = train_cfg.get("max_eval_batches", None)
    if max_eval_batches is not None:
        max_eval_batches = int(max_eval_batches)

    effective_batches_per_epoch = len(train_loader)
    if max_train_batches is not None:
        effective_batches_per_epoch = min(effective_batches_per_epoch, max_train_batches)
    warmup_steps = int(warmup_epochs * max(effective_batches_per_epoch, 0)) if warmup_enabled else 0
    warmup_base_lrs = _param_groups_lrs(optimizer)
    if warmup_steps > 0:
        _set_param_groups_lrs(optimizer, [lr * warmup_start_factor for lr in warmup_base_lrs])

    nan_guard_cfg = stability_cfg.get("nan_guard", {}) or {}
    nan_guard_enabled = bool(nan_guard_cfg.get("enabled", True))
    nan_guard_lr_decay_factor = float(nan_guard_cfg.get("lr_decay_factor", 0.5))
    nan_guard_max_bad_steps = int(nan_guard_cfg.get("max_bad_steps_per_epoch", 100))

    ema_cfg = stability_cfg.get("ema", {}) or {}
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.999))
    ema_update_after_step = int(ema_cfg.get("update_after_step", 0))
    ema_eval_with = bool(ema_cfg.get("eval_with_ema", True))
    ema_model: ModelEMA | None = None
    if ema_enabled:
        ema_model = ModelEMA(model=model, decay=ema_decay, update_after_step=ema_update_after_step)

    best_dev_loss = float("inf")
    bad_epochs = 0
    best_ckpt = output_dir / "best.ckpt"
    last_ckpt = output_dir / "last.ckpt"
    log_path = output_dir / "metrics.jsonl"
    global_step = 0
    total_bad_steps = 0
    lr_reduced_events = 0
    aborted_for_instability = False

    epochs = int(train_cfg["epochs"])
    patience = int(train_cfg["patience"])
    density_cfg = train_cfg.get("density_physics_loss", {}) or {}
    density_enabled = bool(density_cfg.get("enabled", True))
    density_weight = float(density_cfg.get("weight", 0.01))
    density_rho0 = float(density_cfg.get("rho0", 1027.0))
    density_alpha = float(density_cfg.get("alpha", 2.0e-4))
    density_beta = float(density_cfg.get("beta", 7.5e-4))
    density_t0 = float(density_cfg.get("t0", 0.0))
    density_s0 = float(density_cfg.get("s0", 35.0))

    smoothness_cfg_raw = train_cfg.get("smoothness_loss", None)
    legacy_physics_cfg = train_cfg.get("physics_loss", None)
    using_legacy_physics_cfg = smoothness_cfg_raw is None and isinstance(legacy_physics_cfg, dict)
    if using_legacy_physics_cfg:
        smoothness_cfg = legacy_physics_cfg
        _log(
            "DEPRECATED: `train.physics_loss` is interpreted as smoothness regularization. "
            "Please migrate to `train.smoothness_loss`."
        )
    else:
        smoothness_cfg = smoothness_cfg_raw or {}

    smoothness_enabled = bool(smoothness_cfg.get("enabled", False))
    smoothness_weight = float(smoothness_cfg.get("weight", 0.0))
    smoothness_spatial_weight = float(smoothness_cfg.get("spatial_weight", 1.0))
    smoothness_temporal_weight = float(smoothness_cfg.get("temporal_weight", 1.0))
    speed_aux_cfg = train_cfg.get("speed_aux_loss", {}) or {}
    speed_aux_enabled = bool(speed_aux_cfg.get("enabled", True))
    speed_aux_weight = float(speed_aux_cfg.get("weight", 0.1))
    speed_aux_type = str(speed_aux_cfg.get("type", "huber")).lower()
    speed_aux_huber_delta = float(speed_aux_cfg.get("huber_delta", 0.05))
    if speed_aux_type not in ("huber", "mse"):
        raise ValueError(
            f"Unsupported train.speed_aux_loss.type='{speed_aux_type}'. Expected 'huber' or 'mse'."
        )
    if speed_aux_type == "huber" and speed_aux_huber_delta <= 0:
        raise ValueError(
            "train.speed_aux_loss.huber_delta must be > 0 when type='huber'."
        )
    _log(
        "Training plan: "
        f"epochs={epochs}, patience={patience}, max_train_batches={max_train_batches}, "
        f"max_eval_batches={max_eval_batches}, density_enabled={density_enabled}, "
        f"density_weight={density_weight}, smoothness_enabled={smoothness_enabled}, "
        f"smoothness_weight={smoothness_weight}, speed_aux_enabled={speed_aux_enabled}, "
        f"speed_aux_weight={speed_aux_weight}, speed_aux_type={speed_aux_type}"
    )
    _log(
        "Stability plan: "
        f"grad_clip={grad_clip_enabled}(max_norm={grad_clip_max_norm}), "
        f"scheduler={scheduler_enabled}({scheduler_type}, monitor={scheduler_monitor}), "
        f"warmup={warmup_enabled}(epochs={warmup_epochs}, steps={warmup_steps}), "
        f"nan_guard={nan_guard_enabled}(max_bad_steps={nan_guard_max_bad_steps}), "
        f"ema={ema_enabled}(eval_with_ema={ema_eval_with})"
    )

    for epoch in range(1, epochs + 1):
        t_epoch = time.perf_counter()
        _log(f"Epoch {epoch}/{epochs} started.")
        model.train()
        total_train_loss = 0.0
        total_train_data_loss = 0.0
        total_train_density_loss = 0.0
        total_train_smoothness_loss = 0.0
        total_train_speed_aux_loss = 0.0
        n_train_batches = 0
        bad_steps_epoch = 0
        grad_norm_before_sum = 0.0
        grad_norm_after_sum = 0.0
        grad_norm_count = 0
        latest_grad_norm_before = 0.0
        latest_grad_norm_after = 0.0

        for b_idx, batch in enumerate(train_loader):
            if max_train_batches is not None and b_idx >= max_train_batches:
                break
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            if use_amp and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=True)
            elif use_amp:
                autocast_ctx = torch.cuda.amp.autocast(enabled=True)
            else:
                autocast_ctx = nullcontext()

            with autocast_ctx:
                x_in = augment_input_tensor(
                    x=batch["x"],
                    t_in=batch["t_in"],
                    mask=batch["mask"],
                    feature_cfg=input_feature_cfg,
                )
                pred = model(x_in)
                data_loss = masked_mse_loss(pred, batch["y"], batch["mask"])
                if density_enabled and density_weight > 0:
                    density_loss = masked_density_consistency_loss(
                        pred=pred,
                        mask=batch["mask"],
                        target_mean=stats["target_mean"],
                        target_std=stats["target_std"],
                        rho0=density_rho0,
                        alpha=density_alpha,
                        beta=density_beta,
                        t0=density_t0,
                        s0=density_s0,
                    )
                else:
                    density_loss = pred.new_zeros(())
                if smoothness_enabled and smoothness_weight > 0:
                    smoothness_loss = masked_smoothness_loss(
                        pred=pred,
                        mask=batch["mask"],
                        spatial_weight=smoothness_spatial_weight,
                        temporal_weight=smoothness_temporal_weight,
                    )
                else:
                    smoothness_loss = pred.new_zeros(())
                if speed_aux_enabled and speed_aux_weight > 0:
                    speed_aux_loss = _masked_speed_aux_loss(
                        pred_norm=pred,
                        target_norm=batch["y"],
                        mask=batch["mask"],
                        target_mean=stats["target_mean"],
                        target_std=stats["target_std"],
                        loss_type=speed_aux_type,
                        huber_delta=speed_aux_huber_delta,
                    )
                else:
                    speed_aux_loss = pred.new_zeros(())
                loss = (
                    data_loss
                    + density_weight * density_loss
                    + smoothness_weight * smoothness_loss
                    + speed_aux_weight * speed_aux_loss
                )

            if nan_guard_enabled and not torch.isfinite(loss.detach()).all():
                bad_steps_epoch += 1
                total_bad_steps += 1
                old_lrs, new_lrs = _decay_optimizer_lr(
                    optimizer,
                    factor=nan_guard_lr_decay_factor,
                    min_lr=scheduler_min_lr,
                )
                lr_reduced_events += 1
                _log(
                    f"Epoch {epoch} batch {b_idx}: non-finite loss detected; "
                    f"skipped step and decayed lr {old_lrs} -> {new_lrs}"
                )
                optimizer.zero_grad(set_to_none=True)
                if bad_steps_epoch > nan_guard_max_bad_steps:
                    aborted_for_instability = True
                    _log(
                        f"Epoch {epoch}: bad steps exceeded limit "
                        f"({bad_steps_epoch} > {nan_guard_max_bad_steps})."
                    )
                    break
                continue

            scaler.scale(loss).backward()
            if use_amp:
                scaler.unscale_(optimizer)

            if nan_guard_enabled and not _all_grads_finite(params_for_grad):
                bad_steps_epoch += 1
                total_bad_steps += 1
                old_lrs, new_lrs = _decay_optimizer_lr(
                    optimizer,
                    factor=nan_guard_lr_decay_factor,
                    min_lr=scheduler_min_lr,
                )
                lr_reduced_events += 1
                _log(
                    f"Epoch {epoch} batch {b_idx}: non-finite gradients detected; "
                    f"skipped step and decayed lr {old_lrs} -> {new_lrs}"
                )
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                if bad_steps_epoch > nan_guard_max_bad_steps:
                    aborted_for_instability = True
                    _log(
                        f"Epoch {epoch}: bad steps exceeded limit "
                        f"({bad_steps_epoch} > {nan_guard_max_bad_steps})."
                    )
                    break
                continue

            latest_grad_norm_before = _compute_grad_norm(params_for_grad, norm_type=grad_clip_norm_type)
            latest_grad_norm_after = latest_grad_norm_before
            if grad_clip_enabled:
                torch.nn.utils.clip_grad_norm_(
                    params_for_grad,
                    max_norm=grad_clip_max_norm,
                    norm_type=grad_clip_norm_type,
                )
                latest_grad_norm_after = _compute_grad_norm(params_for_grad, norm_type=grad_clip_norm_type)

            grad_norm_before_sum += latest_grad_norm_before
            grad_norm_after_sum += latest_grad_norm_after
            grad_norm_count += 1

            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            if warmup_steps > 0 and global_step <= warmup_steps:
                warmup_progress = float(global_step) / float(max(warmup_steps, 1))
                warmup_factor = warmup_start_factor + (1.0 - warmup_start_factor) * warmup_progress
                _set_param_groups_lrs(
                    optimizer,
                    [base_lr * warmup_factor for base_lr in warmup_base_lrs],
                )

            if ema_model is not None:
                ema_model.update(model=model, step=global_step)

            total_train_loss += float(loss.item())
            total_train_data_loss += float(data_loss.item())
            total_train_density_loss += float(density_loss.item())
            total_train_smoothness_loss += float(smoothness_loss.item())
            total_train_speed_aux_loss += float(speed_aux_loss.item())
            n_train_batches += 1
            if n_train_batches % 200 == 0:
                _log(
                    f"Epoch {epoch}: processed {n_train_batches} train batches; "
                    f"latest_loss={float(loss.item()):.6f}, lr={_param_groups_lrs(optimizer)[0]:.3e}, "
                    f"grad_norm_before_clip={latest_grad_norm_before:.4f}, "
                    f"grad_norm_after_clip={latest_grad_norm_after:.4f}, bad_steps={bad_steps_epoch}"
                )

        train_loss = total_train_loss / max(n_train_batches, 1)
        train_data_loss = total_train_data_loss / max(n_train_batches, 1)
        train_density_loss = total_train_density_loss / max(n_train_batches, 1)
        train_smoothness_loss = total_train_smoothness_loss / max(n_train_batches, 1)
        train_speed_aux_loss = total_train_speed_aux_loss / max(n_train_batches, 1)

        if ema_model is not None and ema_eval_with:
            ema_backup = ema_model.apply_to(model)
            try:
                dev_metrics = evaluate(
                    model,
                    dev_loader,
                    device=device,
                    target_mean=stats["target_mean"],
                    target_std=stats["target_std"],
                    nrmse_denom=stats["nrmse_denom"],
                    input_feature_cfg=input_feature_cfg,
                    speed_bucket_edges=speed_bucket_edges,
                    max_batches=max_eval_batches,
                )
            finally:
                ema_model.restore(model, ema_backup)
        else:
            dev_metrics = evaluate(
                model,
                dev_loader,
                device=device,
                target_mean=stats["target_mean"],
                target_std=stats["target_std"],
                nrmse_denom=stats["nrmse_denom"],
                input_feature_cfg=input_feature_cfg,
                speed_bucket_edges=speed_bucket_edges,
                max_batches=max_eval_batches,
            )

        _log_bucket_eval_metrics(
            metrics=dev_metrics,
            bucket_edges=speed_bucket_edges,
            prefix=f"Epoch {epoch} dev",
        )

        monitor_value = _get_monitor_value(dev_metrics, scheduler_monitor)
        scheduler_action = "disabled"
        if scheduler is not None:
            if warmup_steps > 0 and global_step < warmup_steps:
                scheduler_action = "skipped_warmup"
            else:
                before_lrs = _param_groups_lrs(optimizer)
                scheduler.step(monitor_value)
                after_lrs = _param_groups_lrs(optimizer)
                if any(a < b - 1e-12 for a, b in zip(after_lrs, before_lrs)):
                    scheduler_action = "reduced"
                    lr_reduced_events += 1
                    _log(f"Epoch {epoch}: scheduler reduced lr {before_lrs} -> {after_lrs}")
                else:
                    scheduler_action = "kept"

        row = {
            "epoch": epoch,
            "lr": _param_groups_lrs(optimizer)[0],
            "train_loss": train_loss,
            "train_data_loss": train_data_loss,
            "train_density_loss": train_density_loss,
            "density_consistency_weight": density_weight,
            "train_smoothness_loss": train_smoothness_loss,
            "smoothness_weight": smoothness_weight,
            "train_speed_aux_loss": train_speed_aux_loss,
            "speed_aux_weight": speed_aux_weight,
            "train_grad_norm_before_clip": grad_norm_before_sum / max(grad_norm_count, 1),
            "train_grad_norm_after_clip": grad_norm_after_sum / max(grad_norm_count, 1),
            "bad_steps_epoch": bad_steps_epoch,
            "bad_steps_total": total_bad_steps,
            "scheduler_action": scheduler_action,
            "scheduler_monitor": scheduler_monitor,
            "scheduler_monitor_value": monitor_value,
            "dev_loss": dev_metrics["loss"],
            "dev_nrmse_mean": dev_metrics["nrmse_mean"],
            "dev_nrmse_sst": dev_metrics["nrmse_sst"],
            "dev_nrmse_sss": dev_metrics["nrmse_sss"],
            "dev_nrmse_speed": dev_metrics["nrmse_speed"],
        }
        for bucket_i in range(int(speed_bucket_edges.size) + 1):
            row[f"dev_speed_bucket_{bucket_i}_samples"] = float(
                dev_metrics.get(f"speed_bucket_{bucket_i}_samples", 0.0)
            )
            row[f"dev_speed_bucket_{bucket_i}_rmse"] = float(
                dev_metrics.get(f"speed_bucket_{bucket_i}_rmse", float("nan"))
            )
            row[f"dev_speed_bucket_{bucket_i}_nrmse"] = float(
                dev_metrics.get(f"speed_bucket_{bucket_i}_nrmse", float("nan"))
            )
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False))
        _log(f"Epoch {epoch} finished in {time.perf_counter() - t_epoch:.1f}s")

        ckpt_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict(),
            "stability_state": {
                "best_dev_loss": float(best_dev_loss),
                "global_step": int(global_step),
                "bad_steps_epoch": int(bad_steps_epoch),
                "total_bad_steps": int(total_bad_steps),
                "lr_reduced_events": int(lr_reduced_events),
                "warmup_steps": int(warmup_steps),
                "warmup_done": bool(global_step >= warmup_steps),
                "scheduler_monitor": scheduler_monitor,
                "ema_enabled": ema_enabled,
                "ema_eval_with": ema_eval_with,
            },
            "ema_state": ema_model.state_dict() if ema_model is not None else None,
            "config": cfg,
            "stats": {k: np.asarray(v) for k, v in stats.items()},
            "dev_metrics": dev_metrics,
        }
        torch.save(ckpt_payload, last_ckpt)

        if dev_metrics["loss"] < best_dev_loss:
            best_dev_loss = float(dev_metrics["loss"])
            bad_epochs = 0
            ckpt_payload["stability_state"]["best_dev_loss"] = float(best_dev_loss)
            torch.save(ckpt_payload, best_ckpt)
        else:
            bad_epochs += 1

        if aborted_for_instability:
            _log(f"Stopped training due to instability at epoch={epoch}.")
            break
        if bad_epochs >= patience:
            print(f"Early stopping triggered at epoch={epoch}.")
            _log(f"Early stopping triggered at epoch={epoch}.")
            break

    if not best_ckpt.exists():
        raise RuntimeError("Training ended before producing a valid best checkpoint.")

    _log("Evaluating best checkpoint on test split...")
    best = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])

    best_ema_state = best.get("ema_state")
    best_stability_state = best.get("stability_state", {}) or {}
    run_test_with_ema = bool(best_stability_state.get("ema_eval_with", False)) and best_ema_state is not None
    if run_test_with_ema:
        ema_for_test = ModelEMA(
            model=model,
            decay=float(best_ema_state.get("decay", 0.999)),
            update_after_step=int(best_ema_state.get("update_after_step", 0)),
        )
        ema_for_test.load_state_dict(best_ema_state)
        ema_backup = ema_for_test.apply_to(model)
        try:
            test_metrics, inference_profile = evaluate_with_profile(
                model,
                test_loader,
                device=device,
                target_mean=stats["target_mean"],
                target_std=stats["target_std"],
                nrmse_denom=stats["nrmse_denom"],
                input_feature_cfg=input_feature_cfg,
                speed_bucket_edges=speed_bucket_edges,
                max_batches=max_eval_batches,
            )
        finally:
            ema_for_test.restore(model, ema_backup)
    else:
        test_metrics, inference_profile = evaluate_with_profile(
            model,
            test_loader,
            device=device,
            target_mean=stats["target_mean"],
            target_std=stats["target_std"],
            nrmse_denom=stats["nrmse_denom"],
            input_feature_cfg=input_feature_cfg,
            speed_bucket_edges=speed_bucket_edges,
            max_batches=max_eval_batches,
        )

    result = {
        "best_dev_loss": float(best.get("dev_metrics", {}).get("loss", best_dev_loss)),
        "best_dev_nrmse_mean": float(best.get("dev_metrics", {}).get("nrmse_mean", float("nan"))),
        "test_metrics": test_metrics,
        "inference_profile": inference_profile,
        "device": str(device),
        "best_checkpoint": str(best_ckpt),
        "stability": {
            "scheduler_enabled": scheduler_enabled,
            "scheduler_monitor": scheduler_monitor,
            "warmup_steps": warmup_steps,
            "total_bad_steps": total_bad_steps,
            "lr_reduced_events": lr_reduced_events,
            "ema_enabled": ema_enabled,
            "ema_eval_with": ema_eval_with,
        },
    }
    with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    reader.close()
    _log(f"Training complete. Total elapsed={time.perf_counter() - t0_all:.1f}s")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .config import load_config
from .data.dataset import load_stats
from .data.zip_reader import datetime_to_hour_index, hour_index_to_datetime, parse_datetime64, ZipNetCDFReader
from .metrics import MaskedChannelRMSE, rmse_to_nrmse, summarize_channel_metrics
from .models.registry import build_model
from .utils import resolve_device

CHANNEL_NAMES = ("sst", "sss", "speed")


def _build_model_kwargs(model_cfg: Dict, pred_len: int) -> Dict:
    kwargs = {k: v for k, v in model_cfg.items() if k != "name"}
    kwargs["input_channels"] = 4
    kwargs["output_channels"] = 3
    kwargs["default_pred_len"] = int(pred_len)
    return kwargs


def _stats_from_checkpoint(ckpt: Dict) -> Dict[str, np.ndarray]:
    if "stats" not in ckpt:
        raise KeyError("Checkpoint has no 'stats'. Provide --stats-path to a stats.npz file.")
    raw = ckpt["stats"]
    return {k: np.asarray(v) for k, v in raw.items()}


def _sanitize_time_for_filename(value: str) -> str:
    cleaned = value.replace(":", "").replace(" ", "_")
    cleaned = re.sub(r"[^0-9A-Za-zT_\-]", "", cleaned)
    return cleaned or "unknown_time"


def _sample_frame_indices(total_steps: int, every_hours: int) -> List[int]:
    if total_steps <= 0:
        return []
    if every_hours <= 0:
        raise ValueError(f"sample_every_hours must be > 0, got {every_hours}")
    return list(range(0, total_steps, every_hours))


def _try_collect_ground_truth(
    reader: ZipNetCDFReader,
    ref_map: Dict[int, object],
    forecast_hours: np.ndarray,
    ocean_mask: np.ndarray,
) -> Tuple[np.ndarray | None, str | None]:
    missing = [int(h) for h in forecast_hours if int(h) not in ref_map]
    if missing:
        return None, f"ground-truth missing for forecast hour_index={missing[0]}"

    gt_frames = []
    for hour in forecast_hours:
        frame, _ = reader.read_frame(ref_map[int(hour)])
        speed = np.sqrt(np.square(frame[2]) + np.square(frame[3]))
        target = np.stack([frame[0], frame[1], speed], axis=0).astype(np.float32)
        gt_frames.append(target)

    gt_raw = np.stack(gt_frames, axis=0).astype(np.float32)
    gt_raw = np.where(ocean_mask[None, None, :, :], gt_raw, np.nan).astype(np.float32)
    return gt_raw, None


def _compute_metrics_payload(
    pred_raw: np.ndarray,
    gt_raw: np.ndarray | None,
    ocean_mask: np.ndarray,
    nrmse_denom: np.ndarray,
    reason_if_unavailable: str,
) -> Dict[str, object]:
    if gt_raw is None:
        return {
            "metrics_available": False,
            "metrics_unavailable_reason": reason_if_unavailable,
        }

    meter = MaskedChannelRMSE(n_channels=3)
    pred_t = torch.from_numpy(pred_raw).unsqueeze(0).to(torch.float32)
    gt_t = torch.from_numpy(gt_raw).unsqueeze(0).to(torch.float32)
    mask_t = torch.from_numpy(ocean_mask.astype(np.float32)).unsqueeze(0)
    meter.update(pred_t, gt_t, mask_t)

    rmse = meter.compute_rmse()
    nrmse = rmse_to_nrmse(rmse, nrmse_denom)
    payload: Dict[str, object] = {
        "metrics_available": True,
        **summarize_channel_metrics(rmse, nrmse),
    }
    payload["pass_15pct"] = bool(float(payload["nrmse_mean"]) <= 0.15)
    return payload


def _safe_min_max(values: np.ndarray) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return vmin - 1.0, vmax + 1.0
    return vmin, vmax


def _save_visualizations(
    pred_raw: np.ndarray,
    gt_raw: np.ndarray | None,
    forecast_hours: np.ndarray,
    start_hour: int,
    start_time: np.datetime64,
    lat: np.ndarray,
    lon: np.ndarray,
    viz_dir: Path,
    sample_every_hours: int,
    show: bool,
) -> List[str]:
    import matplotlib.pyplot as plt

    viz_dir.mkdir(parents=True, exist_ok=True)
    lon2d, lat2d = np.meshgrid(lon, lat)
    frame_indices = _sample_frame_indices(pred_raw.shape[0], sample_every_hours)
    start_tag = _sanitize_time_for_filename(str(start_time))
    image_paths: List[str] = []
    has_gt = gt_raw is not None

    for t_idx in frame_indices:
        f_hour = int(forecast_hours[t_idx])
        f_time = str(hour_index_to_datetime(f_hour))
        offset_hour = int(f_hour - start_hour)

        if has_gt:
            fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
            col_titles = ("Pred", "GT", "Error")
            for c, cname in enumerate(CHANNEL_NAMES):
                pred_map = pred_raw[t_idx, c]
                gt_map = gt_raw[t_idx, c]
                err_map = pred_map - gt_map

                vmin, vmax = _safe_min_max(np.stack([pred_map, gt_map], axis=0))
                err_lim = float(np.nanmax(np.abs(err_map[np.isfinite(err_map)]))) if np.any(np.isfinite(err_map)) else 1.0
                if err_lim == 0:
                    err_lim = 1.0

                panels = (
                    (pred_map, col_titles[0], "viridis", vmin, vmax),
                    (gt_map, col_titles[1], "viridis", vmin, vmax),
                    (err_map, col_titles[2], "coolwarm", -err_lim, err_lim),
                )
                for col, (panel_data, title, cmap, pmin, pmax) in enumerate(panels):
                    ax = axes[c, col]
                    mesh = ax.pcolormesh(lon2d, lat2d, panel_data, shading="auto", cmap=cmap, vmin=pmin, vmax=pmax)
                    ax.set_title(f"{cname.upper()} {title}", fontsize=10)
                    ax.set_xlabel("Lon")
                    ax.set_ylabel("Lat")
                    fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        else:
            fig, axes = plt.subplots(3, 1, figsize=(6, 11), constrained_layout=True)
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            for c, cname in enumerate(CHANNEL_NAMES):
                panel_data = pred_raw[t_idx, c]
                vmin, vmax = _safe_min_max(panel_data)
                ax = axes[c]
                mesh = ax.pcolormesh(lon2d, lat2d, panel_data, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_title(f"{cname.upper()} Pred", fontsize=10)
                ax.set_xlabel("Lon")
                ax.set_ylabel("Lat")
                fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"Forecast {f_time} | truth unavailable", fontsize=12)

        if has_gt:
            fig.suptitle(f"Forecast {f_time} | offset +{offset_hour}h", fontsize=12)

        out_path = viz_dir / f"forecast_{start_tag}_t{offset_hour:03d}.png"
        fig.savefig(out_path, dpi=150)
        if show:
            plt.show(block=False)
        plt.close(fig)
        image_paths.append(str(out_path))

    return image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for Yellow-Bohai Sea ConvLSTM baseline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint.")
    parser.add_argument("--start_time", required=True, help="Forecast start time (e.g. 2014-06-01T00:00:00).")
    parser.add_argument(
        "--stats_path",
        default=None,
        help="Optional stats .npz path. If omitted, load stats from checkpoint.",
    )
    parser.add_argument(
        "--viz_dir",
        default="outputs/infer_viz",
        help="Directory for visualization PNG files.",
    )
    parser.add_argument(
        "--sample_every_hours",
        type=int,
        default=9,
        help="Save one visualization every N forecast hours.",
    )
    parser.add_argument(
        "--metrics_json",
        default=None,
        help="Optional metrics JSON path. Default: <viz_dir>/metrics.json",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also show figures in a local window.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--auto_shift_start",
        action="store_true",
        help="If requested start_time is invalid, shift forward to nearest valid start with full history.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    device_cfg = args.device if args.device is not None else cfg["train"]["device"]
    device = resolve_device(device_cfg)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    stats = load_stats(args.stats_path) if args.stats_path else _stats_from_checkpoint(ckpt)

    years = sorted(set(data_cfg["train_years"] + data_cfg["test_years"]))
    reader = ZipNetCDFReader(
        root_dir=data_cfg["root_dir"],
        years=years,
        cache_size=data_cfg.get("cache_size", 8),
    )

    input_len = int(data_cfg["input_len"])
    pred_len = int(data_cfg["pred_len"])
    start_time = parse_datetime64(args.start_time)
    start_hour = datetime_to_hour_index(start_time)
    requested_start_hour = start_hour

    if args.auto_shift_start:
        refs = reader.build_index(
            start_time=parse_datetime64(data_cfg["train_start"]),
            end_time=parse_datetime64(data_cfg["test_end"]),
        )
    else:
        refs = reader.build_index(
            start_time=hour_index_to_datetime(int(start_hour - input_len)),
            end_time=hour_index_to_datetime(int(start_hour + pred_len - 1)),
        )
    ref_map = {r.hour_index: r for r in refs}
    hour_set = set(ref_map.keys())

    def has_history_window(h: int) -> bool:
        return all((hh in hour_set) for hh in range(h - input_len, h))

    if not has_history_window(start_hour):
        if not args.auto_shift_start:
            raise RuntimeError(
                "Missing required history hours for inference. "
                "Try another start_time or use --auto_shift_start."
            )
        max_hour = max(hour_set)
        found = None
        for cand in range(start_hour, max_hour + 1):
            if has_history_window(cand):
                found = cand
                break
        if found is None:
            raise RuntimeError("Could not find a valid shifted start_time with full history.")
        start_hour = int(found)
        start_time = hour_index_to_datetime(start_hour)
        print(
            f"Requested start_time shifted from {hour_index_to_datetime(requested_start_hour)} "
            f"to {start_time} for continuity."
        )

    input_hours = np.arange(start_hour - input_len, start_hour, dtype=np.int64)
    forecast_hours = np.arange(start_hour, start_hour + pred_len, dtype=np.int64)

    missing = [int(h) for h in input_hours if int(h) not in ref_map]
    if missing:
        raise RuntimeError(
            f"Missing required history hours for inference. First missing hour_index={missing[0]}"
        )

    x_frames = []
    for hour in input_hours:
        frame, _ = reader.read_frame(ref_map[int(hour)])
        x_frames.append(frame)
    x_raw = np.stack(x_frames, axis=0).astype(np.float32)  # [Tin, 4, H, W]

    input_mean = stats["input_mean"].astype(np.float32)
    input_std = np.maximum(stats["input_std"].astype(np.float32), 1e-6)
    target_mean = stats["target_mean"].astype(np.float32)
    target_std = np.maximum(stats["target_std"].astype(np.float32), 1e-6)
    ocean_mask = stats["ocean_mask"].astype(np.float32) > 0.5

    x = (x_raw - input_mean[None, :, None, None]) / input_std[None, :, None, None]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device=device, dtype=torch.float32)

    model = build_model(model_name=model_cfg["name"], **_build_model_kwargs(model_cfg, pred_len))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        pred_norm = model(x_tensor, pred_len=pred_len)  # [1, Tout, 3, H, W]
    pred_norm = pred_norm.squeeze(0).cpu().numpy().astype(np.float32)
    pred_raw = pred_norm * target_std[None, :, None, None] + target_mean[None, :, None, None]
    pred_raw = np.where(ocean_mask[None, None, :, :], pred_raw, np.nan).astype(np.float32)

    lat = stats["lat"].astype(np.float32)
    lon = stats["lon"].astype(np.float32)
    gt_raw, missing_reason = _try_collect_ground_truth(reader, ref_map, forecast_hours, ocean_mask)
    metrics_payload = _compute_metrics_payload(
        pred_raw=pred_raw,
        gt_raw=gt_raw,
        ocean_mask=ocean_mask,
        nrmse_denom=stats["nrmse_denom"],
        reason_if_unavailable=missing_reason or "ground truth unavailable for forecast range",
    )

    viz_dir = Path(args.viz_dir)
    image_paths = _save_visualizations(
        pred_raw=pred_raw,
        gt_raw=gt_raw,
        forecast_hours=forecast_hours,
        start_hour=start_hour,
        start_time=start_time,
        lat=lat,
        lon=lon,
        viz_dir=viz_dir,
        sample_every_hours=int(args.sample_every_hours),
        show=bool(args.show),
    )

    metrics_json_path = Path(args.metrics_json) if args.metrics_json else (viz_dir / "metrics.json")
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output = {
        **metrics_payload,
        "device": str(device),
        "start_time": str(start_time),
        "pred_len": pred_len,
        "sample_every_hours": int(args.sample_every_hours),
        "generated_images": len(image_paths),
        "viz_dir": str(viz_dir),
    }
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_output, f, ensure_ascii=False, indent=2)

    summary = {
        "shape": list(pred_raw.shape),
        "device": str(device),
        "start_time": str(start_time),
        "pred_len": pred_len,
        "viz_dir": str(viz_dir),
        "generated_images": len(image_paths),
        "metrics_json": str(metrics_json_path),
        "metrics_available": bool(metrics_output["metrics_available"]),
    }
    if metrics_output["metrics_available"]:
        summary["nrmse_mean"] = float(metrics_output["nrmse_mean"])
        summary["pass_15pct"] = bool(metrics_output["pass_15pct"])
    else:
        summary["metrics_unavailable_reason"] = str(metrics_output["metrics_unavailable_reason"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

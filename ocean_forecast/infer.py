from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .config import load_config
from .data.dataset import load_stats
from .data.zip_reader import datetime_to_hour_index, hour_index_to_datetime, parse_datetime64, ZipNetCDFReader
from .models.registry import build_model
from .utils import resolve_device


def _stats_from_checkpoint(ckpt: Dict) -> Dict[str, np.ndarray]:
    if "stats" not in ckpt:
        raise KeyError("Checkpoint has no 'stats'. Provide --stats-path to a stats.npz file.")
    raw = ckpt["stats"]
    return {k: np.asarray(v) for k, v in raw.items()}


def _save_hourly_stats_csv(path: Path, pred: np.ndarray, forecast_hours: np.ndarray, ocean_mask: np.ndarray) -> None:
    # pred: [T, 3, H, W]
    path.parent.mkdir(parents=True, exist_ok=True)
    channel_names = ("sst", "sss", "speed")
    mask = ocean_mask.astype(bool)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["hour_index", "forecast_time", "channel", "mean", "min", "max"]
        )
        for t_idx in range(pred.shape[0]):
            ts = str(hour_index_to_datetime(int(forecast_hours[t_idx])))
            for c, cname in enumerate(channel_names):
                vals = pred[t_idx, c][mask]
                writer.writerow(
                    [
                        int(forecast_hours[t_idx]),
                        ts,
                        cname,
                        float(np.nanmean(vals)),
                        float(np.nanmin(vals)),
                        float(np.nanmax(vals)),
                    ]
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for Yellow-Bohai Sea ConvLSTM baseline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint.")
    parser.add_argument("--start_time", required=True, help="Forecast start time (e.g. 2014-06-01T00:00:00).")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument(
        "--stats_csv",
        default=None,
        help="Optional CSV path for hourly ocean-grid summary stats.",
    )
    parser.add_argument(
        "--stats_path",
        default=None,
        help="Optional stats .npz path. If omitted, load stats from checkpoint.",
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
            end_time=hour_index_to_datetime(int(start_hour - 1)),
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

    model = build_model(
        model_name=model_cfg["name"],
        input_channels=4,
        output_channels=3,
        hidden_dims=model_cfg["hidden_dims"],
        kernel_size=model_cfg["kernel_size"],
        dropout=model_cfg.get("dropout", 0.0),
        default_pred_len=pred_len,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        pred_norm = model(x_tensor, pred_len=pred_len)  # [1, Tout, 3, H, W]
    pred_norm = pred_norm.squeeze(0).cpu().numpy().astype(np.float32)
    pred_raw = pred_norm * target_std[None, :, None, None] + target_mean[None, :, None, None]
    pred_raw = np.where(ocean_mask[None, None, :, :], pred_raw, np.nan).astype(np.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lat = stats["lat"].astype(np.float32)
    lon = stats["lon"].astype(np.float32)
    forecast_time_str = np.array([str(hour_index_to_datetime(int(h))) for h in forecast_hours], dtype=object)

    np.savez_compressed(
        output_path,
        pred=pred_raw,
        forecast_hours=forecast_hours,
        forecast_times=forecast_time_str,
        lat=lat,
        lon=lon,
        ocean_mask=ocean_mask.astype(np.uint8),
        variables=np.array(["sst", "sss", "speed"], dtype=object),
        start_time=str(start_time),
    )

    if args.stats_csv:
        _save_hourly_stats_csv(Path(args.stats_csv), pred_raw, forecast_hours, ocean_mask)

    summary = {
        "output": str(output_path),
        "shape": list(pred_raw.shape),
        "device": str(device),
        "start_time": str(start_time),
        "pred_len": pred_len,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

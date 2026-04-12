from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .zip_reader import FrameRef, parse_datetime64


TARGET_VARIABLES = ("sst", "sss", "ssu", "ssv")


@dataclass
class SplitRefs:
    train_main: List[FrameRef]
    dev: List[FrameRef]
    test: List[FrameRef]


def select_refs_by_range(
    refs: Sequence[FrameRef],
    start_time: np.datetime64,
    end_time: np.datetime64,
) -> List[FrameRef]:
    return [r for r in refs if start_time <= r.timestamp <= end_time]


def split_refs(refs: Sequence[FrameRef], data_cfg: Dict) -> SplitRefs:
    train_start = parse_datetime64(data_cfg["train_start"])
    train_main_end = parse_datetime64(data_cfg["train_main_end"])
    dev_start = parse_datetime64(data_cfg["dev_start"])
    dev_end = parse_datetime64(data_cfg["dev_end"])
    test_start = parse_datetime64(data_cfg["test_start"])
    test_end = parse_datetime64(data_cfg["test_end"])

    return SplitRefs(
        train_main=select_refs_by_range(refs, train_start, train_main_end),
        dev=select_refs_by_range(refs, dev_start, dev_end),
        test=select_refs_by_range(refs, test_start, test_end),
    )


def build_window_starts(
    refs: Sequence[FrameRef],
    input_len: int,
    pred_len: int,
    stride: int = 1,
) -> List[int]:
    total = input_len + pred_len
    if len(refs) < total:
        return []

    hours = np.array([r.hour_index for r in refs], dtype=np.int64)
    starts: List[int] = []
    for s in range(0, len(refs) - total + 1, stride):
        segment = hours[s : s + total]
        if np.all(np.diff(segment) == 1):
            starts.append(s)
    return starts


def _update_channel_stats(
    sum_arr: np.ndarray,
    sq_arr: np.ndarray,
    cnt_arr: np.ndarray,
    channels: np.ndarray,
    mask: np.ndarray,
) -> None:
    # channels: [C, H, W], mask: [H, W]
    for c in range(channels.shape[0]):
        valid = mask & np.isfinite(channels[c])
        if not np.any(valid):
            continue
        vals = channels[c][valid].astype(np.float64)
        sum_arr[c] += vals.sum()
        sq_arr[c] += np.square(vals).sum()
        cnt_arr[c] += vals.size


def compute_normalization_stats(
    reader: Any,
    refs: Sequence[FrameRef],
) -> Dict[str, np.ndarray]:
    in_sum = np.zeros(4, dtype=np.float64)
    in_sq = np.zeros(4, dtype=np.float64)
    in_cnt = np.zeros(4, dtype=np.float64)

    out_sum = np.zeros(4, dtype=np.float64)
    out_sq = np.zeros(4, dtype=np.float64)
    out_cnt = np.zeros(4, dtype=np.float64)
    speed_sum = 0.0
    speed_sq = 0.0
    speed_cnt = 0.0

    ocean_mask = None
    for ref in refs:
        frame, mask = reader.read_frame(ref)
        if ocean_mask is None:
            ocean_mask = mask.copy()

        _update_channel_stats(in_sum, in_sq, in_cnt, frame, mask)
        _update_channel_stats(out_sum, out_sq, out_cnt, frame, mask)

        speed = np.sqrt(np.square(frame[2]) + np.square(frame[3]))
        valid_speed = mask & np.isfinite(speed)
        if np.any(valid_speed):
            vals = speed[valid_speed].astype(np.float64)
            speed_sum += vals.sum()
            speed_sq += np.square(vals).sum()
            speed_cnt += float(vals.size)

    if ocean_mask is None:
        raise RuntimeError("No valid frames were found in training range.")

    in_mean = in_sum / np.maximum(in_cnt, 1.0)
    out_mean = out_sum / np.maximum(out_cnt, 1.0)

    in_var = in_sq / np.maximum(in_cnt, 1.0) - np.square(in_mean)
    out_var = out_sq / np.maximum(out_cnt, 1.0) - np.square(out_mean)

    in_std = np.sqrt(np.maximum(in_var, 1e-12))
    out_std = np.sqrt(np.maximum(out_var, 1e-12))
    speed_mean = speed_sum / max(speed_cnt, 1.0)
    speed_var = speed_sq / max(speed_cnt, 1.0) - speed_mean * speed_mean
    speed_std = float(np.sqrt(max(speed_var, 1e-12)))

    lat, lon = reader.get_lat_lon()
    return {
        "input_mean": in_mean.astype(np.float32),
        "input_std": in_std.astype(np.float32),
        "target_mean": out_mean.astype(np.float32),
        "target_std": out_std.astype(np.float32),
        "nrmse_denom": np.asarray([out_std[0], out_std[1], speed_std], dtype=np.float32),
        "ocean_mask": ocean_mask.astype(np.float32),
        "lat": lat.astype(np.float32),
        "lon": lon.astype(np.float32),
    }


def save_stats(stats: Dict[str, np.ndarray], path: str | Path) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **stats)


def load_stats(path: str | Path) -> Dict[str, np.ndarray]:
    obj = np.load(Path(path))
    return {k: obj[k] for k in obj.files}


class OceanSeqDataset(Dataset):
    def __init__(
        self,
        reader: Any,
        refs: Sequence[FrameRef],
        window_starts: Sequence[int],
        input_len: int,
        pred_len: int,
        stats: Dict[str, np.ndarray],
        front_seg_enabled: bool = False,
        front_seg_quantile: float = 0.9,
    ):
        self.reader = reader
        self.refs = list(refs)
        self.window_starts = list(window_starts)
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.front_seg_enabled = bool(front_seg_enabled)
        self.front_seg_quantile = float(front_seg_quantile)
        if not (0.0 < self.front_seg_quantile < 1.0):
            raise ValueError(
                f"front_seg_quantile must be in (0, 1), got {self.front_seg_quantile}."
            )

        self.input_mean = stats["input_mean"].astype(np.float32)
        self.input_std = stats["input_std"].astype(np.float32)
        self.target_mean = stats["target_mean"].astype(np.float32)
        self.target_std = stats["target_std"].astype(np.float32)
        self.target_channels = int(self.target_mean.shape[0])
        self.ocean_mask = stats["ocean_mask"].astype(np.float32)
        self.ocean_mask_bool = self.ocean_mask > 0.5

    def __len__(self) -> int:
        return len(self.window_starts)

    @staticmethod
    def _nan_fill(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
        valid = np.asarray(mask, dtype=bool) & np.isfinite(field)
        if np.any(valid):
            fill = float(np.mean(field[valid], dtype=np.float64))
        else:
            fill = 0.0
        out = np.asarray(field, dtype=np.float32).copy()
        out[~np.isfinite(out)] = fill
        out[~np.asarray(mask, dtype=bool)] = fill
        return out

    @classmethod
    def _gradient_mag(cls, field: np.ndarray, mask: np.ndarray) -> np.ndarray:
        filled = cls._nan_fill(field, mask)
        gy, gx = np.gradient(filled)
        mag = np.sqrt(np.square(gx) + np.square(gy)).astype(np.float32, copy=False)
        mag[~np.asarray(mask, dtype=bool)] = np.nan
        return mag

    def _build_front_mask(self, y_raw: np.ndarray) -> np.ndarray:
        # y_raw: [Tout, C, H, W] in denormalized/raw units.
        if y_raw.ndim != 4 or y_raw.shape[1] < 2:
            raise ValueError(
                f"front mask requires y_raw [T,C,H,W] with C>=2 (sst,sss), got {y_raw.shape}"
            )
        mask_t: List[np.ndarray] = []
        valid_ocean = self.ocean_mask_bool
        for t in range(y_raw.shape[0]):
            grad_sst = self._gradient_mag(y_raw[t, 0], valid_ocean)
            grad_sss = self._gradient_mag(y_raw[t, 1], valid_ocean)
            signal = np.maximum(grad_sst, grad_sss).astype(np.float32, copy=False)
            vals = signal[valid_ocean & np.isfinite(signal)]
            if vals.size == 0:
                mask_t.append(np.zeros_like(valid_ocean, dtype=np.float32))
                continue
            thr = float(np.quantile(vals, self.front_seg_quantile))
            front = (signal >= thr) & valid_ocean
            mask_t.append(front.astype(np.float32, copy=False))
        return np.stack(mask_t, axis=0)[:, None, :, :]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.window_starts[idx]
        end = start + self.input_len + self.pred_len
        window_refs = self.refs[start:end]

        frames = []
        hours = []
        for ref in window_refs:
            frame, _ = self.reader.read_frame(ref)
            frames.append(frame)
            hours.append(ref.hour_index)

        seq = np.stack(frames, axis=0)  # [Tin+Tout, 4, H, W]
        x_raw = seq[: self.input_len]
        future = seq[self.input_len :]
        if self.target_channels == 4:
            y_raw = future[:, :4, :, :]  # [Tout, 4, H, W]
        elif self.target_channels == 3:
            speed = np.sqrt(np.square(future[:, 2]) + np.square(future[:, 3]))
            y_raw = np.stack([future[:, 0], future[:, 1], speed], axis=1)  # [Tout, 3, H, W]
        else:
            raise ValueError(
                f"Unsupported target_channels={self.target_channels}; expected 3 (legacy) or 4 (uv)."
            )

        x = (x_raw - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        y = (y_raw - self.target_mean[None, :, None, None]) / self.target_std[None, :, None, None]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        t_in = np.array(hours[: self.input_len], dtype=np.int64)
        t_out = np.array(hours[self.input_len :], dtype=np.int64)
        sample = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "mask": torch.from_numpy(self.ocean_mask.copy()),
            "t_in": torch.from_numpy(t_in),
            "t_out": torch.from_numpy(t_out),
        }
        if self.front_seg_enabled:
            front_mask = self._build_front_mask(y_raw=y_raw).astype(np.float32, copy=False)
            sample["front_mask"] = torch.from_numpy(front_mask)
        return sample


class PreparedSeqDataset(OceanSeqDataset):
    """Alias dataset used by prepared-artifact training pipeline."""

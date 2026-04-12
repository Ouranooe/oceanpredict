from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocean_forecast.config import load_config


@dataclass
class FrameDiag:
    ref: Any
    timestamp: np.datetime64
    year: int
    month: int
    season: str
    mask: np.ndarray
    maps: Dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upgraded nine-step structure analyzer.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"], help="Data split.")
    parser.add_argument("--max_frames", type=int, default=240, help="Max sampled frames for primary pass.")
    parser.add_argument("--min_hour_gap", type=int, default=24, help="Primary sampling gap (hours).")
    parser.add_argument("--time_scales", default="1,6,24", help="Timescale gaps (hours), comma separated.")
    parser.add_argument("--max_frames_per_scale", type=int, default=120, help="Max frames per timescale branch.")
    parser.add_argument("--max_shift", type=int, default=2, help="Max pixel shift for aligned IoU.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--skip_plots", action="store_true", help="Disable PNG plotting.")
    parser.add_argument("--output_json", default="outputs/analysis/structure_analysis_v2.json", help="Output JSON.")
    parser.add_argument("--output_md", default="outputs/analysis/structure_analysis_v2.md", help="Output markdown.")
    parser.add_argument("--plot_dir", default="outputs/analysis/plots", help="Output plot dir.")
    return parser.parse_args()


def _parse_time_scales(text: str) -> List[int]:
    vals = sorted(set(int(x.strip()) for x in text.split(",") if x.strip()))
    if not vals:
        return [1, 6, 24]
    if any(v <= 0 for v in vals):
        raise ValueError(f"Invalid time_scales={vals}, must be > 0.")
    return vals


def _ym(ts: np.datetime64) -> Tuple[int, int]:
    s = str(ts)
    return int(s[:4]), int(s[5:7])


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def _select_refs_by_range(refs: Sequence[Any], start_time: np.datetime64, end_time: np.datetime64) -> List[Any]:
    return [r for r in refs if start_time <= r.timestamp <= end_time]


def _resolve_split_refs(reader: object, data_cfg: Dict[str, Any], split_name: str) -> List[Any]:
    from ocean_forecast.data.zip_reader import parse_datetime64

    train_start = parse_datetime64(str(data_cfg["train_start"]))
    train_main_end = parse_datetime64(str(data_cfg["train_main_end"]))
    dev_start = parse_datetime64(str(data_cfg["dev_start"]))
    dev_end = parse_datetime64(str(data_cfg["dev_end"]))
    test_start = parse_datetime64(str(data_cfg["test_start"]))
    test_end = parse_datetime64(str(data_cfg["test_end"]))

    refs = reader.build_index(start_time=min(train_start, dev_start, test_start), end_time=max(train_main_end, dev_end, test_end))  # type: ignore[attr-defined]
    if split_name == "train":
        return _select_refs_by_range(refs, train_start, train_main_end)
    if split_name == "dev":
        return _select_refs_by_range(refs, dev_start, dev_end)
    return _select_refs_by_range(refs, test_start, test_end)


def _sample_refs(refs: Sequence[Any], max_frames: int, min_hour_gap: int) -> List[Any]:
    if not refs:
        return []
    sampled: List[Any] = []
    last_hour = None
    for ref in refs:
        h = int(ref.hour_index)
        if last_hour is None or (h - int(last_hour)) >= int(min_hour_gap):
            sampled.append(ref)
            last_hour = h
    if len(sampled) <= int(max_frames):
        return sampled
    idx = np.linspace(0, len(sampled) - 1, num=int(max_frames), dtype=np.int64)
    return [sampled[int(i)] for i in idx]


def _masked_values(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(field)
    return np.asarray(field[valid], dtype=np.float64)


def _safe_quantile(vals: np.ndarray, q: float, default: float = 0.0) -> float:
    if vals.size == 0:
        return float(default)
    return float(np.quantile(vals, q))


def _nan_fill(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = _masked_values(field, mask)
    fill = float(np.mean(vals)) if vals.size > 0 else 0.0
    out = np.asarray(field, dtype=np.float32).copy()
    out[~np.isfinite(out)] = fill
    out[~np.asarray(mask, dtype=bool)] = fill
    return out


def _gradient_mag(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    f = _nan_fill(field, mask)
    gy, gx = np.gradient(f)
    mag = np.sqrt(np.square(gx) + np.square(gy)).astype(np.float32, copy=False)
    mag[~np.asarray(mask, dtype=bool)] = np.nan
    return mag


def _uv_diagnostics(u: np.ndarray, v: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_fill = _nan_fill(u, mask)
    v_fill = _nan_fill(v, mask)
    du_dy, du_dx = np.gradient(u_fill)
    dv_dy, dv_dx = np.gradient(v_fill)
    vort = (dv_dx - du_dy).astype(np.float32, copy=False)
    sn = du_dx - dv_dy
    ss = dv_dx + du_dy
    ow = (np.square(sn) + np.square(ss) - np.square(vort)).astype(np.float32, copy=False)
    speed = np.sqrt(np.square(u_fill) + np.square(v_fill)).astype(np.float32, copy=False)
    invalid = ~np.asarray(mask, dtype=bool)
    vort[invalid] = np.nan
    ow[invalid] = np.nan
    speed[invalid] = np.nan
    return vort, ow, speed


def _load_frame_diags(reader: object, refs: Sequence[Any]) -> List[FrameDiag]:
    out: List[FrameDiag] = []
    for ref in refs:
        frame, valid = reader.read_frame(ref)  # type: ignore[attr-defined]
        mask = np.asarray(valid, dtype=bool)
        sst = np.asarray(frame[0], dtype=np.float32)
        sss = np.asarray(frame[1], dtype=np.float32)
        u = np.asarray(frame[2], dtype=np.float32)
        v = np.asarray(frame[3], dtype=np.float32)
        grad_sst = _gradient_mag(sst, mask)
        grad_sss = _gradient_mag(sss, mask)
        vort, ow, speed = _uv_diagnostics(u, v, mask)
        y, m = _ym(ref.timestamp)
        out.append(
            FrameDiag(
                ref=ref,
                timestamp=ref.timestamp,
                year=y,
                month=m,
                season=_season(m),
                mask=mask,
                maps={
                    "sst": sst,
                    "sss": sss,
                    "u": u,
                    "v": v,
                    "speed": speed,
                    "grad_sst": grad_sst,
                    "grad_sss": grad_sss,
                    "vorticity": vort,
                    "ow": ow,
                },
            )
        )
    return out


def _stats_basic(vals: np.ndarray) -> Dict[str, float]:
    if vals.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan"), "q01": float("nan"), "q50": float("nan"), "q99": float("nan")}
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "q01": float(np.quantile(vals, 0.01)),
        "q50": float(np.quantile(vals, 0.50)),
        "q99": float(np.quantile(vals, 0.99)),
    }


def _raw_health(diags: Sequence[FrameDiag]) -> Dict[str, Any]:
    channels = ("sst", "sss", "u", "v")
    vals: Dict[str, List[np.ndarray]] = {k: [] for k in channels}
    mon_mean: Dict[str, Dict[int, List[float]]] = {k: {m: [] for m in range(1, 13)} for k in channels}
    yr_mean: Dict[str, Dict[int, List[float]]] = {k: {} for k in channels}
    miss = {k: 0 for k in channels}
    total = 0
    ocean_ratio = []
    speed_vals: List[np.ndarray] = []
    dir_vals: List[np.ndarray] = []

    for d in diags:
        total += int(d.mask.size)
        ocean_ratio.append(float(np.mean(d.mask.astype(np.float32))))
        for ch in channels:
            arr = np.asarray(d.maps[ch], dtype=np.float32)
            miss[ch] += int(np.sum(~np.isfinite(arr)))
            sea = _masked_values(arr, d.mask)
            if sea.size > 0:
                vals[ch].append(sea)
                mon_mean[ch][d.month].append(float(np.mean(sea)))
                yr_mean[ch].setdefault(d.year, []).append(float(np.mean(sea)))
        sp = _masked_values(d.maps["speed"], d.mask)
        if sp.size > 0:
            speed_vals.append(sp)
        uv_ok = d.mask & np.isfinite(d.maps["u"]) & np.isfinite(d.maps["v"])
        if np.any(uv_ok):
            dir_vals.append(np.arctan2(d.maps["v"][uv_ok], d.maps["u"][uv_ok]).astype(np.float64, copy=False))

    ch_stats = {}
    for ch in channels:
        all_vals = np.concatenate(vals[ch], axis=0) if vals[ch] else np.zeros((0,), dtype=np.float64)
        ch_stats[ch] = _stats_basic(all_vals)
        ch_stats[ch]["missing_ratio"] = float(miss[ch]) / float(max(total, 1))

    speed_all = np.concatenate(speed_vals, axis=0) if speed_vals else np.zeros((0,), dtype=np.float64)
    dir_all = np.concatenate(dir_vals, axis=0) if dir_vals else np.zeros((0,), dtype=np.float64)
    dir_hist, dir_edges = np.histogram(dir_all, bins=36, range=(-math.pi, math.pi)) if dir_all.size > 0 else (np.zeros((36,), dtype=np.int64), np.linspace(-math.pi, math.pi, 37))
    sp_hist, sp_edges = np.histogram(speed_all, bins=40) if speed_all.size > 0 else (np.zeros((40,), dtype=np.int64), np.linspace(0.0, 1.0, 41))

    monthly = {ch: {str(m): float(np.mean(mon_mean[ch][m])) if mon_mean[ch][m] else float("nan") for m in range(1, 13)} for ch in channels}
    yearly = {ch: {str(y): float(np.mean(v)) for y, v in sorted(yr_mean[ch].items())} for ch in channels}
    sst_mon = np.array([monthly["sst"][str(m)] for m in range(1, 13)], dtype=np.float64)
    sss_mon = np.array([monthly["sss"][str(m)] for m in range(1, 13)], dtype=np.float64)
    seas = {
        "sst": float((np.nanmax(sst_mon) - np.nanmin(sst_mon)) / max(float(np.nanstd(sst_mon)), 1e-8)),
        "sss": float((np.nanmax(sss_mon) - np.nanmin(sss_mon)) / max(float(np.nanstd(sss_mon)), 1e-8)),
    }
    return {
        "channel_stats": ch_stats,
        "land_ratio": float(1.0 - np.mean(ocean_ratio)) if ocean_ratio else float("nan"),
        "monthly_mean": monthly,
        "yearly_mean": yearly,
        "speed_stats": _stats_basic(speed_all),
        "u_v_direction_hist": {"counts": dir_hist.tolist(), "bin_edges": dir_edges.tolist()},
        "speed_hist": {"counts": sp_hist.tolist(), "bin_edges": sp_edges.tolist()},
        "seasonality_strength": seas,
    }


def _maybe_import_matplotlib(skip_plots: bool):
    if skip_plots:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _plot_raw_health(raw: Dict[str, Any], out_dir: Path, plt_mod: Any) -> Dict[str, Any]:
    out = {"enabled": bool(plt_mod is not None), "files": []}
    if plt_mod is None:
        out["reason"] = "plot skipped or matplotlib unavailable"
        return out
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt_mod.subplots(2, 2, figsize=(12, 8))
    for i, ch in enumerate(("sst", "sss", "u", "v")):
        ax = axes[i // 2, i % 2]
        series = [raw["monthly_mean"][ch][str(m)] for m in range(1, 13)]
        arr = np.asarray(series, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            ax.hist(arr, bins=20, alpha=0.8)
        ax.set_title(f"{ch.upper()} monthly-mean histogram")
    fig.tight_layout()
    p1 = out_dir / "channel_histograms.png"
    fig.savefig(p1, dpi=150)
    plt_mod.close(fig)
    out["files"].append(str(p1))

    fig, axes = plt_mod.subplots(2, 2, figsize=(12, 8))
    months = list(range(1, 13))
    for i, ch in enumerate(("sst", "sss", "u", "v")):
        ax = axes[i // 2, i % 2]
        vals = [raw["monthly_mean"][ch][str(m)] for m in months]
        ax.boxplot([[v] for v in vals], showfliers=False)
        ax.set_xticks(months)
        ax.set_title(f"{ch.upper()} monthly box")
    fig.tight_layout()
    p2 = out_dir / "monthly_boxplots.png"
    fig.savefig(p2, dpi=150)
    plt_mod.close(fig)
    out["files"].append(str(p2))

    fig, axes = plt_mod.subplots(1, 2, figsize=(12, 4))
    dir_hist = np.asarray(raw["u_v_direction_hist"]["counts"], dtype=np.float64)
    dir_edges = np.asarray(raw["u_v_direction_hist"]["bin_edges"], dtype=np.float64)
    sp_hist = np.asarray(raw["speed_hist"]["counts"], dtype=np.float64)
    sp_edges = np.asarray(raw["speed_hist"]["bin_edges"], dtype=np.float64)
    axes[0].bar(0.5 * (dir_edges[:-1] + dir_edges[1:]), dir_hist, width=np.diff(dir_edges), alpha=0.8)
    axes[0].set_title("u/v direction")
    axes[1].bar(0.5 * (sp_edges[:-1] + sp_edges[1:]), sp_hist, width=np.diff(sp_edges), alpha=0.8)
    axes[1].set_title("speed")
    fig.tight_layout()
    p3 = out_dir / "uv_speed_hist.png"
    fig.savefig(p3, dpi=150)
    plt_mod.close(fig)
    out["files"].append(str(p3))
    return out


def _plot_spatial_maps(diags: Sequence[FrameDiag], out_dir: Path, plt_mod: Any, seed: int) -> Dict[str, Any]:
    out = {"enabled": bool(plt_mod is not None), "samples": []}
    if plt_mod is None or not diags:
        out["reason"] = "plot skipped or matplotlib unavailable"
        return out
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    idx = sorted(rng.sample(list(range(len(diags))), k=min(8, len(diags))))
    fields = ["sst", "sss", "u", "v", "speed", "grad_sst", "grad_sss", "vorticity", "ow"]
    for j, i in enumerate(idx):
        d = diags[int(i)]
        fig, axes = plt_mod.subplots(3, 3, figsize=(12, 10))
        for k, key in enumerate(fields):
            arr = np.asarray(d.maps[key], dtype=np.float32)
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                vmin, vmax = float(np.quantile(finite, 0.01)), float(np.quantile(finite, 0.99))
                if vmin >= vmax:
                    vmin, vmax = float(np.min(finite)), float(np.max(finite))
            else:
                vmin, vmax = 0.0, 1.0
            cmap = "coolwarm" if key in ("u", "v", "vorticity", "ow") else "viridis"
            ax = axes[k // 3, k % 3]
            im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(key)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(str(d.timestamp))
        fig.tight_layout()
        path = out_dir / f"sample_{j:02d}.png"
        fig.savefig(path, dpi=140)
        plt_mod.close(fig)
        out["samples"].append({"index": int(i), "timestamp": str(d.timestamp), "path": str(path)})
    return out


def _shift_bool(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    sy0 = max(0, -dy)
    sy1 = min(h, h - dy) if dy >= 0 else h
    dy0 = max(0, dy)
    dy1 = min(h, h + dy) if dy <= 0 else h
    sx0 = max(0, -dx)
    sx1 = min(w, w - dx) if dx >= 0 else w
    dx0 = max(0, dx)
    dx1 = min(w, w + dx) if dx <= 0 else w
    if sy1 <= sy0 or sx1 <= sx0:
        return out
    out[dy0:dy1, dx0:dx1] = mask[sy0:sy1, sx0:sx1]
    return out


def _best_shift_iou(mask_a: np.ndarray, mask_b: np.ndarray, valid: np.ndarray, max_shift: int) -> Tuple[float, float]:
    best_iou = float("nan")
    best_shift = float("nan")
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = _shift_bool(mask_b, dy, dx)
            inter = np.sum(mask_a & shifted & valid)
            union = np.sum((mask_a | shifted) & valid)
            if union <= 0:
                continue
            iou = float(inter) / float(union)
            if (not np.isfinite(best_iou)) or (iou > best_iou):
                best_iou = iou
                best_shift = math.sqrt(float(dy * dy + dx * dx))
    if not np.isfinite(best_iou):
        return 0.0, float(max_shift)
    return best_iou, best_shift


def _candidate_specs() -> List[Dict[str, Any]]:
    return [
        {"name": "front_grad_sst_q90", "structure": "front", "family": "front_alt", "mode": "front_single", "source": "grad_sst", "q": 0.90},
        {"name": "front_grad_sss_q90", "structure": "front", "family": "front_alt", "mode": "front_single", "source": "grad_sss", "q": 0.90},
        {"name": "front_or_q90", "structure": "front", "family": "front_alt", "mode": "front_or", "q": 0.90},
        {"name": "front_combined_q85", "structure": "front", "family": "front_thr", "mode": "front_combined", "q": 0.85, "alpha": 0.6, "beta": 0.4},
        {"name": "front_combined_q90", "structure": "front", "family": "front_thr", "mode": "front_combined", "q": 0.90, "alpha": 0.6, "beta": 0.4},
        {"name": "front_combined_q95", "structure": "front", "family": "front_thr", "mode": "front_combined", "q": 0.95, "alpha": 0.6, "beta": 0.4},
        {"name": "eddy_vort_q85", "structure": "eddy", "family": "eddy_alt", "mode": "eddy_vort", "vort_q": 0.85},
        {"name": "eddy_vort_q85_ow_q20", "structure": "eddy", "family": "eddy_thr", "mode": "eddy_vort_ow", "vort_q": 0.85, "ow_q": 0.20},
        {"name": "eddy_vort_q90_ow_q15", "structure": "eddy", "family": "eddy_thr", "mode": "eddy_vort_ow", "vort_q": 0.90, "ow_q": 0.15},
        {"name": "strong_current_q80", "structure": "strong_current", "family": "cur_thr", "mode": "current", "q": 0.80},
        {"name": "strong_current_q85", "structure": "strong_current", "family": "cur_thr", "mode": "current", "q": 0.85},
        {"name": "strong_current_q90", "structure": "strong_current", "family": "cur_thr", "mode": "current", "q": 0.90},
    ]


def _candidate_signal(d: FrameDiag, spec: Dict[str, Any]) -> np.ndarray:
    mode = str(spec["mode"])
    if mode == "front_single":
        return np.asarray(d.maps[str(spec["source"])], dtype=np.float32)
    if mode == "front_or":
        return np.maximum(np.asarray(d.maps["grad_sst"], dtype=np.float32), np.asarray(d.maps["grad_sss"], dtype=np.float32))
    if mode == "front_combined":
        a = float(spec.get("alpha", 0.5))
        b = float(spec.get("beta", 0.5))
        return a * np.asarray(d.maps["grad_sst"], dtype=np.float32) + b * np.asarray(d.maps["grad_sss"], dtype=np.float32)
    if mode in ("eddy_vort", "eddy_vort_ow"):
        return np.abs(np.asarray(d.maps["vorticity"], dtype=np.float32))
    return np.asarray(d.maps["speed"], dtype=np.float32)


def _candidate_mask(d: FrameDiag, spec: Dict[str, Any]) -> np.ndarray:
    valid = d.mask
    mode = str(spec["mode"])
    if mode == "front_single":
        src = np.asarray(d.maps[str(spec["source"])], dtype=np.float32)
        thr = _safe_quantile(_masked_values(src, valid), float(spec["q"]))
        return (src >= thr) & valid
    if mode == "front_or":
        sst = np.asarray(d.maps["grad_sst"], dtype=np.float32)
        sss = np.asarray(d.maps["grad_sss"], dtype=np.float32)
        q = float(spec["q"])
        t1 = _safe_quantile(_masked_values(sst, valid), q)
        t2 = _safe_quantile(_masked_values(sss, valid), q)
        return ((sst >= t1) | (sss >= t2)) & valid
    if mode == "front_combined":
        sig = _candidate_signal(d, spec)
        thr = _safe_quantile(_masked_values(sig, valid), float(spec["q"]))
        return (sig >= thr) & valid
    if mode == "eddy_vort":
        vort = np.abs(np.asarray(d.maps["vorticity"], dtype=np.float32))
        tv = _safe_quantile(_masked_values(vort, valid), float(spec["vort_q"]))
        return (vort >= tv) & valid
    if mode == "eddy_vort_ow":
        vort = np.abs(np.asarray(d.maps["vorticity"], dtype=np.float32))
        ow = np.asarray(d.maps["ow"], dtype=np.float32)
        tv = _safe_quantile(_masked_values(vort, valid), float(spec["vort_q"]))
        to = _safe_quantile(_masked_values(ow, valid), float(spec["ow_q"]))
        return (vort >= tv) & (ow <= to) & valid
    speed = np.asarray(d.maps["speed"], dtype=np.float32)
    ts = _safe_quantile(_masked_values(speed, valid), float(spec["q"]))
    return (speed >= ts) & valid


def _perimeter(mask: np.ndarray, valid: np.ndarray) -> float:
    m = mask & valid
    up = np.zeros_like(m)
    up[1:, :] = m[:-1, :]
    dn = np.zeros_like(m)
    dn[:-1, :] = m[1:, :]
    lf = np.zeros_like(m)
    lf[:, 1:] = m[:, :-1]
    rt = np.zeros_like(m)
    rt[:, :-1] = m[:, 1:]
    return float((m & ~up).sum() + (m & ~dn).sum() + (m & ~lf).sum() + (m & ~rt).sum())


def _ecc(mask: np.ndarray, valid: np.ndarray) -> float:
    pts = np.argwhere(mask & valid)
    if pts.shape[0] < 5:
        return 0.0
    centered = pts.astype(np.float64) - np.mean(pts.astype(np.float64), axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eig = np.sort(np.maximum(np.linalg.eigvalsh(cov), 1e-12))
    return float(np.sqrt(max(0.0, 1.0 - float(eig[0] / eig[-1]))))


def _component_sizes(mask: np.ndarray) -> List[int]:
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    pts = np.argwhere(mask)
    sizes: List[int] = []
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for yx in pts:
        y, x = int(yx[0]), int(yx[1])
        if vis[y, x]:
            continue
        st = [(y, x)]
        vis[y, x] = True
        cnt = 0
        while st:
            cy, cx = st.pop()
            cnt += 1
            for dy, dx in neigh:
                ny, nx = cy + dy, cx + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if vis[ny, nx] or (not mask[ny, nx]):
                    continue
                vis[ny, nx] = True
                st.append((ny, nx))
        sizes.append(cnt)
    return sizes


def _evaluate_candidates(diags: Sequence[FrameDiag], specs: Sequence[Dict[str, Any]], max_shift: int) -> Dict[str, Dict[str, Any]]:
    if not diags:
        return {}
    valid_ref = np.asarray(diags[0].mask, dtype=bool)
    out: Dict[str, Dict[str, Any]] = {}
    for spec in specs:
        masks: List[np.ndarray] = []
        areas: List[float] = []
        perims: List[float] = []
        eccs: List[float] = []
        strength: List[float] = []
        contrast: List[float] = []
        area_ratio: List[float] = []
        months: List[int] = []
        seasons: List[str] = []
        for d in diags:
            sig = _candidate_signal(d, spec)
            m = _candidate_mask(d, spec)
            vals = _masked_values(sig, d.mask)
            p90 = _safe_quantile(vals, 0.90)
            p50 = _safe_quantile(vals, 0.50)
            ms = m & d.mask
            in_vals = sig[ms & np.isfinite(sig)]
            out_vals = sig[(~ms) & d.mask & np.isfinite(sig)]
            c = float(np.mean(in_vals)) / max(float(np.mean(out_vals)) if out_vals.size > 0 else 1e-8, 1e-8)
            a = float(np.sum(ms))
            ar = a / max(float(np.sum(d.mask)), 1.0)
            masks.append(ms)
            areas.append(a)
            perims.append(_perimeter(ms, d.mask))
            eccs.append(_ecc(ms, d.mask))
            strength.append(float(p90 / max(p50, 1e-8)))
            contrast.append(c)
            area_ratio.append(ar)
            months.append(int(d.month))
            seasons.append(str(d.season))

        ious, shifts, pers, births, deaths, defs = [], [], [], [], [], []
        for i in range(len(masks) - 1):
            a = masks[i] & valid_ref
            b = masks[i + 1] & valid_ref
            iou, shift = _best_shift_iou(a, b, valid_ref, max_shift=max_shift)
            inter = float(np.sum(a & b))
            aa = float(np.sum(a))
            bb = float(np.sum(b))
            ious.append(iou)
            shifts.append(shift)
            pers.append(inter / max(aa, 1.0))
            births.append(float(np.sum((~a) & b)) / max(bb, 1.0))
            deaths.append(float(np.sum(a & (~b))) / max(aa, 1.0))
            defs.append(0.5 * (abs(perims[i + 1] - perims[i]) / max(perims[i], 1e-8) + abs(eccs[i + 1] - eccs[i])))

        idx = np.linspace(0, len(masks) - 1, num=min(30, len(masks)), dtype=np.int64)
        comp_n, comp_l, comp_f = [], [], []
        for j in idx:
            mm = masks[int(j)][::2, ::2] & valid_ref[::2, ::2]
            area = int(np.sum(mm))
            if area <= 0:
                comp_n.append(0.0)
                comp_l.append(0.0)
                comp_f.append(0.0)
                continue
            sz = _component_sizes(mm)
            comp_n.append(float(len(sz)))
            comp_l.append(float(max(sz)) / float(area))
            comp_f.append(float(len(sz)) / float(area))

        s_strength = float(np.median(strength)) if strength else 0.0
        s_area = float(np.mean(area_ratio)) if area_ratio else 0.0
        s_iou = float(np.mean(ious)) if ious else 0.0
        s_temporal = float(np.clip(0.35 * s_iou + 0.25 * (float(np.mean(pers)) if pers else 0.0) + 0.15 * (1.0 - min(float(np.mean(births)) if births else 0.0, 1.0)) + 0.15 * (1.0 - min(float(np.mean(deaths)) if deaths else 0.0, 1.0)) + 0.10 * (1.0 - min(float(np.mean(defs)) if defs else 0.0, 1.0)), 0.0, 1.0))
        s_conn = float(np.clip(0.7 * (float(np.mean(comp_l)) if comp_l else 0.0) + 0.3 * (1.0 - min((float(np.mean(comp_f)) if comp_f else 0.0) * 40.0, 1.0)), 0.0, 1.0))
        s_single = float(np.clip(0.45 * np.clip((s_strength - 1.2) / 2.5, 0.0, 1.0) + 0.25 * np.exp(-((s_area - (0.12 if spec["structure"] == "front" else (0.08 if spec["structure"] == "eddy" else 0.15))) ** 2) / (2 * 0.08 * 0.08)) + 0.20 * s_conn + 0.10 * np.clip(float(np.median(contrast)) / 4.0, 0.0, 1.0), 0.0, 1.0))
        s_final = float(np.clip(0.55 * s_single + 0.45 * s_temporal, 0.0, 1.0))

        out[str(spec["name"])] = {
            "spec": dict(spec),
            "single": {"strength_ratio_p90_p50": s_strength, "area_ratio": s_area, "contrast_ratio_in_out": float(np.median(contrast)) if contrast else 0.0},
            "connectivity": {"num_components": float(np.mean(comp_n)) if comp_n else 0.0, "largest_component_ratio": float(np.mean(comp_l)) if comp_l else 0.0, "fragmentation_index": float(np.mean(comp_f)) if comp_f else 0.0},
            "temporal": {"temporal_iou": s_iou, "mean_shift_pixels": float(np.mean(shifts)) if shifts else float(max_shift), "persistence_rate": float(np.mean(pers)) if pers else 0.0, "birth_rate": float(np.mean(births)) if births else 0.0, "death_rate": float(np.mean(deaths)) if deaths else 0.0, "deformation_rate": float(np.mean(defs)) if defs else 0.0},
            "scores": {"single_frame_score": s_single, "temporal_score": s_temporal, "connectivity_score": s_conn, "final_score": s_final},
            "grade": "A" if s_final >= 0.78 else ("B" if s_final >= 0.62 else ("C" if s_final >= 0.46 else "D")),
            "_internal": {"masks": masks, "months": months, "seasons": seasons, "area_ratio": area_ratio},
        }
    return out


def _best_by_structure(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for name, row in results.items():
        s = str(row["spec"]["structure"])
        if s not in best or float(row["scores"]["final_score"]) > float(best[s]["scores"]["final_score"]):
            best[s] = {"name": name, **row}
    return best


def _cv_score(values: List[float], norm: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    cv = float(np.std(arr)) / max(abs(float(np.mean(arr))), 1e-8)
    return float(np.clip(1.0 - np.clip(cv / max(norm, 1e-8), 0.0, 1.0), 0.0, 1.0))


def _threshold_robustness(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    fam = {
        "front": ["front_combined_q85", "front_combined_q90", "front_combined_q95"],
        "eddy": ["eddy_vort_q85_ow_q20", "eddy_vort_q90_ow_q15"],
        "strong_current": ["strong_current_q80", "strong_current_q85", "strong_current_q90"],
    }
    out = {}
    for s, names in fam.items():
        rows = [results[n] for n in names if n in results]
        if not rows:
            out[s] = {"score": 0.0}
            continue
        sc = _cv_score([float(r["scores"]["final_score"]) for r in rows], norm=0.6)
        ar = _cv_score([float(r["single"]["area_ratio"]) for r in rows], norm=0.8)
        io = _cv_score([float(r["temporal"]["temporal_iou"]) for r in rows], norm=0.8)
        co = _cv_score([float(r["connectivity"]["largest_component_ratio"]) for r in rows], norm=0.8)
        out[s] = {"score": float(np.clip(0.30 * sc + 0.25 * ar + 0.25 * io + 0.20 * co, 0.0, 1.0))}
    return out


def _seasonal_analysis(best: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for s, row in best.items():
        internal = row["_internal"]
        by = {}
        for sea in ("DJF", "MAM", "JJA", "SON"):
            idx = [i for i, v in enumerate(internal["seasons"]) if v == sea]
            by[sea] = float(np.mean([internal["area_ratio"][i] for i in idx])) if idx else float("nan")
        out[s] = {"area_ratio_by_season": by, "consistency_score": _cv_score([v for v in by.values() if np.isfinite(v)], norm=0.8)}
    return out


def _regional_analysis(best: Dict[str, Dict[str, Any]], valid: np.ndarray) -> Dict[str, Any]:
    h, w = valid.shape
    ys = np.array_split(np.arange(h), 3)
    xs = np.array_split(np.arange(w), 3)
    reg_masks = {}
    for i, y in enumerate(ys):
        m = np.zeros((h, w), dtype=bool)
        m[y, :] = True
        reg_masks[f"lat_band_{i}"] = m & valid
    for i, x in enumerate(xs):
        m = np.zeros((h, w), dtype=bool)
        m[:, x] = True
        reg_masks[f"lon_band_{i}"] = m & valid

    out = {}
    for s, row in best.items():
        masks = row["_internal"]["masks"]
        ratios = {}
        for rn, rm in reg_masks.items():
            den = float(np.sum(rm))
            if den <= 0:
                ratios[rn] = float("nan")
                continue
            ratios[rn] = float(np.mean([float(np.sum(m & rm)) / den for m in masks]))
        out[s] = {"area_ratio_by_region": ratios, "consistency_score": _cv_score([v for v in ratios.values() if np.isfinite(v)], norm=0.8)}
    return out


def _timescale_analysis(reader: object, all_refs: Sequence[Any], best: Dict[str, Dict[str, Any]], scales: Sequence[int], max_frames_per_scale: int, max_shift: int) -> Dict[str, Any]:
    specs = [v["spec"] for v in best.values()]
    by_scale = {}
    for sc in scales:
        refs = _sample_refs(all_refs, max_frames=max_frames_per_scale, min_hour_gap=int(sc))
        diags = _load_frame_diags(reader, refs)
        rr = _evaluate_candidates(diags, specs, max_shift=max_shift)
        by_scale[str(sc)] = {k: {"final_score": float(v["scores"]["final_score"]), "temporal_iou": float(v["temporal"]["temporal_iou"]), "area_ratio": float(v["single"]["area_ratio"]), "mean_shift_pixels": float(v["temporal"]["mean_shift_pixels"])} for k, v in rr.items()}
    cons = {}
    for s, row in best.items():
        name = row["name"]
        vals = [by_scale[str(sc)][name]["final_score"] for sc in scales if name in by_scale[str(sc)]]
        ious = [by_scale[str(sc)][name]["temporal_iou"] for sc in scales if name in by_scale[str(sc)]]
        areas = [by_scale[str(sc)][name]["area_ratio"] for sc in scales if name in by_scale[str(sc)]]
        cons[s] = float(np.clip(0.4 * _cv_score(vals, 0.6) + 0.35 * _cv_score(ious, 0.8) + 0.25 * _cv_score(areas, 0.8), 0.0, 1.0))
    return {"by_scale": by_scale, "consistency_score": cons}


def _multi_metric_table(results: Dict[str, Dict[str, Any]], thr: Dict[str, Any], ts: Dict[str, Any], sea: Dict[str, Any], reg: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for name, row in results.items():
        s = str(row["spec"]["structure"])
        robust = float(thr.get(s, {}).get("score", 0.0))
        tsc = float(ts["consistency_score"].get(s, 0.0))
        sec = float(sea.get(s, {}).get("consistency_score", 0.0))
        rec = float(reg.get(s, {}).get("consistency_score", 0.0))
        overall = float(np.clip(0.45 * float(row["scores"]["final_score"]) + 0.15 * robust + 0.15 * tsc + 0.125 * sec + 0.125 * rec, 0.0, 1.0))
        rows.append({"candidate": name, "structure": s, "single_frame_score": float(row["scores"]["single_frame_score"]), "temporal_score": float(row["scores"]["temporal_score"]), "connectivity_score": float(row["scores"]["connectivity_score"]), "threshold_robustness": robust, "timescale_consistency": tsc, "seasonal_consistency": sec, "regional_consistency": rec, "overall_score": overall, "grade": ("A" if overall >= 0.78 else ("B" if overall >= 0.62 else ("C" if overall >= 0.46 else "D")))})
    rows.sort(key=lambda x: x["overall_score"], reverse=True)
    return rows


def _three_sentences(best: Dict[str, Dict[str, Any]], thr: Dict[str, Any], ts: Dict[str, Any], sea: Dict[str, Any], reg: Dict[str, Any]) -> Dict[str, str]:
    if not best:
        return {"clearest_structure": "No valid result.", "most_continuous_structure": "No valid result.", "most_robust_structure": "No valid result."}
    clear_s = max(best.items(), key=lambda kv: float(kv[1]["scores"]["single_frame_score"]))[0]
    cont_s = max(best.items(), key=lambda kv: float(kv[1]["scores"]["temporal_score"]))[0]
    robust = {}
    for s in best.keys():
        robust[s] = float(np.mean([float(thr.get(s, {}).get("score", 0.0)), float(ts["consistency_score"].get(s, 0.0)), float(sea.get(s, {}).get("consistency_score", 0.0)), float(reg.get(s, {}).get("consistency_score", 0.0))]))
    rob_s = max(robust.items(), key=lambda kv: kv[1])[0]
    return {"clearest_structure": f"The clearest structure is {clear_s}.", "most_continuous_structure": f"The most continuous structure is {cont_s}.", "most_robust_structure": f"The most robust structure across thresholds/seasons/regions/time-scales is {rob_s}."}


def _serializable_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for k, v in results.items():
        out[k] = {"spec": v["spec"], "single": v["single"], "connectivity": v["connectivity"], "temporal": v["temporal"], "scores": v["scores"], "grade": v["grade"]}
    return out


def _to_md(payload: Dict[str, Any]) -> str:
    meta = payload["meta"]
    best = payload["best_by_structure"]
    table = payload["multi_metric_table"]
    lines = [
        "# Structure Analysis V2",
        "",
        f"- split: `{meta['split']}`",
        f"- sampled_frames: `{meta['sampled_frames']}`",
        f"- source_refs: `{meta['source_refs']}`",
        f"- time_scales: `{meta['time_scales']}`",
        "",
        "## Best Per Structure",
    ]
    for s, row in best.items():
        lines.append(f"- `{s}` -> `{row['name']}` | final_score={row['scores']['final_score']:.3f} | grade={row['grade']}")
    lines.extend(["", "## Top 10 Multi-Metric", "", "| candidate | structure | single | temporal | conn | robust | scale | season | region | overall | grade |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
    for r in table[:10]:
        lines.append(f"| {r['candidate']} | {r['structure']} | {r['single_frame_score']:.3f} | {r['temporal_score']:.3f} | {r['connectivity_score']:.3f} | {r['threshold_robustness']:.3f} | {r['timescale_consistency']:.3f} | {r['seasonal_consistency']:.3f} | {r['regional_consistency']:.3f} | {r['overall_score']:.3f} | {r['grade']} |")
    lines.extend(["", "## Final Three Sentences", f"- {payload['final_three_sentences']['clearest_structure']}", f"- {payload['final_three_sentences']['most_continuous_structure']}", f"- {payload['final_three_sentences']['most_robust_structure']}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    years = sorted(set(list(data_cfg.get("train_years", [])) + list(data_cfg.get("test_years", []))))
    try:
        from ocean_forecast.data.reader_factory import build_reader
    except ModuleNotFoundError as exc:
        raise RuntimeError("Failed to import readers. Install at least h5py and numpy.") from exc

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(args.plot_dir)
    scales = _parse_time_scales(args.time_scales)
    plt_mod = _maybe_import_matplotlib(skip_plots=bool(args.skip_plots))

    reader = build_reader(data_cfg=data_cfg, years=years)
    try:
        split_refs = _resolve_split_refs(reader, data_cfg=data_cfg, split_name=args.split)
        sampled_refs = _sample_refs(split_refs, max_frames=int(args.max_frames), min_hour_gap=int(args.min_hour_gap))
        if not sampled_refs:
            raise RuntimeError("No sampled refs. Check ranges/min_hour_gap.")
        diags = _load_frame_diags(reader, sampled_refs)
        raw = _raw_health(diags)
        raw_plot = _plot_raw_health(raw, out_dir=plot_dir / "raw_health", plt_mod=plt_mod)
        spatial_plot = _plot_spatial_maps(diags, out_dir=plot_dir / "spatial_maps", plt_mod=plt_mod, seed=int(args.seed))
        specs = _candidate_specs()
        results = _evaluate_candidates(diags, specs, max_shift=int(args.max_shift))
        best = _best_by_structure(results)
        valid_ref = np.asarray(diags[0].mask, dtype=bool)
        thr = _threshold_robustness(results)
        sea = _seasonal_analysis(best)
        reg = _regional_analysis(best, valid=valid_ref)
        ts = _timescale_analysis(reader, split_refs, best, scales=scales, max_frames_per_scale=int(args.max_frames_per_scale), max_shift=int(args.max_shift))
        table = _multi_metric_table(results, thr=thr, ts=ts, sea=sea, reg=reg)
        three = _three_sentences(best, thr=thr, ts=ts, sea=sea, reg=reg)

        payload = {
            "meta": {"config": str(Path(args.config)), "split": str(args.split), "source_refs": int(len(split_refs)), "sampled_frames": int(len(sampled_refs)), "min_hour_gap": int(args.min_hour_gap), "time_scales": scales, "max_shift": int(args.max_shift)},
            "raw_health": raw,
            "plot_outputs": {"raw_health": raw_plot, "spatial_structure": spatial_plot},
            "candidate_library": specs,
            "candidate_results": _serializable_results(results),
            "best_by_structure": {k: {"name": v["name"], "spec": v["spec"], "single": v["single"], "connectivity": v["connectivity"], "temporal": v["temporal"], "scores": v["scores"], "grade": v["grade"]} for k, v in best.items()},
            "threshold_robustness": thr,
            "timescale_analysis": ts,
            "seasonal_analysis": sea,
            "regional_analysis": reg,
            "multi_metric_table": table,
            "final_three_sentences": three,
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with out_md.open("w", encoding="utf-8") as f:
            f.write(_to_md(payload))
        print(json.dumps({"saved_json": str(out_json), "saved_md": str(out_md), "sampled_frames": len(sampled_refs)}, ensure_ascii=False, indent=2))
        print(f"[analyze_structures_v2] saved: {out_json}")
        print(f"[analyze_structures_v2] saved: {out_md}")
    finally:
        close_fn = getattr(reader, "close", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    main()

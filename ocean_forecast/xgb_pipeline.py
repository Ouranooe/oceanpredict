from __future__ import annotations

import importlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .metrics import rmse_to_nrmse, summarize_channel_metrics, to_eval_channels_numpy

HOURS_PER_DAY = 24.0
HOURS_PER_YEAR = 24.0 * 366.0


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _validate_speed_bucket_edges(edges: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(edges, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("speed bucket edges must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("speed bucket edges must be finite numbers.")
    if not np.all(np.diff(arr) > 0):
        raise ValueError("speed bucket edges must be strictly increasing.")
    return arr


def _assign_speed_bins(speed_gt: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(bin_edges, speed_gt, side="right").astype(np.int64, copy=False)


def _speed_bucket_labels(bin_edges: np.ndarray) -> List[str]:
    labels: List[str] = []
    if bin_edges.size == 0:
        return labels
    labels.append(f"<{bin_edges[0]:g}")
    for i in range(bin_edges.size - 1):
        labels.append(f"{bin_edges[i]:g}-{bin_edges[i + 1]:g}")
    labels.append(f">={bin_edges[-1]:g}")
    return labels


def _append_speed_bucket_metrics(
    metrics: Dict[str, float],
    bucket_sum_sq: np.ndarray,
    bucket_count: np.ndarray,
    bucket_samples: np.ndarray,
    nrmse_denom: np.ndarray,
) -> None:
    speed_denom = float(max(float(nrmse_denom[2]), 1e-12))
    for i in range(int(bucket_sum_sq.size)):
        count = float(bucket_count[i])
        rmse = float(np.sqrt(bucket_sum_sq[i] / max(count, 1.0)))
        nrmse = float(rmse / speed_denom) if count > 0.0 else float("nan")
        metrics[f"speed_bucket_{i}_samples"] = float(bucket_samples[i])
        metrics[f"speed_bucket_{i}_rmse"] = rmse if count > 0.0 else float("nan")
        metrics[f"speed_bucket_{i}_nrmse"] = nrmse


def _masked_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask.astype(np.float32)[None, None, :, :]
    err2 = np.square(pred - target) * valid
    denom = float(np.sum(valid) * pred.shape[0] * pred.shape[1])
    if denom <= 0.0:
        return 0.0
    return float(np.sum(err2) / denom)


def _extract_recent_lags(series: np.ndarray, lag_steps: int) -> np.ndarray:
    if series.ndim != 3:
        raise ValueError(f"series must be [T,H,W], got {series.shape}")
    if lag_steps <= 0:
        raise ValueError(f"lag_steps must be > 0, got {lag_steps}")
    t = int(series.shape[0])
    start = max(0, t - lag_steps)
    selected = series[start:t]
    if selected.shape[0] == lag_steps:
        return selected.astype(np.float32, copy=False)
    pad = np.repeat(series[0:1], repeats=(lag_steps - selected.shape[0]), axis=0)
    return np.concatenate([pad, selected], axis=0).astype(np.float32, copy=False)


def _lat_lon_mesh(lat: np.ndarray, lon: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    lat_arr = np.asarray(lat, dtype=np.float32)
    lon_arr = np.asarray(lon, dtype=np.float32)

    if lat_arr.ndim == 1 and lon_arr.ndim == 1:
        if lat_arr.shape[0] != h or lon_arr.shape[0] != w:
            raise ValueError(
                f"lat/lon length mismatch: lat={lat_arr.shape}, lon={lon_arr.shape}, expected H={h}, W={w}"
            )
        lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
        return lat_grid.astype(np.float32, copy=False), lon_grid.astype(np.float32, copy=False)

    if lat_arr.ndim == 2 and lon_arr.ndim == 2:
        if lat_arr.shape != (h, w) or lon_arr.shape != (h, w):
            raise ValueError(
                f"lat/lon shape mismatch: lat={lat_arr.shape}, lon={lon_arr.shape}, expected {(h, w)}"
            )
        return lat_arr.astype(np.float32, copy=False), lon_arr.astype(np.float32, copy=False)

    raise ValueError(f"Unsupported lat/lon shapes: lat={lat_arr.shape}, lon={lon_arr.shape}")


def _time_features(hour_index: int) -> Tuple[float, float, float, float]:
    hour_phase = (2.0 * math.pi) * (float(hour_index) % HOURS_PER_DAY) / HOURS_PER_DAY
    year_phase = (2.0 * math.pi) * (float(hour_index) % HOURS_PER_YEAR) / HOURS_PER_YEAR
    return (
        float(math.sin(hour_phase)),
        float(math.cos(hour_phase)),
        float(math.sin(year_phase)),
        float(math.cos(year_phase)),
    )


class XGBPointwiseForecaster:
    def __init__(
        self,
        lag_steps: int = 6,
        train_row_cap: int = 2_000_000,
        model_params: Dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> None:
        self.lag_steps = int(lag_steps)
        self.train_row_cap = int(train_row_cap)
        self.random_state = int(random_state)
        self.model_params = self._normalized_model_params(model_params or {})
        self.models: List[Any] = []

        self.input_mean: np.ndarray | None = None
        self.input_std: np.ndarray | None = None
        self.target_mean: np.ndarray | None = None
        self.target_std: np.ndarray | None = None
        self.ocean_mask: np.ndarray | None = None
        self.lat: np.ndarray | None = None
        self.lon: np.ndarray | None = None
        self.feature_dim: int = 0

    @staticmethod
    def _normalized_model_params(overrides: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "device": "cpu",
        }
        params.update(overrides)
        if params.get("random_state") is None:
            params["random_state"] = 42
        return {k: v for k, v in params.items() if v is not None}

    @staticmethod
    def _require_xgboost() -> Any:
        try:
            return importlib.import_module("xgboost")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "xgboost is required for this pipeline. Install it first, e.g. `pip install xgboost`."
            ) from exc

    @property
    def target_channels(self) -> int:
        if self.target_mean is None:
            raise RuntimeError("Model context is not initialized.")
        return int(self.target_mean.shape[0])

    def _set_context_from_stats(self, stats: Dict[str, np.ndarray]) -> None:
        self.input_mean = np.asarray(stats["input_mean"], dtype=np.float32)
        self.input_std = np.asarray(stats["input_std"], dtype=np.float32)
        self.target_mean = np.asarray(stats["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(stats["target_std"], dtype=np.float32)
        self.ocean_mask = np.asarray(stats["ocean_mask"], dtype=np.float32)
        self.lat = np.asarray(stats["lat"], dtype=np.float32)
        self.lon = np.asarray(stats["lon"], dtype=np.float32)

    def _ensure_context_ready(self) -> None:
        if any(v is None for v in [self.input_mean, self.input_std, self.target_mean, self.target_std, self.ocean_mask, self.lat, self.lon]):
            raise RuntimeError("Forecaster context is incomplete. Fit or load a saved model first.")

    def _channel_input_series(self, x_raw: np.ndarray, channel_idx: int) -> np.ndarray:
        c_out = self.target_channels
        if c_out == 3 and channel_idx == 2:
            speed = np.sqrt(np.square(x_raw[:, 2]) + np.square(x_raw[:, 3]))
            return speed.astype(np.float32, copy=False)
        return x_raw[:, channel_idx].astype(np.float32, copy=False)

    def _prepare_window_base_features(
        self,
        x_norm: np.ndarray,
        mask: np.ndarray,
        channel_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_context_ready()
        assert self.input_mean is not None
        assert self.input_std is not None
        assert self.lat is not None
        assert self.lon is not None

        h = int(x_norm.shape[2])
        w = int(x_norm.shape[3])
        mask_bool = np.asarray(mask > 0.5, dtype=bool)
        valid_idx = np.flatnonzero(mask_bool.reshape(-1))

        x_raw = x_norm * self.input_std[None, :, None, None] + self.input_mean[None, :, None, None]
        lag_maps = _extract_recent_lags(self._channel_input_series(x_raw, channel_idx), self.lag_steps)
        lag_flat = lag_maps.reshape(self.lag_steps, -1).transpose(1, 0)
        lag_valid = lag_flat[valid_idx]

        lat_grid, lon_grid = _lat_lon_mesh(self.lat, self.lon, h=h, w=w)
        lat_valid = lat_grid.reshape(-1)[valid_idx].astype(np.float32, copy=False)
        lon_valid = lon_grid.reshape(-1)[valid_idx].astype(np.float32, copy=False)
        return lag_valid, lat_valid, lon_valid, valid_idx

    def _compose_features(
        self,
        lag_valid: np.ndarray,
        lat_valid: np.ndarray,
        lon_valid: np.ndarray,
        hour_idx: int,
        lead_frac: float,
    ) -> np.ndarray:
        n = int(lag_valid.shape[0])
        hour_sin, hour_cos, year_sin, year_cos = _time_features(hour_idx)
        cols = [
            lag_valid.astype(np.float32, copy=False),
            np.full((n, 1), hour_sin, dtype=np.float32),
            np.full((n, 1), hour_cos, dtype=np.float32),
            np.full((n, 1), year_sin, dtype=np.float32),
            np.full((n, 1), year_cos, dtype=np.float32),
            np.full((n, 1), float(lead_frac), dtype=np.float32),
            lat_valid.reshape(n, 1).astype(np.float32, copy=False),
            lon_valid.reshape(n, 1).astype(np.float32, copy=False),
            np.ones((n, 1), dtype=np.float32),
        ]
        x = np.concatenate(cols, axis=1)
        if self.feature_dim <= 0:
            self.feature_dim = int(x.shape[1])
        return x

    def fit(self, dataset: Sequence[Dict[str, Any]], stats: Dict[str, np.ndarray], seed: int | None = None) -> Dict[str, Any]:
        self._set_context_from_stats(stats)
        self._ensure_context_ready()
        xgb = self._require_xgboost()

        if self.lag_steps <= 0:
            raise ValueError(f"lag_steps must be > 0, got {self.lag_steps}")
        if self.train_row_cap <= 0:
            raise ValueError(f"train_row_cap must be > 0, got {self.train_row_cap}")
        if len(dataset) == 0:
            raise RuntimeError("Empty dataset for XGBoost fit.")

        rng = np.random.default_rng(int(self.random_state if seed is None else seed))
        n_channels = self.target_channels
        x_chunks: List[List[np.ndarray]] = [[] for _ in range(n_channels)]
        y_chunks: List[List[np.ndarray]] = [[] for _ in range(n_channels)]
        row_counts = np.zeros((n_channels,), dtype=np.int64)

        order = rng.permutation(len(dataset))
        for ds_idx in order:
            sample = dataset[int(ds_idx)]
            x_norm = _to_numpy(sample["x"]).astype(np.float32, copy=False)
            y_norm = _to_numpy(sample["y"]).astype(np.float32, copy=False)
            t_out = _to_numpy(sample["t_out"]).astype(np.int64, copy=False)
            mask = _to_numpy(sample["mask"]).astype(np.float32, copy=False)
            tout = int(y_norm.shape[0])

            for c in range(n_channels):
                remaining = int(self.train_row_cap - row_counts[c])
                if remaining <= 0:
                    continue

                lag_valid, lat_valid, lon_valid, valid_idx = self._prepare_window_base_features(
                    x_norm=x_norm,
                    mask=mask,
                    channel_idx=c,
                )
                if valid_idx.size == 0:
                    continue

                target_flat = y_norm[:, c].reshape(tout, -1)[:, valid_idx]
                for lead_idx in range(tout):
                    remaining = int(self.train_row_cap - row_counts[c])
                    if remaining <= 0:
                        break
                    n_valid = int(lag_valid.shape[0])
                    if remaining < n_valid:
                        sel = rng.choice(n_valid, size=remaining, replace=False)
                        lag_sel = lag_valid[sel]
                        lat_sel = lat_valid[sel]
                        lon_sel = lon_valid[sel]
                        y_sel = target_flat[lead_idx, sel]
                    else:
                        lag_sel = lag_valid
                        lat_sel = lat_valid
                        lon_sel = lon_valid
                        y_sel = target_flat[lead_idx]
                    lead_frac = float(lead_idx + 1) / float(max(tout, 1))
                    x_block = self._compose_features(
                        lag_valid=lag_sel,
                        lat_valid=lat_sel,
                        lon_valid=lon_sel,
                        hour_idx=int(t_out[lead_idx]),
                        lead_frac=lead_frac,
                    )
                    x_chunks[c].append(x_block.astype(np.float32, copy=False))
                    y_chunks[c].append(y_sel.astype(np.float32, copy=False))
                    row_counts[c] += int(x_block.shape[0])

            if np.all(row_counts >= self.train_row_cap):
                break

        self.models = []
        for c in range(n_channels):
            if not x_chunks[c]:
                raise RuntimeError(f"No training rows were collected for channel={c}.")
            x_train = np.concatenate(x_chunks[c], axis=0).astype(np.float32, copy=False)
            y_train = np.concatenate(y_chunks[c], axis=0).astype(np.float32, copy=False)
            params = dict(self.model_params)
            params["random_state"] = int(seed if seed is not None else self.random_state) + c
            model = xgb.XGBRegressor(**params)
            model.fit(x_train, y_train, verbose=False)
            self.models.append(model)

        return {
            "rows_per_channel": [int(v) for v in row_counts.tolist()],
            "feature_dim": int(self.feature_dim),
            "target_channels": int(n_channels),
            "lag_steps": int(self.lag_steps),
        }

    def predict_window(self, sample: Dict[str, Any]) -> np.ndarray:
        x_norm = _to_numpy(sample["x"]).astype(np.float32, copy=False)
        t_out = _to_numpy(sample["t_out"]).astype(np.int64, copy=False)
        mask = _to_numpy(sample["mask"]).astype(np.float32, copy=False)
        return self.predict_window_arrays(x_norm=x_norm, t_out=t_out, mask=mask)

    def predict_window_arrays(self, x_norm: np.ndarray, t_out: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._ensure_context_ready()
        if not self.models:
            raise RuntimeError("XGBoost models are empty. Fit or load a model first.")
        tout = int(t_out.shape[0])
        h = int(x_norm.shape[2])
        w = int(x_norm.shape[3])
        out = np.zeros((tout, self.target_channels, h, w), dtype=np.float32)

        mask_bool = np.asarray(mask > 0.5, dtype=bool)
        if np.flatnonzero(mask_bool.reshape(-1)).size == 0:
            return out

        for c, model in enumerate(self.models):
            lag_valid, lat_valid, lon_valid, valid_idx_runtime = self._prepare_window_base_features(
                x_norm=x_norm,
                mask=mask.astype(np.float32, copy=False),
                channel_idx=c,
            )
            for lead_idx in range(tout):
                lead_frac = float(lead_idx + 1) / float(max(tout, 1))
                x_feat = self._compose_features(
                    lag_valid=lag_valid,
                    lat_valid=lat_valid,
                    lon_valid=lon_valid,
                    hour_idx=int(t_out[lead_idx]),
                    lead_frac=lead_frac,
                )
                pred_flat = model.predict(x_feat).astype(np.float32, copy=False)
                frame = out[lead_idx, c].reshape(-1)
                frame[valid_idx_runtime] = pred_flat
        return out

    def predict_batch(self, batch: Dict[str, Any]) -> np.ndarray:
        x = _to_numpy(batch["x"]).astype(np.float32, copy=False)
        t_out = _to_numpy(batch["t_out"]).astype(np.int64, copy=False)
        mask = _to_numpy(batch["mask"]).astype(np.float32, copy=False)
        preds = []
        for i in range(int(x.shape[0])):
            preds.append(
                self.predict_window_arrays(
                    x_norm=x[i],
                    t_out=t_out[i],
                    mask=mask[i],
                )
            )
        return np.stack(preds, axis=0).astype(np.float32, copy=False)

    def denormalize_targets(self, y_norm: np.ndarray) -> np.ndarray:
        self._ensure_context_ready()
        assert self.target_mean is not None
        assert self.target_std is not None
        return (
            y_norm * self.target_std[None, :, None, None] + self.target_mean[None, :, None, None]
        ).astype(np.float32, copy=False)

    def save(self, save_dir: str | Path) -> None:
        self._ensure_context_ready()
        if not self.models:
            raise RuntimeError("No trained models to save.")

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self.models):
            model.save_model(str(save_path / f"channel_{i}.ubj"))

        assert self.input_mean is not None
        assert self.input_std is not None
        assert self.target_mean is not None
        assert self.target_std is not None
        assert self.ocean_mask is not None
        assert self.lat is not None
        assert self.lon is not None
        np.savez_compressed(
            save_path / "arrays.npz",
            input_mean=self.input_mean,
            input_std=self.input_std,
            target_mean=self.target_mean,
            target_std=self.target_std,
            ocean_mask=self.ocean_mask,
            lat=self.lat,
            lon=self.lon,
        )
        manifest = {
            "version": 1,
            "lag_steps": int(self.lag_steps),
            "train_row_cap": int(self.train_row_cap),
            "random_state": int(self.random_state),
            "feature_dim": int(self.feature_dim),
            "model_params": self.model_params,
            "target_channels": int(self.target_channels),
            "n_models": int(len(self.models)),
        }
        with (save_path / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, save_dir: str | Path) -> "XGBPointwiseForecaster":
        path = Path(save_dir)
        manifest_path = path / "manifest.json"
        arrays_path = path / "arrays.npz"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        if not arrays_path.exists():
            raise FileNotFoundError(f"Missing arrays file: {arrays_path}")

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        arrays = np.load(arrays_path)
        stats = {k: arrays[k] for k in arrays.files}

        obj = cls(
            lag_steps=int(manifest["lag_steps"]),
            train_row_cap=int(manifest.get("train_row_cap", 2_000_000)),
            model_params=dict(manifest.get("model_params", {})),
            random_state=int(manifest.get("random_state", 42)),
        )
        obj.feature_dim = int(manifest.get("feature_dim", 0))
        obj._set_context_from_stats(stats)

        xgb = cls._require_xgboost()
        n_models = int(manifest.get("n_models", manifest.get("target_channels", 0)))
        obj.models = []
        for i in range(n_models):
            model = xgb.XGBRegressor()
            model.load_model(str(path / f"channel_{i}.ubj"))
            obj.models.append(model)
        return obj


def evaluate_forecaster_on_dataset(
    forecaster: XGBPointwiseForecaster,
    dataset: Sequence[Dict[str, Any]],
    nrmse_denom: np.ndarray,
    speed_bucket_edges: Sequence[float] | np.ndarray | None = None,
    max_windows: int | None = None,
    eval_batch_windows: int = 4,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty, cannot evaluate.")

    speed_edges = None
    if speed_bucket_edges is not None:
        speed_edges = _validate_speed_bucket_edges(speed_bucket_edges)
        n_buckets = int(speed_edges.size) + 1
        bucket_sum_sq = np.zeros((n_buckets,), dtype=np.float64)
        bucket_count = np.zeros((n_buckets,), dtype=np.float64)
        bucket_samples = np.zeros((n_buckets,), dtype=np.float64)
    else:
        bucket_sum_sq = bucket_count = bucket_samples = None

    sum_sq = np.zeros((3,), dtype=np.float64)
    count = np.zeros((3,), dtype=np.float64)
    total_loss = 0.0
    n_windows = 0

    t0 = time.perf_counter()
    limit = int(len(dataset) if max_windows is None else min(int(max_windows), len(dataset)))
    batch_windows = max(int(eval_batch_windows), 1)
    for start in range(0, limit, batch_windows):
        end = min(start + batch_windows, limit)
        samples = [dataset[i] for i in range(start, end)]
        batch = {
            "x": np.stack([_to_numpy(s["x"]).astype(np.float32, copy=False) for s in samples], axis=0),
            "y": np.stack([_to_numpy(s["y"]).astype(np.float32, copy=False) for s in samples], axis=0),
            "mask": np.stack([_to_numpy(s["mask"]).astype(np.float32, copy=False) for s in samples], axis=0),
            "t_out": np.stack([_to_numpy(s["t_out"]).astype(np.int64, copy=False) for s in samples], axis=0),
        }
        pred_norm_batch = forecaster.predict_batch(batch)
        for i in range(pred_norm_batch.shape[0]):
            y_norm = batch["y"][i]
            mask = batch["mask"][i]
            pred_norm = pred_norm_batch[i]
            total_loss += _masked_mse(pred_norm, y_norm, mask)
            n_windows += 1

            pred_raw = forecaster.denormalize_targets(pred_norm)
            y_raw = forecaster.denormalize_targets(y_norm)
            pred_eval = to_eval_channels_numpy(pred_raw)
            y_eval = to_eval_channels_numpy(y_raw)
            valid = (mask > 0.5).astype(np.float64)[None, :, :]

            err2 = np.square(pred_eval - y_eval).astype(np.float64, copy=False)
            sum_sq += np.sum(err2 * valid[:, None, :, :], axis=(0, 2, 3))
            count += np.sum(valid, axis=(0, 1, 2)) * float(pred_eval.shape[0])

            if speed_edges is not None and bucket_sum_sq is not None and bucket_count is not None and bucket_samples is not None:
                speed_target = y_eval[:, 2]
                denom = float(np.sum(valid))
                speed_gt_mean = float(np.sum(speed_target * valid) / max(denom * float(speed_target.shape[0]), 1.0))
                bucket_idx = int(_assign_speed_bins(np.asarray([speed_gt_mean], dtype=np.float64), speed_edges)[0])
                speed_err2 = np.square(pred_eval[:, 2] - y_eval[:, 2]).astype(np.float64, copy=False)
                bucket_sum_sq[bucket_idx] += float(np.sum(speed_err2 * valid))
                bucket_count[bucket_idx] += float(np.sum(valid) * speed_err2.shape[0])
                bucket_samples[bucket_idx] += 1.0

    elapsed = time.perf_counter() - t0
    rmse = np.sqrt(sum_sq / np.maximum(count, 1.0))
    nrmse = rmse_to_nrmse(rmse, nrmse_denom)
    metrics = summarize_channel_metrics(rmse, nrmse)
    metrics["loss"] = float(total_loss / max(n_windows, 1))

    if speed_edges is not None and bucket_sum_sq is not None and bucket_count is not None and bucket_samples is not None:
        _append_speed_bucket_metrics(
            metrics=metrics,
            bucket_sum_sq=bucket_sum_sq,
            bucket_count=bucket_count,
            bucket_samples=bucket_samples,
            nrmse_denom=np.asarray(nrmse_denom, dtype=np.float64),
        )

    profile = {
        "total_eval_seconds": float(elapsed),
        "num_eval_batches": int(n_windows),
        "num_eval_samples": int(n_windows),
        "avg_batch_seconds": float(elapsed / max(n_windows, 1)),
        "avg_sample_seconds": float(elapsed / max(n_windows, 1)),
        "samples_per_second": float(n_windows / elapsed) if elapsed > 0 else 0.0,
        "windows_per_second": float(n_windows / elapsed) if elapsed > 0 else 0.0,
    }
    return metrics, profile


def speed_bucket_debug_info(speed_bucket_edges: Sequence[float] | np.ndarray) -> Dict[str, Any]:
    edges = _validate_speed_bucket_edges(speed_bucket_edges)
    return {"bin_edges": edges.tolist(), "labels": _speed_bucket_labels(edges)}

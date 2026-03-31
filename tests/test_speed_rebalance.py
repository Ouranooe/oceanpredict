from __future__ import annotations

import numpy as np
import pytest
from torch.utils.data import WeightedRandomSampler

from ocean_forecast.train import (
    _assign_speed_bins,
    _build_train_speed_rebalance_sampler,
    _compute_future_window_mean_speed,
)
from ocean_forecast.data.zip_reader import FrameRef


class _DummyReader:
    def __init__(self, speed_by_index: list[float]):
        self.speed_by_index = [float(v) for v in speed_by_index]

    def read_frame(self, ref: FrameRef):
        speed = np.float32(self.speed_by_index[int(ref.time_idx)])
        frame = np.zeros((4, 2, 2), dtype=np.float32)
        frame[2, :, :] = speed
        frame[3, :, :] = 0.0
        mask = np.ones((2, 2), dtype=bool)
        return frame, mask


def _make_refs(n: int) -> list[FrameRef]:
    base = np.datetime64("2000-01-01T00:00:00")
    refs: list[FrameRef] = []
    for i in range(int(n)):
        refs.append(
            FrameRef(
                timestamp=base + np.timedelta64(i, "h"),
                hour_index=i,
                zip_path="unused.npy",
                member_name="2011",
                time_idx=i,
            )
        )
    return refs


def test_assign_speed_bins_boundary_behavior() -> None:
    bin_edges = np.array([0.1, 0.2, 0.4], dtype=np.float32)
    speed_gt = np.array([0.0, 0.099, 0.1, 0.1999, 0.2, 0.3999, 0.4, 0.8], dtype=np.float32)
    bucket_idx = _assign_speed_bins(speed_gt=speed_gt, bin_edges=bin_edges)
    assert bucket_idx.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_compute_future_window_mean_speed_matches_manual() -> None:
    ref_speed = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.50], dtype=np.float32)
    windows = np.array([0, 1], dtype=np.int64)
    out = _compute_future_window_mean_speed(
        ref_speed=ref_speed,
        window_starts=windows,
        input_len=2,
        pred_len=3,
    )
    assert np.allclose(out, np.array([0.30, 0.40], dtype=np.float32), atol=1e-6)


def test_build_train_speed_rebalance_sampler_enabled_and_disabled() -> None:
    refs = _make_refs(5)
    train_windows = np.array([0, 1, 2, 3], dtype=np.int64)
    reader = _DummyReader([0.05, 0.10, 0.20, 0.30, 0.40])

    sampler, meta = _build_train_speed_rebalance_sampler(
        reader=reader,
        train_refs=refs,
        train_windows=train_windows,
        input_len=1,
        pred_len=1,
        speed_rebalance_cfg={
            "enabled": True,
            "bin_edges": [0.1, 0.2, 0.4],
            "bin_weights": [1.0, 1.5, 2.5, 4.0],
            "stat": "future_ocean_mean",
        },
    )
    assert isinstance(sampler, WeightedRandomSampler)
    assert bool(meta["enabled"]) is True
    assert meta["bucket_idx"].tolist() == [1, 2, 2, 3]
    assert meta["bucket_counts"].tolist() == [0, 1, 2, 1]
    assert np.allclose(np.asarray(sampler.weights), np.array([1.5, 2.5, 2.5, 4.0]), atol=1e-6)

    sampler_disabled, meta_disabled = _build_train_speed_rebalance_sampler(
        reader=reader,
        train_refs=refs,
        train_windows=train_windows,
        input_len=1,
        pred_len=1,
        speed_rebalance_cfg={"enabled": False},
    )
    assert sampler_disabled is None
    assert bool(meta_disabled["enabled"]) is False


def test_build_train_speed_rebalance_sampler_invalid_bin_weights_raises() -> None:
    refs = _make_refs(3)
    reader = _DummyReader([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="bin_weights size mismatch"):
        _build_train_speed_rebalance_sampler(
            reader=reader,
            train_refs=refs,
            train_windows=np.array([0], dtype=np.int64),
            input_len=1,
            pred_len=1,
            speed_rebalance_cfg={
                "enabled": True,
                "bin_edges": [0.1, 0.2, 0.4],
                "bin_weights": [1.0, 2.0, 3.0],
                "stat": "future_ocean_mean",
            },
        )

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from ocean_forecast.train import (
    _assign_speed_bins,
    _build_train_speed_rebalance_sampler,
    _compute_batch_speed_gt_means,
    _compute_future_window_mean_speed,
    _masked_speed_aux_loss,
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


def test_masked_speed_aux_loss_uv_huber_and_mse() -> None:
    pred = torch.zeros(1, 2, 4, 2, 2, dtype=torch.float32)
    target = torch.zeros(1, 2, 4, 2, 2, dtype=torch.float32)
    mask = torch.ones(1, 2, 2, dtype=torch.float32)
    target_mean = np.zeros(4, dtype=np.float32)
    target_std = np.ones(4, dtype=np.float32)

    # Introduce speed error through u-channel at t=1.
    pred[:, 1, 2, :, :] = 1.0

    loss_huber = _masked_speed_aux_loss(
        pred_norm=pred,
        target_norm=target,
        mask=mask,
        target_mean=target_mean,
        target_std=target_std,
        loss_type="huber",
        huber_delta=0.1,
    )
    loss_mse = _masked_speed_aux_loss(
        pred_norm=pred,
        target_norm=target,
        mask=mask,
        target_mean=target_mean,
        target_std=target_std,
        loss_type="mse",
        huber_delta=0.1,
    )
    assert torch.isfinite(loss_huber)
    assert torch.isfinite(loss_mse)
    assert float(loss_huber.item()) > 0.0
    assert float(loss_mse.item()) > float(loss_huber.item())


def test_compute_batch_speed_gt_means_matches_masked_average() -> None:
    y_eval = torch.zeros(2, 2, 3, 2, 2, dtype=torch.float32)
    y_eval[0, :, 2, :, :] = 0.2
    y_eval[1, :, 2, :, :] = 0.5
    mask = torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
    means = _compute_batch_speed_gt_means(y_eval=y_eval, mask=mask)
    assert np.allclose(means, np.array([0.2, 0.5], dtype=np.float64), atol=1e-6)

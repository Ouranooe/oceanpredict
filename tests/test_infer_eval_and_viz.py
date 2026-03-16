from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from ocean_forecast.infer import _compute_metrics_payload, _sample_frame_indices, _save_visualizations


def test_compute_metrics_payload_available() -> None:
    pred = np.zeros((2, 3, 2, 2), dtype=np.float32)
    gt = np.ones((2, 3, 2, 2), dtype=np.float32)
    mask = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32) > 0.5
    nrmse_denom = np.array([2.0, 4.0, 5.0], dtype=np.float32)

    payload = _compute_metrics_payload(
        pred_raw=pred,
        gt_raw=gt,
        ocean_mask=mask,
        nrmse_denom=nrmse_denom,
        reason_if_unavailable="unused",
    )

    assert payload["metrics_available"] is True
    assert np.isclose(payload["rmse_sst"], 1.0)
    assert np.isclose(payload["nrmse_sst"], 0.5)
    assert np.isclose(payload["nrmse_sss"], 0.25)
    assert np.isclose(payload["nrmse_speed"], 0.2)
    assert payload["pass_15pct"] is False


def test_compute_metrics_payload_unavailable() -> None:
    pred = np.zeros((2, 3, 2, 2), dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)

    payload = _compute_metrics_payload(
        pred_raw=pred,
        gt_raw=None,
        ocean_mask=mask,
        nrmse_denom=np.ones(3, dtype=np.float32),
        reason_if_unavailable="missing gt",
    )

    assert payload["metrics_available"] is False
    assert payload["metrics_unavailable_reason"] == "missing gt"


def test_sample_frame_indices_every_9h() -> None:
    assert _sample_frame_indices(total_steps=24, every_hours=9) == [0, 9, 18]


def test_save_visualizations_agg_backend() -> None:
    pred = np.random.randn(10, 3, 2, 3).astype(np.float32)
    gt = pred + 0.1
    forecast_hours = np.arange(100, 110, dtype=np.int64)
    lat = np.array([31.0, 32.0], dtype=np.float32)
    lon = np.array([117.0, 118.0, 119.0], dtype=np.float32)

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "viz"
        image_paths = _save_visualizations(
            pred_raw=pred,
            gt_raw=gt,
            forecast_hours=forecast_hours,
            start_hour=100,
            start_time=np.datetime64("2014-06-01T00:00:00"),
            lat=lat,
            lon=lon,
            viz_dir=out_dir,
            sample_every_hours=9,
            show=False,
        )

        assert len(image_paths) == 2
        for path_str in image_paths:
            assert Path(path_str).exists()

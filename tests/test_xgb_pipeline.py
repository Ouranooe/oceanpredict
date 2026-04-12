from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from ocean_forecast.xgb_pipeline import (
    XGBPointwiseForecaster,
    _extract_recent_lags,
    evaluate_forecaster_on_dataset,
)


class _DummyRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full((x.shape[0],), self.value, dtype=np.float32)


def _synthetic_stats(h: int = 2, w: int = 3, target_channels: int = 3) -> dict[str, np.ndarray]:
    return {
        "input_mean": np.zeros((4,), dtype=np.float32),
        "input_std": np.ones((4,), dtype=np.float32),
        "target_mean": np.zeros((target_channels,), dtype=np.float32),
        "target_std": np.ones((target_channels,), dtype=np.float32),
        "ocean_mask": np.ones((h, w), dtype=np.float32),
        "lat": np.linspace(30.0, 31.0, h, dtype=np.float32),
        "lon": np.linspace(120.0, 122.0, w, dtype=np.float32),
        "nrmse_denom": np.ones((3,), dtype=np.float32),
    }


def _synthetic_sample(t_in: int = 4, t_out: int = 3, h: int = 2, w: int = 3, c_out: int = 3) -> dict[str, np.ndarray]:
    x = np.random.randn(t_in, 4, h, w).astype(np.float32)
    y = np.random.randn(t_out, c_out, h, w).astype(np.float32)
    mask = np.array([[1, 1, 0], [1, 0, 1]], dtype=np.float32)
    t_out_idx = np.arange(100, 100 + t_out, dtype=np.int64)
    return {"x": x, "y": y, "mask": mask, "t_out": t_out_idx}


def test_extract_recent_lags_boundary_padding() -> None:
    series = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)
    out = _extract_recent_lags(series, lag_steps=6)
    assert out.shape == (6, 2, 2)
    assert np.allclose(out[0], series[0])
    assert np.allclose(out[-1], series[-1])


def test_predict_window_shape_and_feature_dim_with_dummy_models() -> None:
    stats = _synthetic_stats()
    sample = _synthetic_sample()
    forecaster = XGBPointwiseForecaster(lag_steps=6, train_row_cap=128)
    forecaster._set_context_from_stats(stats)
    forecaster.models = [_DummyRegressor(0.0), _DummyRegressor(0.0), _DummyRegressor(0.0)]

    pred = forecaster.predict_window(sample)
    assert pred.shape == sample["y"].shape
    assert forecaster.feature_dim == 14  # lag_steps(6) + time(4) + lead(1) + lat/lon/mask(3)


def test_missing_xgboost_dependency_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    original = importlib.import_module

    def _fake_import(name: str, package: str | None = None):
        if name == "xgboost":
            raise ModuleNotFoundError("mocked missing xgboost")
        return original(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    with pytest.raises(RuntimeError, match="pip install xgboost"):
        XGBPointwiseForecaster._require_xgboost()


def test_evaluate_forecaster_metrics_keys() -> None:
    stats = _synthetic_stats()
    sample = _synthetic_sample()
    forecaster = XGBPointwiseForecaster(lag_steps=6, train_row_cap=128)
    forecaster._set_context_from_stats(stats)
    forecaster.models = [_DummyRegressor(0.0), _DummyRegressor(0.0), _DummyRegressor(0.0)]

    metrics, profile = evaluate_forecaster_on_dataset(
        forecaster=forecaster,
        dataset=[sample],
        nrmse_denom=stats["nrmse_denom"],
        speed_bucket_edges=[0.1, 0.2, 0.4],
    )
    assert "nrmse_mean" in metrics
    assert "speed_bucket_0_samples" in metrics
    assert profile["num_eval_samples"] == 1


@pytest.mark.skipif(importlib.util.find_spec("xgboost") is None, reason="xgboost not installed in current env")
def test_xgboost_fit_save_load_smoke(tmp_path: Path) -> None:
    stats = _synthetic_stats()
    ds = [_synthetic_sample(), _synthetic_sample()]
    forecaster = XGBPointwiseForecaster(
        lag_steps=4,
        train_row_cap=256,
        model_params={
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.2,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "device": "cpu",
        },
        random_state=7,
    )
    summary = forecaster.fit(ds, stats=stats, seed=7)
    assert len(summary["rows_per_channel"]) == 3
    pred = forecaster.predict_window(ds[0])
    assert pred.shape == ds[0]["y"].shape

    save_dir = tmp_path / "xgb_model"
    forecaster.save(save_dir)
    restored = XGBPointwiseForecaster.load(save_dir)
    pred2 = restored.predict_window(ds[0])
    assert pred2.shape == ds[0]["y"].shape


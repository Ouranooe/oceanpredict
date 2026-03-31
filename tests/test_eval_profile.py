from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ocean_forecast.train import evaluate_with_profile


class _DummyForecastModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return the first 3 channels as forecast to match target channel count.
        return x[:, :, :3, :, :]


class _DummySeqDataset(Dataset):
    def __init__(self, num_samples: int, t: int = 4, c_in: int = 4, h: int = 3, w: int = 3):
        self.num_samples = int(num_samples)
        self.t = int(t)
        self.c_in = int(c_in)
        self.h = int(h)
        self.w = int(w)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        x = torch.randn(self.t, self.c_in, self.h, self.w)
        y = x[:, :3, :, :].clone()
        t_in = torch.arange(self.t, dtype=torch.int64)
        mask = torch.ones(self.h, self.w, dtype=torch.float32)
        return {"x": x, "y": y, "t_in": t_in, "mask": mask}


def test_evaluate_with_profile_returns_expected_keys_and_max_batches() -> None:
    model = _DummyForecastModel()
    loader = DataLoader(_DummySeqDataset(num_samples=5), batch_size=2, shuffle=False)
    input_feature_cfg = {"add_mask": False, "add_time_hour": False, "add_time_year": False}

    metrics, profile = evaluate_with_profile(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        target_mean=np.zeros(3, dtype=np.float32),
        target_std=np.ones(3, dtype=np.float32),
        nrmse_denom=np.ones(3, dtype=np.float32),
        input_feature_cfg=input_feature_cfg,
        max_batches=2,
    )

    assert "loss" in metrics
    assert profile["num_eval_batches"] == 2
    assert profile["num_eval_samples"] == 4
    assert profile["total_eval_seconds"] >= 0.0
    assert profile["avg_batch_seconds"] >= 0.0
    assert profile["avg_sample_seconds"] >= 0.0
    assert profile["samples_per_second"] >= 0.0
    assert profile["windows_per_second"] >= 0.0


def test_evaluate_with_profile_empty_loader_is_zero_safe() -> None:
    model = _DummyForecastModel()
    loader = DataLoader(_DummySeqDataset(num_samples=0), batch_size=2, shuffle=False)
    input_feature_cfg = {"add_mask": False, "add_time_hour": False, "add_time_year": False}

    metrics, profile = evaluate_with_profile(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        target_mean=np.zeros(3, dtype=np.float32),
        target_std=np.ones(3, dtype=np.float32),
        nrmse_denom=np.ones(3, dtype=np.float32),
        input_feature_cfg=input_feature_cfg,
        max_batches=None,
    )

    assert "loss" in metrics
    assert profile["num_eval_batches"] == 0
    assert profile["num_eval_samples"] == 0
    assert profile["avg_batch_seconds"] == 0.0
    assert profile["avg_sample_seconds"] == 0.0
    assert profile["samples_per_second"] == 0.0
    assert profile["windows_per_second"] == 0.0

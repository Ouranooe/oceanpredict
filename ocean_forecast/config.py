from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "data": {
        "base_root": "dataset_base/sea_v1",
        "format": "zip",
        "root_dir": "dataset/海域要素预测",
        "train_years": list(range(1994, 2014)),
        "test_years": [2014],
        "input_len": 72,
        "pred_len": 72,
        "stride": 1,
        "cache_size": 8,
        "train_start": "1994-01-01T00:00:00",
        "train_main_end": "2013-10-31T23:00:00",
        "dev_start": "2013-11-01T00:00:00",
        "dev_end": "2013-12-31T23:00:00",
        "test_start": "2014-01-01T00:00:00",
        "test_end": "2014-12-31T23:00:00",
        "input_features": {
            "add_mask": False,
            "add_time_hour": False,
            "add_time_year": False,
        },
    },
    "model": {
        "name": "convlstm",
        "hidden_dims": [32, 32],
        "kernel_size": 3,
        "dropout": 0.1,
    },
    "train": {
        "batch_size": 2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 20,
        "patience": 5,
        "num_workers": 0,
        "device": "auto",
        "amp": True,
        "seed": 42,
        "shuffle_train": False,
        "output_dir": "outputs/convlstm_baseline",
        "max_train_batches": None,
        "max_eval_batches": None,
        "density_physics_loss": {
            "enabled": True,
            "weight": 0.01,
            "rho0": 1027.0,
            "alpha": 2.0e-4,
            "beta": 7.5e-4,
            "t0": 0.0,
            "s0": 35.0,
        },
        "smoothness_loss": {
            "enabled": False,
            "weight": 0.0,
            "spatial_weight": 1.0,
            "temporal_weight": 1.0,
        },
        "diffdiv_loss": {
            "enabled": False,
            "weight": 0.05,
            "channels": ["sst", "sss"],
            "speed_weight": 0.0,
        },
        "laplacian_smoothness_loss": {
            "enabled": False,
            "weight": 0.0,
            "channels": ["sst", "sss", "speed"],
        },
        "stability": {
            "grad_clip": {
                "enabled": True,
                "max_norm": 1.0,
                "norm_type": 2.0,
            },
            "scheduler": {
                "enabled": True,
                "type": "plateau",
                "monitor": "dev_loss",
                "factor": 0.5,
                "patience": 2,
                "threshold": 1e-4,
                "min_lr": 1e-6,
                "cooldown": 0,
            },
            "warmup": {
                "enabled": True,
                "epochs": 1,
                "start_factor": 0.1,
            },
            "nan_guard": {
                "enabled": True,
                "lr_decay_factor": 0.5,
                "max_bad_steps_per_epoch": 100,
            },
            "ema": {
                "enabled": False,
                "decay": 0.999,
                "update_after_step": 0,
                "eval_with_ema": True,
            },
        },
    },
}


def _deep_update(base: Dict[str, Any], new_values: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in new_values.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = deepcopy(DEFAULT_CONFIG)
    return _deep_update(cfg, user_cfg)


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

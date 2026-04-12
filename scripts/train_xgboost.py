from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ocean_forecast.config import load_config
from ocean_forecast.data.dataset import PreparedSeqDataset
from ocean_forecast.data.prepared import PreparedReader
from ocean_forecast.train import _log, _resolve_or_build_experiment_artifacts
from ocean_forecast.xgb_pipeline import XGBPointwiseForecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost point-wise baseline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Override xgboost device.")
    return parser.parse_args()


def _xgb_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg.get("xgboost", {}) or {})
    out.setdefault("enabled", True)
    out.setdefault("lag_steps", 6)
    out.setdefault("train_row_cap", 2_000_000)
    out.setdefault("eval_batch_windows", 4)
    out.setdefault("n_estimators", 500)
    out.setdefault("max_depth", 8)
    out.setdefault("learning_rate", 0.05)
    out.setdefault("subsample", 0.8)
    out.setdefault("colsample_bytree", 0.8)
    out.setdefault("tree_method", "hist")
    out.setdefault("device", "cpu")
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    xgb_cfg = _xgb_cfg(cfg)

    if not bool(xgb_cfg.get("enabled", True)):
        raise RuntimeError("xgboost.enabled is false in config. Set xgboost.enabled=true for this script.")
    if args.device is not None:
        xgb_cfg["device"] = args.device

    base_root_text = data_cfg.get("base_root")
    if not base_root_text:
        raise RuntimeError("Config missing required data.base_root.")
    base_root = Path(str(base_root_text))
    prepared_root = _resolve_or_build_experiment_artifacts(base_root=base_root, data_cfg=data_cfg)
    reader = PreparedReader(prepared_root=prepared_root)
    try:
        split = reader.load_split_refs()
        train_windows = reader.load_windows("train")
        stats = reader.load_stats()
        manifest = reader.manifest
        train_ds = PreparedSeqDataset(
            reader=reader,
            refs=split.train_main,
            window_starts=train_windows,
            input_len=int(manifest.input_len),
            pred_len=int(manifest.pred_len),
            stats=stats,
        )
        _log(
            "XGBoost train dataset ready: "
            f"prepared_root={prepared_root}, windows={len(train_ds)}, "
            f"pred_len={manifest.pred_len}, target_channels={int(stats['target_mean'].shape[0])}"
        )

        model_params = {
            "n_estimators": int(xgb_cfg["n_estimators"]),
            "max_depth": int(xgb_cfg["max_depth"]),
            "learning_rate": float(xgb_cfg["learning_rate"]),
            "subsample": float(xgb_cfg["subsample"]),
            "colsample_bytree": float(xgb_cfg["colsample_bytree"]),
            "tree_method": str(xgb_cfg["tree_method"]),
            "device": str(xgb_cfg["device"]),
        }
        forecaster = XGBPointwiseForecaster(
            lag_steps=int(xgb_cfg["lag_steps"]),
            train_row_cap=int(xgb_cfg["train_row_cap"]),
            model_params=model_params,
            random_state=int(train_cfg.get("seed", 42)),
        )
        summary = forecaster.fit(
            dataset=train_ds,
            stats=stats,
            seed=int(train_cfg.get("seed", 42)),
        )

        output_dir = Path(str(train_cfg["output_dir"]))
        save_dir = output_dir / "xgboost_model"
        save_dir.mkdir(parents=True, exist_ok=True)
        forecaster.save(save_dir)

        train_summary = {
            "prepared_root": str(prepared_root),
            "model_dir": str(save_dir),
            "rows_per_channel": summary["rows_per_channel"],
            "feature_dim": int(summary["feature_dim"]),
            "target_channels": int(summary["target_channels"]),
            "lag_steps": int(summary["lag_steps"]),
            "train_windows": int(len(train_ds)),
            "xgboost_params": model_params,
        }
        with (save_dir / "train_summary.json").open("w", encoding="utf-8") as f:
            json.dump(train_summary, f, ensure_ascii=False, indent=2)
        _log(f"XGBoost model saved to: {save_dir}")
        _log(f"XGBoost train summary: {train_summary}")
    finally:
        reader.close()


if __name__ == "__main__":
    main()


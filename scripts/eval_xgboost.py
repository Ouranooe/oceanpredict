from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ocean_forecast.config import load_config
from ocean_forecast.data.dataset import PreparedSeqDataset
from ocean_forecast.data.prepared import PreparedReader
from ocean_forecast.train import _log, _resolve_or_build_experiment_artifacts
from ocean_forecast.xgb_pipeline import (
    XGBPointwiseForecaster,
    evaluate_forecaster_on_dataset,
    speed_bucket_debug_info,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate XGBoost point-wise baseline on dev/test.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--split", default="test", choices=["dev", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--ckpt_dir", default=None, help="Model dir. Default: <train.output_dir>/xgboost_model")
    parser.add_argument("--max_eval_windows", type=int, default=None, help="Optional cap on evaluated windows.")
    parser.add_argument("--eval_batch_windows", type=int, default=None, help="Predict N windows per batch.")
    parser.add_argument("--output_json", default=None, help="Optional output json path.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Device tag in output payload.")
    return parser.parse_args()


def _xgb_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg.get("xgboost", {}) or {})
    out.setdefault("eval_batch_windows", 4)
    return out


def _speed_edges_from_cfg(cfg: Dict[str, Any]) -> list[float]:
    speed_rebalance = ((cfg.get("train", {}) or {}).get("speed_rebalance", {}) or {})
    return list(speed_rebalance.get("bin_edges", [0.1, 0.2, 0.4]))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    xgb_cfg = _xgb_cfg(cfg)

    base_root_text = data_cfg.get("base_root")
    if not base_root_text:
        raise RuntimeError("Config missing required data.base_root.")
    base_root = Path(str(base_root_text))
    prepared_root = _resolve_or_build_experiment_artifacts(base_root=base_root, data_cfg=data_cfg)

    output_dir = Path(str(train_cfg["output_dir"]))
    model_dir = Path(args.ckpt_dir) if args.ckpt_dir else (output_dir / "xgboost_model")
    forecaster = XGBPointwiseForecaster.load(model_dir)
    speed_edges = _speed_edges_from_cfg(cfg)
    speed_meta = speed_bucket_debug_info(speed_edges)
    _log(
        "XGBoost eval setup: "
        f"split={args.split}, prepared_root={prepared_root}, model_dir={model_dir}, "
        f"speed_bin_edges={speed_meta['bin_edges']}"
    )

    reader = PreparedReader(prepared_root=prepared_root)
    try:
        split = reader.load_split_refs()
        stats = reader.load_stats()
        manifest = reader.manifest
        window_starts = reader.load_windows(args.split)
        refs = split.dev if args.split == "dev" else split.test
        ds = PreparedSeqDataset(
            reader=reader,
            refs=refs,
            window_starts=window_starts,
            input_len=int(manifest.input_len),
            pred_len=int(manifest.pred_len),
            stats=stats,
        )
        metrics, profile = evaluate_forecaster_on_dataset(
            forecaster=forecaster,
            dataset=ds,
            nrmse_denom=stats["nrmse_denom"],
            speed_bucket_edges=speed_edges,
            max_windows=args.max_eval_windows,
            eval_batch_windows=int(args.eval_batch_windows or xgb_cfg["eval_batch_windows"]),
        )
    finally:
        reader.close()

    metric_key = f"{args.split}_metrics"
    result = {
        metric_key: metrics,
        "inference_profile": profile,
        "device": str(args.device or "cpu"),
        "best_checkpoint": str(model_dir),
        "split": str(args.split),
    }
    if args.split == "test":
        result["test_metrics"] = metrics
    if args.split == "dev":
        result["dev_metrics"] = metrics

    out_path = Path(args.output_json) if args.output_json else (model_dir / "final_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    _log(f"Saved XGBoost evaluation to: {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from ocean_forecast.config import load_config
from ocean_forecast.data.dataset import PreparedSeqDataset
from ocean_forecast.data.prepared import PreparedReader
from ocean_forecast.input_features import (
    BASE_INPUT_CHANNELS,
    compute_model_input_channels,
    parse_input_feature_config,
)
from ocean_forecast.models.registry import build_model
from ocean_forecast.train import (
    ModelEMA,
    _build_model_kwargs,
    _resolve_or_build_experiment_artifacts,
    _log,
    evaluate_with_profile,
)
from ocean_forecast.utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supplement final test evaluation from output_dir/best.ckpt without training."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--device", default="cuda", help="Device override (default: cuda).")
    parser.add_argument("--max_eval_batches", type=int, default=None, help="Optional eval batch cap.")
    return parser.parse_args()


def _load_effective_config(config_path: str | Path) -> tuple[Dict, Path]:
    cfg_from_arg = load_config(config_path)
    output_dir = Path(str(cfg_from_arg["train"]["output_dir"]))
    resolved_cfg = output_dir / "resolved_config.yaml"
    if resolved_cfg.exists():
        cfg = load_config(resolved_cfg)
        return cfg, Path(str(cfg["train"]["output_dir"]))
    return cfg_from_arg, output_dir


def main() -> None:
    args = parse_args()
    cfg, output_dir = _load_effective_config(args.config)

    cfg["train"]["device"] = args.device
    if args.max_eval_batches is not None:
        cfg["train"]["max_eval_batches"] = int(args.max_eval_batches)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    set_seed(int(train_cfg.get("seed", 42)))
    device = resolve_device(train_cfg["device"])
    input_feature_cfg = parse_input_feature_config(data_cfg)
    model_input_channels = compute_model_input_channels(BASE_INPUT_CHANNELS, input_feature_cfg)

    best_ckpt = output_dir / "best.ckpt"
    if not best_ckpt.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {best_ckpt}. This script only accepts best.ckpt."
        )

    base_root_text = data_cfg.get("base_root")
    if not base_root_text:
        raise RuntimeError("Config missing required data.base_root.")
    base_root = Path(str(base_root_text))
    prepared_root = _resolve_or_build_experiment_artifacts(base_root=base_root, data_cfg=data_cfg)
    reader = PreparedReader(prepared_root=prepared_root)
    manifest = reader.manifest
    split = reader.load_split_refs()
    test_windows = reader.load_windows("test")
    stats = reader.load_stats()

    if test_windows.size == 0:
        raise RuntimeError("No test windows found in prepared artifacts.")

    test_ds = PreparedSeqDataset(
        reader=reader,
        refs=split.test,
        window_starts=test_windows,
        input_len=int(manifest.input_len),
        pred_len=int(manifest.pred_len),
        stats=stats,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(
        model_name=model_cfg["name"],
        **_build_model_kwargs(
            model_cfg,
            int(manifest.pred_len),
            model_input_channels,
            output_channels=int(stats["target_mean"].shape[0]),
        ),
    ).to(device)

    _log(f"Loading best checkpoint: {best_ckpt}")
    best = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])

    max_eval_batches = train_cfg.get("max_eval_batches", None)
    if max_eval_batches is not None:
        max_eval_batches = int(max_eval_batches)

    best_ema_state = best.get("ema_state")
    best_stability_state = best.get("stability_state", {}) or {}
    run_test_with_ema = bool(best_stability_state.get("ema_eval_with", False)) and best_ema_state is not None
    if run_test_with_ema:
        ema_for_test = ModelEMA(
            model=model,
            decay=float(best_ema_state.get("decay", 0.999)),
            update_after_step=int(best_ema_state.get("update_after_step", 0)),
        )
        ema_for_test.load_state_dict(best_ema_state)
        ema_backup = ema_for_test.apply_to(model)
        try:
            test_metrics, inference_profile = evaluate_with_profile(
                model,
                test_loader,
                device=device,
                target_mean=stats["target_mean"],
                target_std=stats["target_std"],
                nrmse_denom=stats["nrmse_denom"],
                input_feature_cfg=input_feature_cfg,
                max_batches=max_eval_batches,
            )
        finally:
            ema_for_test.restore(model, ema_backup)
    else:
        test_metrics, inference_profile = evaluate_with_profile(
            model,
            test_loader,
            device=device,
            target_mean=stats["target_mean"],
            target_std=stats["target_std"],
            nrmse_denom=stats["nrmse_denom"],
            input_feature_cfg=input_feature_cfg,
            max_batches=max_eval_batches,
        )

    stability_cfg = train_cfg.get("stability", {}) or {}
    scheduler_cfg = stability_cfg.get("scheduler", {}) or {}
    ema_cfg = stability_cfg.get("ema", {}) or {}

    result = {
        "best_dev_loss": float(best.get("dev_metrics", {}).get("loss", float("nan"))),
        "best_dev_nrmse_mean": float(best.get("dev_metrics", {}).get("nrmse_mean", float("nan"))),
        "test_metrics": test_metrics,
        "inference_profile": inference_profile,
        "device": str(device),
        "best_checkpoint": str(best_ckpt),
        "stability": {
            "scheduler_enabled": bool(scheduler_cfg.get("enabled", True)),
            "scheduler_monitor": str(scheduler_cfg.get("monitor", "dev_loss")),
            "warmup_steps": int(best_stability_state.get("warmup_steps", 0)),
            "total_bad_steps": int(best_stability_state.get("total_bad_steps", 0)),
            "lr_reduced_events": int(best_stability_state.get("lr_reduced_events", 0)),
            "ema_enabled": bool(best_stability_state.get("ema_enabled", ema_cfg.get("enabled", False))),
            "ema_eval_with": bool(
                best_stability_state.get("ema_eval_with", ema_cfg.get("eval_with_ema", True))
            ),
        },
    }

    final_metrics_path = output_dir / "final_metrics.json"
    with final_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    _log(f"Saved supplement evaluation to: {final_metrics_path}")
    reader.close()


if __name__ == "__main__":
    main()



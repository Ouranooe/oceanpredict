from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, save_config
from .data.dataset import (
    OceanSeqDataset,
    build_window_starts,
    compute_normalization_stats,
    save_stats,
    split_refs,
)
from .data.zip_reader import parse_datetime64, ZipNetCDFReader
from .losses import masked_mse_loss
from .metrics import MaskedChannelRMSE, rmse_to_nrmse, summarize_channel_metrics
from .models.registry import build_model
from .utils import resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Yellow-Bohai Sea ConvLSTM baseline.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output_dir", default=None, help="Override output directory.")
    parser.add_argument("--max_train_batches", type=int, default=None, help="Debug: cap train batches per epoch.")
    parser.add_argument("--max_eval_batches", type=int, default=None, help="Debug: cap eval batches.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience.")
    return parser.parse_args()


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    nrmse_denom: np.ndarray,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()
    meter = MaskedChannelRMSE(n_channels=3)
    total_loss = 0.0
    n_batches = 0

    mean_t = torch.tensor(target_mean, dtype=torch.float32, device=device).view(1, 1, 3, 1, 1)
    std_t = torch.tensor(target_std, dtype=torch.float32, device=device).view(1, 1, 3, 1, 1)

    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            batch = _to_device(batch, device)
            pred_norm = model(batch["x"])
            loss = masked_mse_loss(pred_norm, batch["y"], batch["mask"])
            total_loss += float(loss.item())
            n_batches += 1

            pred_raw = pred_norm * std_t + mean_t
            y_raw = batch["y"] * std_t + mean_t
            meter.update(pred_raw, y_raw, batch["mask"])

    rmse = meter.compute_rmse()
    nrmse = rmse_to_nrmse(rmse, nrmse_denom)
    metrics = summarize_channel_metrics(rmse, nrmse)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.output_dir is not None:
        cfg["train"]["output_dir"] = args.output_dir
    if args.max_train_batches is not None:
        cfg["train"]["max_train_batches"] = args.max_train_batches
    if args.max_eval_batches is not None:
        cfg["train"]["max_eval_batches"] = args.max_eval_batches
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.patience is not None:
        cfg["train"]["patience"] = args.patience

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    set_seed(int(train_cfg["seed"]))
    device = resolve_device(train_cfg["device"])
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "resolved_config.yaml")

    years = sorted(set(data_cfg["train_years"] + data_cfg["test_years"]))
    reader = ZipNetCDFReader(
        root_dir=data_cfg["root_dir"],
        years=years,
        cache_size=data_cfg.get("cache_size", 8),
    )
    refs = reader.build_index(
        start_time=parse_datetime64(data_cfg["train_start"]),
        end_time=parse_datetime64(data_cfg["test_end"]),
    )
    split = split_refs(refs, data_cfg)

    input_len = int(data_cfg["input_len"])
    pred_len = int(data_cfg["pred_len"])
    stride = int(data_cfg.get("stride", 1))

    train_windows = build_window_starts(split.train_main, input_len, pred_len, stride=stride)
    dev_windows = build_window_starts(split.dev, input_len, pred_len, stride=stride)
    test_windows = build_window_starts(split.test, input_len, pred_len, stride=stride)
    if not train_windows:
        raise RuntimeError("No training windows found. Check split range and continuity settings.")
    if not dev_windows:
        raise RuntimeError("No dev windows found. Check split range and continuity settings.")
    if not test_windows:
        raise RuntimeError("No test windows found. Check split range and continuity settings.")

    stats = compute_normalization_stats(reader, split.train_main)
    save_stats(stats, output_dir / "stats.npz")

    train_ds = OceanSeqDataset(reader, split.train_main, train_windows, input_len, pred_len, stats)
    dev_ds = OceanSeqDataset(reader, split.dev, dev_windows, input_len, pred_len, stats)
    test_ds = OceanSeqDataset(reader, split.test, test_windows, input_len, pred_len, stats)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=bool(train_cfg.get("shuffle_train", False)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
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
        input_channels=4,
        output_channels=3,
        hidden_dims=model_cfg["hidden_dims"],
        kernel_size=model_cfg["kernel_size"],
        dropout=model_cfg.get("dropout", 0.0),
        default_pred_len=pred_len,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_dev = float("inf")
    bad_epochs = 0
    best_ckpt = output_dir / "best.ckpt"
    last_ckpt = output_dir / "last.ckpt"
    log_path = output_dir / "metrics.jsonl"

    mean_t = torch.tensor(stats["target_mean"], dtype=torch.float32, device=device).view(1, 1, 3, 1, 1)
    std_t = torch.tensor(stats["target_std"], dtype=torch.float32, device=device).view(1, 1, 3, 1, 1)

    epochs = int(train_cfg["epochs"])
    max_train_batches = train_cfg.get("max_train_batches", None)
    max_eval_batches = train_cfg.get("max_eval_batches", None)
    patience = int(train_cfg["patience"])

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0

        for b_idx, batch in enumerate(train_loader):
            if max_train_batches is not None and b_idx >= int(max_train_batches):
                break
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            if use_amp and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=True)
            elif use_amp:
                autocast_ctx = torch.cuda.amp.autocast(enabled=True)
            else:
                autocast_ctx = nullcontext()

            with autocast_ctx:
                pred = model(batch["x"])
                loss = masked_mse_loss(pred, batch["y"], batch["mask"])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += float(loss.item())
            n_train_batches += 1

        train_loss = total_train_loss / max(n_train_batches, 1)
        dev_metrics = evaluate(
            model,
            dev_loader,
            device=device,
            target_mean=stats["target_mean"],
            target_std=stats["target_std"],
            nrmse_denom=stats["nrmse_denom"],
            max_batches=max_eval_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_nrmse_mean": dev_metrics["nrmse_mean"],
            "dev_nrmse_sst": dev_metrics["nrmse_sst"],
            "dev_nrmse_sss": dev_metrics["nrmse_sss"],
            "dev_nrmse_speed": dev_metrics["nrmse_speed"],
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False))

        ckpt_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg,
            "stats": {k: np.asarray(v) for k, v in stats.items()},
            "dev_metrics": dev_metrics,
        }
        torch.save(ckpt_payload, last_ckpt)

        if dev_metrics["nrmse_mean"] < best_dev:
            best_dev = dev_metrics["nrmse_mean"]
            bad_epochs = 0
            torch.save(ckpt_payload, best_ckpt)
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping triggered at epoch={epoch}.")
            break

    best = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    test_metrics = evaluate(
        model,
        test_loader,
        device=device,
        target_mean=stats["target_mean"],
        target_std=stats["target_std"],
        nrmse_denom=stats["nrmse_denom"],
        max_batches=max_eval_batches,
    )
    result = {
        "best_dev_nrmse_mean": float(best_dev),
        "test_metrics": test_metrics,
        "device": str(device),
        "best_checkpoint": str(best_ckpt),
    }
    with (output_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

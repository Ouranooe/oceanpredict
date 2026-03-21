from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke run: tiny prepare + train + infer.")
    parser.add_argument("--config", default="configs/convlstm_baseline.yaml")
    parser.add_argument("--output_dir", default="outputs/smoke")
    parser.add_argument("--prepared_dir", default=None, help="Optional prepared dataset output dir.")
    parser.add_argument("--start_time", default="2014-06-01T00:00:00")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = Path(args.prepared_dir) if args.prepared_dir else (output_dir / "prepared")

    prepare_cmd = [
        "python",
        "scripts/prepare_dataset.py",
        "--config",
        args.config,
        "--out_root",
        str(prepared_dir),
    ]
    subprocess.run(prepare_cmd, check=True)

    train_cmd = [
        "python",
        "train.py",
        "--config",
        str(prepared_dir / "train_ready.yaml"),
        "--output_dir",
        str(output_dir),
        "--epochs",
        "1",
        "--patience",
        "1",
        "--max_train_batches",
        "2",
        "--max_eval_batches",
        "2",
    ]
    subprocess.run(train_cmd, check=True)

    infer_cmd = [
        "python",
        "infer.py",
        "--config",
        args.config,
        "--ckpt",
        str(output_dir / "best.ckpt"),
        "--start_time",
        args.start_time,
        "--viz_dir",
        str(output_dir / "viz"),
        "--auto_shift_start",
    ]
    subprocess.run(infer_cmd, check=True)


if __name__ == "__main__":
    main()

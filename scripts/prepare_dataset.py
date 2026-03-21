from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocean_forecast.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper: build base dataset from YAML and write train-ready config "
            "for train auto-build mode."
        )
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--out_root", required=True, help="Output directory (contains base + train_ready.yaml).")
    return parser.parse_args()


def _dump_train_ready(cfg: dict, out_root: Path, base_root: Path) -> Path:
    data_cfg = cfg["data"]
    train_ready = {
        "data": {
            "base_root": str(base_root),
            "input_len": int(data_cfg["input_len"]),
            "pred_len": int(data_cfg["pred_len"]),
            "stride": int(data_cfg.get("stride", 1)),
            "train_start": str(data_cfg["train_start"]),
            "train_main_end": str(data_cfg["train_main_end"]),
            "dev_start": str(data_cfg["dev_start"]),
            "dev_end": str(data_cfg["dev_end"]),
            "test_start": str(data_cfg["test_start"]),
            "test_end": str(data_cfg["test_end"]),
        },
        "model": cfg["model"],
        "train": cfg["train"],
    }
    dst = out_root / "train_ready.yaml"
    with dst.open("w", encoding="utf-8") as f:
        yaml.safe_dump(train_ready, f, sort_keys=False, allow_unicode=True)
    return dst


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    base_root = out_root / "base"
    years = sorted(set(int(y) for y in (data_cfg["train_years"] + data_cfg["test_years"])))
    years_arg = ",".join(str(y) for y in years)

    cmd1 = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_base_dataset.py"),
        "--src_root",
        str(data_cfg["root_dir"]),
        "--out_root",
        str(base_root),
        "--years",
        years_arg,
    ]
    subprocess.run(cmd1, check=True)

    train_ready = _dump_train_ready(cfg, out_root, base_root)
    print(
        f"[prepare_dataset] done: base_root={base_root}, train_ready={train_ready}. "
        "Training will auto-build split/windows/stats cache."
    )


if __name__ == "__main__":
    main()

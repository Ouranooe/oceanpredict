from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import io
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple
import zipfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocean_forecast.config import load_config
from ocean_forecast.data.zip_reader import INPUT_VARIABLES


@dataclass(frozen=True)
class TimeRef:
    member_name: str
    time_idx: int
    hour_index: int


def _as_scalar_attr(value, default: float) -> float:
    if value is None:
        return float(default)
    arr = np.array(value).reshape(-1)
    if arr.size == 0:
        return float(default)
    return float(arr[0])


def _list_nc_members(zip_path: Path) -> List[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        return sorted(
            n for n in zf.namelist() if n.endswith(".nc") and "__MACOSX" not in n
        )


def _read_member_bytes(zf: zipfile.ZipFile, member_name: str) -> bytes:
    with zf.open(member_name, "r") as entry:
        return entry.read()


def _decode_member_channels(nc_bytes: bytes) -> np.ndarray:
    with h5py.File(io.BytesIO(nc_bytes), "r") as ds:
        channels = []
        for var in INPUT_VARIABLES:
            raw = np.array(ds[var][:], dtype=np.float32)
            fill_value = _as_scalar_attr(ds[var].attrs.get("_FillValue"), np.nan)
            scale_factor = _as_scalar_attr(ds[var].attrs.get("scale_factor"), 1.0)
            add_offset = _as_scalar_attr(ds[var].attrs.get("add_offset"), 0.0)
            raw[raw == fill_value] = np.nan
            decoded = raw * scale_factor + add_offset
            channels.append(decoded)
        return np.stack(channels, axis=1).astype(np.float32)  # [T, 4, H, W]


def _collect_year_refs(zip_path: Path) -> Tuple[List[TimeRef], np.ndarray | None, np.ndarray | None]:
    refs: List[TimeRef] = []
    seen_hours = set()
    lat = None
    lon = None

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member_name in _list_nc_members(zip_path):
            nc_bytes = _read_member_bytes(zf, member_name)
            with h5py.File(io.BytesIO(nc_bytes), "r") as ds:
                time_vals = np.array(ds["time"][:], dtype=np.float64)
                if time_vals.size == 0:
                    continue
                if lat is None:
                    lat = np.array(ds["lat"][:], dtype=np.float32)
                    lon = np.array(ds["lon"][:], dtype=np.float32)

                hour_indices = np.rint(time_vals).astype(np.int64)
                for t_idx, hour_idx in enumerate(hour_indices):
                    h = int(hour_idx)
                    if h in seen_hours:
                        continue
                    refs.append(TimeRef(member_name=member_name, time_idx=int(t_idx), hour_index=h))
                    seen_hours.add(h)

    refs.sort(key=lambda r: r.hour_index)
    return refs, lat, lon


def convert_year_zip_to_npy(zip_path: Path, out_root: Path, year: int) -> Tuple[int, np.ndarray | None, np.ndarray | None]:
    refs, lat, lon = _collect_year_refs(zip_path)
    if not refs:
        return 0, lat, lon

    if lat is None or lon is None:
        raise RuntimeError(f"Missing lat/lon while parsing {zip_path}")
    h, w = int(lat.shape[0]), int(lon.shape[0])

    out_root.mkdir(parents=True, exist_ok=True)
    data_path = out_root / f"{year}_data.npy"
    hours_path = out_root / f"{year}_hours.npy"

    mm = np.lib.format.open_memmap(
        data_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(refs), 4, h, w),
    )
    hours = np.array([r.hour_index for r in refs], dtype=np.int64)
    np.save(hours_path, hours)

    by_member: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for out_idx, ref in enumerate(refs):
        by_member[ref.member_name].append((out_idx, ref.time_idx))

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member_name in sorted(by_member.keys()):
            nc_bytes = _read_member_bytes(zf, member_name)
            decoded = _decode_member_channels(nc_bytes)  # [T,4,H,W]
            for out_idx, t_idx in by_member[member_name]:
                mm[out_idx] = decoded[t_idx]

    mm.flush()
    return len(refs), lat, lon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build yearly NPY shards from zipped NetCDF data.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--out_root", required=True, help="Output root dir for yearly NPY files.")
    parser.add_argument(
        "--years",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit years. Default: union(train_years, test_years) from config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    src_root = Path(data_cfg["root_dir"])
    out_root = Path(args.out_root)
    years = list(args.years) if args.years else sorted(set(data_cfg["train_years"] + data_cfg["test_years"]))

    print(f"[standardize] src_root={src_root}")
    print(f"[standardize] out_root={out_root}")
    print(f"[standardize] years={years}")

    out_root.mkdir(parents=True, exist_ok=True)
    lat_saved = None
    lon_saved = None
    converted = []

    for year in years:
        zip_path = src_root / f"{int(year)}.zip"
        if not zip_path.exists():
            print(f"[standardize] skip year={year}: missing {zip_path}")
            continue

        print(f"[standardize] year={year}: indexing + decoding from {zip_path.name} ...")
        n_refs, lat, lon = convert_year_zip_to_npy(zip_path, out_root, int(year))
        if n_refs == 0:
            print(f"[standardize] year={year}: no valid frames (empty)")
            continue

        if lat_saved is None:
            lat_saved = lat
            lon_saved = lon
        else:
            if lat is not None and not np.array_equal(lat_saved, lat):
                raise RuntimeError(f"lat mismatch detected in year={year}")
            if lon is not None and not np.array_equal(lon_saved, lon):
                raise RuntimeError(f"lon mismatch detected in year={year}")

        converted.append({"year": int(year), "frames": int(n_refs)})
        print(f"[standardize] year={year}: wrote frames={n_refs}")

    if not converted:
        raise RuntimeError("No years converted. Check input root and year list.")

    assert lat_saved is not None and lon_saved is not None
    np.save(out_root / "lat.npy", lat_saved.astype(np.float32))
    np.save(out_root / "lon.npy", lon_saved.astype(np.float32))
    meta = {
        "version": 1,
        "format": "npy_yearly",
        "variables": list(INPUT_VARIABLES),
        "years": [row["year"] for row in converted],
        "converted": converted,
        "source_root": str(src_root),
        "config_path": str(Path(args.config)),
    }
    with (out_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[standardize] done. years={len(converted)}, out_root={out_root}")


if __name__ == "__main__":
    main()

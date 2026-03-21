from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocean_forecast.data.npy_reader import NpyYearReader
from scripts.build_standard_dataset import convert_year_zip_to_npy


DEFAULT_START_YEAR = 1994
DEFAULT_END_YEAR = 2014


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reusable base dataset for all experiments (no YAML required)."
    )
    parser.add_argument("--src_root", required=True, help="Raw source root containing yearly .zip files.")
    parser.add_argument("--out_root", required=True, help="Output base dataset directory.")
    parser.add_argument(
        "--years",
        default=f"{DEFAULT_START_YEAR}:{DEFAULT_END_YEAR}",
        help="Year range/list, e.g. 1994:2014 or 1994,1995,1996",
    )
    return parser.parse_args()


def _parse_years(text: str) -> List[int]:
    s = text.strip()
    if ":" in s:
        left, right = s.split(":", 1)
        start, end = int(left), int(right)
        if start > end:
            raise ValueError(f"Invalid years range: {text}")
        return list(range(start, end + 1))
    years = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not years:
        raise ValueError("No years parsed from --years.")
    return sorted(set(years))


def _copy_year_npy(src_root: Path, out_root: Path, year: int) -> int:
    src_data = src_root / f"{year}_data.npy"
    src_hours = src_root / f"{year}_hours.npy"
    if not src_data.exists() or not src_hours.exists():
        raise FileNotFoundError(f"Missing yearly npy source for year={year} under {src_root}")

    dst_data = out_root / src_data.name
    dst_hours = out_root / src_hours.name
    if src_data.resolve() != dst_data.resolve():
        shutil.copy2(src_data, dst_data)
    if src_hours.resolve() != dst_hours.resolve():
        shutil.copy2(src_hours, dst_hours)
    hours = np.asarray(np.load(dst_hours), dtype=np.int64)
    return int(hours.size)


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    years = _parse_years(args.years)

    print(f"[prepare_base] src_root={src_root}")
    print(f"[prepare_base] out_root={out_root}")
    print(f"[prepare_base] years={years}")

    converted = []
    lat_saved = None
    lon_saved = None
    for year in years:
        zip_path = src_root / f"{year}.zip"
        npy_data = src_root / f"{year}_data.npy"
        npy_hours = src_root / f"{year}_hours.npy"
        if zip_path.exists():
            print(f"[prepare_base] converting year={year} from {zip_path.name}")
            n_refs, lat, lon = convert_year_zip_to_npy(zip_path, out_root, year=year)
            if n_refs <= 0:
                raise RuntimeError(f"Year {year} has no valid frames after conversion.")
            if lat_saved is None:
                lat_saved = lat
                lon_saved = lon
        elif npy_data.exists() and npy_hours.exists():
            print(f"[prepare_base] copying year={year} from yearly npy source")
            n_refs = _copy_year_npy(src_root, out_root, year)
        else:
            raise FileNotFoundError(
                f"Missing source data for year={year}. Need either {zip_path.name} "
                f"or {npy_data.name}/{npy_hours.name} under {src_root}"
            )
        converted.append({"year": int(year), "frames": int(n_refs)})
        print(f"[prepare_base] year={year} frames={n_refs}")

    lat_src = src_root / "lat.npy"
    lon_src = src_root / "lon.npy"
    if lat_src.exists() and lon_src.exists():
        if lat_src.resolve() != (out_root / "lat.npy").resolve():
            shutil.copy2(lat_src, out_root / "lat.npy")
        if lon_src.resolve() != (out_root / "lon.npy").resolve():
            shutil.copy2(lon_src, out_root / "lon.npy")
    else:
        if lat_saved is None or lon_saved is None:
            raise RuntimeError("lat/lon not available from source.")
        np.save(out_root / "lat.npy", lat_saved.astype(np.float32))
        np.save(out_root / "lon.npy", lon_saved.astype(np.float32))

    reader = NpyYearReader(root_dir=out_root, years=years)
    refs = reader.build_index()
    if not refs:
        raise RuntimeError("No refs indexed from base dataset.")
    hour_index = np.asarray([int(r.hour_index) for r in refs], dtype=np.int64)
    year_arr = np.asarray([int(r.member_name) for r in refs], dtype=np.int32)
    time_idx = np.asarray([int(r.time_idx) for r in refs], dtype=np.int32)
    np.savez_compressed(
        out_root / "refs_global.npz",
        hour_index=hour_index,
        year=year_arr,
        time_idx=time_idx,
    )
    reader.close()

    manifest = {
        "version": 1,
        "task": "sea_forecast",
        "format": "base_v1",
        "source_root": str(src_root),
        "years": [int(x["year"]) for x in converted],
        "converted": converted,
        "n_refs_global": int(len(refs)),
        "refs_global_file": "refs_global.npz",
        "lat_file": "lat.npy",
        "lon_file": "lon.npy",
    }
    with (out_root / "manifest_base.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[prepare_base] done: refs_global={len(refs)}")


if __name__ == "__main__":
    main()

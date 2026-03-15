from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import io
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import zipfile

import h5py
import numpy as np


BASE_TIME = np.datetime64("2000-01-01T00:00:00")
INPUT_VARIABLES = ("sst", "sss", "ssu", "ssv")


@dataclass(frozen=True)
class FrameRef:
    timestamp: np.datetime64
    hour_index: int
    zip_path: str
    member_name: str
    time_idx: int


def parse_datetime64(text: str) -> np.datetime64:
    return np.datetime64(text).astype("datetime64[h]")


def hour_index_to_datetime(hour_index: int) -> np.datetime64:
    return BASE_TIME + np.timedelta64(int(hour_index), "h")


def datetime_to_hour_index(value: np.datetime64) -> int:
    delta = value.astype("datetime64[h]") - BASE_TIME
    return int(delta / np.timedelta64(1, "h"))


def _as_scalar_attr(value, default: float) -> float:
    if value is None:
        return float(default)
    arr = np.array(value).reshape(-1)
    if arr.size == 0:
        return float(default)
    return float(arr[0])


class ZipNetCDFReader:
    """Read HYCOM NetCDF frames directly from yearly zip archives via h5py."""

    def __init__(self, root_dir: str | Path, years: Sequence[int | str], cache_size: int = 8):
        self.root_dir = Path(root_dir)
        self.cache_size = int(cache_size)
        self.zip_paths: List[Path] = []
        for y in years:
            zp = self.root_dir / f"{int(y)}.zip"
            if zp.exists():
                self.zip_paths.append(zp)
        if not self.zip_paths:
            raise FileNotFoundError(f"No yearly zip files found under {self.root_dir}")

        self._member_cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        self._lat = None
        self._lon = None

    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._lat is None or self._lon is None:
            _ = self.build_index()
        return self._lat.copy(), self._lon.copy()

    def build_index(
        self,
        start_time: np.datetime64 | None = None,
        end_time: np.datetime64 | None = None,
    ) -> List[FrameRef]:
        refs: List[FrameRef] = []
        seen_hours = set()

        for zip_path in sorted(self.zip_paths):
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = sorted(
                    n for n in zf.namelist() if n.endswith(".nc") and "__MACOSX" not in n
                )
                for member in members:
                    with zf.open(member, "r") as entry:
                        nc_bytes = entry.read()
                    with h5py.File(io.BytesIO(nc_bytes), "r") as ds:
                        time_vals = np.array(ds["time"][:], dtype=np.float64)
                        if time_vals.size == 0:
                            continue

                        if self._lat is None:
                            self._lat = np.array(ds["lat"][:], dtype=np.float32)
                            self._lon = np.array(ds["lon"][:], dtype=np.float32)

                        hour_indices = np.rint(time_vals).astype(np.int64)
                        for t_idx, hour_idx in enumerate(hour_indices):
                            hour_idx = int(hour_idx)
                            if hour_idx in seen_hours:
                                continue
                            ts = hour_index_to_datetime(hour_idx)
                            if start_time is not None and ts < start_time:
                                continue
                            if end_time is not None and ts > end_time:
                                continue
                            refs.append(
                                FrameRef(
                                    timestamp=ts,
                                    hour_index=hour_idx,
                                    zip_path=str(zip_path),
                                    member_name=member,
                                    time_idx=int(t_idx),
                                )
                            )
                            seen_hours.add(hour_idx)

        refs.sort(key=lambda r: r.hour_index)
        return refs

    def read_frame(self, ref: FrameRef) -> Tuple[np.ndarray, np.ndarray]:
        cached = self._load_member(ref.zip_path, ref.member_name)
        frame = cached["data"][ref.time_idx]
        valid = np.isfinite(frame[0]) & np.isfinite(frame[1]) & np.isfinite(frame[2]) & np.isfinite(frame[3])
        return frame, valid

    def _load_member(self, zip_path: str, member_name: str) -> Dict[str, np.ndarray]:
        key = f"{zip_path}::{member_name}"
        if key in self._member_cache:
            self._member_cache.move_to_end(key)
            return self._member_cache[key]

        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(member_name, "r") as entry:
                nc_bytes = entry.read()
        with h5py.File(io.BytesIO(nc_bytes), "r") as ds:
            time_vals = np.array(ds["time"][:], dtype=np.float64)
            hour_indices = np.rint(time_vals).astype(np.int64)

            channels = []
            for var in INPUT_VARIABLES:
                raw = np.array(ds[var][:], dtype=np.float32)
                fill_value = _as_scalar_attr(ds[var].attrs.get("_FillValue"), np.nan)
                scale_factor = _as_scalar_attr(ds[var].attrs.get("scale_factor"), 1.0)
                add_offset = _as_scalar_attr(ds[var].attrs.get("add_offset"), 0.0)
                raw[raw == fill_value] = np.nan
                decoded = raw * scale_factor + add_offset
                channels.append(decoded)

            data = np.stack(channels, axis=1)  # [T, C=4, H, W]

        payload = {
            "time_hours": hour_indices,
            "data": data.astype(np.float32),
        }
        self._member_cache[key] = payload
        self._member_cache.move_to_end(key)
        while len(self._member_cache) > self.cache_size:
            self._member_cache.popitem(last=False)
        return payload


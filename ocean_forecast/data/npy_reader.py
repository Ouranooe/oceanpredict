from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .zip_reader import FrameRef, hour_index_to_datetime


@dataclass(frozen=True)
class YearFiles:
    year: int
    data_path: Path
    hours_path: Path


class NpyYearReader:
    """Read preprocessed yearly NPY shards with mmap."""

    def __init__(self, root_dir: str | Path, years: Sequence[int | str]):
        self.root_dir = Path(root_dir)
        self.year_files: List[YearFiles] = []
        for y in years:
            year = int(y)
            data_path = self.root_dir / f"{year}_data.npy"
            hours_path = self.root_dir / f"{year}_hours.npy"
            if data_path.exists() and hours_path.exists():
                self.year_files.append(YearFiles(year=year, data_path=data_path, hours_path=hours_path))

        if not self.year_files:
            raise FileNotFoundError(f"No yearly npy files found under {self.root_dir}")

        self._hours_cache: Dict[int, np.ndarray] = {}
        self._data_cache: Dict[int, np.ndarray] = {}
        self._lat = None
        self._lon = None

    def close(self) -> None:
        for arr in self._data_cache.values():
            mmap_obj = getattr(arr, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
        self._data_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _load_year_hours(self, year: int) -> np.ndarray:
        if year not in self._hours_cache:
            hours_path = next(yf.hours_path for yf in self.year_files if yf.year == year)
            arr = np.load(hours_path)
            self._hours_cache[year] = np.asarray(arr, dtype=np.int64)
        return self._hours_cache[year]

    def _load_year_data(self, year: int) -> np.ndarray:
        if year not in self._data_cache:
            data_path = next(yf.data_path for yf in self.year_files if yf.year == year)
            arr = np.load(data_path, mmap_mode="r")
            if arr.ndim != 4 or arr.shape[1] != 4:
                raise ValueError(f"Expected [T,4,H,W] data in {data_path}, got {arr.shape}")
            self._data_cache[year] = arr
        return self._data_cache[year]

    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._lat is None or self._lon is None:
            lat_path = self.root_dir / "lat.npy"
            lon_path = self.root_dir / "lon.npy"
            if not lat_path.exists() or not lon_path.exists():
                raise FileNotFoundError(
                    f"lat/lon files not found under {self.root_dir}. Expected lat.npy and lon.npy."
                )
            self._lat = np.asarray(np.load(lat_path), dtype=np.float32)
            self._lon = np.asarray(np.load(lon_path), dtype=np.float32)
        return self._lat.copy(), self._lon.copy()

    def build_index(
        self,
        start_time: np.datetime64 | None = None,
        end_time: np.datetime64 | None = None,
    ) -> List[FrameRef]:
        refs: List[FrameRef] = []
        seen_hours = set()
        for yf in sorted(self.year_files, key=lambda x: x.year):
            hours = self._load_year_hours(yf.year)
            for t_idx, hour_idx in enumerate(hours):
                h = int(hour_idx)
                if h in seen_hours:
                    continue
                ts = hour_index_to_datetime(h)
                if start_time is not None and ts < start_time:
                    continue
                if end_time is not None and ts > end_time:
                    continue
                refs.append(
                    FrameRef(
                        timestamp=ts,
                        hour_index=h,
                        zip_path=str(yf.data_path),
                        member_name=str(yf.year),
                        time_idx=int(t_idx),
                    )
                )
                seen_hours.add(h)
        refs.sort(key=lambda r: r.hour_index)
        return refs

    def read_frame(self, ref: FrameRef) -> Tuple[np.ndarray, np.ndarray]:
        year = int(ref.member_name)
        data = self._load_year_data(year)
        frame = np.asarray(data[ref.time_idx], dtype=np.float32)
        valid = np.isfinite(frame[0]) & np.isfinite(frame[1]) & np.isfinite(frame[2]) & np.isfinite(frame[3])
        return frame, valid

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .dataset import SplitRefs, load_stats
from .zip_reader import FrameRef, hour_index_to_datetime


@dataclass(frozen=True)
class PreparedManifest:
    input_len: int
    pred_len: int
    stride: int
    years: List[int]
    train_years: List[int]
    test_years: List[int]
    train_stride_schedule: List[Dict[str, Any]]
    train_stride_summary: List[Dict[str, Any]]


class PreparedReader:
    """Reader over precomputed training artifacts under a prepared root."""

    def __init__(self, prepared_root: str | Path):
        self.prepared_root = Path(prepared_root)
        if not self.prepared_root.exists():
            raise FileNotFoundError(f"prepared_root not found: {self.prepared_root}")

        manifest_path = self.prepared_root / "manifest_exp.json"
        if not manifest_path.exists():
            manifest_path = self.prepared_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest missing: expected {self.prepared_root / 'manifest_exp.json'} or "
                f"{self.prepared_root / 'manifest.json'}"
            )
        with manifest_path.open("r", encoding="utf-8") as f:
            raw_manifest = json.load(f)
        base_root = raw_manifest.get("base_root")
        self.base_root = Path(base_root) if base_root else self.prepared_root
        if not self.base_root.exists():
            raise FileNotFoundError(f"base_root not found: {self.base_root}")

        years = [int(y) for y in raw_manifest.get("years", [])]
        if not years:
            base_manifest = self.base_root / "manifest_base.json"
            if base_manifest.exists():
                with base_manifest.open("r", encoding="utf-8") as f:
                    years = [int(y) for y in json.load(f).get("years", [])]
        self.manifest = PreparedManifest(
            input_len=int(raw_manifest["input_len"]),
            pred_len=int(raw_manifest["pred_len"]),
            stride=int(raw_manifest["stride"]),
            years=years,
            train_years=[int(y) for y in raw_manifest.get("train_years", [])],
            test_years=[int(y) for y in raw_manifest.get("test_years", [])],
            train_stride_schedule=list(raw_manifest.get("train_stride_schedule", [])),
            train_stride_summary=list(raw_manifest.get("train_stride_summary", [])),
        )

        self._lat = None
        self._lon = None
        self._refs_cache = None
        self._data_cache: Dict[int, np.ndarray] = {}

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

    def _load_year_data(self, year: int) -> np.ndarray:
        if year not in self._data_cache:
            data_path = self.base_root / f"{int(year)}_data.npy"
            if not data_path.exists():
                raise FileNotFoundError(f"missing yearly shard: {data_path}")
            arr = np.load(data_path, mmap_mode="r")
            if arr.ndim != 4 or arr.shape[1] != 4:
                raise ValueError(f"Expected [T,4,H,W] data in {data_path}, got {arr.shape}")
            self._data_cache[year] = arr
        return self._data_cache[year]

    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._lat is None or self._lon is None:
            lat_path = self.base_root / "lat.npy"
            lon_path = self.base_root / "lon.npy"
            if not lat_path.exists() or not lon_path.exists():
                raise FileNotFoundError(f"missing lat/lon in base_root={self.base_root}")
            self._lat = np.asarray(np.load(lat_path), dtype=np.float32)
            self._lon = np.asarray(np.load(lon_path), dtype=np.float32)
        return self._lat.copy(), self._lon.copy()

    def read_frame(self, ref: FrameRef) -> Tuple[np.ndarray, np.ndarray]:
        year = int(ref.member_name)
        data = self._load_year_data(year)
        frame = np.asarray(data[ref.time_idx], dtype=np.float32)
        valid = np.isfinite(frame[0]) & np.isfinite(frame[1]) & np.isfinite(frame[2]) & np.isfinite(frame[3])
        return frame, valid

    def load_stats(self) -> Dict[str, np.ndarray]:
        stats_path = self.prepared_root / "stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(f"missing stats file: {stats_path}")
        return load_stats(stats_path)

    def load_refs(self) -> List[FrameRef]:
        if self._refs_cache is not None:
            return list(self._refs_cache)

        refs_path = self.base_root / "refs_global.npz"
        if not refs_path.exists():
            refs_path = self.prepared_root / "refs.npz"
        if not refs_path.exists():
            raise FileNotFoundError(
                f"missing refs file: expected {self.base_root / 'refs_global.npz'} "
                f"or {self.prepared_root / 'refs.npz'}"
            )
        raw = np.load(refs_path)
        hour_index = np.asarray(raw["hour_index"], dtype=np.int64)
        years = np.asarray(raw["year"], dtype=np.int64)
        time_idx = np.asarray(raw["time_idx"], dtype=np.int64)

        refs: List[FrameRef] = []
        for h, y, t in zip(hour_index, years, time_idx):
            year_i = int(y)
            refs.append(
                FrameRef(
                    timestamp=hour_index_to_datetime(int(h)),
                    hour_index=int(h),
                    zip_path=str(self.base_root / f"{year_i}_data.npy"),
                    member_name=str(year_i),
                    time_idx=int(t),
                )
            )
        self._refs_cache = refs
        return list(self._refs_cache)

    def load_split_refs(self) -> SplitRefs:
        split_path = self.prepared_root / "split_refs.npz"
        if not split_path.exists():
            raise FileNotFoundError(f"missing split refs file: {split_path}")
        split = np.load(split_path)
        refs = self.load_refs()

        train_idx = np.asarray(split["train_main_idx"], dtype=np.int64)
        dev_idx = np.asarray(split["dev_idx"], dtype=np.int64)
        test_idx = np.asarray(split["test_idx"], dtype=np.int64)
        return SplitRefs(
            train_main=[refs[int(i)] for i in train_idx],
            dev=[refs[int(i)] for i in dev_idx],
            test=[refs[int(i)] for i in test_idx],
        )

    def load_windows(self, split_name: str) -> np.ndarray:
        name_map = {
            "train": "train_windows.npy",
            "dev": "dev_windows.npy",
            "test": "test_windows.npy",
        }
        if split_name not in name_map:
            raise ValueError(f"Unknown split_name={split_name}. Expected one of {sorted(name_map)}")
        path = self.prepared_root / name_map[split_name]
        if not path.exists():
            raise FileNotFoundError(f"missing windows file: {path}")
        return np.asarray(np.load(path), dtype=np.int64)

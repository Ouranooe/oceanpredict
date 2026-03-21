from __future__ import annotations

from typing import Any, Sequence

from .npy_reader import NpyYearReader
from .zip_reader import ZipNetCDFReader


def build_reader(data_cfg: dict[str, Any], years: Sequence[int | str]):
    data_format = str(data_cfg.get("format", "zip")).lower()
    if data_format == "zip":
        return ZipNetCDFReader(
            root_dir=data_cfg["root_dir"],
            years=years,
            cache_size=int(data_cfg.get("cache_size", 8)),
        )
    if data_format == "npy_yearly":
        return NpyYearReader(
            root_dir=data_cfg["root_dir"],
            years=years,
        )

    raise ValueError(f"Unknown data.format='{data_format}'. Supported: zip, npy_yearly")

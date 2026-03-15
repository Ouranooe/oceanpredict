from __future__ import annotations

import tempfile
from pathlib import Path
import zipfile

import h5py
import numpy as np

from ocean_forecast.data.dataset import build_window_starts
from ocean_forecast.data.zip_reader import FrameRef, ZipNetCDFReader, hour_index_to_datetime


def _create_nc(path: Path, with_data: bool = True) -> None:
    with h5py.File(path, "w") as f:
        lat = np.array([31.0, 31.08], dtype=np.float32)
        lon = np.array([117.0, 117.08, 117.16], dtype=np.float32)
        f.create_dataset("lat", data=lat)
        f.create_dataset("lon", data=lon)
        if with_data:
            time = np.array([0.2, 1.4], dtype=np.float64)
            shape = (2, 2, 3)
            sst = np.full(shape, 100, dtype=np.int16)
            sss = np.full(shape, 200, dtype=np.int16)
            ssu = np.full(shape, 300, dtype=np.int16)
            ssv = np.full(shape, 400, dtype=np.int16)
            sst[0, 0, 0] = -30000
            f.create_dataset("time", data=time)
            for name, arr, offset in [
                ("sst", sst, 1.0),
                ("sss", sss, 2.0),
                ("ssu", ssu, 0.0),
                ("ssv", ssv, 0.0),
            ]:
                ds = f.create_dataset(name, data=arr)
                ds.attrs["_FillValue"] = np.array([-30000], dtype=np.int16)
                ds.attrs["scale_factor"] = np.array([0.1], dtype=np.float32)
                ds.attrs["add_offset"] = np.array([offset], dtype=np.float32)
        else:
            f.create_dataset("time", data=np.array([], dtype=np.float64))
            shape = (0, 2, 3)
            for name in ("sst", "sss", "ssu", "ssv"):
                ds = f.create_dataset(name, data=np.empty(shape, dtype=np.int16))
                ds.attrs["_FillValue"] = np.array([-30000], dtype=np.int16)
                ds.attrs["scale_factor"] = np.array([0.1], dtype=np.float32)
                ds.attrs["add_offset"] = np.array([0.0], dtype=np.float32)


def test_zip_reader_decode_and_skip_empty() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        nc1 = root / "19940101.nc"
        nc2 = root / "19940102.nc"
        _create_nc(nc1, with_data=True)
        _create_nc(nc2, with_data=False)

        with zipfile.ZipFile(root / "1994.zip", "w") as zf:
            zf.write(nc1, arcname="19940101.nc")
            zf.write(nc2, arcname="19940102.nc")
            zf.writestr("__MACOSX/._19940101.nc", b"dummy")

        reader = ZipNetCDFReader(root_dir=root, years=[1994], cache_size=2)
        refs = reader.build_index()
        assert len(refs) == 2
        assert refs[0].hour_index == 0
        assert refs[1].hour_index == 1

        frame, mask = reader.read_frame(refs[0])
        assert frame.shape == (4, 2, 3)
        assert np.isnan(frame[0, 0, 0])  # fill value to NaN
        assert not mask[0, 0]
        assert np.isclose(frame[0, 0, 1], 11.0)  # 100*0.1 + 1.0
        assert np.isclose(frame[1, 1, 1], 22.0)  # 200*0.1 + 2.0


def test_build_window_starts_continuity() -> None:
    refs = []
    for h in [0, 1, 2, 4, 5, 6]:
        refs.append(
            FrameRef(
                timestamp=hour_index_to_datetime(h),
                hour_index=h,
                zip_path="x.zip",
                member_name="a.nc",
                time_idx=0,
            )
        )
    starts = build_window_starts(refs, input_len=2, pred_len=2, stride=1)
    # windows need 4 continuous hours, only [0,1,2,4] breaks; [2,4,5,6] also breaks
    assert starts == []


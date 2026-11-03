"""IMD 0.25-degree gridded rainfall, the primary truth (MODEL_SPEC 4.3).

imdlib layout: <imd_dir>/rain/<year>.grd. If the files are already there (e.g. uploaded to Drive
because the IMD server is unreachable from cloud machines), they are read in place.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr


def year_file(imd_dir: str | Path, year: int) -> Path:
    return Path(imd_dir) / "rain" / f"{year}.grd"


def missing_years(imd_dir: str | Path, years) -> list[int]:
    return [y for y in years if not year_file(imd_dir, y).exists()]


def download(imd_dir: str | Path, years, log=print, attempts: int = 3, wait_s: float = 10.0) -> list[int]:
    """Download missing years one by one, verifying each file; returns the years that still failed."""
    import time
    import imdlib as imd
    todo = missing_years(imd_dir, years)
    if not todo:
        log("imd: all years present")
        return []
    os.makedirs(imd_dir, exist_ok=True)                  # imdlib does not create its folder
    failed = []
    for y in todo:                                       # one year at a time: resumable
        for attempt in range(1, attempts + 1):
            log(f"imd: downloading {y} (attempt {attempt})")
            try:
                imd.get_data("rain", y, y, fn_format="yearwise", file_dir=str(imd_dir))
            except Exception as exc:                     # server hiccup: retry
                log(f"imd: {y} error {type(exc).__name__}: {exc}")
            f = year_file(imd_dir, y)
            if f.exists() and f.stat().st_size > 1_000_000:
                break
            if f.exists():
                f.unlink()                               # truncated file: remove and retry
            time.sleep(wait_s)
        else:
            failed.append(y)
    if failed:
        log(f"imd: WARNING years still missing after {attempts} attempts: {failed}")
    return failed


def clean(da: xr.DataArray) -> xr.DataArray:
    """-999 -> NaN, ascending lat, float32, named o_rain."""
    da = da.where(da > -998.0)
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.isel(lat=slice(None, None, -1))
    return da.astype("float32").rename("o_rain")


def open_years(imd_dir: str | Path, start: int, end: int) -> xr.DataArray:
    import imdlib as imd
    missing = missing_years(imd_dir, range(start, end + 1))
    if missing:
        raise FileNotFoundError(f"IMD years missing in {imd_dir}/rain: {missing}")
    data = imd.open_data("rain", start, end, "yearwise", str(imd_dir))
    return clean(data.get_xarray()["rain"])


def land_mask(o_rain: xr.DataArray, min_valid_frac: float = 0.95) -> xr.DataArray:
    """Cells with valid IMD data on >= 95% of days (TRD 2.1)."""
    return (o_rain.notnull().mean("time") >= min_valid_frac).rename("land")


def check_grid(o_rain: xr.DataArray, rain_box) -> None:
    """IMD grid must coincide with the working grid (both 0.25 degree, same extents)."""
    lat0, lat1, lon0, lon1 = rain_box
    ok = (np.isclose(float(o_rain.lat.min()), lat0) and np.isclose(float(o_rain.lat.max()), lat1)
          and np.isclose(float(o_rain.lon.min()), lon0) and np.isclose(float(o_rain.lon.max()), lon1))
    if not ok:
        raise ValueError(f"IMD grid {float(o_rain.lat.min())}-{float(o_rain.lat.max())}N, "
                         f"{float(o_rain.lon.min())}-{float(o_rain.lon.max())}E does not match rain_box {rain_box}")

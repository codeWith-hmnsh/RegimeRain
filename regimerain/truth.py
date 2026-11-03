"""Observed rainfall ("truth"), cached as zarr and indexed by RAIN-DAY START DATE (TRD 2.2).

IMD date D + `imd_date_offset_days` is the rain day starting on D; every downstream module sees truth
already shifted, so no other code needs to know IMD's date convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def truth_path(cfg: dict, year: int) -> Path:
    return Path(cfg["paths"]["raw"]) / "truth" / f"o_rain_{year}.zarr"


def prepare_truth(cfg: dict, years, log=print) -> None:
    """Convert IMD .grd years to zarr once (fast reads later)."""
    from regimerain.ingest import imd
    if cfg["sources"]["truth"] != "imd":
        raise NotImplementedError("only IMD truth is built (MODEL_SPEC 4.3)")
    for y in years:
        out = truth_path(cfg, y)
        if out.exists():
            continue
        da = imd.open_years(cfg["paths"]["imd"], y, y)
        imd.check_grid(da, cfg["grid"]["rain_box"])
        out.parent.mkdir(parents=True, exist_ok=True)
        da.to_dataset().to_zarr(out, mode="w")
        log(f"truth: {y} -> {out}")


def load_truth(cfg: dict, years) -> xr.DataArray:
    """o_rain(time=rain-day start date, lat, lon), mm."""
    parts = []
    for y in sorted(set(years)):
        p = truth_path(cfg, y)
        if not p.exists():
            raise FileNotFoundError(f"{p} missing - run `regimerain static` (it prepares truth) or `label` first")
        parts.append(xr.open_zarr(p)["o_rain"].load())
    o = xr.concat(parts, dim="time")
    off = int(cfg["sources"].get("imd_date_offset_days", 0))
    if off:
        o = o.assign_coords(time=o.time.values - np.timedelta64(off, "D"))
    return o.astype("float32")

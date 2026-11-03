"""Static layers, `basic` mode (pilot): land mask from IMD, approximate zones, no terrain.

The full mode (DEM terrain/coast classes, GADM zones and district weights, MODEL_SPEC section 5)
replaces this; `basic` exists so the pipeline runs end-to-end without the DEM/GADM downloads.
Approximate zone boxes (disclosed simplification of IMD's four homogeneous regions):
    south peninsula: lat < 18; northwest: lat >= 24 and lon < 84; east & NE: lon >= 84 and lat >= 21;
    central: the rest.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from regimerain.ingest.imd import land_mask

TERRAIN_VARS = ("elev_mean", "elev_std", "slope_mean", "windward", "dist_coast_km")


def approx_zones(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    zone = np.full(LAT.shape, 1, "int8")
    zone[LAT < 18] = 2
    zone[(LAT >= 24) & (LON < 84)] = 0
    zone[(LON >= 84) & (LAT >= 21)] = 3
    return zone


def build_static_basic(o_rain: xr.DataArray) -> xr.Dataset:
    land = land_mask(o_rain)
    lat, lon = land.lat.values, land.lon.values
    shape = (len(lat), len(lon))
    ds = xr.Dataset({"land": (("lat", "lon"), land.values.astype("int8")),
                     "zone": (("lat", "lon"), approx_zones(lat, lon)),
                     "geo": (("lat", "lon"), np.zeros(shape, "int8"))},
                    coords={"lat": lat, "lon": lon})
    for v in TERRAIN_VARS:
        ds[v] = (("lat", "lon"), np.zeros(shape, "float32"))
    ds.attrs["mode"] = "basic"
    return ds


def static_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["static"]) / "static.zarr"


def load_static(cfg: dict) -> xr.Dataset:
    return xr.open_zarr(static_path(cfg)).load()

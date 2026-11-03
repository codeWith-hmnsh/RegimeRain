"""IFS HRES from WeatherBench2 -> rain-day forecasts for the India domain (MODEL_SPEC 4.1-4.2, 4.5).

Confirmed in Phase 0 (docs/PHASE0_FINDINGS.md): `total_precipitation` is accumulated from init in
metres; leads are 6-hourly to 240 h; latitude ascending; `prediction_timedelta` may arrive as int64
hours. Per season this writes
    <raw>/hres/rain_<year>.zarr   f_rain(init, lead, lat, lon)                 rain domain, mm
    <raw>/hres/dyn_<year>.zarr    u850 v850 mslp w500 pw ivtx ivty (init, lead, lat, lon)  dynamics domain
Dynamics are rain-day means of the 6-hourly instants inside each 03-03 UTC window, and the
moisture column integrals are done here, so pressure levels never need to be stored.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from regimerain.features.dynamics import column_integrals
from regimerain.ingest.align import cumulative_from_buckets, lead_hours, rainday_totals, to_hours

PRECIP_VAR = {"cumulative": "total_precipitation", "tp6": "total_precipitation_6hr"}
COLUMN_LEVELS = [300, 400, 500, 600, 700, 850, 925, 1000]
INSTANT_HOURS = (6, 12, 18, 24)                 # inside rain day 1; +24 per later lead


def open_hres(url: str, **kw) -> xr.Dataset:
    ds = xr.open_zarr(url, storage_options={"token": "anon"}, **kw)
    return normalise(ds)


def normalise(ds: xr.Dataset) -> xr.Dataset:
    """WeatherBench2 names -> internal names; 00 UTC inits only; ascending latitude."""
    ren = {k: v for k, v in {"latitude": "lat", "longitude": "lon", "time": "init"}.items() if k in ds.dims or k in ds.coords}
    ds = ds.rename(ren)
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.isel(lat=slice(None, None, -1))
    hours = pd.DatetimeIndex(ds.init.values).hour
    return ds.isel(init=np.where(hours == 0)[0])


def subset(obj, box):
    lat0, lat1, lon0, lon1 = box
    return obj.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))


def season_inits(ds: xr.Dataset, year: int, max_lead: int) -> np.ndarray:
    """Inits whose rain days (lead 1..max_lead) can fall in 1 Jun - 30 Sep of `year`."""
    start = np.datetime64(f"{year}-06-01") - np.timedelta64(max_lead - 1, "D")
    end = np.datetime64(f"{year}-09-30")
    t = ds.init.values
    return t[(t >= start) & (t <= end)]


def _lead_index(ds: xr.Dataset, max_hour: float) -> np.ndarray:
    return np.where(lead_hours(ds.prediction_timedelta) <= max_hour + 1e-6)[0]


def rain_batch(ds: xr.Dataset, inits, leads, kind: str, box) -> xr.DataArray:
    """Rain-day totals (mm) for a batch of inits."""
    var = PRECIP_VAR[kind]
    idx = _lead_index(ds, 27 + 24 * (max(leads) - 1) + 3)
    tp = subset(ds[var], box).sel(init=inits).isel(prediction_timedelta=idx).load()
    tp = to_hours(tp)
    cum = tp if kind == "cumulative" else cumulative_from_buckets(tp)
    return rainday_totals(cum, leads)


def dyn_batch(ds: xr.Dataset, inits, leads, box) -> xr.Dataset:
    """Rain-day-mean dynamics (and column integrals) for a batch of inits."""
    want = sorted({h + 24 * (L - 1) for L in leads for h in INSTANT_HOURS})
    hrs = lead_hours(ds.prediction_timedelta)
    idx = [int(np.argmin(np.abs(hrs - h))) for h in want]
    if not np.allclose(hrs[idx], want):
        raise ValueError(f"missing instantaneous leads {want}")
    sub = subset(ds, box).sel(init=inits).isel(prediction_timedelta=idx)
    lev = sub.level.values
    need = [l for l in COLUMN_LEVELS if l in lev]
    if len(need) != len(COLUMN_LEVELS):
        raise ValueError(f"missing levels: {sorted(set(COLUMN_LEVELS) - set(need))}")
    q = sub["specific_humidity"].sel(level=need).load()
    u = sub["u_component_of_wind"].sel(level=need).load()
    v = sub["v_component_of_wind"].sel(level=need).load()
    ax = q.dims.index("level")
    pw, ivtx, ivty = column_integrals(q.values, u.values, v.values, need, level_axis=ax)
    dims = tuple(d for d in q.dims if d != "level")
    coords = {d: q[d] for d in dims}
    inst = xr.Dataset({
        "u850": u.sel(level=850, drop=True), "v850": v.sel(level=850, drop=True),
        "mslp": sub["mean_sea_level_pressure"].load(),
        "w500": sub["vertical_velocity"].sel(level=500, drop=True).load(),
        "pw": xr.DataArray(pw, dims=dims, coords=coords),
        "ivtx": xr.DataArray(ivtx, dims=dims, coords=coords),
        "ivty": xr.DataArray(ivty, dims=dims, coords=coords),
    })
    inst = to_hours(inst)
    per_lead = [inst.sel(hour=[h + 24 * (L - 1) for h in INSTANT_HOURS]).mean("hour").expand_dims(lead=[L])
                for L in leads]
    return xr.concat(per_lead, dim="lead").astype("float32")


def _write_atomic(obj: xr.Dataset, path: Path) -> None:
    tmp = path.with_name(path.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    obj.to_zarr(tmp, mode="w")
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def ingest_season(ds: xr.Dataset, year: int, cfg: dict, leads=(1, 2, 3, 4, 5), force: bool = False,
                  log=print) -> dict[str, Path]:
    raw = Path(cfg["paths"]["raw"]) / "hres"
    raw.mkdir(parents=True, exist_ok=True)
    out = {"rain": raw / f"rain_{year}.zarr", "dyn": raw / f"dyn_{year}.zarr"}
    if not force and all(p.exists() for p in out.values()):
        log(f"hres {year}: already ingested, skipping")
        return out
    inits = season_inits(ds, year, max(leads))
    batch = int(cfg["sources"].get("hres_batch_inits", 4))
    kind = cfg["sources"]["hres_precip_kind"]
    rains, dyns = [], []
    for i in range(0, len(inits), batch):
        b = inits[i:i + batch]
        rains.append(rain_batch(ds, b, leads, kind, cfg["grid"]["rain_box"]))
        dyns.append(dyn_batch(ds, b, leads, cfg["grid"]["dyn_box"]))
        log(f"hres {year}: {min(i + batch, len(inits))}/{len(inits)} inits")
    rain = xr.concat(rains, dim="init").to_dataset(name="f_rain")
    dyn = xr.concat(dyns, dim="init")
    rain.attrs.update(source="WeatherBench2 IFS HRES", units="mm per IMD rain day (03-03 UTC)",
                      precip_kind=kind)
    _write_atomic(rain, out["rain"])
    _write_atomic(dyn, out["dyn"])
    return out


def chunk_report(ds: xr.Dataset, variables=("total_precipitation", "specific_humidity")) -> dict:
    """How WeatherBench2 chunks each variable (decides how much is downloaded for an India subset)."""
    rep = {}
    for v in variables:
        if v not in ds:
            continue
        enc = ds[v].encoding.get("chunks") or ds[v].encoding.get("preferred_chunks")
        itemsize = np.dtype(ds[v].dtype).itemsize
        chunk = dict(zip(ds[v].dims, enc)) if isinstance(enc, (tuple, list)) else enc
        mb = float(np.prod(list(chunk.values())) * itemsize / 1e6) if chunk else None
        rep[v] = {"dims": list(ds[v].dims), "chunks": chunk, "chunk_mb": mb}
    return rep

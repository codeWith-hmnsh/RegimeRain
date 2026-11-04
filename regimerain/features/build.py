"""`regimerain features`: one row per (init, lead, land cell) (TRD 3.3, MODEL_SPEC 7.2-7.4).

Output: <paths.table>/lead=<L>/season=<Y>/part.parquet (lead/season live in the path, hive style).
Only rain days with valid date in 1 Jun - 30 Sep of the season are kept.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from regimerain.features import dynamics as dyn
from regimerain.label.build import jjas_days, load_labels
from regimerain.label.depression import haversine_km
from regimerain.static.basic import TERRAIN_VARS, load_static
from regimerain.truth import load_truth

DYN_VARS = ("u850", "v850", "mslp", "w500", "pw", "ivtx", "ivty")


def _index_of(sub: np.ndarray, full: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(full, sub)
    idx = np.clip(idx, 0, len(full) - 1)
    if not np.allclose(full[idx], sub, atol=1e-6):
        raise ValueError("rain grid is not a subset of the dynamics grid")
    return idx


def _box(field, lat, lon, box):
    la = (lat >= box[0]) & (lat <= box[1])
    lo = (lon >= box[2]) & (lon <= box[3])
    return field[np.ix_(la, lo)], lat[la], lon[lo]


def domain_indices(u850, zeta_s, mslp_hpa, latd, lond, fcfg) -> dict:
    sub, la, _ = _box(u850, latd, lond, fcfg["llj_box"])
    w = np.cos(np.deg2rad(la))[:, None] * np.ones(sub.shape[1])[None, :]
    llj = float(np.nansum(sub * w) / np.nansum(w))
    lo = (lond >= fcfg["trough_lon_band"][0]) & (lond <= fcfg["trough_lon_band"][1])
    la_t = (latd >= fcfg["trough_lat_band"][0]) & (latd <= fcfg["trough_lat_band"][1])
    prof = ndimage.uniform_filter1d(np.nanmean(mslp_hpa[:, lo], axis=1), 5)[la_t]
    trough = float(latd[la_t][np.nanargmin(prof)])
    bob, _, _ = _box(zeta_s, latd, lond, fcfg["bob_box"])
    return {"f_llj_index": llj, "f_trough_lat": trough, "f_bob_vort_max": float(np.nanmax(bob))}


def grids_for_init(rain2d, d, latd, lond, lat_r, lon_r, fcfg) -> dict:
    """All forecast feature grids on the rain grid for one (init, lead). `d`: dict of dynamics 2-D arrays."""
    iy, ix = _index_of(lat_r, latd), _index_of(lon_r, lond)
    cut = lambda a: a[np.ix_(iy, ix)]
    u, v = d["u850"], d["v850"]
    zeta_s = ndimage.gaussian_filter(np.nan_to_num(dyn.rel_vorticity(u, v, latd, lond)), 2)
    mslp = d["mslp"] / 100.0
    imfc = dyn.imfc_mm_day(d["ivtx"], d["ivty"], latd, lond)
    g = {
        "f_rain": rain2d,
        "f_rain_nbr3_mean": ndimage.uniform_filter(rain2d, 3, mode="nearest"),
        "f_rain_nbr3_max": ndimage.maximum_filter(rain2d, 3, mode="nearest"),
        "f_rain_nbr5_mean": ndimage.uniform_filter(rain2d, 5, mode="nearest"),
        "f_rain_nbr5_max": ndimage.maximum_filter(rain2d, 5, mode="nearest"),
        "f_u850": cut(u), "f_v850": cut(v), "f_ws850": np.hypot(cut(u), cut(v)),
        "f_vort850_max500km": cut(ndimage.maximum_filter(zeta_s, size=int(fcfg["vort_max_filter_cells"]))),
        "f_pw": cut(d["pw"]), "f_ivt": np.hypot(cut(d["ivtx"]), cut(d["ivty"])), "f_imfc": cut(imfc),
        "f_w500": cut(d["w500"]),
    }
    mr = cut(mslp)
    g["f_mslp_anom"] = mr - np.nanmean(mr)
    sub, la, lo = _box(mslp, latd, lond, (10, 30, 70, 95))
    jy, jx = np.unravel_index(np.nanargmin(sub), sub.shape)
    LAT, LON = np.meshgrid(lat_r, lon_r, indexing="ij")
    g["f_dist_mslp_min_km"] = haversine_km(LAT, LON, la[jy], lo[jx])
    g.update(domain_indices(u, zeta_s, mslp, latd, lond, fcfg))
    return g


def _rmm(cfg):
    p = Path(cfg["paths"]["raw"]) / "mjo" / "rmm.parquet"
    return pd.read_parquet(p) if p.exists() else None


def cmz_mask(cfg: dict, static: xr.Dataset) -> np.ndarray:
    cb = cfg["labels"]["cmz_box"]
    LATg, LONg = np.meshgrid(static.lat.values, static.lon.values, indexing="ij")
    land = static["land"].values.astype(bool)
    return land & (LATg >= cb[0]) & (LATg <= cb[1]) & (LONg >= cb[2]) & (LONg <= cb[3])


def forecast_rows(cfg: dict, init, lead: int, rain2d: np.ndarray, d: dict, latd, lond, static: xr.Dataset,
                  rmm: pd.DataFrame | None, cmz: np.ndarray | None = None) -> pd.DataFrame:
    """Feature rows of one (init, lead) for every land cell: everything known at forecast time, no truth.

    Shared by the training table (`build_season_lead`) and live runs (`regimerain.runs.live`), so the
    model sees features computed the same way in both.
    """
    from regimerain.ingest.mjo import value_on
    init = pd.Timestamp(init)
    lat_r, lon_r = static.lat.values, static.lon.values
    land = static["land"].values.astype(bool)
    iy, ix = np.nonzero(land)
    cmz = cmz_mask(cfg, static) if cmz is None else cmz
    rain2d = np.asarray(rain2d, "float64")
    g = grids_for_init(rain2d, d, latd, lond, lat_r, lon_r, cfg["features"])
    g["f_cmz_rain"] = float(np.nanmean(rain2d[cmz])) if cmz.any() else 0.0
    g["f_cmz_wetfrac"] = float(np.mean(rain2d[cmz] >= 2.5)) if cmz.any() else 0.0
    row = {"lat_idx": iy.astype("int16"), "lon_idx": ix.astype("int16"),
           "lat": lat_r[iy].astype("float32"), "lon": lon_r[ix].astype("float32")}
    for k, a in g.items():
        row[k] = (np.asarray(a)[iy, ix] if np.ndim(a) == 2 else np.full(len(iy), a)).astype("float32")
    for k in ("geo", "zone"):
        row[k] = static[k].values[iy, ix].astype("int8")
    for k in TERRAIN_VARS:
        row[k] = static[k].values[iy, ix].astype("float32")
    m = value_on(rmm, init) if rmm is not None else {"rmm1": 0.0, "rmm2": 0.0, "phase": 0, "amplitude": 0.0}
    row.update(mjo_rmm1=np.float32(m["rmm1"]), mjo_rmm2=np.float32(m["rmm2"]),
               mjo_amp=np.float32(m["amplitude"]), mjo_phase=np.int8(m["phase"]))
    valid = init.normalize() + pd.Timedelta(days=lead - 1)
    doy = valid.day_of_year
    row.update(doy_sin=np.float32(np.sin(2 * np.pi * doy / 365.25)), doy_cos=np.float32(np.cos(2 * np.pi * doy / 365.25)))
    df = pd.DataFrame(row)
    df["init"], df["valid"] = init, valid
    df.loc[df.f_rain < float(cfg["labels"]["dry_threshold_mm"]), "f_rain"] = 0.0     # same drizzle cut as truth
    return df


def build_season_lead(cfg: dict, year: int, lead: int, static: xr.Dataset, log=print) -> pd.DataFrame:
    raw = Path(cfg["paths"]["raw"]) / "hres"
    rain = xr.open_zarr(raw / f"rain_{year}.zarr")["f_rain"].sel(lead=lead).load()
    dz = xr.open_zarr(raw / f"dyn_{year}.zarr").sel(lead=lead).load()
    lat_r, lon_r = static.lat.values, static.lon.values
    if not (np.allclose(rain.lat.values, lat_r) and np.allclose(rain.lon.values, lon_r)):
        raise ValueError("forecast rain grid differs from the IMD/static grid")
    latd, lond = dz.lat.values, dz.lon.values
    land = static["land"].values.astype(bool)
    iy, ix = np.nonzero(land)
    labels = load_labels(cfg, year)
    truth = load_truth(cfg, [year])
    rmm = _rmm(cfg)
    days = set(jjas_days(year))
    cmz = cmz_mask(cfg, static)
    frames = []
    for i, init in enumerate(pd.DatetimeIndex(rain.init.values)):
        valid = init.normalize() + pd.Timedelta(days=lead - 1)
        if valid not in days:
            continue
        d = {k: dz[k].isel(init=i).values.astype("float64") for k in DYN_VARS}
        df = forecast_rows(cfg, init, lead, rain.isel(init=i).values, d, latd, lond, static, rmm, cmz)
        df["o_rain"] = truth.sel(time=valid).values[iy, ix].astype("float32")
        df["y_synoptic"] = labels["y_synoptic"].sel(time=valid).values[iy, ix].astype("int8")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[out.o_rain.notna()].reset_index(drop=True)
    dry = float(cfg["labels"]["dry_threshold_mm"])
    out.loc[out.o_rain < dry, "o_rain"] = 0.0
    if rmm is None:
        log(f"features {year} L{lead}: WARNING no MJO file - MJO features set to neutral")
    return out


def table_path(cfg: dict, lead: int, year: int) -> Path:
    return Path(cfg["paths"]["table"]) / f"lead={lead}" / f"season={year}" / "part.parquet"


def build_features(cfg: dict, seasons, leads, force: bool = False, log=print) -> None:
    static = load_static(cfg)
    for y in seasons:
        for L in leads:
            out = table_path(cfg, L, y)
            if out.exists() and not force:
                log(f"features {y} L{L}: exists, skipping")
                continue
            df = build_season_lead(cfg, y, L, static, log=log)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            log(f"features {y} L{L}: {len(df):,} rows -> {out}")

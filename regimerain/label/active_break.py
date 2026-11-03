"""Active/break labels from IMD rainfall over the core monsoon zone (MODEL_SPEC section 6.1).

Convention (Rajeevan, Gadgil & Bhate 2010): standardised anomaly of area-mean rainfall over the
core monsoon zone; > +1 for >= 3 consecutive days = active, < -1 for >= 3 days = break.
Climatology uses 1981-2015 only (outside the 2016-2022 evaluation window).
"""
from __future__ import annotations

import numpy as np
import xarray as xr

ACTIVE, BREAK, DEPRESSION, NORMAL = 0, 1, 2, 3


def cmz_series(o_rain: xr.DataArray, land: xr.DataArray, box=(18, 28, 65, 88)) -> xr.DataArray:
    sel = dict(lat=slice(box[0], box[1]), lon=slice(box[2], box[3]))
    sub = o_rain.sel(**sel).where(land.sel(**sel))
    w = np.cos(np.deg2rad(sub.lat))
    return sub.weighted(w).mean(["lat", "lon"])


def circular_smooth(values: np.ndarray, window: int = 31) -> np.ndarray:
    v = np.asarray(values, "float64")
    p = window // 2
    padded = np.concatenate([v[-p:], v, v[:p]])
    return np.convolve(padded, np.ones(window) / window, mode="valid")[: len(v)]


def standardised_anomaly(R: xr.DataArray, clim=(1981, 2015), window: int = 31) -> xr.DataArray:
    base = R.sel(time=R.time.dt.year.isin(list(range(clim[0], clim[1] + 1))))
    g = base.groupby("time.dayofyear")
    mu, sd = g.mean(), g.std()
    mu = mu.copy(data=circular_smooth(mu.values, window))
    sd = sd.copy(data=circular_smooth(sd.values, window))
    doy = R.time.dt.dayofyear
    z = (R - mu.sel(dayofyear=doy)) / sd.sel(dayofyear=doy)
    return z.drop_vars("dayofyear", errors="ignore")


def runs_at_least(mask: np.ndarray, k: int) -> np.ndarray:
    """True where `mask` is part of a run of >= k consecutive True values."""
    mask = np.asarray(mask, bool)
    out = np.zeros_like(mask)
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= k:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def active_break(z: xr.DataArray, threshold: float = 1.0, min_run: int = 3) -> xr.DataArray:
    """Per-day label: ACTIVE, BREAK or NORMAL.

    Runs never cross a calendar year or a gap in the dates (e.g. a JJAS-only series), so
    'consecutive' always means consecutive calendar days.
    """
    lab = np.full(z.sizes["time"], NORMAL, "int8")
    t = z.time.values.astype("datetime64[D]")
    years = z.time.dt.year.values
    new_segment = np.ones(len(t), bool)
    new_segment[1:] = (np.diff(t) != np.timedelta64(1, "D")) | (np.diff(years) != 0)
    segment = np.cumsum(new_segment)
    zv = z.values
    for seg in np.unique(segment):
        idx = np.where(segment == seg)[0]
        zz = zv[idx]
        lab[idx[runs_at_least(zz > threshold, min_run)]] = ACTIVE
        lab[idx[runs_at_least(zz < -threshold, min_run)]] = BREAK
    return xr.DataArray(lab, coords={"time": z.time}, dims="time", name="ab")


def synoptic_labels(ab_by_day: np.ndarray, dep_mask: np.ndarray) -> np.ndarray:
    """(day,) active/break/normal + (day, lat, lon) depression mask -> (day, lat, lon) int8 labels."""
    lab = np.broadcast_to(np.asarray(ab_by_day, "int8")[:, None, None], dep_mask.shape).copy()
    lab[np.asarray(dep_mask, bool)] = DEPRESSION
    return lab


def day_level_label(ab: int, dep_frac: float, min_frac: float = 0.05) -> int:
    return DEPRESSION if dep_frac >= min_frac else int(ab)

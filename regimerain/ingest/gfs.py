"""Live GFS forecast for one 00 UTC init (MODEL_SPEC 18.2), in the same units and grids as the HRES ingest.

Phase 0 (docs/PHASE0_FINDINGS.md) confirmed that every GFS 0.25 deg file carries a running precipitation total
from init (`APCP:surface:0-{h} hour acc`), so a rain-day total is one difference of two running totals. The
3/6-hour bucket fallback is kept in case NOAA changes the layout.

Output of `fetch_gfs` (all latitude ascending, like the working grid):
    rain[L]  (nlat, nlon) mm per IMD rain day (03-03 UTC), on the rain grid
    dyn[L]   dict u850 v850 mslp w500 pw ivtx ivty, rain-day means of the 6-hourly instants, on the dynamics grid

GRIB access goes through Herbie (byte-range downloads of just the needed messages, from NOAA NOMADS or the
AWS archive). Install with `pip install -e ".[live]"`.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from regimerain.features.dynamics import column_integrals
from regimerain.ingest.hres import COLUMN_LEVELS, INSTANT_HOURS
from regimerain.runs.fields import lead_hours

LEVEL_RE = "|".join(str(level) for level in COLUMN_LEVELS)
INSTANT_SEARCH = f":(UGRD|VGRD|SPFH):({LEVEL_RE}) mb:|:VVEL:500 mb:|:PRMSL:mean sea level:"
RH_T_SEARCH = f":(RH|TMP):({LEVEL_RE}) mb:"                 # only if a file lacks SPFH (MODEL_SPEC 18.2)


def rain_hours(leads) -> list[int]:
    """Forecast hours whose running totals bound the rain days: 3, 27, 51, ..."""
    return sorted({h for L in leads for h in lead_hours(L)})


def instant_hours(leads) -> list[int]:
    """6-hourly instants inside each rain day (the HRES ingest uses the same ones)."""
    return sorted({h + 24 * (L - 1) for L in leads for h in INSTANT_HOURS})


def rainday_from_cumulative(cum: dict[int, np.ndarray], leads) -> dict[int, np.ndarray]:
    """Rain-day totals from running totals since init: total(27 + 24(L-1)) - total(3 + 24(L-1)), never negative."""
    out = {}
    for L in leads:
        a, b = lead_hours(L)
        out[L] = np.clip(cum[b] - cum[a], 0, None)
    return out


def q_from_rh(rh_pct: np.ndarray, t_k: np.ndarray, p_hpa) -> np.ndarray:
    """Specific humidity (kg/kg) from relative humidity (%) and temperature (K) at pressure p (MODEL_SPEC 18.2)."""
    p = np.asarray(p_hpa, float).reshape((-1,) + (1,) * (np.ndim(t_k) - 1)) if np.ndim(p_hpa) else float(p_hpa)
    tc = t_k - 273.15
    es = 6.112 * np.exp(17.67 * tc / (tc + 243.5))
    e = rh_pct / 100.0 * es
    return 0.622 * e / (p - 0.378 * e)


def instant_fields(q: np.ndarray, u: np.ndarray, v: np.ndarray, mslp: np.ndarray, w500: np.ndarray,
                   levels=COLUMN_LEVELS) -> dict[str, np.ndarray]:
    """One 6-hourly instant -> the seven dynamics fields. q, u, v: (level, lat, lon) on `levels` (hPa)."""
    levels = list(levels)
    pw, ivtx, ivty = column_integrals(q, u, v, levels, level_axis=0)
    k850 = levels.index(850)
    return {"u850": u[k850], "v850": v[k850], "mslp": mslp, "w500": w500, "pw": pw, "ivtx": ivtx, "ivty": ivty}


def rainday_means(inst: dict[int, dict[str, np.ndarray]], leads) -> dict[int, dict[str, np.ndarray]]:
    """Mean of the four instants inside each rain day (06, 12, 18, 24 UTC + 24(L-1)), like the HRES ingest."""
    out = {}
    for L in leads:
        hours = [h + 24 * (L - 1) for h in INSTANT_HOURS]
        out[L] = {k: np.mean([inst[h][k] for h in hours], axis=0) for k in inst[hours[0]]}
    return out


def _box_axes(box, res: float):
    lat0, lat1, lon0, lon1 = box
    return (np.round(np.arange(lat0, lat1 + res / 2, res), 4), np.round(np.arange(lon0, lon1 + res / 2, res), 4))


class Gfs:
    """Herbie-backed reader for one GFS 0.25 deg init. Every field comes back latitude ascending on a box."""

    def __init__(self, init, save_dir: str | Path, priority=("aws", "nomads")):
        self.init = np.datetime64(init, "h")
        self.save_dir = Path(save_dir)
        self.priority = list(priority)

    def _herbie(self, fxx: int):
        from herbie import Herbie
        return Herbie(str(self.init), model="gfs", product="pgrb2.0p25", fxx=fxx, save_dir=self.save_dir,
                      priority=self.priority, verbose=False)

    def download(self, fxx: int, search: str) -> None:
        """Fetch the matching GRIB messages to the local cache (thread-safe: plain HTTP range requests)."""
        self._herbie(fxx).download(search)

    def _read(self, fxx: int, search: str):
        ds = self._herbie(fxx).xarray(search, remove_grib=False)       # reuses the cached subset file
        return ds if isinstance(ds, list) else [ds]

    @staticmethod
    def _on(da, lat, lon) -> np.ndarray:
        lons = da.longitude.values
        if lons.max() > 180 and lon.min() < 0:
            raise ValueError("box crosses the date line")
        return da.sel(latitude=lat, longitude=lon, method="nearest", tolerance=1e-3).values.astype("float64")

    def _apcp(self, fxx: int, window: str, lat, lon) -> np.ndarray:
        ds = self._read(fxx, f":APCP:surface:{window} acc")[0]            # duplicate messages at f003/f006
        return self._on(ds["tp"], lat, lon)

    def cumulative(self, h: int, lat, lon) -> np.ndarray:
        """Precipitation (mm) accumulated from init to init + h hours."""
        windows = [f"0-{h} hour"] + ([f"0-{h // 24} day"] if h % 24 == 0 else [])
        for w in windows:
            try:
                return self._apcp(h, w, lat, lon)
            except Exception:                                             # noqa: BLE001 - try the next layout
                pass
        step = 3 if h % 6 == 3 else 6                                     # bucket fallback
        prev = self.cumulative(h - step, lat, lon) if h - step > 0 else 0.0
        return prev + self._apcp(h, f"{h - step}-{h} hour", lat, lon)

    def _fields(self, h: int, search: str, lat, lon) -> dict[str, np.ndarray]:
        fields: dict[str, np.ndarray] = {}
        for ds in self._read(h, search):
            for name, da in ds.data_vars.items():
                if "isobaricInhPa" in da.dims:                            # a multi-level cube
                    da = da.sel(isobaricInhPa=500 if name == "w" else COLUMN_LEVELS)
                fields[name] = self._on(da, lat, lon)
        return fields

    def instant(self, h: int, lat, lon) -> dict[str, np.ndarray]:
        fields = self._fields(h, INSTANT_SEARCH, lat, lon)
        if "q" not in fields:                                             # SPFH absent: derive it from RH and T
            extra = self._fields(h, RH_T_SEARCH, lat, lon)
            fields["q"] = q_from_rh(extra["r"], extra["t"], COLUMN_LEVELS)
        missing = [k for k in ("q", "u", "v", "w", "prmsl") if k not in fields]
        if missing:
            raise ValueError(f"GFS f{h:03d}: missing {missing}")
        return instant_fields(fields["q"], fields["u"], fields["v"], fields["prmsl"], np.squeeze(fields["w"]))


def _quiet(fn, *args):
    try:
        fn(*args)
    except Exception:                             # noqa: BLE001 - the sequential read reports the real error
        pass


def fetch_gfs(cfg: dict, init, leads, threads: int = 8, log=print, reader=None):
    """Rain-day rain on the rain grid and rain-day-mean dynamics on the dynamics grid, for one 00 UTC init."""
    res = float(cfg["grid"]["res_deg"])
    lat_r, lon_r = _box_axes(cfg["grid"]["rain_box"], res)
    latd, lond = _box_axes(cfg["grid"]["dyn_box"], res)
    g = reader or Gfs(init, Path(cfg["paths"]["raw"]) / "gfs")
    rh, ih = rain_hours(leads), instant_hours(leads)
    log(f"gfs {init}: {len(rh)} precipitation and {len(ih)} dynamics files")
    if hasattr(g, "download"):                    # downloads in parallel; GRIB decoding stays sequential
        jobs = [(h, f":APCP:surface:0-{h} hour acc") for h in rh] + [(h, INSTANT_SEARCH) for h in ih]
        with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
            list(pool.map(lambda j: _quiet(g.download, *j), jobs))
    cum = {h: g.cumulative(h, lat_r, lon_r) for h in rh}
    inst = {h: g.instant(h, latd, lond) for h in ih}
    return {"rain": rainday_from_cumulative(cum, leads), "dyn": rainday_means(inst, leads),
            "lat": lat_r, "lon": lon_r, "latd": latd, "lond": lond}

"""`regimerain label`: synoptic labels per season (MODEL_SPEC 6.1-6.4).

Writes <static-parent>/labels/labels_<year>.zarr with
    y_synoptic(time, lat, lon) int8   0 active, 1 break, 2 depression, 3 normal
    ab(time)                           active/break/normal of the day
    z(time)                            core-monsoon-zone standardised anomaly
    day_label(time)                    day-level label (depression if >= day_depression_min_frac land cells)
`labels.climatology`: "imd" = sources.imd_climatology years (needs those IMD years), "self" = the
seasons themselves (pilot simplification, disclosed in the report).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from regimerain.label.active_break import (DEPRESSION, active_break, cmz_series, day_level_label,
                                           standardised_anomaly, synoptic_labels)
from regimerain.label.depression import depression_mask
from regimerain.static.basic import load_static
from regimerain.truth import load_truth, prepare_truth


def labels_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["static"]).parent / "labels"


def jjas_days(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-06-01", f"{year}-09-30", freq="D")


def climatology_years(cfg: dict, seasons) -> list[int]:
    mode = cfg["labels"].get("climatology", "imd")
    if mode == "self":
        return sorted(seasons)
    a, b = cfg["sources"]["imd_climatology"]
    return list(range(a, b + 1))


def available_climatology(cfg: dict, clim: list[int], seasons, log=print, min_years: int = 25) -> list[int]:
    """Drop climatology years with no IMD data (neither .grd nor converted zarr); fail if too few remain."""
    if cfg["labels"].get("climatology", "imd") == "self":
        return clim
    from regimerain.ingest.imd import year_file
    from regimerain.truth import truth_path
    have = [y for y in clim if truth_path(cfg, y).exists() or year_file(cfg["paths"]["imd"], y).exists()]
    missing = sorted(set(clim) - set(have))
    if missing:
        if len(have) < min_years:
            raise FileNotFoundError(f"IMD climatology years missing {missing}: only {len(have)} available, "
                                    f"need {min_years}. Rerun `ingest --imd-climatology ...`")
        log(f"label: WARNING climatology without {missing} ({len(have)} years used)")
    return have


def build_labels(cfg: dict, seasons, log=print) -> dict[int, Path]:
    lab_cfg = cfg["labels"]
    clim = available_climatology(cfg, climatology_years(cfg, seasons), seasons, log=log)
    prepare_truth(cfg, sorted(set(clim) | set(seasons)), log=log)
    static = load_static(cfg)
    land = static["land"].astype(bool)
    o = load_truth(cfg, sorted(set(clim) | set(seasons)))
    R = cmz_series(o, land, tuple(lab_cfg["cmz_box"]))
    z = standardised_anomaly(R, clim=(min(clim), max(clim)))
    ab = active_break(z, lab_cfg["ab_threshold_sigma"], lab_cfg["ab_min_run_days"])
    tracks_file = Path(cfg["paths"]["raw"]) / "tracks" / "ibtracs_ni.parquet"
    tracks = pd.read_parquet(tracks_file) if tracks_file.exists() else pd.DataFrame(columns=["time", "lat", "lon"])
    if tracks.empty:
        log("label: WARNING no track file - no depression labels (run `ingest --only tracks`)")
    out_dir = labels_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    written, counts = {}, []
    lat, lon = o.lat.values, o.lon.values
    land_np = land.values
    for y in seasons:
        days = jjas_days(y)
        ab_days = ab.sel(time=days).values
        dep = depression_mask(tracks, days, lat, lon, lab_cfg["depression_radius_km"]) & land_np[None]
        syn = synoptic_labels(ab_days, dep)
        dep_frac = dep.reshape(len(days), -1)[:, land_np.ravel()].mean(axis=1)
        day_lab = np.array([day_level_label(a, f, lab_cfg["day_depression_min_frac"]) for a, f in zip(ab_days, dep_frac)],
                           "int8")
        ds = xr.Dataset({"y_synoptic": (("time", "lat", "lon"), syn),
                         "ab": ("time", ab_days.astype("int8")),
                         "z": ("time", z.sel(time=days).values.astype("float32")),
                         "day_label": ("time", day_lab)},
                        coords={"time": days, "lat": lat, "lon": lon})
        ds.attrs["climatology"] = f"{min(clim)}-{max(clim)} ({lab_cfg.get('climatology', 'imd')})"
        path = out_dir / f"labels_{y}.zarr"
        ds.to_zarr(path, mode="w")
        written[y] = path
        cells = syn[:, land_np]
        counts.append({"season": y, **{f"cells_{n}": int((cells == k).sum()) for k, n in
                                       enumerate(["active", "break", "depression", "normal"])},
                       **{f"days_{n}": int((day_lab == k).sum()) for k, n in
                          enumerate(["active", "break", "depression", "normal"])}})
        log(f"label {y}: days active={counts[-1]['days_active']} break={counts[-1]['days_break']} "
            f"depression={counts[-1]['days_depression']} normal={counts[-1]['days_normal']}")
    pd.DataFrame(counts).to_csv(out_dir / "label_counts.csv", index=False)
    return written


def load_labels(cfg: dict, year: int) -> xr.Dataset:
    return xr.open_zarr(labels_dir(cfg) / f"labels_{year}.zarr").load()


__all__ = ["build_labels", "load_labels", "labels_dir", "DEPRESSION"]

"""Phase 0 data checks (docs/MODEL_SPEC.md section 3). Run on a laptop or in Colab:

    pip install -e ".[data,live]"
    python scripts/phase0_check.py            # all checks
    python scripts/phase0_check.py hres imd   # selected checks

Each check prints PASS/FAIL with what it found. Paste the output into docs/PHASE0_FINDINGS.md.
Nothing here writes large files; the IMD check downloads one year (~25 MB) into ./data/raw/imd_probe.
"""
from __future__ import annotations

import io
import os
import sys
import traceback
import urllib.request
from datetime import datetime, timedelta, timezone

import yaml

HRES_URL = "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
IBTRACS_NI = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
              "v04r01/access/csv/ibtracs.NI.list.v04r01.csv")
RMM_URLS = ["http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt",
            "https://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"]
USER_AGENT = "regimerain/0.1 (SIH26080 research; https://github.com/shreyashsri79/080)"
NEEDED_LEVELS = {1000, 925, 850, 700, 600, 500, 400, 300}


def check_hres():
    import numpy as np
    import xarray as xr
    ds = xr.open_zarr(HRES_URL, storage_options={"token": "anon"})
    precip = [v for v in ds.data_vars if "precip" in v]
    from regimerain.ingest.align import lead_hours
    lead_h = lead_hours(ds.prediction_timedelta)
    info_dtype = {"prediction_timedelta_dtype": str(ds.prediction_timedelta.dtype),
                  "prediction_timedelta_units": ds.prediction_timedelta.attrs.get("units")}
    info = {
        "precip_vars": {v: {"units": ds[v].attrs.get("units"), "dims": list(ds[v].dims)} for v in precip},
        "first_leads_h": lead_h[:6].tolist(),
        "max_lead_h": float(lead_h.max()),
        "lat_ascending": bool(ds.latitude.values[0] < ds.latitude.values[-1]),
        "levels": sorted(int(x) for x in ds.level.values) if "level" in ds.dims else None,
        "vars": sorted(ds.data_vars),
        **info_dtype,
    }
    missing_levels = NEEDED_LEVELS - set(info["levels"] or [])
    info["precip_consistency"] = hres_precip_consistency(ds)
    ok = bool(precip) and not missing_levels and info["precip_consistency"]["cumulative_matches_6hr"]
    return ok, dict(info, missing_levels=sorted(missing_levels))


def hres_precip_consistency(ds):
    """Is `total_precipitation` accumulated since init, and do the 6hr/24hr variables match its differences?

    Reads one monsoon init, leads 0-30 h, over a central-India box (small, fast)."""
    import numpy as np
    box = dict(latitude=slice(18, 28), longitude=slice(70, 88))
    one = ds.sel(**box).isel(prediction_timedelta=slice(0, 6))
    one = one.sel(time=one.time.values[(one.time.values >= np.datetime64("2019-07-15"))][0])
    tp = one["total_precipitation"].mean(["latitude", "longitude"]).values.astype(float)
    tp6 = one["total_precipitation_6hr"].mean(["latitude", "longitude"]).values.astype(float)
    d = np.diff(tp)
    rel = float(np.nanmax(np.abs(d - tp6[1:]) / (np.abs(tp6[1:]) + 1e-9)))
    out = {
        "tp_at_leads_0_30h": [round(v, 6) for v in tp.tolist()],
        "tp6_at_leads_0_30h": [None if np.isnan(v) else round(v, 6) for v in tp6.tolist()],
        "tp_is_nondecreasing": bool(np.all(d >= -1e-7)),
        "tp_starts_at_zero": bool(abs(tp[0]) < 1e-6),
        "max_rel_diff_diff(tp)_vs_tp6": round(rel, 4),
        "cumulative_matches_6hr": bool(rel < 0.02 and np.all(d >= -1e-7)),
        "box_mean_mm_0_24h_if_metres": round(float((tp[4] - tp[0]) * 1000), 2),
    }
    if "total_precipitation_24hr" in one:
        tp24 = one["total_precipitation_24hr"].mean(["latitude", "longitude"]).values.astype(float)
        out["tp24_at_24h"] = None if np.isnan(tp24[4]) else round(float(tp24[4]), 6)
        out["tp24_matches_tp(24)-tp(0)"] = bool(abs(tp24[4] - (tp[4] - tp[0])) < 1e-4 + 0.02 * abs(tp24[4]))
    return out


def check_imd():
    import imdlib as imd
    os.makedirs("data/raw/imd_probe", exist_ok=True)          # imdlib does not create its folder
    imd.get_data("rain", 2016, 2016, fn_format="yearwise", file_dir="data/raw/imd_probe")
    da = imd.open_data("rain", 2016, 2016, "yearwise", "data/raw/imd_probe").get_xarray()["rain"]
    da = da.where(da > -998)
    return True, {"dims": dict(da.sizes), "lat_range": [float(da.lat.min()), float(da.lat.max())],
                  "lon_range": [float(da.lon.min()), float(da.lon.max())],
                  "jul_mean_mm": float(da.sel(time=da.time.dt.month == 7).mean())}


def check_ibtracs():
    import pandas as pd
    df = pd.read_csv(IBTRACS_NI, skiprows=[1], low_memory=False, nrows=50000)
    cols = [c for c in ("SID", "ISO_TIME", "LAT", "LON", "NEWDELHI_GRADE", "NEWDELHI_WIND") if c in df.columns]
    return "NEWDELHI_GRADE" in df.columns, {"columns_found": cols, "rows_read": len(df)}


def fetch_rmm_text() -> tuple[str, str]:
    """RMM index text. Order: $RMM_FILE (manually downloaded copy), then BOM with an explicit User-Agent."""
    local = os.environ.get("RMM_FILE")
    if local:
        with open(local, encoding="utf-8", errors="replace") as fh:
            return fh.read(), f"file:{local}"
    errors = []
    for url in RMM_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace"), url
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("RMM download failed; download the file in a browser and set RMM_FILE=<path>. " + " | ".join(errors))


def check_mjo():
    import pandas as pd
    text, source = fetch_rmm_text()
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", skiprows=2, header=None, usecols=range(7),
                     names=["year", "month", "day", "rmm1", "rmm2", "phase", "amplitude"])
    df = df[pd.to_numeric(df.amplitude, errors="coerce") < 100]
    return len(df) > 1000, {"source": source, "first": df.iloc[0].tolist(), "last": df.iloc[-1].tolist()}


def check_gfs():
    from herbie import Herbie
    init = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d 00:00")
    windows = {}
    for fxx in (3, 6, 9, 12, 27):
        inv = Herbie(init, model="gfs", product="pgrb2.0p25", fxx=fxx).inventory(":APCP:")
        windows[fxx] = inv["search_this"].tolist() if "search_this" in inv else inv.to_string()
    return True, {"init": init, "apcp_messages": windows}


def check_hres_chunks():
    """How the store is chunked -> how much data an India-only read really downloads."""
    import xarray as xr
    from regimerain.ingest.hres import chunk_report
    ds = xr.open_zarr(HRES_URL, storage_options={"token": "anon"})
    rep = chunk_report(ds, ("total_precipitation", "specific_humidity", "u_component_of_wind",
                            "mean_sea_level_pressure"))
    for v, r in rep.items():
        c = r["chunks"] or {}
        lat_c, lon_c = c.get("latitude"), c.get("longitude")
        r["india_fraction_of_chunk"] = (round((129 / lat_c) * (135 / lon_c), 3)
                                        if lat_c and lon_c and lat_c >= 129 and lon_c >= 135 else "spatially chunked")
    return True, rep


CHECKS = {"hres": check_hres, "hres_chunks": check_hres_chunks, "imd": check_imd, "ibtracs": check_ibtracs, "mjo": check_mjo, "gfs": check_gfs}


def main(names: list[str]) -> int:
    names = names or list(CHECKS)
    results = {}
    for name in names:
        try:
            ok, info = CHECKS[name]()
        except Exception as exc:  # report and continue with the other checks
            ok, info = False, {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=2)}
        results[name] = {"status": "PASS" if ok else "FAIL", **info}
        print(f"\n=== {name}: {'PASS' if ok else 'FAIL'} ===")
        print(yaml.safe_dump(info, sort_keys=False, width=110))
    failed = [n for n, r in results.items() if r["status"] != "PASS"]
    print("\nSummary:", {n: r["status"] for n, r in results.items()})
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

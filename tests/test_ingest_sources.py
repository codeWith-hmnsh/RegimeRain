"""Ingest code on synthetic data shaped like the real sources (formats confirmed in docs/PHASE0_FINDINGS.md)."""
import io

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from regimerain.config import deep_merge, load_config
from regimerain.features.dynamics import G, column_integrals
from regimerain.ingest import imd
from regimerain.ingest.hres import dyn_batch, ingest_season, normalise, rain_batch, season_inits
from regimerain.ingest.mjo import align_to_dates, parse, value_on
from regimerain.ingest.tracks import parse_ibtracs

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
RATE_MM_H = 1.0


def fake_wb2(first="2019-05-28", days=6):
    """WeatherBench2-like HRES: descending lat, 00/12 UTC inits, int64 lead hours, cumulative tp (m)."""
    lat = np.arange(12.0, 4.99, -1.0)                       # descending on purpose
    lon = np.arange(64.0, 72.01, 1.0)
    time = pd.date_range(first, periods=days * 2, freq="12h")
    hours = np.arange(0, 241, 6).astype("int64")
    shp4 = (len(time), len(hours), len(lat), len(lon))
    tp = np.broadcast_to((hours * RATE_MM_H / 1000.0)[None, :, None, None], shp4).copy()
    lev_shape = (len(time), len(hours), len(LEVELS), len(lat), len(lon))
    coords = {"time": time, "prediction_timedelta": ("prediction_timedelta", hours, {"units": "hours"}),
              "level": LEVELS, "latitude": lat, "longitude": lon}
    d4 = ("time", "prediction_timedelta", "latitude", "longitude")
    d5 = ("time", "prediction_timedelta", "level", "latitude", "longitude")
    return xr.Dataset({
        "total_precipitation": (d4, tp),
        "mean_sea_level_pressure": (d4, np.full(shp4, 100500.0)),
        "specific_humidity": (d5, np.full(lev_shape, 0.01)),
        "u_component_of_wind": (d5, np.full(lev_shape, 10.0)),
        "v_component_of_wind": (d5, np.zeros(lev_shape)),
        "vertical_velocity": (d5, np.full(lev_shape, -0.1)),
    }, coords=coords)


def test_normalise_renames_flips_and_keeps_00utc():
    ds = normalise(fake_wb2())
    assert {"init", "lat", "lon"} <= set(ds.dims)
    assert ds.lat.values[0] < ds.lat.values[-1]
    assert (pd.DatetimeIndex(ds.init.values).hour == 0).all() and ds.sizes["init"] == 6


def test_season_inits_cover_lead_back_to_june_1():
    ds = normalise(fake_wb2(first="2019-05-25", days=10))
    inits = season_inits(ds, 2019, max_lead=5)
    assert str(inits[0])[:10] == "2019-05-28"                # 1 Jun minus 4 days


def test_rain_batch_constant_rate():
    ds = normalise(fake_wb2())
    r = rain_batch(ds, ds.init.values[:2], (1, 2, 5), "cumulative", (5, 12, 64, 72))
    assert set(r.dims) == {"init", "lead", "lat", "lon"}
    assert np.allclose(r.values, 24 * RATE_MM_H)


def test_column_integrals_constant_profile():
    q = np.full((8, 2, 2), 0.01); u = np.full_like(q, 10.0); v = np.zeros_like(q)
    pw, ivtx, ivty = column_integrals(q, u, v, [300, 400, 500, 600, 700, 850, 925, 1000])
    assert np.allclose(pw, 0.01 * 70000 / G) and np.allclose(ivtx, 10 * pw) and np.allclose(ivty, 0)


def test_dyn_batch_values_and_shape():
    ds = normalise(fake_wb2())
    d = dyn_batch(ds, ds.init.values[:2], (1, 2), (5, 12, 64, 72))
    assert set(d.data_vars) == {"u850", "v850", "mslp", "w500", "pw", "ivtx", "ivty"}
    assert d.sizes["lead"] == 2 and d.sizes["init"] == 2
    assert np.allclose(d.u850, 10) and np.allclose(d.pw, 0.01 * 70000 / G, rtol=1e-5)
    assert np.allclose(d.w500, -0.1)


def test_ingest_season_writes_and_skips(tmp_path):
    pytest.importorskip("zarr")
    cfg = deep_merge(load_config(), {"paths": {"raw": str(tmp_path)},
                                     "grid": {"rain_box": [5, 12, 64, 72], "dyn_box": [5, 12, 64, 72]},
                                     "sources": {"hres_batch_inits": 2}})
    ds = normalise(fake_wb2())
    out = ingest_season(ds, 2019, cfg, leads=(1, 2), log=lambda *_: None)
    rain = xr.open_zarr(out["rain"])["f_rain"]
    assert np.allclose(rain.values, 24.0) and "valid" in rain.coords
    assert xr.open_zarr(out["dyn"]).sizes["lead"] == 2
    msgs = []
    ingest_season(ds, 2019, cfg, leads=(1, 2), log=msgs.append)
    assert "skipping" in msgs[0]


RMM_TEXT = """RMM values
year, month, day, RMM1, RMM2, phase, amplitude, Final_value
  2019     7    14  0.5  -0.3   4   0.58  Final_value:_BoM_ACCESS
  2019     7    15  0.7  -0.2   5   0.73  Final_value:_BoM_ACCESS
  2019     7    16  1.0E36 1.0E36 999 1.0E36 Missing_value
"""


def test_rmm_parse_and_staleness():
    df = parse(RMM_TEXT)
    assert len(df) == 2 and df.phase.tolist() == [4, 5]
    assert value_on(df, "2019-07-15")["phase"] == 5
    assert value_on(df, "2019-07-17")["phase"] == 5             # 2 days old: still used
    stale = value_on(df, "2019-07-25")
    assert stale["stale"] and stale["amplitude"] == 0.0 and stale["phase"] == 0
    assert len(align_to_dates(df, pd.date_range("2019-07-14", "2019-07-16"))) == 3


IBTRACS_CSV = """SID,SEASON,ISO_TIME,LAT,LON,NEWDELHI_GRADE,NEWDELHI_WIND,OTHER
 ,Year,,degrees_north,degrees_east,,kts,
2019X1,2019,2019-08-06 00:00:00,20.5,88.0,D,25,a
2019X1,2019,2019-08-06 03:00:00,20.9,87.4,D,25,b
2019X2,2019,2019-12-01 00:00:00,10.0,85.0,CS,40,c
"""


def test_ibtracs_parse_skips_units_row_and_filters_months():
    df = parse_ibtracs(io.StringIO(IBTRACS_CSV), years=(2019, 2019))
    assert list(df.columns) == ["sid", "time", "lat", "lon", "grade", "wind_kt"]
    assert len(df) == 2 and df.lat.dtype.kind == "f"


def test_imd_clean_and_grid_check():
    lat = np.arange(38.5, 6.49, -0.25); lon = np.arange(66.5, 100.01, 0.25)
    da = xr.DataArray(np.full((2, len(lat), len(lon)), -999.0), dims=("time", "lat", "lon"),
                      coords={"time": pd.date_range("2019-07-01", periods=2), "lat": lat, "lon": lon})
    da[:, 10, 10] = 5.0
    c = imd.clean(da)
    assert c.name == "o_rain" and c.lat.values[0] < c.lat.values[-1] and int(c.notnull().sum()) == 2
    assert c.sizes["lat"] == 129 and c.sizes["lon"] == 135
    imd.check_grid(c, [6.5, 38.5, 66.5, 100.0])
    with pytest.raises(ValueError):
        imd.check_grid(c, [6.0, 38.5, 66.5, 100.0])
    assert imd.land_mask(c).sum() == 1


def test_imd_missing_years(tmp_path):
    (tmp_path / "rain").mkdir()
    (tmp_path / "rain" / "2016.grd").write_bytes(b"")
    assert imd.missing_years(tmp_path, [2016, 2017]) == [2017]


def test_cli_ingest_mjo_from_file(tmp_path, monkeypatch):
    from regimerain import cli
    f = tmp_path / "rmm.txt"; f.write_text(RMM_TEXT)
    monkeypatch.setenv("RMM_FILE", str(f))
    assert cli.main(["ingest", "--only", "mjo", "--years", "2019", "--data-root", str(tmp_path)]) == 0
    out = pd.read_parquet(tmp_path / "data" / "raw" / "mjo" / "rmm.parquet")
    assert len(out) == 2


def test_climatology_tolerates_a_few_missing_years(tmp_path):
    from regimerain.label.build import available_climatology
    cfg = deep_merge(load_config(), {"paths": {"imd": str(tmp_path / "imd"), "raw": str(tmp_path / "raw")}})
    (tmp_path / "imd" / "rain").mkdir(parents=True)
    for y in range(1981, 2016):
        if y != 1983:
            (tmp_path / "imd" / "rain" / f"{y}.grd").write_bytes(b"x")
    have = available_climatology(cfg, list(range(1981, 2016)), [2019], log=lambda *_: None)
    assert 1983 not in have and len(have) == 34
    with pytest.raises(FileNotFoundError):
        available_climatology(cfg, list(range(1981, 2016)), [2019], log=lambda *_: None, min_years=35)

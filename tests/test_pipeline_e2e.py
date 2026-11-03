"""End-to-end pilot on a tiny synthetic world: static -> label -> features -> backtest -> report.

Truth has regime-dependent forecast errors (model too dry on 'active' days, too wet on 'break' days),
so the regime-aware corrector has something real to learn.
"""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("lightgbm")
pytest.importorskip("zarr")

from regimerain import cli
from regimerain.config import deep_merge, load_config
from regimerain.truth import truth_path

SEASONS = [2019, 2020, 2021]
LAT = np.arange(10.0, 14.01, 0.25)
LON = np.arange(70.0, 75.01, 0.25)
LATD = np.arange(8.0, 16.01, 0.25)
LOND = np.arange(68.0, 77.01, 0.25)


def make_world(root, rng):
    raw = root / "raw"
    for y in SEASONS:
        days = pd.date_range(f"{y}-06-01", f"{y}-09-30")
        # regime signal: sine-like active/break alternation in the domain-mean rainfall
        phase = np.sin(np.arange(len(days)) / 6.0)
        base = rng.gamma(0.7, 12, (len(days), len(LAT), len(LON)))
        o = base * (1.0 + 1.2 * phase[:, None, None]).clip(0.1)
        o[:, 0, 0] = np.nan                                          # an ocean cell
        truth_path_ = truth_path({"paths": {"raw": str(raw)}}, y)
        truth_path_.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset({"o_rain": (("time", "lat", "lon"), o.astype("float32"))},
                   coords={"time": days, "lat": LAT, "lon": LON}).to_zarr(truth_path_, mode="w")
        inits = days                                                  # lead 1 -> valid == init date
        bias = np.where(phase > 0, 0.6, 1.5)[:, None, None]           # active: too dry, break: too wet
        f = np.nan_to_num(o) * bias * rng.uniform(0.8, 1.2, o.shape)
        rain = xr.DataArray(f[:, None].astype("float32"), dims=("init", "lead", "lat", "lon"),
                            coords={"init": inits, "lead": [1], "lat": LAT, "lon": LON}).to_dataset(name="f_rain")
        (raw / "hres").mkdir(parents=True, exist_ok=True)
        rain.to_zarr(raw / "hres" / f"rain_{y}.zarr", mode="w")
        shp = (len(inits), 1, len(LATD), len(LOND))
        u = (8 + 6 * phase)[:, None, None, None] + rng.normal(0, 1, shp)
        dyn = xr.Dataset({k: (("init", "lead", "lat", "lon"), v.astype("float32")) for k, v in {
            "u850": u, "v850": rng.normal(0, 2, shp), "mslp": 100000 + rng.normal(0, 200, shp),
            "w500": rng.normal(0, 0.1, shp), "pw": 45 + 5 * u / 10, "ivtx": 300 + 20 * u, "ivty": rng.normal(0, 50, shp)}.items()},
            coords={"init": inits, "lead": [1], "lat": LATD, "lon": LOND})
        dyn.to_zarr(raw / "hres" / f"dyn_{y}.zarr", mode="w")
    (raw / "tracks").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sid": ["X"] * 3, "time": pd.to_datetime(["2020-08-10 06:00", "2020-08-10 12:00", "2020-08-11 06:00"]),
                  "lat": [12.0, 12.2, 12.5], "lon": [72.0, 72.5, 73.0], "grade": "D", "wind_kt": 25.0}
                 ).to_parquet(raw / "tracks" / "ibtracs_ni.parquet")


@pytest.fixture()
def cfg_file(tmp_path):
    rng = np.random.default_rng(0)
    make_world(tmp_path, rng)
    over = {
        "seasons": SEASONS, "leads": [1], "num_threads": 2,
        "paths": {k: str(tmp_path / v) for k, v in {"raw": "raw", "static": "data/static", "table": "data/table",
                                                   "cache": "cache", "reports": "reports", "imd": "imd"}.items()},
        "grid": {"rain_box": [10.0, 14.0, 70.0, 75.0]},
        "labels": {"climatology": "self", "cmz_box": [10, 14, 70, 75], "depression_radius_km": 150},
        "features": {"llj_box": [8, 12, 68, 72], "trough_lon_band": [72, 76], "trough_lat_band": [9, 15],
                     "bob_box": [12, 16, 73, 77], "vort_max_filter_cells": 5},
        "classifier": {"params": {"num_boost_round": 60, "min_data_in_leaf": 50}},
        "qm": {"n0_days": 10},
        "verify": {"bootstrap_B": {"fast": 30, "full": 30}},
    }
    import yaml
    f = tmp_path / "test.yaml"
    f.write_text(yaml.safe_dump(over))
    return [str(cli.REPO_ROOT / "config" / "default.yaml"), str(f)], tmp_path


def run(args, cfgs):
    argv = []
    for c in cfgs:
        argv += ["--config", c]
    assert cli.main(args + argv) == 0


def test_pilot_end_to_end(cfg_file, capsys):
    cfgs, root = cfg_file
    run(["static"], cfgs)
    run(["label"], cfgs)
    counts = pd.read_csv(root / "data" / "labels" / "label_counts.csv")
    assert counts.days_active.sum() > 0 and counts.days_break.sum() > 0
    assert counts.loc[counts.season == 2020, "cells_depression"].iloc[0] > 0
    run(["features"], cfgs)
    t = pd.read_parquet(root / "data" / "table" / "lead=1" / "season=2019" / "part.parquet")
    assert len(t) == 122 * (len(LAT) * len(LON) - 1)                   # every JJAS day x land cell
    assert t.f_llj_index.nunique() > 50 and t.o_rain.notna().all()
    run(["backtest", "--variants", "raw,A,B"], cfgs)
    for y in SEASONS:
        assert (root / "cache" / f"fold={y}" / "DONE").exists()
    capsys.readouterr()
    run(["report"], cfgs)
    out = capsys.readouterr().out
    assert "Regime-aware QM" in out and "Where correction did not help" in out
    rep_dir = next((root / "reports").iterdir())
    rep = json.loads((rep_dir / "verification_report.json").read_text())
    pooled = {r["variant"]: r for r in rep["rows"]
              if r["fold"] == "pooled" and r["synoptic"] == "all" and r["threshold"] == 15.6 and r["lead"] == 1}
    assert set(pooled) == {"raw", "A", "B"}
    # regime-dependent bias -> regime-aware correction should beat raw on RMSE in this synthetic world
    assert pooled["B"]["rmse"] < pooled["raw"]["rmse"]
    for png in ("metrics_heavy.png", "fss_vs_scale.png", "map_wettest_day.png"):
        assert (rep_dir / png).exists()

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
                                                   "cache": "cache", "reports": "reports", "imd": "imd",
                                                   "runs": "runs", "models": "models"}.items()},
        "grid": {"rain_box": [10.0, 14.0, 70.0, 75.0]},
        "labels": {"climatology": "self", "cmz_box": [10, 14, 70, 75], "depression_radius_km": 150},
        "features": {"llj_box": [8, 12, 68, 72], "trough_lon_band": [72, 76], "trough_lat_band": [9, 15],
                     "bob_box": [12, 16, 73, 77], "vort_max_filter_cells": 5},
        "classifier": {"params": {"num_boost_round": 60, "min_data_in_leaf": 50}},
        "qm": {"n0_days": 10},
        "exceed": {"params": {"min_data_in_leaf": 50, "num_boost_round": 200, "early_stopping_rounds": 20},
                   "min_events": 5},
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
    for png in ("metrics_heavy.png", "fss_vs_scale.png", "map_wettest_day.png", "reliability_exceed.png"):
        assert (rep_dir / png).exists()
    # heavy-rain probability model (M4): written per fold, scored in the report, calibrated on inner_val only
    preds = pd.read_parquet(root / "cache" / "fold=2021" / "predictions.parquet")
    p = preds[["p_exc_64.5", "p_exc_115.6"]].to_numpy()
    assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1 and (p[:, 1] <= p[:, 0] + 1e-7).all()
    ex = json.loads((root / "cache" / "fold=2021" / "exceed.json").read_text())
    assert ex["models"][0]["calibrated_on"] == [ex["inner_val"]] and ex["inner_val"] != 2021
    exc = [r for r in rep["exceedance"] if r["fold"] == "pooled" and r["threshold"] == 64.5]
    assert len(exc) == 1 and exc[0]["n_events"] > 0 and exc[0]["brier"] is not None
    assert "Heavy-rain probability (M4" in (rep_dir / "summary.md").read_text(encoding="utf-8")
    # skill gate: one table row per unit x lead x regime, a map, and a summary in the report
    gate = pd.read_parquet(rep_dir / "skill_gate.parquet")
    assert set(gate.gate.unique()) <= {-1, 0, 1} and (gate.regime == "all").any()
    assert sum(rep["skill_gate"]["counts_all_regimes"].values()) == int((gate.regime == "all").sum())
    assert (rep_dir / "skill_gate_map.png").exists()
    assert "## Skill gate" in (rep_dir / "summary.md").read_text(encoding="utf-8")
    # replay: real held-out days exported for the web, valid against web contract v2, never synthetic
    from regimerain.runs.contract import check_folder
    capsys.readouterr()
    run(["run", "--source", "replay"], cfgs)
    printed = capsys.readouterr().out
    assert "replay, held-out" in printed
    webs = sorted((root / "runs").glob("*/web"))
    assert webs, "no replay export written"
    for web in webs:
        assert check_folder(web) == []
        man = json.loads((web / "manifest.json").read_text())
        assert man["kind"] == "replay" and man["synthetic"] is False
        assert {"truth", "p_heavy", "p_very_heavy", "corrected_global"} <= set(man["layers"])
        assert man["provenance"]["held_out_season"] in SEASONS and man["truth_source"].startswith("IMD")
        ver = json.loads((web / "verification.json").read_text())
        assert ver["synthetic"] is False and ver["entries"] and ver["reliability"]
    # fit-final: every model refit on all seasons, hashed; the loaded set predicts sane values
    run(["fit-final"], cfgs)
    from regimerain.final import ModelSet
    model_dir = next((root / "models").iterdir())
    for f in ("classifier/booster.txt", "classifier/temperature.json", "qm/experts.parquet", "exceed/64.5.iso.pkl",
              "gate/skill_gate.parquet", "static.npz", "config.yaml", "feature_list.json", "model_card.json"):
        assert (model_dir / f).exists(), f
    ms = ModelSet(model_dir)
    rows = pd.read_parquet(root / "data" / "table" / "lead=1" / "season=2020" / "part.parquet")
    rows = rows[rows.init == rows.init.iloc[0]].assign(lead=np.int16(1))
    out = ms.predict(rows)
    P = out[["p_active", "p_break", "p_depression", "p_normal"]].to_numpy()
    assert np.allclose(P.sum(axis=1), 1, atol=1e-5) and (out.B >= 0).all() and (out.A >= 0).all()
    assert (out["p_exc_115.6"] <= out["p_exc_64.5"] + 1e-7).all()
    card = json.loads((model_dir / "model_card.json").read_text())
    assert card["model_set_id"] == model_dir.name and card["classifier"]["rounds"] >= 10
    # live run: a forecast in the fetch_gfs layout (here the synthetic HRES of 1 Jul 2020) through the model set
    from regimerain.runs.export_web import export_run
    from regimerain.runs.live import live_fields
    init = np.datetime64("2020-07-01")
    rain = xr.open_zarr(root / "raw" / "hres" / "rain_2020.zarr")["f_rain"].sel(init=init)
    dz = xr.open_zarr(root / "raw" / "hres" / "dyn_2020.zarr").sel(init=init)
    forecast = {"rain": {1: rain.sel(lead=1).values}, "lat": rain.lat.values, "lon": rain.lon.values,
                "dyn": {1: {k: dz[k].sel(lead=1).values for k in dz.data_vars}},
                "latd": dz.lat.values, "lond": dz.lon.values}
    from datetime import date
    f = live_fields(json.loads(json.dumps(ms.config)) | {"paths": {"raw": str(root / "raw")}}, date(2020, 7, 1),
                    ms, forecast)
    assert f.kind == "model" and f.source == "gfs" and f.truth is None
    assert {"wind850", "p_heavy", "p_very_heavy", "corrected_global"} <= set(f.layers())
    web = export_run(f, ms.config | {"paths": {"runs": str(root / "runs")}}, root / "runs",
                     curves=ms.curves, regime_days=ms.regime_days)
    assert check_folder(web) == []
    man = json.loads((web / "manifest.json").read_text())
    assert man["kind"] == "model" and man["synthetic"] is False and man["forecast_source"] == "NCEP GFS 0.25"
    assert man["provenance"]["model_set_id"] == model_dir.name and "transfer" in man["provenance"]["note"]
    assert man["provenance"]["model_hashes"] and man["regime_days"]
    with open(model_dir / "config.yaml", "a", encoding="utf-8") as fh:
        fh.write("# edited after fit-final\n")
    with pytest.raises(ValueError, match="changed since fit-final"):
        ModelSet(model_dir)

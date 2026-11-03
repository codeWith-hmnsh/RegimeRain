"""Export one run to web contract v2 (`runs/<run_id>/web/*.json`, BACKEND_BUILD_PLAN section 3).

Inputs: `RunFields` from any predictor, optionally a backtest report (the dict
regimerain.verify.report writes, TRD 3.6), QM curves and per-regime training-day counts. Every file
is built through the pydantic models in `contract.py`, so an export that breaks the contract fails
here rather than in the browser.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from regimerain.features.schema import GEO, SYNOPTIC
from regimerain.runs import contract as C
from regimerain.runs.derive import depression_track, place_values, regime_index, wettest
from regimerain.runs.fields import RunFields, lead_hours, valid_date

KM_PER_DEG = 111.2
METRICS = ("rmse", "ets", "csi", "pod", "far")


def make_run_id(f: RunFields) -> str:
    """TRD 3.5: {init}T00Z_{source}_{model_set_id[:8]}."""
    tag = {"mock": "mock", "replay": f"replay-{(f.provenance.get('backtest_id') or 'x')[:8]}"}.get(
        f.kind, (f.provenance.get("model_set_id") or "model")[:8])
    return f"{f.init:%Y%m%d}T00Z_{f.source}_{tag}"


def _rows(a: np.ndarray, land: np.ndarray, nd: int) -> list[list]:
    """(lead, nlat, nlon) -> per lead, a flat row-major list rounded to nd places, None off land."""
    out = []
    flat_land = land.ravel()
    for plane in a:
        vals = np.round(plane.ravel().astype("float64"), nd)
        out.append([float(v) if m and math.isfinite(v) else None for v, m in zip(vals, flat_land)])
    return out


def _full(a: np.ndarray, nd: int) -> list[list[float]]:
    return [np.round(plane.ravel().astype("float64"), nd).tolist() for plane in a]


def _num(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


# ------------------------------------------------------------------------------------ verification
def fss_windows(windows, step_deg: float) -> list[dict]:
    return [{"key": f"fss_{w}", "cells": int(w), "km": int(round(int(w) * step_deg * KM_PER_DEG))} for w in sorted(windows, key=int)]


def verification_from_report(report: dict, thresholds: dict[str, float], step_deg: float, synthetic: bool) -> dict:
    """Map report rows (one per variant) to web entries (baseline = raw, corrected = B, global = A)."""
    name_of = {round(float(v), 1): k for k, v in thresholds.items()}
    rows = [r for r in report["rows"] if r.get("geo", "all") == "all" and round(float(r["threshold"]), 1) in name_of]
    windows = sorted({int(w) for r in rows for w in (r.get("fss") or {})})

    def scores(r):
        s = {m: _num(r.get(m)) for m in METRICS}
        s.update({f"fss_{w}": _num((r.get("fss") or {}).get(str(w))) for w in windows})
        return s

    table = {}
    for r in rows:
        key = (str(r["fold"]), int(r["lead"]), r["synoptic"], name_of[round(float(r["threshold"]), 1)])
        table.setdefault(key, {})[r["variant"]] = r
    entries = []
    for (fold, lead, syn, th), by_var in table.items():
        if "raw" not in by_var or "B" not in by_var:
            continue
        e = {"fold": fold, "lead": lead, "threshold": th, "regime": None if syn == "all" else syn,
             "baseline": scores(by_var["raw"]), "corrected": scores(by_var["B"]),
             "global": scores(by_var["A"]) if "A" in by_var else None,
             "n_samples": int(by_var["B"].get("n_cell_days", 0)), "n_events": by_var["B"].get("n_events_obs")}
        entries.append(e)
    entries.sort(key=lambda e: (e["fold"] != "pooled", e["fold"], e["lead"], e["threshold"], e["regime"] or ""))

    deltas = []
    for d in report.get("deltas", []):
        t = d.get("threshold")
        if t is not None and round(float(t), 1) not in name_of:
            continue
        ci = d.get("ci90")
        deltas.append({"variant": d["variant"], "vs": d["vs"], "lead": int(d["lead"]),
                       "threshold": None if t is None else name_of[round(float(t), 1)], "metric": d["metric"],
                       "delta": _num(d.get("delta")), "ci90": [_num(x) for x in ci] if ci else None,
                       "significant": d.get("significant")})

    leads = sorted({e["lead"] for e in entries}) or list(report.get("leads", [1]))
    headline = leads[0]
    reliability = None
    exc = report.get("exceedance")
    if exc:
        reliability = {}
        for tval, name in ((thresholds["heavy"], "heavy"), (thresholds["very_heavy"], "very_heavy")):
            rel = (exc.get(str(tval)) or {}).get("lead", {}).get(str(headline), {}).get("reliability")
            if rel:
                reliability[name] = [{"p_forecast": p, "p_observed": o, "n": int(n)}
                                     for p, o, n in zip(rel["bin_mean_p"], rel["obs_freq"], rel["count"])]
        reliability = reliability or None

    clf = report.get("classifier")
    classifier = None
    if isinstance(clf, list) and clf:
        acc = [c["test_metrics"]["accuracy"] for c in clf if c.get("test_metrics")]
        f1 = [c["test_metrics"]["macro_f1"] for c in clf if c.get("test_metrics")]
        classifier = {"accuracy_mean": round(float(np.mean(acc)), 3), "macro_f1_mean": round(float(np.mean(f1)), 3),
                      "folds": [{"season": c["test"], **{k: c["test_metrics"].get(k) for k in ("accuracy", "macro_f1", "per_class_recall")}}
                                for c in clf]}

    seasons = [int(s) for s in report.get("seasons", [])]
    return {"synthetic": synthetic, "backtest_id": str(report.get("backtest_id")),
            "cv": f"leave-one-monsoon-out, {len(seasons)} seasons" + (f" ({min(seasons)}-{max(seasons)})" if seasons else ""),
            "seasons": seasons, "leads": leads, "headline_lead": headline,
            "variants": [v for v in report.get("variants", ["raw", "A", "B"]) if v in ("raw", "A", "B")],
            "fss_windows": fss_windows(windows, step_deg), "entries": entries, "deltas": deltas,
            "reliability": reliability, "classifier": classifier}


# ------------------------------------------------------------------------------------ export
def build_files(f: RunFields, cfg: dict, report: dict | None = None, curves: list[dict] | None = None,
                regime_days: dict | None = None, run_id: str | None = None, truth_source: str | None = None) -> dict:
    """Validated contract models for one run, keyed by file name."""
    f.validate()
    run_id = run_id or make_run_id(f)
    heavy, very_heavy = cfg["exceed"]["thresholds"]
    thresholds = {"heavy": float(heavy), "very_heavy": float(very_heavy)}
    step = float(np.round(np.median(np.diff(f.lat)), 4))
    syn = f.synthetic
    land = f.land

    leads = []
    for k, L in enumerate(f.leads):
        a, b = lead_hours(L)
        leads.append({"index": k, "day": L, "valid_date": valid_date(f.init, L).isoformat(), "lead_hours": b, "hours": [a, b]})
    probs = np.round(f.p_synoptic * 100).astype(int)
    probs[:, ~land] = 0
    layers = {"raw": _rows(f.raw, land, 1), "corrected": _rows(f.corrected, land, 1),
              "regime": [r.ravel().tolist() for r in regime_index(f)]}
    if f.corrected_global is not None:
        layers["corrected_global"] = _rows(f.corrected_global, land, 1)
    for name in ("p_heavy", "p_very_heavy"):
        if getattr(f, name) is not None:
            layers[name] = _rows(getattr(f, name), land, 2)
    if f.truth is not None:
        layers["truth"] = _rows(f.truth, land, 1)
    if f.u850 is not None and f.v850 is not None:
        layers["u850"], layers["v850"] = _full(f.u850, 1), _full(f.v850, 1)
        layers["wind850"] = _full(np.hypot(f.u850, f.v850), 1)

    grid = C.Grid(synthetic=syn, lat0=float(f.lat[0]), lon0=float(f.lon[0]), step=step, nlat=len(f.lat), nlon=len(f.lon),
                  synoptic=list(SYNOPTIC), geo_classes=list(GEO), leads=leads, geo=f.geo.astype(int).ravel().tolist(),
                  layers=layers, regime_probs_pct=[p.reshape(-1).tolist() for p in probs])

    prov = {k: v for k, v in f.provenance.items() if k in C.Provenance.model_fields}
    manifest = C.Manifest(
        synthetic=syn, kind=f.kind, run_id=run_id, created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        forecast_issue_date=f.init.isoformat(),
        forecast_source={"mock": "synthetic sample (not real)", "hres": "ECMWF IFS HRES (WeatherBench2)", "gfs": "NCEP GFS 0.25"}.get(f.source, f.source),
        truth_source=truth_source if not syn else None, grid_step_deg=step, thresholds_mm=thresholds,
        synoptic=list(SYNOPTIC), geo_classes=list(GEO), layers=f.layers(), regime_days=regime_days,
        depression_track=depression_track(f), wettest=wettest(f), provenance=prov)

    files = {"manifest": manifest, "grid": grid,
             "places": C.Places(synthetic=syn, places=place_values(f, curves, regime_days))}
    if curves:
        files["qm_curves"] = C.QmCurves(synthetic=syn, curves=curves)
    if report:
        files["verification"] = C.Verification.model_validate(verification_from_report(report, thresholds, step, syn))
    return files


def write_export(files: dict, out_dir: str | Path) -> Path:
    """Write atomically: build in a temp folder next to the target, then swap it in."""
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=out_dir.parent))
    tmp.chmod(0o755)                      # mkdtemp is 0700; exports are served to other users
    try:
        for name, model in files.items():
            (tmp / f"{name}.json").write_text(C.dump(model), encoding="utf-8")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(tmp, out_dir)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return out_dir


def export_run(f: RunFields, cfg: dict, runs_dir: str | Path, **kw) -> Path:
    """runs/<run_id>/web/ for one run. Returns the web folder."""
    files = build_files(f, cfg, **kw)
    run_id = files["manifest"].run_id
    return write_export(files, Path(runs_dir) / run_id / "web")

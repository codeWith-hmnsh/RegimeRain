"""Replay producer (BACKEND_BUILD_PLAN phase B3): a real held-out backtest day as a web run.

Reads `cache/fold=<season>/predictions.parquet` (written by `regimerain backtest`) and the newest
`reports/*/verification_report.json`. Every value is real and out-of-sample: raw HRES rain, the global (A) and
regime-aware (B) corrections and the M4 heavy-rain probabilities all come from models that never saw that
season, and IMD rain is the truth for the valid days. Nothing is retrained here.

Wind and MSLP layers need the HRES dynamics files, which the Kaggle backtest keeps in scratch only, so a
replay run normally has no wind layer; the exporter then leaves it out and the web hides it.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from regimerain.exceed.model import prob_column
from regimerain.features.schema import REGIME_PROB_FEATURES, SYNOPTIC
from regimerain.folds import fold_dir
from regimerain.runs.fields import RunFields

SHOWCASE = ("depression", "active", "break")


def latest_report(cfg: dict) -> dict | None:
    files = sorted(Path(cfg["paths"]["reports"]).glob("*/verification_report.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def _predictions_path(cfg: dict, season: int) -> Path:
    p = fold_dir(cfg["paths"]["cache"], season) / "predictions.parquet"
    if not p.exists():
        raise FileNotFoundError(f"no held-out predictions for season {season} at {p}: run `backtest --folds {season}`")
    return p


def regime_days(cfg: dict, season: int) -> dict[str, int] | None:
    """Training days behind each regime's correction curves in this fold (from qm_counts_B.csv)."""
    p = fold_dir(cfg["paths"]["cache"], season) / "qm_counts_B.csv"
    if not p.exists():
        return None
    c = pd.read_csv(p)
    c = c[c.regime >= 0] if "regime" in c else c
    days = c.groupby("regime").n_days.max()
    return {SYNOPTIC[int(r)]: int(n) for r, n in days.items() if 0 <= int(r) < len(SYNOPTIC)} or None


def fold_curves(cfg: dict, season: int) -> list[dict] | None:
    """The fold's QM display curves (written by `backtest`), for the web's correction charts."""
    p = fold_dir(cfg["paths"]["cache"], season) / "qm_curves.json"
    return json.loads(p.read_text(encoding="utf-8")) or None if p.exists() else None


def replay_fields(cfg: dict, init: date, report: dict | None = None) -> RunFields:
    from regimerain.static.basic import load_static
    d = pd.read_parquet(_predictions_path(cfg, init.year), filters=[("init", "==", pd.Timestamp(init))])
    if d.empty:
        raise ValueError(f"init {init} is not in the held-out predictions of season {init.year}")
    st = load_static(cfg)
    lat, lon = st.lat.values, st.lon.values
    leads = sorted(int(x) for x in d.lead.unique())
    n_lead, ny, nx = len(leads), len(lat), len(lon)
    k = d.lead.map({L: i for i, L in enumerate(leads)}).to_numpy()
    iy, ix = d.lat_idx.to_numpy(), d.lon_idx.to_numpy()

    def grid(col: str) -> np.ndarray:
        a = np.full((n_lead, ny, nx), np.nan, "float32")
        a[k, iy, ix] = d[col].to_numpy("float32")
        return a

    count = np.zeros((ny, nx), int)
    np.add.at(count, (iy, ix), 1)
    land = st["land"].values.astype(bool) & (count == n_lead)      # a cell is shown only with every lead
    P = np.zeros((n_lead, ny, nx, len(SYNOPTIC)), "float32")
    for j, c in enumerate(REGIME_PROB_FEATURES):
        P[k, iy, ix, j] = d[c].to_numpy("float32")
    geo = np.where(land, st["geo"].values, -1).astype("int8")
    heavy, very_heavy = cfg["exceed"]["thresholds"]
    optional = lambda col: grid(col) if col in d else None
    rep = report or {}
    return RunFields(
        kind="replay", init=init, source="hres", lat=lat, lon=lon, leads=leads, land=land,
        raw=grid("raw").clip(min=0), corrected=grid("B"), p_synoptic=P, geo=geo,
        corrected_global=optional("A"), p_heavy=optional(prob_column(heavy)),
        p_very_heavy=optional(prob_column(very_heavy)), truth=grid("o_rain"),
        provenance={"backtest_id": rep.get("backtest_id"), "config_sha256": rep.get("config_sha256"),
                    "git_commit": rep.get("git_commit"), "held_out_season": int(init.year),
                    "note": f"held-out hindcast: every model of this fold was trained without season {init.year}"})


def showcase_inits(cfg: dict, threshold: float = 64.5) -> list[tuple[date, str]]:
    """One init per showcase regime: the held-out day with the most observed heavy-rain cells (lead 1)."""
    days = []
    for s in cfg["seasons"]:
        try:
            p = _predictions_path(cfg, s)
        except FileNotFoundError:
            continue
        d = pd.read_parquet(p, columns=["init", "lead", "o_rain", "y_synoptic"])
        d = d[d.lead == d.lead.min()]
        y = d.y_synoptic.to_numpy()
        g = pd.DataFrame({"init": d.init.to_numpy(), "heavy": d.o_rain.to_numpy() >= threshold,
                          "depression": y == 2, "active": y == 0, "break": y == 1}).groupby("init").mean()
        g["heavy_cells"] = d.assign(h=d.o_rain >= threshold).groupby("init").h.sum()
        days.append(g)
    if not days:
        raise FileNotFoundError("no backtest predictions in the cache: run `regimerain backtest` first")
    g = pd.concat(days)
    rules = {"depression": g.depression >= 0.05, "active": g.active >= 0.5, "break": g["break"] >= 0.5}
    picks, seen = [], set()
    for label in SHOWCASE:
        cand = g[rules[label]]
        if cand.empty:
            continue
        init = pd.Timestamp(cand.heavy_cells.idxmax()).date()
        if init not in seen:
            picks.append((init, label)); seen.add(init)
    if not picks:
        picks.append((pd.Timestamp(g.heavy_cells.idxmax()).date(), "wettest"))
    return picks

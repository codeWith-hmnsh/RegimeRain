"""Live run (MODEL_SPEC 18, BACKEND_BUILD_PLAN B5): today's GFS through the fitted model set to a web run.

    GFS init -> rain-day rain + dynamics -> features (the same `forecast_rows` as the training table)
             -> ModelSet.predict (M1 regimes, M3 mixture QM, M4 heavy-rain probabilities)
             -> skill gate (raw rain where the gate is OFF) -> RunFields(kind="model", source="gfs")

Nothing is trained here. The models were fitted on IFS HRES, so on GFS the correction is a transfer; the
manifest says so. No observed truth exists for a live run, so it has no truth layer.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from regimerain.exceed.model import prob_column
from regimerain.features.build import _rmm, cmz_mask, forecast_rows
from regimerain.features.schema import REGIME_PROB_FEATURES, SYNOPTIC
from regimerain.runs.fields import RunFields

def season_note(init: date) -> str:
    """Warn when the init is outside the June-September monsoon the models were trained on."""
    if 6 <= init.month <= 9:
        return ""
    return (" OUTSIDE THE MONSOON: the models were trained on June-September only, so the regime names and"
            " corrections are not meaningful for this date; treat this run as a pipeline demo.")


TRANSFER_NOTE = ("models fitted on ECMWF IFS HRES 2016-2022; applied to NOAA GFS here, so the correction is a "
                 "transfer between models (MODEL_SPEC 18.2)")


def latest_model_set(cfg: dict) -> Path:
    root = Path(cfg["paths"]["models"])
    sets = sorted((p for p in root.glob("*") if (p / "model_card.json").exists()), key=lambda p: p.stat().st_mtime)
    if not sets:
        raise FileNotFoundError(f"no model set under {root}: run `regimerain fit-final` (or download ps26080-models)")
    return sets[-1]


def _check_grid(ms, lat, lon) -> None:
    st = ms.static
    if not (np.allclose(st.lat.values, lat) and np.allclose(st.lon.values, lon)):
        raise ValueError("live rain grid differs from the model set's static grid")


def live_fields(cfg: dict, init: date, model_set, forecast: dict, source: str = "gfs") -> RunFields:
    """`forecast`: output of `regimerain.ingest.gfs.fetch_gfs` (or the same layout from another model)."""
    ms = model_set
    st = ms.static
    lat, lon = st.lat.values, st.lon.values
    _check_grid(ms, forecast["lat"], forecast["lon"])
    leads = sorted(forecast["rain"])
    rmm = _rmm(cfg)
    cmz = cmz_mask(cfg, st)
    rows = pd.concat([forecast_rows(cfg, pd.Timestamp(init), L, forecast["rain"][L], forecast["dyn"][L],
                                    forecast["latd"], forecast["lond"], st, rmm, cmz).assign(lead=np.int16(L))
                      for L in leads], ignore_index=True)
    out = ms.predict(rows)
    served = ms.gate_served(out)
    raw = rows.f_rain.to_numpy("float32")
    corrected = np.where(served, out.B.to_numpy("float32"), raw)

    n_lead, ny, nx = len(leads), len(lat), len(lon)
    k = rows.lead.map({L: i for i, L in enumerate(leads)}).to_numpy()
    iy, ix = rows.lat_idx.to_numpy(), rows.lon_idx.to_numpy()

    def grid(values) -> np.ndarray:
        a = np.full((n_lead, ny, nx), np.nan, "float32")
        a[k, iy, ix] = np.asarray(values, "float32")
        return a

    P = np.zeros((n_lead, ny, nx, len(SYNOPTIC)), "float32")
    for j, c in enumerate(REGIME_PROB_FEATURES):
        P[k, iy, ix, j] = out[c].to_numpy("float32")
    land = st["land"].values.astype(bool)
    geo = np.where(land, st["geo"].values, -1).astype("int8")

    # wind and pressure on the rain grid (the dynamics grid contains it), shown as the web's wind layer
    jy = np.searchsorted(forecast["latd"], lat)
    jx = np.searchsorted(forecast["lond"], lon)
    cut = lambda name: np.stack([forecast["dyn"][L][name][np.ix_(jy, jx)] for L in leads]).astype("float32")

    heavy, very_heavy = cfg["exceed"]["thresholds"]
    col = lambda t: grid(out[prob_column(t)]) if prob_column(t) in out else None
    m = rmm is not None and _mjo_fresh(rmm, init)
    card = ms.card
    fields = RunFields(
        kind="model", init=init, source=source, lat=lat, lon=lon, leads=leads, land=land,
        raw=grid(raw).clip(min=0), corrected=grid(corrected), p_synoptic=P, geo=geo,
        corrected_global=grid(out.A), p_heavy=col(heavy), p_very_heavy=col(very_heavy),
        u850=cut("u850"), v850=cut("v850"), mslp=cut("mslp") / 100.0,
        provenance={"model_set_id": card["model_set_id"], "model_hashes": ms.hashes,
                    "config_sha256": card.get("config_sha256"), "git_commit": card.get("git_commit"),
                    "note": (f"{TRANSFER_NOTE}. Skill gate OFF on {int((~served).sum())} land cell-days "
                             f"(raw rain served there). MJO: {'latest RMM' if m else 'neutral (index missing or stale)'}. "
                             f"Trained on seasons {card.get('seasons')}.{season_note(init)}")})
    fields.reasons = place_reasons(fields, ms, rows, out, k)
    return fields


def place_reasons(f: RunFields, ms, rows: pd.DataFrame, out: pd.DataFrame, k_of_row: np.ndarray) -> dict:
    """Reasons for each named place's cell and lead (only those rows: TreeSHAP on every cell is wasted work)."""
    from regimerain.runs.derive import nearest_land_cell
    from regimerain.runs.places import PLACES
    cells = {c for _, la, lo in PLACES if (c := nearest_land_cell(f, la, lo)) is not None}
    if not cells:
        return {}
    key = pd.Series(list(zip(k_of_row, rows.lat_idx, rows.lon_idx)))
    wanted = {(k, i, j) for (i, j) in cells for k in range(len(f.leads))}
    pick = key.isin(wanted).to_numpy()
    sel = rows[pick].reset_index(drop=True)
    reasons = ms.explain(sel, out[pick].reset_index(drop=True))
    return {tuple(int(x) for x in t): r for t, r in zip(key[pick], reasons)}


def _mjo_fresh(rmm: pd.DataFrame, init) -> bool:
    from regimerain.ingest.mjo import value_on
    return not value_on(rmm, init).get("stale", False)

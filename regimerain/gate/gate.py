"""Skill gate (MODEL_SPEC section 14, PRD FR-11).

For each unit × predicted regime × lead, compare the corrected forecast (variant B by default) with raw NWP on
the pooled leave-one-monsoon-out predictions:

- MSE skill score SS = 1 - SSE_corrected / SSE_raw, with a day-block bootstrap 90% interval;
- ETS gain at 64.5 mm/day (only when there are enough observed heavy-rain cells), same bootstrap.

Decision: ON (1) if either measure is significantly better and neither significantly worse; OFF (-1) in the
opposite case; NEUTRAL (0) otherwise. A regime with fewer than `gate.min_days` days inherits the gate of its
unit × lead over all regimes. Live use: serve corrected rain where the gate is ON or NEUTRAL, raw where OFF.

Units are IMD zone × geographic class (12 units) until district boundaries are licensed (PRD §20); the code
only needs a unit column, so districts plug in later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from regimerain.features.schema import GEO, REGIME_PROB_FEATURES, SYNOPTIC, ZONES
from regimerain.verify.bootstrap import block_indices

UNIT = ["zone", "geo"]
STATS = ["n", "se_raw", "se_corr", "H_raw", "M_raw", "FA_raw", "CN_raw", "H_corr", "M_corr", "FA_corr", "CN_corr"]
ON, NEUTRAL, OFF = 1, 0, -1
LABEL = {ON: "ON", NEUTRAL: "NEUTRAL", OFF: "OFF"}


def gate_decision(ss_ci, dets_ci, n_days: int, min_days: int = 30):
    """ss_ci, dets_ci: (lo, hi) intervals; dets_ci may be None. None = too few days, caller falls back."""
    if n_days < min_days:
        return None
    better = ss_ci[0] > 0 or (dets_ci is not None and dets_ci[0] > 0)
    worse = ss_ci[1] < 0 or (dets_ci is not None and dets_ci[1] < 0)
    if better and not worse:
        return ON
    if worse and not better:
        return OFF
    return NEUTRAL


def ets(H, M, FA, CN):
    H, M, FA, CN = (np.asarray(x, float) for x in (H, M, FA, CN))
    n = H + M + FA + CN
    hr = (H + M) * (H + FA) / np.where(n > 0, n, 1)
    den = H + M + FA - hr
    return np.where(den != 0, (H - hr) / np.where(den != 0, den, 1), np.nan)


def unit_day_stats(preds: pd.DataFrame, variant: str = "B", threshold: float = 64.5) -> pd.DataFrame:
    """Summed statistics per (zone, geo, lead, regime, valid); regime = most probable predicted regime per cell."""
    d = preds[["zone", "geo", "lead", "valid", "season", "o_rain", "raw", variant]].copy()
    d["regime"] = np.asarray(preds[REGIME_PROB_FEATURES]).argmax(axis=1).astype("int8")
    d = d[d.o_rain.notna()]
    o, r, c = d.o_rain.to_numpy(float), d.raw.to_numpy(float), d[variant].to_numpy(float)
    ob, rb, cb = o >= threshold, r >= threshold, c >= threshold
    d = d.assign(n=1, se_raw=(r - o) ** 2, se_corr=(c - o) ** 2,
                 H_raw=rb & ob, M_raw=~rb & ob, FA_raw=rb & ~ob, CN_raw=~rb & ~ob,
                 H_corr=cb & ob, M_corr=~cb & ob, FA_corr=cb & ~ob, CN_corr=~cb & ~ob)
    keys = UNIT + ["lead", "regime", "valid", "season"]
    return d.groupby(keys, observed=True)[STATS].sum().reset_index()


def _score(day: pd.DataFrame, B: int, block: int, seed: int, min_events: int, min_days: int) -> dict:
    day = day.sort_values("valid")
    a = {k: day[k].to_numpy(float) for k in STATS}
    n_days, n_events = len(day), int(a["H_raw"].sum() + a["M_raw"].sum())
    ss_point = 1 - a["se_corr"].sum() / a["se_raw"].sum() if a["se_raw"].sum() > 0 else np.nan
    rng = np.random.default_rng(seed)
    idx = np.stack([block_indices(n_days, block, rng) for _ in range(B)]) if n_days else np.zeros((B, 0), int)
    s = {k: a[k][idx].sum(axis=1) for k in STATS}
    ss_boot = 1 - s["se_corr"] / np.where(s["se_raw"] > 0, s["se_raw"], np.nan)
    ss_ci = tuple(np.nanquantile(ss_boot, [0.05, 0.95])) if n_days else (np.nan, np.nan)
    with_ets = n_events >= min_events
    dets = dets_ci = None
    if with_ets:
        dets = float(ets(*(a[f"{k}_corr"].sum() for k in "H M FA CN".split()))
                     - ets(*(a[f"{k}_raw"].sum() for k in "H M FA CN".split())))
        boot = ets(*(s[f"{k}_corr"] for k in "H M FA CN".split())) - ets(*(s[f"{k}_raw"] for k in "H M FA CN".split()))
        dets_ci = tuple(np.nanquantile(boot, [0.05, 0.95]))
    per_season = day.groupby("season")[["se_raw", "se_corr"]].sum()
    return {"n_days": n_days, "n_cell_days": int(a["n"].sum()), "n_obs_heavy": n_events,
            "ss": float(ss_point), "ss_lo": float(ss_ci[0]), "ss_hi": float(ss_ci[1]),
            "dets": dets, "dets_lo": None if dets_ci is None else float(dets_ci[0]),
            "dets_hi": None if dets_ci is None else float(dets_ci[1]),
            "season_wins": int((per_season.se_corr < per_season.se_raw).sum()), "seasons": int(len(per_season)),
            "gate_own": gate_decision(ss_ci, dets_ci, n_days, min_days)}


def skill_gate(preds: pd.DataFrame, cfg: dict, variant: str = "B", B: int = 300, log=print) -> pd.DataFrame:
    """One row per unit × lead × regime ('all' plus each predicted regime) with scores and the final gate."""
    g = cfg.get("gate", {})
    min_days = int(g.get("min_days", 30))
    min_events = int(g.get("min_heavy_events_for_ets", 10))
    block = int(cfg["verify"]["bootstrap_block_days"])
    seed = int(cfg["seed"])
    stats = unit_day_stats(preds, variant)
    rows = []
    all_days = stats.groupby(UNIT + ["lead", "valid", "season"], observed=True)[STATS].sum().reset_index()
    for (z, geo, lead), day in all_days.groupby(UNIT + ["lead"], observed=True):
        rows.append({"zone": int(z), "geo": int(geo), "lead": int(lead), "regime": "all",
                     **_score(day, B, block, seed, min_events, min_days)})
    for (z, geo, lead, reg), day in stats.groupby(UNIT + ["lead", "regime"], observed=True):
        rows.append({"zone": int(z), "geo": int(geo), "lead": int(lead), "regime": SYNOPTIC[int(reg)],
                     **_score(day, B, block, seed, min_events, min_days)})
    df = pd.DataFrame(rows)
    parent = df[df.regime == "all"].set_index(UNIT + ["lead"]).gate_own
    gates, source = [], []
    for r in df.itertuples():
        own = None if pd.isna(r.gate_own) else int(r.gate_own)
        if own is not None:
            gates.append(own); source.append("own")
        else:
            p = parent.get((r.zone, r.geo, r.lead))
            p = None if p is None or pd.isna(p) else int(p)
            gates.append(NEUTRAL if p is None else p)
            source.append("insufficient" if p is None else "unit_lead")
    df["gate"] = np.array(gates, "int8")
    df["gate_source"] = source
    df["unit"] = [f"{ZONES[z]} / {GEO[g]}" for z, g in zip(df.zone, df.geo)]
    log(f"gate: {len(df)} rows, ON {int((df.gate == ON).sum())}, NEUTRAL {int((df.gate == NEUTRAL).sum())}, "
        f"OFF {int((df.gate == OFF).sum())}")
    return df


def gate_summary(df: pd.DataFrame, variant: str) -> dict:
    regime_rows = df[df.regime == "all"]
    return {"variant": variant, "unit": "zone x geo", "rule": "ON/OFF if SS or ETS-gain 90% CI excludes 0",
            "counts_all_regimes": {LABEL[k]: int((regime_rows.gate == k).sum()) for k in (ON, NEUTRAL, OFF)},
            "counts_by_regime": {LABEL[k]: int((df.gate == k).sum()) for k in (ON, NEUTRAL, OFF)},
            "off": df[df.gate == OFF][["unit", "lead", "regime", "ss", "ss_lo", "ss_hi", "dets"]]
                   .to_dict(orient="records")}

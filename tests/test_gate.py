"""Skill gate (MODEL_SPEC 14): rules, and that it finds helpful, harmful and neutral corrections."""
import numpy as np
import pandas as pd

from regimerain.config import deep_merge, load_config
from regimerain.features.schema import REGIME_PROB_FEATURES
from regimerain.gate.gate import NEUTRAL, OFF, ON, gate_decision, gate_summary, skill_gate


def test_gate_rules():
    assert gate_decision((0.01, 0.2), None, 100) == ON
    assert gate_decision((-0.3, -0.05), None, 100) == OFF
    assert gate_decision((-0.1, 0.1), None, 100) == NEUTRAL
    assert gate_decision((-0.1, 0.1), (0.01, 0.05), 100) == ON          # ETS gain alone can switch it on
    assert gate_decision((0.02, 0.1), (-0.05, -0.01), 100) == NEUTRAL   # better on one, worse on the other
    assert gate_decision((-0.2, -0.1), (-0.1, 0.1), 100) == OFF
    assert gate_decision((0.1, 0.2), None, 10, min_days=30) is None     # too few days: caller falls back


def synthetic_preds(seed=0):
    """Zone 0: correction removes most error. Zone 1: correction adds error. Zone 2: correction = raw."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in (2019, 2020, 2021):
        for d in range(90):
            valid = pd.Timestamp(f"{season}-06-01") + pd.Timedelta(days=d)
            n = 60
            zone = np.repeat([0, 1, 2], n // 3)
            o = rng.gamma(0.9, 15, n)
            raw = o + rng.normal(0, 12, n)
            corr = np.where(zone == 0, o + rng.normal(0, 3, n), np.where(zone == 1, o + rng.normal(0, 25, n), raw))
            P = np.zeros((n, 4)); P[:, 3] = 1.0
            P[:5, 3], P[:5, 2] = 0.0, 1.0                     # a thin "depression" regime in zone 0
            rows.append(pd.DataFrame({"zone": zone, "geo": 0, "lead": 1, "valid": valid, "season": season,
                                      "o_rain": o, "raw": raw.clip(0), "B": corr.clip(0),
                                      **{c: P[:, k] for k, c in enumerate(REGIME_PROB_FEATURES)}}))
    return pd.concat(rows, ignore_index=True)


def test_skill_gate_finds_on_off_and_neutral_units():
    cfg = deep_merge(load_config(), {"gate": {"min_days": 30}})
    g = skill_gate(synthetic_preds(), cfg, "B", B=200, log=lambda *_: None)
    every = g[g.regime == "all"].set_index("zone")
    assert every.loc[0, "gate"] == ON and every.loc[0, "ss_lo"] > 0
    assert every.loc[1, "gate"] == OFF and every.loc[1, "ss_hi"] < 0
    assert every.loc[2, "gate"] == NEUTRAL and abs(every.loc[2, "ss"]) < 1e-9
    assert every.loc[0, "season_wins"] == 3 and every.loc[1, "season_wins"] == 0
    dep = g[(g.zone == 0) & (g.regime == "depression")].iloc[0]
    assert dep.gate_source == "own" or dep.gate == every.loc[0, "gate"]
    s = gate_summary(g, "B")
    assert s["counts_all_regimes"] == {"ON": 1, "NEUTRAL": 1, "OFF": 1}
    assert any(o["unit"].startswith("central") for o in s["off"])        # zone 1 = central


def test_thin_regime_inherits_the_unit_gate():
    cfg = deep_merge(load_config(), {"gate": {"min_days": 400}})       # no regime slice reaches 400 days
    g = skill_gate(synthetic_preds(), cfg, "B", B=50, log=lambda *_: None)
    assert (g.gate == NEUTRAL).all() and set(g.gate_source) == {"insufficient"}
    cfg = deep_merge(load_config(), {"gate": {"min_days": 100}})       # 'all' = 270 days, depression = 270 too
    g = skill_gate(synthetic_preds(), cfg, "B", B=50, log=lambda *_: None)
    assert set(g[g.regime == "all"].gate_source) == {"own"}

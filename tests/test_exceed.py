"""Heavy-rain probability model M4 and its cross-fitted inputs (MODEL_SPEC 9.5, 11.7, 12)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")

from regimerain.config import deep_merge, load_config
from regimerain.exceed.metrics import exceed_scores, reliability
from regimerain.exceed.model import fit_exceed, negative_sample, prob_column
from regimerain.features.schema import BASE_FEATURES, EXCEED_FEATURES, REGIME_PROB_FEATURES

NY, NX = 12, 12


def small_cfg(**exceed):
    cfg = load_config()
    return deep_merge(cfg, {
        "num_threads": 2,
        "classifier": {"params": {"num_boost_round": 40, "min_data_in_leaf": 20}, "cell_stride": {"fast": 1, "full": 1},
                       "crossfit_groups": {"fast": 3, "full": 3}},
        "qm": {"n0_days": 5, "min_exceedances": 20},
        "exceed": {"params": {"min_data_in_leaf": 20, "num_boost_round": 300, "early_stopping_rounds": 30},
                   "min_events": 5, "neg_fraction": {"fast": 0.5, "full": 0.5}, **exceed},
    })


def table(seasons, days=40, seed=0):
    """Cell-day rows with every model column. Heavy rain follows forecast rain and the low-level jet."""
    rng = np.random.default_rng(seed)
    iy, ix = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    parts = []
    for s in seasons:
        for d in range(days):
            n = NY * NX
            row = {c: rng.normal(0, 1, n).astype("float32") for c in BASE_FEATURES}
            jet = rng.normal(0, 1)                                        # one value per day: a "regime"
            f = rng.gamma(0.8, 14, n) * (1.6 if jet > 0 else 0.6)
            row.update(f_rain=f.astype("float32"), f_llj_index=np.full(n, jet, "float32"),
                       geo=rng.integers(0, 3, n).astype("int8"), zone=rng.integers(0, 4, n).astype("int8"),
                       mjo_phase=rng.integers(0, 9, n).astype("int8"), lead=np.ones(n, "int16"))
            o = f * rng.lognormal(0, 0.35, n) * (1.3 if jet > 0 else 0.8)
            valid = pd.Timestamp(f"{s}-06-01") + pd.Timedelta(days=d)
            parts.append(pd.DataFrame({**row, "o_rain": o.astype("float32"), "season": np.int16(s),
                                       "y_synoptic": np.full(n, 0 if jet > 0.5 else 1 if jet < -0.5 else 3, "int8"),
                                       "init": valid, "valid": valid, "lat_idx": iy.ravel(), "lon_idx": ix.ravel()}))
    return pd.concat(parts, ignore_index=True)


def with_inputs(df):
    """Stand-in p_* and qm_rain (the backtest supplies cross-fitted ones)."""
    out = df.copy()
    out["qm_rain"] = df.f_rain * 1.1
    p = np.full((len(df), 4), 0.25, "float32")
    for k, c in enumerate(REGIME_PROB_FEATURES):
        out[c] = p[:, k]
    return out


def test_negative_sample_keeps_every_event_and_the_weighted_rate():
    rng = np.random.default_rng(1)
    event = rng.random(200_000) < 0.01
    idx, w = negative_sample(event, 0.25, seed=3)
    assert event[idx].sum() == event.sum()                               # no event row dropped
    assert (w[event[idx]] == 1).all() and np.allclose(w[~event[idx]], 4.0)
    weighted_rate = (w * event[idx]).sum() / w.sum()
    assert abs(weighted_rate - event.mean()) < 0.001
    idx_all, w_all = negative_sample(event, 1.0, seed=3)
    assert len(idx_all) == len(event) and (w_all == 1).all()
    with pytest.raises(ValueError):
        negative_sample(event, 0.0, seed=3)


def test_model_learns_is_monotone_and_calibrated_on_val_only():
    cfg = small_cfg()
    fit, val, test = (with_inputs(table([s], seed=s)) for s in (2019, 2020, 2021))
    model = fit_exceed(fit, val, cfg, log=lambda *_: None)
    assert model.calibrated_on == [2020]                                # inner_val only, never the test season
    assert model.features == EXCEED_FEATURES
    P = model.predict(test)
    assert P.shape == (len(test), 2) and P.min() >= 0 and P.max() <= 1
    assert (P[:, 1] <= P[:, 0] + 1e-7).all()                             # P(>=115.6) <= P(>=64.5)
    y = test.o_rain.to_numpy() >= 64.5
    s = exceed_scores(y, P[:, 0], {"likely": 0.5})
    assert s["n_events"] > 50
    assert s["roc_auc"] > 0.85 and s["bss"] > 0.1                        # real skill on an unseen season
    assert abs(s["mean_forecast"] - s["event_rate"]) < 0.4 * s["event_rate"] + 0.005   # calibrated on average


def test_too_few_events_falls_back_to_the_base_rate():
    cfg = small_cfg(thresholds=[64.5, 5000.0])
    fit, val, test = (with_inputs(table([s], days=10, seed=s)) for s in (2019, 2020, 2021))
    model = fit_exceed(fit, val, cfg, log=lambda *_: None)
    m = model.models[1]
    assert m.booster is None and m.base_rate == 0.0
    assert np.all(model.predict(test)[:, 1] == 0.0)
    assert model.diagnostics()["thresholds"][1]["constant_base_rate"] is True


def test_truth_columns_are_refused_as_features():
    from regimerain.features.schema import LeakageError
    cfg = small_cfg()
    fit, val = with_inputs(table([2019], days=5)), with_inputs(table([2020], days=5))
    with pytest.raises(LeakageError):
        fit_exceed(fit, val, cfg, features=EXCEED_FEATURES + ["o_rain"], log=lambda *_: None)


def test_scores_on_known_forecasts():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0], bool)
    perfect = exceed_scores(y, y.astype(float), {"likely": 0.5})
    assert perfect["brier"] == 0 and perfect["bss"] == 1 and perfect["roc_auc"] == 1
    assert perfect["levels"]["likely"]["pod"] == 1 and perfect["levels"]["likely"]["far"] == 0
    clim = exceed_scores(y, np.full(10, 0.2), {"likely": 0.5})
    assert abs(clim["bss"]) < 1e-12 and clim["levels"]["likely"]["pod"] == 0
    none = exceed_scores(np.zeros(5, bool), np.zeros(5), {"likely": 0.5})
    assert np.isnan(none["roc_auc"]) and np.isnan(none["bss"])


def test_reliability_bins_cover_every_row():
    rng = np.random.default_rng(0)
    p = rng.random(1000)
    bins = reliability(rng.random(1000) < p, p)
    assert len(bins) == 10 and sum(b["n"] for b in bins) == 1000
    assert bins[-1]["mean_forecast"] > 0.9 and bins[0]["mean_forecast"] < 0.1


def test_prob_column_names():
    assert prob_column(64.5) == "p_exc_64.5" and prob_column(115.6) == "p_exc_115.6"


def test_crossfit_never_uses_the_rows_it_predicts():
    """Out-of-fold: changing one season's labels and truth must not change that season's own p_* or qm_rain."""
    from regimerain.classify.crossfit import oof_features
    cfg = small_cfg()
    tr = table([2016, 2017, 2018], days=30, seed=4)
    P1, q1 = oof_features(tr, cfg, (NY, NX), log=lambda *_: None)
    assert P1.shape == (len(tr), 4) and np.allclose(P1.sum(axis=1), 1, atol=1e-5)
    tampered = tr.copy()
    m = tampered.season.to_numpy() == 2017
    tampered.loc[m, "y_synoptic"] = (tampered.loc[m, "y_synoptic"] + 1) % 4
    tampered.loc[m, "o_rain"] = tampered.loc[m, "o_rain"] * 3
    P2, q2 = oof_features(tampered, cfg, (NY, NX), log=lambda *_: None)
    assert np.array_equal(P1[m], P2[m]) and np.array_equal(q1[m], q2[m])     # 2017 rows saw no 2017 data
    assert not np.array_equal(P1[~m], P2[~m])                                 # other seasons did learn from 2017

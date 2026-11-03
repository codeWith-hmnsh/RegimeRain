import numpy as np
import pandas as pd
import pytest

from regimerain.correct.experts import ExpertSet
from regimerain.correct.mixture import correct_mixture, mixture_weights
from regimerain.correct.qm import QMExpert


def rain(n, shape, scale, p_dry, seed):
    rng = np.random.default_rng(seed)
    x = rng.gamma(shape, scale, n)
    x[rng.random(n) < p_dry] = 0.0
    return x


def test_qm_identity_on_same_distribution():
    o = rain(20000, 0.6, 25, 0.4, 0)
    e = QMExpert.fit(o, o, n_days=100)
    probe = np.quantile(o[o > 0], [0.1, 0.5, 0.9, 0.99])
    assert np.allclose(e.transform(probe), probe, rtol=0.01)


def test_qm_monotone():
    e = QMExpert.fit(rain(20000, 0.5, 20, 0.2, 1), rain(20000, 0.7, 30, 0.45, 2), n_days=100)
    x = np.linspace(0, 600, 3001)
    assert np.all(np.diff(e.transform(x)) >= -1e-9)


def test_qm_removes_drizzle():
    f = rain(20000, 0.8, 5, 0.0, 3)          # model never dry
    o = rain(20000, 0.6, 25, 0.5, 4)         # half the days dry
    e = QMExpert.fit(f, o, n_days=100)
    small = np.quantile(f, 0.3)
    assert e.transform([small])[0] == 0.0 and e.transform([np.quantile(f, 0.9)])[0] > 0


def test_gpd_tail_extends_beyond_training_max():
    f = rain(20000, 0.5, 20, 0.2, 5); o = rain(20000, 0.6, 30, 0.3, 6)
    e = QMExpert.fit(f, o, n_days=100)
    assert e.has_tail and 0.0 <= e.o_xi <= 0.4
    assert e.transform([10 * f.max()])[0] > o.max()


def test_without_tail_output_is_capped_at_training_max():
    f = rain(5000, 0.5, 20, 0.2, 7); o = rain(5000, 0.6, 30, 0.3, 8)
    e = QMExpert.fit(f, o, n_days=10, min_exceedances=10**9)
    assert not e.has_tail
    assert e.transform([10 * f.max()])[0] == pytest.approx(o.max(), rel=1e-3)   # q clipped at 1 - 1e-6


def test_record_roundtrip():
    e = QMExpert.fit(rain(5000, 0.5, 20, 0.2, 9), rain(5000, 0.6, 30, 0.3, 10), n_days=40)
    e2 = QMExpert.from_record(e.to_record())
    x = np.linspace(0, 300, 50)
    assert np.allclose(e.transform(x), e2.transform(x), rtol=1e-5, atol=1e-4)


def make_df(seed=0):
    """Two regimes on one geo/zone/lead: regime 0 model too dry (x2 needed), regime 1 too wet (x0.5)."""
    rng = np.random.default_rng(seed)
    rows = []
    for r, factor, n_days in ((0, 2.0, 60), (1, 0.5, 5)):
        for d in range(n_days):
            f = rng.gamma(0.8, 15, 200)
            rows.append(pd.DataFrame({"f_rain": f, "o_rain": f * factor, "geo": 0, "zone": 1, "lead": 1,
                                      "valid": pd.Timestamp("2019-06-01") + pd.Timedelta(days=d + 100 * r),
                                      "y": r}))
    return pd.concat(rows, ignore_index=True)


def test_shrinkage_weight_uses_days():
    df = make_df()
    es = ExpertSet(n0_days=30, min_exceedances=10**9).fit(df, "y")
    assert es.weight((0, 0, 1, 1)) == pytest.approx(60 / 90)
    assert es.weight((1, 0, 1, 1)) == pytest.approx(5 / 35)


def test_regime_experts_learn_opposite_corrections():
    df = make_df()
    es = ExpertSet(n0_days=1, min_exceedances=10**9).fit(df, "y")
    x = np.array([20.0])
    assert es.apply((0, 0, 1, 1), x)[0] > 30      # dry-biased regime corrected upward
    assert es.apply((1, 0, 1, 1), x)[0] < 15      # wet-biased regime corrected downward


def test_unfitted_key_falls_back_to_parent():
    df = make_df()
    es = ExpertSet(n0_days=30).fit(df, "y")
    x = np.array([5.0, 20.0, 50.0])
    assert np.allclose(es.apply((3, 0, 1, 1), x), es.apply((-1, 0, 1, 1), x))


def test_thin_regime_is_pulled_toward_parent():
    df = make_df()
    strong = ExpertSet(n0_days=0.001, min_exceedances=10**9).fit(df, "y")
    shrunk = ExpertSet(n0_days=300, min_exceedances=10**9).fit(df, "y")
    x = np.array([20.0])
    parent = shrunk.apply((-1, 0, 1, 1), x)[0]
    assert abs(shrunk.apply((1, 0, 1, 1), x)[0] - parent) < abs(strong.apply((1, 0, 1, 1), x)[0] - parent)


def test_mixture_weights_rules():
    W = mixture_weights(np.array([[0.6, 0.35, 0.05, 0.0], [0.85, 0.1, 0.05, 0.0], [0.25, 0.25, 0.25, 0.25]]))
    assert np.allclose(W[0], [0.6 / 0.95, 0.35 / 0.95, 0, 0])
    assert np.allclose(W[1], [1, 0, 0, 0])                    # dominant regime -> hard
    assert np.allclose(W[2], [0.25] * 4)
    assert np.allclose(W.sum(axis=1), 1)


def test_mixture_equals_single_expert_for_one_hot():
    df = make_df()
    es = ExpertSet(n0_days=30).fit(df, "y")
    sub = df.head(50).reset_index(drop=True)
    P = np.tile([0, 1.0, 0, 0], (len(sub), 1))
    assert np.allclose(correct_mixture(sub, P, es), es.apply((1, 0, 1, 1), sub.f_rain.values), atol=1e-4)


def test_global_variant_and_sample_counts():
    df = make_df()
    es = ExpertSet(n0_days=30).fit(df, None)
    out = es.apply_global(df.head(10))
    assert out.shape == (10,) and np.all(out >= 0)
    counts = es.sample_counts()
    assert set(counts.regime) == {-1} and {"n", "n_days", "w_shrink"} <= set(counts.columns)
    back = ExpertSet.from_frame(es.to_frame("A"), n0_days=30)
    assert np.allclose(back.apply_global(df.head(10)), out, atol=1e-3)

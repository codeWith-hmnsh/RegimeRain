import math

import numpy as np
import pandas as pd
import pytest

from regimerain.verify.bootstrap import block_bootstrap_delta
from regimerain.verify.metrics import categorical_scores, contingency, fss, rmse, scores_from_stats
from regimerain.verify.stats import cell_slice_stats, fss_day_stats, fss_parts


def test_toy_contingency_table():
    s = categorical_scores(H=10, M=5, FA=5, CN=80)
    assert s["pod"] == pytest.approx(10 / 15)
    assert s["far"] == pytest.approx(5 / 15)
    assert s["csi"] == pytest.approx(0.5)
    hr = 15 * 15 / 100
    assert s["ets"] == pytest.approx((10 - hr) / (20 - hr))
    assert s["freq_bias"] == pytest.approx(1.0)


def test_perfect_and_empty_tables():
    assert categorical_scores(5, 0, 0, 95)["ets"] == pytest.approx(1.0)
    assert math.isnan(categorical_scores(0, 0, 0, 100)["pod"])


def test_contingency_counts():
    f = np.array([0, 70, 70, 10]); o = np.array([0, 80, 10, 90])
    assert contingency(f, o, 64.5) == (1, 1, 1, 1)


def test_rmse_and_scores_from_stats():
    f = np.array([[1.0, 2.0], [3.0, 70.0]]); o = np.array([[1.0, 4.0], [3.0, 60.0]])
    rows = cell_slice_stats(f, o, np.ones_like(f, bool))
    assert len(rows) == 1 and rows[0]["n"] == 4
    s = scores_from_stats(rows[0], 64.5)
    assert s["rmse"] == pytest.approx(math.sqrt((4 + 100) / 4))
    assert math.isnan(s["pod"])                  # obs 60 < 64.5: no observed event, POD undefined
    assert s["far"] == pytest.approx(1.0)       # forecast 70 is a false alarm
    assert rmse(0, 0) != rmse(0, 0)             # nan


def test_slices_by_regime_and_geo():
    f = np.zeros((2, 2)); o = np.zeros((2, 2))
    syn = np.array([[0, 0], [2, 3]]); geo = np.array([[0, 1], [2, 2]])
    rows = cell_slice_stats(f, o, np.ones((2, 2), bool), syn, geo)
    by = {(r["synoptic"], r["geo"]): r["n"] for r in rows}
    assert by[("all", "all")] == 4 and by[(0, "all")] == 2 and by[(2, 2)] == 1
    assert (1, "all") not in by                  # empty slices are skipped


def test_nan_truth_cells_are_excluded():
    o = np.array([[1.0, np.nan]]); f = np.array([[1.0, 99.0]])
    assert cell_slice_stats(f, o, np.ones((1, 2), bool))[0]["n"] == 1


def test_fss_identical_is_one():
    rng = np.random.default_rng(0)
    f = rng.gamma(0.5, 30, (40, 40))
    num, den = fss_parts(f, f, np.ones_like(f, bool), 64.5, 3)
    assert fss(num, den) == pytest.approx(1.0)


def test_fss_disjoint_at_window_one_is_zero():
    f = np.zeros((20, 20)); o = np.zeros((20, 20))
    f[2, 2] = 100; o[15, 15] = 100
    assert fss(*fss_parts(f, o, np.ones_like(f, bool), 64.5, 1)) == pytest.approx(0.0)


def test_fss_rises_with_window_for_shifted_blob():
    f = np.zeros((40, 40)); o = np.zeros((40, 40))
    f[10:15, 10:15] = 100; o[12:17, 13:18] = 100
    land = np.ones_like(f, bool)
    vals = [fss(*fss_parts(f, o, land, 64.5, w)) for w in (1, 3, 5, 9)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])) and vals[-1] > vals[0]


def test_fss_day_stats_keys():
    row = fss_day_stats(np.zeros((5, 5)), np.zeros((5, 5)), np.ones((5, 5), bool), thresholds=(64.5,), windows=(1, 3))
    assert set(row) == {"fssnum_64.5_1", "fssden_64.5_1", "fssnum_64.5_3", "fssden_64.5_3"}


def test_bootstrap_detects_real_improvement():
    rng = np.random.default_rng(1)
    days = 200
    raw = pd.DataFrame({"sse": rng.uniform(90, 110, days), "n": 100})
    corr = pd.DataFrame({"sse": raw.sse * 0.8, "n": 100})
    stat = lambda s: math.sqrt(s["sse"] / s["n"])
    res = block_bootstrap_delta(raw, corr, stat, B=200, block=5, seed=0)
    assert res["delta"] < 0 and res["significant"] and res["ci"][1] < 0


def test_bootstrap_no_difference_not_significant():
    rng = np.random.default_rng(2)
    raw = pd.DataFrame({"sse": rng.uniform(90, 110, 150), "n": 100})
    res = block_bootstrap_delta(raw, raw.copy(), lambda s: s["sse"] / s["n"], B=200, seed=0)
    assert res["delta"] == 0 and not res["significant"]

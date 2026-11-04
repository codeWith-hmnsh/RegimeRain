"""Explanations (MODEL_SPEC 15): TreeSHAP reasons for the heavy-rain probability."""
import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")

from regimerain.explain.explain import phrase, top_reasons  # noqa: E402


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({"f_imfc": rng.normal(0, 10, n), "p_depression": rng.uniform(0, 1, n),
                      "lat": rng.uniform(8, 30, n), "f_pw": rng.normal(50, 5, n)})
    logit = 0.3 * X.f_imfc + 4 * (X.p_depression - 0.5) + 0.4 * (X.lat - 19)       # lat matters but is silent
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)
    b = lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 15, "seed": 0}, lgb.Dataset(X, y), 80)
    return b, X


def test_contributions_add_up_to_the_raw_score(model):
    b, X = model
    c = b.predict(X.iloc[:20], pred_contrib=True)
    assert np.allclose(c.sum(axis=1), b.predict(X.iloc[:20], raw_score=True), atol=1e-6)


def test_top_reasons_name_the_driving_features(model):
    b, _ = model
    row = pd.DataFrame({"f_imfc": [35.0], "p_depression": [0.95], "lat": [29.0], "f_pw": [50.0]})
    r = top_reasons(b, row, k=3)[0]
    feats = [x["feature"] for x in r]
    assert feats[:2] == ["f_imfc", "p_depression"] or set(feats[:2]) == {"f_imfc", "p_depression"}
    assert "lat" not in feats                                       # position never becomes a "reason"
    assert all(x["log_odds"] > 0 for x in r)
    assert any("moisture convergence 35 mm/day" == x["text"] for x in r)
    assert any(x["text"] == "depression/low regime (p=0.95)" for x in r)


def test_no_positive_push_means_no_reasons(model):
    b, _ = model
    row = pd.DataFrame({"f_imfc": [-30.0], "p_depression": [0.0], "lat": [8.0], "f_pw": [50.0]})
    assert all(x["log_odds"] > 0 for x in top_reasons(b, row)[0])
    assert top_reasons(None, row) == [[]]


def test_phrase_unknown_and_valueless():
    assert phrase("windward", 1.0) == "windward slope (orographic lift)"
    assert phrase("some_new_feature", 2.0) == "some new feature"
    assert phrase("mjo_phase", 4) == "MJO phase 4"

"""Model 4: probability that rain reaches an IMD threshold (MODEL_SPEC section 12).

One LightGBM binary model per threshold (64.5 and 115.6 mm/day). Early stopping and isotonic calibration
use the inner-validation season only, never the test season. After calibration P(>= 115.6) <= P(>= 64.5)
is enforced, so a higher threshold is never more likely than a lower one.

Features are EXCEED_FEATURES: the regime classifier's inputs plus the regime-aware corrected rain
(`qm_rain`) and the regime probabilities (`p_*`). On training rows those two come from cross-fitting
(regimerain.classify.crossfit), so they behave like test-time values.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from regimerain.config import by_mode, lgb_params
from regimerain.features.schema import CATEGORICAL, EXCEED_FEATURES, assert_no_leak
from regimerain.verify.metrics import fmt_threshold

DEFAULT_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=1000,
                      feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                      num_boost_round=3000, early_stopping_rounds=100)


def prob_column(threshold: float) -> str:
    """Column name of P(rain >= threshold) in predictions.parquet, e.g. p_exc_64.5."""
    return f"p_exc_{fmt_threshold(threshold)}"


def negative_sample(event: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Row indices and weights: every event row, plus a random `frac` of the others weighted 1/frac.

    Heavy rain is rare (about 1% of cell-days), so most rows carry little information. Dropping part of the
    dry rows cuts training time; the 1/frac weight keeps the weighted event rate equal to the true one.
    """
    event = np.asarray(event, bool)
    if frac >= 1.0:
        return np.arange(len(event)), np.ones(len(event))
    if not 0.0 < frac < 1.0:
        raise ValueError(f"neg_fraction must be in (0, 1], got {frac}")
    keep = event | (np.random.default_rng(seed).random(len(event)) < frac)
    idx = np.flatnonzero(keep)
    return idx, np.where(event[idx], 1.0, 1.0 / frac)


@dataclass
class ThresholdModel:
    threshold: float
    booster: object | None          # None: too few events to learn from -> constant base rate
    iso: object | None
    base_rate: float
    n_events_fit: int
    n_events_val: int
    best_iteration: int = 0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            return np.full(len(X), self.base_rate)
        raw = self.booster.predict(X, num_iteration=self.best_iteration or None)
        return self.iso.predict(raw)

    def top_features(self, n: int = 8) -> list[tuple[str, float]]:
        if self.booster is None:
            return []
        gain = self.booster.feature_importance(importance_type="gain")
        order = np.argsort(gain)[::-1][:n]
        total = gain.sum() or 1.0
        return [(self.booster.feature_name()[i], round(float(gain[i] / total), 4)) for i in order]


@dataclass
class ExceedModel:
    features: list[str]
    models: list[ThresholdModel]                     # sorted by threshold
    calibrated_on: list[int] = field(default_factory=list)

    @property
    def thresholds(self) -> list[float]:
        return [m.threshold for m in self.models]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """(n, n_thresholds) calibrated probabilities, non-increasing along thresholds."""
        X = df[self.features]
        P = np.column_stack([m.predict(X) for m in self.models]).clip(0.0, 1.0)
        return np.minimum.accumulate(P, axis=1).astype("float32")

    def diagnostics(self) -> dict:
        return {"features": self.features, "calibrated_on": self.calibrated_on,
                "thresholds": [{"threshold": m.threshold, "constant_base_rate": m.booster is None,
                                "base_rate": m.base_rate, "n_events_fit": m.n_events_fit,
                                "n_events_val": m.n_events_val, "best_iteration": m.best_iteration,
                                "top_features_gain": m.top_features()} for m in self.models]}


def _train_one(fit: pd.DataFrame, val: pd.DataFrame, threshold: float, idx: np.ndarray, w: np.ndarray,
               cfg: dict, features: list[str]) -> ThresholdModel:
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    ec = cfg["exceed"]
    yf = fit.o_rain.to_numpy() >= threshold
    yv = val.o_rain.to_numpy() >= threshold
    base = float((yf.sum() + yv.sum()) / max(len(yf) + len(yv), 1))
    n_f, n_v = int(yf.sum()), int(yv.sum())
    if min(n_f, n_v) < int(ec.get("min_events", 20)):
        return ThresholdModel(threshold, None, None, base, n_f, n_v)
    params = lgb_params(cfg, {**DEFAULT_PARAMS, **ec.get("params", {})})
    rounds = int(params.pop("num_boost_round"))
    stop = int(params.pop("early_stopping_rounds"))
    dtr = lgb.Dataset(fit[features].iloc[idx], yf[idx].astype(int), weight=w,
                      categorical_feature=CATEGORICAL, free_raw_data=False)
    dva = lgb.Dataset(val[features], yv.astype(int), categorical_feature=CATEGORICAL, reference=dtr,
                      free_raw_data=False)
    booster = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(stop, verbose=False)])
    assert_no_leak(booster.feature_name())
    best = int(booster.best_iteration or 0)
    raw_val = booster.predict(val[features], num_iteration=best or None)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_val, yv.astype(float))
    return ThresholdModel(threshold, booster, iso, base, n_f, n_v, best)


def fit_exceed(fit: pd.DataFrame, val: pd.DataFrame, cfg: dict, features=EXCEED_FEATURES,
               log=print) -> ExceedModel:
    """Train one calibrated model per threshold on `fit`; early stopping and calibration on `val` only."""
    features = list(features)
    assert_no_leak(features)
    missing = [c for c in features if c not in fit.columns or c not in val.columns]
    if missing:
        raise KeyError(f"exceedance features missing from the table: {missing}")
    ec = cfg["exceed"]
    thresholds = sorted(float(t) for t in ec["thresholds"])
    frac = float(by_mode(cfg, ec.get("neg_fraction", 1.0)))
    idx, w = negative_sample(fit.o_rain.to_numpy() >= thresholds[0], frac, int(cfg["seed"]))
    log(f"exceed: {len(idx):,} of {len(fit):,} fit rows (neg_fraction {frac}), val {len(val):,} rows")
    models = []
    for t in thresholds:
        m = _train_one(fit, val, t, idx, w, cfg, features)
        state = ("constant base rate (too few events)" if m.booster is None
                 else f"best iteration {m.best_iteration}")
        log(f"exceed >= {t:g} mm: {m.n_events_fit} fit / {m.n_events_val} val events, {state}")
        models.append(m)
    return ExceedModel(features, models, sorted(int(s) for s in val.season.unique()))

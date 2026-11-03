"""Model 1: synoptic regime classifier (MODEL_SPEC section 9)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from regimerain.config import lgb_params
from regimerain.features.schema import BASE_FEATURES, CATEGORICAL, SYNOPTIC, assert_no_leak

K = len(SYNOPTIC)


def class_weights(y: np.ndarray, k: int = K) -> np.ndarray:
    counts = np.bincount(y, minlength=k).astype(float)
    w = len(y) / (k * np.maximum(counts, 1))
    return w[y]


def train_classifier(fit: pd.DataFrame, val: pd.DataFrame, cfg: dict, features=BASE_FEATURES):
    import lightgbm as lgb
    assert_no_leak(features)
    params = lgb_params(cfg, cfg["classifier"]["params"])
    rounds = int(params.pop("num_boost_round", 2000))
    dtr = lgb.Dataset(fit[features], fit.y_synoptic.to_numpy(), weight=class_weights(fit.y_synoptic.to_numpy()),
                      categorical_feature=CATEGORICAL, free_raw_data=False)
    dva = lgb.Dataset(val[features], val.y_synoptic.to_numpy(), weight=class_weights(val.y_synoptic.to_numpy()),
                      categorical_feature=CATEGORICAL, reference=dtr, free_raw_data=False)
    booster = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    assert_no_leak(booster.feature_name())
    return booster


def raw_logits(booster, df: pd.DataFrame, features=BASE_FEATURES) -> np.ndarray:
    z = booster.predict(df[features], raw_score=True, num_iteration=booster.best_iteration or None)
    return np.asarray(z).reshape(len(df), -1)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(T):
        return -log_softmax(logits / T, axis=1)[np.arange(len(y)), y].mean()
    return float(minimize_scalar(nll, bounds=(0.25, 10.0), method="bounded").x)


def predict_proba(booster, df: pd.DataFrame, T: float, features=BASE_FEATURES) -> np.ndarray:
    return softmax(raw_logits(booster, df, features) / T, axis=1)


def smooth_on_grid(df: pd.DataFrame, P: np.ndarray, shape: tuple[int, int], size: int = 3) -> np.ndarray:
    """Masked size x size spatial mean of probabilities per (init, lead) map, renormalised (MODEL_SPEC 9.4)."""
    if size <= 1:
        return P
    out = np.empty_like(P)
    for _, idx in df.groupby(["init", "lead"], observed=True).indices.items():
        iy, ix = df.lat_idx.to_numpy()[idx], df.lon_idx.to_numpy()[idx]
        m = np.zeros(shape); m[iy, ix] = 1.0
        den = uniform_filter(m, size, mode="constant")
        for k in range(P.shape[1]):
            g = np.zeros(shape); g[iy, ix] = P[idx, k]
            out[idx, k] = (uniform_filter(g, size, mode="constant") / np.maximum(den, 1e-12))[iy, ix]
    out /= out.sum(axis=1, keepdims=True)
    return out


def classifier_metrics(y: np.ndarray, P: np.ndarray) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score
    pred = P.argmax(axis=1)
    onehot = np.eye(K)[y]
    recall = {}
    for k, name in enumerate(SYNOPTIC):
        m = y == k
        recall[name] = float((pred[m] == k).mean()) if m.any() else None
    return {"accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, labels=list(range(K)), average="macro", zero_division=0)),
            "brier": float(((P - onehot) ** 2).sum(axis=1).mean()),
            "per_class_recall": recall,
            "confusion": confusion_matrix(y, pred, labels=list(range(K))).tolist(),
            "class_counts": {n: int((y == k).sum()) for k, n in enumerate(SYNOPTIC)}}

"""Scores for the heavy-rain probability product (MODEL_SPEC 12.3, 12.4).

- Brier score, and its skill against the sample climatology of the same rows (BSS > 0: better than always
  forecasting the event frequency).
- ROC-AUC (ranking) and PR-AUC (average precision; the honest one for rare events).
- Reliability: 10 probability bins with forecast mean, observed frequency and count.
- Alert levels: P >= level counts as "yes", scored with POD / FAR / CSI / ETS like a deterministic forecast.
"""
from __future__ import annotations

import numpy as np

from regimerain.verify.metrics import categorical_scores

NAN = float("nan")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2)) if len(y) else NAN


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = []
    for k in range(n_bins):
        m = b == k
        out.append({"bin": [round(edges[k], 2), round(edges[k + 1], 2)], "n": int(m.sum()),
                    "mean_forecast": float(p[m].mean()) if m.any() else None,
                    "observed_freq": float(y[m].mean()) if m.any() else None})
    return out


def exceed_scores(y: np.ndarray, p: np.ndarray, levels: dict[str, float]) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y, bool)
    p = np.asarray(p, float)
    n, n_ev = len(y), int(y.sum())
    rate = n_ev / n if n else NAN
    bs = brier(y, p)
    bs_ref = rate * (1 - rate) if n else NAN
    both = 0 < n_ev < n
    out = {"n": n, "n_events": n_ev, "event_rate": rate, "mean_forecast": float(p.mean()) if n else NAN,
           "brier": bs, "brier_climatology": bs_ref,
           "bss": (1 - bs / bs_ref) if both and bs_ref > 0 else NAN,
           "roc_auc": float(roc_auc_score(y, p)) if both else NAN,
           "pr_auc": float(average_precision_score(y, p)) if both else NAN,
           "reliability": reliability(y, p), "levels": {}}
    for name, lv in levels.items():
        yes = p >= lv
        H, M = int((yes & y).sum()), int((~yes & y).sum())
        FA, CN = int((yes & ~y).sum()), int((~yes & ~y).sum())
        out["levels"][name] = {"level": float(lv), **categorical_scores(H, M, FA, CN)}
    return out

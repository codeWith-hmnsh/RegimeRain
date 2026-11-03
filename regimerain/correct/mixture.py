"""Soft-regime correction: probability-weighted mixture of experts (MODEL_SPEC section 11.5)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from regimerain.correct.experts import ExpertSet


def mixture_weights(P: np.ndarray, p_min: float = 0.1, p_hard: float = 0.8) -> np.ndarray:
    """Hard-assign rows with a dominant regime, drop small weights, renormalise."""
    W = np.asarray(P, "float64").copy()
    if W.ndim != 2:
        raise ValueError("P must be (n, K)")
    hard = W.max(axis=1) >= p_hard
    if hard.any():
        W[hard] = (W[hard] == W[hard].max(axis=1, keepdims=True)).astype("float64")
    W[W < p_min] = 0.0
    tot = W.sum(axis=1, keepdims=True)
    empty = tot[:, 0] == 0                    # only possible if K > 1/p_min; fall back to the argmax
    if empty.any():
        W[empty] = 0.0
        W[empty, np.argmax(P[empty], axis=1)] = 1.0
        tot = W.sum(axis=1, keepdims=True)
    return W / tot


def correct_mixture(df: pd.DataFrame, P: np.ndarray, experts: ExpertSet,
                    p_min: float = 0.1, p_hard: float = 0.8) -> np.ndarray:
    """df: rows with f_rain, geo, zone, lead. P: (n, K) regime probabilities aligned with df."""
    if len(df) != len(P):
        raise ValueError("df and P must have the same length")
    W = mixture_weights(P, p_min, p_hard)
    out = np.zeros(len(df))
    x_all = df["f_rain"].to_numpy("float64")
    for (g, z, L), idx in df.groupby(["geo", "zone", "lead"], observed=True).indices.items():
        x = x_all[idx]
        for r in range(W.shape[1]):
            wr = W[idx, r]
            m = wr > 0
            if m.any():
                out[idx[m]] += wr[m] * experts.apply((r, int(g), int(z), int(L)), x[m])
    return out.astype("float32")

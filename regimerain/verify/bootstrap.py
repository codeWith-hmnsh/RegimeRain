"""Moving-block bootstrap over days for raw-vs-corrected differences (MODEL_SPEC section 13.3)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def block_indices(n_days: int, block: int, rng: np.random.Generator) -> np.ndarray:
    block = max(1, min(block, n_days))
    n_blocks = int(np.ceil(n_days / block))
    starts = rng.integers(0, n_days - block + 1, n_blocks)
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n_days]


def block_bootstrap_delta(day_raw: pd.DataFrame, day_corr: pd.DataFrame,
                          stat_fn: Callable[[pd.Series], float], B: int = 1000, block: int = 5,
                          seed: int = 0, ci: float = 0.90) -> dict:
    """Delta = stat(corrected) - stat(raw) with a two-sided `ci` interval.

    day_raw / day_corr: one row per day (same order, sorted by date) of summed statistics.
    """
    if len(day_raw) != len(day_corr):
        raise ValueError("raw and corrected must have the same days")
    raw = day_raw.reset_index(drop=True)
    corr = day_corr.reset_index(drop=True)
    delta = stat_fn(corr.sum()) - stat_fn(raw.sum())
    rng = np.random.default_rng(seed)
    deltas = np.empty(B)
    for b in range(B):
        idx = block_indices(len(raw), block, rng)
        deltas[b] = stat_fn(corr.iloc[idx].sum()) - stat_fn(raw.iloc[idx].sum())
    alpha = (1 - ci) / 2
    lo, hi = np.nanquantile(deltas, [alpha, 1 - alpha])
    return {"delta": float(delta), "ci": [float(lo), float(hi)],
            "significant": bool(lo > 0 or hi < 0)}

"""The PS's six named metrics, computed from summed sufficient statistics (PRD section 17.1).

Scores are always computed from sums over cell-days (never averaged per day), so pooling
across days, folds and bootstrap resamples is exact.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

NAN = float("nan")


def contingency(f: np.ndarray, o: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    """Hits, misses, false alarms, correct negatives for `value >= threshold`."""
    fy, oy = np.asarray(f) >= threshold, np.asarray(o) >= threshold
    return (int(np.sum(fy & oy)), int(np.sum(~fy & oy)), int(np.sum(fy & ~oy)), int(np.sum(~fy & ~oy)))


def _div(a: float, b: float) -> float:
    return a / b if b else NAN


def categorical_scores(H: float, M: float, FA: float, CN: float) -> dict[str, float]:
    n = H + M + FA + CN
    h_random = (H + M) * (H + FA) / n if n else NAN
    return {
        "pod": _div(H, H + M),
        "far": _div(FA, H + FA),
        "csi": _div(H, H + M + FA),
        "ets": _div(H - h_random, H + M + FA - h_random) if n else NAN,
        "freq_bias": _div(H + FA, H + M),
    }


def rmse(sse: float, n: float) -> float:
    return math.sqrt(sse / n) if n else NAN


def fss(num_sum: float, den_sum: float) -> float:
    """FSS = 1 - sum (Pf - Po)^2 / (sum Pf^2 + sum Po^2), pooled over days."""
    return 1.0 - num_sum / den_sum if den_sum > 0 else NAN


def scores_from_stats(stats: Mapping[str, float], threshold: float) -> dict[str, float]:
    """`stats` holds summed columns: n, sse, H_<t>, M_<t>, FA_<t>, CN_<t>."""
    t = fmt_threshold(threshold)
    out = categorical_scores(stats[f"H_{t}"], stats[f"M_{t}"], stats[f"FA_{t}"], stats[f"CN_{t}"])
    out["rmse"] = rmse(stats["sse"], stats["n"])
    return out


def fmt_threshold(t: float) -> str:
    """Stable column-name form of a threshold: 64.5 -> '64.5', 115.6 -> '115.6', 2.5 -> '2.5'."""
    return f"{float(t):g}"

"""Per-day sufficient statistics (MODEL_SPEC section 13.1).

Computed once per (variant, lead, day); any slice's metrics and bootstrap CIs are then sums of rows.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from regimerain.verify.metrics import contingency, fmt_threshold

THRESHOLDS = (2.5, 15.6, 64.5, 115.6)
WINDOWS = (1, 3, 5, 9)
ALL = "all"


def cell_slice_stats(f: np.ndarray, o: np.ndarray, land: np.ndarray, syn: np.ndarray | None = None,
                     geo: np.ndarray | None = None, thresholds=THRESHOLDS) -> list[dict]:
    """Stats for one day on (lat, lon) grids, sliced by cell synoptic label and geo class.

    Slices: every combination of synoptic in {all, 0..3} x geo in {all, 0..2} that has cells.
    """
    f = np.nan_to_num(np.asarray(f, "float64"))
    o = np.asarray(o, "float64")
    land = np.asarray(land, bool) & ~np.isnan(o)
    syn_values = [ALL] + ([] if syn is None else [0, 1, 2, 3])
    geo_values = [ALL] + ([] if geo is None else [0, 1, 2])
    rows = []
    for s in syn_values:
        for g in geo_values:
            m = land.copy()
            if s != ALL:
                m &= syn == s
            if g != ALL:
                m &= geo == g
            if not m.any():
                continue
            fm, om = f[m], o[m]
            row = {"synoptic": s, "geo": g, "n": int(m.sum()), "sse": float(np.sum((fm - om) ** 2))}
            for t in thresholds:
                k = fmt_threshold(t)
                row[f"H_{k}"], row[f"M_{k}"], row[f"FA_{k}"], row[f"CN_{k}"] = contingency(fm, om, t)
            rows.append(row)
    return rows


def fractions(binary: np.ndarray, mask: np.ndarray, window: int) -> np.ndarray:
    """Fraction of `binary` cells in an n x n window, counting only masked (valid) cells."""
    num = uniform_filter(binary * mask, size=window, mode="constant")
    den = uniform_filter(mask, size=window, mode="constant")
    return np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), 0.0)


def fss_parts(f: np.ndarray, o: np.ndarray, land: np.ndarray, threshold: float, window: int) -> tuple[float, float]:
    """(sum (Pf-Po)^2, sum Pf^2 + sum Po^2) over land cells for one day."""
    land = np.asarray(land, bool) & ~np.isnan(np.asarray(o, "float64"))
    m = land.astype("float64")
    pf = fractions((np.nan_to_num(f) >= threshold).astype("float64"), m, window)[land]
    po = fractions((np.nan_to_num(o) >= threshold).astype("float64"), m, window)[land]
    return float(np.sum((pf - po) ** 2)), float(np.sum(pf ** 2) + np.sum(po ** 2))


def fss_day_stats(f, o, land, thresholds=THRESHOLDS, windows=WINDOWS) -> dict:
    row = {}
    for t in thresholds:
        k = fmt_threshold(t)
        for w in windows:
            row[f"fssnum_{k}_{w}"], row[f"fssden_{k}_{w}"] = fss_parts(f, o, land, t, w)
    return row

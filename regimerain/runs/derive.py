"""Values derived from a run's fields for display: wettest point, depression track, place values.

Pure functions of `RunFields`. No new numbers are invented here; each output is a lookup or an
argmin/argmax of the run's own grids (NFR-2).
"""
from __future__ import annotations

import numpy as np

from regimerain.features.schema import GEO, SYNOPTIC
from regimerain.runs.fields import RunFields, valid_date
from regimerain.runs.places import PLACES

# The box MODEL_SPEC section 7 searches for the MSLP minimum (f_dist_mslp_min_km).
TRACK_BOX = (10.0, 30.0, 70.0, 95.0)          # lat0, lat1, lon0, lon1
TRACK_MIN_DEPTH_HPA = 3.0                     # a low must be this far below its 10-degree surroundings
BACKGROUND_DEG = 10.0


def _r(x, nd):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def regime_index(f: RunFields) -> np.ndarray:
    """Most probable synoptic regime per (lead, cell); -1 off land."""
    top = f.p_synoptic.argmax(-1).astype(np.int16)
    top[:, ~f.land] = -1
    return top


def wettest(f: RunFields) -> dict:
    """The land cell and lead with the highest corrected rain."""
    c = np.where(f.land[None], f.corrected, -np.inf)
    k, i, j = np.unravel_index(int(np.nanargmax(c)), c.shape)
    top = int(f.p_synoptic[k, i, j].argmax())
    return {"lead": int(k), "valid_date": valid_date(f.init, f.leads[k]).isoformat(),
            "lat": float(f.lat[i]), "lon": float(f.lon[j]),
            "raw_mm": _r(f.raw[k, i, j], 1), "corrected_mm": _r(f.corrected[k, i, j], 1),
            "p_heavy": _r(f.p_heavy[k, i, j], 2) if f.p_heavy is not None else None,
            "p_very_heavy": _r(f.p_very_heavy[k, i, j], 2) if f.p_very_heavy is not None else None,
            "regime": SYNOPTIC[top], "geo": GEO[int(f.geo[i, j])]}


def _box_mean(a: np.ndarray, half: int) -> np.ndarray:
    """Mean over a (2*half+1)^2 window, shrinking at the edges (separable, cumulative sums)."""
    def along(x, axis):
        n = x.shape[axis]
        c = np.cumsum(np.insert(x, 0, 0, axis=axis), axis=axis)
        lo = np.clip(np.arange(n) - half, 0, n)
        hi = np.clip(np.arange(n) + half + 1, 0, n)
        take = lambda idx: np.take(c, idx, axis=axis)
        return take(hi) - take(lo), (hi - lo)
    s, n0 = along(a, 0)
    s, n1 = along(s, 1)
    return s / (n0[:, None] * n1[None, :])


def depression_track(f: RunFields) -> list[dict]:
    """Per lead, the deepest compact MSLP low in the Bay of Bengal / central India box, if deep enough.

    Display only: the position of the minimum of MSLP minus its 10-degree background. Leads without
    such a low are skipped, so a run with no depression has an empty track.
    """
    if f.mslp is None:
        return []
    la0, la1, lo0, lo1 = TRACK_BOX
    bi = (f.lat >= la0) & (f.lat <= la1)
    bj = (f.lon >= lo0) & (f.lon <= lo1)
    step = float(np.median(np.diff(f.lat)))
    half = max(1, int(round(BACKGROUND_DEG / 2 / step)))
    out = []
    for k in range(len(f.leads)):
        p = f.mslp[k]
        anom = p - _box_mean(p, half)
        sub = np.where(bi[:, None] & bj[None, :], anom, np.inf)
        i, j = np.unravel_index(int(np.argmin(sub)), sub.shape)
        if -sub[i, j] >= TRACK_MIN_DEPTH_HPA:
            out.append({"lead": k, "lat": float(f.lat[i]), "lon": float(f.lon[j]), "mslp_hpa": round(float(p[i, j]), 1)})
    return out


def nearest_land_cell(f: RunFields, lat: float, lon: float, max_cells: int = 2) -> tuple[int, int] | None:
    i = int(np.argmin(np.abs(f.lat - lat)))
    j = int(np.argmin(np.abs(f.lon - lon)))
    if f.land[i, j]:
        return i, j
    best = None
    for di in range(-max_cells, max_cells + 1):
        for dj in range(-max_cells, max_cells + 1):
            a, b = i + di, j + dj
            if 0 <= a < len(f.lat) and 0 <= b < len(f.lon) and f.land[a, b]:
                d = abs(di) + abs(dj)
                if best is None or d < best[0]:
                    best = (d, a, b)
    return None if best is None else (best[1], best[2])


def place_values(f: RunFields, curves: list[dict] | None = None, regime_days: dict | None = None) -> list[dict]:
    """Each named place: its nearest land cell and that cell's values per lead."""
    by_regime = {c["regime"]: c for c in (curves or []) if c.get("variant") == "B" and c.get("geo", "all") == "all"}
    out = []
    for name, la, lo in PLACES:
        cell = nearest_land_cell(f, la, lo)
        if cell is None:
            continue
        i, j = cell
        leads = []
        for k in range(len(f.leads)):
            probs = f.p_synoptic[k, i, j]
            reg = SYNOPTIC[int(probs.argmax())]
            curve = by_regime.get(reg)
            leads.append({
                "raw_mm": _r(f.raw[k, i, j], 1), "corrected_mm": _r(f.corrected[k, i, j], 1),
                "corrected_global_mm": _r(f.corrected_global[k, i, j], 1) if f.corrected_global is not None else None,
                "p_heavy": _r(f.p_heavy[k, i, j], 2) if f.p_heavy is not None else None,
                "p_very_heavy": _r(f.p_very_heavy[k, i, j], 2) if f.p_very_heavy is not None else None,
                "truth_mm": _r(f.truth[k, i, j], 1) if f.truth is not None else None,
                "regime": reg, "regime_probs": {s: round(float(p), 2) for s, p in zip(SYNOPTIC, probs)},
                "qm_curve": curve["id"] if curve else None,
                "qm_curve_days": (curve["n_days"] if curve else (regime_days or {}).get(reg)),
            })
        out.append({"name": name, "lat": la, "lon": lo, "cell": {"lat": float(f.lat[i]), "lon": float(f.lon[j])},
                    "geo": GEO[int(f.geo[i, j])], "leads": leads})
    return out

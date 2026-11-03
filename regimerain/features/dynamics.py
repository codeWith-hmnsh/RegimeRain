"""Physics helpers on regular lat/lon grids (MODEL_SPEC section 7.1). Arrays end in (..., lat, lon)."""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

R_EARTH = 6.371e6
G = 9.80665


def ddx(f: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    dx = R_EARTH * np.cos(np.deg2rad(lat))[:, None] * np.deg2rad(np.gradient(lon))[None, :]
    return np.gradient(f, axis=-1) / dx


def ddy(f: np.ndarray, lat: np.ndarray) -> np.ndarray:
    dy = R_EARTH * np.deg2rad(np.gradient(lat))[:, None]
    return np.gradient(f, axis=-2) / dy


def rel_vorticity(u, v, lat, lon):
    return ddx(v, lat, lon) - ddy(u, lat)


def column_integrals(q, u, v, p_hpa, level_axis: int = 0):
    """Precipitable water (mm) and integrated vapour transport components (kg m-1 s-1).

    q, u, v share a pressure-level axis `level_axis` with levels `p_hpa` (any order).
    """
    p = np.asarray(p_hpa, "float64")
    order = np.argsort(p)
    p_pa = p[order] * 100.0
    q = np.take(np.asarray(q, "float64"), order, axis=level_axis)
    u = np.take(np.asarray(u, "float64"), order, axis=level_axis)
    v = np.take(np.asarray(v, "float64"), order, axis=level_axis)
    pw = trapezoid(q, p_pa, axis=level_axis) / G
    ivtx = trapezoid(q * u, p_pa, axis=level_axis) / G
    ivty = trapezoid(q * v, p_pa, axis=level_axis) / G
    return pw, ivtx, ivty


def imfc_mm_day(ivtx, ivty, lat, lon):
    """Vertically integrated moisture-flux convergence, mm/day (kg m-2 day-1)."""
    return -(ddx(ivtx, lat, lon) + ddy(ivty, lat)) * 86400.0

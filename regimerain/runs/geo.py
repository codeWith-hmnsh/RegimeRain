"""Polygon masks for the mock producer and place lookup (numpy only).

Reads the same GeoJSON the web draws: the Survey of India outline (`india.json`, the land mask of an
IMD-like grid) and Natural Earth land (`land.json`, to tell a coast from a land border).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

GEO_DIR = Path(__file__).resolve().parents[2] / "mvp" / "web" / "public" / "geo"


@lru_cache(maxsize=None)
def rings(name: str) -> tuple[np.ndarray, ...]:
    """All outer and inner rings of a FeatureCollection of (Multi)Polygons, as (n, 2) lon/lat arrays."""
    fc = json.loads((GEO_DIR / f"{name}.json").read_text(encoding="utf-8"))
    out = []
    for f in fc["features"]:
        g = f["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            out.extend(np.asarray(r, dtype=float) for r in poly)
    return tuple(out)


def inside(lon: np.ndarray, lat: np.ndarray, rs) -> np.ndarray:
    """Vectorised even-odd point-in-polygon over all rings."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    hit = np.zeros(lon.shape, bool)
    for r in rs:
        if lon.max() < r[:, 0].min() or lon.min() > r[:, 0].max() or lat.max() < r[:, 1].min() or lat.min() > r[:, 1].max():
            continue
        x1, y1 = r[:-1, 0], r[:-1, 1]
        x2, y2 = r[1:, 0], r[1:, 1]
        for a, b, c, d in zip(x1, y1, x2, y2):
            if d == b:
                continue
            cross = (b > lat) != (d > lat)
            hit ^= cross & (lon < (c - a) * (lat - b) / (d - b) + a)
    return hit


def mask(lat: np.ndarray, lon: np.ndarray, name: str) -> np.ndarray:
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    return inside(LON.ravel(), LAT.ravel(), rings(name)).reshape(LAT.shape)


def near(m: np.ndarray, cells: int) -> np.ndarray:
    """True where any cell within `cells` (Chebyshev distance) is True."""
    pad = np.pad(m, cells, constant_values=False)
    out = np.zeros_like(m)
    n0, n1 = m.shape
    for dy in range(-cells, cells + 1):
        for dx in range(-cells, cells + 1):
            out |= pad[cells + dy:cells + dy + n0, cells + dx:cells + dx + n1]
    return out

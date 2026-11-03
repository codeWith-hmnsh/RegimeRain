"""Cell-level depression / monsoon-low labels from track data (MODEL_SPEC section 6.2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi, dl = p2 - p1, np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def depression_mask(tracks: pd.DataFrame, days, lat, lon, radius_km: float = 500.0) -> np.ndarray:
    """tracks: columns time, lat, lon. days: rain-day start dates (window D 03 UTC -> D+1 03 UTC).

    Returns bool (day, lat, lon): a low/depression centre within `radius_km` during the rain day.
    """
    days = pd.DatetimeIndex(days)
    LAT, LON = np.meshgrid(np.asarray(lat), np.asarray(lon), indexing="ij")
    out = np.zeros((len(days), LAT.shape[0], LAT.shape[1]), bool)
    t = pd.to_datetime(tracks["time"])
    for i, d in enumerate(days):
        t0 = d.normalize() + pd.Timedelta(hours=3)
        pts = tracks[(t >= t0) & (t < t0 + pd.Timedelta(hours=24))]
        for la, lo in zip(pts["lat"].to_numpy(), pts["lon"].to_numpy()):
            out[i] |= haversine_km(LAT, LON, la, lo) <= radius_km
    return out

"""`RunFields`: one run's gridded product, in memory, before export (BACKEND_BUILD_PLAN section 2).

Every predictor returns this. A field set to None is *not built* (for example exceedance before
MODEL_SPEC section 12 exists), never zero; the exporter leaves it out and the web hides the layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

import numpy as np

from regimerain.features.schema import GEO, SYNOPTIC

Kind = Literal["mock", "replay", "model"]


def working_grid(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes (ascending) and longitudes of the rain domain (TRD 2.1)."""
    lat0, lat1, lon0, lon1 = cfg["grid"]["rain_box"]
    res = float(cfg["grid"]["res_deg"])
    lat = np.round(np.arange(lat0, lat1 + res / 2, res), 4)
    lon = np.round(np.arange(lon0, lon1 + res / 2, res), 4)
    return lat, lon


def lead_hours(lead: int) -> tuple[int, int]:
    """Forecast hours covered by rain day L (TRD 2.2): 03 UTC to 03 UTC."""
    return 3 + 24 * (lead - 1), 27 + 24 * (lead - 1)


def valid_date(init: date, lead: int) -> date:
    """Rain days are labelled by their start date: day L of an init on D starts on D + (L - 1)."""
    return init + timedelta(days=lead - 1)


@dataclass
class RunFields:
    kind: Kind
    init: date
    source: str                                   # "mock" | "hres" | "gfs"
    lat: np.ndarray                               # (nlat,) ascending
    lon: np.ndarray                               # (nlon,)
    leads: list[int]                              # rain days, 1-based
    land: np.ndarray                              # (nlat, nlon) bool
    raw: np.ndarray                               # (lead, nlat, nlon) mm/day, NaN off land
    corrected: np.ndarray                         # variant B (regime-aware mixture QM)
    p_synoptic: np.ndarray                        # (lead, nlat, nlon, 4) in SYNOPTIC order
    geo: np.ndarray                               # (nlat, nlon) int8 in GEO order, -1 off land
    corrected_global: np.ndarray | None = None    # variant A (one global QM)
    u850: np.ndarray | None = None                # (lead, nlat, nlon) m/s, defined everywhere
    v850: np.ndarray | None = None
    mslp: np.ndarray | None = None                # (lead, nlat, nlon) hPa
    p_heavy: np.ndarray | None = None             # (lead, nlat, nlon) P(rain >= heavy threshold)
    p_very_heavy: np.ndarray | None = None
    truth: np.ndarray | None = None               # replay only: observed rain for the valid days
    provenance: dict = field(default_factory=dict)

    @property
    def synthetic(self) -> bool:
        """Derived, never set by hand: only the mock predictor produces synthetic runs."""
        return self.kind == "mock"

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.leads), len(self.lat), len(self.lon)

    def layers(self) -> list[str]:
        """Names of the layers this run actually has (manifest.layers)."""
        out = ["raw", "corrected", "regime", "geo"]
        if self.corrected_global is not None:
            out.append("corrected_global")
        if self.u850 is not None and self.v850 is not None:
            out.append("wind850")
        for name in ("p_heavy", "p_very_heavy", "truth"):
            if getattr(self, name) is not None:
                out.append(name)
        return out

    def validate(self) -> "RunFields":
        n_lead, n_lat, n_lon = self.shape
        grid3 = (n_lead, n_lat, n_lon)
        if not np.all(np.diff(self.lat) > 0):
            raise ValueError("lat must be ascending (south to north)")
        if self.land.shape != (n_lat, n_lon) or self.geo.shape != (n_lat, n_lon):
            raise ValueError("land/geo must be (nlat, nlon)")
        if self.p_synoptic.shape != grid3 + (len(SYNOPTIC),):
            raise ValueError(f"p_synoptic must be {grid3 + (len(SYNOPTIC),)}, got {self.p_synoptic.shape}")
        for name in ("raw", "corrected", "corrected_global", "u850", "v850", "mslp", "p_heavy", "p_very_heavy", "truth"):
            a = getattr(self, name)
            if a is not None and a.shape != grid3:
                raise ValueError(f"{name} must be {grid3}, got {a.shape}")
        for name in ("raw", "corrected", "corrected_global", "truth"):
            a = getattr(self, name)
            if a is not None and np.nanmin(np.where(self.land, a, np.nan)) < 0:
                raise ValueError(f"{name} has negative rain")
        for name in ("p_heavy", "p_very_heavy"):
            a = getattr(self, name)
            if a is not None and (np.nanmin(a) < 0 or np.nanmax(a) > 1):
                raise ValueError(f"{name} outside [0, 1]")
        if self.p_heavy is not None and self.p_very_heavy is not None:
            if np.any(self.p_very_heavy[:, self.land] > self.p_heavy[:, self.land] + 1e-6):
                raise ValueError("P(very heavy) exceeds P(heavy)")  # MODEL_SPEC 12: monotone thresholds
        sums = self.p_synoptic[:, self.land].sum(-1)
        if not np.allclose(sums, 1.0, atol=1e-3):
            raise ValueError("synoptic probabilities must sum to 1 on land")
        if np.any((self.geo[self.land] < 0) | (self.geo[self.land] >= len(GEO))):
            raise ValueError("geo class missing on a land cell")
        return self

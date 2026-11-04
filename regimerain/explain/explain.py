"""Reasons behind a heavy-rain probability (MODEL_SPEC 15, PRD FR-12).

LightGBM's `pred_contrib=True` splits each prediction (in log-odds, before isotonic calibration) into one
contribution per feature plus a bias, exactly (TreeSHAP). The features that pushed P(heavy rain) up the
most become short phrases, e.g. "depression/low regime (p=0.72), strong moisture convergence (41 mm/day)".
Contributions are kept as numbers for a tooltip; the sentence never shows them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PHRASES = {
    "p_depression": "depression/low regime (p={v:.2f})",
    "p_active": "active monsoon regime (p={v:.2f})",
    "p_break": "break-monsoon regime (p={v:.2f})",
    "p_normal": "normal monsoon regime (p={v:.2f})",
    "f_llj_index": "low-level jet {v:.0f} m/s",
    "f_imfc": "moisture convergence {v:.0f} mm/day",
    "f_pw": "moist column ({v:.0f} mm water)",
    "f_ivt": "moisture transport {v:.0f} kg/m/s",
    "f_vort850_max500km": "cyclonic circulation nearby",
    "f_bob_vort_max": "cyclonic circulation over the Bay of Bengal",
    "f_dist_mslp_min_km": "low-pressure centre {v:.0f} km away",
    "f_mslp_anom": "pressure {v:+.1f} hPa vs domain mean",
    "f_w500": "rising air at 500 hPa",
    "qm_rain": "corrected forecast {v:.0f} mm",
    "f_rain": "raw forecast {v:.0f} mm",
    "f_rain_nbr3_max": "heavy rain forecast nearby ({v:.0f} mm)",
    "f_rain_nbr5_max": "heavy rain forecast nearby ({v:.0f} mm)",
    "f_rain_nbr3_mean": "wet surroundings ({v:.0f} mm average)",
    "f_rain_nbr5_mean": "wet surroundings ({v:.0f} mm average)",
    "f_cmz_rain": "core monsoon zone rain {v:.0f} mm",
    "f_cmz_wetfrac": "{v:.0%} of the core monsoon zone wet",
    "f_trough_lat": "monsoon trough at {v:.1f}°N",
    "windward": "windward slope (orographic lift)",
    "elev_std": "rugged terrain",
    "slope_mean": "steep terrain",
    "dist_coast_km": "{v:.0f} km from the coast",
    "mjo_phase": "MJO phase {v:.0f}",
    "mjo_amp": "MJO amplitude {v:.1f}",
    "f_u850": "850 hPa westerly {v:.0f} m/s",
    "f_ws850": "850 hPa wind {v:.0f} m/s",
}
# position and calendar push the probability too, but "because it is at 19 N" explains nothing to a forecaster
SILENT = {"lat", "lon", "zone", "geo", "doy_sin", "doy_cos", "lead", "mjo_rmm1", "mjo_rmm2", "f_v850", "elev_mean"}


def phrase(name: str, value: float) -> str:
    tmpl = PHRASES.get(name, name.replace("_", " "))
    try:
        return tmpl.format(v=float(value))
    except (TypeError, ValueError):
        return tmpl.split("{")[0].strip()


def top_reasons(booster, X: pd.DataFrame, k: int = 3) -> list[list[dict]]:
    """Per row: up to k features that raised the predicted log-odds most, as {feature, log_odds, text}."""
    if booster is None or len(X) == 0:
        return [[] for _ in range(len(X))]
    names = booster.feature_name()
    contrib = np.asarray(booster.predict(X[names], pred_contrib=True))[:, :-1]          # drop the bias column
    keep = np.array([n not in SILENT for n in names])
    out = []
    for r in range(len(X)):
        c = np.where(keep, contrib[r], -np.inf)
        rows = []
        for i in np.argsort(-c)[:k]:
            if not np.isfinite(c[i]) or c[i] <= 0:
                break
            rows.append({"feature": names[i], "log_odds": round(float(c[i]), 3),
                         "text": phrase(names[i], X.iloc[r][names[i]])})
        out.append(rows)
    return out

"""Display curves for the web (web contract v2 `qm_curves.json`): what each correction learned.

Per synoptic regime, the forecast-vs-observed quantile curve of the fold's training rows (all-India, all
terrain, the first lead), plus the single global curve. These summarise the experts for people; the
corrector itself uses the finer (regime, terrain, zone, lead) experts in `experts.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from regimerain.features.schema import SYNOPTIC


def display_curves(tr: pd.DataFrame, n0_days: float, n_q: int = 41) -> list[dict]:
    d = tr[tr.lead == tr.lead.min()]
    d = d[d.o_rain.notna()]
    qs = np.linspace(0.0, 1.0, n_q)
    out = []

    def curve(rows: pd.DataFrame, **meta) -> dict:
        n_days = int(rows.valid.nunique())
        return {**meta, "lead": int(d.lead.min()), "n_days": n_days,
                "w_shrink": round(n_days / (n_days + n0_days), 3) if meta["variant"] == "B" else None,
                "quantiles": qs.round(3).tolist(),
                "forecast_mm": np.quantile(rows.f_rain.to_numpy(float), qs).round(1).tolist(),
                "truth_mm": np.quantile(rows.o_rain.to_numpy(float), qs).round(1).tolist()}

    for k, reg in enumerate(SYNOPTIC):
        rows = d[d.y_synoptic == k]
        if len(rows):
            out.append(curve(rows, id=f"B/{reg}/all/all-India", variant="B", regime=reg, geo="all"))
    if len(d):
        out.append(curve(d, id="A/global", variant="A", regime="global", geo="all"))
    return out

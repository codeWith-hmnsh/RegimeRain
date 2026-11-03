"""BOM RMM (MJO) index (MODEL_SPEC 4.5).

BOM returns 403 to cloud machines (Phase 0 run 2), so the usual source is a manually downloaded
copy: `sources.rmm_file` in the config, or the RMM_FILE environment variable.
"""
from __future__ import annotations

import io
import os
import urllib.request

import numpy as np
import pandas as pd

RMM_URLS = ["http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt",
            "https://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"]
USER_AGENT = "regimerain/0.1 (SIH26080 research; https://github.com/shreyashsri79/080)"
COLS = ["year", "month", "day", "rmm1", "rmm2", "phase", "amplitude"]


def fetch_text(rmm_file: str | None = None) -> tuple[str, str]:
    path = os.environ.get("RMM_FILE") or rmm_file
    if path:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read(), f"file:{path}"
    errors = []
    for url in RMM_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", errors="replace"), url
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError("RMM download failed (BOM blocks cloud IPs). Download rmm.74toRealtime.txt in a "
                       "browser and set sources.rmm_file or RMM_FILE. " + " | ".join(errors))


def parse(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", skiprows=2, header=None, usecols=range(7), names=COLS)
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    df = df[(df.amplitude < 100) & df.phase.between(1, 8)]
    df["date"] = pd.to_datetime(dict(year=df.year.astype(int), month=df.month.astype(int), day=df.day.astype(int)))
    out = df[["date", "rmm1", "rmm2", "phase", "amplitude"]].reset_index(drop=True)
    out["phase"] = out["phase"].astype("int8")
    return out


def value_on(rmm: pd.DataFrame, date, max_stale_days: int = 3) -> dict:
    """Latest RMM on or before `date`; neutral (amp 0, phase 0) if missing or older than max_stale_days."""
    date = pd.Timestamp(date).normalize()
    past = rmm[rmm.date <= date]
    if past.empty or (date - past.date.iloc[-1]).days > max_stale_days:
        return {"rmm1": 0.0, "rmm2": 0.0, "phase": 0, "amplitude": 0.0, "stale": True}
    r = past.iloc[-1]
    return {"rmm1": float(r.rmm1), "rmm2": float(r.rmm2), "phase": int(r.phase),
            "amplitude": float(r.amplitude), "stale": False}


def align_to_dates(rmm: pd.DataFrame, dates, max_stale_days: int = 3) -> pd.DataFrame:
    rows = [dict(value_on(rmm, d, max_stale_days), date=pd.Timestamp(d).normalize()) for d in dates]
    return pd.DataFrame(rows)

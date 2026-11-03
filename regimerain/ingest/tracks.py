"""IBTrACS North Indian basin tracks (MODEL_SPEC 4.5). Columns confirmed in Phase 0 run 1."""
from __future__ import annotations

import pandas as pd

IBTRACS_NI = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
              "v04r01/access/csv/ibtracs.NI.list.v04r01.csv")
COLUMNS = {"SID": "sid", "ISO_TIME": "time", "LAT": "lat", "LON": "lon",
           "NEWDELHI_GRADE": "grade", "NEWDELHI_WIND": "wind_kt"}


def parse_ibtracs(source, years=(2015, 2022), months=(5, 10)) -> pd.DataFrame:
    """`source`: URL, path or file-like. Row 1 of the CSV is a units row and is skipped."""
    df = pd.read_csv(source, skiprows=[1], low_memory=False, na_values=[" ", ""],
                     usecols=list(COLUMNS), keep_default_na=True)
    df = df.rename(columns=COLUMNS)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for c in ("lat", "lon", "wind_kt"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time", "lat", "lon"])
    keep = df.time.dt.year.between(*years) & df.time.dt.month.between(*months)
    return df[keep].sort_values(["sid", "time"]).reset_index(drop=True)

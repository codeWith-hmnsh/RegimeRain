"""Load the partitioned feature table (TRD 3.3)."""
from __future__ import annotations

import pandas as pd
import pyarrow.dataset as pads


def load_table(cfg: dict, seasons, leads, columns=None, cell_stride: int = 1) -> pd.DataFrame:
    ds = pads.dataset(cfg["paths"]["table"], format="parquet", partitioning="hive")
    filt = pads.field("season").isin([int(s) for s in seasons]) & pads.field("lead").isin([int(l) for l in leads])
    df = ds.to_table(filter=filt, columns=columns).to_pandas()
    if cell_stride > 1:
        df = df[(df.lat_idx % cell_stride == 0) & (df.lon_idx % cell_stride == 0)]
    for c in ("season", "lead"):
        if c in df:
            df[c] = df[c].astype("int16")
    return df.reset_index(drop=True)

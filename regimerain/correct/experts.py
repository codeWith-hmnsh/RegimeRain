"""Expert hierarchy with day-based shrinkage (MODEL_SPEC section 11.4).

Keys are (regime, geo, zone, lead); -1 means "all". Chain:
    (r, g, z, L)  ->  (-1, g, z, L)  ->  (-1, g, -1, L)
Output of a key = w * expert(x) + (1 - w) * parent(x), w = n_days / (n_days + n0_days).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from regimerain.correct.qm import QMExpert

Key = tuple[int, int, int, int]


class ExpertSet:
    def __init__(self, n0_days: float = 30, tail_q: float = 0.95, min_exceedances: int = 50,
                 xi_bounds=(0.0, 0.4), quantiles: int = 1001):
        self.experts: dict[Key, QMExpert] = {}
        self.n0_days = float(n0_days)
        self.fit_kw = dict(tail_q=tail_q, min_exceedances=min_exceedances, xi_bounds=tuple(xi_bounds),
                           quantiles=quantiles)

    @classmethod
    def from_config(cls, qm_cfg: dict) -> "ExpertSet":
        return cls(n0_days=qm_cfg["n0_days"], tail_q=qm_cfg["tail_q"], min_exceedances=qm_cfg["min_exceedances"],
                   xi_bounds=qm_cfg["xi_bounds"], quantiles=qm_cfg["quantiles"])

    def _put(self, key: Key, d: pd.DataFrame) -> None:
        self.experts[key] = QMExpert.fit(d["f_rain"].to_numpy(), d["o_rain"].to_numpy(),
                                         n_days=d["valid"].nunique(), **self.fit_kw)

    def fit(self, df: pd.DataFrame, regime_col: Optional[str] = None) -> "ExpertSet":
        """df columns: f_rain, o_rain, geo, zone, lead, valid [, regime_col]. Fit on LABELLED regimes."""
        for (g, L), d in df.groupby(["geo", "lead"], observed=True):
            self._put((-1, int(g), -1, int(L)), d)
        for (g, z, L), d in df.groupby(["geo", "zone", "lead"], observed=True):
            self._put((-1, int(g), int(z), int(L)), d)
        if regime_col is not None:
            for (r, g, z, L), d in df.groupby([regime_col, "geo", "zone", "lead"], observed=True):
                self._put((int(r), int(g), int(z), int(L)), d)
        return self

    @staticmethod
    def parent(key: Key) -> Optional[Key]:
        r, g, z, L = key
        if r != -1:
            return (-1, g, z, L)
        if z != -1:
            return (-1, g, -1, L)
        return None

    def weight(self, key: Key) -> float:
        e = self.experts.get(key)
        return 0.0 if e is None else e.n_days / (e.n_days + self.n0_days)

    def apply(self, key: Key, x) -> np.ndarray:
        x = np.asarray(x, "float64")
        e = self.experts.get(key)
        par = self.parent(key)
        if e is None:
            return self.apply(par, x) if par is not None else x.copy()   # unfitted root: identity
        y = e.transform(x)
        if par is None:
            return y
        yp = self.apply(par, x)
        w = self.weight(key)
        out = w * y + (1.0 - w) * yp
        if not e.has_tail:                       # above this expert's threshold, trust the parent's tail
            hi = x > e.f_u
            out[hi] = yp[hi]
        return out

    def apply_global(self, df: pd.DataFrame) -> np.ndarray:
        """Variant A: no regime split, (geo, zone, lead) chain only."""
        out = np.zeros(len(df))
        x_all = df["f_rain"].to_numpy("float64")
        for (g, z, L), idx in df.groupby(["geo", "zone", "lead"], observed=True).indices.items():
            out[idx] = self.apply((-1, int(g), int(z), int(L)), x_all[idx])
        return out.astype("float32")

    def sample_counts(self) -> pd.DataFrame:
        rows = [{"regime": k[0], "geo": k[1], "zone": k[2], "lead": k[3], "n": e.n, "n_days": e.n_days,
                 "w_shrink": self.weight(k), "has_tail": e.has_tail} for k, e in self.experts.items()]
        return pd.DataFrame(rows).sort_values(["lead", "geo", "zone", "regime"]).reset_index(drop=True)

    def to_frame(self, variant: str) -> pd.DataFrame:
        rows = []
        for (r, g, z, L), e in self.experts.items():
            rec = e.to_record()
            rec.update(variant=variant, regime=r, geo=g, zone=z, lead=L, w_shrink=self.weight((r, g, z, L)))
            rows.append(rec)
        return pd.DataFrame(rows)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, n0_days: float) -> "ExpertSet":
        s = cls(n0_days=n0_days)
        for rec in df.to_dict("records"):
            s.experts[(int(rec["regime"]), int(rec["geo"]), int(rec["zone"]), int(rec["lead"]))] = QMExpert.from_record(rec)
        return s

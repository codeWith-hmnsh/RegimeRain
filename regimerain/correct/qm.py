"""One quantile-mapping expert: empirical body + GPD upper tail (MODEL_SPEC sections 11.2-11.3).

corrected = F_obs^-1( F_fcst(x) ), with
  * wet-day frequency: forecasts at or below the forecast quantile of the observed dry fraction -> 0;
  * above `tail_q`, both CDFs are spliced with Generalised Pareto tails so extremes can exceed
    the training maximum.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import genpareto

N_QUANTILES = 1001


def qgrid(n: int = N_QUANTILES) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def fit_gpd(exceedances: np.ndarray, xi_bounds=(0.0, 0.4)) -> tuple[float, float]:
    """Fit a GPD (location 0) to positive exceedances; shape xi clipped to `xi_bounds`."""
    xi, _, sigma = genpareto.fit(exceedances, floc=0.0)
    xi_c = float(np.clip(xi, *xi_bounds))
    if xi_c != xi:
        _, _, sigma = genpareto.fit(exceedances, fc=xi_c, floc=0.0)
    return xi_c, float(sigma)


@dataclass
class QMExpert:
    n: int
    n_days: int
    f_q: np.ndarray
    o_q: np.ndarray
    f_dry_thr: float
    tail_q: float
    f_u: float
    o_u: float
    has_tail: bool = False
    f_xi: float = float("nan")
    f_sigma: float = float("nan")
    o_xi: float = float("nan")
    o_sigma: float = float("nan")

    @classmethod
    def fit(cls, f, o, n_days: int, tail_q: float = 0.95, min_exceedances: int = 50,
            xi_bounds=(0.0, 0.4), quantiles: int = N_QUANTILES) -> "QMExpert":
        f = np.asarray(f, "float64")
        o = np.asarray(o, "float64")
        ok = ~(np.isnan(f) | np.isnan(o))
        f, o = f[ok], o[ok]
        if len(f) == 0:
            raise ValueError("no samples to fit")
        q = qgrid(quantiles)
        p_dry = float(np.mean(o <= 0.0))
        e = cls(n=len(f), n_days=int(n_days), f_q=np.quantile(f, q), o_q=np.quantile(o, q),
                f_dry_thr=float(np.quantile(f, p_dry)), tail_q=float(tail_q),
                f_u=float(np.quantile(f, tail_q)), o_u=float(np.quantile(o, tail_q)))
        fe, oe = f[f > e.f_u] - e.f_u, o[o > e.o_u] - e.o_u
        if len(fe) >= min_exceedances and len(oe) >= min_exceedances and e.o_u > 0:
            e.f_xi, e.f_sigma = fit_gpd(fe, xi_bounds)
            e.o_xi, e.o_sigma = fit_gpd(oe, xi_bounds)
            e.has_tail = True
        return e

    @property
    def _q(self) -> np.ndarray:
        return qgrid(len(self.f_q))

    def _fq_strict(self) -> np.ndarray:
        # np.interp needs strictly increasing x; break ties (e.g. many zeros) with a negligible ramp
        return self.f_q + np.arange(len(self.f_q)) * 1e-9

    def transform(self, x) -> np.ndarray:
        x = np.asarray(x, "float64")
        out = np.zeros_like(x)
        wet = x > max(self.f_dry_thr, 0.0)
        if not wet.any():
            return out
        xw = x[wet]
        q = np.interp(xw, self._fq_strict(), self._q)
        if self.has_tail:
            t = xw > self.f_u
            q[t] = self.tail_q + (1 - self.tail_q) * genpareto.cdf(xw[t] - self.f_u, self.f_xi, scale=self.f_sigma)
        q = np.clip(q, 0.0, 1.0 - 1e-6)
        y = np.interp(q, self._q, self.o_q)
        if self.has_tail:
            t = q > self.tail_q
            y[t] = self.o_u + genpareto.ppf((q[t] - self.tail_q) / (1 - self.tail_q), self.o_xi, scale=self.o_sigma)
        out[wet] = np.maximum(y, 0.0)
        return out

    def to_record(self) -> dict:
        d = asdict(self)
        d["f_q"], d["o_q"] = self.f_q.astype("float32").tolist(), self.o_q.astype("float32").tolist()
        return d

    @classmethod
    def from_record(cls, d: dict) -> "QMExpert":
        d = dict(d)
        d["f_q"], d["o_q"] = np.asarray(d["f_q"], "float64"), np.asarray(d["o_q"], "float64")
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

"""MockPredictor: a SYNTHETIC run for building and testing everything downstream of the models.

Nothing here is a forecast, an observation or a measured score. The exporter marks every file
`synthetic: true` because `kind == "mock"`, and the web shows a banner. The fields are shape-true
(the real 0.25 deg working grid, an India-only land mask like IMD's, rain days 1-5, four synoptic
classes plus a geo class, the config thresholds) so the UI cannot behave differently when real runs
arrive. Some regimes and metrics are deliberately made *worse* after correction, so the
"where it did not help" path is always exercised (NFR-4).

Replaces the run half of mvp/tools/make_sample_run.py; same idea, now inside the package.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from regimerain.features.schema import GEO, SYNOPTIC
from regimerain.runs import geo as geomask
from regimerain.runs.fields import RunFields, working_grid

DEFAULT_INIT = date(2022, 7, 14)
DEFAULT_SEED = 26080

# Regime and terrain factors the synthetic "correction" applies (a stand-in for the QM mixture).
SYN_FACTOR = {"active": 1.10, "break": 0.82, "depression": 1.36, "normal": 1.00}
GEO_FACTOR = {"plains": 1.00, "coastal": 0.95, "orographic": 1.15}
# Training days behind each regime's curve (invented; the real numbers come from the QM experts).
MOCK_REGIME_DAYS = {"active": 612, "break": 388, "depression": 141, "normal": 1873}


def _band(x, lo, hi, w=1.2):
    """Soft 0..1 window: ~1 between lo and hi, fading over w degrees."""
    return 1 / (1 + np.exp(-(x - lo) / (w / 4))) / (1 + np.exp((x - hi) / (w / 4)))


def _smooth(a, k=2):
    out = a.copy()
    for _ in range(k):
        out = (out + np.roll(out, 1, 0) + np.roll(out, -1, 0) + np.roll(out, 1, 1) + np.roll(out, -1, 1)) / 5
    return out


def _terrain(LAT, LON):
    ghats_lon = 73.5 + (21 - LAT) * (3.5 / 12.5)
    ghats = np.exp(-((LON - ghats_lon + 0.25) / 0.55) ** 2) * _band(LAT, 8, 21.5)
    foot_lat = np.interp(LON, [76, 80, 84, 88, 92, 96], [31.0, 29.3, 27.8, 27.0, 27.2, 28.2])
    himal = np.exp(-((LAT - foot_lat) / 0.8) ** 2) * _band(LON, 75.5, 96.5, 2)
    khasi = np.exp(-((LAT - 25.3) ** 2 + (LON - 91.7) ** 2) / 0.35)
    arakan = np.exp(-((LON - (93.8 - (LAT - 17) * 0.35)) / 0.6) ** 2) * _band(LAT, 15.5, 22.5)
    return ghats, himal, khasi, arakan


def mock_static(cfg: dict) -> dict:
    """Land mask (inside the Survey of India outline), geo classes, and Natural Earth land for coasts."""
    lat, lon = working_grid(cfg)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    land = geomask.mask(lat, lon, "india")
    ne_land = geomask.mask(lat, lon, "land")
    res = float(cfg["grid"]["res_deg"])
    coast_cells = max(1, int(round(float(cfg["static"]["coastal_km"]) / (111.2 * res))))
    coastal = land & geomask.near(~ne_land, coast_cells)
    ghats, himal, khasi, arakan = _terrain(LAT, LON)
    oro = land & (np.clip(ghats + himal + khasi + arakan, 0, 1) > 0.3)
    g = np.full(land.shape, -1, np.int8)
    g[land] = GEO.index("plains")
    g[coastal] = GEO.index("coastal")
    g[oro] = GEO.index("orographic")
    return {"lat": lat, "lon": lon, "LAT": LAT, "LON": LON, "land": land, "ne_land": ne_land, "geo": g}


def mock_fields(cfg: dict, init: date = DEFAULT_INIT, seed: int = DEFAULT_SEED, leads=None) -> RunFields:
    rng = np.random.default_rng(seed)
    st = mock_static(cfg)
    LAT, LON, land, g = st["LAT"], st["LON"], st["land"], st["geo"]
    leads = list(leads or cfg["leads"] or [1, 2, 3, 4, 5])
    heavy, very_heavy = cfg["exceed"]["thresholds"]
    ghats, himal, khasi, arakan = _terrain(LAT, LON)
    tibet = _band(LAT, 31, 60, 2) * _band(LON, 78, 120, 2)
    shadow = _band(LAT, 0, 17, 2) * _band(LON, 76.5, 80.5, 1.5)
    noise_base = _smooth(rng.normal(0, 1, LAT.shape), 6)
    geo_fac = np.vectorize(lambda k: GEO_FACTOR[GEO[k]] if k >= 0 else 1.0)(g)
    syn_fac = np.array([SYN_FACTOR[s] for s in SYNOPTIC])

    out = {k: [] for k in ("raw", "B", "A", "P", "u", "v", "mslp", "ph", "pvh")}
    n = len(leads)
    for k, L in enumerate(leads):
        t = k / max(1, n - 1)
        clat, clon = 20.0 + 3.6 * t, 87.6 - 10.0 * t          # a Bay of Bengal depression moving north-west
        dx, dy = LON - clon, LAT - clat
        r = np.hypot(dx, dy) + 1e-6

        u = 13 * np.exp(-((LAT - 12.5) / 4.5) ** 2) - 6 * np.exp(-((LAT - 27.5) / 2.5) ** 2) + 2
        v = 4.5 * np.exp(-((LAT - 13) / 6) ** 2)
        vt = 15 * (r / 2.6) * np.exp(1 - r / 2.6)
        u, v = u - vt * dy / r, v + vt * dx / r
        damp = np.where(st["ne_land"], 0.65, 1.0) * (1 - 0.65 * tibet)
        u, v = u * damp, v * damp

        mslp = (1010 - 0.3 * (LAT - 6.5) - 4 * np.exp(-((LON - 72) ** 2 + (LAT - 28) ** 2) / 40)
                - 9 * np.exp(-(r ** 2) / (2 * 1.8 ** 2)) + 0.3 * rng.normal(0, 1, LAT.shape))

        noise = _smooth(noise_base * 0.6 + rng.normal(0, 1, LAT.shape) * 0.8, 3)
        trough_lat = 22.5 + 1.2 * t
        raw = (1.2 + 3.4 * np.clip(u, 0, None) * ghats + 34 * himal + 70 * khasi + 38 * arakan
               + 92 * np.exp(-((dx + 0.9) ** 2 + (dy + 0.7) ** 2) / (2 * 1.5 ** 2))
               + 17 * np.exp(-((LAT - trough_lat) / 2.8) ** 2) * _band(LON, 71, 89, 4))
        raw *= np.exp(0.38 * noise) * (1 - 0.75 * tibet)
        raw *= 1 - 0.65 * _band(LON, 50, 75.5, 3) * _band(LAT, 23.5, 60, 3)   # Thar and the north-west stay dry
        raw *= 1 - 0.55 * shadow                                               # rain shadow east of the Ghats

        scores = np.stack([
            2.2 * np.exp(-((LAT - trough_lat) / 3.2) ** 2) * _band(LON, 71, 89, 3),            # active
            1.4 * shadow + 0.6 * _band(LON, 60, 75.5, 3) * _band(LAT, 23, 40, 3) + 0.25,      # break
            3.6 * np.exp(-(r ** 2) / (2 * 2.0 ** 2)),                                        # depression
            np.full(LAT.shape, 0.9),                                                         # normal
        ], -1) + rng.normal(0, 0.15, LAT.shape + (len(SYNOPTIC),))
        z = scores * 1.7
        e = np.exp(z - z.max(-1, keepdims=True))
        P = e / e.sum(-1, keepdims=True)

        B = raw * (P @ syn_fac) * geo_fac * np.exp(rng.normal(0, 0.035, raw.shape))
        A = raw * 1.05 * (1 + 0.04 * np.clip(raw / 60, 0, 2))
        ph = 1 / (1 + np.exp(-(B - heavy) / 10))
        pvh = np.minimum(ph, 1 / (1 + np.exp(-(B - very_heavy) / 13)))

        off = ~land
        for a in (raw, B, A, ph, pvh):
            a[off] = np.nan
        for key, a in (("raw", raw), ("B", B), ("A", A), ("P", P), ("u", u), ("v", v), ("mslp", mslp), ("ph", ph), ("pvh", pvh)):
            out[key].append(a.astype("float32"))

    return RunFields(
        kind="mock", init=init, source="mock", lat=st["lat"], lon=st["lon"], leads=leads, land=land,
        raw=np.stack(out["raw"]), corrected=np.stack(out["B"]), corrected_global=np.stack(out["A"]),
        p_synoptic=np.stack(out["P"]), geo=g, u850=np.stack(out["u"]), v850=np.stack(out["v"]),
        mslp=np.stack(out["mslp"]), p_heavy=np.stack(out["ph"]), p_very_heavy=np.stack(out["pvh"]),
        provenance={"seed": seed, "note": "Synthetic sample: invented values, not a forecast or a measurement."},
    ).validate()


# ------------------------------------------------------------------------------------ QM curves
def mock_curves(cfg: dict, seed: int = DEFAULT_SEED, n_q: int = 41) -> list[dict]:
    """Variant B parent curves per synoptic regime (all-India, all terrain) and the variant A global curve."""
    rng = np.random.default_rng(seed + 1)
    qs = np.linspace(0, 1, n_q)
    shape = {"active": 1.3, "break": 0.8, "depression": 1.6, "normal": 1.0}
    scale = {"active": 14, "break": 6, "depression": 28, "normal": 9}
    curves, allf, allt = [], [], []
    for reg in SYNOPTIC:
        f = rng.gamma(shape[reg], scale[reg], MOCK_REGIME_DAYS[reg] * 40)
        tail = 1 + 0.35 * (SYN_FACTOR[reg] - 1) * np.clip(f / np.percentile(f, 90), 0, 3)
        tr = f * SYN_FACTOR[reg] * tail * np.exp(rng.normal(0, 0.18, f.size))
        allf.append(f)
        allt.append(tr)
        curves.append({"id": f"B/{reg}/all/all-India", "variant": "B", "regime": reg, "geo": "all",
                       "n_days": MOCK_REGIME_DAYS[reg], "w_shrink": round(MOCK_REGIME_DAYS[reg] / (MOCK_REGIME_DAYS[reg] + 30), 3),
                       "quantiles": qs.round(3).tolist(), "forecast_mm": np.quantile(f, qs).round(1).tolist(),
                       "truth_mm": np.quantile(tr, qs).round(1).tolist()})
    F, T = np.concatenate(allf), np.concatenate(allt)
    curves.append({"id": "A/global", "variant": "A", "regime": "global", "geo": "all", "n_days": sum(MOCK_REGIME_DAYS.values()),
                   "quantiles": qs.round(3).tolist(), "forecast_mm": np.quantile(F, qs).round(1).tolist(),
                   "truth_mm": np.quantile(T, qs).round(1).tolist()})
    return curves


# ------------------------------------------------------------------------------------ backtest report
# Pooled, lead 1, heavy threshold: raw, then the change for A and B. Invented to exercise the UI.
_BASE = {64.5: {"rmse": 14.8, "ets": 0.21, "csi": 0.27, "pod": 0.46, "far": 0.52, "fss": 0.38},
         115.6: {"rmse": 14.8, "ets": 0.09, "csi": 0.12, "pod": 0.24, "far": 0.69, "fss": 0.18}}
_GAIN = {"A": {"rmse": -0.3, "ets": 0.01, "csi": 0.01, "pod": 0.02, "far": 0.01, "fss": 0.02},
         "B": {"rmse": -0.9, "ets": 0.03, "csi": 0.03, "pod": 0.05, "far": 0.02, "fss": 0.05}}
# Regime-aware correction by synoptic regime: a multiplier on B's gain. Break is made worse on purpose.
_REGIME_GAIN = {"active": 1.2, "break": -0.9, "depression": 2.4, "normal": 0.1}
_REGIME_BASE = {"active": 1.0, "break": 0.7, "depression": 1.3, "normal": 0.9}


def _row(variant, fold, lead, synoptic, t, rng, windows):
    b = _BASE[t]
    decay = 1 - 0.07 * (lead - 1)
    rb = _REGIME_BASE.get(synoptic, 1.0)
    gain = 0.0 if variant == "raw" else 1.0
    gmul = _REGIME_GAIN.get(synoptic, 1.0) if variant == "B" else 1.0
    g = _GAIN.get(variant, {})
    j = (lambda: 1 + 0.06 * rng.normal()) if fold != "pooled" else (lambda: 1.0)
    rmse = b["rmse"] * (1 + 0.08 * (lead - 1)) * (2 - rb) * j() + gain * g.get("rmse", 0) * gmul
    s = {"ets": b["ets"] * decay * rb, "csi": b["csi"] * decay * rb, "pod": b["pod"] * decay * rb, "far": b["far"] / max(decay, 0.5)}
    s = {k: v * j() + gain * g.get(k, 0) * gmul for k, v in s.items()}
    fss0 = b["fss"] * decay * rb * j() + gain * g.get("fss", 0) * gmul
    fss = {str(w): float(np.clip(fss0 * (1 + 0.28 * np.log2(w)), 0, 0.97)) for w in windows}
    n = 120 * 4900 * (7 if fold == "pooled" else 1) // (1 if synoptic == "all" else 4)
    return {"variant": variant, "fold": fold, "lead": lead, "synoptic": synoptic, "geo": "all", "threshold": t,
            "n_cell_days": int(n), "n_events_obs": int(n * (0.02 if t == 64.5 else 0.005)),
            "rmse": round(float(rmse), 3), **{k: round(float(np.clip(v, 0, 1)), 4) for k, v in s.items()},
            "fss": {k: round(v, 4) for k, v in fss.items()}}


def mock_report(cfg: dict, seed: int = DEFAULT_SEED) -> dict:
    """A backtest report in the schema regimerain.verify.report writes (TRD 3.6), with invented numbers."""
    rng = np.random.default_rng(seed + 2)
    seasons, leads = list(cfg["seasons"]), list(cfg["leads"])
    thresholds = [t for t in cfg["verify"]["thresholds"] if t in _BASE]
    windows = list(cfg["verify"]["fss_windows"])
    rows = [_row(v, fold, L, syn, t, rng, windows)
            for v in ("raw", "A", "B") for fold in ["pooled"] + seasons for L in leads
            for syn in ["all"] + list(SYNOPTIC) for t in thresholds
            if fold == "pooled" or syn == "all"]
    pooled = {(r["variant"], r["lead"], r["threshold"]): r for r in rows if r["fold"] == "pooled" and r["synoptic"] == "all"}
    deltas = []
    for L in leads:
        for v, ref in (("A", "raw"), ("B", "raw"), ("B", "A")):
            for t in thresholds:
                for m in ("ets", "csi", "pod", "far"):
                    d = pooled[(v, L, t)][m] - pooled[(ref, L, t)][m]
                    half = 0.6 * abs(d) + 0.006
                    deltas.append({"variant": v, "vs": ref, "lead": L, "synoptic": "all", "threshold": t, "metric": m,
                                   "delta": round(d, 4), "ci90": [round(d - half, 4), round(d + half, 4)],
                                   "significant": bool(abs(d) > half)})
            d = pooled[(v, L, thresholds[0])]["rmse"] - pooled[(ref, L, thresholds[0])]["rmse"]
            half = 0.5 * abs(d) + 0.05
            deltas.append({"variant": v, "vs": ref, "lead": L, "synoptic": "all", "threshold": None, "metric": "rmse",
                           "delta": round(d, 3), "ci90": [round(d - half, 3), round(d + half, 3)], "significant": bool(abs(d) > half)})
    classifier = [{"test": s, "test_metrics": {"accuracy": round(0.62 + 0.03 * rng.normal(), 3),
                                               "macro_f1": round(0.49 + 0.03 * rng.normal(), 3),
                                               "per_class_recall": {c: round(float(np.clip(x + 0.05 * rng.normal(), 0, 1)), 3)
                                                                    for c, x in zip(SYNOPTIC, (0.58, 0.51, 0.66, 0.71))}}}
                  for s in seasons]
    bins = np.linspace(0.05, 0.95, 10)
    exceedance = {str(t): {"lead": {str(L): {"reliability": {
        "bin_mean_p": bins.round(2).tolist(),
        "obs_freq": [round(float(np.clip(p * (0.88 if t == thresholds[0] else 0.8) + 0.03 * rng.normal(), 0, 1)), 3) for p in bins],
        "count": [int(4000 * np.exp(-5 * p) + 40) for p in bins]}} for L in leads}} for t in thresholds}
    return {"backtest_id": "mock", "config_sha256": None, "git_commit": None, "truth_source": "mock",
            "forecast_source": "mock", "seasons": seasons, "leads": leads, "variants": ["raw", "A", "B"],
            "rows": rows, "deltas": deltas, "classifier": classifier, "exceedance": exceedance}

"""Model 2: error regimes (MODEL_SPEC 10) and variant C, the error-regime-aware correction (MODEL_SPEC 11).

The synoptic regimes of variant B are named by meteorologists. Error regimes are discovered from the data:
cluster the training days by *how the forecast was wrong* (too wet, heavy area missed, rain displaced north,
extremes too weak, ...), learn to predict that cluster from forecast-time information only, and fit one QM
expert set per cluster. At test time the predicted cluster probabilities weight the experts, as in variant B.

Leakage rules (all inside one LOMO fold, separately per lead):
- clusters are fitted on the fold's `fit` seasons only; inner_val and test days are assigned afterwards;
- the classifier trains on `fit` days, early-stops on `inner_val` days;
- the test season's descriptors (which need the truth) are used only to report agreement, never to fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from regimerain.config import by_mode, lgb_params
from regimerain.features.schema import SYNOPTIC, ZONES
from regimerain.label.depression import haversine_km

HEAVY = 64.5
DESCRIPTORS = ["bias", "larea", "disp_km", "disp_north_km", "corr", "lint"]
DAY_KEYS = ["init", "lead"]
DOMAIN = ["f_llj_index", "f_trough_lat", "f_bob_vort_max", "f_cmz_rain", "f_cmz_wetfrac",
          "mjo_rmm1", "mjo_rmm2", "mjo_amp", "mjo_phase", "doy_sin", "doy_cos"]
ZONE_AGG = {"f_rain": "mean", "f_heavy_frac": "mean", "f_imfc": "mean", "f_pw": "mean"}

PHRASES = {("bias", 1): "too wet", ("bias", -1): "too dry",
           ("larea", 1): "heavy area too large", ("larea", -1): "heavy rain missed",
           ("disp_km", 1): "heavy rain displaced", ("disp_km", -1): "heavy rain well placed",
           ("disp_north_km", 1): "rain too far north", ("disp_north_km", -1): "rain too far south",
           ("corr", 1): "pattern right", ("corr", -1): "pattern wrong",
           ("lint", 1): "extremes too strong", ("lint", -1): "extremes too weak"}


# ------------------------------------------------------------------------------------ 10.1 descriptors
def day_descriptors(rows: pd.DataFrame, n_zones: int = len(ZONES), heavy: float = HEAVY) -> pd.DataFrame:
    """One row per (init, lead): 6 error descriptors x zone (needs o_rain, so training/verification only)."""
    d = rows[DAY_KEYS + ["zone", "lat", "lon", "f_rain", "o_rain"]].copy()
    f, o = d.f_rain.to_numpy("float64"), d.o_rain.to_numpy("float64")
    a, b = np.log1p(f), np.log1p(o)
    hf, ho = np.where(f >= heavy, f, 0.0), np.where(o >= heavy, o, 0.0)
    d = d.assign(f=f, o=o, nf=(f >= heavy).astype(float), no=(o >= heavy).astype(float),
                 wf=hf, wo=ho, wf_lat=hf * d.lat, wf_lon=hf * d.lon, wo_lat=ho * d.lat, wo_lon=ho * d.lon,
                 a=a, b=b, aa=a * a, bb=b * b, ab=a * b, n=1.0)
    g = d.groupby(DAY_KEYS + ["zone"], observed=True)
    s = g[["f", "o", "nf", "no", "wf", "wo", "wf_lat", "wf_lon", "wo_lat", "wo_lon", "a", "b", "aa", "bb", "ab", "n"]].sum()
    q = g[["f", "o"]].quantile(0.95)
    out = pd.DataFrame(index=s.index)
    out["bias"] = (s.f - s.o) / (s.o + 1.0)
    out["larea"] = np.log((s.nf + 1) / (s.no + 1))
    both = (s.wf > 0) & (s.wo > 0)
    cf_lat, cf_lon = s.wf_lat / s.wf.where(s.wf > 0), s.wf_lon / s.wf.where(s.wf > 0)
    co_lat, co_lon = s.wo_lat / s.wo.where(s.wo > 0), s.wo_lon / s.wo.where(s.wo > 0)
    out["disp_km"] = np.where(both, haversine_km(co_lat.fillna(0), co_lon.fillna(0), cf_lat.fillna(0), cf_lon.fillna(0)), 0.0)
    out["disp_north_km"] = np.where(both, (cf_lat - co_lat).fillna(0) * 111.2, 0.0)
    n = s.n
    cov = s.ab / n - (s.a / n) * (s.b / n)
    va, vb = s.aa / n - (s.a / n) ** 2, s.bb / n - (s.b / n) ** 2
    ok = (va > 1e-12) & (vb > 1e-12)
    out["corr"] = np.where(ok, cov / np.sqrt(va.clip(lower=1e-12) * vb.clip(lower=1e-12)), 0.0)
    out["lint"] = np.log((q.f + 1) / (q.o + 1))
    wide = out.unstack("zone")
    wide.columns = [f"{name}_z{int(z)}" for name, z in wide.columns]
    cols = [f"{name}_z{z}" for z in range(n_zones) for name in DESCRIPTORS]
    return wide.reindex(columns=cols).fillna(0.0)                  # a zone outside the domain: no error signal


def day_features(rows: pd.DataFrame, n_zones: int = len(ZONES), heavy: float = HEAVY) -> pd.DataFrame:
    """One row per (init, lead): forecast-time information only (domain indices, MJO, season, zone aggregates)."""
    first = rows.groupby(DAY_KEYS, observed=True)[DOMAIN].first()
    z = rows.assign(f_heavy_frac=(rows.f_rain >= heavy).astype(float)) \
            .groupby(DAY_KEYS + ["zone"], observed=True).agg(ZONE_AGG).unstack("zone")
    z.columns = [f"{name}_z{int(k)}" for name, k in z.columns]
    z = z.reindex(columns=[f"{name}_z{k}" for k in range(n_zones) for name in ZONE_AGG])
    return first.join(z)


def day_synoptic(rows: pd.DataFrame) -> pd.Series:
    """Day-level synoptic label: the most common label over land cells (for the crosstab only)."""
    return rows.groupby(DAY_KEYS, observed=True).y_synoptic.agg(lambda s: int(np.bincount(s.astype(int)).argmax()))


# ------------------------------------------------------------------------------------ 10.2 clustering
def fit_error_regimes(E: np.ndarray, k_range=(4, 8), seed: int = 0, n_boot: int = 20) -> dict:
    """Standardise, PCA to 90 % variance, KMeans; k chosen by silhouette x bootstrap stability (ARI)."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(E)
    pca = PCA(n_components=0.90, random_state=seed).fit(scaler.transform(E))
    Z = pca.transform(scaler.transform(E))
    rng = np.random.default_rng(seed)
    best = None
    for k in range(int(k_range[0]), int(k_range[1]) + 1):
        if k >= len(Z):
            break
        km = KMeans(k, n_init=20, random_state=seed).fit(Z)
        if len(set(km.labels_)) < 2:
            continue
        sil = float(silhouette_score(Z, km.labels_))
        aris = []
        for _ in range(n_boot):                                   # stability under resampling of days
            idx = rng.choice(len(Z), len(Z), replace=True)
            kb = KMeans(k, n_init=5, random_state=int(rng.integers(1_000_000_000))).fit(Z[idx])
            aris.append(adjusted_rand_score(km.labels_, kb.predict(Z)))
        score = sil * float(np.mean(aris))
        if best is None or score > best["score"]:
            best = {"score": score, "k": k, "kmeans": km, "silhouette": sil, "stability": float(np.mean(aris))}
    if best is None:
        raise ValueError("too few days to cluster error regimes")
    best.update(scaler=scaler, pca=pca, labels=best["kmeans"].labels_)
    return best


def assign(model: dict, E: np.ndarray) -> np.ndarray:
    return model["kmeans"].predict(model["pca"].transform(model["scaler"].transform(E)))


def name_clusters(E: pd.DataFrame, labels: np.ndarray, k: int) -> list[str]:
    """Each cluster named by its most distinctive descriptor (standardised cluster mean), e.g. 'too wet (central)'."""
    std = (E - E.mean()) / E.std(ddof=0).replace(0, 1)
    names = []
    for c in range(k):
        m = std[labels == c].mean()
        if m.empty or m.abs().max() == 0:
            names.append(f"cluster {c}")
            continue
        col = m.abs().idxmax()
        desc, z = col.rsplit("_z", 1)
        names.append(f"{PHRASES[(desc, 1 if m[col] > 0 else -1)]} ({ZONES[int(z)]})")
    return names


# ------------------------------------------------------------------------------------ 10.3 day classifier
def train_err_classifier(X_fit, y_fit, X_val, y_val, k: int, cfg: dict):
    import lightgbm as lgb
    p = lgb_params(cfg, {**cfg["errreg"]["params"], "objective": "multiclass", "num_class": k})
    rounds = int(p.pop("num_boost_round", 1000))
    stop = int(p.pop("early_stopping_rounds", 50))
    dtr = lgb.Dataset(X_fit, y_fit, free_raw_data=False)
    dva = lgb.Dataset(X_val, y_val, reference=dtr, free_raw_data=False)
    return lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(stop, verbose=False)])


def predict_days(booster, X: pd.DataFrame, k: int) -> np.ndarray:
    P = np.asarray(booster.predict(X, num_iteration=booster.best_iteration or None)).reshape(len(X), -1)
    return P[:, :k]


# ------------------------------------------------------------------------------------ fold driver
def run_variant_c(cfg: dict, fold: dict, tr: pd.DataFrame, te: pd.DataFrame, log=print):
    """Variant C for one fold. Returns (corrected rain for te rows, diagnostics, expert set)."""
    from regimerain.correct.experts import ExpertSet
    from regimerain.correct.mixture import correct_mixture
    ecfg = cfg["errreg"]
    n_boot = int(by_mode(cfg, ecfg["n_boot"]))
    leads = sorted(int(x) for x in te.lead.unique())
    tr_lab = np.full(len(tr), -1, "int16")
    kmax = 0
    per_lead, P_te = {}, {}
    for L in leads:
        trL, teL = tr[tr.lead == L], te[te.lead == L]
        E = day_descriptors(trL)
        X = day_features(trL).reindex(E.index)
        season = trL.groupby(DAY_KEYS, observed=True).season.first().reindex(E.index).to_numpy()
        is_fit = np.isin(season, fold["fit"])
        is_val = season == fold["inner_val"]
        model = fit_error_regimes(E[is_fit].to_numpy(), tuple(ecfg["k_range"]), int(cfg["seed"]), n_boot)
        k = model["k"]
        labels = assign(model, E.to_numpy())                         # fit days: own clusters; others: nearest
        clf = train_err_classifier(X[is_fit], labels[is_fit], X[is_val], labels[is_val], k, cfg)
        Xte = day_features(teL)
        Pd = predict_days(clf, Xte, k)
        P_te[L] = pd.DataFrame(Pd, index=Xte.index)
        day_label = pd.Series(labels, index=E.index)
        idx = pd.MultiIndex.from_frame(trL[DAY_KEYS])
        tr_lab[np.flatnonzero((tr.lead == L).to_numpy())] = day_label.reindex(idx).to_numpy()
        # diagnostics: names, sizes, crosstab vs synoptic, agreement with the (truth-based) test clusters
        names = name_clusters(E[is_fit], labels[is_fit], k)
        syn = day_synoptic(trL).reindex(E.index).to_numpy()
        ct = pd.crosstab(pd.Series(labels[is_fit], name="cluster"), pd.Series(syn[is_fit], name="synoptic"),
                         normalize="index").reindex(columns=range(len(SYNOPTIC)), fill_value=0.0)
        Ete = day_descriptors(teL).reindex(Xte.index)
        true_te = assign(model, Ete.to_numpy())
        pred_te = Pd.argmax(axis=1)
        majority = float(np.mean(true_te == np.bincount(labels[is_fit], minlength=k).argmax()))
        per_lead[str(L)] = {
            "k": k, "silhouette": round(model["silhouette"], 4), "stability_ari": round(model["stability"], 4),
            "pca_components": int(model["pca"].n_components_), "names": names,
            "fit_days_per_cluster": np.bincount(labels[is_fit], minlength=k).tolist(),
            "crosstab_cluster_by_synoptic": {names[c]: {SYNOPTIC[s]: round(float(ct.loc[c, s]), 3)
                                                        for s in range(len(SYNOPTIC))} for c in ct.index},
            "test_agreement": round(float(np.mean(pred_te == true_te)), 4), "test_majority_baseline": round(majority, 4),
            "best_iteration": int(clf.best_iteration or 0)}
        kmax = max(kmax, k)
        log(f"fold {fold['test']} L{L}: {k} error regimes (silhouette {model['silhouette']:.2f}, "
            f"stability {model['stability']:.2f}); test agreement {per_lead[str(L)]['test_agreement']:.2f} "
            f"vs majority {majority:.2f}")
    if (tr_lab < 0).any():
        raise RuntimeError("variant C: some training rows have no error-regime label")
    EC = ExpertSet.from_config(cfg["qm"]).fit(tr.assign(y_error=tr_lab), "y_error")
    P = np.zeros((len(te), kmax))
    for L, frame in P_te.items():
        m = (te.lead == L).to_numpy()
        rows = frame.reindex(pd.MultiIndex.from_frame(te.loc[m, DAY_KEYS])).to_numpy()
        P[np.flatnonzero(m), :rows.shape[1]] = rows
    q = cfg["qm"]
    return correct_mixture(te, P, EC, q["p_min"], q["p_hard"]), {"leads": per_lead}, EC

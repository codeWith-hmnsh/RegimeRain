"""`regimerain backtest`: leave-one-monsoon-out folds (MODEL_SPEC 13.4). Built so far: variants raw, A, B,
and the heavy-rain probability model M4 (when `exceed.enabled` and variant B runs).

Per test season T writes cache/fold=T/:
    predictions.parquet   keys, o_rain, y_synoptic, raw/A/B rain, p_* regime probabilities,
                          p_exc_<threshold> heavy-rain probabilities (M4)
    classifier.json       test-season classifier metrics, temperature, best iteration
    exceed.json           M4 event counts, best iterations, top features, calibration season
    qm_curves.json        what the corrections learned, for the web (all-India curve per regime + global)
    qm_counts_A.csv, qm_counts_B.csv   expert sample counts and shrinkage weights
    DONE                  config_sha256 + git commit (written last)
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from regimerain.classify.model import (classifier_metrics, fit_temperature, predict_proba, raw_logits,
                                       smooth_on_grid, train_classifier)
from regimerain.config import by_mode, config_sha256
from regimerain.correct.experts import ExpertSet
from regimerain.correct.mixture import correct_mixture
from regimerain.data import load_table
from regimerain.features.schema import REGIME_PROB_FEATURES
from regimerain.folds import fold_dir, is_done, lomo_folds, mark_done

BUILT_VARIANTS = ("raw", "A", "B", "C", "CM")
KEEP = ["init", "valid", "lead", "season", "lat_idx", "lon_idx", "zone", "geo", "y_synoptic", "o_rain"]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent.parent, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _grid_shape(cfg: dict) -> tuple[int, int]:
    from regimerain.static.basic import load_static
    st = load_static(cfg)
    return st.sizes["lat"], st.sizes["lon"]


def run_exceed(cfg: dict, fold: dict, tr: pd.DataFrame, te: pd.DataFrame, P: np.ndarray, preds: pd.DataFrame,
               shape: tuple[int, int], out_dir: Path, log=print, with_cm: bool = False) -> None:
    """M4 heavy-rain probabilities for the test season (MODEL_SPEC 12), written into `preds` as p_exc_<thr>.

    Training rows (fit seasons) and calibration rows (inner_val) get cross-fitted p_* and qm_rain; the test
    season gets the fold's own classifier probabilities and variant-B rain, exactly what a live run would see.
    """
    from regimerain.classify.crossfit import oof_features
    from regimerain.exceed.model import fit_exceed, prob_column
    t0 = time.time()
    P_oof, qm_oof = oof_features(tr, cfg, shape, log)
    tr["qm_rain"] = qm_oof                       # in place: tr is not used after this, and it is large
    for k, c in enumerate(REGIME_PROB_FEATURES):
        tr[c] = P_oof[:, k]
    te_x = te.assign(qm_rain=preds["B"].to_numpy(), **{c: P[:, k] for k, c in enumerate(REGIME_PROB_FEATURES)})
    fit_rows = tr[tr.season.isin(fold["fit"])]
    val_rows = tr[tr.season == fold["inner_val"]]
    groups = sorted(te_x.lead.unique()) if cfg["exceed"].get("per_lead", False) else [None]
    thresholds = sorted(float(t) for t in cfg["exceed"]["thresholds"])
    out = np.zeros((len(te_x), len(thresholds)), "float32")
    diag = {"test": fold["test"], "fit": fold["fit"], "inner_val": fold["inner_val"], "models": []}
    for lead in groups:
        sel = (lambda d: d) if lead is None else (lambda d, L=lead: d[d.lead == L])
        model = fit_exceed(sel(fit_rows), sel(val_rows), cfg, log=log)
        rows = np.arange(len(te_x)) if lead is None else np.flatnonzero(te_x.lead.to_numpy() == lead)
        out[rows] = model.predict(te_x.iloc[rows])
        diag["models"].append({"lead": "all" if lead is None else int(lead), **model.diagnostics()})
    for k, t in enumerate(thresholds):
        preds[prob_column(t)] = out[:, k]
    if with_cm:                                  # variant CM: same rows and features, target o_rain (FR-13)
        from regimerain.exceed.model import fit_conditional_mean, predict_conditional_mean
        cm = fit_conditional_mean(fit_rows, val_rows, cfg, log=log)
        preds["CM"] = predict_conditional_mean(cm, te_x)
        diag["conditional_mean"] = {"best_iteration": int(cm.best_iteration or 0)}
    diag["seconds"] = round(time.time() - t0, 1)
    (out_dir / "exceed.json").write_text(json.dumps(diag, indent=2, default=str))
    log(f"fold {fold['test']}: exceedance model done ({diag['seconds']:.0f}s)")


def run_fold(cfg: dict, fold: dict, variants, log=print) -> dict:
    t0 = time.time()
    leads = cfg["leads"]
    tr = load_table(cfg, fold["train"], leads)
    te = load_table(cfg, [fold["test"]], leads)
    if tr.empty or te.empty:
        raise RuntimeError(f"fold {fold['test']}: empty table (train {len(tr)}, test {len(te)}) - run `features`")
    stride = int(by_mode(cfg, cfg["classifier"]["cell_stride"]))
    sub = lambda d: d[(d.lat_idx % stride == 0) & (d.lon_idx % stride == 0)] if stride > 1 else d
    fit = sub(tr[tr.season.isin(fold["fit"])])
    val = sub(tr[tr.season == fold["inner_val"]])
    log(f"fold {fold['test']}: classifier on {len(fit):,} rows (val {len(val):,})")
    clf = train_classifier(fit, val, cfg)
    T = fit_temperature(raw_logits(clf, val), val.y_synoptic.to_numpy())
    P = predict_proba(clf, te, T)
    shape = _grid_shape(cfg)
    P = smooth_on_grid(te, P, shape, int(cfg["classifier"].get("smooth_size", 3)))

    preds = te[KEEP].copy()
    preds["raw"] = te["f_rain"].to_numpy("float32")
    preds[REGIME_PROB_FEATURES] = P.astype("float32")
    qcfg = cfg["qm"]
    out_dir = fold_dir(cfg["paths"]["cache"], fold["test"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if "A" in variants:
        EA = ExpertSet.from_config(qcfg).fit(tr, None)
        preds["A"] = EA.apply_global(te)
        EA.sample_counts().to_csv(out_dir / "qm_counts_A.csv", index=False)
    if "B" in variants:
        EB = ExpertSet.from_config(qcfg).fit(tr, "y_synoptic")
        preds["B"] = correct_mixture(te, P, EB, qcfg["p_min"], qcfg["p_hard"])
        EB.sample_counts().to_csv(out_dir / "qm_counts_B.csv", index=False)
        if cfg.get("exceed", {}).get("enabled", False):
            run_exceed(cfg, fold, tr, te, P, preds, shape, out_dir, log, with_cm="CM" in variants)
    if "C" in variants:
        from regimerain.errreg.errreg import run_variant_c
        preds["C"], diag_c, EC = run_variant_c(cfg, fold, tr, te, log=log)
        EC.sample_counts().to_csv(out_dir / "qm_counts_C.csv", index=False)
        (out_dir / "errreg.json").write_text(json.dumps(diag_c, indent=2))
    from regimerain.correct.curves import display_curves
    (out_dir / "qm_curves.json").write_text(json.dumps(display_curves(tr, float(qcfg["n0_days"]))))
    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    diag = {"test": fold["test"], "train": fold["train"], "inner_val": fold["inner_val"],
            "temperature": T, "best_iteration": int(clf.best_iteration or 0),
            "test_metrics": classifier_metrics(te.y_synoptic.to_numpy(), P),
            "seconds": round(time.time() - t0, 1)}
    (out_dir / "classifier.json").write_text(json.dumps(diag, indent=2))
    return diag


def run_backtest(cfg: dict, variants=BUILT_VARIANTS, only=None, force: bool = False, log=print) -> list[int]:
    unknown = [v for v in variants if v not in BUILT_VARIANTS]
    if unknown:
        log(f"backtest: variants {unknown} not built yet (MODEL_SPEC 10, 12) - skipping them")
    variants = [v for v in variants if v in BUILT_VARIANTS]
    if "CM" in variants and not ("B" in variants and cfg.get("exceed", {}).get("enabled", False)):
        log("backtest: variant CM needs variant B and exceed.enabled (it reuses the M4 rows) - skipping it")
        variants = [v for v in variants if v != "CM"]
    info = {"config_sha256": config_sha256(cfg), "git_commit": git_commit(), "variants": list(variants),
            "seasons": list(cfg["seasons"]), "leads": list(cfg["leads"])}
    ran = []
    for fold in lomo_folds(cfg["seasons"], only):
        if is_done(cfg["paths"]["cache"], fold["test"]) and not force:
            log(f"fold {fold['test']}: DONE, skipping")
            continue
        diag = run_fold(cfg, fold, variants, log)
        m = diag["test_metrics"]
        log(f"fold {fold['test']}: classifier acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
            f"({diag['seconds']:.0f}s)")
        mark_done(cfg["paths"]["cache"], fold["test"], info)
        ran.append(fold["test"])
    return ran

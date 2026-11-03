"""`regimerain backtest`: leave-one-monsoon-out folds (MODEL_SPEC 13.4). Built so far: variants raw, A, B.

Per test season T writes cache/fold=T/:
    predictions.parquet   keys, o_rain, y_synoptic, raw/A/B rain, p_* regime probabilities
    classifier.json       test-season classifier metrics, temperature, best iteration
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

BUILT_VARIANTS = ("raw", "A", "B")
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
    P = smooth_on_grid(te, P, _grid_shape(cfg), int(cfg["classifier"].get("smooth_size", 3)))

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

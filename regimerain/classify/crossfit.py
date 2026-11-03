"""Out-of-fold regime probabilities and corrected rain for the training rows (MODEL_SPEC 9.5, 11.7).

The heavy-rain model (M4) uses `p_*` and `qm_rain` as features. On its training rows these must come from
models that never saw those rows, or M4 learns from over-confident in-sample values it will never see at
test time. Training seasons are split into groups; for each group, a classifier and a set of QM experts are
fitted on the other training seasons only, then applied to the group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from regimerain.classify.model import (K, fit_temperature, predict_proba, raw_logits, smooth_on_grid,
                                       train_classifier)
from regimerain.config import by_mode
from regimerain.correct.experts import ExpertSet
from regimerain.correct.mixture import correct_mixture
from regimerain.folds import crossfit_groups


def _stride(cfg: dict):
    s = int(by_mode(cfg, cfg["classifier"]["cell_stride"]))
    return (lambda d: d[(d.lat_idx % s == 0) & (d.lon_idx % s == 0)]) if s > 1 else (lambda d: d)


def oof_features(tr: pd.DataFrame, cfg: dict, shape: tuple[int, int], log=print) -> tuple[np.ndarray, np.ndarray]:
    """(P, qm_rain) for every row of `tr`, each row predicted by models fitted without its season.

    P is (n, 4) regime probabilities, smoothed on the grid like the test-time ones; qm_rain is the regime-aware
    (variant B) corrected rain using those probabilities. Rows keep the order of `tr`.
    """
    seasons = sorted(int(s) for s in tr.season.unique())
    if len(seasons) < 2:
        raise ValueError("cross-fitting needs at least two training seasons")
    groups = crossfit_groups(seasons, int(by_mode(cfg, cfg["classifier"]["crossfit_groups"])))
    sub = _stride(cfg)
    smooth = int(cfg["classifier"].get("smooth_size", 3))
    qcfg = cfg["qm"]
    P = np.zeros((len(tr), K), "float32")
    qm = np.zeros(len(tr), "float32")
    season_col = tr.season.to_numpy()
    for g in groups:
        held = np.isin(season_col, g)
        rest = tr[~held]
        rest_seasons = sorted(int(s) for s in rest.season.unique())
        stop = rest_seasons[-1]                       # early stopping + temperature on one of the other seasons
        fit_part = rest[rest.season != stop] if len(rest_seasons) > 1 else rest
        val_part = rest[rest.season == stop]
        if len(rest_seasons) == 1:
            log(f"crossfit: only season {stop} outside group {g}; it serves for both fitting and early stopping")
        clf = train_classifier(sub(fit_part), sub(val_part), cfg)
        v = sub(val_part)
        T = fit_temperature(raw_logits(clf, v), v.y_synoptic.to_numpy())
        hd = tr[held]
        Pg = smooth_on_grid(hd, predict_proba(clf, hd, T), shape, smooth)
        experts = ExpertSet.from_config(qcfg).fit(rest, "y_synoptic")
        qm[held] = correct_mixture(hd, Pg, experts, qcfg["p_min"], qcfg["p_hard"])
        P[held] = Pg
        log(f"crossfit: group {g} predicted from {rest_seasons} ({held.sum():,} rows)")
    return P, qm

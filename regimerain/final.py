"""`regimerain fit-final`: refit every model on all seasons and write a model set (MODEL_SPEC 17, TRD 3.4).

The models shown to users come from here; every number shown comes from the backtest. Inputs: the feature
table and the backtest cache (all folds DONE), plus the newest report for the skill gate.

1. M1 regime classifier: all seasons, num_boost_round = 1.1 x the median best iteration across folds (no early
   stopping); temperature = median of the fold temperatures.
2. M3 QM experts: variants A (global) and B (regime-aware) on all seasons.
3. M4 heavy-rain model: all seasons except the last, with each row's held-out `p_*` and `qm_rain` from the
   backtest (out-of-fold by construction); early stopping and isotonic calibration on the last season.
   (MODEL_SPEC 17.4 calibrates on pooled out-of-fold raw scores, which the backtest does not store; disclosed
   in `model_card.json`.)
4. Skill gate: copied from the newest report (an out-of-sample object by nature).

    models/<model_set_id>/  config.yaml, model_card.json, feature_list.json, hashes.json, static.npz,
                            classifier/booster.txt, classifier/temperature.json, qm/experts.parquet,
                            exceed/<threshold>.txt + .iso.pkl, gate/skill_gate.parquet
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from regimerain.backtest import git_commit
from regimerain.classify.model import class_weights
from regimerain.config import by_mode, config_sha256, lgb_params
from regimerain.correct.experts import ExpertSet
from regimerain.data import load_table
from regimerain.features.schema import BASE_FEATURES, CATEGORICAL, EXCEED_FEATURES, REGIME_PROB_FEATURES, assert_no_leak
from regimerain.folds import check_poolable, fold_dir
from regimerain.static.basic import TERRAIN_VARS
from regimerain.verify.metrics import fmt_threshold

KEYS = ["init", "lead", "lat_idx", "lon_idx"]
STATIC_VARS = ("land", "geo", "zone") + TERRAIN_VARS     # everything a live run needs from the static layers


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_snapshot_hash(cfg: dict) -> str:
    """Identity of the training data: every table file's relative path and size."""
    root = Path(cfg["paths"]["table"])
    items = sorted((str(p.relative_to(root)).replace("\\", "/"), p.stat().st_size) for p in root.rglob("*.parquet"))
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def fold_diagnostics(cfg: dict) -> list[dict]:
    return [json.loads((fold_dir(cfg["paths"]["cache"], s) / "classifier.json").read_text()) for s in cfg["seasons"]]


def fit_classifier_fixed(df: pd.DataFrame, cfg: dict, rounds: int):
    import lightgbm as lgb
    assert_no_leak(BASE_FEATURES)
    params = lgb_params(cfg, cfg["classifier"]["params"])
    params.pop("num_boost_round", None)
    y = df.y_synoptic.to_numpy()
    dtr = lgb.Dataset(df[BASE_FEATURES], y, weight=class_weights(y), categorical_feature=CATEGORICAL)
    booster = lgb.train(params, dtr, num_boost_round=rounds)
    assert_no_leak(booster.feature_name())
    return booster


def oof_inputs(cfg: dict) -> pd.DataFrame:
    """Held-out p_* and variant-B rain for every cell-day, from the backtest predictions."""
    cols = KEYS + REGIME_PROB_FEATURES + ["B"]
    parts = [pd.read_parquet(fold_dir(cfg["paths"]["cache"], s) / "predictions.parquet", columns=cols)
             for s in cfg["seasons"]]
    return pd.concat(parts, ignore_index=True).rename(columns={"B": "qm_rain"})


def fit_final(cfg: dict, log=print) -> Path:
    t0 = time.time()
    info = check_poolable(cfg["paths"]["cache"], cfg["seasons"])
    seasons, leads = list(cfg["seasons"]), list(cfg["leads"])
    commit = git_commit()
    snapshot = data_snapshot_hash(cfg)
    model_set_id = hashlib.sha256(f"{config_sha256(cfg)}|{commit}|{snapshot}".encode()).hexdigest()[:12]
    out = Path(cfg["paths"]["models"]) / model_set_id
    out.mkdir(parents=True, exist_ok=True)
    log(f"fit-final: model set {model_set_id} -> {out}")

    tr = load_table(cfg, seasons, leads)
    diag = fold_diagnostics(cfg)
    best = [d["best_iteration"] for d in diag if d.get("best_iteration")]
    rounds = max(10, int(round(1.1 * float(np.median(best))))) if best else 200
    temperature = float(np.median([d["temperature"] for d in diag]))
    stride = int(by_mode(cfg, cfg["classifier"]["cell_stride"]))
    sub = tr[(tr.lat_idx % stride == 0) & (tr.lon_idx % stride == 0)] if stride > 1 else tr
    log(f"fit-final: classifier on {len(sub):,} rows, {rounds} rounds, T={temperature:.3f}")
    clf = fit_classifier_fixed(sub, cfg, rounds)
    (out / "classifier").mkdir(exist_ok=True)
    clf.save_model(str(out / "classifier" / "booster.txt"))
    (out / "classifier" / "temperature.json").write_text(json.dumps({"T": temperature, "rounds": rounds}))

    qcfg = cfg["qm"]
    (out / "qm").mkdir(exist_ok=True)
    EB = ExpertSet.from_config(qcfg).fit(tr, "y_synoptic")
    experts = pd.concat([ExpertSet.from_config(qcfg).fit(tr, None).to_frame("A"), EB.to_frame("B")], ignore_index=True)
    experts.to_parquet(out / "qm" / "experts.parquet", index=False)
    from regimerain.correct.curves import display_curves
    from regimerain.features.schema import SYNOPTIC
    (out / "qm" / "curves.json").write_text(json.dumps(display_curves(tr, float(qcfg["n0_days"]))))
    counts = EB.sample_counts()                       # same definition as the replay runs (qm_counts_B.csv)
    counts = counts[counts.regime >= 0]
    regime_days = {SYNOPTIC[int(r)]: int(n) for r, n in counts.groupby("regime").n_days.max().items()
                   if int(r) < len(SYNOPTIC)}
    log(f"fit-final: {len(experts)} QM experts (A + B)")

    exceed_card = None
    if cfg.get("exceed", {}).get("enabled", False):
        from regimerain.exceed.model import fit_exceed
        tr = tr.merge(oof_inputs(cfg), on=KEYS, how="inner", validate="one_to_one")
        calib = seasons[-1]
        model = fit_exceed(tr[tr.season != calib], tr[tr.season == calib], cfg, log=log)
        (out / "exceed").mkdir(exist_ok=True)
        for m in model.models:
            name = fmt_threshold(m.threshold)
            if m.booster is not None:
                m.booster.save_model(str(out / "exceed" / f"{name}.txt"), num_iteration=m.best_iteration or None)
            with open(out / "exceed" / f"{name}.iso.pkl", "wb") as f:
                pickle.dump({"iso": m.iso, "base_rate": m.base_rate, "threshold": m.threshold,
                             "constant": m.booster is None}, f)
        exceed_card = {**model.diagnostics(), "calibration_note": (
            f"early stopping and isotonic calibration on season {calib}; trained on {seasons[:-1]}. "
            "MODEL_SPEC 17.4 uses pooled out-of-fold raw scores, which the backtest does not store.")}

    reports = sorted(Path(cfg["paths"]["reports"]).glob("*/skill_gate.parquet"), key=lambda p: p.stat().st_mtime)
    if reports:
        (out / "gate").mkdir(exist_ok=True)
        (out / "gate" / "skill_gate.parquet").write_bytes(reports[-1].read_bytes())

    from regimerain.static.basic import load_static
    st = load_static(cfg)
    np.savez_compressed(out / "static.npz", lat=st.lat.values, lon=st.lon.values,
                        **{k: st[k].values for k in STATIC_VARS})
    (out / "config.yaml").write_text(yaml.safe_dump({k: v for k, v in cfg.items() if k != "paths"}, sort_keys=False))
    (out / "feature_list.json").write_text(json.dumps({"classifier": BASE_FEATURES, "exceed": EXCEED_FEATURES}, indent=1))
    card = {"model_set_id": model_set_id, "config_sha256": config_sha256(cfg), "git_commit": commit,
            "backtest_git_commit": next(iter(info.values())).get("git_commit"), "data_snapshot": snapshot,
            "seasons": seasons, "leads": leads, "classifier": {"rounds": rounds, "temperature": temperature,
            "fold_best_iterations": best}, "regime_days": regime_days, "exceed": exceed_card,
            "gate_from": str(reports[-1].parent.name) if reports else None,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0, 1)}
    (out / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    hashes = {str(p.relative_to(out)).replace("\\", "/"): _sha256(p)
              for p in sorted(out.rglob("*")) if p.is_file() and p.name != "hashes.json"}
    (out / "hashes.json").write_text(json.dumps(hashes, indent=1))
    log(f"fit-final: done in {card['seconds']:.0f}s, {len(hashes)} files hashed")
    return out


class ModelSet:
    """A loaded model set: everything a live run needs, no training (TRD 3.4)."""

    def __init__(self, path: str | Path):
        import lightgbm as lgb
        self.path = Path(path)
        self.card = json.loads((self.path / "model_card.json").read_text())
        self.config = yaml.safe_load((self.path / "config.yaml").read_text())
        stored = json.loads((self.path / "hashes.json").read_text())
        bad = [k for k, h in stored.items() if _sha256(self.path / k) != h]
        if bad:
            raise ValueError(f"model set {self.path.name}: files changed since fit-final: {bad}")
        self.hashes = stored
        curves = self.path / "qm" / "curves.json"
        self.curves = json.loads(curves.read_text()) if curves.exists() else None
        self.regime_days = self.card.get("regime_days")
        self.classifier = lgb.Booster(model_file=str(self.path / "classifier" / "booster.txt"))
        self.temperature = json.loads((self.path / "classifier" / "temperature.json").read_text())["T"]
        frame = pd.read_parquet(self.path / "qm" / "experts.parquet")
        n0 = float(self.config["qm"]["n0_days"])
        self.experts = {v: ExpertSet.from_frame(frame[frame.variant == v], n0) for v in ("A", "B")}
        z = np.load(self.path / "static.npz")
        missing = [k for k in STATIC_VARS if k not in z.files]
        if missing:
            raise ValueError(f"model set {self.path.name}: static.npz lacks {missing}; rerun fit-final")
        self.static = xr.Dataset({k: (("lat", "lon"), z[k]) for k in STATIC_VARS},
                                 coords={"lat": z["lat"], "lon": z["lon"]})
        self.shape = self.static["land"].shape
        gate = self.path / "gate" / "skill_gate.parquet"
        self.gate = pd.read_parquet(gate) if gate.exists() else None
        self.exceed = []
        for t in sorted(float(x) for x in self.config["exceed"]["thresholds"]):
            name = fmt_threshold(t)
            iso_file = self.path / "exceed" / f"{name}.iso.pkl"
            if not iso_file.exists():
                continue
            with open(iso_file, "rb") as f:
                meta = pickle.load(f)
            booster = None if meta["constant"] else lgb.Booster(model_file=str(self.path / "exceed" / f"{name}.txt"))
            self.exceed.append((t, booster, meta))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """df: feature rows for one init (BASE_FEATURES + keys). Returns p_*, A, B and p_exc_* columns."""
        from scipy.special import softmax
        from regimerain.classify.model import smooth_on_grid
        from regimerain.correct.mixture import correct_mixture
        from regimerain.exceed.model import prob_column
        z = np.asarray(self.classifier.predict(df[BASE_FEATURES], raw_score=True)).reshape(len(df), -1)
        P = softmax(z / self.temperature, axis=1)
        P = smooth_on_grid(df, P, self.shape, int(self.config["classifier"].get("smooth_size", 3)))  # as in backtest
        q = self.config["qm"]
        out = df[KEYS].copy()
        for k, c in enumerate(REGIME_PROB_FEATURES):
            out[c] = P[:, k].astype("float32")
        out["A"] = self.experts["A"].apply_global(df).astype("float32")
        out["B"] = correct_mixture(df, P, self.experts["B"], q["p_min"], q["p_hard"])
        X = df.assign(qm_rain=out["B"].to_numpy(), **{c: out[c].to_numpy() for c in REGIME_PROB_FEATURES})
        probs = []
        for t, booster, meta in self.exceed:
            p = (np.full(len(df), meta["base_rate"]) if booster is None
                 else meta["iso"].predict(booster.predict(X[EXCEED_FEATURES])))
            probs.append(np.clip(p, 0, 1))
        if probs:
            Pm = np.minimum.accumulate(np.column_stack(probs), axis=1)
            for k, (t, _, _) in enumerate(self.exceed):
                out[prob_column(t)] = Pm[:, k].astype("float32")
        return out

    def explain(self, rows: pd.DataFrame, out: pd.DataFrame, k: int = 3) -> list[list[dict]]:
        """Top reasons from the heavy-rain (first threshold) model for each row of `rows` (MODEL_SPEC 15)."""
        from regimerain.explain.explain import top_reasons
        if not self.exceed:
            return [[] for _ in range(len(rows))]
        X = rows.assign(qm_rain=out["B"].to_numpy(), **{c: out[c].to_numpy() for c in REGIME_PROB_FEATURES})
        return top_reasons(self.exceed[0][1], X, k)

    def gate_served(self, out: pd.DataFrame, static: xr.Dataset | None = None) -> np.ndarray:
        """True where the corrected rain is served, False where the skill gate is OFF (raw is served instead).

        Lookup per row: the unit (zone x geo) x lead x predicted regime row of the gate table, else the
        unit x lead 'all' row. No gate table, or a unit the gate never scored: served (and disclosed).
        """
        from regimerain.features.schema import SYNOPTIC
        from regimerain.gate.gate import OFF
        if self.gate is None:
            return np.ones(len(out), bool)
        st = self.static if static is None else static
        zone = st["zone"].values[out.lat_idx, out.lon_idx].astype(int)
        geo = st["geo"].values[out.lat_idx, out.lon_idx].astype(int)
        regime = np.array(SYNOPTIC)[out[REGIME_PROB_FEATURES].to_numpy().argmax(axis=1)]
        g = self.gate.set_index(["zone", "geo", "lead", "regime"])["gate"]
        g = g[~g.index.duplicated()]
        keys = pd.MultiIndex.from_arrays([zone, geo, out.lead.astype(int).to_numpy(), regime])
        own = g.reindex(keys).to_numpy()
        keys_all = pd.MultiIndex.from_arrays([zone, geo, out.lead.astype(int).to_numpy(), np.full(len(out), "all")])
        fallback = g.reindex(keys_all).to_numpy()
        gate = np.where(np.isnan(own.astype(float)), fallback.astype(float), own.astype(float))
        return ~(gate == OFF)


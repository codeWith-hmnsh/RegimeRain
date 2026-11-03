"""`regimerain report`: pool all DONE folds -> verification_report.json, summary.md, plots (MODEL_SPEC 13.5).

Every number shown anywhere comes from these files (NFR-2). Negative results are listed, not hidden (NFR-4).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from regimerain.config import by_mode, config_sha256
from regimerain.features.schema import SYNOPTIC
from regimerain.folds import check_poolable, fold_dir
from regimerain.verify.bootstrap import block_bootstrap_delta
from regimerain.verify.metrics import fmt_threshold, fss, scores_from_stats
from regimerain.verify.stats import cell_slice_stats, fss_day_stats

VARIANT_ORDER = ["raw", "A", "B", "C", "CM"]
VARIANT_NAMES = {"raw": "Raw NWP", "A": "Global QM", "B": "Regime-aware QM", "C": "Error-regime QM", "CM": "Conditional mean"}


def load_predictions(cfg: dict) -> pd.DataFrame:
    parts = [pd.read_parquet(fold_dir(cfg["paths"]["cache"], s) / "predictions.parquet").assign(fold=s)
             for s in cfg["seasons"]]
    return pd.concat(parts, ignore_index=True)


def day_stats(preds: pd.DataFrame, variants, shape, thresholds, windows, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (variant, lead, valid) sufficient statistics: cell slices and all-land FSS parts."""
    cell_rows, fss_rows = [], []
    groups = preds.groupby(["lead", "valid"], observed=True)
    for n, ((lead, valid), g) in enumerate(groups):
        iy, ix = g.lat_idx.to_numpy(), g.lon_idx.to_numpy()
        land = np.zeros(shape, bool); land[iy, ix] = True
        o = np.full(shape, np.nan); o[iy, ix] = g.o_rain.to_numpy()
        syn = np.full(shape, -1); syn[iy, ix] = g.y_synoptic.to_numpy()
        geo = np.full(shape, -1); geo[iy, ix] = g.geo.to_numpy()
        ys = g.y_synoptic.to_numpy()
        day_label = 2 if (ys == 2).mean() >= 0.05 else int(pd.Series(ys[ys != 2]).mode().iloc[0]) if (ys != 2).any() else 2
        base = {"lead": int(lead), "valid": valid, "fold": int(g.fold.iloc[0])}
        for v in variants:
            f = np.zeros(shape); f[iy, ix] = g[v].to_numpy()
            for r in cell_slice_stats(f, o, land, syn, geo, thresholds):
                cell_rows.append({**base, "variant": v, **r})
            fss_rows.append({**base, "variant": v, "day_label": day_label, **fss_day_stats(f, o, land, thresholds, windows)})
        if n % 200 == 0:
            log(f"report: day stats {n}/{groups.ngroups}")
    return pd.DataFrame(cell_rows), pd.DataFrame(fss_rows)


def _slice_name(x):
    return x if x == "all" else SYNOPTIC[int(x)]


def score_table(cells: pd.DataFrame, fsd: pd.DataFrame, thresholds, windows, by_fold: bool) -> list[dict]:
    rows = []
    keys = ["variant", "lead"] + (["fold"] if by_fold else [])
    c = cells[cells.geo.astype(str) == "all"]
    stat_cols = [col for col in c.columns if col in ("n", "sse") or col[:2] in ("H_", "M_", "FA", "CN")]
    agg = c.groupby(keys + ["synoptic"], observed=True)[stat_cols].sum().reset_index()
    fcols = [col for col in fsd.columns if col.startswith("fss")]
    f_all = fsd.groupby(keys, observed=True)[fcols].sum().reset_index().assign(synoptic="all")
    f_syn = fsd.groupby(keys + ["day_label"], observed=True)[fcols].sum().reset_index().rename(columns={"day_label": "synoptic"})
    fagg = pd.concat([f_all, f_syn], ignore_index=True)
    fagg["synoptic"] = fagg["synoptic"].astype(str)
    agg["synoptic"] = agg["synoptic"].astype(str)
    for _, r in agg.iterrows():
        fr = fagg[(fagg.variant == r.variant) & (fagg.lead == r.lead) & (fagg.synoptic == r.synoptic)
                  & ((fagg.fold == r.fold) if by_fold else True)]
        for t in thresholds:
            k = fmt_threshold(t)
            s = scores_from_stats(r, t)
            fs = {str(w): (fss(fr[f"fssnum_{k}_{w}"].sum(), fr[f"fssden_{k}_{w}"].sum()) if len(fr) else None)
                  for w in windows}
            rows.append({"variant": r.variant, "fold": int(r.fold) if by_fold else "pooled", "lead": int(r.lead),
                         "synoptic": _slice_name(r.synoptic), "geo": "all", "threshold": float(t),
                         "n_cell_days": int(r.n), "n_events_obs": int(r[f"H_{k}"] + r[f"M_{k}"]), **s, "fss": fs})
    return rows


def deltas(cells: pd.DataFrame, variants, leads, cfg) -> list[dict]:
    """Bootstrap CIs for corrected-minus-raw (and B-minus-A) on pooled, all-cell statistics."""
    c = cells[(cells.geo.astype(str) == "all") & (cells.synoptic.astype(str) == "all")]
    stat_cols = [col for col in c.columns if col in ("n", "sse") or col[:2] in ("H_", "M_", "FA", "CN")]
    B = int(by_mode(cfg, cfg["verify"]["bootstrap_B"]))
    block = int(cfg["verify"]["bootstrap_block_days"])
    pairs = [(v, "raw") for v in variants if v != "raw"] + ([("B", "A")] if {"A", "B"} <= set(variants) else [])
    out = []
    for lead in leads:
        for v, ref in pairs:
            a = c[(c.variant == ref) & (c.lead == lead)].sort_values("valid")[stat_cols].reset_index(drop=True)
            b = c[(c.variant == v) & (c.lead == lead)].sort_values("valid")[stat_cols].reset_index(drop=True)
            for t in (64.5, 115.6):
                for metric in ("ets", "csi", "pod", "far"):
                    fn = lambda s, t=t, m=metric: scores_from_stats(s, t)[m]
                    res = block_bootstrap_delta(a, b, fn, B=B, block=block, seed=int(cfg["seed"]))
                    out.append({"variant": v, "vs": ref, "lead": int(lead), "synoptic": "all", "threshold": t,
                                "metric": metric, **res})
            res = block_bootstrap_delta(a, b, lambda s: scores_from_stats(s, 64.5)["rmse"], B=B, block=block,
                                        seed=int(cfg["seed"]))
            out.append({"variant": v, "vs": ref, "lead": int(lead), "synoptic": "all", "threshold": None,
                        "metric": "rmse", **res})
    return out


def _fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def summary_markdown(report: dict) -> str:
    rows = [r for r in report["rows"] if r["fold"] == "pooled" and r["synoptic"] == "all" and r["threshold"] in (64.5, 115.6)]
    lines = [f"# Verification summary — backtest `{report['backtest_id']}`", "",
             f"Seasons {report['seasons']} (leave-one-monsoon-out), leads {report['leads']}, truth `{report['truth_source']}`, "
             f"forecast `{report['forecast_source']}`. Labels climatology: {report.get('label_climatology')}. "
             f"Static layers: {report.get('static_mode')}.", "",
             "| Lead | Threshold | Variant | RMSE | POD | FAR | CSI | ETS | Freq. bias | FSS 3 | FSS 5 | Obs. events |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {v: i for i, v in enumerate(VARIANT_ORDER)}
    for r in sorted(rows, key=lambda r: (r["lead"], r["threshold"], order.get(r["variant"], 9))):
        lines.append(f"| D{r['lead']} | ≥{r['threshold']:g} | {VARIANT_NAMES.get(r['variant'], r['variant'])} | "
                     f"{_fmt(r['rmse'], 2)} | {_fmt(r['pod'])} | {_fmt(r['far'])} | {_fmt(r['csi'])} | {_fmt(r['ets'])} | "
                     f"{_fmt(r['freq_bias'], 2)} | {_fmt(r['fss'].get('3'))} | {_fmt(r['fss'].get('5'))} | {r['n_events_obs']:,} |")
    lines += ["", "## Differences with 90% bootstrap intervals", "",
              "| Lead | Comparison | Metric | Threshold | Δ | 90% CI | Significant |", "|---|---|---|---|---|---|---|"]
    for d in report["deltas"]:
        thr = "–" if d["threshold"] is None else f"≥{d['threshold']:g}"
        lines.append(f"| D{d['lead']} | {VARIANT_NAMES.get(d['variant'])} vs {VARIANT_NAMES.get(d['vs'])} | "
                     f"{d['metric'].upper()} | {thr} | {_fmt(d['delta'], 4)} | [{_fmt(d['ci'][0], 4)}, {_fmt(d['ci'][1], 4)}] | "
                     f"{'yes' if d['significant'] else 'no'} |")
    worse = [d for d in report["deltas"] if d["significant"] and d["vs"] == "raw"
             and ((d["metric"] in ("ets", "csi", "pod") and d["delta"] < 0) or (d["metric"] in ("far", "rmse") and d["delta"] > 0))]
    lines += ["", "## Where correction did not help (significant, vs raw)", ""]
    lines += [f"- D{d['lead']} {VARIANT_NAMES.get(d['variant'])}: {d['metric'].upper()}"
              f"{'' if d['threshold'] is None else ' ≥' + format(d['threshold'], 'g')} worse by {abs(d['delta']):.4f}"
              for d in worse] or ["- none"]
    lines += ["", "## Regime classifier (test seasons)", "", "| Test season | Accuracy | Macro-F1 | Recall active / break / depression / normal |",
              "|---|---|---|---|"]
    for c in report["classifier"]:
        rc = c["test_metrics"]["per_class_recall"]
        lines.append(f"| {c['test']} | {_fmt(c['test_metrics']['accuracy'])} | {_fmt(c['test_metrics']['macro_f1'])} | "
                     + " / ".join(_fmt(rc[k]) for k in SYNOPTIC) + " |")
    return "\n".join(lines) + "\n"


def plots(report: dict, preds: pd.DataFrame, shape, out: Path, variants) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    rows = [r for r in report["rows"] if r["fold"] == "pooled" and r["synoptic"] == "all" and r["threshold"] == 64.5]
    leads = sorted({r["lead"] for r in rows})
    metrics = ["pod", "far", "csi", "ets"]
    fig, axes = plt.subplots(1, len(leads), figsize=(5 * len(leads), 3.6), squeeze=False)
    for ax, lead in zip(axes[0], leads):
        x = np.arange(len(metrics)); w = 0.8 / len(variants)
        for i, v in enumerate(variants):
            r = next((r for r in rows if r["lead"] == lead and r["variant"] == v), None)
            if r:
                ax.bar(x + i * w, [r[m] if r[m] is not None else np.nan for m in metrics], w, label=VARIANT_NAMES[v])
        ax.set_xticks(x + w * (len(variants) - 1) / 2, [m.upper() for m in metrics]); ax.set_title(f"Day {lead}, rain ≥ 64.5 mm")
        ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout(); p = out / "metrics_heavy.png"; fig.savefig(p, dpi=150); plt.close(fig); files.append(p.name)

    fig, axes = plt.subplots(1, len(leads), figsize=(5 * len(leads), 3.4), squeeze=False)
    for ax, lead in zip(axes[0], leads):
        for v in variants:
            r = next((r for r in rows if r["lead"] == lead and r["variant"] == v), None)
            if r:
                ws = sorted(r["fss"], key=int)
                ax.plot([int(k) * 25 for k in ws], [r["fss"][k] for k in ws], marker="o", label=VARIANT_NAMES[v])
        ax.set_xlabel("window (km, approx.)"); ax.set_ylabel("FSS"); ax.set_title(f"Day {lead}, ≥ 64.5 mm"); ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout(); p = out / "fss_vs_scale.png"; fig.savefig(p, dpi=150); plt.close(fig); files.append(p.name)

    lead0 = leads[0]
    g = preds[preds.lead == lead0]
    wettest = g.groupby("valid").o_rain.apply(lambda s: (s >= 64.5).sum()).idxmax()
    d = g[g.valid == wettest]
    panels = [("o_rain", "Observed (IMD)")] + [(v, VARIANT_NAMES[v]) for v in variants]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.8))
    for ax, (col, title) in zip(np.atleast_1d(axes), panels):
        grid = np.full(shape, np.nan); grid[d.lat_idx.to_numpy(), d.lon_idx.to_numpy()] = d[col].to_numpy()
        im = ax.imshow(grid, origin="lower", vmin=0, vmax=150, cmap="YlGnBu")
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), shrink=0.8, label="mm/day")
    fig.suptitle(f"Day-{lead0} forecast for rain day starting {pd.Timestamp(wettest).date()} (most heavy-rain cells)", fontsize=10)
    p = out / "map_wettest_day.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); files.append(p.name)
    return files


def build_report(cfg: dict, log=print) -> Path:
    info = check_poolable(cfg["paths"]["cache"], cfg["seasons"])
    variants = [v for v in VARIANT_ORDER if v in next(iter(info.values())).get("variants", ["raw"])]
    preds = load_predictions(cfg)
    from regimerain.static.basic import load_static
    st = load_static(cfg)
    shape = (st.sizes["lat"], st.sizes["lon"])
    thresholds = cfg["verify"]["thresholds"]
    windows = cfg["verify"]["fss_windows"]
    cells, fsd = day_stats(preds, variants, shape, thresholds, windows, log)
    backtest_id = f"{time.strftime('%Y%m%d')}_{config_sha256(cfg)[:8]}"
    out = Path(cfg["paths"]["reports"]) / backtest_id
    out.mkdir(parents=True, exist_ok=True)
    leads = sorted(preds.lead.unique().tolist())
    classifier = [json.loads((fold_dir(cfg["paths"]["cache"], s) / "classifier.json").read_text()) for s in cfg["seasons"]]
    report = {
        "backtest_id": backtest_id, "config_sha256": config_sha256(cfg),
        "git_commit": next(iter(info.values())).get("git_commit"),
        "truth_source": cfg["sources"]["truth"], "forecast_source": "hres",
        "label_climatology": cfg["labels"].get("climatology", "imd"), "static_mode": st.attrs.get("mode", "full"),
        "seasons": list(cfg["seasons"]), "leads": leads, "variants": variants,
        "rows": score_table(cells, fsd, thresholds, windows, by_fold=False)
                + score_table(cells, fsd, thresholds, windows, by_fold=True),
        "deltas": deltas(cells, variants, leads, cfg),
        "classifier": classifier,
    }
    (out / "verification_report.json").write_text(json.dumps(report, indent=1, default=str))
    (out / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    report["plots"] = plots(report, preds, shape, out, variants)
    cells.astype({"synoptic": str, "geo": str}).to_parquet(out / "day_stats_cells.parquet", index=False)
    fsd.to_parquet(out / "day_stats_fss.parquet", index=False)
    log(f"report -> {out}")
    return out

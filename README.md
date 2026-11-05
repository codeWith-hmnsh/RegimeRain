# SIH26080 — Regime-Aware Rainfall Post-Processing

[![ci](https://github.com/codeWith-hmnsh/RegimeRain/actions/workflows/ci.yml/badge.svg)](https://github.com/codeWith-hmnsh/RegimeRain/actions/workflows/ci.yml)

**Live site:** https://codewith-hmnsh.github.io/RegimeRain/ shows a real held-out hindcast: 9 Aug 2019, Kerala/Karnataka floods. Every model behind it was trained without the 2019 season. `?run=hindcast-break` opens a break-monsoon day (9 Jul 2017), and `?run=sample` opens the synthetic sample.

**Build an AI/ML system that identifies the prevailing monsoon rainfall regime, then applies a regime-specific bias correction to raw NWP rainfall forecasts.**
Ministry of Earth Sciences (MoES) / NCMRWF · Software · Smart Automation.

Rainfall forecast errors are not uniform: they behave differently in active monsoon, break monsoon, depressions, orographic and coastal settings, and near western disturbances. One global bias-correction curve averages that structure away. We classify the regime first — from forecast-time information only, so it's deployable — then apply a correction fitted to that regime, and report the PS's own named scorecard (RMSE, ETS, CSI, POD, FAR, FSS) against raw NWP, honestly, under a leave-one-monsoon-out split.

**Most teams will detect and correct with one model. We tell the model what kind of day it's forecasting first — and we report all six metrics the problem statement actually asked for.**

## What is here

```
PRD.md                    product requirements — what and why (§1-20)
TECHNICAL_SPEC.md         TRD — grids, data contracts, modules, CLI, API, tests
docs/PS26080_Brief.md     the original problem-statement text, verbatim
docs/PPT_BRIEF.md         deck-ready narrative, claims, demo script, for the PPT team
docs/SOLUTION_EXPLAINED_80.md  full idea explainer: problem → solution → every model, with flowcharts
docs/BACKEND_BUILD_PLAN.md  run producers (mock / replay / model), web contract v2, API, live run
docs/MODEL_SPEC.md        model build guide: every model step by step, with code, config and tests (§23 = Colab, §24 = Kaggle)
notebooks/colab_pipeline.ipynb   Colab CPU notebook (primary): STAGE = data | backtest (resumable folds) | final
notebooks/kaggle_pipeline.ipynb  Kaggle alternative, same stages
```

Build state (5 Nov 2026):

| Part | State |
|---|---|
| `regimerain` ingest, basic static layers, labels, features, LOMO folds | Built |
| Regime classifier (M1), QM variants A (global) and B (regime-aware), backtest, verification (six metrics + block bootstrap) | Built; pilot v2 measured on 7 seasons × 5 leads (`docs/PILOT_RESULTS.md`) |
| Run exporter, mock producer, web contract v2 | Built (`regimerain/runs/`) |
| Web MVP (`mvp/web/`) | Built; live, default run is a real held-out hindcast |
| Heavy-rain probability model M4 (LightGBM + isotonic, cross-fitted inputs) | Built; measured: beats climatology at every lead (Day 1 ≥64.5 mm ROC-AUC 0.894, BSS 0.115) |
| Skill gate (zone × terrain units until district boundaries are licensed) | Built and tested (`regimerain/gate/`, explainer `docs/EXPLAINED_SKILL_GATE.md`) |
| Reasons behind each heavy-rain probability (TreeSHAP on M4, named places, Bulletin page) | Built and tested (`regimerain/explain/`, explainer `docs/EXPLAINED_REASONS.md`); live runs only |
| Error regimes and variant C (clusters of forecast errors, predicted from forecast-time features, one QM expert set per cluster) | Built and tested (`regimerain/errreg/`, explainer `docs/EXPLAINED_ERROR_REGIMES.md`); backtest only |
| Conditional mean, variant CM (Tweedie LightGBM on the M4 rows, expected rain ≥ 0) | Built and tested (`regimerain/exceed/model.py`); backtest only |
| Districts | Not built (boundary licence, PRD §20) |
| Replay runs: real held-out backtest days for the web (`regimerain run --source replay [--init DATE] [--publish NAME]`) | Built; two held-out days published (`mvp/web/public/run/hindcast-*`) |
| `regimerain serve`: read-only API + built web app on one port | Built and tested (`regimerain/api/app.py`) |
| `fit-final`: every model refit on all seasons → hashed model set (`regimerain/final.py`, `ModelSet` loader) | Built; model set `e0cae019e7f5` trained on Kaggle (dataset `ps26080-models`) |
| Live GFS run: today's GFS through the model set, skill gate applied (`regimerain run --source gfs [--init DATE]`) | Built and tested (`regimerain/ingest/gfs.py`, `regimerain/runs/live.py`, explainer `docs/EXPLAINED_LIVE_GFS.md`); needs a model set from Kaggle |
| Model-set run on an HRES init (`run --source hres`) | Not built (CLI exits 2 and names the plan section) |

Pilot v2 headline (7 monsoons 2016-2022, leave-one-monsoon-out, Day 1-5): quantile mapping significantly improves heavy-rain detection at every lead (Day 1, ≥64.5 mm: ETS 0.084 → 0.092, POD 0.152 → 0.183, frequency bias 0.83 → 1.01). The regime classifier now reaches 0.63-0.73 accuracy on unseen seasons. Regime-aware QM is still not clearly better than global QM; that is reported, not hidden. Details and every caveat: `docs/PILOT_RESULTS.md`.

`PRD.md` §18 has the build-phase order, `TECHNICAL_SPEC.md` §9 the CLI, `docs/MODEL_SPEC.md` §22 the model build checklist.

## Status

Selected 29 Sep 2026 alongside PS26081 (built separately by Mridul) to replace the team's two closed slots — PS26143 (Dark Transit, 500/500) and PS26167 (SatQuery, filled). Selection rationale and the live submission-count comparison against seven other candidates: `../extra/problem-statements/SLOT2_CANDIDATES.md`.

Built by the team excluding Mridul. Every algorithm in this plan (gradient-boosted trees, empirical quantile mapping) was chosen to be tabular-scale and not require a dedicated ML specialist — see `PRD.md` §18 and §20.

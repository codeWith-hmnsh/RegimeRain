# SIH26080 — Regime-Aware Rainfall Post-Processing

[![ci](https://github.com/codeWith-hmnsh/RegimeRain/actions/workflows/ci.yml/badge.svg)](https://github.com/codeWith-hmnsh/RegimeRain/actions/workflows/ci.yml)

**Live site:** https://codewith-hmnsh.github.io/RegimeRain/ (shows the clearly labelled synthetic sample run until the Kaggle backtest results are published).

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

Build state (2 Nov 2026):

| Part | State |
|---|---|
| `regimerain` ingest, basic static layers, labels, features, LOMO folds | Built |
| Regime classifier (M1), QM variants A (global) and B (regime-aware), backtest, verification (six metrics + block bootstrap) | Built; pilot v1 measured (`docs/PILOT_RESULTS.md`) |
| Run exporter, mock producer, web contract v2 | Built (`regimerain/runs/`) |
| Web MVP (`mvp/web/`) | Built; shows synthetic data until a replay run exists |
| Heavy-rain probability model M4 (LightGBM + isotonic, cross-fitted inputs) | Built and tested (`regimerain/exceed/`, explainer `docs/EXPLAINED_M4_HEAVY_RAIN.md`); Kaggle results pending |
| Skill gate (zone × terrain units until district boundaries are licensed) | Built and tested (`regimerain/gate/`, explainer `docs/EXPLAINED_SKILL_GATE.md`) |
| Error regimes (variant C), explanations, districts, conditional mean | Not built |
| Replay runs: real held-out backtest days for the web (`regimerain run --source replay [--init DATE] [--publish NAME]`) | Built and tested (`regimerain/runs/replay.py`); needs the Kaggle backtest cache |
| `regimerain serve`: read-only API + built web app on one port | Built and tested (`regimerain/api/app.py`) |
| `fit-final`: every model refit on all seasons → hashed model set (`regimerain/final.py`, `ModelSet` loader) | Built and tested |
| HRES / live GFS runs (feature building for one new init, GFS ingest) | Not built (CLI exits 2 and names the plan section) |

Pilot v1 headline: quantile mapping improves heavy-rain detection (Day 1, ≥64.5 mm: POD 0.171 → 0.204), but regime-aware QM does not yet beat global QM, because the classifier trains on one season per fold. Pilot v2 (`config/v2.yaml`, 7 seasons) is the next run.

`PRD.md` §18 has the build-phase order, `TECHNICAL_SPEC.md` §9 the CLI, `docs/MODEL_SPEC.md` §22 the model build checklist.

## Status

Selected 29 Sep 2026 alongside PS26081 (built separately by Mridul) to replace the team's two closed slots — PS26143 (Dark Transit, 500/500) and PS26167 (SatQuery, filled). Selection rationale and the live submission-count comparison against seven other candidates: `../extra/problem-statements/SLOT2_CANDIDATES.md`.

Built by the team excluding Mridul. Every algorithm in this plan (gradient-boosted trees, empirical quantile mapping) was chosen to be tabular-scale and not require a dedicated ML specialist — see `PRD.md` §18 and §20.

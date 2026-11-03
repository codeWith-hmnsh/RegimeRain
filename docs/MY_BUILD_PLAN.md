# My build plan: RegimeRain (SIH26080)

Goal: a complete, working, deployed project that I can explain line by line in an interview.
Pace: 2–3 hours a day alongside the semester, about 8 weeks. Tick boxes as I go and commit this file.

Rule for myself: I write the code (or type it after understanding it). Every number on the resume comes from my own
Kaggle runs, copied from the report files, never from memory.

---

## Step 0: accounts and logins (one time)

| Where | What to do | Why |
|---|---|---|
| GitHub | Log in. Create a **new public repo** named `regimerain` (no README, no .gitignore) | My code, my commits |
| Git on this laptop | `git config --global user.name "My Name"` (email is already set) | Commits show my name |
| Kaggle (kaggle.com) | Sign up / log in. *Settings → Phone verification* | Without it, notebooks get no internet |
| Kaggle API key (optional) | *Settings → API → Create New Token* → saves `kaggle.json` | Upload datasets from the command line instead of the browser |
| BOM MJO file (optional, later) | Open http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt in a browser, save it | BOM blocks cloud servers; MJO features are neutral without it |

## Step 1: my repo (Day 1)

```
cd C:\samanway-project\080
git init -b main
git add .
git commit -m "Import SIH26080 team baseline (docs, regimerain package, web MVP)"
git remote add origin https://github.com/codeWith-hmnsh/RegimeRain.git
git push -u origin main
```

The first commit message says it is the team baseline: honest, and everything after it is mine.
Then set `REPO` in `notebooks/kaggle_pipeline.ipynb` (cell 2) to my repo URL and commit.

Local check:
```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[data,ml,dev,api]"
pytest -q            # expect 94 passed
```

- [x] GitHub repo created: https://github.com/codeWith-hmnsh/RegimeRain
- [x] `pytest` shows 96 passed on my laptop (`.venv`, Python 3.12, lightgbm 4.7.0, xarray 2026.7.0)
- [ ] Read `docs/SOLUTION_EXPLAINED_80.md` §1–§8; wrote my own 5-line summary of the idea

## Step 2: IMD rainfall to Kaggle (week 1)

IMD (the truth data) may be blocked from Kaggle's servers, so it is downloaded on the laptop and uploaded once.

- [x] `data/kaggle_upload/imd/rain/1981.grd … 2022.grd` present (42 files, 1.0 GB; 2019 read back: 129 × 135 grid, JJAS mean 7.9 mm/day)
- [x] Kaggle → *Datasets → New Dataset* → **`ps26080-imd`**, private (uploaded as `imd.zip`, 93 MB; the uncompressed folder stalled in the browser, an Incognito window worked)
- [x] MJO index downloaded on the laptop: `data/kaggle_upload/mjo/rmm.74toRealtime.txt` (2.6 MB, 1974-06-01 → 2024-02-24).
      Kaggle run 1 failed with `HTTP Error 403: Forbidden` from BOM, which blocks cloud servers.
- [x] Kaggle → *New Dataset* → upload `rmm.74toRealtime.txt` → name **`ps26080-mjo`**, private; attach it to the notebook
      (the code finds it under `/kaggle/input` by name, `regimerain/ingest/mjo.py`)

## Step 3: first training run on Kaggle (week 2)

Notebook: upload `notebooks/kaggle_pipeline.ipynb` to Kaggle (*Create → New Notebook → File → Import*).
Settings every time: *Accelerator: None (CPU)*, *Internet: On*. Long runs: *Save Version → Save & Run All*.

| Run | Set in cell 2 | Attach | After it finishes, save output as dataset |
|---|---|---|---|
| 1 | `STAGE="data"` | `ps26080-imd` | `ps26080-table` |
| 2a | `STAGE="backtest"`, `FOLDS="2016,2017,2018"`, `COMMIT=<sha>` | `ps26080-imd`, `ps26080-table` | `ps26080-folds-a` |
| 2b | `FOLDS="2019,2020"` | same | `ps26080-folds-b` |
| 2c | `FOLDS="2021,2022"` | same | `ps26080-folds-c` |
| 3 | `STAGE="final"` | all of the above | `ps26080-models` |

Note: `final` runs `report`, then `fit-final`. `fit-final` is not built yet (CLI exits 2). Until I build it
(Step 6), the report is the useful output of run 3.

What gets trained here: the regime classifier (LightGBM, 4 classes) and the quantile-mapping correctors, once per
held-out season.

- [ ] Run 1 done, `selftest --quick` passed in the log
- [ ] Runs 2a–2c done (all 7 folds have `DONE`)
- [ ] Run 3: `verification_report.json` downloaded; numbers copied into `docs/PILOT_RESULTS.md` (v2 section)
- [ ] I can explain: what is leave-one-monsoon-out, and why not a random split?

## Step 4: my model, heavy-rain probability (weeks 3–4)

Spec: `docs/MODEL_SPEC.md` §12. Code: `regimerain/exceed/`, `regimerain/classify/crossfit.py`. Explainer: `docs/EXPLAINED_M4_HEAVY_RAIN.md`.

- [x] LightGBM binary model for ≥ 64.5 and ≥ 115.6 mm/day, features from §12.1
- [x] Isotonic calibration on the inner-validation season only (never the test season)
- [x] `P(very heavy) ≤ P(heavy)` enforced; test for it
- [x] Reliability diagram + Brier score in the report (plus BSS, ROC/PR-AUC, POD/FAR at 0.3 and 0.5); cross-fitted inputs with a leakage test
- [ ] Trained on Kaggle through the backtest; numbers in `PILOT_RESULTS.md`

## Step 5: skill gate (week 5)

Spec: `docs/MODEL_SPEC.md` §14. Per region × regime × lead: use the correction only where it helped.

- [x] `regimerain/gate/gate.py` + `test_gate_rules` (units: zone × terrain until districts are licensed)
- [x] Gate table in the report (`skill_gate.parquet`, map, summary section)

## Step 6: real results on the website (week 6)

Spec: `docs/BACKEND_BUILD_PLAN.md` B2–B4.

- [ ] `fit-final` (MODEL_SPEC §17)
- [x] Replay producer: real held-out days (a depression day, a break day, an active day) → `mvp/web` (code done; run it once the Kaggle backtest cache is downloaded)
- [ ] `regimerain serve` (FastAPI) serves the API + built web app
- [ ] The synthetic banner is gone for real runs

## Step 7: launch (week 7)

- [ ] Deployed (Render / Vercel); link at the top of README
- [ ] GitHub Actions runs `pytest` on every push; badge in README
- [ ] 2-minute demo video (problem → map → scorecard → honest limits)
- [ ] README rewritten: one-line problem, diagram, results table from my report

## Step 8: interview prep (week 8)

- [ ] Explain the whole project in 2 minutes without notes
- [ ] Answer: why LightGBM, not a neural network? Why can quantile mapping worsen RMSE?
      How did I prevent leakage? What did NOT work, and why?
- [ ] Resume bullets written with my real numbers

---

## Resume bullet template (fill X/Y only from my report)

> **RegimeRain — regime-aware monsoon rainfall correction** · Python, LightGBM, xarray, FastAPI, React · [live] · [GitHub]
> - Built a pipeline that classifies the monsoon regime and applies regime-specific quantile-mapping correction to ECMWF forecasts over ~4,900 Indian grid cells (SIH 2026, team; my part: …).
> - Trained a calibrated heavy-rain probability model (LightGBM + isotonic); ≥64.5 mm detection POD X → Y, ETS X → Y under leave-one-monsoon-out validation.
> - Leakage-safe evaluation (fold-isolation tests, block-bootstrap CIs) and a React + MapLibre dashboard of raw vs corrected forecasts.

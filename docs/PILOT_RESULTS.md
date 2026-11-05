# Pilot results

Every number here is copied from a report file; the source report is named in each section. Do not
retype numbers into slides from memory: copy them from `summary.md` of the named backtest (NFR-2).

## Pilot v2: backtest `20261105_45e44153` (5 Nov 2026, Kaggle)

Setup: IFS HRES (WeatherBench2) vs IMD 0.25° gridded rain, JJAS 2016-2022 (7 seasons), leave-one-monsoon-out,
leads Day 1 to Day 5, 4,964 land cells, `--mode fast`, variants raw / A (global QM) / B (regime-aware QM) /
C (error-regime QM) / CM (conditional mean). Code commit `11aeaf9`, config sha `45e44153…`. Final model set
`e0cae019e7f5`. Source: `reports/20261105_45e44153/summary.md` (Kaggle datasets `ps26080-folds`, `ps26080-models`).
Simplifications still in place: approximate zones, no terrain classes (`static: basic`).

**What it shows**
- Quantile mapping improves heavy-rain (≥ 64.5 mm) detection at every lead, and the 90% block-bootstrap CI
  excludes 0 each time. ETS for global QM: D1 0.084 -> 0.092, D3 0.070 -> 0.080, D5 0.064 -> 0.071.
  Day 1 POD 0.152 -> 0.183, frequency bias 0.83 -> 1.01, FSS (5 cells) 0.345 -> 0.364.
- RMSE: Day 1 is worse after QM (18.13 -> 18.54, significant), the known cost of distribution matching.
  Day 3 to Day 5 RMSE improves (Day 3 19.17 -> 18.91, significant). Day 2 shows no significant change.
- Regime-aware QM (B) is still not clearly better than global QM (A). ETS difference is not significant at
  Day 1-3; at Day 4-5 B is significantly better by only +0.0007 / +0.0011 ETS, and its RMSE is
  significantly worse at Day 2-5 (+0.05 to +0.16). Reported as it is.
- Regime classifier improved a lot over v1 (one training season per fold then, five now): test accuracy
  0.63-0.73 per season, macro-F1 0.29-0.54 (v1: ~0.43 / ~0.2). Break-day recall is still weak in several seasons.
- Error-regime QM (C): heavy-rain ETS about the same as A; best RMSE of the QM variants at Day 3-5
  (Day 4: -0.34 vs raw, A: -0.28). Predicted error cluster agrees with the true one above the majority
  baseline in every season (e.g. 2018: 0.40 vs 0.17), but the clusters are weak (silhouette 0.07-0.08).
- Conditional mean (CM) has by far the lowest RMSE (Day 1: 14.20 vs raw 18.13) but almost never forecasts
  heavy rain (Day 1 POD 0.033). It is a mean forecast, not a warning forecast.
- Heavy-rain probability model (M4) beats climatology at every lead and threshold. Day 1 ≥ 64.5 mm:
  BSS 0.115, ROC-AUC 0.894; ≥ 115.6 mm: ROC-AUC 0.933. Probabilities are low, so a 0.5 alert level
  catches little (POD 0.022); alert levels need calibrating to the user's cost of a miss.
- Skill gate (zone × terrain unit × lead, all regimes): ON 9, NEUTRAL 9, OFF 2. Both OFF cells are
  east & north-east plains at Day 1-2, where the live product serves raw NWP, flagged.
- Run cost on Kaggle (4 CPU cores): 21-27 min per fold (peak 12.1 GB RAM), `report` 9.7 min,
  `fit-final` 11.1 min.

**On the web:** two held-out days are published as replay runs: 9 Aug 2019 (depression; the Kerala/Karnataka
floods; held-out season 2019) and 9 Jul 2017 (break). Every model behind each run was trained without that season.

## Pilot v1: backtest `20260929_d47ede94` (29 Sep 2026, Kaggle)

Setup: IFS HRES (WeatherBench2) vs IMD 0.25° gridded rain, JJAS 2019-2021, leave-one-monsoon-out,
leads Day 1 and Day 3, 4,964 land cells, 3.6 M cell-days. Simplifications: active/break climatology from
the 3 seasons themselves, approximate zones, no terrain classes, no MJO file, variants raw / A (global QM) /
B (regime-aware QM).

**What it shows**
- Quantile mapping improves heavy-rain detection significantly (90% block-bootstrap CIs exclude 0).
  Day 1, >= 64.5 mm: POD 0.171 -> 0.204, ETS 0.095 -> 0.104, frequency bias 0.85 -> 1.00.
  Day 3, >= 64.5 mm: ETS 0.077 -> 0.086, FAR 0.840 -> 0.828, RMSE 20.15 -> 19.38.
- Day-1 RMSE got worse (18.61 -> 18.95 for B, -> 19.10 for A): the expected cost of distribution matching
  (PRD section 17.4). Reported, not hidden.
- Regime-aware QM is not yet better than global QM. The regime classifier is weak (accuracy ~0.43,
  macro-F1 ~0.2, break and depression recall ~0) because with 3 seasons each fold trains the classifier
  on ONE season. The mixture then falls back to the global correction.
- Case, 9 Aug 2019 (Day 1): the Kerala/Karnataka extreme rain of the August 2019 floods is badly
  under-forecast by raw NWP; correction adds intensity but cannot move misplaced rain (displacement
  correction is on the roadmap).

**Next (v2):** all 7 seasons (classifier trains on 5), IMD 1981-2015 climatology for labels, new
features `f_cmz_rain` / `f_cmz_wetfrac` (forecast core-monsoon-zone rain), MJO file if available.
Config: `config/v2.yaml`.

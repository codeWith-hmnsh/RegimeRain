# Pilot results

Every number here is copied from a report file; the source report is named in each section. Do not
retype numbers into slides from memory: copy them from `summary.md` of the named backtest (NFR-2).

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

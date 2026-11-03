# Regime-Aware Rainfall Post-Processing — Technical Requirements Document (TRD)

**Revision:** B (supersedes A) · **Date:** 29 September 2026

Engineering counterpart to `PRD.md`. This document covers the system **how**: grids, time conventions, data contracts field by field, module design, configuration, numerical rules, CLI, API schemas, sizing and tests. The model-level **how to build it**, with code, is in `docs/MODEL_SPEC.md`. No MVP exists yet; every number here is a design target, not a measurement. Measured numbers go into `reports/<backtest_id>/verification_report.json` and are the only numbers quoted anywhere.

Package name: **`regimerain`** (matching the `../143/darktransit` convention).

### Revision B changes

| Area | Rev A | Rev B |
|---|---|---|
| Rain day | UTC calendar day (00–24 UTC) | **IMD rain day, 03–03 UTC**, because IMD gridded rainfall is now the primary truth |
| Truth | CHIRPS default | **IMD 0.25° primary**, CHIRPS fallback; no fine-grid truth in the live product |
| Regime labels | One categorical with precedence | **Two fields**: `synoptic` ∈ {active, break, depression, normal} and static `geo` ∈ {orographic, coastal, plains} |
| QM | Per (regime, season); top-decile ratio extrapolation | Experts per (synoptic, geo, zone, lead); **GPD tail**; **shrinkage**; **mixture** blend |
| Thresholds | 64.5 / 124.5 | **64.5 / 115.6** (plus 2.5, 15.6 for metrics) |
| New modules | — | `errreg` (error regimes), `gate` (skill gate), `explain` |
| Depression radius | 300 km (proposed) | **500 km** (configurable; recorded in manifest) |

---

## 1. Scope and traceability

| PRD requirement | Module(s) | MODEL_SPEC § | Test(s) |
|---|---|---|---|
| FR-1 synoptic regime probabilities | `features`, `classify` | §7, §9 | `test_leakage`, `test_classifier_probs_sum_to_one` |
| FR-2 geographic class | `static` | §5 | `test_geo_classes_cover_land` |
| FR-3 mixture QM correction | `correct` | §11 | `test_qm_identity`, `test_qm_monotone`, `test_gpd_tail_extends`, `test_shrinkage_weights` |
| FR-4 exceedance probabilities | `exceed` | §12 | `test_exceed_monotone`, `test_calibration_fold_isolation` |
| FR-5 district product | `aggregate` | §16 | `test_district_weights_sum_to_one` |
| FR-6 verification | `verify` | §13 | `test_metrics_toy_tables`, `test_fss_known_cases`, `test_bootstrap_ci` |
| FR-7 live mode | `live` | §18 | `test_live_smoke` (network-marked) |
| FR-8 map toggles | `web` | — | manual + `test_no_hardcoded_numbers` |
| FR-9 traceability | all | §17 | `test_no_hardcoded_numbers`, `test_manifest_complete` |
| FR-10 error regimes | `errreg` | §10 | `test_errreg_fold_internal` |
| FR-11 skill gate | `gate` | §14 | `test_gate_rules` |
| FR-12 explanations | `explain` | §15 | `test_reason_top3` |
| FR-13 conditional mean | `condmean` | §12.5 | `test_condmean_nonnegative` |

---

## 2. Grids, domains, time and units

### 2.1 Spatial

| Item | Value |
|---|---|
| **Working grid** | 0.25° regular lat/lon, identical to the IMD gridded rainfall grid |
| **Rain domain** | 6.5°N–38.5°N, 66.5°E–100.0°E → 129 lat × 135 lon = 17,415 cells; ~4,900 land cells inside India (IMD mask) |
| **Dynamics domain** (for regime features) | 0°N–40°N, 40°E–110°E, 0.25°. Needed because the low-level jet (Arabian Sea) and Bay of Bengal lows lie partly outside the rain domain |
| **Latitude order** | Ascending (south → north) in every internal array. Flip on ingest if the source is descending |
| **CRS** | WGS84 (EPSG:4326) everywhere. The **only** reprojection is to EPSG:6933 (equal-area) inside `aggregate` to compute overlap areas |
| **Land mask** | IMD grid valid-data mask (cells with non-missing IMD rain on ≥ 95% of days) |
| **Zones** | IMD's four homogeneous monsoon regions (Northwest, Central, South Peninsula, East & Northeast), rasterised onto the working grid from a state → zone table (MODEL_SPEC §5.4) |

### 2.2 Time

| Item | Value |
|---|---|
| **Rain day** | 24 h ending **03 UTC** (08:30 IST). A rain day is labelled by its **start date** D, covering D 03 UTC → D+1 03 UTC. This is the single convention in all artefacts |
| **Initialisation** | 00 UTC runs only |
| **Lead days** | Day L ∈ {1..5} covers forecast hours [3 + 24(L−1), 27 + 24(L−1)] |
| **Training seasons** | JJAS 2016–2022: valid rain days 1 Jun – 30 Sep; the init date is valid date − (L−1) days |
| **Climatology window** | 1981–2015 (active/break standardisation only; outside the evaluation window) |
| **HRES accumulation** | WeatherBench2 HRES leads are 6-hourly. Rain-day totals are computed by linear interpolation of **cumulative** precipitation to the 03 UTC boundaries (MODEL_SPEC §4.2). Documented in `docs/units.md` |
| **GFS accumulation** | 3-hourly; exact 03 UTC boundaries (MODEL_SPEC §18.2) |
| **IMD date convention** | Verified empirically in Phase 0 by lag correlation against the forecast (MODEL_SPEC §4.3); stored in config as `imd_date_offset_days` |
| **Timestamps** | ISO-8601 UTC in every artefact |

### 2.3 Units

Rainfall mm/day (float32, ≥ 0). Winds m/s. Specific humidity kg/kg. Pressure Pa internally, hPa in display. Vorticity s⁻¹. Moisture-flux convergence kg m⁻² s⁻¹ (column) or mm/day equivalent. Distances km. Conversion happens **once**, in `ingest`; no unit conversion in any downstream module.

---

## 3. Data contracts

### 3.1 Raw inputs (`data/raw/`)

| File | Contents | Dims | Units |
|---|---|---|---|
| `hres/rain_{year}.zarr` | Rain-day forecast precipitation, rain domain | `init, lead, lat, lon` | mm/day |
| `hres/dyn_{year}.zarr` | Rain-day-mean dynamics on the dynamics domain: `u850`, `v850`, `mslp`, `w500`, and the column integrals `pw` (mm), `ivtx`, `ivty` (kg m⁻¹ s⁻¹) computed at ingest from q, u, v at 300–1000 hPa, so pressure levels are never stored | `init, lead, lat, lon` | SI |
| `imd/rain_{year}.nc` | IMD gridded rain | `time, lat, lon` | mm/day |
| `chirps/rain_{year}.nc` | CHIRPS regridded (conservative) to 0.25° | `time, lat, lon` | mm/day |
| `tracks/ibtracs_ni.parquet` | `sid, time, lat, lon, grade, wind_kt` | row per 3/6-hourly fix | — |
| `tracks/lps.parquet` | Same schema from the LPS dataset or our tracker | row per fix | — |
| `mjo/rmm.parquet` | `date, rmm1, rmm2, phase, amplitude` | row per day | — |
| `gfs/{init}/...` | Live GRIB2 subset (see MODEL_SPEC §18) | — | — |

### 3.2 Static layers (`data/static/static.nc`)

| Variable | Dims | Description |
|---|---|---|
| `land` | `lat, lon` | bool land mask |
| `elev_mean`, `elev_std` | `lat, lon` | m, from DEM aggregated to 0.25° |
| `slope_mean` | `lat, lon` | degrees |
| `windward` | `lat, lon` | mean positive upslope component along 250° flow, dimensionless |
| `dist_coast_km` | `lat, lon` | km to the nearest coastline |
| `geo` | `lat, lon` | int8: 0 plains, 1 coastal, 2 orographic |
| `zone` | `lat, lon` | int8: 0 NW, 1 Central, 2 South Peninsula, 3 East & NE |
| `district_weights` | sparse table in `district_weights.parquet`: `district_id, lat_idx, lon_idx, weight` | — |

### 3.3 Labelled and feature tables (`data/table/lead={L}/season={Y}/part.parquet`)

One row per (init date, lead, land cell). Column contract:

| Column | Type | Source | Allowed in features? |
|---|---|---|---|
| `init`, `valid` | date | index | no |
| `lead` | int8 | index | **yes** |
| `lat_idx`, `lon_idx` | int16 | index | no |
| `lat`, `lon` | float32 | static | yes |
| `geo`, `zone` | int8 | static | yes |
| `elev_mean`, `elev_std`, `slope_mean`, `windward`, `dist_coast_km` | float32 | static | yes |
| `f_rain` | float32 | forecast | yes |
| `f_rain_nbr3_mean`, `f_rain_nbr3_max`, `f_rain_nbr5_mean`, `f_rain_nbr5_max` | float32 | forecast | yes |
| `f_u850`, `f_v850`, `f_ws850` | float32 | forecast | yes |
| `f_llj_index`, `f_trough_lat`, `f_bob_vort_max` | float32 | forecast (domain indices, same for all cells on that init/lead) | yes |
| `f_vort850_max500km`, `f_dist_mslp_min_km`, `f_mslp_anom` | float32 | forecast | yes |
| `f_pw`, `f_ivt`, `f_imfc` | float32 | forecast | yes |
| `f_w500` | float32 | forecast | yes |
| `mjo_rmm1`, `mjo_rmm2`, `mjo_amp`, `mjo_phase` | float32/int8 | MJO on the init date | yes |
| `doy_sin`, `doy_cos` | float32 | valid date | yes |
| `o_rain` | float32 | **truth** | **no** |
| `y_synoptic` | int8 | **label** (0 active, 1 break, 2 depression, 3 normal) | **no** |
| `y_err_regime` | int8 | **label**, fold-specific, written to the fold cache, not the base table | **no** |

**Leakage rule:** feature columns are exactly those prefixed `f_`, `mjo_`, `doy_` plus the static list above and `lead`. The feature builder selects by allow-list, not deny-list. `test_leakage` asserts that no column starting with `o_` or `y_` is in any model's `feature_name()`.

### 3.4 Model artefacts (`models/<model_set_id>/`)

```
models/<model_set_id>/
  config.yaml                      # frozen copy of the config used
  classifier/booster.txt           # LightGBM model (lead as feature) or booster_L{1..5}.txt
  classifier/temperature.json      # {"T": float}
  errreg/scaler.pkl, pca.pkl, kmeans.pkl, classifier.txt
  qm/experts.parquet               # one row per expert key: quantile arrays + GPD params + n
  exceed/heavy.txt, very_heavy.txt # LightGBM boosters
  exceed/iso_heavy.pkl, iso_very_heavy.pkl
  condmean/booster.txt             # optional
  gate/skill_gate.parquet
  feature_list.json                # exact ordered feature names per model
  hashes.json                      # sha256 of every file above
```

`qm/experts.parquet` columns: `variant` (A/B/C), `regime` (int or −1 for parent), `geo`, `zone` (−1 = all-India), `lead`, `n`, `n_days`, `f_q` (float32[1001]), `o_q` (float32[1001]), `f_dry_thr`, `tail_q`, `f_xi`, `f_sigma`, `o_xi`, `o_sigma`, `f_u`, `o_u`, `parent_key`, `w_shrink`.

### 3.5 Run artefacts (`runs/<run_id>/`)

`run_id` = `{init:%Y%m%d}T00Z_{source}_{model_set_id[:8]}`.

| File | Contents |
|---|---|
| `manifest.json` | See §3.7 |
| `regime_probs.nc` | `p(lead, lat, lon, synoptic)` float32, plus `p_err(lead, lat, lon, k)` if variant C is active |
| `corrected_rainfall.nc` | `raw`, `qm`, `cond_mean` (optional), all `(lead, lat, lon)` float32 |
| `exceedance_probs.nc` | `p_exceed(lead, lat, lon, threshold)` with threshold ∈ {64.5, 115.6} |
| `gate.nc` | `gate(lead, lat, lon)` int8: 1 ON, 0 NEUTRAL, −1 OFF; `served(lead, lat, lon)` = rainfall actually served |
| `district_table.csv/.json` | §3.8 |
| `explain.parquet` | `district_id, lead, rank, feature, contribution, phrase` |

### 3.6 Verification report schema (`reports/<backtest_id>/verification_report.json`)

```json
{
  "backtest_id": "string",
  "model_set_id": "string",
  "truth_source": "imd|chirps",
  "forecast_source": "hres",
  "seasons": [2016, 2017, 2018, 2019, 2020, 2021, 2022],
  "rows": [
    {
      "variant": "raw|A|B|C",
      "fold": "2019|pooled",
      "lead": 1,
      "synoptic": "all|active|break|depression|normal",
      "geo": "all|orographic|coastal|plains",
      "threshold": 64.5,
      "n_cell_days": 0,
      "n_events_obs": 0,
      "rmse": 0.0, "pod": 0.0, "far": 0.0, "csi": 0.0, "ets": 0.0,
      "fss": {"1": 0.0, "3": 0.0, "5": 0.0, "9": 0.0},
      "freq_bias": 0.0
    }
  ],
  "deltas": [
    {"variant": "B", "vs": "raw", "lead": 1, "synoptic": "all", "geo": "all",
     "threshold": 64.5, "metric": "ets", "delta": 0.0, "ci90": [0.0, 0.0], "significant": false}
  ],
  "classifier": {"lead": {"1": {"accuracy": 0.0, "macro_f1": 0.0, "brier": 0.0,
                  "confusion": [[0]], "per_class_recall": {"active": 0.0}}}},
  "exceedance": {"64.5": {"lead": {"1": {"brier": 0.0, "roc_auc": 0.0, "pr_auc": 0.0,
                  "reliability": {"bin_mean_p": [], "obs_freq": [], "count": []}}}}},
  "qm_sample_counts": [{"variant": "B", "regime": "depression", "geo": "plains", "zone": "Central", "lead": 1, "n": 0, "w_shrink": 0.0}],
  "error_regimes": {"k": 0, "silhouette": 0.0, "crosstab": [[0]]}
}
```

RMSE is computed on all land cell-days, independent of threshold; it is repeated in each threshold row for convenience. Metric definitions: PRD §17.1.

### 3.7 Manifest schema

```json
{
  "run_id": "", "created_utc": "", "git_commit": "", "config_sha256": "",
  "forecast_source": "gfs|hres", "forecast_init_utc": "",
  "truth_source": "imd|chirps", "truth_version": "",
  "model_set_id": "", "model_hashes": {"classifier": "", "qm": "", "exceed_heavy": "", "exceed_very_heavy": "", "gate": ""},
  "seed": 20260929,
  "params": {"depression_radius_km": 500, "imd_date_offset_days": 0, "qm_variant": "B"},
  "timings_s": {"ingest": 0.0, "classify": 0.0, "correct": 0.0, "exceed": 0.0, "aggregate": 0.0}
}
```

### 3.8 District table schema

| Column | Type | Description |
|---|---|---|
| `district_id`, `district_name`, `state_name` | str | GADM ids/names |
| `init`, `valid`, `lead` | date, date, int | |
| `raw_mean`, `corr_mean`, `corr_max_cell` | float mm | area-weighted mean, max over touching cells |
| `p_heavy_max`, `p_vheavy_max` | float | max over touching cells |
| `area_frac_heavy50` | float | area fraction with P(heavy) ≥ 0.5 |
| `category` | str | highest IMD category with P ≥ decision level |
| `regime_top`, `regime_top_p` | str, float | dominant synoptic regime (area-weighted mean of probabilities) |
| `geo_mix` | str | e.g. `orographic 0.6, plains 0.4` |
| `gate` | str | ON / NEUTRAL / OFF (district, dominant regime, lead) |
| `reason` | str | templated SHAP sentence |

---

## 4. Module design

Each module is a Python subpackage of `regimerain` with a pure function API (inputs → outputs, no global state) and a CLI entry in `regimerain/cli.py`. The algorithms and reference code are in `docs/MODEL_SPEC.md`.

### 4.1 `ingest`
- HRES from WeatherBench2 (anonymous GCS), subset to the rain and dynamics domains, 00 UTC inits, init months May–Sep. Rain-day precipitation via cumulative interpolation; dynamics averaged over the four 6-hourly steps inside each rain day.
- IMD via `imdlib` (1981–2022); CHIRPS via HTTPS, conservatively regridded to 0.25°. If `paths.imd` already holds the `rain/<year>.grd` files (e.g., the uploaded Kaggle dataset), read them in place and skip the download.
- IBTrACS NI CSV → parquet; LPS dataset → parquet (or run `tracker` on ERA5).
- MJO RMM text → parquet.
- Idempotent: output paths include source versions; re-running with the same inputs is a no-op (checked by content hash).

### 4.2 `static`
DEM → `elev_mean`, `elev_std`, `slope_mean`, `windward`; coastline → `dist_coast_km`; rules → `geo`; state table → `zone`; districts → `district_weights.parquet`. Runs once.

### 4.3 `label`
Active/break series from IMD (climatology 1981–2015), depression masks from tracks within `depression_radius_km`, combined into `y_synoptic`. Writes label counts per season/regime/geo/zone to `data/table/label_counts.csv`.

### 4.4 `features`
Builds the §3.3 table from `data/raw` and `data/static`. Domain indices (LLJ, trough latitude, BoB vorticity max) are computed once per (init, lead) and broadcast. Output is partitioned parquet (float32). The allow-list is defined once in `features/schema.py`.

### 4.5 `folds`
Yields LOMO folds `(train_seasons, inner_val_season, test_season)`. `inner_val_season` = the training season immediately before the test season (cyclic). Provides a fold-scoped cache directory `cache/fold={test}/`.

### 4.6 `classify`
LightGBM multi-class plus temperature scaling plus spatial smoothing. Also produces **cross-fitted** out-of-fold probabilities on training seasons (inner LOMO) for use as features in `exceed` (MODEL_SPEC §9.5).

### 4.7 `errreg`
Error descriptors → scaler → PCA → k-means → error-regime classifier, all fitted per fold. Crosstab against `y_synoptic`.

### 4.8 `correct`
QM experts (variants A/B/C), GPD tails, shrinkage hierarchy, mixture blend. Pure NumPy/SciPy; the fitted state serialises to `experts.parquet`.

### 4.9 `exceed` and `condmean`
Binary LightGBM per threshold with isotonic calibration on the inner validation season; the optional Tweedie LightGBM regressor for conditional mean.

### 4.10 `verify`
Per-day sufficient statistics (contingency counts, SSE, FSS numerator/denominator per window) are computed once, then aggregated for any slice and bootstrapped by day. Writes the §3.6 report and PNGs.

### 4.11 `gate`
Reads pooled LOMO per-day, per-district statistics → gate table per district × synoptic × lead, using PRD §10.6 rules.

### 4.12 `aggregate` and `explain`
District weights × grids → district table; LightGBM `pred_contrib` → top-3 phrases.

### 4.13 `live`
GFS download (Herbie or direct S3), GFS → the same raw schema → features → the trained model set → run artefacts. No fitting.

### 4.14 `api` and `web`
FastAPI reads `runs/` and `reports/` only; it does no computation other than serving and PNG rendering of grids. The web client renders only what the API returns; no client-side metrics (NFR-2).

---

## 5. Configuration

One YAML file, `config/default.yaml`, frozen into every model set and hashed into every manifest. The full reference file is in MODEL_SPEC §19. Every threshold, radius, hyperparameter and path lives there; code carries no magic numbers.

---

## 6. Determinism and numerical rules

- Global seed from config; LightGBM `seed`, `bagging_seed`, `feature_fraction_seed`, `deterministic=true`, `num_threads` fixed. k-means `n_init=20, random_state=seed`. Bootstrap uses `np.random.default_rng(seed)`.
- All artefacts are content-hashed (sha256) and the hashes recorded (§3.4, §3.7).
- Rainfall below 0.1 mm/day is set to 0 in both forecast and truth before any fitting or scoring (`dry_threshold_mm`).
- float32 storage, float64 accumulation for metrics.
- A metric or corrected value is never computed by two code paths. `verify` is the single source for metrics, and `correct` for corrected values.

---

## 7. Sizing and performance

| Item | Estimate |
|---|---|
| Land cells | ~4,900 |
| Cell-days per lead, 7 seasons | 854 × 4,900 ≈ 4.2 M |
| Feature table, all leads (~40 float32 columns) | 5 × 4.2 M × 40 × 4 B ≈ 3.4 GB (parquet compressed ≈ 1–1.5 GB) |
| HRES download (rain + dynamics, India/dynamics domains, 7 × JJAS) | a few GB; download-bound, ~1–3 h |
| IMD 1981–2022 | ~1 GB |
| Classifier fit (lead as feature, 1-in-4 cell subsample ≈ 5 M rows) | ~3–8 min per fit on 8 cores |
| QM experts fit, one fold | < 2 min |
| Exceedance fit per threshold per fold | ~3–6 min |
| Full LOMO backtest, variants A+B, `fast` mode | ~3–5 h |
| Live run after download | < 60 s (NFR-3) |

RAM: load one lead at a time. Use `pyarrow` dataset filters by season; never hold all leads in memory at once.

---

## 8. Repository layout

```
regimerain/
  __init__.py
  cli.py
  config.py                 # load + validate YAML (pydantic)
  ingest/{hres.py, imd.py, chirps.py, tracks.py, mjo.py, gfs.py, align.py}
  static/{dem.py, coast.py, zones.py, districts.py}
  label/{active_break.py, depression.py, tracker.py}
  features/{schema.py, dynamics.py, build.py}
  folds.py
  classify/{train.py, calibrate.py, predict.py}
  errreg/{descriptors.py, cluster.py, predict.py}
  correct/{qm.py, gpd.py, experts.py, mixture.py}
  exceed/{train.py, calibrate.py}
  condmean/train.py
  verify/{metrics.py, fss.py, stats.py, bootstrap.py, report.py, plots.py}
  gate/gate.py
  aggregate/aggregate.py
  explain/explain.py
  live/run.py
  api/app.py
web/                        # MapLibre client (from ../167/)
config/default.yaml
data/  cache/  models/  runs/  reports/   # git-ignored
tests/
docs/
```

---

## 9. CLI

```bash
python -m regimerain.cli ingest   --years 2016-2022 --truth imd        # module 1
python -m regimerain.cli ingest   --imd-climatology 1981-2015
python -m regimerain.cli static                                         # static layers, zones, district weights
python -m regimerain.cli label                                          # synoptic labels
python -m regimerain.cli features --leads 1-5                           # feature tables
python -m regimerain.cli backtest --variants raw,A,B,C --mode fast      # full LOMO: classify, errreg, correct, exceed, verify, gate
python -m regimerain.cli backtest --variants raw,A,B,C --mode fast --folds 2016,2017,2018   # only these test seasons -> cache/fold=Y/
python -m regimerain.cli report                                         # pool every cache/fold=Y/ -> verification report + skill gate
python -m regimerain.cli fit-final                                      # refit on all seasons -> models/<id>/
python -m regimerain.cli run      --source gfs --init today             # live run -> runs/<run_id>/
python -m regimerain.cli selftest                                       # unit + leakage + fold-isolation + no-hardcoded-number tests
python -m regimerain.cli selftest --quick                               # leakage + table-schema checks only (after `features`)
python -m regimerain.cli serve                                          # http://127.0.0.1:8000/
```

**Global options** (all commands): `--config PATH` (default `config/default.yaml`; later files override earlier ones when repeated, e.g. `--config config/default.yaml --config config/kaggle.yaml`) and `--data-root PATH` (overrides `paths.*` in the config).

**Resumable backtest.** `backtest --folds` runs only the listed test seasons and writes each to `cache/fold=<Y>/` with a `DONE` marker. A fold with a `DONE` marker is skipped unless `--force` is given. `report` refuses to run unless all 7 folds have `DONE` markers and share one `config_sha256`, which prevents pooling folds trained with different settings. This is what makes session-limited platforms (Colab, `docs/MODEL_SPEC.md` §23; Kaggle, §24) workable.

---

## 10. API schemas

- `GET /runs/latest` → `{"run_id": str, "manifest": {...}}`
- `GET /runs/{run_id}/regime?lead=1&format=png|geojson` → PNG overlay (with `bounds` header) or GeoJSON of cells with `{"p": {"active": f, ...}, "geo": str}`; plus `summary`: `{"active": frac_cells, ...}`
- `GET /runs/{run_id}/rainfall?variant=raw|corrected|cond_mean&lead=1&format=png|json` → grid
- `GET /runs/{run_id}/exceedance/{heavy|very_heavy}?lead=1` → grid
- `GET /runs/{run_id}/gate?lead=1` → grid of −1/0/1
- `GET /runs/{run_id}/districts?lead=1` → list of §3.8 rows
- `GET /reports/{backtest_id}/verification?variant=B&lead=1&threshold=64.5` → filtered §3.6 rows and deltas
- `POST /runs/live` → `202 {"run_id": str, "status": "running"}`; `GET /runs/{run_id}/status` → `{"status": "running|done|failed", "log_tail": str}`

Errors: `404` for an unknown run, lead or threshold; `409` if a live run is already in progress.

---

## 11. Test plan

| Test | What it asserts |
|---|---|
| `test_leakage` | No `o_*` / `y_*` column in any model's `feature_name()` (AC-1) |
| `test_fold_isolation` | For every fold, no test-season date in the training rows of the classifier, QM, errreg, exceed or calibration (AC-9) |
| `test_align_rainday` | A synthetic constant-rate forecast gives exactly 24 × rate per rain day; 6-hourly interpolation matches the analytic value |
| `test_imd_offset` | The configured offset maximises the forecast–truth lag correlation (run on data, marked `slow`) |
| `test_metrics_toy_tables` | POD, FAR, CSI, ETS, RMSE on hand-computed 2×2 tables |
| `test_fss_known_cases` | FSS = 1 for identical fields; FSS = 0 for non-overlapping fields at window 1; rises monotonically with window for a shifted blob |
| `test_qm_identity` | QM fitted on (o, o) maps o ≈ o (max abs error < 1% at quantiles) |
| `test_qm_monotone` | Corrected output is non-decreasing in raw input for every expert |
| `test_gpd_tail_extends` | Values above the training max map above the observed max when ξ ≥ 0 |
| `test_shrinkage_weights` | w = n/(n+N_min); an expert with n = 0 equals its parent |
| `test_mixture_weights` | Output equals a single expert when p = one-hot; weights renormalise after the 0.1 cut |
| `test_exceed_monotone` | P(very heavy) ≤ P(heavy) everywhere |
| `test_calibration_fold_isolation` | The isotonic fit uses only the inner-val season |
| `test_gate_rules` | Synthetic statistics produce the expected ON/NEUTRAL/OFF |
| `test_district_weights_sum_to_one` | Per district, weights sum to 1 ± 1e-6 |
| `test_no_hardcoded_numbers` | The web source contains no numeric literals in score/probability contexts (AC-5) |
| `test_manifest_complete` | Every run manifest has all §3.7 keys |
| `test_live_smoke` | `run --source gfs` on a cached GRIB fixture produces all §3.5 files (network test marked) |

---

## 12. Licensing and safety

- All data sources are open. There is no PII and no credentials. Record each dataset's licence and citation in `docs/DATA_LICENSES.md`.
- District boundaries: if redistribution terms are unclear, do not commit the file; download it at setup time from the source URL.
- The product shows **probabilities and confidence flags**, and the UI carries a footer: "Experimental research product, not an official IMD/NCMRWF forecast."

---

## 13. NCUM adapter interface

To run on NCMRWF's model operationally, implement one function:

```python
def load_forecast(init: datetime, leads: list[int]) -> tuple[xr.DataArray, xr.Dataset]:
    """Return (rain[lead, lat, lon] in mm per IMD rain day on the 0.25 deg working grid,
               dyn Dataset with the variables of TRD section 3.1 on the dynamics domain)."""
```

`ingest/hres.py` and `ingest/gfs.py` are the two reference implementations. Retraining on NCUM hindcasts uses the same `backtest` / `fit-final` commands with `--source ncum`.

---

## 14. Training environment

Training runs on **Google Colab** (CPU runtime) with Google Drive as the persistent store. The storage model, stages, settings lock, quota and notebook are in `docs/MODEL_SPEC.md` §23 and `notebooks/colab_pipeline.ipynb`. Kaggle is a documented alternative (`docs/MODEL_SPEC.md` §24, `notebooks/kaggle_pipeline.ipynb`). The platform override files (`config/colab.yaml`, `config/kaggle.yaml`) change only `paths.*`, `num_threads`, `exceed.per_lead` and `classifier.cell_stride`. **Use the same `num_threads` and model settings for every fold and for the final fit**, because LightGBM's deterministic mode is reproducible only for a fixed thread count. The Colab notebook enforces this with a settings lock on Drive.

## 15. Demo deployment

- Single laptop: `serve` runs FastAPI plus static web on port 8000. Model set and latest run are pre-computed before the demo; `POST /runs/live` is shown live with a pre-downloaded GFS fallback if venue internet fails.
- A `make demo` target runs `run --source gfs --init today || run --source gfs --init cached`.

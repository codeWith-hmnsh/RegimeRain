# Phase 0 findings

Results of `scripts/phase0_check.py` (docs/MODEL_SPEC.md section 3). Update this file every time a check is rerun.

**Run 1: 29 Sep 2026, Google Colab (CPU), code at `7e02258`**

| Check | Status | Finding | Consequence |
|---|---|---|---|
| **HRES (WeatherBench2)** | PASS | Store `gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr` opens anonymously. Precipitation variables: `total_precipitation` (units `m`), `total_precipitation_6hr` and `total_precipitation_24hr` (units not set). Dims `time, prediction_timedelta, latitude, longitude`. Latitude **ascending**. Levels 50–1000 hPa (13), including all needed for moisture integrals. Also present: `u/v_component_of_wind`, `specific_humidity`, `mean_sea_level_pressure`, `vertical_velocity`, `geopotential`, `temperature`, `surface_pressure`. | Variable names in MODEL_SPEC 4.1/4.5 confirmed. **Pending:** whether `total_precipitation` is accumulated since init (the run-2 consistency check decides `sources.hres_precip_kind`). The run-1 lead-hour printout was a script bug (fixed); leads are 6-hourly. |
| **IBTrACS NI v04r01** | PASS | CSV reachable; `SID, ISO_TIME, LAT, LON, NEWDELHI_GRADE, NEWDELHI_WIND` present. | `ingest/tracks.py` as specified. |
| **GFS 0.25° (AWS via Herbie)** | PASS | Each file has both a bucket and a **running total from init**: f009 has `6-9` and `0-9`, f012 has `6-12` and `0-12`, f027 has `24-27` and `0-27`. At f003/f006 the two coincide (`0-3`, `0-6`, listed twice). | Rain day L = APCP(`0-{27+24(L-1)}`) − APCP(`0-{3+24(L-1)}`), exact at 03 UTC. Take the first message where duplicates exist. MODEL_SPEC 18.2 updated. |
| **IMD (imdlib)** | FAIL, script bug | `imdlib.get_data` does not create its target folder, so the IMD server was never contacted. | Script fixed (creates the folder). **Rerun.** |
| **MJO RMM (BOM)** | FAIL | `HTTP 403 Forbidden` from `bom.gov.au`. Probably a block on Python's default User-Agent. | Script now sends an explicit User-Agent and accepts a manually downloaded copy via `RMM_FILE=<path>`. **Rerun.** If it still fails, download `rmm.74toRealtime.txt` in a browser and upload it to `MyDrive/ps26080/inputs/mjo/`. |

**Run 2: 29 Sep 2026, Google Colab, code at `55754fb`**

| Check | Status | Finding | Consequence |
|---|---|---|---|
| **HRES** | FAIL, code bug | `prediction_timedelta` arrives as **int64 with a `units` attribute (hours)**, not timedelta64: recent xarray no longer decodes timedeltas by default. | Added `regimerain.ingest.align.lead_hours()` (handles both forms), used by `to_hours` and the script, plus a regression test. **Rerun.** |
| **MJO RMM (BOM)** | FAIL | 403 on http and https, even with an explicit User-Agent. BOM blocks cloud-server addresses. | **Manual download**: save `rmm.74toRealtime.txt` in a browser, upload it to `MyDrive/ps26080/inputs/mjo/`, point `RMM_FILE` at it. The live demo needs a fresh copy the same way; if it is more than 3 days old, MJO features are set to neutral (MODEL_SPEC 18.2). |

**Run 3: 29 Sep 2026, Google Colab, code at `bbab350`**

| Check | Status | Finding | Consequence |
|---|---|---|---|
| **HRES** | PASS | Leads 0–240 h every 6 h (`prediction_timedelta` int64, units `hours`). `total_precipitation` is **accumulated from init, in metres**: starts at 0, never decreases, its 6-h differences match `total_precipitation_6hr` to 0.01%, and `total_precipitation_24hr` at 24 h equals tp(24) − tp(0). Central-India box mean on 15 Jul 2019, 0–24 h: 2.82 mm. | `sources.hres_precip_kind: cumulative` (reads `total_precipitation` directly). `regimerain/ingest/hres.py` written against these formats. |

## Still to check

- [x] HRES precipitation consistency (run 3).
- [ ] HRES chunking (`hres_chunks` check): how much data an India-only read downloads.
- [ ] One-season HRES ingest timing on Colab.
- [ ] IMD download from Colab, and IMD's date convention (MODEL_SPEC 4.3; needs the forecast ingest first).
- [ ] MJO from the manually downloaded file.
- [ ] Monsoon LPS track dataset host (not scripted).
- [ ] DEM and GADM downloads and licence notes (not scripted).

# Live GFS run explained

Code: `regimerain/ingest/gfs.py` (download), `regimerain/runs/live.py` (predict), `regimerain run --source gfs`
in `regimerain/cli.py`. Uses the model set from `regimerain fit-final` (`regimerain/final.py`, `ModelSet`).
Tests: `tests/test_gfs.py`, and the live step at the end of `tests/test_pipeline_e2e.py`.

## The idea in one line

Take today's free NOAA GFS forecast, build the same features the models were trained on, run the frozen
model set (regime classifier, regime-aware correction, heavy-rain probabilities), switch the correction off
where the skill gate says it did not help, and export a run for the website.

## The steps

1. **Download** (`fetch_gfs`). Herbie downloads only the GRIB messages needed (byte ranges via the `.idx`
   file), from the AWS archive or NOAA NOMADS:
   - rain: running total since init at 3, 27, 51, 75, 99, 123 h. Rain day L = total(27 + 24(L−1)) −
     total(3 + 24(L−1)), the IMD 03–03 UTC day, the same window as the HRES training data;
   - dynamics: u, v, specific humidity at 1000–300 hPa, 500 hPa vertical velocity and sea-level pressure,
     every 6 h. The four instants inside each rain day are averaged, exactly like the HRES ingest. If a
     file has no specific humidity, it is computed from relative humidity and temperature.
2. **Features** (`forecast_rows` in `regimerain/features/build.py`). The training table and the live run call
   the same function, so a feature cannot be computed one way in training and another way live.
3. **Predict** (`ModelSet.predict`): regime probabilities (temperature-scaled, then smoothed on the map the
   same way as in the backtest), corrections A and B, and P(≥ 64.5 mm) and P(≥ 115.6 mm).
4. **Gate** (`ModelSet.gate_served`): for each cell, look up its zone × terrain unit, lead and predicted regime
   in the gate table. Where it is OFF, the served rain is the raw GFS rain.
5. **Export**: web contract v2 with `kind: "model"`, the wind layer, the model set id and every file hash
   in the manifest. There is no truth layer: a live forecast has no observations yet.

Downloaded on the laptop for one rain day it took about 2 minutes (15 Jul and 1 Aug 2026 inits). All five days
need 6 rain files and 20 dynamics files.

## Honest limits (say these before anyone asks)

- **Model transfer.** The models were trained on ECMWF IFS HRES, but live runs use GFS. GFS has different
  biases, so the correction is a transfer, not a GFS-trained correction. The manifest and the Bulletin
  page say so. The proper fix is to retrain on GFS reforecasts.
- **MJO.** The MJO index comes from a manually downloaded file (BOM blocks cloud servers). If it is more than
  3 days old, the MJO features are set to neutral and the manifest says so.

## Interview answers

- **How do you stop training and live features from drifting apart?** One function builds the features for
  both, and the end-to-end test runs a live forecast through the fitted model set and validates the export.
- **Why not download whole GFS files?** A full 0.25° file is about 500 MB. Herbie reads the index and
  downloads only the needed messages, a few MB each.
- **What happens where the model does not help?** The skill gate serves raw GFS there, and says how many
  cell-days that was in the manifest note.

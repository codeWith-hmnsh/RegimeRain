# Reasons ("why heavy rain here?") explained

Code: `regimerain/explain/explain.py`, called from `ModelSet.explain` (`regimerain/final.py`) for live runs
(`regimerain/runs/live.py`). Shown on the Bulletin page under each place. Tests: `tests/test_explain.py` and
the live step of `tests/test_pipeline_e2e.py`.

## The idea in one line

For each named place and day, list the two or three things that pushed the heavy-rain probability up the
most, in words a forecaster uses: *"depression/low regime (p=0.72), moisture convergence 41 mm/day, MJO
phase 4"*.

## How it works

1. The heavy-rain model (M4, P(rain ≥ 64.5 mm)) is a LightGBM tree ensemble. LightGBM can split any single
   prediction into one number per feature plus a bias (`pred_contrib=True`, the TreeSHAP algorithm). The
   numbers add up exactly to the model's raw score (log-odds); a test checks this.
2. Sort features by contribution and keep the top three that pushed the probability **up** (positive).
3. Turn each into a phrase with its value (`PHRASES`). Position and calendar features (latitude, longitude,
   zone, day of year, lead) are skipped: "because it is at 19°N" explains nothing to a forecaster.
4. The log-odds numbers go in a tooltip only; the sentence never shows them.

Only the cells of named places are explained (about 40 places × 5 days): TreeSHAP on every land cell would be
wasted work. Replay runs have no reasons, because the backtest does not keep each fold's heavy-rain model.

## Honest limits

- A contribution says what the model relied on, not what causes rain in the atmosphere. Correlated features
  (moisture transport and moist column) share credit in ways that can look arbitrary.
- The contributions are for the score before isotonic calibration, so they rank causes but are not
  percentage points of probability.

## Interview answers

- **Why TreeSHAP and not plain feature importance?** Importance is one global ranking. TreeSHAP explains
  this place on this day, and for tree models it is exact and fast.
- **Why hide some features?** They carry real signal (some regions rain more), but a reason must be
  something a forecaster can check on a weather map.

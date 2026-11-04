# Error regimes and variant C explained

Code: `regimerain/errreg/errreg.py`, run by `regimerain backtest --variants raw,A,B,C`. Output per fold:
`cache/fold=<season>/errreg.json` (clusters, names, crosstab, agreement), `qm_counts_C.csv`, column `C` in
`predictions.parquet`; the report scores variant C next to A and B and adds an "Error regimes" section.
Tests: `tests/test_errreg.py` and the backtest step of `tests/test_pipeline_e2e.py`.

## The idea in one line

Variant B corrects rain separately for each weather regime that meteorologists name (active, break,
depression, normal). Variant C lets the data say how the forecast goes wrong: it clusters the training days by
the *type of error*, then corrects each type separately.

## How it works (inside every leave-one-monsoon-out fold, separately per lead)

1. **Describe each day's error** (needs the truth, so training days only): for each of the 4 IMD zones,
   6 numbers: wet/dry bias, heavy-rain area too big or missed, how far the heavy rain was displaced (and
   north/south), pattern correlation, and whether the extremes were too strong or too weak. 24 numbers per day.
2. **Cluster** the fit-season days: standardise, PCA to 90 % of the variance, k-means for k = 4…8. Pick k by
   silhouette × stability. Stability is the adjusted Rand index between the clustering and re-clusterings of
   bootstrap-resampled days: a cluster that disappears when you resample days is not real.
3. **Name** each cluster by its most distinctive descriptor, e.g. "too wet (central)" or
   "extremes too weak (south_peninsula)".
4. **Predict the cluster from forecast-time information only** (the truth is not known when forecasting):
   a small LightGBM day-level classifier on domain indices (monsoon jet, trough position, Bay of Bengal
   vorticity), MJO, season, and per-zone forecast aggregates. It trains on fit-season days and early-stops on
   the inner validation season.
5. **Correct**: one quantile-mapping expert set per cluster (same shrinkage chain as B). On a test day, the
   predicted cluster probabilities weight the experts, exactly like the regime mixture in B.

The test season's descriptors are only used afterwards, to report how often the predicted cluster matched the
real one ("agreement"), next to the score of always guessing the biggest cluster.

## How to read the result

- If agreement is barely above the majority baseline, the error type cannot be predicted from the forecast,
  and C will score close to A. That is a finding, not a failure: say it plainly.
- The crosstab (cluster × synoptic regime) shows whether error types line up with the named regimes. If a
  cluster is mostly "break" days, the error regimes rediscover what B already uses.

## Limits

- About 730 training days per lead: enough for a few clusters, not for fine distinctions.
- Variant C is evaluated in the backtest only; the live model set uses B (fit-final does not refit C).

## Interview answers

- **Why cluster errors instead of weather?** Two days with the same weather can fail differently; correcting
  by error type targets the failure directly.
- **How did you pick k?** Silhouette for separation, bootstrap ARI for stability; their product, so a
  well-separated but unstable split loses.
- **Isn't using the truth to make clusters leakage?** The truth defines the clusters on training days only.
  At test time the cluster is predicted from forecast-time features; the test truth is used only to score.

# Interview prep: RegimeRain

Numbers come from backtest `20261105_45e44153` (`docs/PILOT_RESULTS.md`). Topic explainers with more
interview answers: `EXPLAINED_M4_HEAVY_RAIN.md`, `EXPLAINED_SKILL_GATE.md`, `EXPLAINED_ERROR_REGIMES.md`,
`EXPLAINED_REASONS.md`, `EXPLAINED_LIVE_GFS.md`.

## The 2-minute pitch

1. **Problem.** Weather models (NWP) get Indian monsoon rain wrong in a structured way. Errors differ on
   active-monsoon days, break days and depression days. One global bias correction averages that away.
2. **Idea.** First classify the kind of day from forecast-time data only. Then apply a correction
   fitted for that kind of day.
3. **Data.** ECMWF HRES forecasts (WeatherBench2) against IMD 0.25° gridded rain. 4,964 land cells,
   June to September 2016-2022, lead Day 1 to Day 5.
4. **Models.**
   - LightGBM regime classifier (4 classes).
   - Quantile-mapping correctors: global (A), per regime (B) and per error type (C).
   - LightGBM + isotonic heavy-rain probability model (M4).
   - A skill gate that turns the correction off where it hurt.
5. **Validation.** Leave-one-monsoon-out: every score comes from a season the models never saw.
   Block-bootstrap 90% confidence intervals.
6. **Result.**
   - Correction improves heavy-rain detection at every lead (Day 1 POD 0.152 → 0.183, ETS 0.084 → 0.092).
   - The heavy-rain model has ROC-AUC 0.894.
   - Regime-aware correction is not clearly better than global correction yet. I report that.
7. **Product.** A static website replays real held-out days, for example the 9 Aug 2019 Kerala floods.
   A live mode runs today's GFS forecast through the final model set.

## Questions to expect

**Why leave-one-monsoon-out and not a random split?**
Rain on neighbouring days and cells is strongly correlated. A random split puts 14 July in the training set
and 15 July in the test set, so the model "remembers" the weather system and the score is inflated. Holding
out a whole season tests what deployment needs: a season the model has never seen.

**Why can quantile mapping make RMSE worse?**
Raw forecasts are too smooth: too much drizzle, too few extremes. Quantile mapping restores the observed
distribution, so it forecasts more heavy rain. Heavy rain in the right amount but slightly the wrong place
is punished twice by RMSE. That is why Day 1 RMSE went from 18.13 to 18.54, while ETS and POD improved.
The conditional-mean variant shows the opposite trade-off. It has the lowest RMSE (14.20) but almost never
predicts heavy rain (POD 0.033). The right metric depends on the user. A flood warning needs POD and ETS.

**Why LightGBM, not a neural network?**
The data is tabular: one row per cell, day and lead with engineered features. Gradient-boosted trees are
strong on tabular data, train in minutes on CPU, handle missing values and give TreeSHAP explanations.
A CNN on whole maps is the next step (it could fix displacement errors). It needs a GPU and more seasons.

**How did you prevent leakage?**
- Every model is refit inside each fold.
- Calibration and early stopping use an inner validation season, never the test season.
- The heavy-rain model gets regime probabilities that are cross-fitted, so its inputs never come from a
  model that saw the same rows.
- Regime labels use an IMD 1981-2015 climatology, so no test season (2016-2022) shapes its own labels.
- Tests check it: `test_lomo_folds_isolation` and `test_leak_is_detected` in `tests/test_structure.py`.

**What did NOT work?**
- Regime-aware QM versus global QM: no significant ETS difference at Day 1-3. At Day 4-5 it is
  significantly better, but only by +0.001 ETS, and its RMSE is worse. A likely reason: the classifier
  still misses most break days (break recall below 0.43 in five of the six test seasons that had break days).
- Error-type clusters (variant C) are weak (silhouette 0.07-0.08). They are predictable above chance, but
  they do not beat global QM on heavy rain.
- Quantile mapping cannot move rain that is in the wrong place.

**How good is the regime classifier?**
Accuracy 0.63-0.73 per unseen season, macro-F1 0.29-0.54. In the first pilot (3 seasons, one training season
per fold) it was about 0.43 and 0.2. More seasons helped most.

**What is the skill gate?**
For each zone × terrain unit × lead, it compares corrected and raw forecasts with a bootstrap confidence
interval. If the correction is significantly worse, the product serves raw NWP there and flags it.
Result: ON 9, NEUTRAL 9, OFF 2. Both OFF cells are the east and north-east plains at Day 1-2.

**How did you run it without a GPU?**
Kaggle CPU notebooks (4 cores, 33 GB RAM, 12 h limit). The pipeline runs in stages and each stage's output
becomes a Kaggle dataset: HRES download, feature table, backtest folds in parallel chunks, final fit.
- Each backtest fold took 21-27 min.
- The final fit took 11 min.
- One pinned code commit and one config hash for all folds; `report` refuses to pool folds with different
  configs.

**What would you do next?**
- Better break-day detection (MJO and monsoon-trough features).
- Real district boundaries and terrain classes instead of approximate zones.
- A neural displacement correction.
- An operational trial on live GFS forecasts during a monsoon season.

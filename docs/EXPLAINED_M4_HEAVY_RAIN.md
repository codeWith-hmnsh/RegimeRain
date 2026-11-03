# M4 explained: the heavy-rain probability model

Read this before any interview. Code: `regimerain/exceed/`, `regimerain/classify/crossfit.py`,
`run_exceed()` in `regimerain/backtest.py`. Tests: `tests/test_exceed.py`, `tests/test_pipeline_e2e.py`.

## 1. What it answers

"What is the probability that tomorrow's rain at this grid cell (about 25 km × 25 km) reaches IMD's heavy
(≥ 64.5 mm) or very heavy (≥ 115.6 mm) level?"

Output: two numbers between 0 and 1 for every cell and lead day, stored as `p_exc_64.5` and `p_exc_115.6`.

## 2. Why a separate model, and not "is the corrected rain above 64.5?"

A single number cannot say how sure we are. Corrected rain of 60 mm could mean "almost certainly heavy" on a
depression day or "unlikely" on a break day. A probability carries that uncertainty, and disaster officers
act on probabilities ("likely" at P ≥ 0.5, "possible" at P ≥ 0.3).

## 3. How it works, in four steps

1. **Inputs (features).** Everything the regime classifier sees (forecast rain and its neighbourhood, winds,
   moisture, jet index, MJO, terrain, day of year, lead), plus two outputs of earlier models: the regime
   probabilities `p_active … p_normal` and the regime-corrected rain `qm_rain`.
2. **Model.** LightGBM binary classifier (gradient-boosted decision trees), one per threshold.
   Why trees and not a neural network: tabular data, only 7 seasons, fast on a CPU, and explainable
   (feature importance is saved in `exceed.json`).
3. **Calibration.** A raw score of 0.7 is not automatically a 70 % chance. Isotonic regression maps the raw
   scores to observed frequencies, fitted on the inner-validation season only.
4. **Consistency.** P(very heavy) can never exceed P(heavy): enforced after calibration.

## 4. The two traps it avoids (the interview questions)

**Trap 1: leakage through earlier models.** If the training rows used regime probabilities from a
classifier that was trained on those same rows, those probabilities would be over-confident, and M4 would
learn to trust them too much. At test time they are weaker, so M4 would fail.
Fix: **cross-fitting**. The training seasons are split into groups; each group's `p_*` and `qm_rain` come
from a classifier and QM experts fitted without that group. Test: `test_crossfit_never_uses_the_rows_it_predicts`
changes one season's labels and truth and checks that season's own features do not change.

**Trap 2: calibrating on the test season.** That makes the reliability diagram look perfect and means
nothing. Fix: early stopping and isotonic calibration use the inner-validation season only; the test season
is never touched. `exceed.json` records `calibrated_on`, and the end-to-end test checks it.

## 5. Rare events

Heavy rain is about 1 % of cell-days. Training keeps every heavy-rain row and a random 25 % of the others
(`neg_fraction`, fast mode), each kept dry row weighted 4. The weighted event rate stays the true rate
(`test_negative_sample_keeps_every_event_and_the_weighted_rate`), and training is about 3× faster.
If a threshold has fewer than `min_events` (20) events in fit or validation, no model is trained and the
output is the base rate, flagged in `exceed.json`. That is honest, rather than a model fitted on noise.

## 6. How it is scored (`verification_report.json → exceedance`, `summary.md`)

| Score | Meaning | Good value |
|---|---|---|
| Brier score | mean of (probability − outcome)² | lower is better |
| BSS | skill against always forecasting the event frequency | > 0 means useful |
| ROC-AUC | does it rank heavy-rain cells above dry ones? | 0.5 = random, 1 = perfect |
| PR-AUC | precision vs recall for the rare class; honest for rare events | higher than the event rate |
| Reliability diagram | forecast 30 % → observed about 30 %? | points on the diagonal |
| POD / FAR at 0.3 and 0.5 | the alert levels used as yes/no forecasts | trade-off, reported both |

All numbers are on held-out seasons, pooled and per season.

## 7. Two-line answer for "what did you build?"

"I built a calibrated LightGBM model that gives the probability of IMD heavy and very heavy rain per grid
cell. Its training inputs are cross-fitted so no earlier model leaks the answer, it is calibrated only on a
validation season, and I score it with Brier skill, reliability and ROC/PR-AUC on unseen monsoon seasons."

## 8. Questions to practise

1. Why isotonic regression and not Platt scaling? (Isotonic makes no shape assumption; with millions of
   validation rows it does not overfit.)
2. Why is BSS measured against climatology? (A model must beat the trivial forecast "always the average
   rate" to be worth anything.)
3. Why PR-AUC as well as ROC-AUC? (With 1 % events, ROC-AUC can look high even when most alerts are false.)
4. What would you do with more time? (Per-region calibration; displacement-aware features; SHAP reasons per
   alert, MODEL_SPEC §15.)

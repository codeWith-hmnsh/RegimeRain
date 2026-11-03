# Skill gate explained

Code: `regimerain/gate/gate.py`, called by `regimerain report`. Output: `reports/<id>/skill_gate.parquet`,
`skill_gate_map.png`, the "Skill gate" section of `summary.md`. Tests: `tests/test_gate.py`.

## The idea in one line

A correction that helps on average can still hurt in some places. The gate checks, region by region and
regime by regime, whether the correction actually helped on seasons it never saw, and switches it off where
it did not.

## How it decides

For each **unit** (IMD zone × terrain class) × **lead day** × **predicted regime**, on the pooled
leave-one-monsoon-out predictions:

1. **MSE skill score** SS = 1 − error²(corrected) / error²(raw). Above 0 means corrected is closer to the truth.
2. **ETS gain** at 64.5 mm: does heavy-rain detection improve? Only computed with ≥ 10 observed heavy cells.
3. **90 % confidence intervals** by resampling 5-day blocks of days (days next to each other are correlated,
   so single days cannot be resampled independently).
4. Decision:
   - **ON** if either measure is significantly better and neither is significantly worse;
   - **OFF** if either is significantly worse and neither is significantly better;
   - **NEUTRAL** otherwise.
5. A regime with fewer than 30 days uses the gate of its unit × lead over all regimes (too little data to
   decide alone).

Live use: corrected rain where ON or NEUTRAL, raw NWP where OFF, and the map flags it. A forecaster never
gets a correction that was measured to make things worse.

## Why units are zones, not districts

PRD FR-11 asks for districts, but district boundaries wait on a licence check (PRD §20). The code needs only
a unit column, so district polygons replace zone × terrain later without changing the logic.

## Interview answers

- **Why a gate at all?** Honest degradation: if the correction does not help somewhere, the product says so
  and falls back to raw NWP, instead of hiding it in an all-India average.
- **Why block bootstrap?** Rain on consecutive days is correlated; resampling single days would make the
  intervals too narrow and claim significance that is not there.
- **Isn't it tuned on the test seasons?** It is built only from held-out (out-of-fold) predictions: each
  season's corrections come from models that never saw that season.

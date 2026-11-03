# PS26080 — official brief

**PS Number:** SIH26080
**Title:** Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts
**Organisation:** Ministry of Earth Sciences (MoES)
**Department:** National Centre for Medium Range Weather Forecasting (NCMRWF)
**Category:** Software · **Theme:** Smart Automation
**Portal deadline:** 20 September 2026 (idea-submission deadline; live counts tracked separately, see `../../extra/problem-statements/SLOT2_CANDIDATES.md`)

Verbatim from the SIH portal dump (`extra/problem-statements/sih2026_ps.json`, `ps_id: "26080"`):

> **Problem Statement.** Rainfall forecast errors over India vary with weather regimes such as active monsoon, break monsoon, monsoon lows/depressions, orographic rainfall, coastal rainfall and western disturbances. A single bias-correction method may not work equally well in all situations. The challenge is to build an AI/ML-based rainfall post-processing system that first identifies the prevailing weather regime and then applies suitable correction to the raw NWP rainfall forecast. The aim is to improve district/grid-level rainfall forecasts, especially for heavy and very heavy rainfall events.
>
> **Expected Outcome.**
> - **Weather regime classifier** — classification of active, break, depression, coastal/orographic rainfall regimes.
> - **Bias-corrected rainfall forecast** — improved rainfall forecast compared to raw NWP output.
> - **Heavy rainfall probability** — probability of rainfall exceeding operational thresholds.
> - **District-level rainfall product** — user-friendly rainfall forecast table/map.
> - **Verification report** — skill comparison using RMSE, ETS, CSI, POD, FAR and FSS.

The statement supplies its own yardstick: it names six verification metrics by acronym. That is unusual — most PS texts in this batch give no dataset and no metric. Treat it as the judges' rubric, not a suggestion.

## Where this sits in the team's ranking

Full comparative analysis, live submission counts, and the seven other candidates considered: `extra/problem-statements/SLOT2_CANDIDATES.md`. As of the 27 Sep pull, PS26080 ranked 1st by the team's `score = (3/submissions) × feasibility factor` rule (21 submissions, high feasibility). The separate 108-statement composite scoring pass in `extra/problem-statements/shortlist_ranked.md` places SIH26080 at rank 21/108 overall (composite 68.0) on quality/winnability/data/crowding/ceiling/fit.

Chosen 29 Sep 2026, alongside PS26081 (Hybrid AI–NWP Forecast Blending, built separately by Mridul), to replace both PS26143 (Dark Transit — 500/500, closed) and PS26167 (SatQuery — reported filled).

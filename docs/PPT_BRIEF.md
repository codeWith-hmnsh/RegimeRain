# PPT brief — SIH26080 idea-submission deck

Slide-by-slide content for the team building the PPT. Follows the official `templates-and-examples/SIH2026-IDEA-Presentation-Format.pptx` and copies the layout patterns of `templates-and-examples/sih_winning.pdf` (Visioncraft, PS 25159).

Facts come from `../PRD.md` and `../TECHNICAL_SPEC.md` (both Rev B). Every number below is a definition, threshold or design target from those two files. The only measured numbers are in `PILOT_RESULTS.md` (pilot v1, 3 seasons); quote them only from there, with their caveats (regime-aware QM does not yet beat global QM; Day-1 RMSE got worse). Do not add any other skill score, "% improvement" or latency figure.

**Graphics:** Mermaid sources for every diagram are in `PPT_DIAGRAMS.md`. Render them at https://mermaid.live and restyle in the deck. The flow diagram on slide 3 is the reference for the main graphic.

## Submission rules (from the template's last slide)

- Maximum **6 slides including the title slide**. Delete the "Important Pointers" slide.
- Use the provided template. Keep its heading pointers unchanged.
- Points, diagrams and pictures. No paragraphs. Short, precise lines.
- Export to **PDF** and upload the PDF. No PPTX.
- Team name, logo oval (top-left) and blue footer bar carry over on slides 2 to 6, as in the winning deck.

## What the winning deck did (copy these patterns)

| Slide | Winning-deck pattern |
|---|---|
| 1 Title | Bullet list of six fields, big SIH logo graphic on right |
| 2 Solution | Five rounded grey-lilac "pill" boxes, key phrases in **red**, icon-hub infographic (centre shield + 6 labelled icons) on the right |
| 3 Technical approach | Left column: numbered tech stack + logos + prototype/video links. Right: one big flow diagram with decision branches and a feedback loop |
| 4 Feasibility | Three-column coloured table (cyan / red / pink), then three bold-lead bullets |
| 5 Impact | Two rounded panels (pink left, cyan right), numbered headings in green with two sub-bullets each, three small visuals along the bottom |
| 6 Research | Conventional-vs-proposed table, then numbered references and dataset links |

Colour convention: black body text, **red** for the key claim phrases, green for numbered sub-headings on slide 5.

---

## Slide 1 — Title page

Title: SMART INDIA HACKATHON 2026 (use template graphic)

- Problem Statement ID: **SIH26080**
- Problem Statement Title: **Regime-Aware AI Post-Processing of Monsoon Rainfall Forecasts**
- Theme: **Smart Automation**
- PS Category: **Software**
- Team ID: *(fill from portal)*
- Team Name: *(as registered on portal)*

Organisation line, small, if space allows: Ministry of Earth Sciences (MoES), NCMRWF.

---

## Slide 2 — Proposed solution

**Slide title (top, centred):** RegimeRain: Classify the Weather Regime First, Then Correct the Rainfall
**Sub-heading (template):** Proposed Solution (Describe your Idea/Solution/Prototype)

### Left: five pill boxes (red = highlighted phrase)

1. Raw NWP rainfall forecasts err **differently on different days**: active monsoon, break spell, depression, orographic and coastal rain. **One global bias-correction curve** averages that structure away.
2. Our system first **predicts the weather regime** from forecast-time inputs only (forecast rainfall, 850 hPa wind, moisture flux convergence, MJO phase, terrain masks), so it is **deployable, not a hindsight label**.
3. It then applies **a bias correction fitted to that regime** (per-regime empirical quantile mapping) with a **tail rule that keeps heavy-rain values** instead of clipping them.
4. A **calibrated exceedance model** gives the probability of **heavy (≥64.5 mm/day)** and **very-heavy (≥115.6 mm/day)** rainfall, the IMD operational thresholds.
5. Output is a **district-level table and map** an officer can act on, plus a **verification report with all six metrics the PS names: RMSE, ETS, CSI, POD, FAR, FSS**, raw vs corrected, under **leave-one-monsoon-out** testing.

### Right: icon-hub infographic

Centre icon: India map with rain cloud. Six icons around it, each with a two-line label:

| Position | Label |
|---|---|
| Top-left | Regime Classifier (active / break / depression / coastal / orographic) |
| Top-right | Per-Regime Bias Correction |
| Mid-left | Heavy-Rain Probability (calibrated) |
| Mid-right | District Table and Map |
| Bottom-left | Six-Metric Verification Report |
| Bottom-right | Forecast-Time Inputs Only (no leakage) |

### "Innovation and uniqueness" strip (template pointer, one line under the pills or as a 6th slim pill)

**Uniqueness = rigour, not a new algorithm:** classify-then-correct, all six PS metrics, a leakage-proof split, and reported failure cases. State it this way. Do not write "novel AI breakthrough".

---

## Slide 3 — Technical approach

**Slide title:** TECHNICAL APPROACH

### Left column: Techstack used

1. Python (xarray, zarr, NumPy)
2. LightGBM (regime classifier + exceedance models)
3. scikit-learn (isotonic calibration, metrics)
4. Empirical quantile mapping (custom experts per regime × terrain class × zone × lead, GPD tail, shrinkage)
5. GeoPandas (district area-weighted aggregation)
6. FastAPI (serves run artefacts)
7. MapLibre GL (district map + scorecard)
8. Open data: WeatherBench2 (IFS HRES), IMD 0.25° gridded rainfall (CHIRPS 2.0 fallback), IBTrACS, MJO RMM, NOAA GFS

Logos row (as in winning deck): Python, LightGBM, scikit-learn, FastAPI, MapLibre.

Below the stack, template-style lines:
- **Prototype:** *(link, only if one exists by submission; otherwise delete the line)*
- **Prototype video link:** *(same rule)*

### Right: flow diagram (draw as boxes and arrows, like the winning deck)

Top row, left to right:

1. **Inputs**: forecast (IFS HRES / NOAA GFS), IMD gridded rainfall (CHIRPS fallback), IBTrACS, MJO RMM, terrain + coast masks
2. **Ingestion**: clip to India box (6.5–38.5°N, 66.5–100°E), 0.25° grid, mm/day, IMD rain day (03–03 UTC)
3. **Regime labelling** *(offline, training only)*, on two axes: **synoptic** (active / break / depression / normal: core-monsoon-zone rain anomaly ±1σ for ≥3 days; depression within 500 km from IBTrACS) × **geographic** (orographic / coastal / plains, fixed from terrain and coastline)

Middle: **Regime classifier** (multi-class LightGBM, forecast-time features only) outputs a **probability per synoptic regime**. The geographic class is fixed per cell.

The classifier output splits into two parallel branches:

- **Branch A: Regime-blended quantile mapping.** One curve per (regime, terrain class, zone, lead), shrunk toward its parent curve when data is thin, with a GPD tail so heavy values are not clipped. Curves are blended by regime probability. Output: **corrected rainfall grid**.
- **Branch B: Exceedance model.** Two GBM classifiers (heavy, very-heavy), isotonic calibration. Output: **P(heavy), P(very-heavy) grid**.

Both branches join, then:

- **District aggregation.** Area-weighted mean **and** max per district (mean can hide a local heavy cell). Output: **district table + choropleth map**.
- **Verification module.** RMSE, ETS, CSI, POD, FAR, FSS (75 km and 125 km windows) for **raw vs corrected**, at both thresholds, plus reliability diagram. Output: **verification report**.

Feedback loop (mirror the winning deck's "continuous learning" loop box):
**Verification report → weak regimes / low-sample regimes flagged → refit curves and classifier → re-verify.**

Add a small labelled bracket around the flow: "Leave-one-monsoon-out CV: each June–September season held out whole, so no season leaks into its own training."

---

## Slide 4 — Feasibility and viability

**Slide title:** FEASIBILITY AND VIABILITY

Table, three columns exactly as the winning deck (cyan | red | pink):

| Analysis of the feasibility of the idea | Potential challenges and risks | Strategies for overcoming these challenges |
|---|---|---|
| **Technical: Achievable** | Rare regimes (e.g. depression) have few historical days, so the quantile curve is noisy | Tabular-scale **LightGBM + empirical quantile mapping** (trains in minutes, no GPU). **Report the sample count behind every regime curve**. Blend the top-2 regimes near transitions if time allows |
| **Data: Available and open** | NCMRWF's NCUM/NEPS-G output is not public. IMD gridded rainfall may not be reachable | Validate on **open IFS HRES + ERA5 with CHIRPS truth**. Provide an **adapter interface for NCUM**. IMD gridded rainfall is a drop-in truth upgrade if reachable, not a dependency |
| **Accuracy: Verifiable** | Bias-correction gains are typically small. Regime labels can leak hindsight into a forecast-time model | **Leave-one-monsoon-out CV**. A **unit test rejects any truth-derived feature**. All six metrics reported, **including where correction does not help** |
| **Operational: Practical** | District-boundary licence. Pipeline must run quickly for daily use | Use **GADM** boundaries with attribution (licence checked). **No retraining in daily mode**: load fitted models, run correction, aggregation and verification only |

Bottom text, same style as the winning deck:

Given that feasibility and viability are our top priority, we will keep everything:

- **Technically viable:** proven, open-source tools (LightGBM, xarray, FastAPI), no deep model, no GPU.
- **Economically feasible:** 100% open data, runs on a laptop or a small cloud VM.
- **Operationally practical:** one pipeline, fixed seeds and hashed models, every output traceable to a run artefact, district table a disaster-management officer can read.

---

## Slide 5 — Impact and benefits

**Slide title:** IMPACT AND BENEFITS

Keep all claims qualitative. No invented percentages. Wording is "expected" or "supports", never "achieves".

### Left panel (pink): Potential impact on target audience

1. **NCMRWF / IMD forecasters**
   - See which regime was predicted and which correction was applied, so every change is auditable.
   - Get skill against raw NWP on the six metrics they already use.
2. **District disaster-management officers**
   - Read corrected rainfall and heavy-rain probability per district in mm/day.
   - Mean and max per district show localised heavy cells.
3. **State agencies (water, flood, agriculture, urban)**
   - Better-targeted heavy and very-heavy rainfall guidance at district scale.
4. **Model developers and researchers**
   - Per-regime error curves show where NWP fails, feeding model improvement.
5. **Scientific reviewers**
   - Leakage-proof split and reported failure cases make the results trustworthy.

### Right panel (cyan): Benefits of the solution

1. **Social**
   - Earlier, sharper warning of heavy rain for at-risk districts.
   - Clear probability, not just a rainfall number, supports evacuation and response decisions.
2. **Economic**
   - Fewer losses from missed heavy-rain events and from false alarms.
   - Zero data cost: all inputs are open.
3. **Environmental**
   - Runs on tabular models on a laptop, so low compute and energy.
   - Better flood and landslide preparedness in orographic and coastal belts.
4. **Technological**
   - Reusable classify-then-correct pipeline for any NWP model.
   - Reproducible runs (fixed seed, hashed models, manifest file).
5. **Strategic**
   - Supports MoES/NCMRWF post-processing goals with an NCUM-ready adapter.
   - Open methods that can be extended to other regions and seasons.

### Bottom row: three visuals (like the winning deck's chart, pie and image)

1. **Conceptual chart, labelled "Illustrative, not measured":** two quantile-mapping curves. One flat grey line "single global curve", and three coloured lines "active / break / depression". Shows why one curve mis-corrects.
2. **IMD threshold ladder:** vertical bar with two marks, **Heavy ≥ 64.5 mm/day** and **Very heavy ≥ 115.6 mm/day**. Caption: "The events our exceedance probability targets."
3. **District product mockup:** small India map with a few district polygons shaded by heavy-rain probability and one table row (`district | corrected mean | corrected max | P(heavy) | P(very heavy)`). Use placeholders or clearly dummy shading. Caption: "Output format".

---

## Slide 6 — Research and references

**Slide title:** RESEARCH AND REFERENCES

### Research survey table (cyan, three columns: Aspect | Conventional systems | Proposed system)

| Aspect | Conventional systems | Proposed system |
|---|---|---|
| Correction | One global curve or scaling factor for all days | Curves **per weather regime**, blended by regime probability |
| Regime awareness | None | **Classify first**, then correct |
| Regime inputs | Hindsight labels | **Forecast-time features only** |
| Heavy rain | Mean bias focus, tails clipped | **GPD tail + calibrated P(heavy)** |
| Verification | Often RMSE alone | **RMSE, ETS, CSI, POD, FAR, FSS**, leave-one-monsoon-out |
| Output | Gridded file | **District table + map** |
| Reporting | Best-case results | **Failure cases stated** |

### Reference: research papers

Check every citation and URL against the source before you export. Do not submit one you have not opened.

1. Wheeler, M. C. and Hendon, H. H., "An All-Season Real-Time Multivariate MJO Index," *Monthly Weather Review* 132 (2004).
2. Rajeevan, M., Gadgil, S., Bhate, J., "Active and break spells of the Indian summer monsoon," *J. Earth System Science* 119 (2010).
3. Themeßl, M. J., Gobiet, A., Leuprecht, A., "Empirical-statistical downscaling and error correction of daily precipitation from regional climate models," *Int. J. Climatology* 31 (2011).
4. Roberts, N. M. and Lean, H. W., "Scale-selective verification of rainfall accumulations from high-resolution forecasts of convective events," *Monthly Weather Review* 136 (2008). *(Origin of FSS)*
5. Funk, C. et al., "The climate hazards infrared precipitation with stations (CHIRPS)," *Scientific Data* 2 (2015).
6. Rasp, S. et al., "WeatherBench 2: A benchmark for the next generation of data-driven global weather models," *J. Advances in Modeling Earth Systems* (2024).
7. Ke, G. et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," *NeurIPS* (2017).

### Dataset links (three-across row, as in the winning deck)

- [1] WeatherBench2 (ERA5, IFS HRES): https://sites.research.google/weatherbench/
- [2] IMD 0.25° gridded rainfall (truth): https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html · CHIRPS 2.0 (fallback): https://www.chc.ucsb.edu/data/chirps
- [3] IBTrACS depression tracks: https://www.ncei.noaa.gov/products/international-best-track-archive
- [4] BoM MJO RMM index: http://www.bom.gov.au/climate/mjo/
- [5] NOAA GFS open data (AWS): https://registry.opendata.aws/noaa-gfs-bdp-pds/
- [6] GADM district boundaries: https://gadm.org

Six links do not fit in one row: use two rows of three.

---

## Hard rules for whoever builds the deck

1. **No invented numbers.** Allowed: 64.5 and 115.6 mm/day, 0.25°, six metric names, FSS windows of 1/3/5/9 cells (≈25/75/125/225 km), ±1σ, ≥3 days, 500 km, and pilot numbers copied from `PILOT_RESULTS.md`. Anything else needs a source in `../PRD.md` or `../TECHNICAL_SPEC.md`.
2. **Do not write** "novel", "breakthrough", or any improvement percentage. The claim is rigour: six metrics, leakage-proof split, failure cases reported.
3. **Say IMD 0.25° gridded rainfall** for truth (used in pilot v1, `PILOT_RESULTS.md`), with CHIRPS as the fallback.
4. **Say we validate on open models** (IFS HRES for training, GFS for the live demo). Do not imply NCUM/NEPS-G was used.
5. The probability-weighted regime blend is core design (PRD Rev B §10.2). Describe it as designed and piloted; do not claim it beats global QM until a backtest report shows it.
6. No live-demo content, team-planning notes, or references to other PS numbers on any slide.
7. Keep the winning deck's density: max about 5 pills or 5 numbered groups per slide, one line each where possible.
8. Export PDF, check it is at most 6 pages, and that the "Important Pointers" slide is gone.

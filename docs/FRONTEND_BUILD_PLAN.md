# Frontend Build Plan (SIH26080)

**Owner: Shreyash.** Scope: the web application: landing, forecast map, scorecard, method, bulletin.

Requirements come from `../PRD.md` §6, §9 (FR-5, FR-7, FR-8, FR-9), §11–§13, §16 (NFR-1 to NFR-4), §19 (AC-4, AC-5, AC-6) and `../TECHNICAL_SPEC.md` §3.3, §4.8, §4.9, §5. Adapted from the PS26167 (SatQuery) frontend plan. Code lives in `../mvp/web/`.

> **Direction override (29 Sep, by instruction).** PRD §13 and §15 ask for the plain "Field Atlas" look. This plan keeps its **light theme** and map-first framing but adds bold display type, heavy data-driven motion, and a Ventusky-style map workstation. Every other PRD rule holds, especially **NFR-2: no number on screen that is not read from a run artefact.**

---

## 1. The brief, stated plainly

**Be unique. Do not build a boring website.** No SaaS dashboard. No dark navy with a purple gradient. No glass cards. No hero with three feature tiles. NCMRWF judges look at weather maps all day. Give them one that is better to use than the ones they know, then show the scorecard.

**Light theme, not dark.** Rainfall colour ramps read truer on a light ground, and most weather sites are dark.

**Bold, but not minimalist.** Enormous display type, oversized numerals, the map full-bleed like a photograph. Fill the frame with the product's own material: the rainfall field, the monsoon wind, regime labels, skill scores.

**Animate a lot, but only real things.** Rain fields develop, wind particles follow the 850 hPa flow, lead days play forward, numbers count to their value from the run. Motion that shows data reads as an instrument. Motion that shows nothing reads as a template.

**Borrow patterns, not looks.** Take solved components from the libraries in §4, restyle to the tokens, credit them in `mvp/web/CREDITS.md`.

---

## 2. Theme: "Monsoon Chart Room"

A forecaster's plotting table: the rainfall chart pinned flat, wind arrows sketched over it, the day's regime written in the margin, the verification scores ruled underneath.

### 2.1 The signature move

**The forecast is the headline.**

- The rainfall field of India is **full-bleed**, edge to edge, with the 850 hPa monsoon wind drawn as moving particles over it.
- A **huge headline** sits across it: Bricolage Grotesque, weight 800, tracking −0.04em, `clamp(56px, 11vw, 200px)`.
- **The run's own numbers are a second headline at the same scale**, in IBM Plex Mono with tabular numerals: the wettest point's coordinate, its corrected mm/day, its P(heavy), its regime, the lead time. Nobody else gives a forecast value the weight of a slogan.

### 2.2 Tokens

| Token | Hex | Use |
|---|---|---|
| `--paper` | `#F2F4F1` | page ground, never pure white |
| `--surface` | `#FFFFFF` | raised surfaces |
| `--ink` | `#0E1A1F` | display type, coastline |
| `--ink-2` / `--ink-3` | `#46565C` / `#7A878B` | secondary / small labels (≥ 11 px) |
| `--sea` | `#DCE5E6` | map sea |
| `--land` | `#EEF0EA` | map land |
| `--accent` | `#0B6E7F` | **interaction only**: links, focus, primary action |
| `--worse` | `#C0263D` | metric got worse, failure, refusal only |
| `--good` / `--warn` | `#237A57` / `#A1740F` | semantic states |
| Regime marks | active `#1B7F79`, break `#C9A227`, depression `#B3261E`, coastal `#3C6E9E`, orographic `#7A5C9E`, other `#9AA5A6` | dots, rules, the regime map layer |

Rain ramp and probability ramp are defined once in `src/lib/color.ts`; the heavy and very-heavy breaks are read from `manifest.thresholds_mm` (64.5 / 115.6, BACKEND_BUILD_PLAN D2), never hard-coded. The heavy and very-heavy breaks sit exactly on IMD's thresholds so the colour change *is* the threshold.

Rules: one accent carries interaction. Regime hues are marks, not page fills (the regime map layer is the one exception). Red means worse, nothing else.

### 2.3 Type

| Role | Face | Setting |
|---|---|---|
| Display | Bricolage Grotesque (variable) | 800, −0.04em. Hero `clamp(56px, 11vw, 200px)`; section `clamp(36px, 6vw, 96px)` |
| Body / UI | IBM Plex Sans | 15–16 px / 1.55 |
| Data | IBM Plex Mono | `tabular-nums` always; runs at display scale for the signature move |

Fonts self-hosted through `@fontsource`. No runtime CDN.

### 2.4 Layout

Editorial grid for story pages. **Ventusky-style workstation** for the map: map fills the screen, controls float over it.

---

## 3. Motion plan

| Moment | Motion | Tied to |
|---|---|---|
| Landing load | Rainfall field **develops from the south-west to the north-east**, the way the monsoon advances | Run's corrected field |
| Always on map | Wind particles advected along the 850 hPa field | Run's `u850`, `v850` |
| Headline | Line-mask reveal of display type | — |
| Telemetry | Numbers count up to the run value | Run artefact values |
| Capability strip | Marquee of real terms: metric names, thresholds, grid, datasets | PRD §14, §17 |
| Landing scroll | Pinned map changes layer per step: raw → regime → corrected → P(heavy) → scorecard | The pipeline order in PRD §8 |
| Timeline play | Lead days advance, field crossfades, depression moves | Run lead days |
| Raw vs corrected | Drag-to-compare swipe | FR-8 |
| Scorecard | Bars grow from the raw value to the corrected value, so the baseline stays visible | `verification.json` |
| Worse metric | Row lands in `--worse`, stays visible | AC-6, NFR-4 |

**Non-negotiable:** `prefers-reduced-motion` makes every transition instant and stops particles and marquee. No animation blocks input.

---

## 4. Component sourcing

| Source | Take |
|---|---|
| **Ventusky** (reference, not code) | Map UI grammar: left layer rail, bottom timeline with play, bottom-right legend bar, value under the cursor, place labels with values, point panel on click |
| **Magic UI** | Number Ticker, Marquee (pattern adapted to Motion + tokens) |
| **Motion Primitives** | Image comparison (the raw vs corrected swipe) |
| **React Bits / Aceternity** | Text reveal, spotlight hover (restyled) |
| **shadcn/ui** | Tabs, dialog, toast patterns if needed |
| **MapLibre GL** | Map engine, no tile server |
| **Natural Earth** (public domain) | Land polygons for coastline and land mask. No political boundaries are drawn |

Rules: restyle everything to tokens; skip fingerprinted shader backgrounds (Silk, Iridescence, LiquidChrome); list every borrowed piece in `CREDITS.md`; bundle everything locally.

---

## 5. Pages

### 5.1 Landing (`/`)
- **Hero:** full-bleed rainfall field with wind particles, headline "Classify first. Correct second.", telemetry headline from the run's wettest point.
- **Marquee** of the real vocabulary.
- **Scroll narrative:** the five pipeline stages on a pinned map (PRD §8).
- **Where it didn't help:** the worst metric delta from `verification.json`, in red.
- **Numbers band:** skill deltas, shown as `—` with "measured after backtest" while the run is synthetic.
- **Disclosure line**, permanent, when the run is synthetic.

### 5.2 Forecast (`/forecast`) — the workstation, Ventusky-style
- Full-screen map: sea, land, coastline, rainfall field, wind particles, place labels with values.
- **Left rail:** layers: Rainfall, Regime, Heavy chance, Very heavy chance, Wind.
- **Top bar:** place search; Raw / Corrected / Compare toggle (FR-8).
- **Bottom:** lead-day timeline with play (FR-7).
- **Bottom-right:** legend bar, thresholds marked.
- **Hover:** value under cursor.
- **Click:** point panel: raw vs corrected across lead days, regime probabilities (FR-2), and the **correction trace**: forecast value → predicted regime → quantile-mapping curve used (with sample count, PRD §10.2) → corrected value.

### 5.3 Scorecard (`/scorecard`)
All six metrics (FSS at two scales), raw vs corrected, both thresholds (FR-6, AC-3). Per-regime delta matrix with worse cells in red (AC-6). Per-season folds (PRD §17.2). Reliability diagram (§17.3). Quantile-mapping curves per regime with sample counts.

### 5.4 Method (`/method`)
Data sources with real access status (PRD §14), the three model components and their state, IMD thresholds, known limits (§20).

### 5.5 Bulletin (`/bulletin`)
Print-styled place table for a chosen lead day: corrected rain, P(heavy), P(very heavy), regime; plus "what this cannot establish". District rows replace place rows once the boundary licence is cleared (PRD §20).

---

## 6. Engineering

- **Stack:** React + TypeScript, Vite, Tailwind v4 with the tokens above, Motion, MapLibre GL, react-router.
- **Data contract:** the frontend reads `public/run/<run_id>/` (`manifest.json`, `grid.json`, `places.json`, `verification.json`, `qm_curves.json`). The same files come from the API (PRD §12) once the backend exists. The UI only formats; it never computes a metric (TECHNICAL_SPEC §4.9).
- **Map:** live OpenStreetMap raster tiles under the field when online (attribution always shown, nothing prefetched); plain Natural Earth land when offline. The field is drawn to a canvas in Web Mercator rows and placed as an image source.
- **Budget:** lazy-load the map page. Zero console errors.
- **Quality floor:** WCAG AA text; focus rings in `--accent`; keyboard reachable; regime never colour-only (label always present).
- **Checks:** `data-testid` hooks for landing load, layer switch, timeline play, point panel, compare swipe, scorecard render.

---

## 7. Calendar (from 29 Sep; finale date TBC)

| When | Deliverable |
|---|---|
| Week 1 | Tokens, fonts, synthetic run v2 (lead days, wind, places, curves), landing, forecast workstation |
| Week 2 | Scorecard, method, bulletin; reduced-motion and keyboard pass |
| Week 3 | Wire to real `regimerain` run artefacts; remove sample flag for real runs |
| Before finale | District boundaries (once licence cleared), live `POST /runs/live`, venue-laptop test |

---

## 8. Definition of done

- [ ] Light theme; no dark tokens
- [ ] Hero with real run values at display scale
- [ ] Motion tied to run data; reduced motion honoured
- [ ] Ventusky-grade map: layers, timeline, legend, hover value, point panel, compare swipe
- [ ] Correction trace visible for any point
- [ ] Scorecard with all six metrics, worse deltas in red
- [ ] Synthetic banner whenever `manifest.synthetic` is true
- [ ] Borrowed pieces credited in `CREDITS.md`
- [ ] Zero console errors; works offline after build (basemap falls back to plain land; `npm run smoke` runs with external requests blocked)

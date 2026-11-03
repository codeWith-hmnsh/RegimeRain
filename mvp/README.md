# RegimeRain MVP

Frontend first. It renders a run folder; it never computes a forecast or a metric. Plan: `../docs/FRONTEND_BUILD_PLAN.md`.

```
mvp/
  tools/make_sample_run.py     writes a SYNTHETIC run + coastline for UI work (numpy only)
  web/                         React + TS + Vite app
    public/run/<run_id>/       manifest.json, grid.json, places.json, qm_curves.json, verification.json
    public/geo/land.json       Natural Earth land, coastline only
    src/pages/                 Landing, Forecast (map workstation), Scorecard, Method, Bulletin
    tests/smoke.mjs            headless checks via data-testid
```

## Run

```bash
cd mvp/web
npm install
npm run sample            # writes the synthetic run through regimerain's mock producer (needs numpy + pydantic)
npm run dev               # http://localhost:5173
```

Production check:

```bash
npm run build && npm run preview &     # http://localhost:4173
CHROME=/usr/bin/google-chrome npm run smoke
```

Fonts, MapLibre and run data are local. The street basemap is live OpenStreetMap tiles: with no internet the map falls back to plain land (Natural Earth) and the rainfall field still renders. `npm run smoke` blocks every external request by default to prove this (`ONLINE=1` lets tiles through). `?run=<run_id>` loads another run folder.

On Windows, point `CHROME` at Chrome, e.g. `CHROME="C:/Program Files/Google/Chrome/Application/chrome.exe"`, and run `npm run preview` in a second terminal instead of `&`.

## Pages

| Route | What it shows |
|---|---|
| `/` | Hero rainfall field with 850 hPa wind particles, the run's wettest point at display scale, pipeline story on a pinned map, where correction did not help, headline scores |
| `/forecast` | Ventusky-style map: layer rail (rain, regime, heavy, very heavy, wind), raw / corrected / compare swipe, lead-day timeline with play, legend, hover value, place labels, depression track, point panel with correction trace |
| `/scorecard` | Six metrics raw vs corrected, per season, per regime (worse in red), QM curves, reliability |
| `/method` | Models, data sources and access state, known limits |
| `/bulletin` | Print-styled place table for a lead day |

Keyboard on the map: Shift+←/→ changes lead day, Esc closes the point panel, arrows move the compare handle.

## Contract with the pipeline (web contract v2)

Every run, mock or real, is written by one exporter in the `regimerain` package (`regimerain/runs/`, plan: `../docs/BACKEND_BUILD_PLAN.md`). The shapes are pydantic models in `regimerain/runs/contract.py`; `web/src/lib/types.ts` mirrors them and `tests/test_runs.py` fails if the two drift. `regimerain contract --check <dir>` validates a folder.

- `manifest.json`: `contract_version: 2`, `kind` (`mock` / `replay` / `model`; `synthetic` is derived from it), thresholds from config (64.5 / 115.6), `layers` actually present, `depression_track` (MSLP minimum), `wettest` point, provenance (model set, hashes, config hash, backtest id).
- `grid.json`: 0.25° rain domain, rows south to north. Per rain day: `raw`, `corrected` (regime-aware, variant B), `corrected_global` (variant A), `p_heavy`, `p_very_heavy`, `truth` (null off land); `regime` (most probable of active / break / depression / normal); static `geo` (plains / coastal / orographic); `u850`, `v850`, `wind850`; `regime_probs_pct` (4 per cell). Any layer can be absent: absent means not built, and the UI hides it.
- `places.json`: named places, nearest land cell, terrain class, values per rain day, the regime curve used and its training days.
- `qm_curves.json`: per-regime (variant B) and global (variant A) quantile curves with `n_days`. Optional.
- `verification.json`: from the backtest report: `entries` per fold × rain day × threshold (pooled, per season, per regime) with raw / corrected / global scores, FSS per window (`fss_windows`, km from the grid step), bootstrap `deltas`, `reliability`. Optional.

## Known gaps

- Sample values, including every score, are invented. The banner, bulletin and footer say so.
- District polygons wait on the boundary licence (PRD §20). Places stand in.
- Backend: mock producer and web export built (phases B0–B1). API, replay, model and live runs are next: `../docs/BACKEND_BUILD_PLAN.md`.

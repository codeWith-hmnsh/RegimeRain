# Credits

What was borrowed, from where, and how it was changed. If a judge asks "did you build this?", this file is the answer.

## Libraries (npm, bundled locally)

| Package | Use | Licence |
|---|---|---|
| React, React DOM | UI | MIT |
| react-router-dom | Routing | MIT |
| Motion (`motion/react`) | All animation | MIT |
| MapLibre GL JS | Forecast map engine | BSD-3-Clause |
| Tailwind CSS v4 | Styling with project tokens | MIT |
| @fontsource: Bricolage Grotesque, IBM Plex Sans, IBM Plex Mono | Self-hosted fonts | OFL-1.1 |

## Data

| Source | Use | Licence |
|---|---|---|
| Natural Earth 1:50m land | Coastline and land fill (`public/geo/land.json`, built by `tools/make_sample_run.py`). Plain ground when the street map is off or offline. | Public domain |
| Survey of India boundary via datameet/maps, simplified | India outline (`public/geo/india.json`), drawn on top of any basemap because OSM tiles show the de facto Line of Control in J&K; also the in/out-of-India mask for the home hero | CC BY 2.5 IN |
| OpenStreetMap raster tiles (tile.openstreetmap.org) | Street map under the field on the home hero and forecast map. Loaded live for the view on screen only; never prefetched or stored for offline use (OSM tile usage policy). Attribution always shown. | © OpenStreetMap contributors, ODbL |

## Patterns adapted (written here, restyled to tokens)

| Component in this repo | Adapted from | Change |
|---|---|---|
| `components/NumberTicker.tsx` | Magic UI "Number Ticker" | Rewritten on Motion `animate`; counts from a baseline value, not only from 0; honours reduced motion |
| `components/Marquee.tsx` | Magic UI "Marquee" | CSS-only, stops under reduced motion |
| `components/Reveal.tsx` | React Bits / Motion Primitives text reveals | Line-mask reveal with the in-view trigger on the unclipped wrapper |
| Compare swipe in `pages/Forecast.tsx` | Motion Primitives "Image Comparison" | Two synced MapLibre maps with a clip-path, keyboard operable |
| Forecast workstation layout | Ventusky (reference only, no code) | Left layer rail, bottom timeline with play, legend bar, value under cursor, place labels with values, point panel |
| Wind particles in `lib/field.ts` | Common "earth"/Ventusky particle technique | Written from scratch; advects on the run's own 850 hPa u/v grid |
| OSM basemap in `lib/osm.ts`, `FieldStage.tsx`, `useForecastMaps.ts` | Our SIH/081 `WeatherMap.tsx` | Ported to this stack: canvas tiles for the hero, a MapLibre raster source on the forecast map |

## Built here

Field renderer (Web Mercator rows, land clip, edge fade), point panel with correction trace, scorecard charts (QM curves, delta matrix, season dumbbells, reliability), bulletin, sample-run generator. Chart colours were checked with the dataviz palette validator.

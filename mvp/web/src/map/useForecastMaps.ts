import * as maplibregl from 'maplibre-gl'
type MLMap = maplibregl.Map
import 'maplibre-gl/dist/maplibre-gl.css'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import { useEffect, useRef, useState } from 'react'
import { dataId, fieldData, gridBounds, renderField, type FieldOpts } from '../lib/field'
import { OSM_TILES } from '../lib/osm'
import type { FieldLayer, Run } from '../lib/types'

maplibregl.setWorkerUrl(workerUrl)

const urlCache = new WeakMap<HTMLCanvasElement, string>()
const toUrl = (c: HTMLCanvasElement) => {
  let u = urlCache.get(c)
  if (!u) urlCache.set(c, (u = c.toDataURL()))
  return u
}
/** Field opacity over plain ground, and over the street map (place names show through). */
const FIELD_PLAIN = 0.9, FIELD_OVER_TILES = 0.75
const INDIA: [[number, number], [number, number]] = [[67.5, 6], [97.5, 36.5]]

function trackGeo(run: Run): GeoJSON.FeatureCollection {
  const pts = run.manifest.depression_track
  return {
    type: 'FeatureCollection',
    features: [
      { type: 'Feature', properties: {}, geometry: { type: 'LineString', coordinates: pts.map((p) => [p.lon, p.lat]) } },
      ...pts.map((p) => ({ type: 'Feature' as const, properties: { lead: p.lead }, geometry: { type: 'Point' as const, coordinates: [p.lon, p.lat] } })),
    ],
  }
}

type FieldState = { active: 'A' | 'B'; key: string; opacity: number }
const fieldState = (map: MLMap) => (map as MLMap & { _field: FieldState })._field

/** Over the street map the field is feathered out at the grid edge; over plain ground it stops at a dashed outline. */
const fieldOpts = (tiles: boolean): FieldOpts => ({ edge: tiles ? 'feather' : 'hard' })
const fieldKey = (run: Run, layer: FieldLayer, lead: number, tiles: boolean) =>
  `${dataId(fieldData(run.grid, layer, lead))}:${layer}:${fieldOpts(tiles).edge}`

function extentGeo(run: Run): GeoJSON.Feature {
  const b = gridBounds(run.grid)
  return { type: 'Feature', properties: {}, geometry: { type: 'LineString', coordinates: [[b.west, b.north], [b.east, b.north], [b.east, b.south], [b.west, b.south], [b.west, b.north]] } }
}

/**
 * Street map on or off. `tiles` is whether tiles are actually showing (on, and not failed): without
 * them the plain ground, coastline and dashed data extent come back.
 */
export function setBasemap(map: MLMap, on: boolean, tiles: boolean) {
  // callers wait for 'load' (see setField on isStyleLoaded)
  map.setLayoutProperty('osm', 'visibility', on ? 'visible' : 'none')
  map.setLayoutProperty('coast', 'visibility', tiles ? 'none' : 'visible')
  map.setLayoutProperty('extent', 'visibility', tiles ? 'none' : 'visible')
}

export function createMap(container: HTMLElement, run: Run, first: FieldLayer, lead: number, tiles: boolean): MLMap {
  const b = gridBounds(run.grid)
  const coords: [[number, number], [number, number], [number, number], [number, number]] = [[b.west, b.north], [b.east, b.north], [b.east, b.south], [b.west, b.south]]
  const url = toUrl(renderField(run.grid, run.land, first, lead, fieldOpts(tiles)))
  const opacity = tiles ? FIELD_OVER_TILES : FIELD_PLAIN
  const map = new maplibregl.Map({
    container,
    bounds: INDIA,
    fitBoundsOptions: { padding: { top: 70, bottom: 110, left: 90, right: 40 } },
    minZoom: 2.8,
    maxZoom: 8.5,
    maxBounds: [[40, -8], [112, 46]],
    dragRotate: false,
    pitchWithRotate: false,
    renderWorldCopies: false,
    // attribution is drawn by the page, above the timeline, so it is never collapsed or covered
    attributionControl: false,
    style: {
      version: 8,
      sources: {
        land: { type: 'geojson', data: run.land },
        // live tiles only; never prefetched or stored for offline use (OSM tile usage policy)
        osm: { type: 'raster', tiles: [OSM_TILES], tileSize: 256, maxzoom: 19 },
        india: { type: 'geojson', data: run.india },
        extent: { type: 'geojson', data: extentGeo(run) },
        fieldA: { type: 'image', url, coordinates: coords },
        fieldB: { type: 'image', url, coordinates: coords },
        track: { type: 'geojson', data: trackGeo(run) },
      },
      layers: [
        { id: 'sea', type: 'background', paint: { 'background-color': '#dce5e6' } },
        { id: 'land', type: 'fill', source: 'land', paint: { 'fill-color': '#eef0ea' } },
        // OSM under the field, desaturated like the canvas maps' filter saturate(0.45) contrast(0.92) brightness(1.03)
        { id: 'osm', type: 'raster', source: 'osm', layout: { visibility: tiles ? 'visible' : 'none' }, paint: { 'raster-saturation': -0.55, 'raster-contrast': -0.08, 'raster-brightness-min': 0.03 } },
        { id: 'fieldA', type: 'raster', source: 'fieldA', paint: { 'raster-opacity': opacity, 'raster-fade-duration': 0, 'raster-resampling': 'linear', 'raster-opacity-transition': { duration: 420, delay: 0 } } },
        { id: 'fieldB', type: 'raster', source: 'fieldB', paint: { 'raster-opacity': 0, 'raster-fade-duration': 0, 'raster-resampling': 'linear', 'raster-opacity-transition': { duration: 420, delay: 0 } } },
        { id: 'coast', type: 'line', source: 'land', layout: { visibility: tiles ? 'none' : 'visible' }, paint: { 'line-color': '#0e1a1f', 'line-width': ['interpolate', ['linear'], ['zoom'], 3, 0.6, 7, 1.4], 'line-opacity': 0.75 } },
        { id: 'extent', type: 'line', source: 'extent', layout: { visibility: tiles ? 'none' : 'visible' }, paint: { 'line-color': '#0e1a1f', 'line-opacity': 0.4, 'line-width': 1, 'line-dasharray': [4, 4] } },
        // OSM shows the de facto line in J&K; the Survey of India outline always goes on top
        { id: 'india-halo', type: 'line', source: 'india', layout: { 'line-join': 'round' }, paint: { 'line-color': '#f8f7f5', 'line-opacity': 0.8, 'line-width': 3 } },
        { id: 'india-line', type: 'line', source: 'india', layout: { 'line-join': 'round' }, paint: { 'line-color': '#0e2129', 'line-width': 1.3 } },
        { id: 'track-line', type: 'line', source: 'track', filter: ['==', ['geometry-type'], 'LineString'], paint: { 'line-color': '#0e1a1f', 'line-width': 1.5, 'line-dasharray': [2, 2] } },
        { id: 'track-pt', type: 'circle', source: 'track', filter: ['==', ['geometry-type'], 'Point'], paint: { 'circle-radius': 4, 'circle-color': '#ffffff', 'circle-stroke-color': '#0e1a1f', 'circle-stroke-width': 1.5 } },
      ],
    },
  })
  map.touchZoomRotate.disableRotation()
  ;(map as MLMap & { _field: FieldState })._field = { active: 'A', key: fieldKey(run, first, lead, tiles), opacity }
  return map
}

/** Crossfades the field to a new layer/lead (or edge treatment) using the two image sources. */
export function setField(map: MLMap, run: Run, layer: FieldLayer, lead: number, tiles: boolean) {
  // callers wait for 'load'. No isStyleLoaded() guard: it is false whenever street-map tiles are
  // still loading, which would silently drop a layer or lead change.
  const st = fieldState(map)
  const opacity = tiles ? FIELD_OVER_TILES : FIELD_PLAIN
  if (st.opacity !== opacity) {
    st.opacity = opacity
    map.setPaintProperty(`field${st.active}`, 'raster-opacity', opacity)
  }
  // keyed on the data array, not only the layer name: a new array always gets a new image
  const key = fieldKey(run, layer, lead, tiles)
  if (st.key === key) return
  st.key = key
  const next = st.active === 'A' ? 'B' : 'A'
  const url = toUrl(renderField(run.grid, run.land, layer, lead, fieldOpts(tiles)))
  const img = new Image()
  img.src = url
  img.decode().then(() => {
    if (st.key !== key) return
    const b = gridBounds(run.grid)
    ;(map.getSource(`field${next}`) as maplibregl.ImageSource).updateImage({ url, coordinates: [[b.west, b.north], [b.east, b.north], [b.east, b.south], [b.west, b.south]] })
    requestAnimationFrame(() => requestAnimationFrame(() => {
      if (st.key !== key) return
      map.setPaintProperty(`field${next}`, 'raster-opacity', st.opacity)
      map.setPaintProperty(`field${st.active}`, 'raster-opacity', 0)
      st.active = next
    }))
  })
}

export function setTrack(map: MLMap, lead: number, visible: boolean) {
  for (const id of ['track-line', 'track-pt']) map.setLayoutProperty(id, 'visibility', visible ? 'visible' : 'none')
  map.setPaintProperty('track-pt', 'circle-radius', ['case', ['==', ['get', 'lead'], lead], 7, 3.5])
  map.setPaintProperty('track-pt', 'circle-color', ['case', ['==', ['get', 'lead'], lead], '#0e1a1f', '#ffffff'])
}

export type OsmStatus = 'pending' | 'ok' | 'failed'

/** Two synced maps (main + compare). Returns refs, a ready flag and whether street-map tiles are loading. */
export function useForecastMaps(run: Run, a: React.RefObject<HTMLDivElement | null>, b: React.RefObject<HTMLDivElement | null>, first: FieldLayer, lead: number, basemap: boolean) {
  const maps = useRef<{ A: MLMap | null; B: MLMap | null }>({ A: null, B: null })
  const [ready, setReady] = useState(false)
  const [osm, setOsm] = useState<OsmStatus>('pending')
  useEffect(() => {
    const A = createMap(a.current!, run, first, lead, basemap)
    const B = createMap(b.current!, run, 'corrected', lead, basemap)
    maps.current = { A, B }
    // any tile loaded: tiles work. Only errors so far: offline or blocked, fall back to plain ground.
    let loaded = 0
    A.on('sourcedata', (e) => { if (e.sourceId === 'osm' && e.tile && !loaded++) setOsm('ok') })
    // MapLibre does not repaint after a tile error, so with every tile failing 'load' would never fire
    const onError = (m: MLMap) => (e: maplibregl.ErrorEvent & { sourceId?: string }) => {
      if (e.sourceId !== 'osm') return console.error(e.error)
      m.triggerRepaint()
      if (m === A && !loaded) setOsm('failed')
    }
    A.on('error', onError(A))
    B.on('error', onError(B))
    let syncing = false
    const link = (from: MLMap, to: MLMap) => from.on('move', () => {
      if (syncing) return
      syncing = true
      to.jumpTo({ center: from.getCenter(), zoom: from.getZoom() })
      syncing = false
    })
    link(A, B)
    link(B, A)
    let n = 0
    const done = () => { if (++n === 2) setReady(true) }
    A.once('load', done)
    B.once('load', done)
    return () => { A.remove(); B.remove(); maps.current = { A: null, B: null }; setReady(false); setOsm('pending') }
    // maps are created once per run; layer/lead/basemap changes go through setField and setBasemap
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run])
  return { maps, ready, osm }
}

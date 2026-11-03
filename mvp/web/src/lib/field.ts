import { PROB, RAIN, REGIME_ORDER, WIND, rampRGB, regimeRGB, type Stop } from './color'
import type { FieldLayer, Grid } from './types'

export const mercY = (lat: number) => Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360))
export const invMercY = (y: number) => (360 / Math.PI) * Math.atan(Math.exp(y)) - 90

/** Outer edges of the data grid, in degrees. */
export function gridBounds(g: Grid) {
  const h = g.step / 2
  return { west: g.lon0 - h, east: g.lon0 + (g.nlon - 1) * g.step + h, south: g.lat0 - h, north: g.lat0 + (g.nlat - 1) * g.step + h }
}

/** Value of the nearest grid cell (what the hover readout and point panel show: a run value, not an interpolation). */
export function cellAt(g: Grid, lat: number, lon: number) {
  const i = Math.round((lat - g.lat0) / g.step), j = Math.round((lon - g.lon0) / g.step)
  if (i < 0 || j < 0 || i >= g.nlat || j >= g.nlon) return null
  return { i, j, idx: i * g.nlon + j, lat: g.lat0 + i * g.step, lon: g.lon0 + j * g.step }
}

/** Layers drawn on the rain ramp. */
export const isRain = (layer: FieldLayer) => layer === 'raw' || layer === 'corrected' || layer === 'corrected_global' || layer === 'truth'

export function stopsFor(layer: FieldLayer): Stop[] {
  if (isRain(layer)) return RAIN
  if (layer === 'wind850') return WIND
  return PROB
}

/** 0 at the edge of the data, 1 once FEATHER degrees inside (smoothstep): the field dissolves into the basemap. */
const FEATHER = 4
export function feather(g: Grid, lon: number, lat: number) {
  const b = gridBounds(g)
  const t = Math.min(1, Math.max(0, Math.min(lon - b.west, b.east - lon, lat - b.south, b.north - lat) / FEATHER))
  return t * t * (3 - 2 * t)
}

/** Bilinear sample for drawing only; ignores null (sea) corners. */
function bilinear(arr: ArrayLike<number | null>, g: Grid, fi: number, fj: number): number | null {
  const i0 = Math.floor(fi), j0 = Math.floor(fj)
  const ti = fi - i0, tj = fj - j0
  let s = 0, w = 0
  for (const [di, dj, ww] of [[0, 0, (1 - ti) * (1 - tj)], [0, 1, (1 - ti) * tj], [1, 0, ti * (1 - tj)], [1, 1, ti * tj]] as const) {
    const i = Math.min(g.nlat - 1, Math.max(0, i0 + di)), j = Math.min(g.nlon - 1, Math.max(0, j0 + dj))
    const v = arr[i * g.nlon + j]
    if (v != null && ww > 0) { s += v * ww; w += ww }
  }
  return w > 0.05 ? s / w : null
}

export type Land = GeoJSON.FeatureCollection

const ringsOf = (fc: Land) => fc.features.flatMap((f) => {
  const g = f.geometry as GeoJSON.Polygon | GeoJSON.MultiPolygon
  return g.type === 'Polygon' ? g.coordinates : g.coordinates.flat()
})

/** Traces (multi)polygons onto ctx given a lon/lat -> pixel projector. */
export function traceLand(ctx: CanvasRenderingContext2D, land: Land, project: (lon: number, lat: number) => [number, number]) {
  ctx.beginPath()
  for (const ring of ringsOf(land)) {
    ring.forEach(([x, y], k) => {
      const [px, py] = project(x, y)
      if (k === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    })
    ctx.closePath()
  }
}

const maskCache = new WeakMap<Grid, Float32Array>()
/** Per cell: 1 if the cell centre is inside the Survey of India outline, else 0 (even-odd over all rings). */
export function indiaMask(g: Grid, india: Land): Float32Array {
  const hit = maskCache.get(g)
  if (hit) return hit
  const rings = ringsOf(india)
  const m = new Float32Array(g.nlat * g.nlon)
  for (let i = 0; i < g.nlat; i++) {
    const lat = g.lat0 + i * g.step
    for (let j = 0; j < g.nlon; j++) {
      const lon = g.lon0 + j * g.step
      let inside = false
      for (const r of rings)
        for (let a = 0, b = r.length - 1; a < r.length; b = a++) {
          const [xa, ya] = r[a], [xb, yb] = r[b]
          if ((ya > lat) !== (yb > lat) && lon < ((xb - xa) * (lat - ya)) / (yb - ya) + xa) inside = !inside
        }
      m[i * g.nlon + j] = inside ? 1 : 0
    }
  }
  maskCache.set(g, m)
  return m
}

/** Field opacity at a point: feathered at the data edge, and `outside` (0..1) away from India, bilinear across the mask. */
export function emphasis(g: Grid, mask: Float32Array | null, outside: number) {
  return (lon: number, lat: number) => {
    let a = feather(g, lon, lat)
    if (mask && outside < 1) a *= outside + (1 - outside) * (bilinear(mask, g, (lat - g.lat0) / g.step, (lon - g.lon0) / g.step) ?? 0)
    return a
  }
}

export interface FieldOpts {
  /** 'feather' fades to 0 over the last few degrees (over a basemap); 'hard' stops at the grid edge (a dashed outline marks it). */
  edge?: 'feather' | 'hard'
  /** Opacity outside India, with the mask it is read from. */
  outside?: { mask: Float32Array; k: number }
}

/** The data array a layer/lead is drawn from. Images and keys hang off this, never off the layer name alone. */
export const fieldData = (g: Grid, layer: FieldLayer, lead: number): object => {
  const planes = g.layers[layer]
  if (!planes) throw new Error(`layer ${layer} is not in this run`)   // callers check manifest.layers first
  return planes[lead]
}
/** Whether a run has this layer (absent = not built yet, never zero). */
export const hasLayer = (g: Grid, layer: FieldLayer) => !!g.layers[layer]

let nextId = 0
const ids = new WeakMap<object, number>()
/** Stable id for a data array, so a string key changes whenever the data does. */
export const dataId = (o: object) => { let id = ids.get(o); if (id == null) ids.set(o, (id = ++nextId)); return id }

const imgCache = new WeakMap<object, Map<string, HTMLCanvasElement>>()

/**
 * Draws one layer of one lead day to a canvas whose rows are linear in Web Mercator Y,
 * so MapLibre can place it as an image source without distortion. Clipped to land.
 */
export function renderField(g: Grid, land: Land, layer: FieldLayer, lead: number, opts: FieldOpts = {},
  // ~500 px across whatever the grid step: 8 per cell at 0.5 deg, 4 at 0.25 deg
  scale = Math.max(2, Math.round(480 / g.nlon))): HTMLCanvasElement {
  const { edge = 'feather', outside } = opts
  // Keyed on the data array itself: a placeholder field kept while the next one loads is never
  // painted with another layer's colour scale.
  const data = fieldData(g, layer, lead)
  let byOpts = imgCache.get(data)
  if (!byOpts) imgCache.set(data, (byOpts = new Map()))
  const key = `${layer}:${edge}:${outside ? outside.k : 1}:${scale}`
  const hit = byOpts.get(key)
  if (hit) return hit
  const fade = edge === 'feather' ? emphasis(g, outside?.mask ?? null, outside?.k ?? 1) : null
  const b = gridBounds(g)
  const W = g.nlon * scale
  const y0 = mercY(b.north), y1 = mercY(b.south)
  const H = Math.round(W * ((y0 - y1) / ((b.east - b.west) * Math.PI / 180)))
  const c = document.createElement('canvas')
  c.width = W; c.height = H
  const ctx = c.getContext('2d')!
  const img = ctx.createImageData(W, H)
  const isRegime = layer === 'regime'
  const arr = isRegime ? null : (data as (number | null)[])
  const reg = data as number[]
  const stops = stopsFor(layer)
  const lowCut = isRain(layer) ? 4 : layer === 'wind850' ? 1 : 0.06
  for (let py = 0; py < H; py++) {
    const lat = invMercY(y0 - ((py + 0.5) / H) * (y0 - y1))
    const fi = (lat - g.lat0) / g.step
    for (let px = 0; px < W; px++) {
      const lon = b.west + ((px + 0.5) / W) * (b.east - b.west)
      const fj = (lon - g.lon0) / g.step
      let rgb: number[] | null = null
      let lowV = 1
      if (isRegime) {
        const i = Math.round(fi), j = Math.round(fj)
        if (i >= 0 && j >= 0 && i < g.nlat && j < g.nlon) {
          const r = reg[i * g.nlon + j]
          if (r >= 0) rgb = regimeRGB(r)
          if (r === REGIME_ORDER.indexOf('normal')) lowV = 0.3 // "normal" is the background state: keep it faint
        }
      } else {
        const v = bilinear(arr!, g, fi, fj)
        if (v != null) { rgb = rampRGB(stops, v); lowV = v }
      }
      if (!rgb) continue
      // dry cells stay unpainted (the ground shows through); feathered at the grid edge, faint outside India
      const a = (fade ? fade(lon, lat) : 1) * (isRegime ? lowV : Math.min(1, lowV / lowCut))
      const o = (py * W + px) * 4
      img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = Math.round(255 * a)
    }
  }
  ctx.putImageData(img, 0, 0)
  if (layer !== 'wind850') {
    ctx.globalCompositeOperation = 'destination-in'
    ctx.fillStyle = '#000'
    traceLand(ctx, land, (lon, lat) => [((lon - b.west) / (b.east - b.west)) * W, ((y0 - mercY(lat)) / (y0 - y1)) * H])
    ctx.fill('evenodd')
    ctx.globalCompositeOperation = 'source-over'
  }
  byOpts.set(key, c)
  return c
}

/* ------------------------------------------------------------ wind particles */

export interface Particles {
  /** Advance and draw one frame. `dt` is elapsed time in 60 fps frames, so a slow frame rate never slows the flow. */
  step(ctx: CanvasRenderingContext2D, project: (lon: number, lat: number) => [number, number], w: number, h: number, dt?: number): void
  /** Respawn every particle. */
  reset(): void
  /** Forget last screen positions (keep the particles): call after the view moves or resizes. */
  clearTrails(): void
}

/**
 * Trail fade per 60 fps frame. The canvas stores 8-bit alpha, and a fade f leaves every pixel with
 * alpha a < 0.5 / (1 - f) stuck forever: 0.9 leaves up to 5/255 ink (a visible grey box over a
 * street map), 0.8 at most 2/255, which is invisible.
 */
const TRAIL_FADE = 0.8

/**
 * Particles advected by the run's 850 hPa wind (u east, v north, m/s). Motion here *is* the data.
 * Speed is constant on screen (`speed` px per 60 fps frame per m/s, like the 081 map) and time-based.
 * `fade` (read every frame) gives each particle's opacity 0..1, drawn in four alpha buckets.
 * Cost per frame: one projection and one line segment per particle.
 */
export function makeParticles(g: Grid, lead: () => number, count = 1400, opacity = 0.45, fade: () => ((lon: number, lat: number) => number) | null = () => null, speed = 0.3): Particles {
  const b = gridBounds(g)
  const midLon = (b.west + b.east) / 2, midLat = (b.south + b.north) / 2
  type P = { lon: number; lat: number; age: number; max: number }
  const spawn = (p?: P): P => {
    const q = p ?? ({} as P)
    q.lon = b.west + Math.random() * (b.east - b.west)
    q.lat = b.south + Math.random() * (b.north - b.south)
    q.age = 0
    q.max = 50 + Math.random() * 70
    return q
  }
  // staggered ages: particles expire and respawn continuously, never all at once
  let ps: P[] = Array.from({ length: count }, () => { const p = spawn(); p.age = Math.random() * p.max; return p })
  const sx = new Float32Array(count), sy = new Float32Array(count)
  const seen = new Uint8Array(count)
  const wind = (lat: number, lon: number): [number, number] => {
    const L = lead()
    const fi = (lat - g.lat0) / g.step, fj = (lon - g.lon0) / g.step
    if (!g.layers.u850 || !g.layers.v850) return [0, 0]
    return [bilinear(g.layers.u850[L], g, fi, fj) ?? 0, bilinear(g.layers.v850[L], g, fi, fj) ?? 0]
  }
  return {
    reset() { ps = ps.map((p) => spawn(p)); seen.fill(0) },
    clearTrails() { seen.fill(0) },
    step(ctx, project, w, h, dt = 1) {
      dt = Math.min(3, Math.max(0.25, dt))
      ctx.globalCompositeOperation = 'destination-in'
      ctx.fillStyle = `rgba(0,0,0,${TRAIL_FADE ** dt})`
      ctx.fillRect(0, 0, w, h)
      ctx.globalCompositeOperation = 'source-over'
      ctx.lineWidth = 1.1
      ctx.lineCap = 'round'
      // screen pixels per degree of longitude at the data centre: degrees per (m/s) this frame
      const [ax] = project(midLon, midLat), [bx] = project(midLon + 1, midLat)
      const k = (speed * dt) / (Math.abs(bx - ax) || 1)
      const f = fade()
      const buckets = [new Path2D(), new Path2D(), new Path2D(), new Path2D()]
      for (let i = 0; i < ps.length; i++) {
        const p = ps[i]
        const [u, v] = wind(p.lat, p.lon)
        p.lon += u * k
        p.lat += v * k * Math.cos((p.lat * Math.PI) / 180) // Mercator: equal speed on screen in y
        p.age += dt
        if (p.age > p.max || p.lon < b.west || p.lon > b.east || p.lat < b.south || p.lat > b.north || Math.hypot(u, v) < 0.4) {
          spawn(p)
          seen[i] = 0
          continue
        }
        const [x, y] = project(p.lon, p.lat)
        if (seen[i] && x > -20 && y > -20 && x < w + 20 && y < h + 20) {
          const a = f ? f(p.lon, p.lat) : 1
          if (a >= 0.08) {
            const path = buckets[Math.min(3, Math.floor(a * 4))]
            path.moveTo(sx[i], sy[i])
            path.lineTo(x, y)
          }
        }
        sx[i] = x; sy[i] = y; seen[i] = 1
      }
      buckets.forEach((path, n) => { ctx.strokeStyle = `rgba(14,26,31,${(opacity * (n + 1)) / 4})`; ctx.stroke(path) })
    },
  }
}

export const prefersReducedMotion = () => typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches

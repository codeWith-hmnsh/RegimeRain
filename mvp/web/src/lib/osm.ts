/**
 * OpenStreetMap raster tiles for the canvas maps. Tiles are requested live, only for the view on
 * screen at one zoom, and kept in memory for this page load. Nothing is prefetched or stored for
 * offline use (OSM tile usage policy: https://operations.osmfoundation.org/policies/tiles/).
 */

export const OSM_TILES = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
export const OSM_COPYRIGHT = 'https://www.openstreetmap.org/copyright'
/** Desaturated so the forecast colours lead. */
export const OSM_FILTER = 'saturate(0.45) contrast(0.92) brightness(1.03)'

type Tile = { img: HTMLImageElement; ready: HTMLCanvasElement | null; failed: boolean }
const tiles = new Map<string, Tile>()
const listeners = new Set<() => void>()
let loaded = 0, failed = 0

/** 'failed' only when every tile tried so far has failed (offline, blocked): callers fall back to plain ground. */
export const osmStatus = (): 'pending' | 'ok' | 'failed' => (loaded ? 'ok' : failed ? 'failed' : 'pending')

export function onTile(f: () => void) {
  listeners.add(f)
  return () => { listeners.delete(f) }
}

function tile(z: number, x: number, y: number): Tile {
  const key = `${z}/${x}/${y}`
  let t = tiles.get(key)
  if (t) return t
  const img = new Image()
  img.crossOrigin = 'anonymous'
  const entry: Tile = { img, ready: null, failed: false }
  img.onload = () => {
    // filter once per tile rather than on every redraw
    const c = document.createElement('canvas')
    c.width = img.naturalWidth; c.height = img.naturalHeight
    const ctx = c.getContext('2d')!
    ctx.filter = OSM_FILTER
    ctx.drawImage(img, 0, 0)
    entry.ready = c
    loaded++
    listeners.forEach((f) => f())
  }
  img.onerror = () => { entry.failed = true; failed++; listeners.forEach((f) => f()) }
  img.src = OSM_TILES.replace('{z}', String(z)).replace('{x}', String(x)).replace('{y}', String(y))
  tiles.set(key, entry)
  return entry
}

/**
 * Draws the tiles covering a w×h view in which screen = (ox + λ·k, oy − y·k), with λ the longitude
 * in radians and y the Web Mercator ordinate. Zoom is chosen so a tile pixel is about one CSS pixel,
 * which keeps place names readable. Returns how many tiles were drawn.
 */
export function drawTiles(ctx: CanvasRenderingContext2D, view: { k: number; ox: number; oy: number; w: number; h: number }) {
  const { k, ox, oy, w, h } = view
  const z = Math.max(1, Math.min(18, Math.round(Math.log2((2 * Math.PI * k) / 256))))
  const n = 2 ** z, size = (2 * Math.PI * k) / n
  const col = (x: number) => Math.floor(((x - ox) / k + Math.PI) / (2 * Math.PI) * n)
  const row = (y: number) => Math.floor(((1 - (oy - y) / k / Math.PI) / 2) * n)
  const xA = col(0), xB = col(w)
  const yA = Math.max(0, row(0)), yB = Math.min(n - 1, row(h))
  let drawn = 0
  for (let ty = yA; ty <= yB; ty++)
    for (let tx = xA; tx <= xB; tx++) {
      const t = tile(z, ((tx % n) + n) % n, ty)
      if (!t.ready) continue
      const x = ox + ((tx / n) * 2 * Math.PI - Math.PI) * k
      const y = oy - Math.PI * (1 - (2 * ty) / n) * k
      ctx.drawImage(t.ready, x, y, size + 0.5, size + 0.5)
      drawn++
    }
  return drawn
}

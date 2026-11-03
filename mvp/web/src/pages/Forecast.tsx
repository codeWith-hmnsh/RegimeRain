import * as maplibregl from 'maplibre-gl'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Drop } from '../components/Chrome'
import { GEO, PROB, RAIN, REGIME, REGIME_ORDER, WIND, type Stop } from '../lib/color'
import { cellAt, feather, isRain, makeParticles, prefersReducedMotion } from '../lib/field'
import { fmtDay, fmtLat, fmtLon, fmtMm, fmtPct } from '../lib/format'
import { useRun } from '../lib/run'
import type { FieldLayer, Place, Regime, Run, Threshold } from '../lib/types'
import { OSM_COPYRIGHT } from '../lib/osm'
import { setBasemap, setField, setTrack, useForecastMaps } from '../map/useForecastMaps'

type Layer = 'rain' | 'regime' | 'heavy' | 'very_heavy' | 'wind'
type Variant = 'raw' | 'corrected' | 'compare'

const LAYERS: { id: Layer; label: string; short: string; icon: React.ReactNode }[] = [
  { id: 'rain', label: 'Rainfall', short: 'Rain', icon: <path d="M12 3c-3.3 5-6 8-6 11a6 6 0 0 0 12 0c0-3-2.7-6-6-11z" /> },
  { id: 'regime', label: 'Regime', short: 'Regime', icon: <><rect x="4" y="4" width="7" height="7" rx="1" /><rect x="13" y="4" width="7" height="7" rx="1" /><rect x="4" y="13" width="7" height="7" rx="1" /><rect x="13" y="13" width="7" height="7" rx="1" /></> },
  { id: 'heavy', label: 'Heavy rain chance', short: 'Heavy', icon: <><path d="M12 3c-3.3 5-6 8-6 11a6 6 0 0 0 12 0c0-3-2.7-6-6-11z" /><path d="M12 10v4M12 17v.5" /></> },
  { id: 'very_heavy', label: 'Very heavy chance', short: 'V. heavy', icon: <><path d="M8 5c-2 3-3.5 5-3.5 7a3.5 3.5 0 0 0 7 0C11.5 10 10 8 8 5z" /><path d="M16 9c-2 3-3.5 5-3.5 7a3.5 3.5 0 0 0 7 0c0-2-1.5-4-3.5-7z" /></> },
  { id: 'wind', label: 'Wind 850 hPa', short: 'Wind', icon: <path d="M3 8h11a3 3 0 1 0-3-3M3 12h15a3 3 0 1 1-3 3M3 16h8" /> },
]
const MAJOR = new Set(['Mumbai', 'Delhi', 'Kolkata', 'Chennai', 'Bengaluru', 'Hyderabad', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Bhopal', 'Nagpur', 'Patna', 'Bhubaneswar', 'Guwahati', 'Kochi', 'Srinagar', 'Raipur'])

/** A layer shows only if the run has its data (manifest.layers): not-built layers are hidden, never faked. */
const available = (layers: string[], l: Layer) =>
  l === 'rain' || l === 'regime' ? true : l === 'heavy' ? layers.includes('p_heavy') : l === 'very_heavy' ? layers.includes('p_very_heavy') : layers.includes('wind850')

const fieldOf = (l: Layer, v: Variant, side: 'A' | 'B'): FieldLayer =>
  l === 'rain' ? (v === 'raw' || (v === 'compare' && side === 'A') ? 'raw' : 'corrected')
    : l === 'regime' ? 'regime' : l === 'heavy' ? 'p_heavy' : l === 'very_heavy' ? 'p_very_heavy' : 'wind850'

function valueText(run: Run, f: FieldLayer, lead: number, lat: number, lon: number, short = false): string | null {
  const c = cellAt(run.grid, lat, lon)
  if (!c) return null
  if (f === 'regime') {
    const r = run.grid.layers.regime[lead][c.idx]
    return r < 0 ? null : REGIME[REGIME_ORDER[r]].label
  }
  const v = run.grid.layers[f]?.[lead][c.idx]
  if (v == null) return null
  if (f === 'wind850') return short ? `${Math.round(v)}` : `${v.toFixed(1)} m/s`
  if (isRain(f)) return short ? `${Math.round(v)}` : `${v.toFixed(1)} mm/day`
  return fmtPct(v)
}

export default function Forecast() {
  const run = useRun()
  const { grid, manifest } = run
  const reduce = useReducedMotion()
  const [layer, setLayer] = useState<Layer>('rain')
  const [variant, setVariant] = useState<Variant>('corrected')
  const [lead, setLead] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [particlesOn, setParticlesOn] = useState(!prefersReducedMotion())
  const [placesOn, setPlacesOn] = useState(true)
  const [trackOn, setTrackOn] = useState(true)
  const [basemap, setBasemapOn] = useState(true)
  const [sel, setSel] = useState<{ lat: number; lon: number; place?: Place } | null>(null)
  const [hover, setHover] = useState<{ x: number; y: number; text: string; side: 'A' | 'B' } | null>(null)
  const [swipe, setSwipe] = useState(0.5)

  const stage = useRef<HTMLDivElement>(null)
  const aEl = useRef<HTMLDivElement>(null)
  const bEl = useRef<HTMLDivElement>(null)
  const pCanvas = useRef<HTMLCanvasElement>(null)
  const { maps, ready, osm } = useForecastMaps(run, aEl, bEl, 'corrected', 0, true)
  /** Street-map tiles actually showing: switched on and not failed. Otherwise the plain ground comes back. */
  const tiles = basemap && osm !== 'failed'
  const tilesRef = useRef(tiles)
  tilesRef.current = tiles
  const compare = layer === 'rain' && variant === 'compare'
  const leadRef = useRef(lead)
  leadRef.current = lead

  /* field + track on every change */
  useEffect(() => {
    if (!ready) return
    const { A, B } = maps.current
    setField(A!, run, fieldOf(layer, variant, 'A'), lead, tiles)
    setField(B!, run, fieldOf(layer, variant, 'B'), lead, tiles)
    setTrack(A!, lead, trackOn)
    setTrack(B!, lead, trackOn)
  }, [ready, layer, variant, lead, trackOn, run, maps, tiles])

  useEffect(() => {
    if (!ready) return
    setBasemap(maps.current.A!, basemap, tiles)
    setBasemap(maps.current.B!, basemap, tiles)
  }, [ready, basemap, tiles, maps])

  useEffect(() => {
    if (!ready || !compare) return
    maps.current.B!.resize()
    maps.current.B!.jumpTo({ center: maps.current.A!.getCenter(), zoom: maps.current.A!.getZoom() })
  }, [ready, compare, maps])

  /* timeline play */
  useEffect(() => {
    if (!playing) return
    const t = setInterval(() => setLead((l) => (l + 1) % grid.leads.length), 1700)
    return () => clearInterval(t)
  }, [playing, grid.leads.length])

  /* hover + click on both maps */
  useEffect(() => {
    if (!ready) return
    const offs: (() => void)[] = []
    ;(['A', 'B'] as const).forEach((side) => {
      const m = maps.current[side]!
      const move = (e: maplibregl.MapMouseEvent) => {
        const f = fieldOf(layer, variant, side)
        const t = valueText(run, f, leadRef.current, e.lngLat.lat, e.lngLat.lng)
        const r = stage.current!.getBoundingClientRect()
        const mr = m.getContainer().getBoundingClientRect()
        setHover(t ? { x: e.point.x + mr.left - r.left, y: e.point.y + mr.top - r.top, text: t, side } : null)
      }
      const leave = () => setHover(null)
      const click = (e: maplibregl.MapMouseEvent) => {
        const c = cellAt(grid, e.lngLat.lat, e.lngLat.lng)
        if (!c || grid.layers.raw[0][c.idx] == null) return setSel(null)
        const place = run.places.find((p) => Math.abs(p.cell.lat - c.lat) < 1e-6 && Math.abs(p.cell.lon - c.lon) < 1e-6)
        setSel({ lat: c.lat, lon: c.lon, place })
      }
      m.on('mousemove', move); m.on('mouseout', leave); m.on('click', click)
      offs.push(() => { m.off('mousemove', move); m.off('mouseout', leave); m.off('click', click) })
    })
    return () => offs.forEach((f) => f())
  }, [ready, layer, variant, run, grid, maps])

  /* particles over both maps */
  useEffect(() => {
    if (!ready || !particlesOn) return
    const cv = pCanvas.current!, ctx = cv.getContext('2d')!
    const A = maps.current.A!
    // over the street map, particles fade out toward the data edge like the field does
    const fade = (lon: number, lat: number) => feather(grid, lon, lat)
    const parts = makeParticles(grid, () => leadRef.current, 2600, layer === 'wind' ? 0.7 : 0.5, () => (tilesRef.current ? fade : null))
    let raf = 0, moving = false, dpr = 1, W = 0, H = 0
    const size = () => {
      const r = stage.current!.getBoundingClientRect()
      dpr = Math.min(2, devicePixelRatio || 1)
      W = r.width; H = r.height
      cv.width = W * dpr; cv.height = H * dpr
      parts.clearTrails()
    }
    const onMove = () => { moving = true }
    // trails hold screen positions, which a pan or zoom invalidates
    const onEnd = () => { moving = false; parts.clearTrails() }
    A.on('move', onMove); A.on('moveend', onEnd)
    const ro = new ResizeObserver(size)
    ro.observe(stage.current!)
    size()
    const proj = (lon: number, lat: number): [number, number] => { const p = A.project([lon, lat]); return [p.x, p.y] }
    let last = -1
    const loop = (t: number) => {
      const dt = last < 0 ? 1 : (t - last) / (1000 / 60)
      last = t
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      if (moving) ctx.clearRect(0, 0, W, H)
      else parts.step(ctx, proj, W, H, dt)
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => { cancelAnimationFrame(raf); ro.disconnect(); A.off('move', onMove); A.off('moveend', onEnd); ctx.clearRect(0, 0, cv.width, cv.height) }
  }, [ready, particlesOn, grid, layer, maps])

  /* place labels with values */
  const markers = useRef<{ side: 'A' | 'B'; place: Place; el: HTMLDivElement; mk: maplibregl.Marker }[]>([])
  useEffect(() => {
    if (!ready) return
    const list: typeof markers.current = []
    ;(['A', 'B'] as const).forEach((side) => {
      for (const p of run.places) {
        const el = document.createElement('div')
        el.className = 'place'
        el.addEventListener('click', (e) => { e.stopPropagation(); setSel({ lat: p.cell.lat, lon: p.cell.lon, place: p }) })
        const mk = new maplibregl.Marker({ element: el, anchor: 'left', offset: [-3, 0] }).setLngLat([p.lon, p.lat]).addTo(maps.current[side]!)
        list.push({ side, place: p, el, mk })
      }
    })
    markers.current = list
    // greedy label placement: major cities first, hide anything that would overlap a label already placed
    const order = [...run.places].sort((a, b) => Number(MAJOR.has(b.name)) - Number(MAJOR.has(a.name)))
    const placeLabels = () => {
      const A = maps.current.A!
      const taken: { x0: number; y0: number; x1: number; y1: number }[] = []
      for (const p of order) {
        const pt = A.project([p.lon, p.lat])
        const w = 14 + p.name.length * 6.8, h = 18
        const box = { x0: pt.x - 4, y0: pt.y - h / 2, x1: pt.x + w, y1: pt.y + h / 2 }
        const hit = taken.some((t) => box.x0 < t.x1 && box.x1 > t.x0 && box.y0 < t.y1 && box.y1 > t.y0)
        if (!hit) taken.push(box)
        for (const m of list) if (m.place === p) m.el.style.display = hit ? 'none' : ''
      }
    }
    maps.current.A!.on('moveend', placeLabels)
    placeLabels()
    return () => { list.forEach((m) => m.mk.remove()); maps.current.A?.off('moveend', placeLabels) }
  }, [ready, run, maps])

  useEffect(() => {
    for (const m of markers.current) {
      const t = valueText(run, fieldOf(layer, variant, m.side), lead, m.place.cell.lat, m.place.cell.lon, true)
      m.el.innerHTML = `<span>${m.place.name}</span>${t ? `<b>${t}</b>` : ''}`
      m.el.setAttribute('title', `${m.place.name}: ${valueText(run, fieldOf(layer, variant, m.side), lead, m.place.cell.lat, m.place.cell.lon) ?? 'no value'}`)
      m.el.style.visibility = placesOn ? 'visible' : 'hidden'
    }
  }, [layer, variant, lead, placesOn, run, ready])

  /* keyboard: arrows move lead, space plays */
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input, [role=slider]')) return
      if (e.key === 'ArrowRight' && e.shiftKey) { setLead((l) => Math.min(grid.leads.length - 1, l + 1)); e.preventDefault() }
      if (e.key === 'ArrowLeft' && e.shiftKey) { setLead((l) => Math.max(0, l - 1)); e.preventDefault() }
      if (e.key === 'Escape') setSel(null)
    }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [grid.leads.length])

  const flyTo = useCallback((p: Place) => {
    maps.current.A?.flyTo({ center: [p.lon, p.lat], zoom: Math.max(5.4, maps.current.A.getZoom()), duration: reduce ? 0 : 1100 })
    setSel({ lat: p.cell.lat, lon: p.cell.lon, place: p })
  }, [maps, reduce])

  const swipeX = `${swipe * 100}%`

  return (
    <main className="relative h-[calc(100svh_-_var(--banner,0px))] overflow-hidden bg-sea" data-testid="forecast" style={{ ['--banner' as string]: manifest.synthetic ? '30px' : '0px' }}>
      <div ref={stage} className="absolute inset-0">
        {/* MapLibre forces position: relative on its container, so each map sits inside its own absolute box */}
        <div className="absolute inset-0"><div ref={aEl} className="h-full w-full" data-testid="map" /></div>
        <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${swipeX})`, visibility: compare ? 'visible' : 'hidden' }}><div ref={bEl} className="h-full w-full" /></div>
        <canvas ref={pCanvas} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true" style={{ display: particlesOn ? '' : 'none' }} />
        {compare && <Swipe value={swipe} onChange={setSwipe} />}
        {compare && (
          <>
            <span className="tape tape-ink pointer-events-none absolute top-[74px] -translate-x-[calc(100%+14px)] font-mono text-[11px] uppercase tracking-[0.08em]" style={{ left: swipeX }}>Raw forecast</span>
            <span className="tape tape-ink pointer-events-none absolute top-[74px] translate-x-[14px] font-mono text-[11px] uppercase tracking-[0.08em]" style={{ left: swipeX }}>Corrected</span>
          </>
        )}
        {hover && (
          <div className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-[calc(100%+10px)] whitespace-nowrap rounded-[3px] bg-ink px-2 py-1 font-mono text-[12px] text-paper shadow-md" style={{ left: hover.x, top: hover.y }} data-testid="hover-value">
            {hover.text}{layer === 'rain' && <span className="ml-1.5 text-[#9fb2b7]">{fieldOf(layer, variant, hover.side)}</span>}
          </div>
        )}
      </div>

      {/* top bar */}
      <div className="pointer-events-none absolute inset-x-0 top-0 z-30 flex flex-wrap items-start gap-2 p-3">
        <Link to="/" className="pointer-events-auto flex h-10 items-center gap-2 rounded-[4px] bg-paper px-3 text-ink no-underline shadow-[0_1px_4px_rgba(14,26,31,0.18)]">
          <Drop /><span className="font-display text-[17px] font-extrabold tracking-[-0.03em]">RegimeRain</span>
        </Link>
        <PlaceSearch places={run.places} onPick={flyTo} />
        {layer === 'rain' && (
          <div className="pointer-events-auto flex h-10 overflow-hidden rounded-[4px] bg-paper p-1 shadow-[0_1px_4px_rgba(14,26,31,0.18)]" role="radiogroup" aria-label="Forecast variant" data-testid="variant">
            {(['raw', 'corrected', 'compare'] as Variant[]).map((v) => (
              <button key={v} role="radio" aria-checked={variant === v} onClick={() => setVariant(v)}
                className={`rounded-[3px] px-3 text-[13.5px] font-medium capitalize transition-colors ${variant === v ? 'bg-ink text-paper' : 'text-ink hover:bg-[#e3e8e6]'}`}>
                {v}
              </button>
            ))}
          </div>
        )}
        <nav className="pointer-events-auto ml-auto hidden h-10 items-center gap-1 rounded-[4px] bg-paper px-1.5 shadow-[0_1px_4px_rgba(14,26,31,0.18)] md:flex" aria-label="Pages">
          {[['/scorecard', 'Scorecard'], ['/method', 'Method'], ['/bulletin', 'Bulletin']].map(([to, l]) => (
            <Link key={to} to={to} className="rounded-[3px] px-2.5 py-1 text-[13.5px] font-medium text-ink no-underline hover:bg-[#e3e8e6]">{l}</Link>
          ))}
        </nav>
      </div>

      {/* layer rail */}
      <aside className="absolute bottom-[132px] left-3 z-30 flex flex-col gap-1 rounded-[6px] bg-paper p-1.5 shadow-[0_1px_6px_rgba(14,26,31,0.2)] md:bottom-auto md:top-[72px]" aria-label="Layers" data-testid="layer-rail">
        <div role="radiogroup" aria-label="Map layer" className="flex flex-col gap-1">
          {LAYERS.filter((l) => available(manifest.layers, l.id)).map((l) => (
            <button key={l.id} role="radio" aria-checked={layer === l.id} onClick={() => setLayer(l.id)} title={l.label}
              className={`group flex w-[58px] flex-col items-center gap-0.5 rounded-[4px] px-1 py-1.5 text-[10.5px] font-medium leading-tight transition-colors md:w-auto md:flex-row md:gap-2.5 md:px-2.5 md:py-2 md:text-[13px] ${layer === l.id ? 'bg-ink text-paper' : 'text-ink hover:bg-[#e3e8e6]'}`}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{l.icon}</svg>
              <span className="md:hidden">{l.short}</span>
              <span className="hidden md:inline">{l.label}</span>
            </button>
          ))}
        </div>
        <div className="mt-1 hidden border-t border-rule pt-1.5 md:block">
          <Toggle on={particlesOn} set={setParticlesOn} label="Wind particles" />
          <Toggle on={placesOn} set={setPlacesOn} label="Place values" />
          <Toggle on={trackOn} set={setTrackOn} label="Depression track" />
          <Toggle on={basemap} set={setBasemapOn} label="Street map" />
        </div>
      </aside>

      {/* legend */}
      <div className="absolute bottom-[92px] right-3 z-30 w-[min(360px,calc(100%-96px))] rounded-[6px] bg-paper px-3 py-2.5 shadow-[0_1px_6px_rgba(14,26,31,0.2)]" data-testid="legend">
        <LegendBar layer={layer} variant={variant} th={manifest.thresholds_mm} />
      </div>

      {/* attribution: always visible, in the strip under the timeline */}
      <div className="absolute bottom-0 right-3 z-40 rounded-t-[3px] bg-paper/90 px-1.5 text-[10.5px] leading-4 text-ink-2" data-testid="osm-attribution">
        {tiles && <>© <a href={OSM_COPYRIGHT} target="_blank" rel="noreferrer" className="text-ink-2 underline hover:text-ink">OpenStreetMap</a> contributors · </>}
        Coastline: Natural Earth
      </div>

      {/* timeline */}
      <Timeline run={run} lead={lead} setLead={(l) => { setLead(l); setPlaying(false) }} playing={playing} setPlaying={setPlaying} />

      {/* point panel */}
      <AnimatePresence>
        {sel && <PointPanel key={`${sel.lat},${sel.lon}`} run={run} sel={sel} lead={lead} setLead={setLead} onClose={() => setSel(null)} />}
      </AnimatePresence>
    </main>
  )
}

/* ------------------------------------------------------------------ pieces */

function Toggle({ on, set, label }: { on: boolean; set: (v: boolean) => void; label: string }) {
  return (
    <button role="switch" aria-checked={on} onClick={() => set(!on)} className="flex w-full items-center gap-2.5 rounded-[4px] px-2.5 py-1.5 text-[12.5px] text-ink hover:bg-[#e3e8e6]">
      <span className={`relative h-[14px] w-[24px] rounded-full transition-colors ${on ? 'bg-accent' : 'bg-[#b9c4c1]'}`}>
        <span className={`absolute top-[2px] h-[10px] w-[10px] rounded-full bg-white transition-all ${on ? 'left-[12px]' : 'left-[2px]'}`} />
      </span>
      {label}
    </button>
  )
}

function Swipe({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const ref = useRef<HTMLDivElement>(null)
  const drag = useRef(false)
  const setFrom = (clientX: number) => {
    const r = ref.current!.parentElement!.getBoundingClientRect()
    onChange(Math.min(0.97, Math.max(0.03, (clientX - r.left) / r.width)))
  }
  return (
    <div ref={ref} role="slider" tabIndex={0} aria-label="Compare raw and corrected" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value * 100)} data-testid="swipe"
      className="absolute inset-y-0 z-20 w-8 -translate-x-1/2 cursor-ew-resize touch-none"
      style={{ left: `${value * 100}%` }}
      onPointerDown={(e) => { drag.current = true; e.currentTarget.setPointerCapture(e.pointerId) }}
      onPointerMove={(e) => drag.current && setFrom(e.clientX)}
      onPointerUp={() => { drag.current = false }}
      onKeyDown={(e) => {
        if (e.key === 'ArrowLeft') { onChange(Math.max(0.03, value - 0.03)); e.preventDefault() }
        if (e.key === 'ArrowRight') { onChange(Math.min(0.97, value + 0.03)); e.preventDefault() }
      }}>
      <span className="absolute inset-y-0 left-1/2 w-[2px] -translate-x-1/2 bg-ink" />
      <span className="absolute left-1/2 top-1/2 grid h-9 w-9 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-ink text-paper shadow-lg">
        <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><path d="M6 4 2 8l4 4M10 4l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.8" /></svg>
      </span>
    </div>
  )
}

function LegendBar({ layer, variant, th }: { layer: Layer; variant: Variant; th: Record<Threshold, number> }) {
  if (layer === 'regime')
    return (
      <div>
        <p className="mb-1.5 font-mono text-[10.5px] uppercase tracking-[0.1em] text-ink-2">Most probable regime</p>
        <ul className="grid grid-cols-2 gap-x-3 gap-y-1 text-[12px] sm:grid-cols-4">
          {REGIME_ORDER.map((r) => <li key={r} className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: REGIME[r].color }} />{REGIME[r].label}</li>)}
        </ul>
      </div>
    )
  const [stops, unit, ticks, title]: [Stop[], string, number[], string] =
    layer === 'rain' ? [RAIN, 'mm/day', [1, 15, 35, th.heavy, th.very_heavy, 200], variant === 'raw' ? 'Raw forecast rain' : variant === 'compare' ? 'Rain · raw | corrected' : 'Corrected rain']
      : layer === 'wind' ? [WIND, 'm/s', [0, 4, 8, 12, 16, 22], 'Wind speed at 850 hPa']
        : [PROB, '', [0, 0.3, 0.5, 0.7, 1], `Chance of ≥ ${layer === 'heavy' ? th.heavy : th.very_heavy} mm/day`]
  const max = stops[stops.length - 1][0]
  const grad = stops.map(([v, c]) => `${c} ${(v / max) * 100}%`).join(',')
  const fmt = (v: number) => (layer === 'heavy' || layer === 'very_heavy' ? `${Math.round(v * 100)}%` : `${v}`)
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <p className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-ink-2">{title}</p>
        <p className="font-mono text-[11px] text-ink-2">{unit}</p>
      </div>
      <div className="relative h-3 rounded-[2px] border border-[#c9d2cf]" style={{ background: `linear-gradient(90deg, ${grad})` }}>
        {layer === 'rain' && [th.heavy, th.very_heavy].map((t) => <span key={t} className="absolute -top-1 h-5 w-[2px] bg-ink" style={{ left: `${(t / max) * 100}%` }} />)}
      </div>
      <div className="relative mt-1 h-4 font-mono text-[10.5px] text-ink-2">
        {ticks.map((t) => <span key={t} className="absolute -translate-x-1/2" style={{ left: `${(t / max) * 100}%` }}>{fmt(t)}</span>)}
      </div>
      {layer === 'rain' && <p className="mt-0.5 text-[11px] text-ink-2">Marks: IMD heavy ({th.heavy}) and very heavy ({th.very_heavy})</p>}
    </div>
  )
}

function Timeline({ run, lead, setLead, playing, setPlaying }: { run: Run; lead: number; setLead: (l: number) => void; playing: boolean; setPlaying: (p: boolean) => void }) {
  const { grid, manifest } = run
  return (
    <div className="absolute inset-x-3 bottom-4 z-30 flex h-[70px] items-stretch gap-2 rounded-[6px] bg-paper p-1.5 shadow-[0_1px_6px_rgba(14,26,31,0.2)]" data-testid="timeline">
      <button onClick={() => setPlaying(!playing)} aria-label={playing ? 'Pause' : 'Play lead days'} data-testid="play"
        className="grid w-11 shrink-0 place-items-center rounded-[4px] bg-ink text-paper hover:bg-accent sm:w-14">
        {playing
          ? <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><rect x="4" y="3" width="3.5" height="12" fill="currentColor" /><rect x="10.5" y="3" width="3.5" height="12" fill="currentColor" /></svg>
          : <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true"><path d="M5 3l10 6-10 6z" fill="currentColor" /></svg>}
      </button>
      <div className="hidden shrink-0 flex-col justify-center border-r border-rule pr-3 md:flex">
        <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-ink-2">Issued</span>
        <span className="num text-[13px]">{fmtDay(manifest.forecast_issue_date)}</span>
      </div>
      <div className="flex flex-1 gap-1" role="radiogroup" aria-label="Lead day">
        {grid.leads.map((l) => {
          const on = l.index === lead
          return (
            <button key={l.index} role="radio" aria-checked={on} onClick={() => setLead(l.index)}
              className={`relative flex min-w-0 flex-1 flex-col justify-center overflow-hidden whitespace-nowrap rounded-[4px] px-1.5 text-left sm:px-2 transition-colors ${on ? 'bg-ink text-paper' : 'text-ink hover:bg-[#e3e8e6]'}`}>
              <span className="hidden text-[13.5px] font-semibold sm:inline">{fmtDay(l.valid_date)}</span>
              <span className="text-[13px] font-semibold sm:hidden">{fmtDay(l.valid_date).split(' ').slice(0, 2).join(' ')}</span>
              <span className={`num text-[11px] ${on ? 'text-[#a9bcc1]' : 'text-ink-2'}`}>+{l.lead_hours} h</span>
              {on && playing && <motion.span className="absolute bottom-0 left-0 h-[3px] bg-accent" initial={{ width: 0 }} animate={{ width: '100%' }} transition={{ duration: 1.7, ease: 'linear' }} />}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function PlaceSearch({ places, onPick }: { places: Place[]; onPick: (p: Place) => void }) {
  const [q, setQ] = useState('')
  const [open, setOpen] = useState(false)
  const [idx, setIdx] = useState(0)
  const hits = useMemo(() => (q ? places.filter((p) => p.name.toLowerCase().includes(q.toLowerCase())).slice(0, 7) : []), [q, places])
  const pick = (p: Place) => { onPick(p); setQ(p.name); setOpen(false) }
  return (
    <div className="pointer-events-auto relative">
      <input value={q} onChange={(e) => { setQ(e.target.value); setOpen(true); setIdx(0) }} onFocus={() => setOpen(true)} onBlur={() => setTimeout(() => setOpen(false), 120)}
        onKeyDown={(e) => {
          if (e.key === 'ArrowDown') { setIdx((i) => Math.min(hits.length - 1, i + 1)); e.preventDefault() }
          if (e.key === 'ArrowUp') { setIdx((i) => Math.max(0, i - 1)); e.preventDefault() }
          if (e.key === 'Enter' && hits[idx]) pick(hits[idx])
          if (e.key === 'Escape') setOpen(false)
        }}
        role="combobox" aria-expanded={open && hits.length > 0} aria-controls="place-list" aria-autocomplete="list" aria-label="Find a place"
        placeholder="Find a place" data-testid="search"
        className="h-10 w-[180px] rounded-[4px] border-0 bg-paper px-3 text-[14px] shadow-[0_1px_4px_rgba(14,26,31,0.18)] outline-none placeholder:text-ink-3 focus-visible:outline-2 focus-visible:outline-accent sm:w-[220px]" />
      {open && hits.length > 0 && (
        <ul id="place-list" role="listbox" className="absolute left-0 top-11 w-full overflow-hidden rounded-[4px] bg-paper py-1 shadow-[0_4px_16px_rgba(14,26,31,0.2)]">
          {hits.map((p, i) => (
            <li key={p.name} role="option" aria-selected={i === idx} onMouseDown={() => pick(p)}
              className={`flex cursor-pointer justify-between px-3 py-1.5 text-[13.5px] ${i === idx ? 'bg-[#e3e8e6]' : ''}`}>
              <span>{p.name}</span><span className="num text-[11px] text-ink-2">{fmtLat(p.lat)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ point panel with correction trace */

function PointPanel({ run, sel, lead, setLead, onClose }: { run: Run; sel: { lat: number; lon: number; place?: Place }; lead: number; setLead: (l: number) => void; onClose: () => void }) {
  const { grid, manifest, curves } = run
  const reduce = useReducedMotion()
  const c = cellAt(grid, sel.lat, sel.lon)!
  /** Value at this cell, or null when the layer is not in this run. */
  const at = (k: 'raw' | 'corrected' | 'p_heavy' | 'p_very_heavy', l: number) => grid.layers[k]?.[l][c.idx] ?? null
  const regIdx = grid.layers.regime[lead][c.idx]
  const regime: Regime = REGIME_ORDER[regIdx]
  const nSyn = grid.synoptic.length
  const probs = grid.synoptic.map((r, k) => ({ r, p: grid.regime_probs_pct[lead][c.idx * nSyn + k] })).sort((a, b) => b.p - a.p)
  const curve = curves.find((q) => q.variant === 'B' && q.regime === regime && q.geo === 'all')
  const geo = grid.geo_classes[grid.geo[c.idx]]
  const hasP = manifest.layers.includes('p_heavy')
  const series = grid.leads.map((l) => ({ l, raw: at('raw', l.index) ?? 0, cor: at('corrected', l.index) ?? 0 }))
  const yMax = Math.max(140, ...series.flatMap((s) => [s.raw, s.cor])) * 1.08
  const th = manifest.thresholds_mm

  return (
    <motion.aside initial={reduce ? false : { x: 40, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={reduce ? undefined : { x: 40, opacity: 0 }} transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
      className="absolute inset-x-3 bottom-[88px] top-auto z-40 max-h-[62%] overflow-y-auto rounded-[6px] bg-paper shadow-[0_8px_30px_rgba(14,26,31,0.25)] md:inset-x-auto md:bottom-[92px] md:right-3 md:top-[64px] md:max-h-none md:w-[372px]"
      aria-label="Point forecast" data-testid="point-panel">
      <header className="sticky top-0 z-10 flex items-start justify-between gap-3 border-b border-rule bg-paper px-4 py-3">
        <div>
          <h2 className="font-display text-[24px] font-extrabold leading-tight tracking-[-0.03em]">{sel.place?.name ?? 'Grid cell'}</h2>
          <p className="num text-[11.5px] text-ink-2">cell {fmtLat(c.lat)} {fmtLon(c.lon)} · {grid.step}° · {fmtDay(grid.leads[lead].valid_date)}</p>
        </div>
        <button onClick={onClose} aria-label="Close point forecast" className="-mr-1 grid h-8 w-8 place-items-center rounded-[4px] text-[20px] leading-none hover:bg-[#e3e8e6]">×</button>
      </header>

      <div className="grid grid-cols-2 gap-px bg-rule">
        {[['Corrected', `${fmtMm(at('corrected', lead))}`, 'mm/day', true], ['Raw forecast', `${fmtMm(at('raw', lead))}`, 'mm/day', false],
          ...(hasP ? [['P(heavy)', fmtPct(at('p_heavy', lead)), `≥ ${th.heavy} mm`, false], ['P(very heavy)', fmtPct(at('p_very_heavy', lead)), `≥ ${th.very_heavy} mm`, false]] : [])].map(([k, v, u, big]) => (
          <div key={k as string} className="bg-paper px-4 py-3">
            <div className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-ink-2">{k}</div>
            <div className={`num leading-none tracking-[-0.03em] ${big ? 'mt-1.5 text-[36px] font-medium' : 'mt-2 text-[22px]'}`}>{v}</div>
            <div className="num mt-1 text-[11px] text-ink-2">{u}</div>
          </div>
        ))}
      </div>

      <section className="px-4 py-4">
        <h3 className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-ink-2">Next {grid.leads.length} days · raw vs corrected</h3>
        <svg viewBox="0 0 340 150" className="mt-2 w-full" role="img" aria-label="Raw and corrected rainfall for each lead day">
          {[th.heavy, th.very_heavy].map((t) => (
            <g key={t}>
              <line x1="0" x2="340" y1={128 - (t / yMax) * 118} y2={128 - (t / yMax) * 118} stroke="#0e1a1f" strokeDasharray="3 3" strokeOpacity="0.5" />
              <text x="338" y={124 - (t / yMax) * 118} textAnchor="end" fontSize="9.5" fill="#46565c" className="num">{t}</text>
            </g>
          ))}
          <line x1="0" x2="340" y1="128" y2="128" stroke="#d3dad7" />
          {series.map((s, i) => {
            const cx = 12 + i * 64
            const on = s.l.index === lead
            return (
              <g key={i} className="cursor-pointer" onClick={() => setLead(s.l.index)}>
                <rect x={cx - 6} y="0" width="60" height="148" fill={on ? '#e3e8e6' : 'transparent'} rx="3" />
                <rect x={cx} y={128 - (s.raw / yMax) * 118} width="20" height={(s.raw / yMax) * 118} fill="#8e999c" rx="2" />
                <rect x={cx + 22} y={128 - (s.cor / yMax) * 118} width="20" height={(s.cor / yMax) * 118} fill="#0b6e7f" rx="2" />
                <text x={cx + 21} y="143" textAnchor="middle" fontSize="10" fill="#46565c" className="num">+{s.l.lead_hours}h</text>
              </g>
            )
          })}
        </svg>
        <div className="mt-1 flex gap-4 text-[11.5px] text-ink-2">
          <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-[2px] bg-[#8e999c]" />Raw</span>
          <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-[2px] bg-accent" />Corrected</span>
          <span className="ml-auto">dashed: IMD thresholds</span>
        </div>
      </section>

      <section className="border-t border-rule px-4 py-4">
        <h3 className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-ink-2">Regime probabilities</h3>
        <ul className="mt-2 grid gap-1.5">
          {probs.map(({ r, p }) => (
            <li key={r} className="grid grid-cols-[88px_1fr_36px] items-center gap-2 text-[12.5px]">
              <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: REGIME[r].color }} />{REGIME[r].label}</span>
              <span className="h-2 overflow-hidden rounded-r-[4px] bg-[#e3e8e6]"><motion.span className="block h-full rounded-r-[4px]" style={{ background: REGIME[r].color }} initial={reduce ? false : { width: 0 }} animate={{ width: `${p}%` }} transition={{ duration: 0.7 }} /></span>
              <span className="num text-right">{p}%</span>
            </li>
          ))}
        </ul>
      </section>

      <section className="border-t border-rule px-4 py-4" data-testid="correction-trace">
        <h3 className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-ink-2">How this value was corrected</h3>
        <ol className="mt-3 grid gap-0">
          {[
            ['Raw forecast', `${fmtMm(at('raw', lead))} mm/day`, manifest.forecast_source],
            ['Regime predicted', `${REGIME[regime].label} · ${probs[0].p}%`, 'forecast-time inputs only'],
            ['Terrain class', geo ? GEO[geo].label : '—', 'static: from elevation and distance to the coast'],
            ['Curves blended', curve?.id ?? '—', curve ? `regime curves weighted by probability; ${REGIME[regime].label.toLowerCase()} curve fitted on ${curve.n_days.toLocaleString()} past days` : 'no curve file in this run'],
            ['Corrected', `${fmtMm(at('corrected', lead))} mm/day`, hasP ? `P(heavy) ${fmtPct(at('p_heavy', lead))}` : 'heavy-rain chance: not in this run'],
          ].map(([k, v, note], i, arr) => (
            <motion.li key={k} initial={reduce ? false : { opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 + i * 0.12 }}
              className="relative grid grid-cols-[18px_1fr] gap-3 pb-3">
              <span className="relative flex justify-center">
                <span className={`mt-1 h-2.5 w-2.5 rounded-full ${i === arr.length - 1 ? 'bg-accent' : 'border-2 border-ink bg-paper'}`} />
                {i < arr.length - 1 && <span className="absolute bottom-[-4px] top-4 w-px bg-ink-3" />}
              </span>
              <span>
                <span className="block text-[11.5px] text-ink-2">{k}</span>
                <span className="num block text-[14px]">{v}</span>
                <span className="block text-[11.5px] text-ink-3">{note}</span>
              </span>
            </motion.li>
          ))}
        </ol>
      </section>
    </motion.aside>
  )
}

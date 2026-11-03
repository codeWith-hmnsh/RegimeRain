import { motion, useInView, useReducedMotion } from 'motion/react'
import { useMemo, useRef, useState, type ReactNode } from 'react'
import { REGIME, REGIME_ORDER, THRESHOLDS } from '../lib/color'
import { METRIC_KEYS, METRIC_META, fmtMetric, verdict } from '../lib/format'
import { pooledEntry, score, seasonEntries } from '../lib/verify'
import type { MetricKey, QmCurve, Regime, Threshold, Verification, VerificationEntry } from '../lib/types'

const INK2 = '#46565c'
const RULE = '#d3dad7'
const RAW = '#8e999c'
const COR = '#0b6e7f'

/* ------------------------------------------------------------------ shared bits */

function useTip() {
  const [tip, setTip] = useState<{ x: number | string; y: number; body: ReactNode } | null>(null)
  const el = tip && (
    <div role="status" className="pointer-events-none absolute z-10 min-w-[140px] -translate-x-1/2 -translate-y-[calc(100%+12px)] rounded-[4px] border border-ink bg-surface px-2.5 py-2 font-mono text-[11.5px] leading-[1.5] shadow-[0_6px_18px_rgba(14,26,31,0.14)]"
      style={{ left: tip.x, top: tip.y }}>
      {tip.body}
    </div>
  )
  return { setTip, el }
}

export function Legend({ items }: { items: { color: string; label: string; dash?: boolean }[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-[12.5px] text-ink-2">
      {items.map((i) => (
        <li key={i.label} className="flex items-center gap-1.5">
          <svg width="18" height="8" aria-hidden="true"><line x1="1" y1="4" x2="17" y2="4" stroke={i.color} strokeWidth="2.5" strokeDasharray={i.dash ? '3 3' : undefined} strokeLinecap="round" /></svg>
          {i.label}
        </li>
      ))}
    </ul>
  )
}

const niceMax = (v: number) => {
  const p = Math.pow(10, Math.floor(Math.log10(v)))
  return Math.ceil(v / p) * p
}

/* ------------------------------------------------------------------ QM curves */

export function QmChart({ curves, show, height = 300, compact = false }: { curves: QmCurve[]; show: (Regime | 'global')[]; height?: number; compact?: boolean }) {
  const wrap = useRef<HTMLDivElement>(null)
  const inView = useInView(wrap, { once: true })
  const reduce = useReducedMotion()
  const { setTip, el } = useTip()
  const sel = curves.filter((c) => show.includes(c.regime))
  const cut = (c: QmCurve) => c.quantiles.map((q, i) => ({ q, f: c.forecast_mm[i], t: c.truth_mm[i] })).filter((p) => p.q <= 0.975)
  const max = niceMax(Math.max(...sel.flatMap((c) => cut(c).flatMap((p) => [p.f, p.t]))))
  const W = 480, H = height, m = { l: 44, r: 12, t: 10, b: 34 }
  const x = (v: number) => m.l + (v / max) * (W - m.l - m.r)
  const y = (v: number) => H - m.b - (v / max) * (H - m.t - m.b)
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((k) => k * max)
  const color = (r: Regime | 'global') => (r === 'global' ? '#0e1a1f' : REGIME[r].color)
  const label = (r: Regime | 'global') => (r === 'global' ? 'One global curve' : REGIME[r].label)

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const box = e.currentTarget.getBoundingClientRect()
    const fx = ((e.clientX - box.left) / box.width) * W
    const fv = ((fx - m.l) / (W - m.l - m.r)) * max
    if (fv < 0 || fv > max) return setTip(null)
    const rows = sel.map((c) => {
      const pts = cut(c)
      const p = pts.reduce((a, b) => (Math.abs(b.f - fv) < Math.abs(a.f - fv) ? b : a))
      return { c, p }
    })
    setTip({
      x: ((x(fv)) / W) * box.width, y: e.clientY - box.top,
      body: <>
        <div className="mb-1 text-ink-2">Forecast ≈ {fv.toFixed(0)} mm</div>
        {rows.map(({ c, p }) => <div key={c.id} className="flex justify-between gap-3"><span>{label(c.regime)}</span><span>→ {p.t.toFixed(1)} mm</span></div>)}
      </>,
    })
  }

  return (
    <div ref={wrap} className="relative">
      {!compact && <div className="mb-2"><Legend items={[...sel.map((c) => ({ color: color(c.regime), label: `${label(c.regime)} · ${c.n_days} days`, dash: c.regime === 'global' })), { color: RULE, label: 'No correction (1:1)', dash: true }]} /></div>}
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Quantile-mapping curves: forecast rainfall against corrected rainfall" onPointerMove={onMove} onPointerLeave={() => setTip(null)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} stroke={RULE} strokeWidth="1" />
            <text x={m.l - 6} y={y(t) + 4} textAnchor="end" fontSize="11" fill={INK2} className="num">{t.toFixed(0)}</text>
            <text x={x(t)} y={H - m.b + 16} textAnchor="middle" fontSize="11" fill={INK2} className="num">{t.toFixed(0)}</text>
          </g>
        ))}
        <text x={(W + m.l) / 2} y={H - 4} textAnchor="middle" fontSize="11" fill={INK2}>Raw forecast, mm/day</text>
        <text x={12} y={(H - m.b) / 2} textAnchor="middle" fontSize="11" fill={INK2} transform={`rotate(-90 12 ${(H - m.b) / 2})`}>Corrected, mm/day</text>
        <line x1={x(0)} y1={y(0)} x2={x(max)} y2={y(max)} stroke={RULE} strokeWidth="2" strokeDasharray="4 4" />
        {sel.map((c, k) => {
          const d = cut(c).map((p, i) => `${i ? 'L' : 'M'}${x(p.f).toFixed(1)},${y(p.t).toFixed(1)}`).join('')
          return (
            <motion.path key={c.id} d={d} fill="none" stroke={color(c.regime)} strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round"
              strokeDasharray={c.regime === 'global' ? '6 5' : undefined}
              initial={reduce ? false : { pathLength: 0 }} animate={inView ? { pathLength: 1 } : undefined}
              transition={{ duration: 1.3, delay: 0.15 * k, ease: [0.16, 1, 0.3, 1] }} />
          )
        })}
      </svg>
      {el}
    </div>
  )
}

/* ------------------------------------------------------------------ metric rows (raw -> corrected) */

export function MetricBars({ entry, keys = METRIC_KEYS, big = false }: { entry: VerificationEntry; keys?: MetricKey[]; big?: boolean }) {
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { once: true, margin: '-10% 0px' })
  const reduce = useReducedMotion()
  return (
    <div ref={ref} className="grid gap-4" data-testid="metric-bars">
      {keys.map((k) => {
        const a = score(entry.baseline, k), b = score(entry.corrected, k)
        if (a == null || b == null) return null
        const max = k === 'rmse' ? niceMax(Math.max(a, b)) : 1
        const v = verdict(k, a, b)
        return (
          <div key={k} className="grid grid-cols-[minmax(84px,136px)_1fr_auto] items-center gap-x-4 gap-y-1">
            <div>
              <div className={`font-display font-extrabold tracking-tight ${big ? 'text-[22px]' : 'text-[17px]'}`}>{METRIC_META[k].name}</div>
              <div className="text-[11px] text-ink-3">{METRIC_META[k].low ? 'lower is better' : 'higher is better'}</div>
            </div>
            <div className="grid gap-[3px]">
              {[['Raw', a, RAW], ['Corrected', b, COR]].map(([lab, val, col]) => (
                <div key={lab as string} className="grid grid-cols-[64px_1fr_48px] items-center gap-2 text-[11.5px]">
                  <span className="text-ink-2">{lab}</span>
                  <span className="h-[9px] overflow-hidden rounded-r-[4px] bg-[#e6ebe9]">
                    <motion.span className="block h-full rounded-r-[4px]" style={{ background: col as string }}
                      initial={reduce ? false : { width: lab === 'Raw' ? 0 : `${(a / max) * 100}%` }}
                      animate={inView ? { width: `${((val as number) / max) * 100}%` } : undefined}
                      transition={{ duration: 1.1, delay: lab === 'Raw' ? 0 : 0.6, ease: [0.16, 1, 0.3, 1] }} />
                  </span>
                  <span className="num text-right">{fmtMetric(k, val as number)}</span>
                </div>
              ))}
            </div>
            <VerdictTag v={v} />
          </div>
        )
      })}
    </div>
  )
}

export function VerdictTag({ v }: { v: 'better' | 'worse' | 'same' | 'na' }) {
  const m = { better: ['▲', 'Better', 'text-good'], worse: ['▼', 'Worse', 'text-worse'], same: ['■', 'No change', 'text-ink-2'], na: ['', 'Not defined', 'text-ink-3'] }[v]
  return <span className={`num whitespace-nowrap text-[12px] font-medium ${m[2]}`}><span aria-hidden="true">{m[0]} </span>{m[1]}</span>
}

/* ------------------------------------------------------------------ regime x metric delta matrix */

export function DeltaMatrix({ ver, th }: { ver: Verification; th: Threshold }) {
  const { setTip, el } = useTip()
  const rows = REGIME_ORDER.map((r) => pooledEntry(ver, th, r)).filter(Boolean) as VerificationEntry[]
  const colMax = useMemo(() => Object.fromEntries(METRIC_KEYS.map((k) => [k, Math.max(1e-6, ...rows.map((e) => Math.abs((score(e.corrected, k) ?? 0) - (score(e.baseline, k) ?? 0))))])), [rows])
  return (
    <div className="relative overflow-x-auto" data-testid="delta-matrix">
      <table className="w-full min-w-[640px] border-separate border-spacing-[2px] text-[12.5px]">
        <thead>
          <tr>
            <th className="px-2 py-1.5 text-left font-mono text-[10.5px] font-medium uppercase tracking-[0.08em] text-ink-2">Regime</th>
            {METRIC_KEYS.map((k) => <th key={k} className="px-2 py-1.5 text-right font-mono text-[10.5px] font-medium uppercase tracking-[0.08em] text-ink-2">{METRIC_META[k].name}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((e) => (
            <tr key={e.regime}>
              <th scope="row" className="whitespace-nowrap px-2 py-2 text-left font-medium">
                <span className="mr-2 inline-block h-2.5 w-2.5 rounded-[2px] align-[-1px]" style={{ background: REGIME[e.regime!].color }} />
                {REGIME[e.regime!].label}
              </th>
              {METRIC_KEYS.map((k) => {
                const a = score(e.baseline, k), b = score(e.corrected, k)
                if (a == null || b == null) return <td key={k} className="num rounded-[3px] bg-[#eef1ef] px-2 py-2 text-right text-ink-3">—</td>
                const d = b - a
                const v = verdict(k, a, b)
                const s = Math.min(1, Math.abs(d) / colMax[k])
                const bg = v === 'better' ? `rgba(11,110,127,${0.1 + 0.45 * s})` : v === 'worse' ? `rgba(192,38,61,${0.12 + 0.5 * s})` : '#eef1ef'
                return (
                  <td key={k} className="num cursor-default rounded-[3px] px-2 py-2 text-right"
                    style={{ background: bg }}
                    onPointerEnter={(ev) => {
                      const box = (ev.currentTarget.closest('.relative') as HTMLElement).getBoundingClientRect()
                      const r = ev.currentTarget.getBoundingClientRect()
                      setTip({ x: r.left - box.left + r.width / 2, y: r.top - box.top, body: <>
                        <div className="text-ink-2">{REGIME[e.regime!].label} · {METRIC_META[k].name}</div>
                        <div>Raw {fmtMetric(k, a)} → corrected {fmtMetric(k, b)}</div>
                        <VerdictTag v={v} />
                      </> })
                    }}
                    onPointerLeave={() => setTip(null)}>
                    <span aria-hidden="true" className={v === 'worse' ? 'text-worse' : v === 'better' ? 'text-good' : 'text-ink-3'}>{v === 'worse' ? '▼ ' : v === 'better' ? '▲ ' : ''}</span>
                    {d >= 0 ? '+' : '−'}{fmtMetric(k, Math.abs(d))}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {el}
    </div>
  )
}

/* ------------------------------------------------------------------ per-season dumbbells */

export function SeasonDumbbells({ ver, th, metric }: { ver: Verification; th: Threshold; metric: MetricKey }) {
  const { setTip, el } = useTip()
  const rows = seasonEntries(ver, th).filter((e) => score(e.baseline, metric) != null && score(e.corrected, metric) != null)
  const vals = rows.flatMap((e) => [score(e.baseline, metric)!, score(e.corrected, metric)!])
  const lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.15 || 0.05
  const W = 480, rowH = 30, m = { l: 52, r: 16, t: 8, b: 26 }, H = m.t + m.b + rows.length * rowH
  const x = (v: number) => m.l + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (W - m.l - m.r)
  return (
    <div className="relative">
      <div className="mb-2"><Legend items={[{ color: RAW, label: 'Raw forecast' }, { color: COR, label: 'Corrected' }]} /></div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${METRIC_META[metric].name} per held-out season, raw and corrected`}>
        {rows.map((e, i) => {
          const cy = m.t + i * rowH + rowH / 2
          const a = score(e.baseline, metric)!, b = score(e.corrected, metric)!
          const v = verdict(metric, a, b)
          return (
            <g key={e.fold} onPointerEnter={() => setTip({ x: (x(b) / W) * 100 + '%', y: cy, body: <><div className="text-ink-2">Season {e.fold} held out</div><div>Raw {fmtMetric(metric, a)} → {fmtMetric(metric, b)}</div><VerdictTag v={v} /></> })}
              onPointerLeave={() => setTip(null)}>
              <rect x={0} y={cy - rowH / 2} width={W} height={rowH} fill="transparent" />
              <text x={m.l - 10} y={cy + 4} textAnchor="end" fontSize="11.5" fill={INK2} className="num">{e.fold}</text>
              <line x1={m.l} x2={W - m.r} y1={cy} y2={cy} stroke="#edf1ef" />
              <line x1={x(a)} x2={x(b)} y1={cy} y2={cy} stroke={v === 'worse' ? '#c0263d' : '#9fb3b6'} strokeWidth="2" />
              <circle cx={x(a)} cy={cy} r="5" fill={RAW} stroke="#fff" strokeWidth="2" />
              <circle cx={x(b)} cy={cy} r="5" fill={COR} stroke="#fff" strokeWidth="2" />
            </g>
          )
        })}
        {[lo, (lo + hi) / 2, hi].map((t) => <text key={t} x={x(t)} y={H - 6} textAnchor="middle" fontSize="11" fill={INK2} className="num">{fmtMetric(metric, t)}</text>)}
      </svg>
      {el}
    </div>
  )
}

/* ------------------------------------------------------------------ reliability */

export function ReliabilityChart({ ver }: { ver: Verification }) {
  const { setTip, el } = useTip()
  const W = 420, H = 380, m = { l: 44, r: 14, t: 10, b: 38 }
  const x = (v: number) => m.l + v * (W - m.l - m.r)
  const y = (v: number) => H - m.b - v * (H - m.t - m.b)
  const series = ([['heavy', '#2a78d6', `Heavy ≥ ${THRESHOLDS.heavy} mm`], ['very_heavy', '#eb6834', `Very heavy ≥ ${THRESHOLDS.very_heavy} mm`]] as [Threshold, string, string][])
    .filter(([th]) => ver.reliability?.[th]?.length)
  return (
    <div className="relative">
      <div className="mb-2"><Legend items={[...series.map(([, c, l]) => ({ color: c, label: l })), { color: RULE, label: 'Perfectly calibrated', dash: true }]} /></div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-[520px]" role="img" aria-label="Reliability diagram: forecast probability against observed frequency">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} stroke={RULE} />
            <text x={m.l - 6} y={y(t) + 4} textAnchor="end" fontSize="11" fill={INK2} className="num">{t * 100}%</text>
            <text x={x(t)} y={H - m.b + 16} textAnchor="middle" fontSize="11" fill={INK2} className="num">{t * 100}%</text>
          </g>
        ))}
        <text x={(W + m.l) / 2} y={H - 4} textAnchor="middle" fontSize="11" fill={INK2}>Forecast probability</text>
        <text x={12} y={(H - m.b) / 2} textAnchor="middle" fontSize="11" fill={INK2} transform={`rotate(-90 12 ${(H - m.b) / 2})`}>Observed frequency</text>
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke={RULE} strokeWidth="2" strokeDasharray="4 4" />
        {series.map(([th, c]) => {
          const pts = ver.reliability![th]!
          return (
            <g key={th}>
              <polyline points={pts.map((p) => `${x(p.p_forecast)},${y(p.p_observed)}`).join(' ')} fill="none" stroke={c} strokeWidth="2" />
              {pts.map((p) => (
                <circle key={p.p_forecast} cx={x(p.p_forecast)} cy={y(p.p_observed)} r="5" fill={c} stroke="#fff" strokeWidth="2"
                  onPointerEnter={() => setTip({ x: (x(p.p_forecast) / W) * 100 + '%', y: y(p.p_observed), body: <><div className="text-ink-2">{th === 'heavy' ? 'Heavy' : 'Very heavy'}</div><div>Forecast {Math.round(p.p_forecast * 100)}% → observed {(p.p_observed * 100).toFixed(1)}%</div><div className="text-ink-2">{p.n.toLocaleString()} cases</div></> })}
                  onPointerLeave={() => setTip(null)} />
              ))}
            </g>
          )
        })}
      </svg>
      {el}
    </div>
  )
}

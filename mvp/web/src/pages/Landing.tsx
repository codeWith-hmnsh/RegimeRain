import { useInView } from 'motion/react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Footer, Nav } from '../components/Chrome'
import { FieldStage } from '../components/FieldStage'
import { Marquee } from '../components/Marquee'
import { NumberTicker } from '../components/NumberTicker'
import { FadeUp, RevealLines } from '../components/Reveal'
import { MetricBars, QmChart, VerdictTag } from '../components/charts'
import { GEO_ORDER, RAIN, REGIME, REGIME_ORDER } from '../lib/color'
import { FSS_HEADLINE, METRIC_KEYS, METRIC_META, fmtDay, fmtLat, fmtLon, fmtMetric, verdict } from '../lib/format'
import { pooledEntry, regimeEntries, score } from '../lib/verify'
import { useRun } from '../lib/run'
import type { FieldLayer, MetricKey, Regime, Threshold } from '../lib/types'

/** Glass panel for text over the live map: translucent surface, blurred and saturated backdrop. */
const glass = 'pointer-events-auto rounded-[4px] border border-ink/25 bg-paper/80 shadow-[0_2px_14px_rgba(14,26,31,0.1)] backdrop-blur-[3px] backdrop-saturate-150'

/** Real vocabulary for the marquee: metric names and thresholds come from the run. */
const vocab = (th: Record<Threshold, number>, step: number) => [...METRIC_KEYS.map((k) => METRIC_META[k].name),
  `heavy ≥ ${th.heavy} mm/day`, `very heavy ≥ ${th.very_heavy} mm/day`, `${step}° grid`, 'IMD gridded rainfall 0.25°', 'IFS HRES',
  'IBTrACS', 'MJO RMM', 'leave-one-monsoon-out', 'EPSG:4326', 'June–September']

export default function Landing() {
  const run = useRun()
  const { manifest: m, verification: ver, grid } = run
  const w = m.wettest
  const [wide, setWide] = useState(() => matchMedia('(min-width: 900px)').matches)
  useEffect(() => {
    const q = matchMedia('(min-width: 900px)')
    const f = () => setWide(q.matches)
    q.addEventListener('change', f)
    return () => q.removeEventListener('change', f)
  }, [])
  const mark = useMemo(() => ({ lat: w.lat, lon: w.lon }), [w.lat, w.lon])
  const heavy = pooledEntry(ver, 'heavy')

  return (
    <main data-testid="landing">
      {/* ---------------------------------------------------------------- hero: the forecast is the headline */}
      <section className="relative h-[calc(100svh_-_30px)] min-h-[640px] overflow-hidden">
        {/* full-bleed street map backdrop: no wheel or drag, so the page still scrolls */}
        <FieldStage run={run} layer="corrected" lead={w.lead} anchorX={wide ? 0.68 : 0.5} anchorY={wide ? 0.5 : 0.32} reveal particles mark={mark} basemap outsideIndia={0.2} className="absolute inset-0" />
        {/* a light wash on the text side, so the panel reads over any tile without hiding the map */}
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-paper/70 via-paper/20 via-35% to-transparent to-55% lg:bg-gradient-to-r lg:from-paper/60 lg:via-paper/15 lg:via-30% lg:to-transparent lg:to-45%" aria-hidden="true" />
        <Nav over />
        <div className="pointer-events-none absolute inset-0 flex items-end px-4 pb-6 pt-24 md:px-8 lg:items-center lg:pb-10">
          <div className={`${glass} w-full p-5 sm:max-w-[clamp(500px,42vw,660px)] sm:p-8`}>
            <p className="eyebrow mb-3 sm:mb-5">SIH 2026 · SIH26080 · MoES / NCMRWF</p>
            <h1 className="display text-[length:clamp(40px,4.4vw,76px)]">
              <RevealLines lines={['Classify first.', 'Correct second.']} delay={0.3} />
            </h1>
            <p className="mt-4 hidden max-w-[48ch] text-[16px] leading-[1.6] text-ink-2 sm:block">
              Rain forecasts go wrong in different ways on different days. RegimeRain names the kind of monsoon day first, from the forecast itself, then corrects the rain with a curve fitted on past days of that kind, and scores the result against what fell.
            </p>
            <div className="mt-5 flex flex-wrap gap-2 sm:mt-6 sm:gap-2.5">
              <Link to="/forecast" data-testid="hero-cta" className="group inline-flex items-center gap-2 rounded-[3px] bg-ink px-3.5 py-2.5 text-[14px] font-medium text-paper sm:px-4 sm:text-[15px] no-underline shadow-[0_2px_10px_rgba(14,26,31,0.25)] transition-colors hover:bg-accent">
                Open the forecast map <span aria-hidden="true" className="transition-transform duration-200 group-hover:translate-x-1">→</span>
              </Link>
              <Link to="/method" className="hidden items-center rounded-[3px] sm:inline-flex border border-ink px-3.5 py-2.5 text-[14px] font-medium text-ink sm:px-4 sm:text-[15px] no-underline transition-colors hover:bg-ink hover:text-paper">How it works</Link>
            </div>
            <dl className="mt-5 grid grid-cols-4 gap-3 border-t border-ink pt-4 sm:mt-7 sm:gap-5" data-testid="hero-telemetry">
              {[
                [<NumberTicker key="c" value={w.corrected_mm} decimals={1} duration={2.2} />, 'mm/day corrected, wettest point'],
                [w.raw_mm.toFixed(1), 'mm/day raw forecast there'],
                w.p_heavy != null ? [`${Math.round(w.p_heavy * 100)}%`, `chance of heavy rain (≥ ${m.thresholds_mm.heavy} mm)`] : [REGIME[w.regime].label, 'regime predicted there'],
                [`${REGIME_ORDER.length}×${GEO_ORDER.length}`, 'regimes × terrain classes, each with its own curves'],
              ].map(([v, label], i) => (
                <div key={i} className="min-w-0">
                  <dt className="sr-only">{label}</dt>
                  <dd className="num text-[length:clamp(20px,2vw,30px)] font-medium leading-none tracking-[-0.03em]">{v}</dd>
                  <dd className="mt-1.5 text-[11.5px] leading-[1.35] text-ink-2 sm:text-[12.5px]">{label}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
        {/* what the backdrop is showing */}
        <div className={`${glass} absolute bottom-6 right-4 hidden w-[330px] p-3.5 md:right-8 lg:block`} data-testid="hero-legend">
          <p className="eyebrow">Corrected rain · mm/day · {fmtDay(w.valid_date)}</p>
          <div className="mt-2 h-2.5 rounded-[2px] border border-ink/15" style={{ background: `linear-gradient(90deg, ${RAIN.map(([v, c]) => `${c} ${(v / 200) * 100}%`).join(',')})` }} />
          <div className="relative mt-1 h-3.5 font-mono text-[10.5px] text-ink-2">
            {[1, 35, m.thresholds_mm.heavy, m.thresholds_mm.very_heavy, 200].map((t) => <span key={t} className="absolute -translate-x-1/2 last:-translate-x-full" style={{ left: `${(t / 200) * 100}%` }}>{t}</span>)}
          </div>
          <p className="mt-2 text-[12px] leading-snug text-ink-2">
            +{grid.leads[w.lead].lead_hours} h ahead. Circle marks the wettest point: {fmtLat(w.lat)} {fmtLon(w.lon)}, a{' '}
            <span className="inline-flex items-center gap-1 text-ink"><span className="inline-block h-2 w-2 rounded-[2px]" style={{ background: REGIME[w.regime].color }} aria-hidden="true" />{REGIME[w.regime].label.toLowerCase()}</span> day. Lines show 850 hPa wind. Faded outside India.
          </p>
        </div>
      </section>

      {/* ---------------------------------------------------------------- vocabulary strip */}
      <div className="border-y border-ink bg-ink py-3 font-mono text-[13px] tracking-wide text-paper">
        <Marquee seconds={55}>
          {vocab(m.thresholds_mm, m.grid_step_deg).map((v) => <span key={v} className="flex items-center gap-10 whitespace-nowrap">{v}<span className="text-[#5d7077]">/</span></span>)}
        </Marquee>
      </div>

      {/* ---------------------------------------------------------------- the problem */}
      <section className="px-5 py-24 md:px-8 md:py-32">
        <div className="grid gap-12 lg:grid-cols-[1fr_1.1fr]">
          <h2 className="display text-[length:var(--section)]">
            <RevealLines lines={['Rain forecasts', 'go wrong in', 'different ways', 'on different days.']} />
          </h2>
          <FadeUp className="self-end">
            <p className="max-w-[46ch] text-[18px] leading-[1.55] text-ink-2">
              Most correction tools fit one curve to every day of the year. That curve is never right for the day that matters. We name the kind of day first, from the forecast itself, then apply the curve built for it.
            </p>
          </FadeUp>
        </div>
        <ul className="mt-16 border-t border-ink">
          {REGIME_ORDER.map((r, i) => (
            <FadeUp key={r} delay={i * 0.05}>
              <li className="grid items-baseline gap-x-8 gap-y-1 border-b border-rule py-5 md:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)_auto]">
                <span className="flex items-baseline gap-4">
                  <span className="inline-block h-[0.55em] w-[0.55em] shrink-0 rounded-[3px] text-[length:clamp(32px,5vw,72px)]" style={{ background: REGIME[r].color }} aria-hidden="true" />
                  <span className="display text-[length:clamp(32px,5vw,72px)]">{REGIME[r].label}</span>
                </span>
                <span className="text-[16px] text-ink-2">{REGIME[r].desc}</span>
                <span className="num text-[13px] text-ink-2">{m.regime_days?.[r] != null ? `${m.regime_days[r].toLocaleString()} training days` : ''}</span>
              </li>
            </FadeUp>
          ))}
        </ul>
      </section>

      {/* ---------------------------------------------------------------- the pipeline, on a pinned map */}
      <Pipeline />

      {/* ---------------------------------------------------------------- honesty */}
      {ver && <WhereItDidNotHelp />}

      {/* ---------------------------------------------------------------- numbers band */}
      {heavy && (
        <section className="border-t border-ink px-5 py-24 md:px-8">
          <div className="flex flex-wrap items-baseline justify-between gap-4">
            <h2 className="display text-[length:var(--section)]"><RevealLines lines={['Six scores, raw', 'against corrected.']} /></h2>
            <p className="eyebrow">Heavy rain · pooled over held-out seasons{m.synthetic && ' · sample values'}</p>
          </div>
          <div className="mt-14 grid gap-x-8 gap-y-12 sm:grid-cols-2 lg:grid-cols-4">
            {([FSS_HEADLINE, 'pod', 'ets', 'rmse'] as MetricKey[]).map((k) => {
              const a = score(heavy.baseline, k), b = score(heavy.corrected, k)
              if (a == null || b == null) return null
              return (
                <FadeUp key={k}>
                  <div className="border-t-2 border-ink pt-3">
                    <div className="flex items-baseline justify-between"><span className="font-display text-[22px] font-extrabold tracking-tight">{METRIC_META[k].name}</span><VerdictTag v={verdict(k, a, b)} /></div>
                    <div className="num mt-4 text-[length:clamp(56px,6.5vw,104px)] font-medium leading-none tracking-[-0.05em]">
                      <NumberTicker value={b} from={a} decimals={k === 'rmse' ? 1 : 2} duration={1.8} />
                    </div>
                    <div className="num mt-2 text-[13px] text-ink-2">from {fmtMetric(k, a)} raw · {METRIC_META[k].long.toLowerCase()}</div>
                  </div>
                </FadeUp>
              )
            })}
          </div>
          <Link to="/scorecard" className="mt-14 inline-block text-[15px] font-medium text-accent">All six metrics, per regime and per season →</Link>
        </section>
      )}

      {/* ---------------------------------------------------------------- close */}
      <section className="relative overflow-hidden border-t border-ink">
        <FieldStage run={run} layer="p_heavy" lead={Math.min(grid.leads.length - 1, 2)} anchorX={wide ? 0.72 : 0.5} particles track className="absolute inset-0 opacity-90" />
        <div className="relative px-5 py-28 md:px-8 md:py-40">
          <Link to="/forecast" className="group no-underline" data-testid="cta-forecast">
            <span className="display block text-[length:var(--hero)] text-ink">
              <span className="tape">Open the</span><br /><span className="tape">forecast map</span>{' '}
              <span className="tape tape-ink inline-block transition-transform duration-300 group-hover:translate-x-3">→</span>
            </span>
          </Link>
          <div className="mt-8 flex flex-wrap gap-2 text-[15px]">
            <Link to="/scorecard" className="tape font-medium text-accent">Read the scorecard</Link>
            <Link to="/method" className="tape font-medium text-accent">How it works</Link>
            <Link to="/bulletin" className="tape font-medium text-accent">Print the bulletin</Link>
          </div>
        </div>
      </section>
      <Footer />
    </main>
  )
}

/* ------------------------------------------------------------------ scroll narrative */

const STEPS: { layer: FieldLayer; title: string; body: string }[] = [
  { layer: 'raw', title: 'Start from the raw forecast.', body: 'Daily rainfall from an open global model, clipped to India on a 0.25° grid. This is what forecasters get today.' },
  { layer: 'regime', title: 'Name the kind of day.', body: 'A gradient-boosted classifier reads only what is known when the forecast is issued: the forecast rain, 850 hPa wind, moisture flux, MJO phase, terrain and coast. It gives a probability for each regime, per cell.' },
  { layer: 'corrected', title: 'Correct with that day’s curve.', body: 'Each regime has its own quantile-mapping curve, fitted on past days of that regime only. The tail is extrapolated, not clipped, so heavy rain keeps its size.' },
  { layer: 'p_heavy', title: 'Say how likely heavy rain is.', body: 'A calibrated model gives the chance of reaching IMD’s heavy ({heavy} mm) and very heavy ({very_heavy} mm) thresholds.' },
  { layer: 'corrected', title: 'Score it against the truth.', body: 'Every season is held out in turn. RMSE, ETS, CSI, POD, FAR and FSS are reported for raw and corrected, including where correction made things worse.' },
]

function Step({ i, onActive, children }: { i: number; onActive: (i: number) => void; children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null)
  const inView = useInView(ref, { margin: '-45% 0px -45% 0px' })
  useEffect(() => { if (inView) onActive(i) }, [inView, i, onActive])
  return <div ref={ref} className="flex min-h-[80vh] flex-col justify-center py-16">{children}</div>
}

function Pipeline() {
  const run = useRun()
  const [active, setActive] = useState(0)
  const heavy = pooledEntry(run.verification, 'heavy')
  const lead = run.manifest.wettest.lead
  const th = run.manifest.thresholds_mm
  const steps = useMemo(() => STEPS.filter((st) => st.layer !== 'p_heavy' || run.manifest.layers.includes('p_heavy'))
    .map((st) => ({ ...st, body: st.body.replace('{heavy}', String(th.heavy)).replace('{very_heavy}', String(th.very_heavy)) })), [run.manifest.layers, th])
  return (
    <section className="relative border-t border-ink" aria-label="How the pipeline works">
      <div className="grid lg:grid-cols-[1.15fr_1fr]">
        <div className="sticky top-0 h-[55vh] lg:h-svh">
          <FieldStage run={run} layer={steps[active].layer} lead={lead} anchorX={0.5} particles={active !== 1} className="relative h-full w-full" />
          <div className="absolute left-4 top-4 flex flex-col gap-1">
            <span className="eyebrow"><span className="tape">Layer shown</span></span>
            <span className="num text-[15px]"><span className="tape tape-ink">{steps[active].layer === 'p_heavy' ? 'P(heavy)' : steps[active].layer}</span></span>
          </div>
          {active === 1 && (
            <ul className="absolute bottom-4 left-4 flex max-w-[90%] flex-wrap gap-1 text-[12.5px]">
              {REGIME_ORDER.map((r) => <li key={r} className="tape flex items-center gap-1.5"><span className="inline-block h-2.5 w-2.5 rounded-[2px]" style={{ background: REGIME[r].color }} />{REGIME[r].label}</li>)}
            </ul>
          )}
        </div>
        <div className="px-5 md:px-10">
          {steps.map((s, i) => (
            <Step key={i} i={i} onActive={setActive}>
              <p className="num text-[13px] text-ink-2">Step {i + 1} of {steps.length}</p>
              <h3 className="display mt-3 text-[length:clamp(34px,4.2vw,64px)]">{s.title}</h3>
              <p className="mt-5 max-w-[44ch] text-[17px] leading-[1.6] text-ink-2">{s.body}</p>
              {s.title.startsWith('Correct') && (
                <div className="mt-8 max-w-[480px] rounded-[4px] border border-rule bg-surface p-4">
                  <QmChart curves={run.curves} show={['global', 'depression', 'break']} height={260} />
                </div>
              )}
              {s.title.startsWith('Score') && heavy && (
                <div className="mt-8 max-w-[520px] rounded-[4px] border border-rule bg-surface p-4">
                  <MetricBars entry={heavy} keys={['rmse', 'pod', 'far', FSS_HEADLINE]} />
                </div>
              )}
            </Step>
          ))}
        </div>
      </div>
    </section>
  )
}

/* ------------------------------------------------------------------ where it did not help */

function WhereItDidNotHelp() {
  const { verification: ver, manifest } = useRun()
  const worse = regimeEntries(ver!)
    .flatMap((e) => METRIC_KEYS.filter((k) => verdict(k, score(e.baseline, k), score(e.corrected, k)) === 'worse').map((k) => {
      const a = score(e.baseline, k)!, b = score(e.corrected, k)!
      return { e, k, a, b, rel: Math.abs(b - a) / Math.max(1e-6, Math.abs(a)) }
    }))
    .sort((a, b) => b.rel - a.rel)
  const total = regimeEntries(ver!).length * METRIC_KEYS.length
  if (!worse.length) return null
  const top = worse[0]
  const reg = top.e.regime as Regime
  return (
    <section className="border-t border-ink bg-surface px-5 py-24 md:px-8" data-testid="did-not-help">
      <p className="eyebrow">Where it did not help{manifest.synthetic && ' · sample values'}</p>
      <div className="mt-6 grid gap-10 lg:grid-cols-[1.3fr_1fr] lg:items-end">
        <h2 className="display text-[length:var(--section)]">
          <RevealLines lines={[`${REGIME[reg].label} spells:`, `${METRIC_META[top.k].name} got worse.`]} />
        </h2>
        <FadeUp>
          <div className="num text-[length:clamp(48px,6vw,96px)] font-medium leading-none tracking-[-0.04em] text-worse">
            {fmtMetric(top.k, top.a)} → {fmtMetric(top.k, top.b)}
          </div>
          <p className="mt-4 max-w-[46ch] text-[16px] text-ink-2">
            {worse.length} of {total} regime-and-metric results got worse after correction. They stay on the scorecard in red, not in a footnote.
          </p>
          <Link to="/scorecard" className="mt-4 inline-block font-medium text-accent">See every regime →</Link>
        </FadeUp>
      </div>
    </section>
  )
}

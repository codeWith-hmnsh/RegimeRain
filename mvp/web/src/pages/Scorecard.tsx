import { useState } from 'react'
import { Footer, Nav } from '../components/Chrome'
import { FadeUp, RevealLines } from '../components/Reveal'
import { DeltaMatrix, MetricBars, QmChart, ReliabilityChart, SeasonDumbbells } from '../components/charts'
import { REGIME, REGIME_ORDER } from '../lib/color'
import { FSS_HEADLINE, METRIC_KEYS, METRIC_META } from '../lib/format'
import { pooledEntry } from '../lib/verify'
import { useRun } from '../lib/run'
import type { MetricKey, Regime, Threshold } from '../lib/types'

function Seg<T extends string>({ value, set, options, label }: { value: T; set: (v: T) => void; options: [T, string][]; label: string }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-[4px] border border-ink p-0.5">
      {options.map(([v, l]) => (
        <button key={v} role="radio" aria-checked={value === v} onClick={() => set(v)}
          className={`rounded-[3px] px-3 py-1.5 text-[13.5px] font-medium ${value === v ? 'bg-ink text-paper' : 'text-ink hover:bg-[#e3e8e6]'}`}>{l}</button>
      ))}
    </div>
  )
}

function Block({ title, note, children }: { title: string; note: string; children: React.ReactNode }) {
  return (
    <FadeUp className="rounded-[6px] border border-rule bg-surface p-5 md:p-6">
      <h2 className="font-display text-[26px] font-extrabold tracking-[-0.03em]">{title}</h2>
      <p className="mb-5 mt-1 max-w-[60ch] text-[14px] text-ink-2">{note}</p>
      {children}
    </FadeUp>
  )
}

export default function Scorecard() {
  const { verification: ver, manifest, curves } = useRun()
  const [th, setTh] = useState<Threshold>('heavy')
  const [metric, setMetric] = useState<MetricKey>(FSS_HEADLINE)
  const [shown, setShown] = useState<Regime[]>(['depression', 'active', 'break'])
  const pooled = pooledEntry(ver, th)
  const lead = ver ? `rain day ${ver.headline_lead}` : ''

  return (
    <>
      <Nav />
      <main className="px-5 pb-24 pt-14 md:px-8" data-testid="scorecard">
        <p className="eyebrow">Verification report · {ver ? `${ver.cv} · ${lead}` : 'not loaded'}{manifest.synthetic && ' · sample values'}</p>
        <h1 className="display mt-4 text-[length:var(--section)]"><RevealLines lines={['The scorecard the', 'problem statement asked for.']} /></h1>
        <p className="mt-6 max-w-[62ch] text-[17px] text-ink-2">
          All six named metrics, for the raw forecast and the corrected one, with every monsoon season held out in turn. Results that got worse are shown in red and stay on the page.
        </p>

        {!ver || !pooled ? (
          <div className="mt-12 rounded-[6px] border border-rule bg-surface p-8" data-testid="no-report">
            <p className="font-display text-[24px] font-extrabold">No verification report in this run</p>
            <p className="mt-2 text-ink-2">Run the leave-one-monsoon-out backtest to write <code className="num">verification.json</code> into the run folder.</p>
          </div>
        ) : (
          <>
            <div className="sticky top-[57px] z-30 -mx-5 mt-10 flex flex-wrap items-center gap-3 border-y border-rule bg-paper/95 px-5 py-3 backdrop-blur md:-mx-8 md:px-8">
              <span className="eyebrow">Threshold</span>
              <Seg value={th} set={setTh} label="Threshold" options={[['heavy', `Heavy ≥ ${manifest.thresholds_mm.heavy} mm`], ['very_heavy', `Very heavy ≥ ${manifest.thresholds_mm.very_heavy} mm`]]} />
            </div>

            <div className="mt-8 grid gap-6 xl:grid-cols-[1.1fr_1fr]">
              <Block title="Pooled over all seasons" note="Grey is the raw forecast, teal is after correction. Bars grow from the raw value so the baseline never disappears.">
                <MetricBars key={th} entry={pooled} big />
              </Block>
              <Block title="Per held-out season" note="Each row is one June–September season scored by a model that never saw it. A red link means that season got worse.">
                <div className="mb-4 flex flex-wrap gap-1">
                  {METRIC_KEYS.map((k) => (
                    <button key={k} onClick={() => setMetric(k)} aria-pressed={metric === k}
                      className={`rounded-[3px] border px-2 py-1 font-mono text-[11.5px] ${metric === k ? 'border-ink bg-ink text-paper' : 'border-rule hover:border-ink'}`}>{METRIC_META[k].name}</button>
                  ))}
                </div>
                <SeasonDumbbells ver={ver} th={th} metric={metric} />
              </Block>
            </div>

            <div className="mt-6">
              <Block title="By regime: where correction helped and where it hurt" note="Change after correction (corrected minus raw). Teal ▲ is better, red ▼ is worse, grey is no meaningful change. Hover a cell for both values.">
                <DeltaMatrix ver={ver} th={th} />
              </Block>
            </div>

            <div className="mt-6 grid gap-6 xl:grid-cols-2">
              {curves.length > 0 && <Block title="Quantile-mapping curves" note="How each regime's curve maps a raw value to a corrected one. The dashed black line is the single global curve most tools use. Sample counts show how much history is behind each curve.">
                <div className="mb-3 flex flex-wrap gap-1">
                  {REGIME_ORDER.map((r) => {
                    const on = shown.includes(r)
                    return (
                      <button key={r} aria-pressed={on} onClick={() => setShown(on ? shown.filter((x) => x !== r) : [...shown, r].slice(-4))}
                        className={`flex items-center gap-1.5 rounded-[3px] border px-2 py-1 text-[12.5px] ${on ? 'border-ink bg-surface' : 'border-rule text-ink-2 hover:border-ink'}`}>
                        <span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: on ? REGIME[r].color : '#c9d2cf' }} />{REGIME[r].label}
                      </button>
                    )
                  })}
                </div>
                <QmChart key={shown.join()} curves={curves} show={['global', ...shown]} />
              </Block>}
              {ver.reliability && <Block title="Reliability of heavy-rain probability" note="When the model says 60%, does heavy rain happen about 60% of the time? Points on the dashed line are perfectly calibrated.">
                <ReliabilityChart ver={ver} />
              </Block>}
            </div>

            <FadeUp className="mt-6 rounded-[6px] border border-rule bg-surface p-5 md:p-6">
              <h2 className="font-display text-[22px] font-extrabold tracking-[-0.03em]">Definitions</h2>
              <dl className="mt-3 grid gap-x-8 gap-y-2 text-[14px] md:grid-cols-2">
                {[
                  ['RMSE', '√(mean of (forecast − truth)²). Continuous error in mm.'],
                  ['POD', 'hits / (hits + misses)'],
                  ['FAR', 'false alarms / (hits + false alarms)'],
                  ['CSI', 'hits / (hits + misses + false alarms)'],
                  ['ETS', 'CSI corrected for hits expected by chance'],
                  ['FSS', `Fractions skill score over square windows of ${ver.fss_windows.map((w) => `${w.cells}×${w.cells} cells (~${w.km} km)`).join(', ')}: rewards the right event in nearly the right place`],
                ].map(([k, v]) => (
                  <div key={k} className="grid grid-cols-[64px_1fr] gap-3 border-t border-rule pt-2"><dt className="num font-medium">{k}</dt><dd className="m-0 text-ink-2">{v}</dd></div>
                ))}
              </dl>
            </FadeUp>
          </>
        )}
      </main>
      <Footer />
    </>
  )
}

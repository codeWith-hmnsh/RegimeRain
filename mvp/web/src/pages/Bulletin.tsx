import { useMemo, useState } from 'react'
import { Nav } from '../components/Chrome'
import { GEO, REGIME } from '../lib/color'
import { fmtDay, fmtMm, fmtPct } from '../lib/format'
import { useRun } from '../lib/run'

export default function Bulletin() {
  const { places, grid, manifest } = useRun()
  const [lead, setLead] = useState(0)
  const probs = manifest.layers.includes('p_heavy')
  // most likely heavy rain first; without the exceedance model, wettest corrected value first
  const rows = useMemo(() => [...places].sort((a, b) => probs
    ? (b.leads[lead].p_heavy ?? 0) - (a.leads[lead].p_heavy ?? 0)
    : b.leads[lead].corrected_mm - a.leads[lead].corrected_mm), [places, lead, probs])
  const th = manifest.thresholds_mm
  return (
    <>
      <Nav />
      <main className="mx-auto max-w-[1100px] px-5 pb-24 pt-10 md:px-8 print:max-w-none print:p-0" data-testid="bulletin">
        <div className="no-print mb-8 flex flex-wrap items-center gap-3">
          <span className="eyebrow">Lead day</span>
          <div role="radiogroup" aria-label="Lead day" className="inline-flex rounded-[4px] border border-ink p-0.5">
            {grid.leads.map((l) => (
              <button key={l.index} role="radio" aria-checked={lead === l.index} onClick={() => setLead(l.index)}
                className={`rounded-[3px] px-3 py-1.5 text-[13px] font-medium ${lead === l.index ? 'bg-ink text-paper' : 'hover:bg-[#e3e8e6]'}`}>{fmtDay(l.valid_date)}</button>
            ))}
          </div>
          <button onClick={() => print()} className="ml-auto rounded-[4px] bg-ink px-4 py-2 text-[14px] font-medium text-paper hover:bg-accent">Print bulletin</button>
        </div>

        <article className="rounded-[6px] border border-ink bg-surface p-6 md:p-10 print:border-0 print:p-0">
          <header className="flex flex-wrap items-end justify-between gap-4 border-b-2 border-ink pb-4">
            <div>
              <p className="eyebrow">Heavy rainfall guidance · regime-corrected</p>
              <h1 className="display mt-2 text-[clamp(36px,5vw,64px)]">{fmtDay(grid.leads[lead].valid_date)}</h1>
            </div>
            <dl className="num grid grid-cols-[auto_auto] gap-x-4 gap-y-0.5 text-[12px]">
              <dt className="text-ink-2">Issued</dt><dd className="m-0">{manifest.forecast_issue_date}</dd>
              <dt className="text-ink-2">Lead</dt><dd className="m-0">+{grid.leads[lead].lead_hours} h</dd>
              <dt className="text-ink-2">Forecast</dt><dd className="m-0">{manifest.forecast_source}</dd>
              <dt className="text-ink-2">Run</dt><dd className="m-0">{manifest.run_id}</dd>
            </dl>
          </header>
          {manifest.synthetic && <p className="mt-4 rounded-[3px] bg-[#fff1c9] px-3 py-2 font-mono text-[12px] text-[#5f4300]">SAMPLE RUN. Invented values. Do not use for any decision.</p>}

          <table className="mt-6 w-full text-[13.5px]">
            <thead>
              <tr className="border-b border-ink text-left">
                {['Place', 'Regime', 'Terrain', 'Raw mm', 'Corrected mm', ...(probs ? [`P ≥ ${th.heavy}`, `P ≥ ${th.very_heavy}`] : [])].map((h, i) => (
                  <th key={h} className={`py-2 font-mono text-[10.5px] font-medium uppercase tracking-[0.08em] text-ink-2 ${i > 2 ? 'text-right' : ''}`}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => {
                const l = p.leads[lead]
                return (
                  <tr key={p.name} className="border-b border-rule">
                    <td className="py-1.5 font-medium">{p.name}</td>
                    <td className="py-1.5"><span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-[2px] align-[-1px]" style={{ background: REGIME[l.regime].color }} />{REGIME[l.regime].label}</td>
                    <td className="py-1.5 text-ink-2">{GEO[p.geo].label}</td>
                    <td className="num py-1.5 text-right text-ink-2">{fmtMm(l.raw_mm)}</td>
                    <td className={`num py-1.5 text-right ${l.corrected_mm >= th.heavy ? 'font-semibold' : ''}`}>{fmtMm(l.corrected_mm)}</td>
                    {probs && <td className="num py-1.5 text-right">{fmtPct(l.p_heavy)}</td>}
                    {probs && <td className="num py-1.5 text-right">{fmtPct(l.p_very_heavy)}</td>}
                  </tr>
                )
              })}
            </tbody>
          </table>

          <section className="mt-8 grid gap-6 border-t-2 border-ink pt-5 md:grid-cols-2">
            <div>
              <h2 className="font-display text-[20px] font-extrabold tracking-tight">How to read this</h2>
              <p className="mt-2 text-[13.5px] text-ink-2">Values are for the {manifest.grid_step_deg}° grid cell containing each place, in mm per IMD rain day (08:30 to 08:30 IST). {probs ? <>Probabilities are calibrated chances of reaching IMD’s heavy ({th.heavy} mm) and very heavy ({th.very_heavy} mm) thresholds.</> : <>Heavy-rain probabilities are not in this run yet.</>}</p>
            </div>
            <div>
              <h2 className="font-display text-[20px] font-extrabold tracking-tight">What this cannot establish</h2>
              <ul className="mt-2 list-disc pl-5 text-[13.5px] text-ink-2">
                <li>Rain at a single street or gauge: one cell averages a large area.</li>
                <li>District totals, until district boundaries are added.</li>
                <li>Skill on NCMRWF’s own models: validated on open models only.</li>
              </ul>
            </div>
          </section>
        </article>
      </main>
    </>
  )
}

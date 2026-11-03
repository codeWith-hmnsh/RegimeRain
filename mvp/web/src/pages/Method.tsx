import { Footer, Nav } from '../components/Chrome'
import { FadeUp, RevealLines } from '../components/Reveal'
import { useRun } from '../lib/run'

// Sources and access state as recorded in docs/PHASE0_FINDINGS.md; models as in docs/MODEL_SPEC.md section 1.
const DATA = [
  ['IFS HRES forecast', 'WeatherBench2, public zarr', 'Raw forecast being corrected, 0.25°, 00 UTC inits, 2016–2022', 'Confirmed'],
  ['IMD gridded rainfall', 'IMD, via imdlib', 'Truth: 0.25° rain days, 08:30 to 08:30 IST', 'Check pending'],
  ['IBTrACS', 'NOAA NCEI', 'Depression tracks, North Indian Ocean', 'Confirmed'],
  ['MJO RMM index', 'Bureau of Meteorology', 'Daily MJO phase and amplitude', 'Manual download'],
  ['NOAA GFS', 'AWS Open Data', 'Today’s forecast for live runs', 'Confirmed'],
  ['CHIRPS 2.0', 'Climate Hazards Center', 'Fallback truth if IMD is unavailable', 'Fallback'],
  ['District boundaries', 'GADM / Survey of India', 'District table and map', 'Licence open'],
]

const MODELS = [
  ['Synoptic regime classifier', 'LightGBM, 4 classes, temperature scaling, spatial smoothing', 'Forecast rain, 850 hPa wind, moisture flux, MSLP, MJO phase, terrain and coast, day of year', 'P(active, break, depression, normal) per cell'],
  ['Regime-aware correction', 'Quantile-mapping experts with a GPD tail and shrinkage', 'One curve per regime × terrain class × zone × lead, blended by regime probability', 'Corrected rainfall grid'],
  ['Exceedance model', 'Two LightGBM classifiers + isotonic calibration', 'Features, corrected rain, regime probabilities', 'P(heavy), P(very heavy)'],
]

export default function Method() {
  const { manifest } = useRun()
  const state = manifest.synthetic ? 'Not trained yet · sample run' : 'Trained · see manifest'
  return (
    <>
      <Nav />
      <main className="px-5 pb-24 pt-14 md:px-8" data-testid="method">
        <p className="eyebrow">Method</p>
        <h1 className="display mt-4 text-[length:var(--section)]"><RevealLines lines={['Three models,', 'all open data,', 'no GPU.']} /></h1>

        <section className="mt-16 grid gap-10 border-t border-ink pt-10 md:grid-cols-4">
          {[
            [`${manifest.thresholds_mm.heavy}`, 'mm/day', 'IMD heavy rainfall'],
            [`${manifest.thresholds_mm.very_heavy}`, 'mm/day', 'IMD very heavy rainfall'],
            [`${manifest.grid_step_deg}°`, 'grid', manifest.synthetic ? 'This run (spec: 0.25°)' : 'Forecast working grid'],
            ['6', 'metrics', 'RMSE, ETS, CSI, POD, FAR, FSS'],
          ].map(([v, u, l]) => (
            <FadeUp key={l}>
              <div className="num text-[length:clamp(56px,7vw,112px)] font-medium leading-none tracking-[-0.05em]">{v}</div>
              <div className="num mt-2 text-[13px] text-ink-2">{u}</div>
              <div className="mt-1 text-[15px]">{l}</div>
            </FadeUp>
          ))}
        </section>

        <section className="mt-24">
          <h2 className="display text-[length:clamp(32px,4vw,56px)]">The models</h2>
          <div className="mt-8 grid gap-4 lg:grid-cols-3">
            {MODELS.map(([name, kind, inputs, out], i) => (
              <FadeUp key={name} delay={i * 0.08} className="flex flex-col rounded-[6px] border border-rule bg-surface p-5">
                <p className="num text-[12px] text-ink-2">Stage {i + 1} of 3</p>
                <h3 className="mt-2 font-display text-[26px] font-extrabold leading-tight tracking-[-0.03em]">{name}</h3>
                <p className="mt-1 text-[14px] font-medium text-accent">{kind}</p>
                <dl className="mt-4 grid gap-3 text-[14px]">
                  <div><dt className="eyebrow">Inputs</dt><dd className="m-0 mt-1 text-ink-2">{inputs}</dd></div>
                  <div><dt className="eyebrow">Output</dt><dd className="m-0 mt-1">{out}</dd></div>
                  <div><dt className="eyebrow">State</dt><dd className={`num m-0 mt-1 text-[13px] ${manifest.synthetic ? 'text-warn' : 'text-good'}`}>{state}</dd></div>
                </dl>
              </FadeUp>
            ))}
          </div>
        </section>

        <section className="mt-24">
          <h2 className="display text-[length:clamp(32px,4vw,56px)]">The data</h2>
          <div className="mt-8 overflow-x-auto rounded-[6px] border border-rule bg-surface">
            <table className="w-full min-w-[720px] text-[14px]">
              <thead>
                <tr className="border-b border-ink text-left">
                  {['Dataset', 'Source', 'Used for', 'Access'].map((h) => <th key={h} className="px-4 py-3 font-mono text-[10.5px] font-medium uppercase tracking-[0.1em] text-ink-2">{h}</th>)}
                </tr>
              </thead>
              <tbody>
                {DATA.map(([d, s, u, a]) => (
                  <tr key={d} className="border-b border-rule last:border-0">
                    <td className="px-4 py-3 font-medium">{d}</td>
                    <td className="px-4 py-3 text-ink-2">{s}</td>
                    <td className="px-4 py-3 text-ink-2">{u}</td>
                    <td className="px-4 py-3"><span className={`num text-[12.5px] ${a === 'Confirmed' ? 'text-good' : 'text-warn'}`}>{a === 'Confirmed' ? '● ' : '○ '}{a}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[13px] text-ink-2">NCMRWF’s own NCUM / NEPS-G output is not public. The pipeline validates on open models and exposes an adapter for NCUM.</p>
        </section>

        <section className="mt-24 grid gap-10 lg:grid-cols-2">
          <FadeUp>
            <h2 className="display text-[length:clamp(32px,4vw,56px)]">No leaks</h2>
            <p className="mt-4 max-w-[52ch] text-[16px] text-ink-2">Each June–September season is held out whole and scored by models that never saw it. A unit test rejects any classifier feature derived from the truth data. Every run records its seed, model hashes and data versions in a manifest.</p>
          </FadeUp>
          <FadeUp delay={0.1}>
            <h2 className="display text-[length:clamp(32px,4vw,56px)]">Known limits</h2>
            <ul className="mt-4 grid gap-2 text-[16px] text-ink-2">
              <li>Gains from bias correction are usually small. The scorecard shows the real size.</li>
              <li>Depression days are rare, so that curve is noisier. Its sample count is shown with it.</li>
              <li>CHIRPS is the truth until IMD gridded rainfall is confirmed.</li>
              <li>Places stand in for districts until the boundary licence is cleared.</li>
            </ul>
          </FadeUp>
        </section>
      </main>
      <Footer />
    </>
  )
}

import { NavLink, Link } from 'react-router-dom'
import { useRun } from '../lib/run'

export function SyntheticBanner() {
  const { manifest } = useRun()
  if (!manifest.synthetic) return null
  return (
    <div data-testid="synthetic-banner" className="no-print relative z-50 flex h-[30px] items-center justify-center overflow-hidden text-ellipsis whitespace-nowrap bg-[#fff1c9] px-4 text-center font-mono text-[11.5px] tracking-wide text-[#5f4300]">
      <span className="sm:hidden">Sample run · invented values · not a forecast</span>
      <span className="hidden sm:inline">Sample run with invented values. Not a forecast, not a measurement. Real runs replace it without UI changes.</span>
    </div>
  )
}

const LINKS = [
  ['/forecast', 'Forecast map'],
  ['/scorecard', 'Scorecard'],
  ['/method', 'Method'],
  ['/bulletin', 'Bulletin'],
] as const

export function Nav({ over = false }: { over?: boolean }) {
  return (
    <header className={`no-print z-40 flex items-center gap-6 px-5 py-3 md:px-8 ${over ? 'absolute inset-x-0 top-0' : 'sticky top-0 border-b border-rule bg-paper/90 backdrop-blur'}`}>
      <Link to="/" className="flex items-center gap-2 bg-paper px-1.5 py-0.5 text-ink no-underline">
        <Drop />
        <span className="font-display text-[19px] font-extrabold tracking-[-0.03em]">RegimeRain</span>
      </Link>
      <nav className="ml-auto flex flex-wrap items-center gap-1" aria-label="Main">
        {LINKS.map(([to, label]) => (
          <NavLink key={to} to={to}
            className={({ isActive }) => `tape rounded-[3px] px-2.5 py-1 text-[13.5px] font-medium no-underline transition-colors ${isActive ? 'tape-ink' : 'text-ink hover:text-accent'}`}>
            {label}
          </NavLink>
        ))}
      </nav>
    </header>
  )
}

export function Drop({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="14" fill="none" stroke="#0e1a1f" strokeWidth="2.5" />
      <path d="M16 6c-4 6-7 9.5-7 13a7 7 0 0 0 14 0c0-3.5-3-7-7-13z" fill="#1c5cab" />
    </svg>
  )
}

export function Footer() {
  const { manifest } = useRun()
  return (
    <footer className="no-print border-t border-rule px-5 py-8 text-[13px] text-ink-2 md:px-8">
      <div className="flex flex-wrap items-baseline gap-x-8 gap-y-2">
        <span className="font-display text-[15px] font-extrabold tracking-tight text-ink">RegimeRain · SIH26080</span>
        <span>MoES / NCMRWF · Regime-aware post-processing of monsoon rainfall forecasts</span>
        <span className="ml-auto font-mono text-[11.5px]">run {manifest.run_id} · coastline © Natural Earth (public domain)</span>
      </div>
      {manifest.synthetic && <p className="mt-3 font-mono text-[11.5px]">Every number on this site comes from the loaded run. This run is synthetic sample data.</p>}
    </footer>
  )
}

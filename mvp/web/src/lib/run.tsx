import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { configureThresholds } from './color'
import { configureMetrics } from './format'
import type { Run } from './types'

/** Loads one run folder. The UI only reads these files; it never computes a forecast or a score. */
async function loadRun(id: string): Promise<Run> {
  const base = `${import.meta.env.BASE_URL}run/${id}/`
  const get = async <T,>(url: string): Promise<T> => {
    const r = await fetch(url)
    if (!r.ok) throw new Error(`${url} returned HTTP ${r.status}`)
    return r.json() as Promise<T>
  }
  const [manifest, grid, places, curves, land, india] = await Promise.all([
    get<Run['manifest']>(base + 'manifest.json'),
    get<Run['grid']>(base + 'grid.json'),
    get<{ places: Run['places'] }>(base + 'places.json'),
    get<{ curves: Run['curves'] }>(base + 'qm_curves.json').catch(() => ({ curves: [] })),   // absent until curves are exported
    get<Run['land']>(`${import.meta.env.BASE_URL}geo/land.json`),
    get<Run['india']>(`${import.meta.env.BASE_URL}geo/india.json`),
  ])
  if (manifest.contract_version !== 2)
    throw new Error(`run ${id} uses web contract v${manifest.contract_version ?? 1}; this app reads v2. Re-export it: regimerain run ... --publish ${id}`)
  const verification = await get<Run['verification']>(base + 'verification.json').catch(() => null)
  // colour breaks, threshold labels and FSS names follow the run, never literals in the code
  configureThresholds(manifest.thresholds_mm)
  configureMetrics(verification?.fss_windows ?? [])
  return { manifest, grid, places: places.places, curves: curves.curves, land, india, verification }
}

type State = { run: Run | null; error: string | null }
const Ctx = createContext<State>({ run: null, error: null })

export function RunProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<State>({ run: null, error: null })
  useEffect(() => {
    const id = new URLSearchParams(location.search).get('run') ?? 'sample'
    loadRun(id).then(
      (run) => setState({ run, error: null }),
      (e: Error) => setState({ run: null, error: e.message }),
    )
  }, [])
  return <Ctx.Provider value={state}>{children}</Ctx.Provider>
}

export const useRunState = () => useContext(Ctx)
export function useRun(): Run {
  const { run } = useContext(Ctx)
  if (!run) throw new Error('useRun called before the run loaded')
  return run
}

import type { MetricKey } from './types'

export const fmtLat = (v: number) => `${Math.abs(v).toFixed(2)}°${v >= 0 ? 'N' : 'S'}`
export const fmtLon = (v: number) => `${Math.abs(v).toFixed(2)}°${v >= 0 ? 'E' : 'W'}`
export const fmtMm = (v: number | null | undefined) => (v == null ? '—' : v.toFixed(1))
export const fmtPct = (v: number | null | undefined) => (v == null ? '—' : `${Math.round(v * 100)}%`)
export const fmtDay = (iso: string) =>
  new Date(iso + 'T00:00:00Z').toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', timeZone: 'UTC' })
type Meta = { name: string; low: boolean; unit: string; long: string }
const BASE_META: Record<string, Meta> = {
  rmse: { name: 'RMSE', low: true, unit: 'mm', long: 'Root-mean-square error' },
  ets: { name: 'ETS', low: false, unit: '', long: 'Equitable threat score' },
  csi: { name: 'CSI', low: false, unit: '', long: 'Critical success index' },
  pod: { name: 'POD', low: false, unit: '', long: 'Probability of detection' },
  far: { name: 'FAR', low: true, unit: '', long: 'False alarm ratio' },
}
/** Metric names and directions. FSS entries are added from the run's verification (its windows, in km). */
export const METRIC_META: Record<string, Meta> = { ...BASE_META }
/** rmse, ets, csi, pod, far, then one FSS key per window the report has. */
export const METRIC_KEYS: MetricKey[] = ['rmse', 'ets', 'csi', 'pod', 'far']
/** The widest FSS window: the headline neighbourhood score. */
export let FSS_HEADLINE: MetricKey = 'fss_1'

/** Called once when a run loads. */
export function configureMetrics(windows: { key: MetricKey; cells: number; km: number }[]) {
  METRIC_KEYS.splice(5)
  for (const w of windows) {
    METRIC_META[w.key] = { name: `FSS ${w.km} km`, low: false, unit: '', long: `Fractions skill score, ${w.cells}×${w.cells} cells (~${w.km} km)` }
    METRIC_KEYS.push(w.key)
  }
  if (windows.length) FSS_HEADLINE = windows[windows.length - 1].key
}
export const fmtMetric = (k: MetricKey, v: number | null | undefined) => (v == null ? '—' : k === 'rmse' ? v.toFixed(1) : v.toFixed(2))
/** Display-only comparison of two report values. Returns 'better' | 'worse' | 'same' (or 'na' when undefined). */
export function verdict(k: MetricKey, raw: number | null | undefined, cor: number | null | undefined) {
  if (raw == null || cor == null) return 'na' as const
  const eps = k === 'rmse' ? 0.05 : 0.005
  const d = cor - raw
  const low = METRIC_META[k]?.low ?? false
  if (Math.abs(d) <= eps) return 'same' as const
  return (low ? d < 0 : d > 0) ? ('better' as const) : ('worse' as const)
}

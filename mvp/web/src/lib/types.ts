/**
 * Web contract v2 (docs/BACKEND_BUILD_PLAN.md section 3). Source of truth: regimerain/runs/contract.py;
 * tests/test_runs.py fails if a field defined there is missing here.
 * Optional fields are layers or values the run does not have (not built yet), never zeros.
 */

/** MODEL_SPEC: four synoptic regimes predicted per cell, plus a static geographic class. */
export type Regime = 'active' | 'break' | 'depression' | 'normal'
export type GeoClass = 'plains' | 'coastal' | 'orographic'
export type Threshold = 'heavy' | 'very_heavy'
export type RunKind = 'mock' | 'replay' | 'model'
/** rmse, ets, csi, pod, far, and fss_<window in cells> from verification.fss_windows. */
export type MetricKey = 'rmse' | 'ets' | 'csi' | 'pod' | 'far' | `fss_${number}`
export type FieldLayer = 'raw' | 'corrected' | 'corrected_global' | 'truth' | 'p_heavy' | 'p_very_heavy' | 'regime' | 'wind850'
export type LayerName = 'raw' | 'corrected' | 'corrected_global' | 'regime' | 'geo' | 'wind850' | 'p_heavy' | 'p_very_heavy' | 'truth'

export interface Provenance {
  model_set_id?: string; model_hashes?: Record<string, string>; config_sha256?: string; git_commit?: string
  backtest_id?: string; held_out_season?: number; seed?: number; note?: string
}

export interface Manifest {
  contract_version: 2
  synthetic: boolean
  kind: RunKind
  run_id: string
  created_utc: string
  forecast_issue_date: string
  forecast_source: string
  truth_source?: string
  grid_step_deg: number
  unit: string
  thresholds_mm: Record<Threshold, number>
  synoptic: Regime[]
  geo_classes: GeoClass[]
  /** Layers this run actually has. The UI hides anything not listed. */
  layers: LayerName[]
  regime_days?: Record<Regime, number>
  depression_track: { lead: number; lat: number; lon: number; mslp_hpa: number }[]
  wettest: {
    lead: number; valid_date: string; lat: number; lon: number
    raw_mm: number; corrected_mm: number; p_heavy?: number; p_very_heavy?: number; regime: Regime; geo: GeoClass
  }
  provenance: Provenance
}

type Row = (number | null)[]

export interface Grid {
  contract_version: 2
  synthetic: boolean
  lat0: number; lon0: number; step: number; nlat: number; nlon: number
  row_order: 'south_to_north'
  synoptic: Regime[]
  geo_classes: GeoClass[]
  /** index into grid.leads; `day` is rain day L; valid_date is the rain day's start (03 UTC to 03 UTC). */
  leads: { index: number; day: number; valid_date: string; lead_hours: number; hours: [number, number] }[]
  /** Static geo class index per cell, -1 off land. */
  geo: number[]
  layers: {
    raw: Row[]; corrected: Row[]; regime: number[][]
    corrected_global?: Row[]; p_heavy?: Row[]; p_very_heavy?: Row[]; truth?: Row[]
    u850?: number[][]; v850?: number[][]; wind850?: number[][]
  }
  /** Per lead: nlat * nlon * synoptic.length, percent. */
  regime_probs_pct: number[][]
}

export interface PlaceLead {
  raw_mm: number; corrected_mm: number; corrected_global_mm?: number
  p_heavy?: number; p_very_heavy?: number; truth_mm?: number
  regime: Regime; regime_probs: Record<Regime, number>; qm_curve?: string; qm_curve_days?: number
}
export interface Place { name: string; lat: number; lon: number; cell: { lat: number; lon: number }; geo: GeoClass; leads: PlaceLead[] }

export type Scores = Partial<Record<MetricKey, number | null>>
export interface VerificationEntry {
  fold: string; lead: number; threshold: Threshold; regime?: Regime
  /** raw NWP */
  baseline: Scores
  /** variant B: regime-aware correction */
  corrected: Scores
  /** variant A: one global curve */
  global?: Scores
  n_samples: number; n_events?: number
}
export interface Delta {
  variant: string; vs: string; lead: number; threshold?: Threshold; metric: string
  delta?: number; ci90?: [number | null, number | null]; significant?: boolean
}
export interface Verification {
  contract_version: 2
  synthetic: boolean
  backtest_id: string
  cv: string
  seasons: number[]
  leads: number[]
  /** Rain day the headline numbers use. */
  headline_lead: number
  variants: string[]
  fss_windows: { key: MetricKey; cells: number; km: number }[]
  entries: VerificationEntry[]
  deltas: Delta[]
  reliability?: Partial<Record<Threshold, { p_forecast: number; p_observed: number; n: number }[]>>
  classifier?: { accuracy_mean: number; macro_f1_mean: number; folds: unknown[] }
}

export interface QmCurve {
  id: string; variant: 'A' | 'B'; regime: Regime | 'global'; geo: GeoClass | 'all'; lead?: number
  n_days: number; w_shrink?: number
  quantiles: number[]; forecast_mm: number[]; truth_mm: number[]
}

export interface Run {
  manifest: Manifest; grid: Grid; places: Place[]; verification: Verification | null
  curves: QmCurve[]; land: GeoJSON.FeatureCollection
  /** Survey of India outline, drawn over any basemap. */
  india: GeoJSON.FeatureCollection
}

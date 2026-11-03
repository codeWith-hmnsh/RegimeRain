import type { MetricKey, Regime, Scores, Threshold, Verification, VerificationEntry } from './types'

/** Entries use one rain day at a time; pages show the report's headline lead. */
const atHeadline = (ver: Verification) => (e: VerificationEntry) => e.lead === ver.headline_lead

/** Pooled over held-out seasons, all regimes (or one regime). */
export const pooledEntry = (ver: Verification | null | undefined, th: Threshold, regime?: Regime) =>
  ver?.entries.find((e) => atHeadline(ver)(e) && e.fold === 'pooled' && e.threshold === th && (regime ? e.regime === regime : !e.regime))

/** One entry per held-out season, all regimes. */
export const seasonEntries = (ver: Verification, th: Threshold) =>
  ver.entries.filter((e) => atHeadline(ver)(e) && e.fold !== 'pooled' && e.threshold === th && !e.regime)

/** Pooled per-regime entries at the headline lead. */
export const regimeEntries = (ver: Verification, th?: Threshold) =>
  ver.entries.filter((e) => atHeadline(ver)(e) && e.fold === 'pooled' && !!e.regime && (!th || e.threshold === th))

/** A score, or null when the report has none (e.g. POD with no observed event). */
export const score = (s: Scores | undefined, k: MetricKey): number | null => {
  const v = s?.[k]
  return v == null || !Number.isFinite(v) ? null : v
}

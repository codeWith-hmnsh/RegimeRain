import { animate, useInView } from 'motion/react'
import { useEffect, useRef } from 'react'
import { prefersReducedMotion } from '../lib/field'

/** Counts up to a run value on arrival. Pattern adapted from Magic UI "Number Ticker". */
export function NumberTicker({ value, decimals = 1, from = 0, className, duration = 1.6 }: { value: number; decimals?: number; from?: number; className?: string; duration?: number }) {
  const ref = useRef<HTMLSpanElement>(null)
  const inView = useInView(ref, { once: true, margin: '-10% 0px' })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (!inView) { el.textContent = from.toFixed(decimals); return }
    if (prefersReducedMotion()) { el.textContent = value.toFixed(decimals); return }
    const c = animate(from, value, { duration, ease: [0.16, 1, 0.3, 1], onUpdate: (v) => { el.textContent = v.toFixed(decimals) } })
    return () => c.stop()
  }, [inView, value, decimals, from, duration])
  return <span ref={ref} className={`num ${className ?? ''}`} aria-label={value.toFixed(decimals)}>{from.toFixed(decimals)}</span>
}

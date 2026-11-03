import type { ReactNode } from 'react'

/** Continuous strip. Pattern adapted from Magic UI "Marquee"; stops under reduced motion. */
export function Marquee({ children, seconds = 40 }: { children: ReactNode; seconds?: number }) {
  return (
    <div className="group relative flex overflow-hidden" aria-hidden="true">
      {[0, 1].map((k) => (
        <div key={k} className="flex shrink-0 items-center gap-10 pr-10 motion-safe:animate-[marquee_var(--d)_linear_infinite]" style={{ ['--d' as string]: `${seconds}s` }}>
          {children}
        </div>
      ))}
      <style>{`@keyframes marquee { from { transform: translateX(0) } to { transform: translateX(-100%) } }`}</style>
    </div>
  )
}

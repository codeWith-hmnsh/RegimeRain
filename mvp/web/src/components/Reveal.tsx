import { motion, useReducedMotion } from 'motion/react'
import type { ReactNode } from 'react'

/** Line-mask reveal for display type (pattern from React Bits / Motion Primitives text reveals). */
export function RevealLines({ lines, className, delay = 0, lineClass }: { lines: ReactNode[]; className?: string; delay?: number; lineClass?: string }) {
  const reduce = useReducedMotion()
  // the in-view trigger sits on the unclipped wrapper; each line slides up inside its own mask
  return (
    <motion.span className={`block ${className ?? ''}`} initial={reduce ? false : 'hidden'} whileInView="show" viewport={{ once: true }}>
      {lines.map((l, i) => (
        <span key={i} className="block overflow-hidden pb-[0.06em]">
          <motion.span
            className={`block ${lineClass ?? ''}`}
            variants={{ hidden: { y: '105%' }, show: { y: 0, transition: { duration: 0.9, delay: delay + i * 0.12, ease: [0.16, 1, 0.3, 1] } } }}
          >
            {l}
          </motion.span>
        </span>
      ))}
    </motion.span>
  )
}

export function FadeUp({ children, delay = 0, className }: { children: ReactNode; delay?: number; className?: string }) {
  const reduce = useReducedMotion()
  return (
    <motion.div className={className} initial={reduce ? false : { opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-8% 0px' }} transition={{ duration: 0.7, delay, ease: [0.16, 1, 0.3, 1] }}>
      {children}
    </motion.div>
  )
}

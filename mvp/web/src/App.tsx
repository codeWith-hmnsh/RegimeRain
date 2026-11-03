import { lazy, Suspense, useEffect } from 'react'
import { BrowserRouter, Route, Routes, useLocation } from 'react-router-dom'
import { RunProvider, useRunState } from './lib/run'
import { SyntheticBanner } from './components/Chrome'
import Landing from './pages/Landing'

const Forecast = lazy(() => import('./pages/Forecast'))
const Scorecard = lazy(() => import('./pages/Scorecard'))
const Method = lazy(() => import('./pages/Method'))
const Bulletin = lazy(() => import('./pages/Bulletin'))

function ScrollTop() {
  const { pathname } = useLocation()
  useEffect(() => { window.scrollTo(0, 0) }, [pathname])
  return null
}

function Loading({ text }: { text: string }) {
  return (
    <div className="grid min-h-svh place-items-center px-6 text-center">
      <p className="font-mono text-[13px] text-ink-2">{text}</p>
    </div>
  )
}

function Shell() {
  const { run, error } = useRunState()
  if (error)
    return (
      <div className="grid min-h-svh place-items-center px-6">
        <div className="max-w-md" data-testid="run-error">
          <p className="display text-[40px]">Run not found</p>
          <p className="mt-3 text-ink-2">{error}. Check the <code className="num">?run=</code> name, or run <code className="num">python3 mvp/tools/make_sample_run.py</code>.</p>
        </div>
      </div>
    )
  if (!run) return <Loading text="Loading run…" />
  return (
    <>
      <SyntheticBanner />
      <ScrollTop />
      <Suspense fallback={<Loading text="Loading page…" />}>
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/forecast" element={<Forecast />} />
          <Route path="/scorecard" element={<Scorecard />} />
          <Route path="/method" element={<Method />} />
          <Route path="/bulletin" element={<Bulletin />} />
          <Route path="*" element={<Loading text="No page here. Use the menu to go back." />} />
        </Routes>
      </Suspense>
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <RunProvider>
        <Shell />
      </RunProvider>
    </BrowserRouter>
  )
}

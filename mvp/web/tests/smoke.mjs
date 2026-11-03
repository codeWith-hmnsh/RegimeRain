// Headless smoke test (FRONTEND_BUILD_PLAN §6). Uses data-testid hooks only.
// Run: npm run build && npm run preview &   then   CHROME=/usr/bin/google-chrome npm run smoke
import puppeteer from 'puppeteer-core'

const BASE = process.env.BASE ?? 'http://localhost:4173'
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME ?? '/usr/bin/google-chrome',
  headless: 'new',
  args: ['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
})
const errors = []
const badText = []   // NaN / undefined visible to a user
let failed = 0
const check = (name, ok) => { console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}`); if (!ok) failed++ }
const wait = (ms) => new Promise((r) => setTimeout(r, ms))

async function open(path) {
  const page = await browser.newPage()
  await page.setViewport({ width: 1440, height: 900 })
  // Offline by default (venue internet is not guaranteed): block every request that leaves BASE, e.g. OSM tiles.
  // The app must still render without them. ONLINE=1 lets them through.
  if (!process.env.ONLINE) {
    await page.setRequestInterception(true)
    page.on('request', (r) => (r.url().startsWith(BASE) || r.url().startsWith('data:') || r.url().startsWith('blob:') ? r.continue() : r.abort()))
  }
  page.on('console', (m) => {
    const blocked = !process.env.ONLINE && /ERR_FAILED|tile\.openstreetmap\.org/.test(m.text() + (m.location()?.url ?? ''))
    if ((m.type() === 'error' || m.type() === 'warning') && !blocked) errors.push(`${path}: ${m.text()}`)
  })
  page.on('pageerror', (e) => errors.push(`${path}: ${e.message}`))
  await page.goto(BASE + path, { waitUntil: 'networkidle0' })
  await wait(1500)
  const text = await page.evaluate(() => document.body.innerText)
  const bad = text.match(/.{0,30}\b(?:NaN|undefined)\b.{0,30}/)
  if (bad) badText.push(`${path}: ${bad[0]}`)
  return page
}
const has = (page, id) => page.$(`[data-testid="${id}"]`).then(Boolean)

let p = await open('/')
check('landing loads', await has(p, 'landing'))
check('hero shows run telemetry', await has(p, 'hero-telemetry'))
check('synthetic banner shown for sample run', await has(p, 'synthetic-banner'))

p = await open('/forecast')
check('forecast map mounts', (await p.$$('.maplibregl-canvas')).length === 2)
check('place labels placed', (await p.$$('.place')).length > 0)
await p.click('[data-testid=layer-rail] [role=radio]:nth-child(2)')
await wait(600)
check('layer switch to regime', await p.$eval('[data-testid=layer-rail] [role=radio]:nth-child(2)', (e) => e.getAttribute('aria-checked') === 'true'))
await p.click('[data-testid=layer-rail] [role=radio]:nth-child(1)')
await p.click('[data-testid=variant] [role=radio]:nth-child(3)')
await wait(600)
check('compare swipe appears', await has(p, 'swipe'))
await p.click('[data-testid=play]')
// software WebGL in headless Chrome slows timers; allow a few ticks, then check the lead moved off day 1
// (measured on Windows headless Chrome: first tick after ~17 s), so poll up to 25 s instead of a fixed wait
let advanced = false
for (let i = 0; i < 25 && !advanced; i++) {
  await wait(1000)
  advanced = await p.$eval('[data-testid=timeline] [role=radio]:nth-child(1)', (e) => e.getAttribute('aria-checked') === 'false')
}
check('timeline play advances lead', advanced)
await p.click('[data-testid=play]')
await p.type('[data-testid=search]', 'Bhub')
await p.keyboard.press('Enter')
await wait(1800)
check('search opens point panel', await has(p, 'point-panel'))
check('correction trace shown', await has(p, 'correction-trace'))

p = await open('/scorecard')
check('scorecard renders metric bars', await has(p, 'metric-bars'))
check('regime delta matrix renders', await has(p, 'delta-matrix'))
p = await open('/method')
check('method renders', await has(p, 'method'))
p = await open('/bulletin')
check('bulletin renders', await has(p, 'bulletin'))
await p.reload({ waitUntil: 'networkidle0' })
check('deep link survives refresh', await has(p, 'bulletin'))

check(`no NaN / undefined on screen (${badText.length})`, badText.length === 0)
check(`zero console errors or warnings (${errors.length})`, errors.length === 0)
badText.forEach((t) => console.log('   ', t))
errors.forEach((e) => console.log('   ', e.slice(0, 240)))
await browser.close()
process.exit(failed ? 1 : 0)

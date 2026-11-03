// Runs Python with the given arguments in the current directory, on any OS.
// macOS/Linux: python3, then python. Windows: python, then `py -3`, because `python3` there is often only the
// Microsoft Store stub. Set PYTHON=/path/to/python to choose one (for example a virtualenv).
import { spawnSync } from 'node:child_process'

const args = process.argv.slice(2)
const candidates = process.env.PYTHON
  ? [[process.env.PYTHON]]
  : process.platform === 'win32'
    ? [['python'], ['py', '-3']]
    : [['python3'], ['python']]

for (const [cmd, ...pre] of candidates) {
  const r = spawnSync(cmd, [...pre, ...args], { stdio: 'inherit' })
  if (r.error && r.error.code === 'ENOENT') continue
  if (r.error) throw r.error
  process.exit(r.status ?? 1)
}
console.error('No Python interpreter found. Install Python 3.10+ or set PYTHON=/path/to/python.')
process.exit(1)

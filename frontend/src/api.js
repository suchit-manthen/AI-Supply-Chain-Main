const BASE = '/api'

async function handle(res) {
  if (!res.ok) {
    let msg = `Request failed (${res.status})`
    try {
      const data = await res.json()
      if (data && data.error) msg = data.error
    } catch { /* ignore */ }
    throw new Error(msg)
  }
  return res.json()
}

export async function getMeta() {
  return handle(await fetch(`${BASE}/meta`))
}

export async function uploadFile(file) {
  const fd = new FormData()
  fd.append('file', file)
  return handle(await fetch(`${BASE}/upload`, { method: 'POST', body: fd }))
}

export async function loadDemo() {
  return handle(await fetch(`${BASE}/demo`, { method: 'POST' }))
}

export async function runForecast(payload) {
  return handle(
    await fetch(`${BASE}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  )
}

export function fmt(n, digits = 1) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return '—'
  return Number(n).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })
}

// z-score lookup for common service levels (client-side inventory recompute)
export const Z_TABLE = { 0.9: 1.282, 0.95: 1.645, 0.98: 2.054, 0.99: 2.326 }
export function zFor(level) {
  return Z_TABLE[level] ?? 1.645
}

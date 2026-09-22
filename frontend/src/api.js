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

export async function recommend(payload) {
  return handle(
    await fetch(`${BASE}/recommend`, {
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

export const RISK_LABELS = { high: 'High', medium: 'Medium', low: 'Low', unknown: 'Unknown' }
export const STATUS_LABELS = {
  stockout: 'Stockout risk',
  reorder: 'Reorder',
  overstock: 'Overstocked',
  ok: 'OK',
  unknown: 'No stock data',
}

// Emoji + tone + plain label per status, for a glanceable manager view.
export const STATUS_META = {
  stockout: { emoji: '🔴', label: 'Order now', tone: 'red' },
  reorder: { emoji: '🟠', label: 'Reorder', tone: 'amber' },
  overstock: { emoji: '🟡', label: 'Overstocked', tone: 'amber' },
  ok: { emoji: '🟢', label: 'Healthy', tone: 'green' },
  unknown: { emoji: '⚪', label: 'No stock data', tone: '' },
}

// Plain-language explanations surfaced on hover for the key terms.
export const TERM_TIPS = {
  'Expected demand': 'How many units the forecast says you will sell during the supplier lead time.',
  'Safety stock': 'Extra stock held to absorb surprise demand spikes during the lead time, so you rarely run out.',
  'Reorder point': 'The stock level that triggers a new order: expected lead-time demand plus safety stock.',
  EOQ: 'Economic Order Quantity — the theoretically cheapest batch size. Shown as a reference; it is not the automatic order.',
  'Lead time': 'Days between placing an order with the supplier and receiving it.',
  'Service level': 'How confident you want to be of not running out (higher = more safety stock).',
}

// Recompute the inventory plan client-side so policy sliders feel instant.
// Mirrors the backend rules: intermittency-capped safety stock, a days-of-cover
// overstock threshold, and demand-based (not probabilistic) risk levels.
export function computeInventory(item, { serviceLevel, leadTime, orderingCost, holdingRate }) {
  const z = zFor(serviceLevel)
  const holding = item.base_price * holdingRate
  const annual = item.annual_demand || 0
  const avgDaily = item.avg_daily_demand || 0
  const leadDemand = avgDaily * leadTime

  let safety = z * item.demand_std * Math.sqrt(leadTime)
  if ((item.intermittency ?? 0) >= 0.2) safety = Math.min(safety, leadDemand)
  const reorder = leadDemand + safety
  const eoq = holding > 0 && annual > 0 ? Math.sqrt((2 * annual * orderingCost) / holding) : null

  const stock = item.current_stock
  let recommended = null
  let daysCover = null
  let stockoutRisk = 'unknown'
  let overstockRisk = 'unknown'
  let status = 'unknown'
  let reason = 'Current stock not provided — upload on-hand inventory to get a reorder quantity.'

  if (avgDaily <= 0) {
    status = 'ok'; stockoutRisk = 'low'; overstockRisk = 'low'; recommended = 0
    reason = 'No recent demand — no order needed.'
  } else if (stock !== null && stock !== undefined) {
    daysCover = stock / avgDaily
    if (stock < leadDemand) {
      status = 'stockout'; stockoutRisk = 'high'; overstockRisk = 'low'
      recommended = Math.max(0, Math.ceil(reorder - stock))
      reason = `Current stock (${stock}) may not cover expected demand (${Math.round(leadDemand)}) during the ${leadTime}-day supplier lead time. Order ${recommended} units now.`
    } else if (stock < reorder) {
      status = 'reorder'; stockoutRisk = 'medium'; overstockRisk = 'low'
      recommended = Math.ceil(reorder - stock)
      reason = `Stock (${stock}) is below the reorder point (${Math.round(reorder)}). Reorder ${recommended} units.`
    } else if (daysCover > Math.max(3 * leadTime, 14)) {
      status = 'overstock'; stockoutRisk = 'low'; overstockRisk = 'high'
      recommended = 0
      reason = `Stock (${stock}) covers ~${Math.round(daysCover)} days — well above the reorder point (${Math.round(reorder)}). No order needed; consider reducing future orders.`
    } else {
      status = 'ok'; stockoutRisk = 'low'; overstockRisk = 'low'
      recommended = 0
      reason = `Stock level is healthy (${daysCover.toFixed(1)} days of cover). No order needed.`
    }
  }

  return {
    ...item,
    lead_time: leadTime,
    expected_demand_lead_time: leadDemand,
    safety_stock: safety,
    reorder_point: reorder,
    eoq,
    recommended_order: recommended,
    days_cover: daysCover ? +daysCover.toFixed(1) : null,
    stockout_risk: stockoutRisk,
    overstock_risk: overstockRisk,
    status,
    reason,
  }
}

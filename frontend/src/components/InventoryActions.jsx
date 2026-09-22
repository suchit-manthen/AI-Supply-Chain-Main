import { useMemo, useState } from 'react'
import { fmt, computeInventory, STATUS_META, TERM_TIPS } from '../api.js'

export default function InventoryActions({ recommendation, runRecommend, recBusy }) {
  const p = recommendation?.inventory_params || {}
  const [serviceLevel, setServiceLevel] = useState(p.service_level ?? 0.95)
  const [leadTime, setLeadTime] = useState(p.lead_time ?? 3)
  const [orderingCost, setOrderingCost] = useState(p.ordering_cost ?? 50)
  const [holdingRate, setHoldingRate] = useState(p.holding_rate ?? 0.2)
  const [filter, setFilter] = useState('all')

  const items = useMemo(() => {
    if (!recommendation) return []
    return (recommendation.inventory || []).map((it) =>
      computeInventory(it, { serviceLevel, leadTime, orderingCost, holdingRate }),
    )
  }, [recommendation, serviceLevel, leadTime, orderingCost, holdingRate])

  if (!recommendation) {
    return (
      <div className="empty-state">
        <div className="empty-icon">📦</div>
        <h2>No inventory plan yet</h2>
        <p className="muted">Generate recommendations from the Overview first.</p>
        <button className="btn primary" onClick={() => runRecommend()} disabled={recBusy}>Generate Recommendations</button>
      </div>
    )
  }

  const filtered = items.filter((it) => {
    if (filter === 'attention') return it.status === 'stockout' || it.status === 'reorder'
    if (filter === 'stockout') return it.stockout_risk === 'high'
    if (filter === 'overstock') return it.overstock_risk === 'high'
    return true
  })

  const orderRank = { stockout: 0, reorder: 1, unknown: 2, overstock: 3, ok: 4 }
  const sorted = [...filtered].sort(
    (a, b) => orderRank[a.status] - orderRank[b.status] || (b.recommended_order || 0) - (a.recommended_order || 0),
  )

  const totalOrder = sorted.reduce((a, b) => a + (b.recommended_order || 0), 0)
  const nAttention = items.filter((i) => i.status === 'stockout' || i.status === 'reorder').length
  const nStockout = items.filter((i) => i.stockout_risk === 'high').length

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h2>Inventory Actions</h2>
          <p className="muted">What to order, for every product — based on the demand forecast.</p>
        </div>
      </div>

      <div className="card">
        <h3>Scenario settings</h3>
        <div className="toolbar">
          <div className="field">
            <label>Service Level</label>
            <select value={serviceLevel} onChange={(e) => setServiceLevel(Number(e.target.value))}>
              <option value={0.9}>90%</option>
              <option value={0.95}>95%</option>
              <option value={0.98}>98%</option>
              <option value={0.99}>99%</option>
            </select>
          </div>
          <div className="field">
            <label>Lead Time (days)</label>
            <input type="number" min="1" max="60" value={leadTime} onChange={(e) => setLeadTime(Number(e.target.value) || 1)} />
          </div>
          <div className="field">
            <label>Ordering Cost ($)</label>
            <input type="number" min="1" value={orderingCost} onChange={(e) => setOrderingCost(Number(e.target.value) || 1)} />
          </div>
          <div className="field">
            <label>Holding Rate (% / yr)</label>
            <input type="number" min="0.01" max="1" step="0.01" value={holdingRate} onChange={(e) => setHoldingRate(Number(e.target.value) || 0.01)} />
          </div>
        </div>
      </div>

      <div className="kpis">
        <KPI label="Total to order" value={fmt(totalOrder, 0)} hint="units across all products" accent="green" />
        <KPI label="Products to reorder" value={fmt(nAttention, 0)} hint="below reorder point" accent={nAttention ? 'red' : 'green'} />
        <KPI label="Stockout risk" value={fmt(nStockout, 0)} hint="may run out during lead time" accent={nStockout ? 'red' : 'green'} />
        <KPI label="Service level" value={`${fmt(serviceLevel * 100, 0)}%`} hint="target fill rate" />
      </div>

      <div className="card">
        <div className="toolbar" style={{ marginBottom: 12 }}>
          <h3 style={{ margin: 0 }}>Recommended actions</h3>
          <div className="spacer" />
          <div className="field">
            <label>Show</label>
            <select value={filter} onChange={(e) => setFilter(e.target.value)}>
              <option value="all">All products</option>
              <option value="attention">Needs attention</option>
              <option value="stockout">Stockout risk</option>
              <option value="overstock">Overstocked</option>
            </select>
          </div>
        </div>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Product</th>
                <th>Status</th>
                <th>Current stock</th>
                <Th tip={TERM_TIPS['Lead time']}>Lead time</Th>
                <Th tip={TERM_TIPS['Expected demand']}>Expected demand</Th>
                <Th tip={TERM_TIPS['Safety stock']}>Safety stock</Th>
                <Th tip={TERM_TIPS['Reorder point']}>Reorder point</Th>
                <th>Action</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <tr key={r.product_id}>
                  <td><b>{r.product_name}</b> <span className="muted">{r.product_id}</span></td>
                  <td><StatusBadge status={r.status} /></td>
                  <td>{r.current_stock === null ? '—' : fmt(r.current_stock, 0)}</td>
                  <td>{r.lead_time}d</td>
                  <td>{fmt(r.expected_demand_lead_time, 0)}</td>
                  <td>{fmt(r.safety_stock, 0)}</td>
                  <td>{fmt(r.reorder_point, 0)}</td>
                  <td>
                    {r.recommended_order
                      ? <span className="action">{STATUS_META[r.status]?.emoji} Order {fmt(r.recommended_order, 0)} units</span>
                      : <span className="muted">—</span>}
                  </td>
                  <td className="reason-cell">{r.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="legend-note">
          <span className="tip" title={TERM_TIPS['Reorder point']}>Reorder point</span> = expected lead-time demand +{' '}
          <span className="tip" title={TERM_TIPS['Safety stock']}>safety stock</span>. Recommended order = reorder point −
          current stock. <span className="tip" title={TERM_TIPS.EOQ}>EOQ</span> is a reference batch size, not the automatic order.
        </p>
      </div>
    </div>
  )
}

function Th({ children, tip }) {
  return <th>{children}{tip && <span className="tip" title={tip}> ⓘ</span>}</th>
}

function StatusBadge({ status }) {
  const meta = STATUS_META[status] || {}
  return <span className={`badge ${meta.tone}`}>{meta.emoji} {meta.label}</span>
}

function KPI({ label, value, hint, accent }) {
  return (
    <div className="card kpi">
      <div className="label">{label}</div>
      <div className="value" style={accent ? { color: `var(--${accent})` } : undefined}>{value}</div>
      <div className="hint">{hint}</div>
    </div>
  )
}

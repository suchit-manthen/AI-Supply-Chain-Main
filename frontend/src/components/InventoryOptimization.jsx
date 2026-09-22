import { useMemo, useState } from 'react'
import { fmt, zFor } from '../api.js'

export default function InventoryOptimization({ results, onGoForecast }) {
  const [serviceLevel, setServiceLevel] = useState(0.95)
  const [leadTime, setLeadTime] = useState(3)
  const [orderingCost, setOrderingCost] = useState(50)
  const [holdingRate, setHoldingRate] = useState(0.2)

  const items = useMemo(() => {
    const raw = results?.inventory || []
    const z = zFor(serviceLevel)
    return raw.map((it) => {
      const holding = it.base_price * holdingRate
      const safety = z * it.demand_std * Math.sqrt(leadTime)
      const reorder = it.avg_daily_demand * leadTime + safety
      const eoq = holding ? Math.sqrt((2 * it.annual_demand * orderingCost) / holding) : 0
      return { ...it, safety_stock: safety, reorder_point: reorder, eoq, holding_cost_per_unit: holding }
    })
  }, [results, serviceLevel, leadTime, orderingCost, holdingRate])

  if (!results) {
    return <Empty onGo={onGoForecast} />
  }

  const totalSafety = items.reduce((a, b) => a + (b.safety_stock || 0), 0)
  const totalReorder = items.reduce((a, b) => a + (b.reorder_point || 0), 0)

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h2>Inventory Optimization</h2>
          <p className="muted">Safety Stock, Reorder Point and EOQ per SKU — recomputed live.</p>
        </div>
      </div>

      <div className="card">
        <h3>Policy Parameters</h3>
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
          <div className="field">
            <label>Z-score</label>
            <div style={{ paddingTop: 8, fontWeight: 600 }}>{fmt(zFor(serviceLevel), 3)}</div>
          </div>
        </div>
      </div>

      <div className="kpis">
        <KPI label="Total Safety Stock" value={fmt(totalSafety, 0)} hint="units to buffer demand variability" />
        <KPI label="Total Reorder Point" value={fmt(totalReorder, 0)} hint="inventory level to trigger reorder" />
        <KPI label="Service Level" value={`${fmt(serviceLevel * 100, 0)}%`} hint="target fill rate" />
        <KPI label="Lead Time" value={fmt(leadTime, 0)} hint="days to replenish" />
      </div>

      <div className="card">
        <h3>Replenishment Plan per SKU</h3>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>SKU</th><th>Product</th><th>Category</th>
                <th>Avg Daily Demand</th><th>Demand Std</th>
                <th>Safety Stock</th><th>Reorder Point</th><th>EOQ</th>
              </tr>
            </thead>
            <tbody>
              {items.map((r) => (
                <tr key={r.product_id}>
                  <td><b>{r.product_id}</b></td>
                  <td>{r.product_name}</td>
                  <td><span className="badge">{r.category}</span></td>
                  <td>{fmt(r.avg_daily_demand, 1)}</td>
                  <td>{fmt(r.demand_std, 1)}</td>
                  <td><span className="badge amber">{fmt(r.safety_stock, 0)}</span></td>
                  <td><span className="badge red">{fmt(r.reorder_point, 0)}</span></td>
                  <td>{fmt(r.eoq, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="legend-note">
          Safety Stock = z · σ<sub>demand</sub> · √lead_time &nbsp;·&nbsp; Reorder Point = avg_demand × lead_time + safety stock &nbsp;·&nbsp; EOQ = √(2·D·S/H)
        </p>
      </div>
    </div>
  )
}

function KPI({ label, value, hint }) {
  return (
    <div className="card kpi">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      <div className="hint">{hint}</div>
    </div>
  )
}

function Empty({ onGo }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">📦</div>
      <h2>No inventory data yet</h2>
      <p className="muted">Run a forecast to compute safety stock, reorder point and EOQ.</p>
      <button className="btn primary" onClick={onGo}>Go to Demand Forecast</button>
    </div>
  )
}

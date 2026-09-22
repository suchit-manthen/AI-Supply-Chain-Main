import { useMemo, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts'
import { fmt } from '../api.js'

export default function DemandForecast({ recommendation, runRecommend, recBusy }) {
  const products = recommendation?.products || []
  const [pid, setPid] = useState('')

  const active = pid || products[0]?.product_id || ''

  if (!recommendation) {
    return (
      <div className="empty-state">
        <div className="empty-icon">📈</div>
        <h2>No forecast yet</h2>
        <p className="muted">Generate recommendations from the Overview first.</p>
        <button className="btn primary" onClick={() => runRecommend()} disabled={recBusy}>Generate Recommendations</button>
      </div>
    )
  }

  const product = products.find((p) => p.product_id === active)
  const pattern = recommendation.patterns?.[active] || {}
  const rec = recommendation.recommendation?.[active] || {}
  const inv = (recommendation.inventory || []).find((i) => i.product_id === active)

  const hist = recommendation.history?.[active] || []
  const fc = recommendation.forecast?.[active] || []
  const data = useMemo(() => {
    const byDate = {}
    hist.forEach((h) => { byDate[h.date] = byDate[h.date] || {}; byDate[h.date].actual = h.sales })
    fc.forEach((f) => { byDate[f.date] = byDate[f.date] || {}; byDate[f.date].forecast = f.predicted })
    return Object.keys(byDate).sort().map((d) => ({ date: d, ...byDate[d] }))
  }, [hist, fc])

  const nextSum = fc.reduce((a, b) => a + (b.predicted || 0), 0)

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h2>Demand Forecast</h2>
          <p className="muted">Past demand and the expected demand ahead.</p>
        </div>
      </div>

      <div className="card">
        <div className="toolbar">
          <div className="field" style={{ flex: 1, minWidth: 280 }}>
            <label>Product</label>
            <select value={active} onChange={(e) => setPid(e.target.value)}>
              {products.map((p) => (
                <option key={p.product_id} value={p.product_id}>{p.product_name} — {p.product_id}</option>
              ))}
            </select>
          </div>
          {product && (
            <div className="field">
              <label>Category</label>
              <div style={{ paddingTop: 8 }}><span className="badge">{product.category}</span></div>
            </div>
          )}
          {inv && (
            <>
              <div className="field">
                <label>Current stock</label>
                <div style={{ paddingTop: 8, fontWeight: 600 }}>{fmt(inv.current_stock, 0)}</div>
              </div>
              <div className="field">
                <label>Expected next {recommendation.horizon}d</label>
                <div style={{ paddingTop: 8, fontWeight: 600 }}>{fmt(nextSum, 0)}</div>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="card">
        <h3>Demand pattern</h3>
        <div className="chips">
          {(rec.tag_labels || []).map((t) => <span key={t} className="badge">{t}</span>)}
        </div>
        {rec.reason && <p className="reason-cell" style={{ marginTop: 10 }}>{rec.reason}</p>}
      </div>

      <div className="card">
        <h3>Historical demand + future forecast</h3>
        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={40} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v) => fmt(v, 1)} />
            <Legend />
            <Line type="monotone" dataKey="actual" stroke="#2563eb" dot={false} strokeWidth={2} name="Past demand" />
            <Line type="monotone" dataKey="forecast" stroke="#10b981" dot={false} strokeWidth={2} name="Forecast" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

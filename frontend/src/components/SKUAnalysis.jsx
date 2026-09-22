import { useMemo, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts'
import { fmt, RISK_LABELS, STATUS_META, TERM_TIPS } from '../api.js'

export default function SKUAnalysis({ recommendation, runRecommend, recBusy }) {
  const products = recommendation?.products || []
  const [pid, setPid] = useState('')
  const active = pid || products[0]?.product_id || ''

  if (!recommendation) {
    return (
      <div className="empty-state">
        <div className="empty-icon">🏷️</div>
        <h2>No results yet</h2>
        <p className="muted">Generate recommendations from the Overview first.</p>
        <button className="btn primary" onClick={() => runRecommend()} disabled={recBusy}>Generate Recommendations</button>
      </div>
    )
  }

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

  const signals = useMemo(() => {
    const p = recommendation.patterns?.[active] || {}
    return [
      { label: 'Weekly rhythm', value: p.weekly, fmt: (v) => v >= 0.15 ? 'strong' : 'weak' },
      { label: 'Seasonal pattern', value: p.yearly, fmt: (v) => v >= 0.12 ? 'strong' : 'weak' },
      { label: 'Promotion sensitivity', value: p.promo, fmt: (v) => v >= 0.2 ? 'high' : 'low' },
      { label: 'Holiday spikes', value: p.holiday, fmt: (v) => v >= 0.15 ? 'strong' : 'weak' },
      { label: 'Day-to-day variation', value: Math.min(p.volatility / 0.8, 1), fmt: (v) => v >= 0.45 ? 'high' : 'low' },
      { label: 'Irregular (zero) days', value: Math.min(p.intermittency / 0.3, 1), fmt: (v) => v >= 0.15 ? 'frequent' : 'rare' },
    ]
  }, [recommendation, active])

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h2>Product Analysis</h2>
          <p className="muted">A closer look at one product's demand and plan.</p>
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
          {inv && (
            <div className="field">
              <label>Category</label>
              <div style={{ paddingTop: 8 }}><span className="badge">{inv.category}</span></div>
            </div>
          )}
        </div>
      </div>

      <div className="row">
        <div className="card">
          <h3>Demand profile</h3>
          <div className="chips" style={{ marginBottom: 14 }}>
            {(rec.tag_labels || []).map((t) => <span key={t} className="badge">{t}</span>)}
          </div>
          {signals.map((s) => (
            <div className="signal" key={s.label}>
              <div className="signal-head">
                <span>{s.label}</span>
                <span className="muted">{s.fmt(s.value)}</span>
              </div>
              <div className="signal-bar"><div className="signal-fill" style={{ width: `${Math.round((s.value || 0) * 100)}%` }} /></div>
            </div>
          ))}
          {rec.reason && <p className="reason-cell" style={{ marginTop: 14 }}>{rec.reason}</p>}
        </div>

        <div className="card">
          <h3>Plan for this product</h3>
          {inv ? (
            <table>
              <tbody>
                <tr>
                  <td className="muted">Status</td>
                  <td><span className={`badge ${STATUS_META[inv.status]?.tone}`}>{STATUS_META[inv.status]?.emoji} {STATUS_META[inv.status]?.label}</span></td>
                </tr>
                <tr><td className="muted">Current stock</td><td><b>{fmt(inv.current_stock, 0)}</b></td></tr>
                <tr><td className="muted"><span className="tip" title={TERM_TIPS['Expected demand']}>Expected demand (lead time) ⓘ</span></td><td><b>{fmt(inv.expected_demand_lead_time, 0)}</b></td></tr>
                <tr><td className="muted"><span className="tip" title={TERM_TIPS['Safety stock']}>Safety stock ⓘ</span></td><td>{fmt(inv.safety_stock, 0)}</td></tr>
                <tr><td className="muted"><span className="tip" title={TERM_TIPS['Reorder point']}>Reorder point ⓘ</span></td><td>{fmt(inv.reorder_point, 0)}</td></tr>
                <tr><td className="muted"><span className="tip" title={TERM_TIPS.EOQ}>Suggested order size (EOQ) ⓘ</span></td><td>{fmt(inv.eoq, 0)}</td></tr>
                <tr>
                  <td className="muted">Action</td>
                  <td>{inv.recommended_order
                    ? <span className="action">{STATUS_META[inv.status]?.emoji} Order {fmt(inv.recommended_order, 0)} units</span>
                    : <span className="muted">— no order needed</span>}</td>
                </tr>
                <tr><td className="muted">Stockout risk</td><td><span className={`badge ${inv.stockout_risk === 'high' ? 'red' : inv.stockout_risk === 'medium' ? 'amber' : 'green'}`}>{RISK_LABELS[inv.stockout_risk]}</span></td></tr>
              </tbody>
            </table>
          ) : <p className="muted">No plan available.</p>}
          {inv?.reason && <p className="reason-cell" style={{ marginTop: 12 }}>{inv.reason}</p>}
        </div>
      </div>

      <div className="card">
        <h3>Demand history + forecast</h3>
        <ResponsiveContainer width="100%" height={300}>
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

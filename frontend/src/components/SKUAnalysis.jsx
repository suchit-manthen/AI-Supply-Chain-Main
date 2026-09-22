import { useMemo, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts'
import { fmt } from '../api.js'

export default function SKUAnalysis({ results, onGoForecast }) {
  const products = results?.products || []
  const [pid, setPid] = useState('')
  const [model, setModel] = useState('')

  const active = pid || products[0]?.product_id || ''

  const skuRows = useMemo(() => {
    const rows = {}
    for (const m of results?.models || []) {
      for (const row of results?.sku_metrics?.[m] || []) {
        if (row.product_id !== active) continue
        rows[m] = row
      }
    }
    return rows
  }, [results, active])

  const sortedModels = useMemo(() => {
    return [...(results?.models || [])].sort((a, b) => (skuRows[b]?.mape ?? 1e9) - (skuRows[a]?.mape ?? 1e9))
  }, [results, skuRows])

  const chosenModel = model || sortedModels[0] || ''
  const series = results?.actual_vs_predicted?.[chosenModel]?.[active] || []
  const product = products.find((p) => p.product_id === active)

  if (!results) {
    return (
      <div className="empty-state">
        <div className="empty-icon">🏷️</div>
        <h2>No results yet</h2>
        <p className="muted">Run a forecast to explore per-product accuracy.</p>
        <button className="btn primary" onClick={onGoForecast}>Go to Demand Forecast</button>
      </div>
    )
  }

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h2>Product / SKU Analysis</h2>
          <p className="muted">Drill into a single product's demand and model accuracy.</p>
        </div>
      </div>

      <div className="card">
        <div className="toolbar" style={{ marginBottom: 0 }}>
          <div className="field">
            <label>Product</label>
            <select value={active} onChange={(e) => setPid(e.target.value)}>
              {products.map((p) => (
                <option key={p.product_id} value={p.product_id}>{p.product_id} — {p.product_name}</option>
              ))}
            </select>
          </div>
          {product && (
            <div className="field">
              <label>Info</label>
              <div style={{ paddingTop: 8 }}>
                <span className="badge">{product.category}</span>{' '}
                <span className="muted" style={{ marginLeft: 6 }}>avg {fmt(product.avg_sales, 1)} units/day</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <h3>Model Accuracy for {active}</h3>
        <table>
          <thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE</th></tr></thead>
          <tbody>
            {sortedModels.map((m) => {
              const row = skuRows[m]
              return (
                <tr key={m} style={m === sortedModels[0] ? { background: '#ecfdf5' } : undefined}>
                  <td>
                    {results.model_labels?.[m] || m}
                    {m === sortedModels[0] && <span className="badge green" style={{ marginLeft: 8 }}>Best</span>}
                  </td>
                  <td>{row ? fmt(row.mae, 2) : '—'}</td>
                  <td>{row ? fmt(row.rmse, 2) : '—'}</td>
                  <td><b>{row ? `${fmt(row.mape, 2)}%` : '—'}</b></td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="card">
        <div className="toolbar" style={{ marginBottom: 8 }}>
          <div className="field">
            <label>Model</label>
            <select value={chosenModel} onChange={(e) => setModel(e.target.value)}>
              {(results.models || []).map((m) => <option key={m} value={m}>{results.model_labels?.[m] || m}</option>)}
            </select>
          </div>
        </div>
        <h3>Actual vs Predicted — {active}</h3>
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={series} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={40} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v) => fmt(v, 1)} />
            <Legend />
            <Line type="monotone" dataKey="actual" stroke="#2563eb" dot={false} strokeWidth={2} name="Actual" />
            <Line type="monotone" dataKey="predicted" stroke="#ef4444" dot={false} strokeWidth={1.5} name="Predicted" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

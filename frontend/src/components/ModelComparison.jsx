import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts'
import { fmt } from '../api.js'

export default function ModelComparison({ results, onGoForecast }) {
  if (!results) {
    return <Empty onGo={onGoForecast} title="No results yet" text="Run a Compare All forecast to see model performance side by side." />
  }

  const models = [...results.models].sort(
    (a, b) => (results.test_metrics[b]?.mape ?? 1e9) - (results.test_metrics[a]?.mape ?? 1e9),
  )
  const chartData = models.map((m) => {
    const t = results.test_metrics[m] || {}
    return { model: results.model_labels?.[m] || m, mae: t.mae, rmse: t.rmse, mape: t.mape }
  })

  // Build a per-SKU leaderboard: best model per SKU by MAPE.
  const skuBest = {}
  for (const m of models) {
    for (const row of results.sku_metrics?.[m] || []) {
      if (!(row.product_id in skuBest) || row.mape < skuBest[row.product_id].mape) {
        skuBest[row.product_id] = { model: m, mape: row.mape }
      }
    }
  }
  const winCount = {}
  for (const v of Object.values(skuBest)) winCount[v.model] = (winCount[v.model] || 0) + 1

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h2>Model Comparison</h2>
          <p className="muted">Measured on the uploaded dataset. Lower MAPE = better fit for this data.</p>
        </div>
      </div>

      <div className="card">
        <h3>Accuracy by Model</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="model" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip formatter={(v) => fmt(v, 2)} />
            <Legend />
            <Bar dataKey="mae" fill="#2563eb" radius={[4, 4, 0, 0]} name="MAE" />
            <Bar dataKey="rmse" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="RMSE" />
          </BarChart>
        </ResponsiveContainer>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="model" tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} unit="%" />
            <Tooltip formatter={(v) => [`${fmt(v, 2)}%`, 'MAPE']} />
            <Bar dataKey="mape" fill="#10b981" radius={[4, 4, 0, 0]} name="MAPE (%)" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="row">
        <div className="card">
          <h3>Leaderboard</h3>
          <table>
            <thead><tr><th>Rank</th><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE</th></tr></thead>
            <tbody>
              {models.map((m, i) => {
                const t = results.test_metrics[m]
                return (
                  <tr key={m} style={i === 0 ? { background: '#ecfdf5' } : undefined}>
                    <td>{i + 1}</td>
                    <td>{results.model_labels?.[m] || m}</td>
                    <td>{t ? fmt(t.mae, 2) : '—'}</td>
                    <td>{t ? fmt(t.rmse, 2) : '—'}</td>
                    <td><b>{t ? `${fmt(t.mape, 2)}%` : '—'}</b></td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          <p className="legend-note">No single algorithm is universally best — the ranking here reflects only this dataset's structure.</p>
        </div>

        <div className="card">
          <h3>Best Model per SKU</h3>
          <table>
            <thead><tr><th>Model</th><th>SKUs won</th></tr></thead>
            <tbody>
              {models.map((m) => (
                <tr key={m}>
                  <td>{results.model_labels?.[m] || m}</td>
                  <td><span className="badge">{winCount[m] || 0}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="legend-note">Counts how many SKUs each model achieved the lowest MAPE.</p>
        </div>
      </div>
    </div>
  )
}

function Empty({ onGo, title, text }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">⚖️</div>
      <h2>{title}</h2>
      <p className="muted">{text}</p>
      <button className="btn primary" onClick={onGo}>Go to Demand Forecast</button>
    </div>
  )
}

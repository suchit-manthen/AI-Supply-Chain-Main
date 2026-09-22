import { useEffect, useMemo, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from 'recharts'
import { runForecast, fmt } from '../api.js'

export default function DemandForecast({ dataset, meta, results, onResults, onRequest }) {
  const modelLabels = meta?.model_labels || {}
  const modelDescriptions = meta?.model_descriptions || {}
  const defaults = meta?.defaults || {}

  const [category, setCategory] = useState('all')
  const [product, setProduct] = useState('all')
  const [selected, setSelected] = useState([]) // model keys
  const [horizon, setHorizon] = useState(30)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const [chartProduct, setChartProduct] = useState('all')
  const [chartModel, setChartModel] = useState('')

  const products = dataset?.products || []
  const categories = dataset?.report?.categories || []

  const filteredProducts = useMemo(
    () => (category === 'all' ? products : products.filter((p) => p.category === category)),
    [products, category],
  )

  useEffect(() => {
    setProduct('all')
  }, [category])

  function toggleModel(m) {
    setSelected((s) => (s.includes(m) ? s.filter((x) => x !== m) : [...s, m]))
  }

  function selectAll() {
    setSelected(meta?.models ? [...meta.models] : [])
  }

  async function run() {
    if (selected.length === 0) {
      setError('Select at least one algorithm.')
      return
    }
    setBusy(true)
    setError(null)
    const payload = {
      dataset_id: dataset.dataset_id,
      models: selected,
      products: product === 'all' ? null : [product],
      category: category === 'all' ? null : category,
      horizon,
    }
    try {
      const res = await runForecast(payload)
      onResults(res)
      onRequest(payload)
      setChartModel(res.models[0])
      setChartProduct('all')
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const modelKeys = meta?.models || ['sarima', 'prophet', 'random_forest', 'xgboost', 'lstm_gru']

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>Demand Forecast</h2>
          <p className="muted">Select products and algorithms, then run forecasting.</p>
        </div>
      </div>

      <div className="card">
        <h3>1 · Select Scope</h3>
        <div className="toolbar">
          <div className="field">
            <label>Category</label>
            <select value={category} onChange={(e) => setCategory(e.target.value)}>
              <option value="all">All categories</option>
              {categories.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Product</label>
            <select value={product} onChange={(e) => setProduct(e.target.value)}>
              <option value="all">All products ({filteredProducts.length})</option>
              {filteredProducts.map((p) => (
                <option key={p.product_id} value={p.product_id}>{p.product_id} — {p.product_name}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Future Horizon (days)</label>
            <input type="number" min="1" max="180" value={horizon} onChange={(e) => setHorizon(Number(e.target.value) || 30)} />
          </div>
        </div>

        <h3>2 · Select Algorithm</h3>
        <div className="model-picker">
          {modelKeys.map((m) => {
            const on = selected.includes(m)
            return (
              <button
                key={m}
                type="button"
                className={`model-card ${on ? 'on' : ''}`}
                onClick={() => toggleModel(m)}
                aria-pressed={on}
              >
                <span className={`m-check ${on ? 'checked' : ''}`}>{on ? '✓' : ''}</span>
                <span className="m-body">
                  <span className="m-name">{modelLabels[m] || m}</span>
                  <span className="m-desc">{modelDescriptions[m] || ''}</span>
                </span>
              </button>
            )
          })}
        </div>
        <div className="toolbar" style={{ marginTop: 12 }}>
          <button className="btn ghost" onClick={selectAll}>Compare All</button>
          <button className="btn ghost" onClick={() => setSelected([])}>Clear</button>
          <button className="btn primary" onClick={run} disabled={busy}>
            {busy ? 'Running models…' : 'Run Forecasting'}
          </button>
        </div>
        {busy && <p className="muted" style={{ marginTop: 8 }}>Training on the selected scope. Compare All on many SKUs can take a minute or two.</p>}
      </div>

      {error && <div className="error banner">{error}</div>}

      {results && (
        <Results
          results={results}
          modelLabels={modelLabels}
          chartProduct={chartProduct}
          setChartProduct={setChartProduct}
          chartModel={chartModel}
          setChartModel={setChartModel}
          productOptions={results.products || []}
          horizon={results.horizon}
        />
      )}
    </div>
  )
}

function Results({ results, modelLabels, chartProduct, setChartProduct, chartModel, setChartModel, productOptions, horizon }) {
  const sorted = [...results.models].sort((a, b) => (results.test_metrics[b]?.mape ?? 1e9) - (results.test_metrics[a]?.mape ?? 1e9))
  const chosenModel = chartModel || results.models[0]
  const pid = chartProduct === 'all' ? (productOptions[0]?.product_id) : chartProduct

  const series = results.actual_vs_predicted?.[chosenModel]?.[pid] || []
  const future = results.future?.[chosenModel]?.[pid] || []

  return (
    <div className="stack">
      <div className="card">
        <h3>3 · Test-Set Accuracy (measured on this dataset)</h3>
        <table>
          <thead>
            <tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>Time (s)</th></tr>
          </thead>
          <tbody>
            {sorted.map((m) => {
              const t = results.test_metrics[m]
              return (
                <tr key={m} style={m === sorted[0] ? { background: '#ecfdf5' } : undefined}>
                  <td>
                    {modelLabels[m] || m}
                    {m === sorted[0] && <span className="badge green" style={{ marginLeft: 8 }}>Lowest MAPE</span>}
                  </td>
                  <td>{t ? fmt(t.mae, 2) : '—'}</td>
                  <td>{t ? fmt(t.rmse, 2) : '—'}</td>
                  <td><b>{t ? `${fmt(t.mape, 2)}%` : '—'}</b></td>
                  <td className="muted">{fmt(results.runtime_sec?.[m], 1)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        {Object.keys(results.errors || {}).length > 0 && (
          <div className="error" style={{ marginTop: 12 }}>
            {Object.entries(results.errors).map(([m, e]) => <div key={m}>{modelLabels[m] || m}: {e}</div>)}
          </div>
        )}
      </div>

      <div className="card">
        <div className="toolbar" style={{ marginBottom: 8 }}>
          <div className="field">
            <label>Product</label>
            <select value={chartProduct} onChange={(e) => setChartProduct(e.target.value)}>
              {productOptions.length > 1 && <option value="all">First product</option>}
              {productOptions.map((p) => (
                <option key={p.product_id} value={p.product_id}>{p.product_id} — {p.product_name}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Model</label>
            <select value={chosenModel} onChange={(e) => setChartModel(e.target.value)}>
              {results.models.map((m) => <option key={m} value={m}>{modelLabels[m] || m}</option>)}
            </select>
          </div>
        </div>
        <h3>Predicted vs Actual Demand (test period)</h3>
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

      <div className="card">
        <h3>Future Demand Forecast ({horizon} days ahead)</h3>
        {future.length ? (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={future} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={40} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip formatter={(v) => fmt(v, 1)} />
              <Line type="monotone" dataKey="predicted" stroke="#10b981" dot={false} strokeWidth={2} name="Forecast" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="muted">No future forecast available for this model.</p>
        )}
      </div>
    </div>
  )
}

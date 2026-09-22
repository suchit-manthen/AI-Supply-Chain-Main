import { useState } from 'react'
import ModelComparison from './ModelComparison.jsx'
import { runForecast } from '../api.js'

export default function AdvancedAnalytics({ dataset, comparison, setComparison }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [horizon, setHorizon] = useState(30)

  async function run() {
    setBusy(true)
    setError(null)
    try {
      const res = await runForecast({ dataset_id: dataset.dataset_id, models: 'all', horizon })
      setComparison(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h2>Advanced Analytics</h2>
          <p className="muted">
            Model-level comparison for technical users: SARIMA, Prophet, Random Forest, XGBoost and LSTM/GRU.
          </p>
        </div>
      </div>

      <div className="card">
        <h3>Run model comparison</h3>
        <div className="toolbar">
          <div className="field">
            <label>Horizon (days)</label>
            <input type="number" min="1" max="180" value={horizon} onChange={(e) => setHorizon(Number(e.target.value) || 30)} />
          </div>
          <button className="btn primary" onClick={run} disabled={busy}>
            {busy ? 'Running all models…' : 'Compare all algorithms'}
          </button>
        </div>
        {busy && <p className="muted" style={{ marginTop: 10 }}>Training all five algorithms on every SKU. This can take a minute or two.</p>}
        {error && <div className="error banner">{error}</div>}
        <p className="legend-note">
          The manager dashboard does <b>not</b> simply pick the lowest-MAPE model — it matches each product to the most
          suitable approach based on its demand pattern. Use this page to inspect raw accuracy per algorithm.
        </p>
      </div>

      {comparison
        ? <ModelComparison results={comparison} onGoForecast={() => {}} />
        : <p className="muted">Run the comparison above to see accuracy and per-SKU model wins.</p>}
    </div>
  )
}

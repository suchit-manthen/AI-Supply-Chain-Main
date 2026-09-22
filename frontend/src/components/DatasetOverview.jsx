import { useRef, useState } from 'react'
import { uploadFile, loadDemo, fmt } from '../api.js'

export default function DatasetOverview({ dataset, onLoaded, onProceed }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [drag, setDrag] = useState(false)
  const inputRef = useRef(null)

  async function handleFile(file) {
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      const data = await uploadFile(file)
      onLoaded(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleDemo() {
    setBusy(true)
    setError(null)
    try {
      const data = await loadDemo()
      onLoaded(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  if (dataset) {
    const r = dataset.report
    return (
      <div>
        <div className="page-head">
          <div>
            <h2>Dataset Overview</h2>
            <p className="muted">Validated and ready for forecasting.</p>
          </div>
          <button className="btn primary" onClick={onProceed}>Proceed to Forecast →</button>
        </div>

        <div className="kpis">
          <KPI label="Rows" value={fmt(r.n_rows, 0)} />
          <KPI label="Products (SKUs)" value={fmt(r.n_products, 0)} />
          <KPI label="Categories" value={fmt(r.n_categories, 0)} />
          <KPI label="Days" value={fmt(r.n_days, 0)} />
          <KPI label="Date Range" value={`${r.date_min} → ${r.date_max}`} small />
        </div>

        <div className="row">
          <div className="card">
            <h3>Detected Columns</h3>
            <table>
              <thead><tr><th>Field</th><th>Source Column</th></tr></thead>
              <tbody>
                {r.columns.map((c) => (
                  <tr key={c.name}>
                    <td><b>{c.name}</b></td>
                    <td className="muted">{r.mapped[c.name] || '(derived / default)'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="card">
            <h3>Categories</h3>
            {r.categories.length === 0 && <p className="muted">No category column detected.</p>}
            <div className="chips">
              {r.categories.map((c) => (
                <span key={c.name} className="badge">{c.name} · {c.count}</span>
              ))}
            </div>

            {Object.keys(r.defaulted).length > 0 && (
              <>
                <h3 style={{ marginTop: 18 }}>Defaults Applied</h3>
                <ul className="note-list">
                  {Object.entries(r.defaulted).map(([k, v]) => (
                    <li key={k}><b>{k}</b>: {v}</li>
                  ))}
                </ul>
              </>
            )}
          </div>
        </div>

        <div className="card">
          <h3>Data Preview</h3>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>{r.columns.map((c) => <th key={c.name}>{c.name}</th>)}</tr>
              </thead>
              <tbody>
                {r.preview.map((row, i) => (
                  <tr key={i}>
                    {r.columns.map((c) => <td key={c.name}>{row[c] ?? ''}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="upload-wrap">
      <div className="page-head">
        <div>
          <h2>Dataset Overview</h2>
          <p className="muted">Upload a supermarket sales CSV to start.</p>
        </div>
      </div>

      <div
        className={`dropzone ${drag ? 'drag' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); handleFile(e.dataTransfer.files[0]) }}
      >
        <div className="dz-icon">⬆️</div>
        <div className="dz-title">Drop your CSV here or click to browse</div>
        <div className="muted">e.g. supermarket_supply_chain_test_dataset.csv</div>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
        <button className="btn primary" style={{ marginTop: 16 }} disabled={busy}>
          {busy ? 'Uploading…' : 'Choose CSV'}
        </button>
      </div>

      <div className="demo-bar">
        <span className="muted">No file handy?</span>
        <button className="btn ghost" onClick={handleDemo} disabled={busy}>
          {busy ? 'Loading…' : 'Load built-in demo dataset'}
        </button>
      </div>

      <div className="req-hint">
        <h3>Expected columns</h3>
        <p className="muted">Required: <code>date</code>, <code>product_id</code> (or sku), <code>sales</code> (or demand/units).
        Optional: <code>price</code>, <code>is_promo</code>, <code>discount_pct</code>, <code>is_holiday</code>, <code>category</code>, <code>base_price</code>, <code>product_name</code>.
        Missing optional columns are defaulted automatically.</p>
      </div>

      {error && <div className="error banner">{error}</div>}
    </div>
  )
}

function KPI({ label, value, small }) {
  return (
    <div className="card kpi">
      <div className="label">{label}</div>
      <div className="value" style={small ? { fontSize: 16 } : undefined}>{value}</div>
    </div>
  )
}

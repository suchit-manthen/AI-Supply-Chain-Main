import { useRef, useState } from 'react'
import { uploadFile, loadDemo } from '../api.js'

export default function DatasetOverview({ onLoaded }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [drag, setDrag] = useState(false)
  const inputRef = useRef(null)

  async function handleFile(file) {
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      onLoaded(await uploadFile(file))
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
      onLoaded(await loadDemo())
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="upload-wrap">
      <div className="page-head">
        <div>
          <h2>Welcome to the Supply Chain Assistant</h2>
          <p className="muted">Upload your sales history and we'll tell you what to order.</p>
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
        <p className="muted">
          Required: <code>date</code>, <code>product_id</code> (or sku), <code>sales</code> (or demand/units).
          <br />
          Optional: <code>price</code>, <code>is_promo</code>, <code>category</code>, <code>product_name</code>,
          and — for reorder recommendations — <code>on_hand</code> (current stock) and <code>lead_time_days</code>.
        </p>
      </div>

      {error && <div className="error banner">{error}</div>}
    </div>
  )
}

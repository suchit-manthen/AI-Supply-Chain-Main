import { useEffect, useState } from 'react'
import DatasetOverview from './components/DatasetOverview.jsx'
import DemandForecast from './components/DemandForecast.jsx'
import ModelComparison from './components/ModelComparison.jsx'
import InventoryOptimization from './components/InventoryOptimization.jsx'
import SKUAnalysis from './components/SKUAnalysis.jsx'
import { getMeta } from './api.js'

const SECTIONS = [
  { id: 'dataset', label: 'Dataset Overview', icon: '📊' },
  { id: 'forecast', label: 'Demand Forecast', icon: '📈' },
  { id: 'compare', label: 'Model Comparison', icon: '⚖️' },
  { id: 'inventory', label: 'Inventory Optimization', icon: '📦' },
  { id: 'sku', label: 'Product / SKU Analysis', icon: '🏷️' },
]

export default function App() {
  const [meta, setMeta] = useState(null)
  const [section, setSection] = useState('dataset')
  const [dataset, setDataset] = useState(null)   // { dataset_id, report, products }
  const [results, setResults] = useState(null)   // full run results
  const [runRequest, setRunRequest] = useState(null) // the last run config for context

  useEffect(() => {
    getMeta().then(setMeta).catch(() => setMeta({ models: [], model_labels: {}, model_descriptions: {}, defaults: {} }))
  }, [])

  const datasetReady = !!dataset

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="side-brand">
          <span className="brand-mark">🛒</span>
          <div>
            <div className="side-title">AI Supply Chain</div>
            <div className="side-sub">Demand Forecasting</div>
          </div>
        </div>
        <nav className="side-nav">
          {SECTIONS.map((s) => {
            const disabled = s.id !== 'dataset' && !datasetReady
            return (
              <button
                key={s.id}
                className={`side-item ${section === s.id ? 'active' : ''} ${disabled ? 'disabled' : ''}`}
                onClick={() => !disabled && setSection(s.id)}
              >
                <span className="side-icon">{s.icon}</span>
                <span>{s.label}</span>
                {s.id !== 'dataset' && !datasetReady && <span className="lock">🔒</span>}
              </button>
            )
          })}
        </nav>
        <div className="side-foot">
          {datasetReady ? (
            <div className="ds-chip" title={dataset.report ? `${dataset.report.n_rows} rows` : ''}>
              <span className="dot" /> {dataset.report?.n_products ?? '—'} SKUs loaded
            </div>
          ) : (
            <div className="ds-chip muted">No dataset loaded</div>
          )}
        </div>
      </aside>

      <div className="main">
        <main className="content">
          {section === 'dataset' && (
            <DatasetOverview dataset={dataset} onLoaded={setDataset} onProceed={() => setSection('forecast')} />
          )}
          {section === 'forecast' && (
            <DemandForecast
              dataset={dataset}
              meta={meta}
              results={results}
              onResults={setResults}
              onRequest={setRunRequest}
            />
          )}
          {section === 'compare' && (
            <ModelComparison results={results} request={runRequest} onGoForecast={() => setSection('forecast')} />
          )}
          {section === 'inventory' && (
            <InventoryOptimization results={results} onGoForecast={() => setSection('forecast')} />
          )}
          {section === 'sku' && (
            <SKUAnalysis results={results} onGoForecast={() => setSection('forecast')} />
          )}
        </main>
      </div>
    </div>
  )
}

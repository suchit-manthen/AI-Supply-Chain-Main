import { useEffect, useState } from 'react'
import DatasetOverview from './components/DatasetOverview.jsx'
import Overview from './components/Overview.jsx'
import DemandForecast from './components/DemandForecast.jsx'
import InventoryActions from './components/InventoryActions.jsx'
import SKUAnalysis from './components/SKUAnalysis.jsx'
import AdvancedAnalytics from './components/AdvancedAnalytics.jsx'
import { getMeta, recommend } from './api.js'

const SECTIONS = [
  { id: 'overview', label: 'Overview', icon: '🏠' },
  { id: 'forecast', label: 'Demand Forecast', icon: '📈' },
  { id: 'inventory', label: 'Inventory Actions', icon: '📦' },
  { id: 'sku', label: 'Product Analysis', icon: '🏷️' },
  { id: 'advanced', label: 'Advanced Analytics', icon: '⚙️' },
]

const DEFAULT_PARAMS = { horizon: 30, service_level: 0.95, lead_time: 3, ordering_cost: 50, holding_rate: 0.2 }

export default function App() {
  const [meta, setMeta] = useState(null)
  const [section, setSection] = useState('overview')
  const [dataset, setDataset] = useState(null)
  const [recommendation, setRecommendation] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [params, setParams] = useState(DEFAULT_PARAMS)
  const [recBusy, setRecBusy] = useState(false)
  const [recError, setRecError] = useState(null)

  useEffect(() => {
    getMeta().then(setMeta).catch(() => setMeta({ models: [], model_labels: {}, model_descriptions: {}, defaults: {} }))
  }, [])

  async function runRecommend(overrides = {}) {
    const p = { ...params, ...overrides }
    setParams(p)
    setRecBusy(true)
    setRecError(null)
    try {
      const res = await recommend({ dataset_id: dataset.dataset_id, ...p })
      setRecommendation(res)
      setSection('overview')
    } catch (e) {
      setRecError(e.message)
    } finally {
      setRecBusy(false)
    }
  }

  if (!dataset) {
    return (
      <div className="app">
        <div className="main">
          <main className="content">
            <DatasetOverview
              onLoaded={(d) => { setDataset(d); setRecommendation(null); setComparison(null) }}
            />
          </main>
        </div>
      </div>
    )
  }

  const shared = {
    dataset,
    meta,
    recommendation,
    setRecommendation,
    comparison,
    setComparison,
    params,
    setParams,
    recBusy,
    recError,
    runRecommend,
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="side-brand">
          <span className="brand-mark">🛒</span>
          <div>
            <div className="side-title">AI Supply Chain</div>
            <div className="side-sub">Demand &amp; Inventory</div>
          </div>
        </div>
        <nav className="side-nav">
          {SECTIONS.map((s) => (
            <button
              key={s.id}
              className={`side-item ${section === s.id ? 'active' : ''}`}
              onClick={() => setSection(s.id)}
            >
              <span className="side-icon">{s.icon}</span>
              <span>{s.label}</span>
              {s.id === 'advanced' && <span className="side-tag">technical</span>}
            </button>
          ))}
        </nav>
        <div className="side-foot">
          <div className="ds-chip" title={`${dataset.report?.n_rows ?? ''} rows`}>
            <span className="dot" /> {dataset.report?.n_products ?? '—'} SKUs loaded
          </div>
        </div>
      </aside>

      <div className="main">
        <main className="content">
          {section === 'overview' && <Overview {...shared} />}
          {section === 'forecast' && <DemandForecast {...shared} />}
          {section === 'inventory' && <InventoryActions {...shared} />}
          {section === 'sku' && <SKUAnalysis {...shared} />}
          {section === 'advanced' && <AdvancedAnalytics {...shared} />}
        </main>
      </div>
    </div>
  )
}

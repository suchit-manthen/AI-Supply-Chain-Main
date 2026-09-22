import { fmt, STATUS_META } from '../api.js'

export default function Overview({ dataset, recommendation, params, setParams, recBusy, recError, runRecommend }) {
  const report = dataset?.report

  if (!recommendation) {
    return (
      <div>
        <div className="page-head">
          <div>
            <h2>Overview</h2>
            <p className="muted">We'll analyse demand, forecast and tell you what to order.</p>
          </div>
        </div>

        <div className="card">
          <h3>Generate Recommendations</h3>
          <div className="toolbar">
            <div className="field">
              <label>Forecast Horizon (days)</label>
              <input type="number" min="1" max="180" value={params.horizon}
                onChange={(e) => setParams({ ...params, horizon: Number(e.target.value) || 30 })} />
            </div>
            <div className="field">
              <label>Service Level</label>
              <select value={params.service_level} onChange={(e) => setParams({ ...params, service_level: Number(e.target.value) })}>
                <option value={0.9}>90%</option>
                <option value={0.95}>95%</option>
                <option value={0.98}>98%</option>
                <option value={0.99}>99%</option>
              </select>
            </div>
            <button className="btn primary" onClick={() => runRecommend()} disabled={recBusy}>
              {recBusy ? 'Analysing…' : 'Generate Recommendations'}
            </button>
          </div>
          {recBusy && <p className="muted" style={{ marginTop: 10 }}>Analysing demand patterns and training the forecast models. This can take a minute.</p>}
          {recError && <div className="error banner">{recError}</div>}
          <p className="legend-note">
            No technical setup needed — we automatically pick the right forecasting approach for each product.
          </p>
        </div>

        {report && <div className="card" style={{ marginTop: 16 }}><DataSummary report={report} /></div>}
      </div>
    )
  }

  const inv = recommendation.inventory || []
  const horizon = recommendation.horizon || 30
  const expectedNext = Math.round(inv.reduce((a, b) => a + (b.avg_daily_demand || 0), 0) * horizon)

  const groups = {
    urgent: inv.filter((i) => i.status === 'stockout'),
    reorder: inv.filter((i) => i.status === 'reorder'),
    overstock: inv.filter((i) => i.status === 'overstock'),
    healthy: inv.filter((i) => i.status === 'ok'),
  }

  return (
    <div className="stack">
      <div className="page-head">
        <div>
          <h2>What needs my attention?</h2>
          <p className="muted">Here's what to act on, and what's fine.</p>
        </div>
        <button className="btn ghost" onClick={() => runRecommend()} disabled={recBusy}>
          {recBusy ? 'Analysing…' : '↻ Re-run'}
        </button>
      </div>

      {recError && <div className="error banner">{recError}</div>}

      <div className="kpis">
        <KPI label="🔴 Order now" value={fmt(groups.urgent.length, 0)} hint="may run out during lead time" accent="red" />
        <KPI label="🟠 Reorder" value={fmt(groups.reorder.length, 0)} hint="below reorder point" accent="amber" />
        <KPI label="🟡 Overstocked" value={fmt(groups.overstock.length, 0)} hint="too much on hand" accent="amber" />
        <KPI label="🟢 Healthy" value={fmt(groups.healthy.length, 0)} hint="no action needed" accent="green" />
        <KPI label="Expected demand" value={fmt(expectedNext, 0)} hint={`units over next ${horizon} days`} />
      </div>

      <ActionGroup
        emoji="🔴"
        title="Act now — stock may run out"
        items={groups.urgent}
        empty="No urgent stockout risks. 🎉"
      />

      <ActionGroup
        emoji="🟠"
        title="Reorder soon"
        items={groups.reorder}
        empty="Nothing below the reorder point."
      />

      <ActionGroup
        emoji="🟡"
        title="Overstocked — consider reducing future orders"
        items={groups.overstock}
        empty="No overstocked products."
      />

      <div className="card explainer">
        <h3>How this works</h3>
        <p className="muted">
          For each product we detect its demand pattern, forecast the days ahead, then compare that with your current
          stock and supplier lead time. The result is a simple instruction — order now, reorder soon, or do nothing —
          plus the exact quantity to order.
        </p>
      </div>

      {report && <div className="card"><DataSummary report={report} /></div>}
    </div>
  )
}

function ActionGroup({ emoji, title, items, empty }) {
  if (items.length === 0) {
    return (
      <div className="card attention-group">
        <h3>{emoji} {title}</h3>
        <p className="muted">{empty}</p>
      </div>
    )
  }
  return (
    <div className="card attention-group">
      <h3>{emoji} {title} <span className="count">{items.length}</span></h3>
      <div className="table-scroll">
        <table>
          <thead>
            <tr><th>Product</th><th>Status</th><th>Current stock</th><th>Expected demand</th><th>Action</th><th>Reason</th></tr>
          </thead>
          <tbody>
            {items.map((r) => {
              const meta = STATUS_META[r.status] || {}
              return (
                <tr key={r.product_id}>
                  <td><b>{r.product_name}</b> <span className="muted">{r.product_id}</span></td>
                  <td><StatusBadge status={r.status} /></td>
                  <td>{fmt(r.current_stock, 0)}</td>
                  <td>{fmt(r.expected_demand_lead_time, 0)}</td>
                  <td>
                    {r.recommended_order
                      ? <span className="action">{meta.emoji} Order {fmt(r.recommended_order, 0)} units</span>
                      : <span className="muted">—</span>}
                  </td>
                  <td className="reason-cell">{r.reason}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function StatusBadge({ status }) {
  const meta = STATUS_META[status] || {}
  return <span className={`badge ${meta.tone}`}>{meta.emoji} {meta.label}</span>
}

function DataSummary({ report }) {
  return (
    <div>
      <h3>Loaded dataset</h3>
      <div className="chips">
        <span className="badge">{fmt(report.n_rows, 0)} rows</span>
        <span className="badge">{fmt(report.n_products, 0)} products</span>
        <span className="badge">{fmt(report.n_categories, 0)} categories</span>
        <span className="badge">{fmt(report.n_days, 0)} days</span>
        <span className="badge">{report.date_min} → {report.date_max}</span>
      </div>
    </div>
  )
}

function KPI({ label, value, hint, accent }) {
  return (
    <div className="card kpi">
      <div className="label">{label}</div>
      <div className="value" style={accent ? { color: `var(--${accent})` } : undefined}>{value}</div>
      <div className="hint">{hint}</div>
    </div>
  )
}

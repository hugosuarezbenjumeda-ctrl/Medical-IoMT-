import { useEffect, useMemo, useState } from 'react'

const FEATURE_META = {
  ack_count: { label: 'ACK Packet Count', description: 'Acknowledgement packets seen in this flow window.' },
  avg: { label: 'Average Value', description: 'Average packet statistic for the current flow window.' },
  covariance: { label: 'Covariance', description: 'Joint variation between paired packet statistics.' },
  cwr_flag_number: { label: 'CWR Flag Count', description: 'Congestion Window Reduced TCP flags observed.' },
  drate: { label: 'Destination Packet Rate', description: 'Estimated packets per second received by destination.' },
  ece_flag_number: { label: 'ECE Flag Count', description: 'ECN-Echo TCP flags observed.' },
  fin_count: { label: 'FIN Packet Count', description: 'Connection-closing FIN packets observed.' },
  fin_flag_number: { label: 'FIN Flag Number', description: 'Feature-engineered FIN flag count indicator.' },
  header_length: { label: 'Header Length', description: 'Packet header size information in the flow window.' },
  iat: { label: 'Inter-Arrival Time (IAT)', description: 'Time gap between packets.' },
  magnitue: { label: 'Magnitude', description: 'Composite traffic magnitude feature from the dataset.' },
  max: { label: 'Maximum Value', description: 'Maximum packet statistic inside this flow window.' },
  min: { label: 'Minimum Value', description: 'Minimum packet statistic inside this flow window.' },
  number: { label: 'Packet Count', description: 'Number of packets observed in this flow window.' },
  psh_flag_number: { label: 'PSH Flag Count', description: 'TCP PUSH flags observed.' },
  radius: { label: 'Radius', description: 'Spread feature derived from the traffic distribution.' },
  rate: { label: 'Packet Rate', description: 'Estimated packets per second in the flow window.' },
  rst_count: { label: 'RST Packet Count', description: 'Connection reset packets observed.' },
  rst_flag_number: { label: 'RST Flag Number', description: 'Feature-engineered RST flag count indicator.' },
  srate: { label: 'Source Packet Rate', description: 'Estimated packets per second sent by source.' },
  std: { label: 'Standard Deviation', description: 'Variation level of packet statistics in this flow window.' },
  syn_count: { label: 'SYN Packet Count', description: 'Connection-open SYN packets observed.' },
  syn_flag_number: { label: 'SYN Flag Number', description: 'Feature-engineered SYN flag count indicator.' },
  tot_size: { label: 'Total Bytes', description: 'Total packet bytes transferred in this flow window.' },
  tot_sum: { label: 'Total Sum', description: 'Aggregate flow-level sum feature from the dataset.' },
  variance: { label: 'Variance', description: 'Variance of packet statistics in the flow window.' },
  weight: { label: 'Weight', description: 'Composite weighted traffic feature from the dataset.' },
}

function safeNumber(value, fallback = 0) {
  const n = Number(value)
  return Number.isFinite(n) ? n : fallback
}

function formatSimClock(seconds) {
  const total = Math.max(0, Math.floor(seconds))
  const h = String(Math.floor(total / 3600)).padStart(2, '0')
  const m = String(Math.floor((total % 3600) / 60)).padStart(2, '0')
  const s = String(total % 60).padStart(2, '0')
  return `${h}:${m}:${s}`
}

function normalizeFeatureKey(feature) {
  return String(feature || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
}

function fallbackFeatureLabel(feature) {
  return String(feature || '')
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

function getFeatureMeta(feature) {
  const key = normalizeFeatureKey(feature)
  const meta = FEATURE_META[key]
  if (meta) return meta
  return {
    label: fallbackFeatureLabel(feature),
    description: 'Flow-level feature used by the model.',
  }
}

function directionText(direction) {
  if (direction === 'attack') return 'pushes this decision toward attack'
  if (direction === 'benign') return 'pulls this decision toward benign'
  return 'influences this decision'
}

function buildAlertNarrative(alert) {
  if (!alert) return ''
  const top = (alert.local_explanation || []).slice(0, 3).map((f) => getFeatureMeta(f.feature).label)
  const score = safeNumber(alert.score_attack, 0)
  const threshold = safeNumber(alert.threshold, 0.5)
  const family = String(alert.attack_family || '').trim()

  let text = `Flagged because attack score ${score.toFixed(4)} is above the ${String(alert.protocol || '').toUpperCase()} threshold ${threshold.toFixed(4)}.`
  if (top.length > 0) text += ` Main drivers: ${top.join(', ')}.`
  if (family && family.toLowerCase() !== 'n/a') text += ` Predicted family context: ${family}.`
  return text
}

async function apiGet(path) {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

async function apiPost(path) {
  const res = await fetch(path, { method: 'POST' })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

function App() {
  const [meta, setMeta] = useState(null)
  const [globalExplanations, setGlobalExplanations] = useState(null)
  const [state, setState] = useState(null)
  const [selectedAlertId, setSelectedAlertId] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    apiGet('/api/init')
      .then((payload) => {
        if (!active) return
        setMeta(payload.meta || {})
        setGlobalExplanations(payload.global_explanations || {})
        setState(payload.state || {})
      })
      .catch((e) => {
        if (!active) return
        setError(`Failed to initialize realtime API: ${String(e?.message || e)}`)
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (!state) return undefined
    let active = true
    const timer = setInterval(() => {
      apiGet('/api/state')
        .then((payload) => {
          if (!active) return
          setState(payload.state || {})
        })
        .catch((e) => {
          if (!active) return
          setError(`Realtime API connection failed: ${String(e?.message || e)}`)
        })
    }, 1000)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [state])

  useEffect(() => {
    const alerts = state?.recent_alerts || []
    if (!selectedAlertId && alerts.length > 0) {
      setSelectedAlertId(alerts[0].id)
      return
    }
    if (selectedAlertId && alerts.length > 0 && !alerts.some((a) => a.id === selectedAlertId)) {
      setSelectedAlertId(alerts[0].id)
    }
  }, [selectedAlertId, state])

  const selectedAlert = useMemo(() => {
    const alerts = state?.recent_alerts || []
    return alerts.find((a) => a.id === selectedAlertId) || null
  }, [selectedAlertId, state])

  const toggleRunning = async () => {
    try {
      const payload = state?.running ? await apiPost('/api/pause') : await apiPost('/api/start')
      setState(payload.state || {})
      setError('')
    } catch (e) {
      setError(`Failed to change run state: ${String(e?.message || e)}`)
    }
  }

  const reset = async () => {
    try {
      const payload = await apiPost('/api/reset')
      setState(payload.state || {})
      setSelectedAlertId('')
      setError('')
    } catch (e) {
      setError(`Failed to reset: ${String(e?.message || e)}`)
    }
  }

  if (error && !state) {
    return (
      <div className="app-shell">
        <h1>MIoT IDS Prototype</h1>
        <p className="error">{error}</p>
      </div>
    )
  }

  if (!meta || !state || !globalExplanations) {
    return (
      <div className="app-shell">
        <h1>MIoT IDS Prototype</h1>
        <p>Connecting to realtime IDS API...</p>
      </div>
    )
  }

  const flowsProcessed = safeNumber(state.flows_processed, 0)
  const totalRows = safeNumber(meta.total_rows, 0)
  const alertsDetected = safeNumber(state.alerts_detected, 0)
  const alertsSurfaced = safeNumber(state.alerts_surfaced, 0)
  const progress = totalRows > 0 ? Math.min(1, flowsProcessed / totalRows) : 0
  const alertRate = flowsProcessed > 0 ? alertsDetected / flowsProcessed : 0
  const surfacedRate = alertsDetected > 0 ? alertsSurfaced / alertsDetected : 0

  const currentMix = Object.entries(state.protocol_flow_counts || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([proto, count]) => `${proto}: ${count}`)
    .join(' | ')

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <h1>MIoT IDS Prototype</h1>
          <p className="subtitle">
            Live XGBoost inference from <code>metadata_test.csv</code> via local realtime API.
          </p>
        </div>
        <div className="controls">
          <button className="primary" onClick={toggleRunning} disabled={Boolean(state.ended)}>
            {state.running ? 'Pause' : 'Start Simulation'}
          </button>
          <button className="ghost" onClick={reset}>Reset</button>
        </div>
      </header>

      {error ? <p className="error">{error}</p> : null}

      <section className="stats-grid">
        <div className="card stat">
          <div className="label">Sim Clock</div>
          <div className="value">{formatSimClock(safeNumber(state.sim_seconds, 0))}</div>
        </div>
        <div className="card stat">
          <div className="label">Flows Processed</div>
          <div className="value">{Math.floor(flowsProcessed).toLocaleString()}</div>
        </div>
        <div className="card stat">
          <div className="label">Alerts Detected</div>
          <div className="value">{Math.floor(alertsDetected).toLocaleString()}</div>
        </div>
        <div className="card stat">
          <div className="label">Alerts Surfaced</div>
          <div className="value">{Math.floor(alertsSurfaced).toLocaleString()}</div>
        </div>
        <div className="card stat">
          <div className="label">Flow Rate</div>
          <div className="value">{safeNumber(meta.rows_per_second, 1).toFixed(1)} flows/s</div>
        </div>
      </section>

      <section className="progress-wrap">
        <div className="progress-label">
          Runtime: {state.running ? 'running' : state.ended ? 'ended' : 'paused'} | Replay: {String(meta.replay_order || 'n/a')} | Current protocol mix: {currentMix || 'n/a'}
        </div>
        {state.first_alert_sim_second != null ? (
          <div className="progress-label">
            First detected alert at {formatSimClock(safeNumber(state.first_alert_sim_second, 0))} (row {safeNumber(state.first_alert_row, 0)}).
          </div>
        ) : null}
        <div className="progress-label">
          Row {Math.floor(flowsProcessed).toLocaleString()} / {Math.floor(totalRows).toLocaleString()} | Alert ratio: {alertRate.toFixed(4)} | Surfaced share: {(surfacedRate * 100).toFixed(2)}%
        </div>
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress * 100}%` }} />
        </div>
      </section>

      <section className="main-grid">
        <div className="card panel">
          <h2>Sequential Alerts</h2>
          {(state.recent_alerts || []).length === 0 ? (
            <p className="muted">No alerts have arrived yet.</p>
          ) : (
            <ul className="alert-list">
              {(state.recent_alerts || []).map((a) => (
                <li key={a.id}>
                  <button
                    className={`alert-item ${selectedAlertId === a.id ? 'active' : ''}`}
                    onClick={() => setSelectedAlertId(a.id)}
                  >
                    <span className="badge">{a.protocol}</span>
                    <span className="alert-main">score={safeNumber(a.score_attack).toFixed(4)} thr={safeNumber(a.threshold).toFixed(4)}</span>
                    <span className="alert-meta">{formatSimClock(safeNumber(a.sim_second, 0))}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="card panel">
          <h2>Local Explanation</h2>
          {!selectedAlert ? (
            <p className="muted">Select an alert to inspect local feature contributions.</p>
          ) : (
            <>
              <div className="alert-detail-head">
                <span className="badge">{selectedAlert.protocol}</span>
                <span>attack_family: {selectedAlert.attack_family || 'n/a'}</span>
                <span>row: {selectedAlert.global_row_index}</span>
              </div>
              <p className="narrative">{buildAlertNarrative(selectedAlert)}</p>
              <ul className="feature-list">
                {(selectedAlert.local_explanation || []).map((f) => {
                  const metaForFeature = getFeatureMeta(f.feature)
                  return (
                    <li key={`${selectedAlert.id}-${f.feature}`}>
                      <div className="feature-row">
                        <span className="feature-name">{metaForFeature.label}</span>
                        <span className="feature-val">{safeNumber(f.contribution).toFixed(4)}</span>
                      </div>
                      <div className="feature-meta">{f.feature} | {metaForFeature.description}</div>
                      <div className="feature-meta">{directionText(String(f.direction || ''))}</div>
                      <div className="bar-wrap">
                        <div className="bar" style={{ width: `${Math.min(100, safeNumber(f.abs_contribution) * 18)}%` }} />
                      </div>
                    </li>
                  )
                })}
              </ul>
            </>
          )}
        </div>
      </section>

      <section className="card panel">
        <h2>Global Explanations</h2>
        <div className="global-grid">
          <div>
            <h3>Overall Top Features</h3>
            <ul className="simple-list">
              {(globalExplanations?.overall_top_features || []).slice(0, 12).map((r) => {
                const m = getFeatureMeta(r.feature)
                return (
                  <li key={r.feature}>
                    <span>{m.label} <span className="feature-tech">({r.feature})</span></span>
                    <span>{safeNumber(r.score).toFixed(1)}</span>
                  </li>
                )
              })}
            </ul>
          </div>
          <div>
            <h3>Per-Protocol Drivers</h3>
            <div className="protocol-columns">
              {(globalExplanations?.protocols || []).map((p) => (
                <div key={p.protocol} className="protocol-card">
                  <div className="protocol-head">
                    <span className="badge">{p.protocol}</span>
                    <span>thr={safeNumber(p.threshold).toFixed(4)}</span>
                  </div>
                  <ul className="simple-list">
                    {(p.top_features || []).slice(0, 8).map((f) => {
                      const m = getFeatureMeta(f.feature)
                      return (
                        <li key={`${p.protocol}-${f.feature}`}>
                          <span>{m.label} <span className="feature-tech">({f.feature})</span></span>
                          <span>{safeNumber(f.mean_abs_contribution).toFixed(3)}</span>
                        </li>
                      )
                    })}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default App

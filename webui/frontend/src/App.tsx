import { useCallback } from 'react'
import { Link, NavLink, Route, Routes, useParams } from 'react-router-dom'
import { api } from './api'
import { useAsync, useLive } from './hooks'
import type { LiveEvent, RunSummary } from './types'
import { KindTag, ScoreChip, StatusBadge, shortId } from './ui'
import RunDetail from './components/RunDetail'
import SetupPage from './components/SetupPage'
import LaunchPage from './components/LaunchPage'

const navClass = ({ isActive }: { isActive: boolean }) => `side-link ${isActive ? 'active' : ''}`

function Sidebar({ runs, connected }: { runs: RunSummary[]; connected: boolean }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark" />
        <div>
          <h1>Mimosa Observatory</h1>
          <small>evolving workflows · live view</small>
        </div>
      </div>
      <div className="side-nav">
        <NavLink to="/launch" className={navClass}>✦ New run</NavLink>
        <NavLink to="/setup" className={navClass}>Setup</NavLink>
      </div>
      <div className="sidebar-head">
        <span>Runs · {runs.length}</span>
        <span className={`live-dot ${connected ? 'on' : ''}`}>
          <i />{connected ? 'live' : 'offline'}
        </span>
      </div>
      <nav className="run-list">
        {runs.map((r) => (
          <NavLink
            key={r.id}
            to={`/runs/${r.id}`}
            className={({ isActive }) => `run-item ${isActive ? 'active' : ''}`}
          >
            <div className="goal">{r.goal || <span className="muted">no goal recorded</span>}</div>
            <div className="meta">
              <StatusBadge status={r.status} />
              {r.score != null && <ScoreChip score={r.score} />}
              <KindTag kind={r.evolution_kind} />
              <span className="rid">{shortId(r.id)}</span>
            </div>
          </NavLink>
        ))}
        {runs.length === 0 && <div className="hint" style={{ padding: 12 }}>No runs found.</div>}
      </nav>
    </aside>
  )
}

export default function App() {
  const { data: runs, refetch } = useAsync(() => api.runs(), [])

  const onEvent = useCallback((e: LiveEvent) => {
    // Any run-level lifecycle event may change the run list / statuses.
    if (['run_finished', 'execution_complete', 'iteration_complete', 'workflow_crafted'].includes(e.type)) {
      refetch()
    }
  }, [refetch])

  const { connected } = useLive(onEvent)

  return (
    <div className="app">
      <Sidebar runs={runs || []} connected={connected} />
      <main className="main">
        <Routes>
          <Route index element={<Welcome count={runs?.length ?? 0} />} />
          <Route path="/runs/:id" element={<RunRoute />} />
          <Route path="/launch" element={<LaunchPage />} />
          <Route path="/setup" element={<SetupPage />} />
        </Routes>
      </main>
    </div>
  )
}

function RunRoute() {
  const { id } = useParams()
  return id ? <RunDetail runId={id} /> : null
}

function Welcome({ count }: { count: number }) {
  return (
    <div className="empty">
      <div className="brand-mark" style={{ width: 46, height: 46 }} />
      <div style={{ textAlign: 'center' }}>
        <h2 style={{ color: 'var(--text)', marginBottom: 6 }}>Mimosa Observatory</h2>
        <div className="hint">
          {count} evolution run{count === 1 ? '' : 's'} on disk — pick one to inspect its
          lineage, workflow, agent replay and workspace.
        </div>
        <Link to="/launch" className="side-link" style={{ marginTop: 14, padding: '8px 18px' }}>
          ✦ Start a new run
        </Link>
      </div>
    </div>
  )
}

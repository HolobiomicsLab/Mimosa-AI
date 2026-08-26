import { Component, useCallback, useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Link, NavLink, Route, Routes, useLocation, useNavigate, useParams } from 'react-router-dom'
import { api } from './api'
import { useAsync, useLive } from './hooks'
import type { LaunchInfo, LiveEvent, RunSummary } from './types'
import { KindTag, ScoreChip, StatusBadge, shortId } from './ui'
import { ToastHost, toast } from './components/Toasts'
import RunDetail from './components/RunDetail'
import SetupPage from './components/SetupPage'
import LaunchPage from './components/LaunchPage'
import AtlasView from './components/AtlasView'

const navClass = ({ isActive }: { isActive: boolean }) => `side-link ${isActive ? 'active' : ''}`

const GEN_FAIL_TEXT =
  'The LLM failed to generate a valid workflow. Certain models — especially smaller ones — '
  + 'often cannot produce one; consider a stronger orchestration model in Setup.'

function notifyFailure(launch: LaunchInfo, navigate: (to: string) => void) {
  const isGeneration = launch.failure_hint === 'workflow_generation'
  toast({
    kind: 'error',
    text: isGeneration
      ? GEN_FAIL_TEXT
      : `Run failed${launch.error_line ? ` — ${launch.error_line}` : ''}. Check the launch log.`,
    actions: [
      { label: 'View log', onClick: () => navigate(`/launch?focus=${launch.id}`) },
      ...(isGeneration ? [{ label: 'Open Setup', onClick: () => navigate('/setup') }] : []),
    ],
  })
}

/** Poll launches app-wide; toast once per launch on failure; allow stopping. */
function useLaunchWatcher() {
  const [launches, setLaunches] = useState<LaunchInfo[]>([])
  const notified = useRef(new Set<string>())
  const navigate = useNavigate()

  useEffect(() => {
    let alive = true
    let timer: ReturnType<typeof setTimeout>
    const tick = () => {
      api.launches()
        .then((ls) => {
          if (!alive) return
          setLaunches(ls)
          ls.forEach((l) => {
            // Toast on a hard failure, or a live run whose crafting LLM is
            // failing to produce a workflow — but not on an ultimately-clean run.
            const flag = l.failed || (l.running && l.failure_hint === 'workflow_generation')
            if (flag && !notified.current.has(l.id)) {
              notified.current.add(l.id)
              notifyFailure(l, navigate)
            }
          })
          timer = setTimeout(tick, ls.some((l) => l.running) ? 3000 : 12000)
        })
        .catch(() => { timer = setTimeout(tick, 12000) })
    }
    tick()
    return () => { alive = false; clearTimeout(timer) }
  }, [navigate])

  const stop = (id: string) => {
    if (!window.confirm('Stop this run?')) return
    api.cancelLaunch(id)
      .then((st) => setLaunches((cur) => cur.map((l) => (l.id === id ? { ...l, ...st } : l))))
      .catch((e: Error) => toast({ kind: 'error', text: e.message }))
  }

  return { launches, stop }
}

function Sidebar({ runs, connected, launches, onStop }: {
  runs: RunSummary[]
  connected: boolean
  launches: LaunchInfo[]
  onStop: (id: string) => void
}) {
  const running = launches.filter((l) => l.running)
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
        <NavLink to="/atlas" className={navClass}>QD Atlas</NavLink>
        <NavLink to="/setup" className={navClass}>Setup</NavLink>
      </div>
      {running.length > 0 && (
        <div className="launch-strip">
          {running.map((l) => (
            <div key={l.id} className="launch-chip">
              <Link to={`/launch?focus=${l.id}`} className="launch-chip-label" title={l.objective}>
                <i className="dot run-pulse" />
                <span>{l.mode}{l.learn ? ' · learn' : ''} · {l.objective}</span>
              </Link>
              <button className="stop-btn" title="Stop / interrupt this run" onClick={() => onStop(l.id)}>
                ■
              </button>
            </div>
          ))}
        </div>
      )}
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

/** Last-resort boundary: a crashing view renders its error with a reason
 * instead of React unmounting the whole app to a blank page. Remounted per
 * route (keyed on pathname) so navigating away clears the error. */
class ViewErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) {
    return { error }
  }
  render() {
    if (this.state.error) {
      return (
        <div className="empty">
          <div className="hint">
            This view crashed — {String(this.state.error.message || this.state.error)}.
            Pick another run or edit the URL; the rest of the app is unaffected.
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

export default function App() {
  const { data: runs, refetch } = useAsync(() => api.runs(), [])
  const { launches, stop } = useLaunchWatcher()

  const onEvent = useCallback((e: LiveEvent) => {
    // Any run-level lifecycle event may change the run list / statuses.
    if (['run_finished', 'execution_complete', 'iteration_complete', 'workflow_crafted'].includes(e.type)) {
      refetch()
    }
  }, [refetch])

  const { connected } = useLive(onEvent)
  const { pathname } = useLocation()

  return (
    <div className="app">
      <ToastHost />
      <Sidebar runs={runs || []} connected={connected} launches={launches} onStop={stop} />
      <main className="main">
        <ViewErrorBoundary key={pathname}>
          <Routes>
            <Route index element={<Welcome count={runs?.length ?? 0} />} />
            <Route path="/runs/:id" element={<RunRoute />} />
            <Route path="/atlas" element={<AtlasView />} />
            <Route path="/launch" element={<LaunchPage />} />
            <Route path="/setup" element={<SetupPage />} />
          </Routes>
        </ViewErrorBoundary>
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

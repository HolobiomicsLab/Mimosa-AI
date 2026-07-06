import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, artifactUrl } from '../api'
import { useAsync } from '../hooks'
import type { Artifact, RunDetail as RunDetailT } from '../types'
import {
  KindTag, ScoreChip, Spinner, StatusBadge, fmtCost, fmtDuration,
} from '../ui'
import LineageTree from './LineageTree'
import RewardChart from './RewardChart'
import MemoryReplay from './MemoryReplay'
import WorkspacePanel from './WorkspacePanel'

// Scientist-first ordering: outputs → what the agents did → the workflow. The
// neuroevolution machinery lives under "Evolution", shown only for learning runs.
type Tab = 'results' | 'replay' | 'workflow' | 'evolution' | 'artifacts'

export default function RunDetail({ runId }: { runId: string }) {
  const { data: run, loading, error } = useAsync<RunDetailT>(() => api.run(runId), [runId])
  const [tab, setTab] = useState<Tab>('results')

  useEffect(() => setTab('results'), [runId])

  if (loading) return <Spinner label="Loading run…" />
  if (error || !run) return <div className="empty"><div className="hint">Run not found.</div></div>

  const tabs: [Tab, string, number?][] = [
    ['results', 'Results'],
    ['replay', 'Replay'],
    ['workflow', 'Workflow'],
    ...(run.learning_mode ? [['evolution', 'Evolution'] as [Tab, string]] : []),
    ['artifacts', 'Artifacts', run.artifacts.length],
  ]

  return (
    <>
      <div className="detail-head">
        <div className="rid">{run.id}</div>
        <h2>{run.goal || 'Untitled run'}</h2>
        <div className="stats">
          <div className="stat"><b><StatusBadge status={run.status} /></b><span>status</span></div>
          <div className="stat"><b><ScoreChip score={run.score} /></b><span>score</span></div>
          <div className="stat"><b>{fmtCost(run.cost_usd)}</b><span>cost</span></div>
          <div className="stat"><b>{fmtDuration(run.wall_time_s)}</b><span>wall time</span></div>
          {run.learning_mode && <div className="stat"><b><KindTag kind={run.evolution_kind} /></b><span>evolution</span></div>}
          {run.is_single_agent && <div className="stat"><b>single</b><span>agent mode</span></div>}
        </div>
      </div>

      <div className="tabs">
        {tabs.map(([id, label, count]) => (
          <button key={id} className={`tab ${tab === id ? 'active' : ''}`} onClick={() => setTab(id)}>
            {label}{count != null && <span className="count">{count}</span>}
          </button>
        ))}
      </div>

      <div className="tab-body">
        {tab === 'results' && <Results run={run} />}
        {tab === 'replay' && <MemoryReplay runId={runId} />}
        {tab === 'workflow' && <Workflow run={run} />}
        {tab === 'evolution' && <Evolution run={run} />}
        {tab === 'artifacts' && <Artifacts run={run} />}
      </div>
    </>
  )
}

/** Landing view: did it succeed, and what did it produce. */
function Results({ run }: { run: RunDetailT }) {
  const ev = run.evaluation_scores
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {ev && <EvalCard ev={ev} />}
      <div>
        <div className="card-head" style={{ padding: '0 0 10px', border: 'none' }}>
          output files
        </div>
        <WorkspacePanel runId={run.id} />
      </div>
    </div>
  )
}

function EvalCard({ ev }: { ev: Record<string, unknown> }) {
  const num = (k: string) => (typeof ev[k] === 'number' ? (ev[k] as number) : undefined)
  const chips: [string, unknown, string][] = [
    ['claims', ev.n_claims, 'var(--text)'],
    ['pass', ev.n_pass, 'var(--ok)'],
    ['fail', ev.n_fail, 'var(--bad)'],
    ['error', ev.n_error, 'var(--warn)'],
    ['unsure', ev.n_unsure, 'var(--text-dim)'],
  ]
  const pass = num('n_pass') ?? 0
  const claims = num('n_claims') ?? 0
  return (
    <div className="card">
      <div className="card-head">
        <span>verification · {String(ev.eval_type)}</span>
        <span className="mono">score {String(ev.overall_score)}{ev.hard_fail_capped ? ' (capped)' : ''}</span>
      </div>
      <div className="card-body">
        <div style={{ display: 'flex', gap: 16, marginBottom: 12, flexWrap: 'wrap' }}>
          {chips.map(([label, val, color]) => (
            <div key={label} className="stat">
              <b style={{ color }}>{val != null ? String(val) : '—'}</b><span>{label}</span>
            </div>
          ))}
        </div>
        {claims > 0 && (
          <div style={{ height: 8, borderRadius: 5, overflow: 'hidden', background: '#ef6a6a44', display: 'flex' }}>
            <div style={{ width: `${(pass / claims) * 100}%`, background: 'var(--ok)' }} />
          </div>
        )}
      </div>
    </div>
  )
}

function Workflow({ run }: { run: RunDetailT }) {
  const hasGraph = run.artifacts.some((a) => a.name === 'workflow_graph' && !a.empty)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <div className="card">
        <div className="card-head">task</div>
        <div className="card-body">
          <p style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{run.original_task || run.goal}</p>
        </div>
      </div>
      <div className="grid-2">
        <div className="card">
          <div className="card-head">executed workflow graph · LangGraph</div>
          <div className="card-body">
            {hasGraph
              ? <img className="artifact-img" src={artifactUrl(run.id, 'workflow_graph')} alt="workflow graph" />
              : <div className="hint">No workflow diagram (run crashed before execution).</div>}
          </div>
        </div>
        <div className="card">
          <div className="card-head">how it was built · generated source</div>
          <div className="card-body">
            {run.genotype ? <pre className="code tight">{run.genotype}</pre> : <div className="hint">No source recorded.</div>}
          </div>
        </div>
      </div>
    </div>
  )
}

/** Expert view: the evolutionary search that produced this workflow. */
function Evolution({ run }: { run: RunDetailT }) {
  const runId = run.id
  const tree = useAsync(() => api.tree(runId), [runId])
  const series = useAsync(() => api.series(runId), [runId])
  const navigate = useNavigate()
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <div className="card">
        <div className="card-head">evolution lineage · click a node to open it</div>
        {tree.loading
          ? <Spinner />
          : tree.data
            ? <LineageTree tree={tree.data} onSelect={(id) => navigate(`/runs/${id}`)} />
            : <div className="hint">No lineage.</div>}
      </div>
      <div className="card">
        <div className="card-head">reward &amp; cost across iterations</div>
        <div className="card-body">
          {series.loading ? <Spinner /> : series.data ? <RewardChart series={series.data} /> : null}
        </div>
      </div>
      <div className="grid-2">
        {run.textual_gradient && (
          <div className="card">
            <div className="card-head">textual gradient · steers the next mutation</div>
            <div className="card-body"><pre className="code">{run.textual_gradient}</pre></div>
          </div>
        )}
        {run.evolution_prompt && (
          <div className="card">
            <div className="card-head">evolution prompt</div>
            <div className="card-body"><pre className="code">{run.evolution_prompt}</pre></div>
          </div>
        )}
      </div>
    </div>
  )
}

function Artifacts({ run }: { run: RunDetailT }) {
  const [sel, setSel] = useState<Artifact | null>(
    run.artifacts.find((a) => a.name === 'workflow_graph') || run.artifacts[0] || null,
  )
  return (
    <div className="split narrow">
      <div className="card">
        <div className="card-head">files</div>
        <div>
          {run.artifacts.map((a) => (
            <button
              key={a.name}
              className="ws-row"
              data-active={sel?.name === a.name}
              onClick={() => setSel(a)}
              style={ROW}
            >
              <span style={{ flex: 1, fontFamily: 'var(--mono)', fontSize: 12 }}>{a.name}</span>
              <span className="muted" style={{ fontSize: 10.5 }}>{a.kind}</span>
            </button>
          ))}
        </div>
      </div>
      <div>{sel && <ArtifactView run={run} artifact={sel} />}</div>
    </div>
  )
}

const ROW: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 8, width: '100%', textAlign: 'left',
  background: 'transparent', border: 'none', borderBottom: '1px solid var(--border)',
  borderRadius: 0, padding: '9px 12px',
}

function ArtifactView({ run, artifact }: { run: RunDetailT; artifact: Artifact }) {
  const url = artifactUrl(run.id, artifact.name)
  const [text, setText] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    if (artifact.kind === 'image') return
    setLoading(true)
    fetch(url).then((r) => r.text()).then(setText).catch(() => setText(null)).finally(() => setLoading(false))
  }, [url, artifact.kind])

  return (
    <div className="card">
      <div className="card-head">
        <span>{artifact.filename}</span>
        <a href={url} target="_blank" rel="noreferrer">open ↗</a>
      </div>
      <div className="card-body">
        {artifact.empty && <div className="hint">Empty file (run incomplete).</div>}
        {!artifact.empty && artifact.kind === 'image' && <img className="artifact-img" src={url} alt={artifact.name} />}
        {!artifact.empty && artifact.kind !== 'image' && (loading ? <Spinner /> : <pre className="code">{text}</pre>)}
      </div>
    </div>
  )
}

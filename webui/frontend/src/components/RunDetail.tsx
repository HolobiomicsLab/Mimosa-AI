import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, artifactUrl } from '../api'
import { useAsync, useLive } from '../hooks'
import type { Artifact, ClaimStatus, EvaluationClaim, LiveEvent, RunDetail as RunDetailT } from '../types'
import {
  KindTag, ScoreChip, Spinner, StatusBadge, fmtCost, fmtDuration,
} from '../ui'
import EvolutionReplay from './EvolutionReplay'
import MemoryReplay from './MemoryReplay'
import WorkspacePanel from './WorkspacePanel'
import ProvenancePanel from './ProvenancePanel'
import TaskPanel from './TaskPanel'
import LiveFeed from './LiveFeed'
import { SmartText } from '../render'

// Scientist-first ordering: outputs → what the agents did → the workflow. The
// neuroevolution machinery lives under "Evolution", shown only for learning runs.
const TAB_IDS = ['results', 'task', 'verification', 'provenance', 'replay', 'workflow', 'evolution', 'artifacts'] as const
type Tab = (typeof TAB_IDS)[number]

/** Per-view query params that only make sense inside the tab that set them. */
const TAB_SCOPED_PARAMS = ['call', 'from', 'file']

/** Objective shown truncated (first 512 chars) with a toggle to expand/collapse. */
const OBJECTIVE_LIMIT = 512

function ObjectiveText({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false)
  const long = text.length > OBJECTIVE_LIMIT
  const shown = expanded || !long ? text : `${text.slice(0, OBJECTIVE_LIMIT)}…`
  return (
    <h2 className="objective">
      {shown}
      {long && (
        <button className="objective-toggle" onClick={() => setExpanded((v) => !v)}>
          {expanded ? 'show less' : 'show more'}
        </button>
      )}
    </h2>
  )
}

export default function RunDetail({ runId }: { runId: string }) {
  const { data: run, loading, error, refetch } = useAsync<RunDetailT>(() => api.run(runId), [runId])
  // The active tab lives in the URL (?tab=…) so run views are deep-linkable;
  // an absent or unknown value falls back to the default tab.
  const [params, setParams] = useSearchParams()
  const rawTab = params.get('tab')
  const urlTab: Tab = (TAB_IDS as readonly string[]).includes(rawTab ?? '') ? (rawTab as Tab) : 'results'
  const setTab = (t: Tab) => {
    setParams((prev) => {
      const next = new URLSearchParams(prev)
      if (t === 'results') next.delete('tab')
      else next.set('tab', t)
      TAB_SCOPED_PARAMS.forEach((p) => next.delete(p))
      return next
    })
  }
  // Live remount keys: bumping one refetches that panel from disk.
  const [wsVersion, setWsVersion] = useState(0)
  const [provVersion, setProvVersion] = useState(0)
  const [sawActivity, setSawActivity] = useState(false)

  useEffect(() => { setSawActivity(false) }, [runId])

  /** This run's filesystem events, subscribed here (not in LiveFeed) so a
   * completed run still surfaces the feed the moment new activity arrives. */
  const onLive = (e: LiveEvent) => {
    if (e.run_id && e.run_id !== runId) return
    setSawActivity(true)
    if (['iteration_complete', 'execution_complete', 'run_finished', 'evaluation_updated'].includes(e.type)) {
      refetch()
      setWsVersion((v) => v + 1)
    }
    if (e.type === 'astra_updated' || e.type === 'evaluation_capsule_updated') {
      setProvVersion((v) => v + 1)
    }
  }
  useLive(onLive)

  if (loading) return <Spinner label="Loading run…" />
  if (error || !run) return <div className="empty"><div className="hint">Run not found.</div></div>

  const live = run.status === 'running' || sawActivity

  const tabs: [Tab, string, number?][] = [
    ['results', 'Results'],
    ['task', 'Task'],
    ...(run.evaluation_scores ? [['verification', 'Verification'] as [Tab, string]] : []),
    ['provenance', 'Provenance'],
    ['replay', 'Replay'],
    ['workflow', 'Workflow'],
    ...(run.learning_mode ? [['evolution', 'Evolution'] as [Tab, string]] : []),
    ['artifacts', 'Evolution Artifact (Expert)', run.artifacts.length],
  ]
  // A deep link may name a tab this run does not show (e.g. ?tab=evolution on
  // a plain task run); fall back to the default rather than a blank body.
  const tab: Tab = tabs.some(([id]) => id === urlTab) ? urlTab : 'results'

  return (
    <>
      <div className="detail-head">
        <div className="rid">{run.id}</div>
        <ObjectiveText text={run.goal || 'Untitled run'} />
        <div className="stats">
          <div className="stat"><b><StatusBadge status={run.status} /></b></div>
          <div className="stat"><b><ScoreChip score={run.score} /></b></div>
          <div className="stat"><b>{fmtCost(run.cost_usd)}</b></div>
          <div className="stat"><b>{fmtDuration(run.wall_time_s)}</b></div>
          {run.learning_mode && <div className="stat"><b><KindTag kind={run.evolution_kind} /></b></div>}
          {run.is_single_agent && <div className="stat"><b>single</b></div>}
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
        {tab === 'results' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            {live && <LiveFeed runId={runId} />}
            <WorkspacePanel key={wsVersion} runId={run.id} />
          </div>
        )}
        {tab === 'task' && <TaskPanel runId={runId} />}
        {tab === 'verification' && <Verification run={run} />}
        {tab === 'provenance' && <ProvenancePanel key={provVersion} runId={runId} />}
        {tab === 'replay' && <MemoryReplay runId={runId} />}
        {tab === 'workflow' && <Workflow run={run} />}
        {tab === 'evolution' && <EvolutionReplay runId={runId} />}
        {tab === 'artifacts' && <Artifacts run={run} />}
      </div>
    </>
  )
}


/** Verification view: the verifier's score breakdown and per-claim results. */
function Verification({ run }: { run: RunDetailT }) {
  const ev = run.evaluation_scores
  if (!ev) return <div className="empty"><div className="hint">No verification data for this run.</div></div>
  return <EvalCard ev={ev} claims={run.evaluation_claims} />
}

function EvalCard({ ev, claims }: { ev: Record<string, unknown>; claims: EvaluationClaim[] | null }) {
  const num = (k: string) => (typeof ev[k] === 'number' ? (ev[k] as number) : undefined)
  const chips: [string, unknown, string][] = [
    ['claims', ev.n_claims, 'var(--text)'],
    ['pass', ev.n_pass, 'var(--ok)'],
    ['fail', ev.n_fail, 'var(--bad)'],
    ['error', ev.n_error, 'var(--warn)'],
    ['unsure', ev.n_unsure, 'var(--text-dim)'],
  ]
  const pass = num('n_pass') ?? 0
  const nClaims = num('n_claims') ?? 0
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
        {nClaims > 0 && (
          <div style={{ height: 8, borderRadius: 5, overflow: 'hidden', background: '#ef6a6a44', display: 'flex' }}>
            <div style={{ width: `${(pass / nClaims) * 100}%`, background: 'var(--ok)' }} />
          </div>
        )}
        {claims && claims.length > 0 && <ClaimList claims={claims} />}
      </div>
    </div>
  )
}

const CLAIM_RANK: Record<ClaimStatus, number> = { fail: 0, error: 1, unsure: 2, pass: 3 }
const CLAIM_CLASS: Record<ClaimStatus, string> = {
  pass: 'ok', fail: 'bad', error: 'warn', unsure: 'dim',
}

/** Per-claim pass/fail breakdown, failures first so they're immediately visible. */
function ClaimList({ claims }: { claims: EvaluationClaim[] }) {
  const [showPass, setShowPass] = useState(false)
  const sorted = [...claims].sort(
    (a, b) =>
      (CLAIM_RANK[a.status ?? 'pass'] ?? 9) - (CLAIM_RANK[b.status ?? 'pass'] ?? 9) ||
      b.importance - a.importance,
  )
  const passCount = claims.filter((c) => c.status === 'pass').length
  const shown = showPass ? sorted : sorted.filter((c) => c.status !== 'pass')

  return (
    <div className="claim-list">
      {shown.map((c) => (
        <div key={c.id} className="claim-row">
          <span className={`claim-status ${CLAIM_CLASS[c.status ?? 'unsure'] ?? 'dim'}`}>
            {c.status ?? '—'}
          </span>
          <div className="claim-body">
            <div className="claim-desc">{c.description}</div>
            {c.details && <div className="claim-details">{c.details}</div>}
            <div className="claim-tags">
              <span className="muted">importance {c.importance}</span>
              {c.relevant_files.map((f) => <span key={f} className="claim-file">{f}</span>)}
            </div>
          </div>
        </div>
      ))}
      {passCount > 0 && (
        <button className="claim-toggle" onClick={() => setShowPass((v) => !v)}>
          {showPass ? 'hide' : 'show'} {passCount} passing claim{passCount === 1 ? '' : 's'}
        </button>
      )}
    </div>
  )
}

function Workflow({ run }: { run: RunDetailT }) {
  const hasGraph = run.artifacts.some((a) => a.name === 'workflow_graph' && !a.empty)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <div className="card">
        <div className="card-head">executed workflow graph · LangGraph</div>
        <div className="card-body">
          {hasGraph
            ? <img className="artifact-img" src={artifactUrl(run.id, 'workflow_graph')} alt="workflow graph" />
            : <div className="hint">No workflow diagram (run crashed before execution).</div>}
        </div>
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
        {!artifact.empty && artifact.kind !== 'image' && (
          loading
            ? <Spinner />
            : <SmartText text={text ?? ''} filename={artifact.filename} kind={artifact.kind} />
        )}
      </div>
    </div>
  )
}

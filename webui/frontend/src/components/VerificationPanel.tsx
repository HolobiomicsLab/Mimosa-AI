import { Fragment } from 'react'
import { useNavigate } from 'react-router-dom'
import type {
  ExecutionRecord, PairEvidence, PairOutcome, PolicyBTVerification, PolicyCriterion,
  PolicyPair, ProcessRecord, RunDetail,
} from '../types'
import { ScoreChip, fmtDuration, fmtScore, scoreColor, shortId } from '../ui'

// Policy-BT verifier view: reproduction gate → pair verdicts → BT reward.
// Payload shape: webui/backend/app/verification.py.

const OUTCOME_CLASS: Record<PairOutcome, string> = { loss: 'bad', tie: 'dim', win: 'ok' }
const OUTCOME_RANK: Record<PairOutcome, number> = { loss: 0, tie: 1, win: 2 }
const DIRECTION_LABEL: Record<PolicyCriterion['direction'], string> = {
  higher: '↑ higher is better', lower: '↓ lower is better', satisfy: '= must hold',
}
const CHANNEL_LABEL: Record<string, string> = { code: 'comparison policy', visual: 'visual tie-break' }

export default function VerificationPanel({ run }: { run: RunDetail }) {
  const v = run.verification
  if (!v) return <div className="empty"><div className="hint">No Policy-BT verification for this run.</div></div>
  return (
    <div className="vstack">
      <SummaryCard v={v} runId={run.id} />
      {run.textual_gradient && <GradientCard text={run.textual_gradient} />}
      <ReproductionCard execution={v.execution} />
      <PairsCard pairs={v.pairs} criteria={v.criteria ?? []} />
      {v.criteria && v.criteria.length > 0 && <CriteriaCard criteria={v.criteria} />}
    </div>
  )
}

const tally = (pairs: PolicyPair[], outcome: PairOutcome) => pairs.filter((p) => p.outcome === outcome).length

/** Reward headline: current BT rating, arrival rating, rank, W/L/T and gate state. */
function SummaryCard({ v, runId }: { v: PolicyBTVerification; runId: string }) {
  const ratings = Object.values(v.ratings)
  const own = v.ratings[runId] ?? v.reward
  const rank = own == null || ratings.length === 0 ? null : 1 + ratings.filter((r) => r > own).length
  const tiedRank = own != null && ratings.filter((r) => r === own).length > 1
  const [wins, losses, ties] = (['win', 'loss', 'tie'] as PairOutcome[]).map((o) => tally(v.pairs, o))
  const total = v.pairs.length
  const stats: [string, string | number, string][] = [
    ['at arrival', fmtScore(v.arrival_reward), 'var(--text)'],
    ['rank', rank == null ? '—' : `${tiedRank ? '=' : ''}${rank} / ${ratings.length}`, 'var(--text)'],
    ['wins', wins, 'var(--ok)'],
    ['losses', losses, 'var(--bad)'],
    ['ties', ties, 'var(--text-dim)'],
  ]
  return (
    <div className="card">
      <div className="card-head">
        <span>verification · policy-BT tournament</span>
        <span className="mono">tournament v{v.tournament_version ?? '—'}</span>
      </div>
      <div className="card-body">
        <div style={{ display: 'flex', gap: 22, flexWrap: 'wrap', alignItems: 'flex-end', marginBottom: 12 }}>
          <div className="stat"><b><ScoreChip score={v.reward} /></b><span>reward (current)</span></div>
          {stats.map(([label, value, color]) => (
            <div key={label} className="stat"><b style={{ color }}>{value}</b><span>{label}</span></div>
          ))}
          <GateBadge execution={v.execution} />
        </div>
        {total > 0 && (
          <div className="outcome-bar" title={`${wins} won · ${ties} tied · ${losses} lost`}>
            <div style={{ width: `${(wins / total) * 100}%`, background: 'var(--ok)' }} />
            <div style={{ width: `${(ties / total) * 100}%`, background: '#6b7688' }} />
            <div style={{ width: `${(losses / total) * 100}%`, background: 'var(--bad)' }} />
          </div>
        )}
        <div className="hint" style={{ marginTop: 8 }}>
          Reward is the regularized Bradley–Terry rating over all pair verdicts of this task
          session; it is refreshed every time a later generation is evaluated.
        </div>
      </div>
    </div>
  )
}

function GateBadge({ execution }: { execution: ExecutionRecord }) {
  const ok = execution.eligible === true
  return (
    <span className={`badge ${ok ? 'completed' : 'crashed'}`} title={execution.reason ?? ''}>
      <span className="tick" style={{ background: 'currentColor' }} />
      {ok ? 'reproducible' : (execution.status ?? 'ineligible').replace(/_/g, ' ')}
    </span>
  )
}

/** The verifier's diagnosis handed to the next mutation. */
function GradientCard({ text }: { text: string }) {
  return (
    <div className="card">
      <div className="card-head">textual gradient · feedback to the next mutation</div>
      <div className="card-body"><pre className="code">{text}</pre></div>
    </div>
  )
}

/** Reproduction contract: the `.sh` script, its rerun, and which outputs came back. */
function ReproductionCard({ execution: ex }: { execution: ExecutionRecord }) {
  const missing = new Set(ex.files?.missing_outputs ?? [])
  const regenerated = new Set(ex.files?.regenerated ?? [])
  const outputs = ex.contract?.generated_outputs ?? []
  const facts: [string, string][] = [
    ['script', ex.script ?? '—'],
    ['exit code', ex.returncode == null ? '—' : String(ex.returncode)],
    ['duration', fmtDuration(ex.duration_s)],
    ['rerun', ex.attempted ? (ex.timed_out ? 'timed out' : 'attempted') : ex.enabled === false ? 'disabled' : 'not attempted'],
  ]
  if (ex.contract) {
    facts.push(['installs requirements', ex.contract.installs_requirements ? 'yes' : 'no'])
    facts.push(['regenerates outputs', ex.contract.regenerates_outputs ? 'yes' : 'no'])
  }
  return (
    <div className="card">
      <div className="card-head">
        <span>reproduction contract</span>
        <GateBadge execution={ex} />
      </div>
      <div className="card-body">
        {ex.reason && <div className={`gate-reason ${ex.eligible ? 'ok' : 'bad'}`}>{ex.reason}</div>}
        <dl className="kv">
          {facts.map(([k, val]) => <Fragment key={k}><dt>{k}</dt><dd>{val}</dd></Fragment>)}
        </dl>
        {outputs.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <div className="hint">declared outputs {missing.size > 0 && `· ${missing.size} missing after rerun`}</div>
            <div className="verdict-tags">
              {outputs.map((f) => (
                <span key={f} className={`verdict-file ${missing.has(f) ? 'bad' : regenerated.has(f) ? 'ok' : ''}`}
                      title={missing.has(f) ? 'missing after rerun' : regenerated.has(f) ? 'regenerated' : ''}>
                  {f}
                </span>
              ))}
            </div>
          </div>
        )}
        <FileDiff files={ex.files} />
        {ex.contract?.reason && <Fold label="why the contract was accepted"><div className="verdict-sub">{ex.contract.reason}</div></Fold>}
        <ProcessFold label="rerun output" record={ex.run} />
        {ex.provision && (ex.provision.returncode !== 0 || ex.provision.error) && (
          <ProcessFold label="venv provisioning" record={ex.provision} />
        )}
      </div>
    </div>
  )
}

/** Counts of workspace files the rerun created, changed or removed. */
function FileDiff({ files }: { files: ExecutionRecord['files'] }) {
  if (!files) return null
  const groups = (['created', 'changed', 'removed', 'regenerated'] as const)
    .map((key) => [key, files[key] ?? []] as [string, string[]])
    .filter(([, list]) => list.length > 0)
  return (
    <div className="hint" style={{ marginTop: 12 }}>
      rerun file diff: {groups.map(([k, list]) => `${list.length} ${k}`).join(' · ') || 'no change'}
      {` · ${files.unchanged_count} unchanged`}
      {groups.length > 0 && (
        <Fold label="show changed paths">
          {groups.map(([k, list]) => (
            <div key={k} className="verdict-tags"><span className="muted">{k}</span>
              {list.map((f) => <span key={f} className="verdict-file">{f}</span>)}
            </div>
          ))}
        </Fold>
      )}
    </div>
  )
}

function ProcessFold({ label, record }: { label: string; record: ProcessRecord | null }) {
  if (!record || (!record.stdout && !record.stderr && !record.error)) return null
  const head = `${label} · exit ${record.returncode ?? '—'}${record.timed_out ? ' · timed out' : ''}`
  return (
    <Fold label={head}>
      {record.argv && <div className="hint mono" style={{ marginBottom: 6 }}>$ {record.argv.join(' ')}</div>}
      {record.error && <pre className="code" style={{ color: '#ef9a9a' }}>{record.error}</pre>}
      {record.stderr && <><div className="hint">stderr (tail)</div><pre className="code" style={{ color: '#ef9a9a' }}>{record.stderr}</pre></>}
      {record.stdout && <><div className="hint">stdout (tail)</div><pre className="code">{record.stdout}</pre></>}
    </Fold>
  )
}

function Fold({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <details className="vdetails">
      <summary>{label}</summary>
      <div className="vdetails-body">{children}</div>
    </details>
  )
}

/** Every pair verdict involving this run, losses first (the gradient's lead). */
function PairsCard({ pairs, criteria }: { pairs: PolicyPair[]; criteria: PolicyCriterion[] }) {
  const sorted = [...pairs].sort(
    (a, b) => OUTCOME_RANK[a.outcome] - OUTCOME_RANK[b.outcome] || a.stage - b.stage,
  )
  return (
    <div className="card">
      <div className="card-head">
        <span>pair verdicts · {pairs.length}</span>
        <span className="mono">
          {tally(pairs, 'win')}W · {tally(pairs, 'loss')}L · {tally(pairs, 'tie')}T
        </span>
      </div>
      <div className="card-body">
        {sorted.length === 0
          ? <div className="hint">No predecessor in this task session yet — nothing to compare against.</div>
          : <div className="verdict-list">
              {sorted.map((p) => <PairRow key={p.opponent} pair={p} criteria={criteria} />)}
            </div>}
      </div>
    </div>
  )
}

function stageLabel(pair: PolicyPair, criteria: PolicyCriterion[]): string {
  if (pair.channel === 'execution') return 'execution gate'
  if (pair.stage === 0) return 'no decisive stage'
  const names = criteria.filter((c) => c.stage === pair.stage).map((c) => c.id)
  return `stage ${pair.stage}${names.length ? ` · ${names.join(', ')}` : ''}`
}

function PairRow({ pair, criteria }: { pair: PolicyPair; criteria: PolicyCriterion[] }) {
  const navigate = useNavigate()
  const hasDetail = pair.opponent_feedback || pair.evidence.length > 0 || pair.values
  return (
    <div className="verdict-row">
      <span className={`verdict-badge ${OUTCOME_CLASS[pair.outcome]}`}>{pair.outcome}</span>
      <div className="verdict-body">
        <div className="verdict-title">
          <span className="muted">vs</span>
          <button className="link-btn mono" onClick={() => navigate(`/runs/${pair.opponent}`)} title={pair.opponent}>
            {shortId(pair.opponent)}
          </button>
          {pair.opponent_reward != null && (
            <span className="score-chip" style={{ background: scoreColor(pair.opponent_reward) }}>
              {fmtScore(pair.opponent_reward)}
            </span>
          )}
          <span className="kind-tag">{stageLabel(pair, criteria)}</span>
          {pair.channel && pair.channel !== 'execution' && (
            <span className="kind-tag">· {CHANNEL_LABEL[pair.channel] ?? pair.channel}</span>
          )}
        </div>
        {pair.feedback && <div className="verdict-text">{pair.feedback}</div>}
        {hasDetail && (
          <Fold label={pair.values || pair.evidence.length > 0 ? 'evidence' : "opponent's feedback"}>
            {pair.values && <ValuesTable pair={pair} criteria={criteria} />}
            {pair.evidence.length > 0 && <EvidenceList evidence={pair.evidence} />}
            {pair.opponent_feedback && (
              <div className="verdict-sub" style={{ marginTop: 8 }}>
                <span className="muted">opponent's feedback: </span>{pair.opponent_feedback}
              </div>
            )}
          </Fold>
        )}
      </div>
    </div>
  )
}

const fmtValue = (value: unknown) =>
  value === undefined || value === null ? '—' : typeof value === 'object' ? JSON.stringify(value) : String(value)

/** Normalized criterion values of both sides; the deciding stage is highlighted. */
function ValuesTable({ pair, criteria }: { pair: PolicyPair; criteria: PolicyCriterion[] }) {
  const values = pair.values!
  const known = criteria.map((c) => c.id)
  const extra = Object.keys({ ...values.self, ...values.opponent }).filter((id) => !known.includes(id))
  const decisive = new Set(criteria.filter((c) => c.stage === pair.stage).map((c) => c.id))
  const byId = new Map(criteria.map((c) => [c.id, c]))
  return (
    <table className="vtable">
      <thead><tr><th>criterion</th><th>this run</th><th>opponent</th></tr></thead>
      <tbody>
        {[...known, ...extra].map((id) => {
          const mine = fmtValue(values.self?.[id])
          const theirs = fmtValue(values.opponent?.[id])
          const cls = decisive.has(id) && pair.outcome !== 'tie' ? 'decisive' : mine !== theirs ? 'differs' : ''
          const c = byId.get(id)
          return (
            <tr key={id} className={cls}>
              <td title={c?.description}>
                {c && <span className="muted">{c.stage} · </span>}{id}
                {c && <span className="muted"> {DIRECTION_LABEL[c.direction]?.split(' ')[0]}</span>}
              </td>
              <td className="mono">{mine}</td>
              <td className="mono">{theirs}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function EvidenceList({ evidence }: { evidence: PairEvidence[] }) {
  return (
    <div style={{ marginTop: 8 }}>
      <div className="hint">decisive evidence</div>
      {evidence.map((e, i) => (
        <div key={i} className="verdict-tags">
          <span className={`kind-tag ${e.side === 'self' ? '' : 'muted'}`}>{e.side === 'self' ? 'this run' : 'opponent'}</span>
          {e.criterion && <span className="mono" style={{ fontSize: 12 }}>{e.criterion}</span>}
          {e.path && <span className="verdict-file">{e.path}</span>}
          {e.value !== undefined && <span className="mono" style={{ fontSize: 12 }}>= {fmtValue(e.value)}</span>}
          {(e.detail || e.observation) && <span className="verdict-sub" style={{ marginTop: 0 }}>{e.detail ?? e.observation}</span>}
        </div>
      ))}
    </div>
  )
}

/** The frozen, goal-derived criteria ladder every pair is decided on. */
function CriteriaCard({ criteria }: { criteria: PolicyCriterion[] }) {
  return (
    <div className="card">
      <div className="card-head">comparison criteria · frozen for this task session</div>
      <div className="card-body">
        <table className="vtable">
          <thead><tr><th>stage</th><th>criterion</th><th>direction</th><th>requirement</th></tr></thead>
          <tbody>
            {criteria.map((c) => (
              <tr key={c.id}>
                <td className="mono">{c.stage}</td>
                <td className="mono">{c.id}</td>
                <td>{DIRECTION_LABEL[c.direction] ?? c.direction}</td>
                <td>{c.description ?? ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

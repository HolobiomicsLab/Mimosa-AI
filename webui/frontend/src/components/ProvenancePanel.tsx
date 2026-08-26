import { useEffect } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAsync } from '../hooks'
import type {
  AstraDecision, DecisionsEra, EvalVerdict, JudgeLayer, Provenance, RunEvaluation,
} from '../types'
import { CopyLink, Spinner, shortId } from '../ui'

/**
 * Provenance tab — the run rendered FROM its structured record (the MySTRA
 * principle): the ASTRA capsule's decisions with their alternatives, and the
 * ASB-as-evaluator capsules that judged the workspace independently of the
 * run's own verifier.
 */
export default function ProvenancePanel({ runId }: { runId: string }) {
  const { data, loading, error } = useAsync<Provenance>(() => api.provenance(runId), [runId])
  const { hash } = useLocation()

  // Deep-linked decision anchors (#decision-<slug>, D16: the slug is the
  // capsule's export-time decision id) scroll into view once data is here —
  // SPA navigation never triggers the browser's native fragment scroll.
  useEffect(() => {
    if (!data || !hash.startsWith('#')) return
    document.getElementById(decodeURIComponent(hash.slice(1)))
      ?.scrollIntoView({ block: 'start' })
  }, [data, hash])

  if (loading) return <Spinner label="Loading provenance…" />
  if (error || !data) return <div className="empty"><div className="hint">No provenance data.</div></div>

  const empty = !data.astra && data.family_capsules.length === 0 && data.evaluations.length === 0
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {data.astra ? <AstraCard capsule={data.astra} /> : <NoCapsule family={data.family_capsules} />}
      {data.evaluations.map((ev) => <EvaluationCard key={ev.source} ev={ev} />)}
      {empty && (
        <div className="empty">
          <div className="hint">
            No ASTRA capsule (only a family's best run exports one) and no
            independent evaluation found. Run <span className="mono">asb_eval</span> against
            this run's workspace to add the evaluation layer.
          </div>
        </div>
      )}
    </div>
  )
}

function NoCapsule({ family }: { family: string[] }) {
  const navigate = useNavigate()
  if (family.length === 0) return null
  return (
    <div className="card">
      <div className="card-head"><span>ASTRA capsule</span></div>
      <div className="card-body">
        <div className="hint">
          The transparency exporter writes one capsule per evolution family — its
          best run. This run has none of its own; the family's record lives at:
        </div>
        <div style={{ display: 'flex', gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
          {family.map((id) => (
            <button key={id} className="mono" onClick={() => navigate(`/runs/${id}`)}>
              {shortId(id)}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

function AstraCard({ capsule }: { capsule: NonNullable<Provenance['astra']> }) {
  const decisions = Object.entries(capsule.decisions)
  return (
    <div className="card">
      <div className="card-head">
        <span>ASTRA capsule · decisions with alternatives</span>
        <span className="mono">v{capsule.version ?? '?'}</span>
      </div>
      <div className="card-body">
        {capsule.description && <div className="hint" style={{ marginBottom: 10 }}>{capsule.description}</div>}
        {decisions.length === 0 && <div className="hint">The capsule records no decisions.</div>}
        {decisions.map(([slug, d]) => (
          <DecisionRow key={slug} slug={slug} d={d} era={capsule.decisions_era} />
        ))}
        {capsule.universes.length > 0 && (
          <div style={{ marginTop: 12 }}>
            {capsule.universes.map((u) => (
              <div key={u.id ?? '?'} className="hint">
                universe <b className="mono">{u.id}</b>
                {u.decisions && ' — ' + Object.entries(u.decisions)
                  .map(([k, v]) => `${k}: ${String(v)}`).join(' · ')}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

/** One decision: the chosen option first and green, alternatives dim below,
 * plus deep links into the memory-trace evidence it was extracted from. */
function DecisionRow({ slug, d, era }: { slug: string; d: AstraDecision; era: DecisionsEra | null }) {
  const options = Object.entries(d.options ?? {})
  const chosen = d.default != null ? String(d.default) : null
  const anchor = `decision-${slug}`
  const { hash } = useLocation()
  const targeted = hash === `#${encodeURIComponent(anchor)}` || hash === `#${anchor}`
  return (
    <div
      id={anchor}
      style={{
        padding: '10px 0', borderTop: '1px solid #ffffff14', scrollMarginTop: 12,
        ...(targeted ? { background: '#f2c14e10', outline: '1px solid var(--gold-dim)', borderRadius: 6, padding: '10px 8px' } : {}),
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
        <b>{d.label ?? slug}</b>
        <span style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {d.model && <span className="mono" style={{ color: 'var(--text-dim)', fontSize: 11 }}>{d.model}</span>}
          <CopyLink url={`?tab=provenance#${encodeURIComponent(anchor)}`} label={`link to decision ${slug}`} />
        </span>
      </div>
      {d.rationale && <div className="hint" style={{ margin: '4px 0 6px' }}>{d.rationale}</div>}
      <EvidenceLinks slug={slug} d={d} era={era} />
      {options.map(([oid, o]) => {
        const isChosen = oid === chosen
        return (
          <div key={oid} style={{
            display: 'flex', gap: 8, alignItems: 'baseline', padding: '2px 0',
            color: isChosen ? 'var(--ok)' : 'var(--text-dim)',
          }}>
            <span className="mono" style={{ width: 14, flexShrink: 0 }}>{isChosen ? '✓' : '·'}</span>
            <span>
              <b style={{ fontWeight: isChosen ? 600 : 400 }}>{o.label ?? oid}</b>
              {o.description && <span style={{ fontSize: 12 }}> — {o.description}</span>}
              {o.excluded_reason && <span style={{ fontSize: 12 }}> (excluded: {o.excluded_reason})</span>}
            </span>
          </div>
        )
      })}
    </div>
  )
}

/** Deep links from a decision to the memory calls it was extracted from.
 *
 * A decision without `source_steps` renders an explicit muted marker with the
 * BACKEND's reason verbatim (never a synthesized one); only when that
 * per-decision reason is absent does the capsule-level `decisions_era` marker
 * stand in. Partially-parsed tags surface the `unparsed_trace_tags` count so
 * a degraded join is never mistaken for a complete one.
 */
function EvidenceLinks({ slug, d, era }: { slug: string; d: AstraDecision; era: DecisionsEra | null }) {
  const steps = d.source_steps ?? []
  const unparsed = d.unparsed_trace_tags ?? 0
  const absentReason = d.source_steps_absent_reason ?? era ?? 'no evidence link recorded'
  return (
    <div style={{ margin: '2px 0 6px', fontSize: 12 }}>
      {steps.length > 0 ? (
        <span>
          <span className="muted">evidence: </span>
          {steps.map((n) => (
            <Link
              key={n}
              className="mono"
              style={{ marginRight: 8 }}
              title={`open memory call astra_decision_step_${n}`}
              to={`?tab=replay&call=astra_decision_step_${n}&from=${encodeURIComponent(slug)}`}
            >
              step {n} ↗
            </Link>
          ))}
        </span>
      ) : (
        <span className="muted" style={{ fontStyle: 'italic' }}>no evidence link — {absentReason}</span>
      )}
      {unparsed > 0 && (
        <span className="muted"> · {unparsed} trace tag{unparsed === 1 ? '' : 's'} unparseable</span>
      )}
    </div>
  )
}

const VERDICT_COLOR: Record<string, string> = {
  pass: 'var(--ok)', fail: 'var(--bad)',
  undecided: 'var(--text-dim)', undecidable: 'var(--text-dim)', uncertain: 'var(--text-dim)',
}
const VERDICT_MARK: Record<string, string> = {
  pass: '✓', fail: '✗', undecided: '∅', undecidable: '∅', uncertain: '∅',
}

function EvaluationCard({ ev }: { ev: RunEvaluation }) {
  const s = ev.summary
  const num = (k: string) => (typeof s[k] === 'number' ? (s[k] as number) : undefined)
  const decided = num('checks_decided') ?? 0
  const passed = num('passed') ?? 0
  return (
    <div className="card">
      <div className="card-head">
        <span>independent evaluation · {ev.criteria_source ?? ev.name ?? ev.source}</span>
        <span className="mono">
          {passed}/{decided} decided passed · {num('checks_undecidable') ?? '—'} of {num('checks_total') ?? '—'} undecidable
        </span>
      </div>
      <div className="card-body">
        <div className="hint" style={{ marginBottom: 10 }}>
          Verdicts come from the ASB card's own criteria measured against the
          workspace artifacts — the evaluator never reads the run's
          self-assessment. Score is over decided checks only.
        </div>
        {ev.verdicts.map((v, i) => <VerdictRow key={v.check_id ?? i} v={v} />)}
        {ev.workspace_flags.length > 0 && (
          <div style={{ marginTop: 10 }}>
            <b style={{ fontSize: 12 }}>workspace provenance flags (advisory)</b>
            {ev.workspace_flags.map((f, i) => (
              <div key={i} className="hint">
                <span className="mono">{f.artefact}</span>: {f.kind}{f.detail ? ` — ${f.detail}` : ''}
              </div>
            ))}
          </div>
        )}
        {ev.judge && <JudgeBlock judge={ev.judge} />}
      </div>
    </div>
  )
}

function VerdictRow({ v }: { v: EvalVerdict }) {
  const status = v.status ?? '?'
  const color = VERDICT_COLOR[status] ?? 'var(--warn)'
  const detail: string[] = []
  if (v.failure_kind) detail.push(v.failure_kind)
  if (v.observed !== undefined && v.observed !== null) {
    detail.push(`observed ${String(v.observed)}${v.expected != null ? ` vs ${String(v.expected)}` : ''}`)
  }
  if (v.note) detail.push(v.note)
  return (
    <div style={{ display: 'flex', gap: 8, padding: '3px 0', alignItems: 'baseline' }}>
      <span className="mono" style={{ color, width: 14, flexShrink: 0 }}>{VERDICT_MARK[status] ?? '?'}</span>
      <span style={{ minWidth: 0 }}>
        <span className="mono" style={{ color, fontSize: 12 }}>{v.check_id}</span>
        {v.provenance_flags && v.provenance_flags.length > 0 && (
          <span className="mono" style={{ color: 'var(--warn)', fontSize: 11 }}> ⚑{v.provenance_flags.length}</span>
        )}
        {detail.length > 0 && (
          <span className="hint" style={{ fontSize: 12 }}> — {detail.join('; ')}</span>
        )}
      </span>
    </div>
  )
}

function JudgeBlock({ judge }: { judge: JudgeLayer }) {
  const inst = judge.instrument ?? {}
  const s = judge.summary ?? {}
  return (
    <div style={{ marginTop: 12, paddingTop: 10, borderTop: '1px solid #ffffff14' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, flexWrap: 'wrap' }}>
        <b style={{ fontSize: 12 }}>judge layer (separate — never merged into the executor score)</b>
        <span className="mono" style={{ fontSize: 11, color: 'var(--text-dim)' }}>
          {String(inst.model ?? '?')} · prompt {String(inst.prompt_sha ?? '?')} · {String(inst.passes ?? '?')} pass(es)
        </span>
      </div>
      <div className="hint" style={{ margin: '4px 0 6px' }}>
        judged {String(s.judged_total ?? '—')}: {String(s.judge_pass ?? 0)} pass
        · {String(s.judge_fail ?? 0)} fail · {String(s.judge_uncertain ?? 0)} uncertain
        · {judge.refused?.length ?? 0} refused
      </div>
      {(judge.verdicts ?? []).map((v, i) => (
        <div key={v.item_id ?? i} style={{ display: 'flex', gap: 8, padding: '2px 0', alignItems: 'baseline' }}>
          <span className="mono" style={{
            color: VERDICT_COLOR[v.verdict ?? ''] ?? 'var(--warn)', width: 14, flexShrink: 0,
          }}>{VERDICT_MARK[v.verdict ?? ''] ?? '?'}</span>
          <span style={{ fontSize: 12 }}>
            <span className="mono">{v.item_id}</span>
            <span style={{ color: 'var(--text-dim)' }}> ({v.layer})</span>
            {v.rationale && <span className="hint"> — {v.rationale}</span>}
          </span>
        </div>
      ))}
    </div>
  )
}

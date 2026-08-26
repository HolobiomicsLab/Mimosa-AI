import { api } from '../api'
import { useAsync } from '../hooks'
import type { TaskView } from '../types'
import { Spinner } from '../ui'
import { JsonTree } from '../render'

/**
 * Task tab — the task the run was actually asked to do, joined back from the
 * run dir outward (see webui/backend/app/taskdef.py). Four layers, each honest
 * about absence: the VERBATIM prompt (what the agent saw — not the card), the
 * parsed task ref, the run's self-declared grounding statistics, and the ASB
 * benchmark card rendered as a schema-driven field list.
 */
export default function TaskPanel({ runId }: { runId: string }) {
  const { data, loading, error } = useAsync<TaskView>(() => api.task(runId), [runId])

  if (loading) return <Spinner label="Loading task definition…" />
  if (error || !data) return <div className="empty"><div className="hint">No task endpoint response.</div></div>

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <PromptCard view={data} />
      <TaskRefCard view={data} />
      <GroundingCard view={data} />
      <AsbCard view={data} />
    </div>
  )
}

function AbsentLine({ reason }: { reason: string | null }) {
  return (
    <div className="hint" style={{ fontStyle: 'italic' }}>
      absent — {reason ?? 'no reason recorded'}
    </div>
  )
}

function PromptCard({ view }: { view: TaskView }) {
  return (
    <div className="card">
      <div className="card-head">
        <span>task prompt · verbatim</span>
        {view.prompt_file && <span className="mono">{view.prompt_file}</span>}
      </div>
      <div className="card-body">
        <div className="hint" style={{ marginBottom: 10 }}>
          the agent saw this text, not the card
        </div>
        {view.prompt != null
          ? (
            <pre className="code" style={{ whiteSpace: 'pre-wrap', maxHeight: '55vh', overflowY: 'auto' }}>
              {view.prompt}
            </pre>
          )
          : <AbsentLine reason={view.prompt_absent_reason} />}
      </div>
    </div>
  )
}

function TaskRefCard({ view }: { view: TaskView }) {
  const ref = view.task_ref
  return (
    <div className="card">
      <div className="card-head">
        <span>task ref · benchmark identity</span>
        {ref && (
          <span className="mono">
            {ref.source === 'task_ref.json' ? 'stamped (task_ref.json)' : 'parsed (prompt tag)'}
          </span>
        )}
      </div>
      <div className="card-body">
        {ref
          ? (
            <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap' }}>
              <div className="stat"><b className="mono">{ref.challenge ?? '—'}</b><span>challenge</span></div>
              <div className="stat"><b className="mono">{ref.task_id ?? '—'}</b><span>task id</span></div>
              <div className="stat"><b className="mono">{ref.csv_row ?? '—'}</b><span>csv row</span></div>
            </div>
          )
          : <AbsentLine reason={view.task_ref_absent_reason} />}
      </div>
    </div>
  )
}

function GroundingCard({ view }: { view: TaskView }) {
  return (
    <div className="card">
      <div className="card-head">
        <span>grounding · self-declared</span>
        <span className="mono">run_metrics.json</span>
      </div>
      <div className="card-body">
        <div className="hint" style={{ marginBottom: 10 }}>
          These numbers come from the run's own metrics file — the run declaring
          them about itself, not an independent observation.
        </div>
        {view.grounding
          ? <JsonTree value={view.grounding} />
          : <AbsentLine reason={view.grounding_absent_reason} />}
      </div>
    </div>
  )
}

/** The ASB card's top-level schema, grouped for reading order. A field the
 * card lacks renders an explicit "absent" row — never a dropped row — and
 * fields outside this list still render at the end (the schema guides
 * ordering; it never filters). */
const CARD_SCHEMA: [string, string[]][] = [
  ['identity', [
    'schema_version', 'project_id', 'scenario_id', 'task_id', 'title',
    'crossref_doi', 'github_name', 'article_type', 'created_at',
    'provenance_source', 'source_package', 'synthesized_from',
  ]],
  ['science', [
    'primary_domain', 'domain', 'subdomains', 'keywords', 'techniques',
    'research_question', 'finding', 'methodology_summary',
    'workflow_description', 'domain_knowledge',
  ]],
  ['task', [
    'task_objective', 'task_description', 'task_inputs', 'task_outputs',
    'inputs', 'input_from', 'parameters', 'skills', 'tools',
    'subtask_categories',
  ]],
  ['data', [
    'data_in', 'data_out', 'data_accessions', 'available_artifacts',
    'artifact_refs', 'script_refs', 'evidence_snippets', 'section_provenance',
  ]],
  ['execution', [
    'executable', 'execution_environment', 'execution_profile', 'run_command',
    'run_cwd', 'run_timeout_seconds', 'resumption_contract',
  ]],
  ['evaluation', [
    'evaluation_strategy', 'expected_outputs', 'expected_artifact_name',
    'landmark_outputs', 'linked_result_ids', 'review_questions',
    'reproducibility_tier', 'missing_information', 'uncertainty_notes', 'prov',
  ]],
]

const SCHEMA_FIELDS = new Set(CARD_SCHEMA.flatMap(([, fields]) => fields))

function AsbCard({ view }: { view: TaskView }) {
  const card = view.card
  const isRecord = typeof card === 'object' && card !== null && !Array.isArray(card)
  return (
    <div className="card">
      <div className="card-head">
        <span>ASB card · benchmark ground truth</span>
      </div>
      <div className="card-body">
        <div className="hint" style={{ marginBottom: 10 }}>
          The corpus card this task was generated from. The agent never saw it —
          see the verbatim prompt above for what it did see.
        </div>
        {!isRecord && card != null && (
          // A card that is valid JSON but not an object still renders verbatim.
          <JsonTree value={card} />
        )}
        {card == null && <AbsentLine reason={view.card_absent_reason} />}
        {isRecord && <CardFields card={card as Record<string, unknown>} />}
      </div>
    </div>
  )
}

function CardFields({ card }: { card: Record<string, unknown> }) {
  const extras = Object.keys(card).filter((k) => !SCHEMA_FIELDS.has(k)).sort()
  return (
    <div>
      {CARD_SCHEMA.map(([group, fields]) => (
        <div key={group} style={{ marginBottom: 10 }}>
          <div className="muted" style={{
            fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.6, margin: '8px 0 4px',
          }}>{group}</div>
          {fields.map((f) => <FieldRow key={f} name={f} present={f in card} value={card[f]} />)}
        </div>
      ))}
      {extras.length > 0 && (
        <div>
          <div className="muted" style={{
            fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.6, margin: '8px 0 4px',
          }}>outside the known schema</div>
          {extras.map((f) => <FieldRow key={f} name={f} present value={card[f]} />)}
        </div>
      )}
    </div>
  )
}

/** How many characters of JSON a value may occupy before it collapses. */
const INLINE_LIMIT = 160

function FieldRow({ name, present, value }: { name: string; present: boolean; value: unknown }) {
  return (
    <div style={{ display: 'flex', gap: 10, padding: '3px 0', alignItems: 'baseline', borderTop: '1px solid #ffffff0a' }}>
      <span className="mono" style={{ width: 210, flexShrink: 0, fontSize: 11, color: 'var(--text-dim)' }}>
        {name}
      </span>
      <div style={{ minWidth: 0, flex: 1, fontSize: 12 }}>
        {present ? <FieldValue value={value} /> : (
          <span className="muted" style={{ fontStyle: 'italic' }}>absent — field not present in card</span>
        )}
      </div>
    </div>
  )
}

function FieldValue({ value }: { value: unknown }) {
  if (value === null) return <span className="muted mono">null</span>
  if (typeof value === 'string') {
    if (value === '') return <span className="muted mono">"" (empty string)</span>
    return <span style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{value}</span>
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return <span className="mono">{String(value)}</span>
  }
  const json = JSON.stringify(value)
  if (json === '[]' || json === '{}') {
    return <span className="muted mono">{json} (empty)</span>
  }
  if (json.length <= INLINE_LIMIT) return <span className="mono" style={{ overflowWrap: 'anywhere' }}>{json}</span>
  return <JsonTree value={value} initialDepth={1} />
}

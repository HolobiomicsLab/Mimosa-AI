import { useState } from 'react'
import { api } from '../api'
import { useAsync } from '../hooks'
import type { McpHealth, SetupConfig, SetupInfo } from '../types'
import { Spinner } from '../ui'

/** Web version of the CLI onboarding steps 1–5: keys, models, workspace, MCP. */
export default function SetupPage() {
  const { data, loading, error, refetch } = useAsync<SetupInfo>(() => api.setup(), [])

  if (loading) return <Spinner label="Loading setup…" />
  if (error || !data) return <div className="empty"><div className="hint">Setup API unavailable: {error}</div></div>

  return (
    <>
      <div className="page-head">
        <h2>Setup</h2>
        <div className="hint mono">{data.config._config_path}</div>
      </div>
      <div className="page">
        {!data.bridge.available && (
          <div className="banner warn">
            Mimosa bridge unavailable — {data.bridge.error}. Config editing works, but
            refinement, mode suggestion and launching need the Mimosa venv.
          </div>
        )}
        <KeysCard info={data} onSaved={refetch} />
        <ModelsCard config={data.config} presets={data.presets} onSaved={refetch} />
        <WorkspaceCard config={data.config} onSaved={refetch} />
        <LearningCard config={data.config} onSaved={refetch} />
        <McpCard />
      </div>
    </>
  )
}

/** Shared PATCH-config helper with busy/feedback state per card. */
function useSaver(onSaved: () => void) {
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const save = (patch: Partial<SetupConfig>) => {
    setBusy(true)
    setMsg(null)
    api.saveConfig(patch)
      .then(() => { setMsg('✓ saved'); onSaved() })
      .catch((e: Error) => setMsg(e.message))
      .finally(() => setBusy(false))
  }
  return { busy, msg, save }
}

function SaveRow({ busy, msg, onSave }: { busy: boolean; msg: string | null; onSave: () => void }) {
  return (
    <div className="save-row">
      <button className="btn-gold" disabled={busy} onClick={onSave}>{busy ? 'Saving…' : 'Save'}</button>
      {msg && <span className={`hint ${msg.startsWith('✓') ? 'ok-text' : 'bad-text'}`}>{msg}</span>}
    </div>
  )
}

function KeysCard({ info, onSaved }: { info: SetupInfo; onSaved: () => void }) {
  const [name, setName] = useState(info.keys.keys[0]?.name ?? '')
  const [value, setValue] = useState('')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)

  const submit = () => {
    setBusy(true)
    setMsg(null)
    api.saveKey(name, value)
      .then((r) => { setMsg(`✓ saved to ${r.saved_to}`); setValue(''); onSaved() })
      .catch((e: Error) => setMsg(e.message))
      .finally(() => setBusy(false))
  }

  return (
    <div className="card setup-card">
      <div className="card-head">API keys · at least one provider required</div>
      <div className="card-body">
        <div className="key-grid">
          {info.keys.keys.map((k) => (
            <div key={k.name} className="key-row mono">
              <i className={`dot ${k.present ? 'ok' : 'off'}`} />
              <span>{k.name}</span>
              <span className="muted">{k.source ?? 'not set'}</span>
            </div>
          ))}
        </div>
        <div className="field-row" style={{ marginTop: 12 }}>
          <select className="input mono" value={name} onChange={(e) => setName(e.target.value)}>
            {info.keys.keys.map((k) => <option key={k.name}>{k.name}</option>)}
          </select>
          <input
            className="input mono" type="password" placeholder="paste key value"
            value={value} onChange={(e) => setValue(e.target.value)} style={{ flex: 1 }}
          />
          <button className="btn-gold" disabled={busy || !value.trim()} onClick={submit}>
            {busy ? 'Saving…' : 'Save key'}
          </button>
        </div>
        {msg && <div className={`hint ${msg.startsWith('✓') ? 'ok-text' : 'bad-text'}`} style={{ marginTop: 6 }}>{msg}</div>}
      </div>
    </div>
  )
}

const ROLES = [
  { key: 'orchestration', label: 'Orchestration', hint: 'plans goals and designs workflows (planner + workflow-crafting)' },
  { key: 'agent', label: 'Agent', hint: 'executes code inside workflow agents (also names run capsules)' },
  { key: 'judge', label: 'Judge', hint: 'evaluates run results' },
] as const

function ModelsCard({ config, presets, onSaved }: {
  config: SetupConfig
  presets: SetupInfo['presets']
  onSaved: () => void
}) {
  const [models, setModels] = useState({
    orchestration: config.planner_llm_model ?? '',
    agent: config.smolagent_model_id ?? '',
    judge: config.judge_model ?? '',
  })
  const { busy, msg, save } = useSaver(onSaved)

  const submit = () => save({
    planner_llm_model: models.orchestration,
    workflow_llm_model: models.orchestration,
    smolagent_model_id: models.agent,
    capsule_namer_model: models.agent,
    judge_model: models.judge,
  })

  return (
    <div className="card setup-card">
      <div className="card-head">models · litellm ids (provider/model)</div>
      <div className="card-body">
        {ROLES.map(({ key, label, hint }) => (
          <div className="field" key={key}>
            <label>{label} <span className="muted normal-case">— {hint}</span></label>
            <input
              className="input mono" value={models[key]} placeholder="Mimosa default"
              onChange={(e) => setModels({ ...models, [key]: e.target.value })}
            />
            <div className="pill-row">
              {presets[key]?.map((m) => (
                <button
                  key={m} className={`pill ${models[key] === m ? 'active' : ''}`}
                  onClick={() => setModels({ ...models, [key]: m })}
                >{m}</button>
              ))}
            </div>
          </div>
        ))}
        <SaveRow busy={busy} msg={msg} onSave={submit} />
      </div>
    </div>
  )
}

function WorkspaceCard({ config, onSaved }: { config: SetupConfig; onSaved: () => void }) {
  const [dir, setDir] = useState(config.workspace_dir ?? '')
  const { busy, msg, save } = useSaver(onSaved)
  return (
    <div className="card setup-card">
      <div className="card-head">
        <span>toolomics workspace · shared with MCP tool servers</span>
        <span className={config._workspace_exists ? 'ok-text' : 'bad-text'}>
          {config._workspace_exists ? 'directory found' : 'directory missing'}
        </span>
      </div>
      <div className="card-body">
        <div className="field">
          <label>Absolute path</label>
          <input className="input mono" value={dir} onChange={(e) => setDir(e.target.value)} />
        </div>
        <SaveRow busy={busy} msg={msg} onSave={() => save({ workspace_dir: dir })} />
      </div>
    </div>
  )
}

function LearningCard({ config, onSaved }: { config: SetupConfig; onSaved: () => void }) {
  const [draft, setDraft] = useState({
    reasoning_effort: config.reasoning_effort ?? 'medium',
    max_tokens: config.max_tokens ?? 8192,
    learned_score_threshold: config.learned_score_threshold ?? 0.92,
    max_learning_evolve_iterations: config.max_learning_evolve_iterations ?? 25,
    export_astra: config.export_astra ?? false,
  })
  const { busy, msg, save } = useSaver(onSaved)
  const set = (patch: Partial<typeof draft>) => setDraft({ ...draft, ...patch })

  return (
    <div className="card setup-card">
      <div className="card-head">learning · when a workflow counts as proven</div>
      <div className="card-body">
        <div className="grid-2">
          <div className="field">
            <label>Reasoning effort</label>
            <select
              className="input" value={draft.reasoning_effort}
              onChange={(e) => set({ reasoning_effort: e.target.value })}
            >
              {['minimal', 'low', 'medium', 'high'].map((v) => <option key={v}>{v}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Max tokens per LLM call</label>
            <input
              className="input mono" type="number" min={256} value={draft.max_tokens}
              onChange={(e) => set({ max_tokens: Number(e.target.value) })}
            />
          </div>
          <div className="field">
            <label>Proven-workflow score threshold</label>
            <input
              className="input mono" type="number" step={0.01} min={0} max={1}
              value={draft.learned_score_threshold}
              onChange={(e) => set({ learned_score_threshold: Number(e.target.value) })}
            />
          </div>
          <div className="field">
            <label>Max evolution iterations</label>
            <input
              className="input mono" type="number" min={1} value={draft.max_learning_evolve_iterations}
              onChange={(e) => set({ max_learning_evolve_iterations: Number(e.target.value) })}
            />
          </div>
        </div>
        <label className="check-row">
          <input
            type="checkbox" checked={draft.export_astra}
            onChange={(e) => set({ export_astra: e.target.checked })}
          />
          Export the best run as an ASTRA capsule
        </label>
        <SaveRow busy={busy} msg={msg} onSave={() => save(draft)} />
      </div>
    </div>
  )
}

function McpCard() {
  const [health, setHealth] = useState<McpHealth | null>(null)
  const [busy, setBusy] = useState(false)
  const scan = () => {
    setBusy(true)
    api.mcpHealth().then(setHealth).catch(() => setHealth(null)).finally(() => setBusy(false))
  }
  return (
    <div className="card setup-card">
      <div className="card-head">
        <span>toolomics MCP servers</span>
        <button disabled={busy} onClick={scan}>{busy ? 'Scanning…' : 'Scan ports'}</button>
      </div>
      <div className="card-body">
        {!health && <div className="hint">Probes the configured discovery port range (default 5000–5200).</div>}
        {health && health.reachable && (
          <div className="pill-row">
            {health.open_ports.map((p) => (
              <span key={p.port} className="pill active">:{p.port}</span>
            ))}
          </div>
        )}
        {health && !health.reachable && <div className="bad-text hint">{health.note}</div>}
        {health && (
          <div className="hint" style={{ marginTop: 8 }}>
            {health.scanned} ports probed. Tool availability (the <code>execute_command</code> shell
            agents need) is verified again at launch time.
          </div>
        )}
      </div>
    </div>
  )
}

import { useEffect, useRef, useState } from 'react'
import { Link, useLocation, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import { useAsync } from '../hooks'
import type { ClassifyResult, LaunchInfo, ObjectiveHistoryEntry, RunMode } from '../types'
import { fmtBytes, Spinner } from '../ui'

const MIN_OBJECTIVE_CHARS = 10
const MAX_REFINE_ROUNDS = 5
const POLL_MS = 2000

/** Web version of CLI onboarding steps 6–9: objective → refine → mode → launch. */
export default function LaunchPage() {
  const launches = useAsync<LaunchInfo[]>(() => api.launches(), [])
  const [params] = useSearchParams()
  const location = useLocation()
  const focusId = params.get('focus')
  const [activeId, setActiveId] = useState<string | null>(focusId)

  // ?focus=<id> deep-links straight to a launch's monitor (used by failure toasts).
  useEffect(() => { if (focusId) setActiveId(focusId) }, [focusId, location.key])

  const onLaunched = (id: string) => { setActiveId(id); launches.refetch() }
  const others = (launches.data ?? []).filter((l) => l.id !== activeId)

  return (
    <>
      <div className="page-head">
        <h2>New run</h2>
        <div className="hint">Refine an objective, pick a mode, and launch Mimosa.</div>
      </div>
      <div className="page wide">
        {others.length > 0 && (
          <div className="pill-row" style={{ marginBottom: 16 }}>
            {others.map((l) => (
              <button key={l.id} className="pill" onClick={() => setActiveId(l.id)}>
                {l.running ? '● ' : ''}{l.mode} · {l.objective.slice(0, 40)}…
              </button>
            ))}
          </div>
        )}
        {activeId
          ? <LaunchMonitor id={activeId} onBack={() => { setActiveId(null); launches.refetch() }} />
          : <Wizard onLaunched={onLaunched} />}
      </div>
    </>
  )
}

type Step = 'objective' | 'refine' | 'launch'
const STEP_LABELS: [Step, string][] = [
  ['objective', '1 · Objective'], ['refine', '2 · Refine'], ['launch', '3 · Launch'],
]

function Wizard({ onLaunched }: { onLaunched: (id: string) => void }) {
  const [step, setStep] = useState<Step>('objective')
  const [objective, setObjective] = useState('')

  return (
    <div>
      <div className="wizard-steps">
        {STEP_LABELS.map(([id, label]) => (
          <span key={id} className={`ws-step ${step === id ? 'active' : ''}`}>{label}</span>
        ))}
      </div>
      {step === 'objective' && (
        <ObjectiveStep objective={objective} setObjective={setObjective} onNext={() => setStep('refine')} />
      )}
      {step === 'refine' && (
        <RefineStep
          objective={objective}
          onDone={(refined) => { setObjective(refined); setStep('launch') }}
        />
      )}
      {step === 'launch' && (
        <LaunchStep objective={objective} onBack={() => setStep('objective')} onLaunched={onLaunched} />
      )}
    </div>
  )
}

function ObjectiveStep({ objective, setObjective, onNext }: {
  objective: string
  setObjective: (v: string) => void
  onNext: () => void
}) {
  const tooShort = objective.trim().length < MIN_OBJECTIVE_CHARS
  const [uploading, setUploading] = useState(false)
  const history = useAsync(() => api.objectiveHistory(), [])
  const entries: ObjectiveHistoryEntry[] = history.data?.result?.entries ?? []
  return (
    <div className="card setup-card">
      <div className="card-head">what should Mimosa work on?</div>
      <div className="card-body objective-split">
        <div className="objective-editor">
          <textarea
            className="input" rows={9} autoFocus
            placeholder="e.g. Analyse the metabolomics CSV in the workspace and report which pathways separate treated from control samples…"
            value={objective} onChange={(e) => setObjective(e.target.value)}
          />
          <div className="save-row">
            <button className="btn-gold" disabled={tooShort || uploading} onClick={onNext}>Continue</button>
            {tooShort && <span className="hint">at least {MIN_OBJECTIVE_CHARS} characters</span>}
          </div>
          <WorkspaceFilesRow onBusy={setUploading} />
        </div>
        <div className="objective-history">
          <div className="field" style={{ marginBottom: 8 }}>
            <label>Past objectives</label>
          </div>
          {history.loading && <Spinner label="Loading history…" />}
          {!history.loading && entries.length === 0 && (
            <div className="hint">No past objectives yet — launched runs are recorded here.</div>
          )}
          {entries.map((e, i) => (
            <button
              key={i} type="button" className="objective-history-item"
              title="Click to reuse this objective"
              onClick={() => setObjective(e.objective)}
            >
              <span className="objective-history-text">{e.objective}</span>
              <span className="hint">
                {e.timestamp ?? ''}{e.mode ? ` · ${String(e.mode).toUpperCase()}` : ''}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

/** Upload input files into the live workspace and list what's already there.
 *  Everything shown is what the next launch snapshots as its initial state.
 *  ``onBusy`` gates the wizard's Continue button while an upload streams. */
function WorkspaceFilesRow({ onBusy }: { onBusy: (busy: boolean) => void }) {
  const listing = useAsync(() => api.workspaceFiles('live'), [])
  const inputRef = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const upload = (picked: FileList | null) => {
    if (!picked?.length) return
    setBusy(true)
    onBusy(true)
    setError(null)
    api.uploadWorkspaceFiles(Array.from(picked))
      .catch((e: Error) => setError(e.message))
      .finally(() => {
        listing.refetch() // also on failure — show what actually persisted
        setBusy(false)
        onBusy(false)
        if (inputRef.current) inputRef.current.value = ''
      })
  }

  const remove = (path: string) => {
    setError(null)
    api.deleteWorkspaceFile(path)
      .catch((e: Error) => setError(e.message))
      .finally(() => listing.refetch())
  }

  const files = listing.data?.files ?? []
  return (
    <div className="field" style={{ marginTop: 14 }}>
      <label>Files for this run</label>
      <div className="pill-row" style={{ marginTop: 4 }}>
        <button className="pill" disabled={busy} onClick={() => inputRef.current?.click()}>
          {busy ? 'Uploading…' : '＋ Add files'}
        </button>
        {files.map((f) => (
          <span key={f.path} className="pill file-chip" title={f.path}>
            {f.path} · {fmtBytes(f.size)}
            <button className="chip-x" title="Remove from workspace" onClick={() => remove(f.path)}>×</button>
          </span>
        ))}
      </div>
      <input
        ref={inputRef} type="file" multiple style={{ display: 'none' }}
        onChange={(e) => upload(e.target.files)}
      />
      {error && <div className="bad-text hint" style={{ marginTop: 6 }}>{error}</div>}
      <div className="hint" style={{ marginTop: 6 }}>
        {files.length === 0 && !listing.loading
          ? 'The workspace is empty — the run starts blank unless you add input files. '
          : 'Mimosa reads these as the run\'s starting workspace. '}
        {listing.data && <span className="mono">{listing.data.root}</span>}
      </div>
    </div>
  )
}

/** Clarifier loop: the planner LLM either asks one question or proposes a refinement. */
function RefineStep({ objective, onDone }: { objective: string; onDone: (refined: string) => void }) {
  const [history, setHistory] = useState<{ question: string; answer: string }[]>([])
  const [answer, setAnswer] = useState('')
  const [refined, setRefined] = useState<string | null>(null)
  const round = useAsync(() => api.refine(objective, history), [history])

  const res = round.data?.result
  const failed = round.data ? !round.data.ok : false
  const asksQuestion = !!res && !res.is_clear && !!res.question && history.length < MAX_REFINE_ROUNDS

  useEffect(() => {
    if (res && (res.is_clear || history.length >= MAX_REFINE_ROUNDS)) setRefined(res.refined_prompt)
  }, [res, history.length])

  const submitAnswer = () => {
    setHistory([...history, { question: res?.question ?? '', answer }])
    setAnswer('')
  }

  return (
    <div className="card setup-card">
      <div className="card-head">
        <span>refining the objective · round {history.length + 1}/{MAX_REFINE_ROUNDS}</span>
        <button onClick={() => onDone(objective)}>Skip refinement</button>
      </div>
      <div className="card-body">
        <div className="hint mono" style={{ marginBottom: 12 }}>{objective}</div>
        {round.loading && <Spinner label="Asking the planner model…" />}
        {(failed || round.error) && (
          <div className="bad-text hint" style={{ marginBottom: 10 }}>
            Refinement unavailable — {round.data?.error ?? round.error}. You can continue with your
            objective as written.
          </div>
        )}
        {res?.degraded && (
          <div className="hint" style={{ marginBottom: 10 }}>
            Auto-refinement wasn't available this time — your objective is shown below as written;
            edit it if you'd like before continuing.
          </div>
        )}
        {asksQuestion && refined === null && (
          <div className="field">
            <label>The planner asks</label>
            <p style={{ margin: '2px 0 8px' }}>{res.question}</p>
            <input
              className="input" value={answer} autoFocus placeholder="your answer"
              onChange={(e) => setAnswer(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && answer.trim() && submitAnswer()}
            />
            <div className="save-row">
              <button className="btn-gold" disabled={!answer.trim()} onClick={submitAnswer}>Answer</button>
            </div>
          </div>
        )}
        {refined !== null && (
          <div className="field">
            <label>Refined objective — edit freely</label>
            <textarea
              className="input" rows={5} value={refined}
              onChange={(e) => setRefined(e.target.value)}
            />
            <div className="save-row">
              <button className="btn-gold" onClick={() => onDone(refined)}>Use this objective</button>
              <button onClick={() => onDone(objective)}>Keep my original</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function LaunchStep({ objective, onBack, onLaunched }: {
  objective: string
  onBack: () => void
  onLaunched: (id: string) => void
}) {
  const suggestion = useAsync<ClassifyResult>(() => api.classify(objective), [objective])
  const [mode, setMode] = useState<RunMode>('task')
  const [modeTouched, setModeTouched] = useState(false)
  const [learn, setLearn] = useState(false)
  const [judge, setJudge] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const cls = suggestion.data?.result
  // Adopt the suggestion only until the user picks a mode themselves, so a
  // late classify response can't snap their choice back.
  useEffect(() => { if (cls && !modeTouched) setMode(cls.mode) }, [cls, modeTouched])
  const pickMode = (m: RunMode) => { setMode(m); setModeTouched(true) }

  const launch = () => {
    setBusy(true)
    setError(null)
    api.launch({ objective, mode, learn: mode === 'task' ? learn : false, judge })
      .then((l) => onLaunched(l.id))
      .catch((e: Error) => { setError(e.message); setBusy(false) })
  }

  return (
    <div className="card setup-card">
      <div className="card-head">
        <span>mode &amp; launch</span>
        <button onClick={onBack}>← Edit objective</button>
      </div>
      <div className="card-body">
        <div className="hint mono" style={{ marginBottom: 14 }}>{objective}</div>

        {suggestion.loading && <Spinner label="Classifying goal vs task…" />}
        {cls && (
          <div className="hint" style={{ marginBottom: 10 }}>
            Suggested <b className="ok-text">{cls.mode} mode</b>
            {cls.confidence != null && <> · confidence {(cls.confidence * 100).toFixed(0)}%</>}
            {cls.reasoning && <> — {cls.reasoning}</>}
          </div>
        )}

        <div className="field">
          <label>Mode</label>
          <div className="seg">
            <button className={mode === 'task' ? 'active' : ''} onClick={() => pickMode('task')}>
              Task — one workflow
            </button>
            <button className={mode === 'goal' ? 'active' : ''} onClick={() => pickMode('goal')}>
              Goal — plan of tasks
            </button>
          </div>
        </div>

        {mode === 'task' ? (
          <label className="check-row">
            <input type="checkbox" checked={learn} onChange={(e) => setLearn(e.target.checked)} />
            Learning mode — evolve the workflow over iterations until it is proven
          </label>
        ) : (
          <div className="hint" style={{ margin: '4px 0 10px' }}>
            Goal mode always evolves each task's workflow while executing the plan.
          </div>
        )}
        <label className="check-row">
          <input type="checkbox" checked={judge} onChange={(e) => setJudge(e.target.checked)} />
          Judge — score results with the evaluation model
        </label>

        <div className="save-row">
          <button className="btn-gold" disabled={busy} onClick={launch}>
            {busy ? 'Launching…' : '✦ Launch Mimosa'}
          </button>
          {error && <span className="bad-text hint">{error}</span>}
        </div>
      </div>
    </div>
  )
}

/** Poll a launch: status, plain-text log tail, stop/interrupt, failure hints. */
function LaunchMonitor({ id, onBack }: { id: string; onBack: () => void }) {
  const [info, setInfo] = useState<LaunchInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const logRef = useRef<HTMLPreElement>(null)

  useEffect(() => {
    let alive = true
    let timer: ReturnType<typeof setTimeout>
    const tick = () => {
      api.launchStatus(id)
        .then((s) => {
          if (!alive) return
          setInfo(s)
          if (s.running) timer = setTimeout(tick, POLL_MS)
        })
        .catch((e: Error) => alive && setError(e.message))
    }
    tick()
    return () => { alive = false; clearTimeout(timer) }
  }, [id])

  useEffect(() => {
    const el = logRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [info?.log_tail])

  // "Focus the log" on failure: bring the log panel into view. Show the hint on
  // a hard failure, or a live run whose crafting LLM is failing — not on success.
  const showFail = !!info && (info.failed || (info.running && info.failure_hint === 'workflow_generation'))
  const failKey = showFail ? (info?.failure_hint ?? 'crash') : null
  useEffect(() => {
    if (failKey) logRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [failKey])

  const stop = () => {
    if (!window.confirm('Stop this run?')) return
    api.cancelLaunch(id).then(setInfo).catch((e: Error) => setError(e.message))
  }

  if (error) return <div className="empty"><div className="hint">{error}</div></div>
  if (!info) return <Spinner label="Loading launch…" />

  const outcome = info.running ? null : info.returncode === 0 ? 'finished' : `failed (exit ${info.returncode})`
  return (
    <div className="card setup-card">
      <div className="card-head">
        <span>
          <span className={`badge ${info.running ? 'running' : info.returncode === 0 ? 'completed' : 'crashed'}`}>
            <span className="tick" style={{ background: 'currentColor' }} />
            {info.running ? 'running' : outcome}
          </span>
          <span style={{ marginLeft: 10 }} className="mono">{info.mode} mode{info.learn ? ' · learning' : ''}</span>
        </span>
        <span>
          {info.running && (
            <button className="btn-danger" onClick={stop} style={{ marginRight: 8 }}>
              ■ Stop run
            </button>
          )}
          <button onClick={onBack}>← New run</button>
        </span>
      </div>
      <div className="card-body">
        <p style={{ margin: '0 0 10px', whiteSpace: 'pre-wrap' }}>{info.objective}</p>
        {failKey && <FailureBanner info={info} />}
        <pre className={`code log-tail ${failKey ? 'failed' : ''}`} ref={logRef}>
          {info.log_tail || 'Waiting for output…'}
        </pre>
        <div className="hint" style={{ marginTop: 8 }}>
          The run appears in the sidebar as soon as it writes its first artifacts — click it there
          to watch results, replay and workflow live.
        </div>
      </div>
    </div>
  )
}

function FailureBanner({ info }: { info: LaunchInfo }) {
  if (info.failure_hint === 'workflow_generation') {
    return (
      <div className="banner error">
        The LLM failed to generate a valid workflow{info.running ? ' (retrying)' : ''}. Certain
        models — especially smaller ones — often cannot produce valid workflow code. Consider a
        stronger orchestration model in <Link to="/setup">Setup → Models</Link>.
      </div>
    )
  }
  return (
    <div className="banner error">
      The run failed{info.error_line ? <> — <code>{info.error_line}</code></> : null}. See the log below.
    </div>
  )
}

import { useState } from 'react'
import { api } from '../api'
import { useAsync } from '../hooks'
import type { CallDetail, StepDetail } from '../types'
import { Spinner, fmtCost, fmtDuration, fmtEpoch } from '../ui'

function agentHue(name: string): number {
  let h = 0
  for (const c of name) h = (h * 31 + c.charCodeAt(0)) % 360
  return h
}
const agentColor = (name: string) => `hsl(${agentHue(name)}, 55%, 60%)`

type Sel =
  | { kind: 'step'; agent: string; index: number }
  | { kind: 'call'; name: string }
  | null

export default function MemoryReplay({ runId }: { runId: string }) {
  const timeline = useAsync(() => api.timeline(runId), [runId])
  const mem = useAsync(() => api.memory(runId), [runId])
  const [tab, setTab] = useState<'steps' | 'calls'>('steps')
  const [sel, setSel] = useState<Sel>(null)

  if (timeline.loading || mem.loading) return <Spinner />
  if (timeline.error) return <div className="hint">No agent memory for this run.</div>

  const steps = timeline.data?.steps || []
  const calls = mem.data?.calls || []

  return (
    <div className="split" style={{ ['--split-list' as string]: '340px' }}>
      <div className="card">
        <div className="card-head" style={{ padding: 8 }}>
          <div className="pill-row">
            <button className={`pill ${tab === 'steps' ? 'active' : ''}`} onClick={() => setTab('steps')}>
              agent steps · {steps.length}
            </button>
            <button className={`pill ${tab === 'calls' ? 'active' : ''}`} onClick={() => setTab('calls')}>
              LLM calls · {calls.length}
            </button>
          </div>
        </div>
        <div style={{ maxHeight: '68vh', overflowY: 'auto' }}>
          {tab === 'steps' && steps.map((s) => {
            const active = sel?.kind === 'step' && sel.agent === s.agent && sel.index === s.index
            return (
              <button
                key={s.order}
                className="step-row"
                data-active={active}
                onClick={() => setSel({ kind: 'step', agent: s.agent, index: s.index })}
              >
                <span className="step-bar" style={{ background: agentColor(s.agent) }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="step-top">
                    <span style={{ color: agentColor(s.agent), fontWeight: 600 }}>{s.agent}</span>
                    <span className="muted">#{s.step_number}</span>
                    {s.is_final_answer && <span className="tagpill ok">final</span>}
                    {s.error_type && <span className="tagpill bad">error</span>}
                  </div>
                  <div className="step-code">{s.code || s.output_text || '(no code)'}</div>
                  <div className="step-foot muted">
                    {fmtDuration(s.duration)} · {s.tokens?.total_tokens ?? '—'} tok
                  </div>
                </div>
              </button>
            )
          })}
          {tab === 'calls' && calls.map((c) => {
            const active = sel?.kind === 'call' && sel.name === c.name
            return (
              <button
                key={c.name}
                className="step-row"
                data-active={active}
                onClick={() => setSel({ kind: 'call', name: c.name })}
              >
                <span className="step-bar" style={{ background: '#5db8f0' }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="step-top"><span style={{ fontWeight: 600 }}>{c.name}</span></div>
                  <div className="step-foot muted">{fmtCost(c.cost_usd)} · {c.total_tokens ?? '—'} tok · {c.model}</div>
                </div>
              </button>
            )
          })}
        </div>
      </div>

      <div>
        {!sel && <div className="hint" style={{ padding: 20 }}>Select a step or LLM call to inspect its inputs, code and outputs.</div>}
        {sel?.kind === 'step' && <StepView runId={runId} agent={sel.agent} index={sel.index} />}
        {sel?.kind === 'call' && <CallView runId={runId} name={sel.name} />}
      </div>
      <style>{STYLE}</style>
    </div>
  )
}

function Block({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card" style={{ marginBottom: 14 }}>
      <div className="card-head">{title}</div>
      <div className="card-body">{children}</div>
    </div>
  )
}

function StepView({ runId, agent, index }: { runId: string; agent: string; index: number }) {
  const { data, loading } = useAsync<StepDetail>(() => api.step(runId, agent, index), [runId, agent, index])
  if (loading || !data) return <Spinner />
  return (
    <div>
      <div className="stats" style={{ display: 'flex', gap: 18, marginBottom: 14, flexWrap: 'wrap' }}>
        <div className="stat"><b style={{ color: agentColor(agent) }}>{agent}</b><span>agent</span></div>
        <div className="stat"><b>#{data.step_number}</b><span>step</span></div>
        <div className="stat"><b>{fmtDuration(data.timing?.duration)}</b><span>duration</span></div>
        <div className="stat"><b>{data.tokens?.total_tokens ?? '—'}</b><span>tokens</span></div>
        <div className="stat"><b>{fmtEpoch(data.timing?.start_time)}</b><span>start</span></div>
      </div>
      {data.error && (
        <Block title={`error · ${data.error.type || ''}`}>
          <pre className="code" style={{ color: '#ef9a9a' }}>{data.error.message}</pre>
        </Block>
      )}
      {data.code && <Block title="executed code"><pre className="code tight">{data.code}</pre></Block>}
      {data.output_text && <Block title="reasoning / model output"><pre className="code">{data.output_text}</pre></Block>}
      {data.observations && <Block title="observations"><pre className="code">{data.observations}</pre></Block>}
      {data.action_output && <Block title="action output"><pre className="code">{data.action_output}</pre></Block>}
      {data.input_messages.length > 0 && (
        <details>
          <summary className="hint" style={{ cursor: 'pointer', padding: '6px 0' }}>
            input messages ({data.input_messages.length})
          </summary>
          {data.input_messages.map((m, i) => (
            <Block key={i} title={m.role}><pre className="code">{m.content}</pre></Block>
          ))}
        </details>
      )}
    </div>
  )
}

function CallView({ runId, name }: { runId: string; name: string }) {
  const { data, loading } = useAsync<CallDetail>(() => api.call(runId, name), [runId, name])
  if (loading || !data) return <Spinner />
  return (
    <div>
      <div className="stats" style={{ display: 'flex', gap: 18, marginBottom: 14, flexWrap: 'wrap' }}>
        <div className="stat"><b>{data.model}</b><span>model</span></div>
        <div className="stat"><b>{fmtCost(data.cost_usd)}</b><span>cost</span></div>
        <div className="stat"><b>{data.tokens.total ?? '—'}</b><span>tokens</span></div>
        <div className="stat"><b>{data.temperature ?? '—'}</b><span>temp</span></div>
        <div className="stat"><b>{data.reasoning_effort || '—'}</b><span>effort</span></div>
      </div>
      {data.messages.map((m, i) => (
        <Block key={i} title={`prompt · ${m.role}`}><pre className="code">{m.content}</pre></Block>
      ))}
      <Block title="response"><pre className="code">{data.response}</pre></Block>
    </div>
  )
}

const STYLE = `
.step-row { display: flex; gap: 0; width: 100%; text-align: left; background: transparent;
  border: none; border-bottom: 1px solid var(--border); border-radius: 0; padding: 0; }
.step-row:hover { background: var(--panel-2); }
.step-row[data-active="true"] { background: #f2c14e10; }
.step-bar { width: 3px; align-self: stretch; flex: none; }
.step-row > div { padding: 9px 11px; }
.step-top { display: flex; align-items: center; gap: 7px; font-size: 12px; margin-bottom: 3px; }
.step-code { font-family: var(--mono); font-size: 11px; color: var(--text-dim);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.step-foot { font-size: 10.5px; margin-top: 3px; }
.tagpill { font-size: 9px; text-transform: uppercase; letter-spacing: 0.4px; padding: 1px 5px; border-radius: 4px; }
.tagpill.ok { color: var(--ok); background: #4ec98a1e; }
.tagpill.bad { color: var(--bad); background: #ef6a6a1e; }
`

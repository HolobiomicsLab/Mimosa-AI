import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { useLive } from '../hooks'
import type { AstraDecision, LiveEvent } from '../types'

/**
 * Live activity feed for one run: filesystem events streamed over the /live
 * WebSocket, translated into a human narrative — agent steps landing, the
 * gradient being written, and (the decision layer) ASTRA capsule updates
 * diffed decision-by-decision so each recorded choice appears as its own
 * feed line the moment the exporter writes it.
 */

interface FeedItem {
  ts: string
  kind: 'step' | 'call' | 'artifact' | 'decision' | 'capsule'
  text: string
}

const MAX_ITEMS = 200

function label(e: LiveEvent): { kind: FeedItem['kind']; text: string } | null {
  switch (e.type) {
    case 'workflow_crafted':
      return { kind: 'artifact', text: 'workflow genotype written' }
    case 'step_appended':
      return {
        kind: 'step',
        text: `agent step trace updated — ${e.filename.replace(/^task_|\.json$/g, '')}`,
      }
    case 'llm_call_logged':
      return { kind: 'call', text: `LLM call logged — ${e.filename.replace(/\.json$/, '')}` }
    case 'iteration_complete':
      return { kind: 'artifact', text: 'iteration metrics written (score available)' }
    case 'execution_complete':
      return { kind: 'artifact', text: 'execution finished — state_result.json written' }
    case 'gradient_updated':
      return { kind: 'artifact', text: 'textual gradient written — feedback for the next mutation' }
    case 'evaluation_updated':
      return { kind: 'artifact', text: 'verifier evaluation written' }
    case 'astra_updated':
      return { kind: 'capsule', text: 'ASTRA capsule updated' }
    case 'evaluation_capsule_updated':
      return { kind: 'capsule', text: 'independent evaluation capsule updated (asb_eval)' }
    default:
      return null
  }
}

const KIND_COLOR: Record<FeedItem['kind'], string> = {
  step: 'var(--accent)',
  call: 'var(--text-dim)',
  artifact: 'var(--warn)',
  decision: 'var(--gold)',
  capsule: 'var(--ok)',
}

export default function LiveFeed({ runId, onActivity }: {
  runId: string
  onActivity?: (e: LiveEvent) => void
}) {
  const [items, setItems] = useState<FeedItem[]>([])
  const decisions = useRef<Map<string, string> | null>(null)
  const onActivityRef = useRef(onActivity)
  onActivityRef.current = onActivity

  const push = useCallback((entries: Omit<FeedItem, 'ts'>[]) => {
    if (!entries.length) return
    const ts = new Date().toLocaleTimeString()
    setItems((cur) => [...entries.map((e) => ({ ...e, ts })), ...cur].slice(0, MAX_ITEMS))
  }, [])

  /** Diff the capsule's decision layer against the last snapshot. */
  const diffDecisions = useCallback((silent: boolean) => {
    api.provenance(runId)
      .then((view) => {
        const next = new Map<string, string>()
        const decs: [string, AstraDecision][] = Object.entries(view.astra?.decisions ?? {})
        for (const [slug, d] of decs) {
          const chosen = d.default != null
            ? (d.options?.[String(d.default)]?.label ?? String(d.default))
            : '—'
          next.set(slug, `${d.label ?? slug} — chose ${chosen}`)
        }
        const prev = decisions.current
        decisions.current = next
        if (silent || prev == null) return
        const fresh: Omit<FeedItem, 'ts'>[] = []
        for (const [slug, line] of next) {
          if (!prev.has(slug)) fresh.push({ kind: 'decision', text: `decision recorded: ${line}` })
          else if (prev.get(slug) !== line) fresh.push({ kind: 'decision', text: `decision changed: ${line}` })
        }
        push(fresh)
      })
      .catch(() => { /* capsule may not exist yet — that's normal mid-run */ })
  }, [runId, push])

  // Baseline snapshot so a pre-existing capsule doesn't flood the feed.
  useEffect(() => {
    decisions.current = null
    setItems([])
    diffDecisions(true)
  }, [runId, diffDecisions])

  const { connected } = useLive((e) => {
    // Events without a run id (eval capsules) may still concern this run.
    if (e.run_id && e.run_id !== runId) return
    const l = label(e)
    if (!l) return
    push([l])
    onActivityRef.current?.(e)
    if (e.type === 'astra_updated' || e.type === 'evaluation_capsule_updated') diffDecisions(false)
  })

  return (
    <div className="card">
      <div className="card-head">
        <span>live activity</span>
        <span className={`live-dot ${connected ? 'on' : ''}`}><i />{connected ? 'watching' : 'offline'}</span>
      </div>
      <div className="card-body" style={{ maxHeight: 260, overflowY: 'auto' }}>
        {items.length === 0 && (
          <div className="hint">
            Waiting for activity — events appear here the moment Mimosa writes an
            artifact: agent steps, the textual gradient, evaluations, and each
            ASTRA decision as the capsule records it.
          </div>
        )}
        {items.map((it, i) => (
          <div key={`${it.ts}-${i}`} style={{ display: 'flex', gap: 10, padding: '3px 0', alignItems: 'baseline' }}>
            <span className="mono" style={{ color: 'var(--text-faint)', fontSize: 11, flexShrink: 0 }}>{it.ts}</span>
            <span className="mono" style={{ color: KIND_COLOR[it.kind], fontSize: 11, flexShrink: 0 }}>●</span>
            <span style={{ fontSize: 12, color: it.kind === 'decision' ? 'var(--gold)' : 'var(--text)' }}>
              {it.text}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

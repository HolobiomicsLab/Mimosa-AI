import { type ReactNode, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAsync } from '../hooks'
import type { EvolutionKind, EvolutionNode, FamilyEvolution } from '../types'
import {
  KindTag, ScoreChip, Spinner, StatusBadge,
  fmtCost, fmtDuration, fmtScore, scoreColor, shortId,
} from '../ui'
import RewardChart from './RewardChart'

// ── replay geometry & timing ────────────────────────────────────────────────

const NODE_R = 16
const GAP_X = 96
const GAP_Y = 120
const PAD_X = 48
const PAD_Y = 44
const TICK_MS = 1600
const SPEEDS = [0.5, 1, 2, 4]
const LATE = Number.MAX_SAFE_INTEGER // sort key for nodes without an iteration

const KIND_FILL: Record<string, string> = {
  seed: '#6e7891',
  mutation: 'var(--run)',
  crossover: '#c78bf0',
}
const kindFill = (k: EvolutionKind) => KIND_FILL[k ?? 'seed'] ?? KIND_FILL.seed

/** Reveal order: iteration first, then created_at (id as tiebreak). */
function orderNodes(nodes: EvolutionNode[]): EvolutionNode[] {
  return [...nodes].sort((a, b) => {
    const ia = a.iteration ?? LATE
    const ib = b.iteration ?? LATE
    if (ia !== ib) return ia - ib
    const ka = a.created_at ?? a.id
    const kb = b.created_at ?? b.id
    return ka < kb ? -1 : ka > kb ? 1 : a.id.localeCompare(b.id)
  })
}

interface Geo {
  pos: Map<string, { x: number; y: number }>
  minX: number
  width: number
  height: number
}

/** Rows by iteration, siblings evenly spaced and centred per row. */
function layoutNodes(ordered: EvolutionNode[]): Geo {
  const rows = new Map<number, EvolutionNode[]>()
  for (const nd of ordered) {
    const it = nd.iteration ?? LATE
    const row = rows.get(it) ?? []
    row.push(nd)
    rows.set(it, row)
  }
  const keys = [...rows.keys()].sort((a, b) => a - b)
  const pos = new Map<string, { x: number; y: number }>()
  let half = GAP_X / 2
  keys.forEach((key, r) => {
    const sibs = rows.get(key)!
    sibs.forEach((nd, i) => pos.set(nd.id, {
      x: i * GAP_X - ((sibs.length - 1) * GAP_X) / 2,
      y: r * GAP_Y + PAD_Y,
    }))
    half = Math.max(half, ((sibs.length - 1) * GAP_X) / 2)
  })
  return {
    pos,
    minX: -(half + PAD_X),
    width: 2 * (half + PAD_X),
    height: (keys.length - 1) * GAP_Y + 2 * PAD_Y,
  }
}

/** Vertical S-curve between two node rims. */
function edgePath(a: { x: number; y: number }, b: { x: number; y: number }): string {
  const y0 = a.y + NODE_R
  const y1 = b.y - NODE_R
  const my = (y0 + y1) / 2
  return `M ${a.x} ${y0} C ${a.x} ${my}, ${b.x} ${my}, ${b.x} ${y1}`
}

/** The node plus every transitive parent — the glowing lineage. */
function ancestorIds(startId: string, byId: Map<string, EvolutionNode>): Set<string> {
  const seen = new Set<string>()
  const stack = [startId]
  while (stack.length) {
    const id = stack.pop()!
    if (seen.has(id)) continue
    seen.add(id)
    for (const p of byId.get(id)?.parents ?? []) stack.push(p)
  }
  return seen
}

/** Index of the highest-scored node among the first k revealed; -1 when none scored. */
function bestRevealedIndex(ordered: EvolutionNode[], k: number): number {
  let best = -1
  for (let i = 0; i < k; i++) {
    const s = ordered[i].score
    if (s != null && (best === -1 || s > (ordered[best].score ?? -Infinity))) best = i
  }
  return best
}

/** Score delta vs the first parent that carries a score. */
function parentDelta(node: EvolutionNode, byId: Map<string, EvolutionNode>): number | null {
  if (node.score == null) return null
  for (const pid of node.parents) {
    const p = byId.get(pid)
    if (p?.score != null) return node.score - p.score
  }
  return null
}

function selectionLine(sel: NonNullable<EvolutionNode['selection']>): string {
  const parts: string[] = []
  if (sel.improvement_type) parts.push(`selected via ${sel.improvement_type}`)
  if (sel.is_validated != null) parts.push(sel.is_validated ? 'validated' : 'not validated')
  if (sel.confidence != null) parts.push(`conf ${sel.confidence.toFixed(2)}`)
  return parts.join(' · ')
}

// ── component ───────────────────────────────────────────────────────────────

/**
 * Evolution tab: the family's self-evolving search, replayed as an animated
 * SVG lineage. Nodes reveal one per tick, the best-so-far badge chases the
 * frontier, and the info panel narrates the current (or selected) run.
 */
export default function EvolutionReplay({ runId }: { runId: string }) {
  const fam = useAsync(() => api.evolution(runId), [runId])
  const series = useAsync(() => api.series(runId), [runId])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      {fam.loading
        ? <Spinner />
        : fam.data && fam.data.nodes.length > 1
          ? <Replay fam={fam.data} />
          : <SingleNodeCard error={fam.error} />}
      <div className="card">
        <div className="card-head">reward &amp; cost across iterations</div>
        <div className="card-body">
          {series.loading ? <Spinner /> : series.data ? <RewardChart series={series.data} /> : null}
        </div>
      </div>
      <style>{STYLE}</style>
    </div>
  )
}

function SingleNodeCard({ error }: { error: string | null }) {
  return (
    <div className="card">
      <div className="card-head">evolution replay</div>
      <div className="card-body">
        <div className="hint">
          {error
            ? 'No evolution data for this run.'
            : <>This run is a single <b>seed</b> with no offspring yet — the animated replay
              lights up once learning-mode evolution produces mutations and crossovers.</>}
        </div>
      </div>
    </div>
  )
}

interface Tip { x: number; y: number; node: EvolutionNode }

function Replay({ fam }: { fam: FamilyEvolution }) {
  const navigate = useNavigate()
  const ordered = useMemo(() => orderNodes(fam.nodes), [fam])
  const byId = useMemo(() => new Map(ordered.map((nd) => [nd.id, nd])), [ordered])
  const orderIdx = useMemo(() => new Map(ordered.map((nd, i) => [nd.id, i])), [ordered])
  const geo = useMemo(() => layoutNodes(ordered), [ordered])
  const n = ordered.length

  const [k, setK] = useState(1)
  const [playing, setPlaying] = useState(true)
  const [speedIdx, setSpeedIdx] = useState(1)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [tip, setTip] = useState<Tip | null>(null)

  const kSafe = Math.min(k, n)
  const current = (selectedId ? byId.get(selectedId) : null) ?? ordered[kSafe - 1]
  const lineage = useMemo(() => ancestorIds(current.id, byId), [current.id, byId])
  const bestIdx = bestRevealedIndex(ordered, kSafe)
  const bestPos = bestIdx >= 0 ? geo.pos.get(ordered[bestIdx].id) : undefined

  useEffect(() => {
    if (!playing) return
    const t = setInterval(() => setK((v) => Math.min(v + 1, n)), TICK_MS / SPEEDS[speedIdx])
    return () => clearInterval(t)
  }, [playing, speedIdx, n])
  useEffect(() => {
    if (playing && kSafe >= n) setPlaying(false)
  }, [playing, kSafe, n])

  const togglePlay = () => {
    if (!playing && kSafe >= n) setK(1)
    setPlaying((p) => !p)
  }
  const step = (d: number) => {
    setPlaying(false)
    setSelectedId(null)
    setK((v) => Math.max(1, Math.min(v + d, n)))
  }
  const reset = () => {
    setPlaying(false)
    setSelectedId(null)
    setK(1)
  }
  const jumpToBest = () => {
    const i = bestRevealedIndex(ordered, n)
    if (i < 0) return
    setPlaying(false)
    setSelectedId(null)
    setK(i + 1)
  }
  const seek = (i: number) => {
    setSelectedId(null)
    setK(i + 1)
  }
  const onKeyDown = (e: React.KeyboardEvent) => {
    if ((e.target as HTMLElement).tagName === 'BUTTON') return
    if (e.key === ' ') { e.preventDefault(); togglePlay() }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1) }
    else if (e.key === 'ArrowRight') { e.preventDefault(); step(1) }
  }

  return (
    <div className="evo-grid">
      <div className="card evo-wrap" tabIndex={0} onKeyDown={onKeyDown}>
        <div className="card-head">
          evolution replay · space plays/pauses · ←/→ step · click a node to inspect
        </div>
        <div className="card-body">
          <div className="evo-controls">
            <button className="evo-btn" onClick={reset} title="reset">⏮</button>
            <button className="evo-btn" onClick={togglePlay} title="play/pause (space)">
              {playing ? '⏸' : '▶'}
            </button>
            <button className="evo-btn" onClick={() => step(-1)} title="step back">◀</button>
            <button className="evo-btn" onClick={() => step(1)} title="step forward">▶</button>
            <button
              className="evo-btn"
              onClick={() => setSpeedIdx((i) => (i + 1) % SPEEDS.length)}
              title="cycle playback speed"
            >
              ×{SPEEDS[speedIdx]}
            </button>
            <button className="evo-btn" onClick={jumpToBest} title="jump to best run">⭐ best</button>
            <span className="evo-readout">
              {kSafe}/{n} runs · gen {ordered[kSafe - 1].iteration ?? '—'}
            </span>
          </div>

          <svg
            className="evo-svg"
            viewBox={`${geo.minX} 0 ${geo.width} ${geo.height}`}
            preserveAspectRatio="xMidYMid meet"
          >
            {fam.edges.map((e) => {
              const a = geo.pos.get(e.source)
              const b = geo.pos.get(e.target)
              if (!a || !b) return null
              const revealed =
                (orderIdx.get(e.source) ?? n) < kSafe && (orderIdx.get(e.target) ?? n) < kSafe
              const onPath = lineage.has(e.source) && lineage.has(e.target)
              const xover = e.kind === 'crossover'
              const cls = [
                'evo-edge',
                revealed ? 'on' : '',
                xover ? 'xover' : '',
                onPath ? 'lit' : 'dim',
              ].join(' ')
              return (
                <path
                  key={`${e.source}->${e.target}`}
                  className={cls}
                  d={edgePath(a, b)}
                  pathLength={1}
                  stroke={xover ? '#c78bf0' : '#7a879e'}
                />
              )
            })}

            {ordered.map((nd, i) => {
              const p = geo.pos.get(nd.id)!
              const revealed = i < kSafe
              const isCurrent = nd.id === current.id
              return (
                <g
                  key={nd.id}
                  className={`evo-node ${revealed ? 'on' : ''}`}
                  onClick={() => setSelectedId((s) => (s === nd.id ? null : nd.id))}
                  onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, node: nd })}
                  onMouseLeave={() => setTip(null)}
                >
                  {isCurrent && <circle className="evo-halo" cx={p.x} cy={p.y} r={NODE_R} />}
                  {nd.is_focus && <circle className="evo-focus-ring" cx={p.x} cy={p.y} r={NODE_R + 6} />}
                  <circle
                    cx={p.x} cy={p.y} r={NODE_R}
                    fill={kindFill(nd.evolution_kind)}
                    stroke={scoreColor(nd.score)}
                    strokeWidth={2.5}
                  />
                  <text className="evo-num" x={p.x} y={p.y + 4}>{nd.iteration ?? '·'}</text>
                  {nd.on_error && <text className="evo-err" x={p.x + 11} y={p.y - 9}>✕</text>}
                </g>
              )
            })}

            {bestPos && (
              <g
                className="evo-best"
                style={{ transform: `translate(${bestPos.x}px, ${bestPos.y - NODE_R - 14}px)` }}
              >
                <path d="M 0 -6 L 6 0 L 0 6 L -6 0 Z" />
              </g>
            )}
          </svg>

          <div className="evo-scrub">
            {ordered.map((nd, i) => (
              <button
                key={nd.id}
                className={`evo-scrub-bar ${i < kSafe ? 'on' : ''} ${i === kSafe - 1 ? 'cur' : ''}`}
                style={{ height: 4 + (nd.score ?? 0) * 30, background: scoreColor(nd.score) }}
                onClick={() => seek(i)}
                title={`${shortId(nd.id)} · ${fmtScore(nd.score)}`}
              />
            ))}
          </div>
        </div>
      </div>

      <InfoPanel node={current} byId={byId} onOpen={(id) => navigate(`/runs/${id}`)} />

      {tip && (
        <div
          className="evo-tip"
          style={{ left: Math.min(tip.x + 14, window.innerWidth - 240), top: tip.y + 12 }}
        >
          <div className="mono" style={{ fontSize: 11, marginBottom: 2 }}>{shortId(tip.node.id)}</div>
          <div>
            {tip.node.evolution_kind ?? 'seed'} · score {fmtScore(tip.node.score)} ·{' '}
            {fmtCost(tip.node.iteration_cost_usd)}
          </div>
        </div>
      )}
    </div>
  )
}

// ── info panel ──────────────────────────────────────────────────────────────

function Kv({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="evo-kv">
      <span className="evo-kv-l">{label}</span>
      <span className="evo-kv-v">{children}</span>
    </div>
  )
}

function DeltaTag({ delta }: { delta: number }) {
  const up = delta >= 0
  return (
    <span className="evo-delta" style={{ color: up ? 'var(--ok)' : 'var(--bad)' }}>
      {up ? `▲ +${delta.toFixed(3)}` : `▼ ${delta.toFixed(3)}`}
    </span>
  )
}

function ClaimChips({ claims }: { claims: NonNullable<EvolutionNode['claims']> }) {
  return (
    <span className="evo-claims">
      <span className="evo-claim pass">✓ {claims.passed}</span>
      <span className="evo-claim fail">✗ {claims.failed}</span>
      <span className="evo-claim err">! {claims.error}</span>
      <span className="evo-claim unsure">? {claims.unsure}</span>
    </span>
  )
}

function InfoPanel({ node, byId, onOpen }: {
  node: EvolutionNode
  byId: Map<string, EvolutionNode>
  onOpen: (id: string) => void
}) {
  const delta = parentDelta(node, byId)
  const uncappedDiffers =
    node.score_uncapped != null && node.score != null && node.score_uncapped !== node.score
  return (
    <div className="card">
      <div className="card-head">run under the lens</div>
      <div className="card-body">
        <div className="evo-idrow">
          <span className="mono" style={{ fontSize: 12 }}>{shortId(node.id)}</span>
          <button className="evo-btn" onClick={() => onOpen(node.id)}>open ↗</button>
        </div>
        <div className="evo-tags">
          <KindTag kind={node.evolution_kind ?? 'seed'} />
          <span className="hint">iter {node.iteration ?? '—'}</span>
          <StatusBadge status={node.status} />
        </div>

        <Kv label="score">
          <ScoreChip score={node.score} />
          {delta != null && <DeltaTag delta={delta} />}
        </Kv>
        {uncappedDiffers && <Kv label="uncapped">{fmtScore(node.score_uncapped)}</Kv>}
        <Kv label="qd score">{fmtScore(node.qd_score)}</Kv>
        <Kv label="novelty">{fmtScore(node.novelty_score)}</Kv>
        {node.claims && <Kv label="claims"><ClaimChips claims={node.claims} /></Kv>}
        <Kv label="cost">
          iter {fmtCost(node.iteration_cost_usd)} · total {fmtCost(node.cumulative_cost_usd)}
        </Kv>
        <Kv label="wall time">{fmtDuration(node.wall_time_s)}</Kv>

        {node.selection && (
          <div className="evo-selline">
            <span>{selectionLine(node.selection)}</span>
            {node.selection.admit_rejected && <span className="evo-rej">admit rejected</span>}
          </div>
        )}

        {node.gradient_snippet && (
          <>
            <div className="hint" style={{ marginTop: 12 }}>
              textual gradient — the feedback that steered the next mutation
            </div>
            <pre className="evo-gradient">{node.gradient_snippet}</pre>
          </>
        )}
      </div>
    </div>
  )
}

// ── styles ──────────────────────────────────────────────────────────────────

const STYLE = `
.evo-grid { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 14px; align-items: start; }
@media (max-width: 1100px) { .evo-grid { grid-template-columns: minmax(0, 1fr); } }
.evo-wrap { outline: none; }
.evo-wrap:focus-visible { box-shadow: 0 0 0 2px #5db8f044; }

.evo-controls { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
.evo-btn { background: var(--panel-2); border: 1px solid var(--border); color: var(--text);
  border-radius: 6px; padding: 3px 10px; font-family: var(--mono); font-size: 12px;
  line-height: 1.6; cursor: pointer; }
.evo-btn:hover { border-color: var(--border-strong); }
.evo-readout { margin-left: auto; font-family: var(--mono); font-size: 11.5px;
  color: var(--text-dim); white-space: nowrap; }

.evo-svg { width: 100%; height: 460px; display: block; background: #0c111b;
  border: 1px solid var(--border); border-radius: 8px; user-select: none; }

.evo-edge { fill: none; stroke-width: 1.6; stroke-dasharray: 1; stroke-dashoffset: 1; opacity: 0;
  transition: stroke-dashoffset .7s ease, opacity .5s ease, stroke .3s ease, stroke-width .3s ease; }
.evo-edge.on { stroke-dashoffset: 0; opacity: 1; }
.evo-edge.on.xover { stroke-dasharray: .055 .04; }
.evo-edge.on.dim { opacity: .3; }
.evo-edge.on.lit { stroke: var(--gold); stroke-width: 2.2; filter: drop-shadow(0 0 3px #f2c14e99); }

.evo-node { transform-box: fill-box; transform-origin: center; opacity: 0; transform: scale(.25);
  pointer-events: none; transition: transform .45s cubic-bezier(.34, 1.56, .64, 1), opacity .35s ease; }
.evo-node.on { opacity: 1; transform: scale(1); pointer-events: auto; cursor: pointer; }
.evo-num { font: 700 11px var(--mono); fill: #0b0e14; text-anchor: middle; pointer-events: none; }
.evo-err { font: 700 10px var(--sans); fill: var(--bad); paint-order: stroke; stroke: #0c111b;
  stroke-width: 3px; text-anchor: middle; pointer-events: none; }
.evo-focus-ring { fill: none; stroke: var(--gold); stroke-width: 2; opacity: .9; }
.evo-halo { fill: none; stroke: var(--gold); stroke-width: 3; transform-box: fill-box;
  transform-origin: center; animation: evo-pulse 1.5s ease-out infinite; pointer-events: none; }
@keyframes evo-pulse {
  0% { transform: scale(1); opacity: .6; }
  75% { transform: scale(2); opacity: 0; }
  100% { transform: scale(2); opacity: 0; }
}
.evo-best { transition: transform .6s cubic-bezier(.22, 1, .36, 1); pointer-events: none; }
.evo-best path { fill: var(--gold); filter: drop-shadow(0 0 3px #f2c14eb0); }

.evo-scrub { display: flex; align-items: flex-end; gap: 2px; height: 42px; margin-top: 10px; }
.evo-scrub-bar { flex: 1 1 0; min-width: 2px; border: none; padding: 0;
  border-radius: 2px 2px 0 0; opacity: .25; cursor: pointer; transition: opacity .25s ease; }
.evo-scrub-bar.on { opacity: 1; }
.evo-scrub-bar.cur { box-shadow: 0 0 0 1.5px var(--gold); }

.evo-tip { position: fixed; z-index: 60; pointer-events: none; background: var(--bg-elev);
  border: 1px solid var(--border-strong); border-radius: 8px; padding: 6px 10px;
  font-size: 12px; color: var(--text); box-shadow: 0 8px 24px #000a; }

.evo-idrow { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 8px; }
.evo-tags { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
.evo-kv { display: flex; justify-content: space-between; align-items: center; gap: 10px;
  padding: 5px 0; border-bottom: 1px dashed var(--border); font-size: 12.5px; }
.evo-kv-l { color: var(--text-faint); font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }
.evo-kv-v { display: flex; align-items: center; gap: 7px; font-family: var(--mono); font-size: 12px; }
.evo-delta { font-family: var(--mono); font-size: 12px; }
.evo-claims { display: flex; gap: 6px; flex-wrap: wrap; }
.evo-claim { font-family: var(--mono); font-size: 11px; padding: 1px 7px; border-radius: 5px; }
.evo-claim.pass { color: var(--ok); background: #4ec98a1e; }
.evo-claim.fail { color: var(--bad); background: #ef6a6a1e; }
.evo-claim.err { color: var(--warn); background: #f0a7421e; }
.evo-claim.unsure { color: var(--text-faint); background: #6b768822; }
.evo-selline { font-size: 12px; color: var(--text-dim); margin-top: 9px;
  display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
.evo-rej { font-size: 10px; text-transform: uppercase; letter-spacing: .4px; color: var(--bad);
  background: #ef6a6a1e; padding: 1px 6px; border-radius: 4px; }
.evo-gradient { max-height: 180px; overflow-y: auto; background: #0c111b;
  border: 1px solid var(--border); border-radius: 8px; padding: 10px; margin-top: 6px;
  font-family: var(--mono); font-size: 11px; line-height: 1.5; white-space: pre-wrap;
  color: var(--text-dim); }
`

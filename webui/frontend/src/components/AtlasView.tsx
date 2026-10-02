import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAsync } from '../hooks'
import type { AtlasData, AtlasPoint } from '../types'
import { KindTag, Spinner, fmtCost, fmtScore, scoreColor, shortId } from '../ui'

/**
 * QD Atlas: every run projected onto the 2-D PCA of Mimosa's own
 * quality-diversity behaviour space (the 384-dim qd_descriptor).
 * Parent→child trails show how mutations moved through behaviour space.
 * Interactive: colour modes, family filter, time replay, zoom/pan, tooltips.
 */

const PLOT_W = 1000
const PLOT_H = 640
const POINT_R = 6
const MIN_ZOOM = 0.5
const MAX_ZOOM = 8
const REPLAY_MS = 120
const GREY = '#6b7688'
const AXIS_STROKE = '#1e2637'
const BASE_VIEW = { x: 0, y: 0, w: PLOT_W, h: PLOT_H }

type ColorMode = 'score' | 'family' | 'iteration'
interface View { x: number; y: number; w: number; h: number }
interface Placed extends AtlasPoint { sx: number; sy: number; order: number }
interface Tip { p: Placed; left: number; top: number; flip: boolean }

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

const famColor = (i: number) => `hsl(${(i * 137.5) % 360}, 55%, 55%)`

/** Light-blue → gold ramp for iteration depth (hue 210 → 45). */
const iterColor = (t: number) => `hsl(${210 + (45 - 210) * clamp(t, 0, 1)}, 65%, 58%)`

/** Map data coords into a fixed plot space (8% margin, y flipped) + time order. */
function placePoints(points: AtlasPoint[]) {
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)]
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)]
  const padX = (x1 - x0 || 1) * 0.08
  const padY = (y1 - y0 || 1) * 0.08
  const sx = (x: number) => ((x - x0 + padX) / (x1 - x0 + 2 * padX || 1)) * PLOT_W
  const sy = (y: number) => PLOT_H - ((y - y0 + padY) / (y1 - y0 + 2 * padY || 1)) * PLOT_H

  const orderOf = new Map(
    [...points]
      .sort((a, b) => (a.started_at || a.id).localeCompare(b.started_at || b.id))
      .map((p, i) => [p.id, i] as const),
  )
  const placed = points.map((p) => ({ ...p, sx: sx(p.x), sy: sy(p.y), order: orderOf.get(p.id)! }))
  const ordered = [...placed].sort((a, b) => a.order - b.order)
  const byId = new Map(placed.map((p) => [p.id, p]))
  const mean = {
    sx: placed.reduce((s, p) => s + p.sx, 0) / placed.length,
    sy: placed.reduce((s, p) => s + p.sy, 0) / placed.length,
  }
  return { ordered, byId, mean }
}

/** Family membership counts, palette index per family, max iteration. */
function familyStats(points: AtlasPoint[]) {
  const counts = new Map<number, number>()
  let maxIter = 1
  for (const p of points) {
    if (p.family != null) counts.set(p.family, (counts.get(p.family) || 0) + 1)
    if (p.iteration != null && p.iteration > maxIter) maxIter = p.iteration
  }
  const list = [...counts.entries()].sort((a, b) => a[0] - b[0]).map(([family, count]) => ({ family, count }))
  const index = new Map(list.map((f, i) => [f.family, i] as const))
  return { list, index, maxIter }
}

/** Quadratic parent→child curve with a small perpendicular bow + arrowhead. */
function trailGeometry(a: Placed, b: Placed, z: number) {
  const dx = b.sx - a.sx
  const dy = b.sy - a.sy
  const len = Math.hypot(dx, dy) || 1
  const bow = Math.min(20, len * 0.15)
  const cx = (a.sx + b.sx) / 2 - (dy / len) * bow
  const cy = (a.sy + b.sy) / 2 + (dx / len) * bow
  const tlen = Math.hypot(b.sx - cx, b.sy - cy) || 1
  const tx = (b.sx - cx) / tlen // tangent at the child end
  const ty = (b.sy - cy) / tlen
  const tipX = b.sx - tx * ((POINT_R + 3) / z)
  const tipY = b.sy - ty * ((POINT_R + 3) / z)
  const s = 6 / z
  const head = `${tipX},${tipY} ${tipX - tx * s - ty * s * 0.45},${tipY - ty * s + tx * s * 0.45}`
    + ` ${tipX - tx * s + ty * s * 0.45},${tipY - ty * s - tx * s * 0.45}`
  return { d: `M ${a.sx} ${a.sy} Q ${cx} ${cy} ${b.sx} ${b.sy}`, head }
}

/** Zoom the viewBox about an svg-space anchor point, clamped 0.5×–8×. */
function zoomAt(v: View, px: number, py: number, factor: number): View {
  const w = clamp(v.w * factor, PLOT_W / MAX_ZOOM, PLOT_W / MIN_ZOOM)
  const s = w / v.w
  return { x: px - (px - v.x) * s, y: py - (py - v.y) * s, w, h: v.h * s }
}

export type AtlasSpace = 'qd' | 'genotype'

export default function AtlasView() {
  const [space, setSpace] = useState<AtlasSpace>('qd')
  const { data, loading, error } = useAsync(() => api.atlas(space), [space])
  if (loading) return <Spinner label="Projecting embedding space…" />
  if (error || !data) return <div className="empty"><div className="hint">No atlas data.</div></div>
  if (data.points.length < 3) {
    return (
      <div className="card" style={{ margin: 18 }}>
        <div className="card-head">evolution atlas</div>
        <div className="card-body hint">
          Not enough embedded runs to project — the atlas needs at least 3 runs with a
          descriptor ({data.skipped.length} runs on disk have none).
        </div>
      </div>
    )
  }
  return <Atlas data={data} space={space} setSpace={setSpace} />
}

function Atlas({ data, space, setSpace }: {
  data: AtlasData
  space: AtlasSpace
  setSpace: (s: AtlasSpace) => void
}) {
  const navigate = useNavigate()
  const [mode, setMode] = useState<ColorMode>('score')
  const [family, setFamily] = useState('all')
  const [view, setView] = useState<View>(BASE_VIEW)
  const [tip, setTip] = useState<Tip | null>(null)

  const layout = useMemo(() => placePoints(data.points), [data.points])
  const fams = useMemo(() => familyStats(data.points), [data.points])
  const n = layout.ordered.length
  const [timeIndex, setTimeIndex] = useState(n)
  const [playing, setPlaying] = useState(false)

  // ── family trajectory: step point-by-point through one family's search ──
  const famSel = family === 'all' ? null : Number(family)
  const famOrdered = useMemo(
    () => (famSel == null ? [] : layout.ordered.filter((p) => p.family === famSel)),
    [layout, famSel],
  )
  const trajOrder = useMemo(
    () => new Map(famOrdered.map((p, i) => [p.id, i] as const)),
    [famOrdered],
  )
  const m = famOrdered.length
  const [trajIndex, setTrajIndex] = useState(0)
  const [trajPlaying, setTrajPlaying] = useState(false)
  const trajectory = famSel != null && m > 1

  useEffect(() => {
    // entering/leaving a family resets the trajectory to fully revealed and
    // releases the fleet-wide time filter so the two replays never compound
    setTrajIndex(m)
    setTrajPlaying(false)
    setTimeIndex(n)
    setPlaying(false)
  }, [famSel, m, n])
  useEffect(() => {
    if (!trajPlaying) return
    const t = setInterval(() => setTrajIndex((i) => Math.min(m, i + 1)), 900)
    return () => clearInterval(t)
  }, [trajPlaying, m])
  useEffect(() => { if (trajIndex >= m) setTrajPlaying(false) }, [trajIndex, m])

  const toggleTrajPlay = () => {
    if (trajPlaying) { setTrajPlaying(false); return }
    if (trajIndex >= m) setTrajIndex(1)
    setTrajPlaying(true)
  }
  const trajStep = (d: number) => {
    setTrajPlaying(false)
    setTrajIndex((i) => clamp(i + d, 1, m))
  }
  const trajCurrent = trajectory && trajIndex > 0 ? famOrdered[trajIndex - 1] : null
  const trajPrev = trajectory && trajIndex > 1 ? famOrdered[trajIndex - 2] : null
  const trajDelta = trajCurrent?.score != null && trajPrev?.score != null
    ? trajCurrent.score - trajPrev.score
    : null
  /** Degenerate trajectory: every family point at one spot (QD embeds the task). */
  const famCollapsed = useMemo(() => {
    if (famOrdered.length < 2) return false
    const [f] = famOrdered
    return famOrdered.every((p) => Math.hypot(p.sx - f.sx, p.sy - f.sy) < 1)
  }, [famOrdered])

  const svgRef = useRef<SVGSVGElement | null>(null)
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const dragRef = useRef<{ cx: number; cy: number; active: boolean } | null>(null)
  const movedRef = useRef(false)

  // ── time replay ──
  useEffect(() => {
    if (!playing) return
    const t = setInterval(() => setTimeIndex((i) => Math.min(n, i + 1)), REPLAY_MS)
    return () => clearInterval(t)
  }, [playing, n])
  useEffect(() => { if (timeIndex >= n) setPlaying(false) }, [timeIndex, n])

  const togglePlay = () => {
    if (playing) { setPlaying(false); return }
    if (timeIndex >= n) setTimeIndex(0)
    setPlaying(true)
  }

  // ── wheel zoom (native listener: React's delegated wheel is passive) ──
  useEffect(() => {
    const svg = svgRef.current
    if (!svg) return
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      const ctm = svg.getScreenCTM()
      if (!ctm) return
      const pt = svg.createSVGPoint()
      pt.x = e.clientX
      pt.y = e.clientY
      const p = pt.matrixTransform(ctm.inverse())
      setView((v) => zoomAt(v, p.x, p.y, Math.exp(e.deltaY * 0.0016)))
    }
    svg.addEventListener('wheel', onWheel, { passive: false })
    return () => svg.removeEventListener('wheel', onWheel)
  }, [])

  // ── drag pan (capture only once the drag threshold is crossed) ──
  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0) return
    dragRef.current = { cx: e.clientX, cy: e.clientY, active: false }
    movedRef.current = false
  }
  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const d = dragRef.current
    if (!d) return
    const dx = e.clientX - d.cx
    const dy = e.clientY - d.cy
    if (!d.active) {
      if (Math.hypot(dx, dy) < 4) return
      d.active = true
      movedRef.current = true
      e.currentTarget.setPointerCapture(e.pointerId)
      setTip(null)
    }
    d.cx = e.clientX
    d.cy = e.clientY
    const ctm = svgRef.current?.getScreenCTM()
    if (ctm) setView((v) => ({ ...v, x: v.x - dx / ctm.a, y: v.y - dy / ctm.d }))
  }
  const onPointerUp = () => { dragRef.current = null }

  // ── tooltip ──
  const showTip = (e: React.PointerEvent, p: Placed) => {
    if (dragRef.current?.active) return
    const rect = wrapRef.current?.getBoundingClientRect()
    if (!rect) return
    const left = e.clientX - rect.left
    setTip({ p, left, top: e.clientY - rect.top, flip: left > rect.width - 220 })
  }

  // ── per-point colour, filter + replay visibility ──
  const colorOf = (p: AtlasPoint): string => {
    if (mode === 'score') return scoreColor(p.score)
    if (mode === 'family') return p.family == null ? GREY : famColor(fams.index.get(p.family) ?? 0)
    return p.iteration == null ? GREY : iterColor(p.iteration / fams.maxIter)
  }
  const isShown = (p: Placed) => p.order < timeIndex
  const inFamily = (p: Placed) => famSel == null || p.family === famSel
  /** In trajectory mode, family points not yet reached are ghosted. */
  const trajFuture = (p: Placed) =>
    trajectory && inFamily(p) && (trajOrder.get(p.id) ?? 0) >= trajIndex

  const z = PLOT_W / view.w
  const [v1 = 0, v2 = 0] = data.variance_explained
  const pctBoth = ((v1 + v2) * 100).toFixed(1)
  const last = timeIndex > 0 ? layout.ordered[timeIndex - 1] : null
  const readoutDate = last?.started_at ? new Date(last.started_at).toLocaleString() : '—'

  return (
    <div className="atlas-page">
      <style>{STYLE}</style>

      <div className="card">
        <div className="card-body atl-headrow">
          <div>
            <h2>evolution atlas · {space === 'qd' ? 'behaviour space' : 'genotype space'}</h2>
            <div className="hint">
              {space === 'qd'
                ? <>Each run is Mimosa's own {data.n_dimensions}-dim QD behaviour descriptor
                  PCA-projected to 2D. Note: the descriptor embeds the <i>task</i>, so runs of
                  one family coincide — switch to genotype space for within-family drift.</>
                : <>Each run's evolved workflow code ({data.n_dimensions}-token TF-IDF)
                  PCA-projected to 2D — mutations of one family spread apart as the code
                  itself evolves, so a family trajectory shows the search exploring.</>}
            </div>
            <div className="seg" style={{ marginTop: 8, display: 'inline-flex' }}>
              <button className={space === 'qd' ? 'active' : ''} onClick={() => setSpace('qd')}>behaviour (QD)</button>
              <button className={space === 'genotype' ? 'active' : ''} onClick={() => setSpace('genotype')}>genotype (code)</button>
            </div>
          </div>
          <div className="atl-chips">
            <span className="atl-chip"><b>{data.points.length}</b> embedded runs</span>
            <span className="atl-chip"><b>{data.edges.length}</b> trails</span>
            <span className="atl-chip"><b>{data.skipped.length}</b> skipped (no descriptor)</span>
            <span className="atl-chip">PC1+PC2 = <b>{pctBoth}%</b> of {data.n_dimensions}-dim variance</span>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-head atl-toolbar">
          <div className="seg">
            {(['score', 'family', 'iteration'] as ColorMode[]).map((m) => (
              <button key={m} className={mode === m ? 'active' : ''} onClick={() => setMode(m)}>
                {m}
              </button>
            ))}
          </div>
          <select value={family} onChange={(e) => setFamily(e.target.value)} title="Filter by family">
            <option value="all">All families</option>
            {fams.list.map((f) => (
              <option key={f.family} value={f.family}>family {f.family} · {f.count}</option>
            ))}
          </select>
          <span style={{ flex: 1 }} />
          <button onClick={() => setView(BASE_VIEW)}>reset view</button>
        </div>

        <div className="card-body">
          <div className="atl-plot-wrap" ref={wrapRef}>
            <svg
              ref={svgRef}
              className="atl-svg"
              viewBox={`${view.x} ${view.y} ${view.w} ${view.h}`}
              preserveAspectRatio="xMidYMid meet"
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
            >
              <g>
                <line x1={0} y1={layout.mean.sy} x2={PLOT_W} y2={layout.mean.sy} stroke={AXIS_STROKE} strokeWidth={1 / z} />
                <line x1={layout.mean.sx} y1={0} x2={layout.mean.sx} y2={PLOT_H} stroke={AXIS_STROKE} strokeWidth={1 / z} />
                <text x={PLOT_W - 8} y={layout.mean.sy - 8} textAnchor="end" fill={GREY} fontSize={12 / z}>
                  PC1 ({(v1 * 100).toFixed(1)}%)
                </text>
                <text x={layout.mean.sx + 8} y={20} fill={GREY} fontSize={12 / z}>
                  PC2 ({(v2 * 100).toFixed(1)}%)
                </text>
              </g>

              {data.edges.map((e, i) => {
                const a = layout.byId.get(e.source)
                const b = layout.byId.get(e.target)
                if (!a || !b) return null
                const geo = trailGeometry(a, b, z)
                const color = colorOf(b)
                const cls = 'atl-trail'
                  + (isShown(a) && isShown(b) ? '' : ' t-hide')
                  + (inFamily(b) ? '' : ' f-fade')
                return (
                  <g key={i} className={cls}>
                    <path d={geo.d} fill="none" stroke={color} strokeOpacity={0.35} strokeWidth={1.5 / z} />
                    <polygon points={geo.head} fill={color} fillOpacity={0.55} />
                  </g>
                )
              })}

              {trajectory && trajIndex > 1 && famOrdered.slice(1, trajIndex).map((p, i) => {
                const a = famOrdered[i]
                // comet trail: recent segments bright, older ones fade
                const t = (i + 1) / Math.max(1, trajIndex - 1)
                return (
                  <line
                    key={`traj-${p.id}`}
                    className="atl-traj-seg"
                    x1={a.sx} y1={a.sy} x2={p.sx} y2={p.sy}
                    stroke="var(--gold)"
                    strokeWidth={2 / z}
                    strokeOpacity={0.2 + 0.7 * t}
                  />
                )
              })}
              {trajCurrent && (
                <circle
                  className="atl-halo"
                  cx={trajCurrent.sx} cy={trajCurrent.sy}
                  r={(POINT_R + 7) / z}
                  fill="none" stroke="var(--gold)" strokeWidth={2 / z}
                />
              )}

              {layout.ordered.map((p) => {
                const cls = 'atl-pt'
                  + (isShown(p) ? '' : ' t-hide')
                  + (inFamily(p) ? '' : ' f-fade')
                  + (trajFuture(p) ? ' traj-future' : '')
                const fill = colorOf(p)
                const r = POINT_R / z
                return (
                  <g
                    key={p.id}
                    className={cls}
                    transform={`translate(${p.sx}, ${p.sy})`}
                    onClick={() => { if (!movedRef.current) navigate(`/runs/${p.id}`) }}
                    onPointerEnter={(e) => showTip(e, p)}
                    onPointerMove={(e) => showTip(e, p)}
                    onPointerLeave={() => setTip(null)}
                  >
                    <g className="atl-pt-in">
                      {p.evolution_kind === 'crossover'
                        ? <rect x={-r} y={-r} width={r * 2} height={r * 2} transform="rotate(45)" fill={fill} stroke="#0b0e14" strokeWidth={1.5 / z} />
                        : <circle r={r} fill={fill} stroke="#0b0e14" strokeWidth={1.5 / z} />}
                    </g>
                  </g>
                )
              })}
            </svg>
            {tip && <PointTooltip tip={tip} />}
          </div>

          {trajectory ? (
            <div className="atl-replay">
              <span className="atl-traj-label">family {famSel} trajectory</span>
              <button className="atl-play" onClick={() => trajStep(-1)} title="Previous point">◀</button>
              <button className="atl-play" onClick={toggleTrajPlay} title="Replay this family's path point by point">
                {trajPlaying ? '⏸' : '▶'}
              </button>
              <button className="atl-play" onClick={() => trajStep(1)} title="Next point">▶︎</button>
              <input
                type="range"
                min={1}
                max={m}
                value={Math.max(1, trajIndex)}
                onChange={(e) => { setTrajPlaying(false); setTrajIndex(Number(e.target.value)) }}
              />
              <span className="atl-readout mono">
                {trajIndex} / {m}
                {trajCurrent && <> · {shortId(trajCurrent.id)} · gen {trajCurrent.iteration ?? '—'} · </>}
                {trajCurrent && (
                  <b style={{ color: scoreColor(trajCurrent.score) }}>{fmtScore(trajCurrent.score)}</b>
                )}
                {trajDelta != null && (
                  <span style={{ color: trajDelta >= 0 ? 'var(--ok)' : 'var(--bad)' }}>
                    {' '}{trajDelta >= 0 ? '▲' : '▼'}{Math.abs(trajDelta).toFixed(3)}
                  </span>
                )}
              </span>
              {famCollapsed && space === 'qd' && (
                <button className="atl-collapsed-hint" onClick={() => setSpace('genotype')}>
                  this family's points coincide (the QD descriptor embeds the task) —
                  switch to genotype space to see its drift
                </button>
              )}
            </div>
          ) : (
            <div className="atl-replay">
              <button className="atl-play" onClick={togglePlay} title="Replay the search over time">
                {playing ? '⏸' : '▶'}
              </button>
              <input
                type="range"
                min={0}
                max={n}
                value={timeIndex}
                onChange={(e) => { setPlaying(false); setTimeIndex(Number(e.target.value)) }}
              />
              <span className="atl-readout mono">{timeIndex} / {n} · {readoutDate}</span>
            </div>
          )}
        </div>
      </div>

      <SkippedDetails skipped={data.skipped} />
    </div>
  )
}

function PointTooltip({ tip }: { tip: Tip }) {
  const p = tip.p
  return (
    <div className={`atl-tip${tip.flip ? ' flip' : ''}`} style={{ left: tip.left, top: tip.top }}>
      <div className="atl-tip-id">{shortId(p.id)}<KindTag kind={p.evolution_kind} /></div>
      <div className="atl-tip-row"><span>family</span><span>{p.family ?? '—'}</span></div>
      <div className="atl-tip-row"><span>iteration</span><span>{p.iteration ?? '—'}</span></div>
      <div className="atl-tip-row">
        <span>score</span>
        <b style={{ color: scoreColor(p.score) }}>{fmtScore(p.score)}</b>
      </div>
      <div className="atl-tip-row"><span>cost</span><span>{fmtCost(p.cost)}</span></div>
    </div>
  )
}

function SkippedDetails({ skipped }: { skipped: AtlasData['skipped'] }) {
  if (skipped.length === 0) return null
  const byReason = new Map<string, string[]>()
  for (const s of skipped) byReason.set(s.reason, [...(byReason.get(s.reason) || []), s.id])
  return (
    <details className="atl-skip card">
      <summary>skipped runs · {skipped.length} without a behaviour descriptor</summary>
      {[...byReason.entries()].map(([reason, ids]) => (
        <details key={reason} className="atl-skip-group">
          <summary className="mono">{reason}: {ids.length}</summary>
          <div className="atl-skip-ids">{ids.join('  ')}</div>
        </details>
      ))}
    </details>
  )
}

const STYLE = `
.atlas-page { margin: 18px; display: flex; flex-direction: column; gap: 14px; }
.atl-headrow { display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; flex-wrap: wrap; }
.atl-headrow h2 { font-size: 17px; margin-bottom: 4px; }
.atl-headrow .hint { max-width: 560px; }
.atl-chips { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.atl-chip {
  font-size: 12px; color: var(--text-dim); background: var(--panel-2);
  border: 1px solid var(--border); border-radius: 999px; padding: 4px 11px; white-space: nowrap;
}
.atl-chip b { color: var(--text); font-weight: 600; }
.atl-toolbar { gap: 10px; justify-content: flex-start; flex-wrap: wrap; text-transform: none; letter-spacing: normal; font-size: 13px; }
.atl-toolbar .seg { flex-shrink: 0; }
.atl-toolbar select {
  background: var(--panel-2); color: var(--text); border: 1px solid var(--border-strong);
  border-radius: 7px; padding: 6px 10px; font-size: 13px; font-family: inherit;
}
.atl-plot-wrap { position: relative; }
.atl-svg { display: block; width: 100%; height: 70vh; cursor: grab; touch-action: none; user-select: none; }
.atl-svg:active { cursor: grabbing; }
.atl-pt { cursor: pointer; transition: opacity 0.25s ease; }
.atl-pt-in { transition: opacity 0.3s ease, transform 0.3s ease; transform-box: fill-box; transform-origin: center; }
.atl-pt.t-hide { pointer-events: none; }
.atl-pt.t-hide .atl-pt-in { opacity: 0; transform: scale(0.3); }
.atl-pt.f-fade { opacity: 0.08; pointer-events: none; }
.atl-trail { transition: opacity 0.3s ease; pointer-events: none; }
.atl-trail.t-hide { opacity: 0; }
.atl-trail.f-fade { opacity: 0.08; }
.atl-pt.traj-future .atl-pt-in { opacity: 0.15; }
.atl-pt.traj-future { pointer-events: none; }
.atl-traj-seg { pointer-events: none; }
.atl-halo { pointer-events: none; animation: atl-pulse 1.6s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }
@keyframes atl-pulse {
  0%, 100% { opacity: 0.9; transform: scale(1); }
  50% { opacity: 0.35; transform: scale(1.25); }
}
.atl-traj-label {
  font-size: 11px; color: var(--gold); text-transform: uppercase; letter-spacing: 0.5px;
  white-space: nowrap; border: 1px solid var(--gold-dim); border-radius: 999px; padding: 3px 9px;
}
.atl-collapsed-hint {
  font-size: 11px; color: var(--warn); background: none; border: 1px dashed var(--warn);
  border-radius: 7px; padding: 4px 9px; cursor: pointer; text-align: left; line-height: 1.4;
}
.atl-collapsed-hint:hover { color: var(--text); }
.atl-replay { display: flex; align-items: center; gap: 12px; margin-top: 10px; }
.atl-replay input[type='range'] { flex: 1; accent-color: var(--gold); }
.atl-play { width: 36px; padding: 5px 0; text-align: center; }
.atl-readout { font-size: 12px; color: var(--text-dim); white-space: nowrap; }
.atl-tip {
  position: absolute; z-index: 5; pointer-events: none; min-width: 150px;
  background: var(--bg-elev); border: 1px solid var(--border-strong); border-radius: 8px;
  padding: 8px 11px; font-size: 12px; color: var(--text-dim);
  box-shadow: 0 6px 22px rgba(0, 0, 0, 0.45); transform: translate(14px, 10px);
}
.atl-tip.flip { transform: translate(calc(-100% - 14px), 10px); }
.atl-tip-id { display: flex; align-items: center; gap: 6px; color: var(--text); font-family: var(--mono); margin-bottom: 4px; }
.atl-tip-row { display: flex; justify-content: space-between; gap: 14px; }
.atl-skip { padding: 10px 15px; }
.atl-skip summary { cursor: pointer; font-size: 13px; color: var(--text-dim); }
.atl-skip summary:hover { color: var(--text); }
.atl-skip-group { margin: 8px 0 0 14px; }
.atl-skip-group summary { font-size: 12px; }
.atl-skip-ids {
  font-family: var(--mono); font-size: 11px; color: var(--text-faint);
  line-height: 1.7; word-break: break-all; margin: 6px 0 4px;
}
`

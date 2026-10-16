import type { RunStatus, EvolutionKind } from './types'

/** Red→amber→green, matching Mimosa's own evolution-tree colormap. */
export function scoreColor(score: number | null | undefined): string {
  if (score == null) return '#6b7688'
  const s = Math.max(0, Math.min(1, score))
  const hue = 0 + s * 125 // 0=red … 125=green
  return `hsl(${hue}, 68%, 52%)`
}

export const fmtScore = (s: number | null | undefined) =>
  s == null ? '—' : s.toFixed(s === 1 || s === 0 ? 0 : 3)

export const fmtCost = (c: number | null | undefined) =>
  c == null ? '—' : `$${c.toFixed(c < 0.01 ? 4 : c < 1 ? 3 : 2)}`

export const fmtDuration = (s: number | null | undefined) => {
  if (s == null) return '—'
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60)
  return `${m}m ${Math.round(s % 60)}s`
}

export const fmtBytes = (n: number) => {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

export const fmtEpoch = (t: number | null | undefined) =>
  t == null ? '—' : new Date(t * 1000).toLocaleTimeString()

export const shortId = (id: string) => id.replace(/^single_agent_/, 'sa·').slice(0, 15)

export function StatusBadge({ status }: { status: RunStatus }) {
  const label = { completed: 'completed', running: 'running', crashed: 'crashed', error: 'error' }[status]
  return (
    <span className={`badge ${status}`}>
      <span className="tick" style={{ background: 'currentColor' }} />
      {label}
    </span>
  )
}

export function ScoreChip({ score }: { score: number | null }) {
  if (score == null) return <span className="kind-tag">unscored</span>
  return (
    <span className="score-chip" style={{ background: scoreColor(score) }}>
      {fmtScore(score)}
    </span>
  )
}

export function KindTag({ kind }: { kind: EvolutionKind }) {
  if (!kind) return null
  return <span className={`kind-tag kind-${kind}`}>{kind}</span>
}

export function Spinner({ label = 'Loading…' }: { label?: string }) {
  return <div className="spinner">{label}</div>
}

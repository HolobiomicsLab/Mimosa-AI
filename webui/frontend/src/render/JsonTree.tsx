import { useMemo, useState } from 'react'
import CodeView from './CodeView'

interface Props {
  /** JSON source; parsed here. Falls back to a CodeView on parse failure. */
  text?: string
  /** Already-parsed value (takes precedence over `text`). */
  value?: unknown
  /** Nodes deeper than this start collapsed. */
  initialDepth?: number
}

const STR_LIMIT = 280

function isComposite(v: unknown): v is Record<string, unknown> | unknown[] {
  return v !== null && typeof v === 'object'
}

/** Hand-rolled collapsible JSON tree in the Observatory theme. */
export default function JsonTree({ text, value, initialDepth = 2 }: Props) {
  const parsed = useMemo(() => {
    if (value !== undefined) return { ok: true as const, value }
    try {
      return { ok: true as const, value: JSON.parse(text ?? '') as unknown }
    } catch {
      return { ok: false as const, value: undefined }
    }
  }, [text, value])

  // Bumping the epoch remounts the tree so every node re-reads openDepth.
  const [epoch, setEpoch] = useState(0)
  const [openDepth, setOpenDepth] = useState(initialDepth)

  if (!parsed.ok) return <CodeView text={text ?? ''} lang="json" />

  const reset = (depth: number) => {
    setOpenDepth(depth)
    setEpoch((e) => e + 1)
  }

  return (
    <div className="json-tree">
      <div className="jt-head">
        <button className="jt-btn" onClick={() => reset(Infinity)}>expand all</button>
        <button className="jt-btn" onClick={() => reset(1)}>collapse all</button>
      </div>
      <div className="jt-body">
        <TreeNode key={epoch} k={null} v={parsed.value} depth={0} openDepth={openDepth} />
      </div>
    </div>
  )
}

function TreeNode({ k, v, depth, openDepth }: {
  k: string | null
  v: unknown
  depth: number
  openDepth: number
}) {
  const [open, setOpen] = useState(depth < openDepth)
  const indent = { paddingLeft: depth * 16 }
  const keyEl = k !== null && (
    <>
      <span className="jt-key">{k}</span>
      <span className="jt-colon">: </span>
    </>
  )

  if (!isComposite(v)) {
    return (
      <div className="jt-row" style={indent}>
        <span className="jt-arrow" />
        {keyEl}
        <Leaf v={v} />
      </div>
    )
  }

  const isArr = Array.isArray(v)
  const entries = isArr
    ? (v as unknown[]).map((x, i) => [String(i), x] as const)
    : Object.entries(v as Record<string, unknown>)
  const summary = isArr
    ? `[…] ${entries.length} item${entries.length === 1 ? '' : 's'}`
    : `{…} ${entries.length} key${entries.length === 1 ? '' : 's'}`

  return (
    <div>
      <div className="jt-row jt-toggle-row" style={indent} onClick={() => setOpen((o) => !o)}>
        <span className="jt-arrow">{open ? '▾' : '▸'}</span>
        {keyEl}
        <span className="jt-summary">{open ? (isArr ? '[' : '{') : summary}</span>
      </div>
      {open && entries.map(([ck, cv]) => (
        <TreeNode key={ck} k={ck} v={cv} depth={depth + 1} openDepth={openDepth} />
      ))}
      {open && (
        <div className="jt-row" style={indent}>
          <span className="jt-arrow" />
          <span className="jt-summary">{isArr ? ']' : '}'}</span>
        </div>
      )}
    </div>
  )
}

function Leaf({ v }: { v: unknown }) {
  if (typeof v === 'string') return <StringLeaf s={v} />
  if (typeof v === 'number') return <span className="jt-num">{String(v)}</span>
  return <span className="jt-bool">{String(v)}</span>
}

function StringLeaf({ s }: { s: string }) {
  const [full, setFull] = useState(false)
  const long = s.length > STR_LIMIT
  const shown = full || !long ? s : s.slice(0, STR_LIMIT)
  return (
    <span className="jt-str">
      "{shown}{long && !full ? '…' : ''}"
      {long && (
        <button className="jt-more" onClick={() => setFull((f) => !f)}>
          {full ? 'less' : `+${s.length - STR_LIMIT} chars`}
        </button>
      )}
    </span>
  )
}

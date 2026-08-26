import { useEffect, useState } from 'react'
import { api, workspaceFileUrl } from '../api'
import { useAsync } from '../hooks'
import type { WorkspaceListing } from '../types'
import { Spinner, fmtBytes } from '../ui'
import { SmartText } from '../render'

const TEXTY = new Set(['report', 'code', 'json', 'data'])

/** Client-side mirror of the backend's extension→kind map (workspace.py
 * _EXT_KIND — keep in lock-step, a reciprocal note sits there), for previewing
 * a deep-linked file that exists on disk but fell outside the ranked listing. */
const EXT_KIND: Record<string, string> = {
  png: 'image', jpg: 'image', jpeg: 'image', gif: 'image', svg: 'image', webp: 'image',
  csv: 'data', tsv: 'data', parquet: 'data',
  md: 'report', txt: 'report', rst: 'report',
  py: 'code', sh: 'code', r: 'code', ipynb: 'code',
  json: 'json', yaml: 'json', yml: 'json',
}
const guessKind = (path: string) =>
  EXT_KIND[path.split('.').pop()?.toLowerCase() ?? ''] ?? 'other'

export default function WorkspacePanel({ runId, initialFile }: { runId: string; initialFile?: string | null }) {
  const scopes = useAsync(() => api.workspaceScopes(), [])
  const [scope, setScope] = useState<string>('live')

  // Prefer this run's snapshot if one exists (durable across workspace resets).
  useEffect(() => {
    if (scopes.data?.snapshots.includes(runId)) setScope(runId)
  }, [scopes.data, runId])

  if (scopes.loading) return <Spinner />
  const snaps = scopes.data?.snapshots || []

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginBottom: 14, flexWrap: 'wrap' }}>
        <div className="card-head" style={{ padding: 0, border: 'none' }}>
          output files
        </div>
        <div className="pill-row" style={{ marginBottom: 0 }}>
          <button className={`pill ${scope === 'live' ? 'active' : ''}`} onClick={() => setScope('live')}>
            live workspace
          </button>
          {snaps.includes(runId) && (
            <button className={`pill ${scope === runId ? 'active' : ''}`} onClick={() => setScope(runId)}>
              snapshot (this run)
            </button>
          )}
        </div>
      </div>
      <ScopeView scope={scope} initialFile={initialFile} />
    </div>
  )
}

function ScopeView({ scope, initialFile }: { scope: string; initialFile?: string | null }) {
  const { data, loading, error } = useAsync<WorkspaceListing>(() => api.workspaceFiles(scope), [scope])
  const [selected, setSelected] = useState<string | null>(null)

  // A ?file= deep link wins over the ranked auto-preview.
  useEffect(() => { setSelected(initialFile || (data?.auto_preview ?? null)) }, [data, initialFile])

  if (loading) return <Spinner />
  if (error || !data) return <div className="hint">Scope not available.</div>
  if (data.files.length === 0) {
    return (
      <div className="hint" style={{ padding: 16 }}>
        No files in <code>{data.root}</code>. The live workspace is wiped between runs;
        mid-run artifacts live in the remote MCP sandbox and surface through agent
        observations in <b>Replay</b>. Per-run snapshots (if present) retain the outputs.
      </div>
    )
  }

  // A deep-linked file can exist on disk yet fall outside the capped ranked
  // listing; preview it anyway with a kind guessed from its extension.
  const cur = data.files.find((f) => f.path === selected)
    ?? (selected ? { path: selected, kind: guessKind(selected) } : undefined)
  return (
    <div className="split" style={{ ['--split-list' as string]: '300px' }}>
      <div className="card">
        <div className="card-head">files · ranked by relevance</div>
        <div style={{ maxHeight: '68vh', overflowY: 'auto' }}>
          {data.files.map((f) => (
            <button
              key={f.path}
              className="ws-row"
              data-active={f.path === selected}
              onClick={() => setSelected(f.path)}
            >
              <span className={`ws-kind k-${f.kind}`}>{f.kind}</span>
              <span className="ws-path">{f.path}</span>
              <span className="ws-size muted">{fmtBytes(f.size)}</span>
            </button>
          ))}
        </div>
      </div>
      <div>{cur ? <FilePreview scope={scope} file={cur.path} kind={cur.kind} /> : null}</div>
      <style>{STYLE}</style>
    </div>
  )
}

function FilePreview({ scope, file, kind }: { scope: string; file: string; kind: string }) {
  const url = workspaceFileUrl(scope, file)
  const [text, setText] = useState<string | null>(null)
  // A failed or non-OK fetch is its own explicit state — a stale ?file= deep
  // link must say "not retrievable", never render blank or show a 404 body.
  const [failed, setFailed] = useState(false)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (kind === 'image') return
    setLoading(true)
    setFailed(false)
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.text()
      })
      .then(setText)
      .catch(() => { setText(null); setFailed(true) })
      .finally(() => setLoading(false))
  }, [url, kind])

  return (
    <div className="card">
      <div className="card-head">
        <span>{file}</span>
        <a href={url} target="_blank" rel="noreferrer">open ↗</a>
      </div>
      <div className="card-body">
        {kind === 'image' && <img className="artifact-img" src={url} alt={file} />}
        {kind !== 'image' && loading && <Spinner />}
        {kind !== 'image' && !loading && failed && (
          <div className="hint">
            File not retrievable at this path in this scope — a stale deep link,
            or the snapshot changed since the link was made.
          </div>
        )}
        {kind !== 'image' && !loading && !failed && (
          TEXTY.has(kind)
            ? <SmartText text={text ?? ''} filename={file} kind={kind} />
            : <div className="hint">Binary file — <a href={url} target="_blank" rel="noreferrer">download</a>.</div>
        )}
      </div>
    </div>
  )
}

const STYLE = `
.ws-row { display: flex; align-items: center; gap: 8px; width: 100%; text-align: left;
  background: transparent; border: none; border-bottom: 1px solid var(--border);
  border-radius: 0; padding: 8px 11px; font-size: 12px; }
.ws-row:hover { background: var(--panel-2); }
.ws-row[data-active="true"] { background: #f2c14e10; }
.ws-path { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  font-family: var(--mono); }
.ws-size { flex: none; font-size: 10.5px; }
.ws-kind { flex: none; font-size: 9px; text-transform: uppercase; letter-spacing: 0.4px;
  padding: 1px 6px; border-radius: 4px; color: #0b0e14; font-weight: 700; }
.k-image { background: #f2c14e; } .k-report { background: #4ec98a; } .k-data { background: #5db8f0; }
.k-code { background: #c78bf0; } .k-json { background: #9aa7bd; } .k-other, .k-binary { background: #6b7688; }
`

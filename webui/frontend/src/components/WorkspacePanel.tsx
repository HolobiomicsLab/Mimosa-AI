import { useEffect, useState } from 'react'
import { api, workspaceFileUrl } from '../api'
import { useAsync } from '../hooks'
import type { WorkspaceListing } from '../types'
import { Spinner, fmtBytes, shortId } from '../ui'

const TEXTY = new Set(['report', 'code', 'json', 'data'])

export default function WorkspacePanel({ runId }: { runId: string }) {
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
      <div className="pill-row" style={{ marginBottom: 14 }}>
        <button className={`pill ${scope === 'live' ? 'active' : ''}`} onClick={() => setScope('live')}>
          live workspace
        </button>
        {snaps.map((s) => (
          <button key={s} className={`pill ${scope === s ? 'active' : ''}`} onClick={() => setScope(s)}>
            snapshot {s === runId ? '(this run)' : shortId(s)}
          </button>
        ))}
      </div>
      <ScopeView scope={scope} />
    </div>
  )
}

function ScopeView({ scope }: { scope: string }) {
  const { data, loading, error } = useAsync<WorkspaceListing>(() => api.workspaceFiles(scope), [scope])
  const [selected, setSelected] = useState<string | null>(null)

  useEffect(() => { setSelected(data?.auto_preview ?? null) }, [data])

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

  const cur = data.files.find((f) => f.path === selected)
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
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (kind === 'image') return
    setLoading(true)
    fetch(url).then((r) => r.text()).then(setText).catch(() => setText(null)).finally(() => setLoading(false))
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
        {kind !== 'image' && !loading && (
          TEXTY.has(kind)
            ? <pre className="code">{text}</pre>
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

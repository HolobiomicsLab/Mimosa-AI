import { useEffect, useMemo, useState } from 'react'
import { stripAnsi } from './ansi'
import CodeView from './CodeView'
import CsvTable from './CsvTable'
import JsonTree from './JsonTree'
import MarkdownView from './MarkdownView'

interface Props {
  text: string
  filename?: string
  /** Backend file kind (image/code/markdown/text/json/data/report/…). */
  kind?: string
}

type Mode =
  | { view: 'code'; lang: string }
  | { view: 'json' }
  | { view: 'markdown' }
  | { view: 'csv' }
  | { view: 'plain' }

function ext(filename?: string): string {
  const m = /\.(\w+)$/.exec(filename ?? '')
  return m ? m[1].toLowerCase() : ''
}

function parsesToJson(text: string): boolean {
  const t = text.trim()
  if (!t.startsWith('{') && !t.startsWith('[')) return false
  try {
    const v = JSON.parse(t) as unknown
    return typeof v === 'object' && v !== null
  } catch {
    return false
  }
}

function pickMode(text: string, filename?: string, kind?: string): Mode {
  const e = ext(filename)
  if (e === 'py') return { view: 'code', lang: 'python' }
  if (e === 'sh' || e === 'bash' || e === 'zsh') return { view: 'code', lang: 'bash' }
  if (e === 'r') return { view: 'code', lang: 'r' }
  if (e === 'json' || kind === 'json' || parsesToJson(text)) return { view: 'json' }
  if (e === 'md' || kind === 'markdown') return { view: 'markdown' }
  if (e === 'yaml' || e === 'yml') return { view: 'code', lang: 'yaml' }
  if (e === 'csv' || e === 'tsv' || kind === 'data') return { view: 'csv' }
  return { view: 'plain' }
}

const TOGGLE_LABELS: Partial<Record<Mode['view'], [string, string]>> = {
  json: ['tree', 'raw'],
  markdown: ['rendered', 'raw'],
  csv: ['table', 'raw'],
}

/** Dispatches a text blob to the richest sensible renderer, with a raw toggle. */
export default function SmartText({ text, filename, kind }: Props) {
  const mode = useMemo(() => pickMode(text, filename, kind), [text, filename, kind])
  const [raw, setRaw] = useState(false)

  useEffect(() => setRaw(false), [text, filename])

  const labels = TOGGLE_LABELS[mode.view]

  let body: React.ReactNode
  switch (mode.view) {
    case 'code':
      body = <CodeView text={text} lang={mode.lang} />
      break
    case 'json':
      body = raw ? <CodeView text={text} lang="json" /> : <JsonTree text={text} />
      break
    case 'markdown':
      body = raw ? <CodeView text={text} lang="markdown" /> : <MarkdownView text={text} />
      break
    case 'csv':
      body = raw ? <pre className="code">{text}</pre> : <CsvTable text={text} filename={filename} />
      break
    default:
      body = <pre className="code">{stripAnsi(text)}</pre>
  }

  return (
    <div className="smart-text">
      {labels && (
        <div className="mode-pills">
          <button className={`mode-pill ${raw ? '' : 'active'}`} onClick={() => setRaw(false)}>
            {labels[0]}
          </button>
          <button className={`mode-pill ${raw ? 'active' : ''}`} onClick={() => setRaw(true)}>
            {labels[1]}
          </button>
        </div>
      )}
      {body}
    </div>
  )
}

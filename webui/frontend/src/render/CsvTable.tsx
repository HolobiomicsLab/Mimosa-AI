import { useMemo } from 'react'

interface Props {
  text: string
  filename?: string
}

const MAX_ROWS = 200
const MAX_COLS = 40
const NUM_RE = /^-?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/

function sniffDelimiter(text: string, filename?: string): string {
  if (filename?.toLowerCase().endsWith('.tsv')) return '\t'
  const nl = text.indexOf('\n')
  const first = nl === -1 ? text : text.slice(0, nl)
  const tabs = (first.match(/\t/g) ?? []).length
  const commas = (first.match(/,/g) ??
    []).length
  return tabs > commas ? '\t' : ','
}

/** Split one line on `delim`, honouring simple double-quoted fields. */
function splitLine(line: string, delim: string): string[] {
  const out: string[] = []
  let cur = ''
  let quoted = false
  for (let i = 0; i < line.length; i++) {
    const c = line[i]
    if (quoted) {
      if (c === '"') {
        if (line[i + 1] === '"') { cur += '"'; i++ } else quoted = false
      } else cur += c
    } else if (c === '"') {
      quoted = true
    } else if (c === delim) {
      out.push(cur)
      cur = ''
    } else {
      cur += c
    }
  }
  out.push(cur)
  return out
}

/** Naive CSV/TSV preview: header + up to 200 rows × 40 cols. */
export default function CsvTable({ text, filename }: Props) {
  const { header, rows, nRows, nCols } = useMemo(() => {
    const delim = sniffDelimiter(text, filename)
    const lines = text.replace(/\r\n?/g, '\n').split('\n').filter((l) => l.length > 0)
    const all = lines.map((l) => splitLine(l, delim))
    const nCols = all.reduce((m, r) => Math.max(m, r.length), 0)
    return {
      header: (all[0] ?? []).slice(0, MAX_COLS),
      rows: all.slice(1, 1 + MAX_ROWS).map((r) => r.slice(0, MAX_COLS)),
      nRows: Math.max(all.length - 1, 0),
      nCols,
    }
  }, [text, filename])

  if (header.length === 0) return <div className="hint">Empty table.</div>

  const truncated = nRows > MAX_ROWS || nCols > MAX_COLS
  return (
    <div>
      <div className="csv-wrap">
        <table className="csv-table">
          <thead>
            <tr>{header.map((h, i) => <th key={i}>{h}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r, ri) => (
              <tr key={ri}>
                {r.map((cell, ci) => (
                  <td key={ci} className={NUM_RE.test(cell.trim()) ? 'num' : undefined}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {truncated && (
        <div className="csv-note">
          showing {Math.min(nRows, MAX_ROWS)} of {nRows} rows · {Math.min(nCols, MAX_COLS)} of {nCols} columns
        </div>
      )}
    </div>
  )
}

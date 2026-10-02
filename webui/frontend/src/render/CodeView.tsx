import { useEffect, useState } from 'react'
import { highlight } from './highlight'

interface Props {
  text: string
  lang: string
  maxHeight?: number
  wrap?: boolean
}

/** Syntax-highlighted code block. Shows a plain <pre class="code"> instantly,
 * then swaps in shiki HTML once the (lazily loaded) highlighter is ready. */
export default function CodeView({ text, lang, maxHeight = 460, wrap = true }: Props) {
  const [html, setHtml] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    let alive = true
    setHtml(null)
    highlight(text, lang).then((h) => { if (alive) setHtml(h) })
    return () => { alive = false }
  }, [text, lang])

  const copy = () => {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    }).catch(() => { /* clipboard unavailable */ })
  }

  return (
    <div className={`codeview ${wrap ? 'wrap' : 'nowrap'}`}>
      <button className="copy-btn" onClick={copy} title="copy to clipboard">
        {copied ? '✓ copied' : 'copy'}
      </button>
      {html ? (
        <div
          className="codeview-host"
          style={{ maxHeight }}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      ) : (
        <pre className={`code ${wrap ? '' : 'tight'}`} style={{ maxHeight }}>{text}</pre>
      )}
    </div>
  )
}

import { isValidElement } from 'react'
import type { ReactNode } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import CodeView from './CodeView'

function flattenText(n: ReactNode): string {
  if (n == null || typeof n === 'boolean') return ''
  if (typeof n === 'string' || typeof n === 'number') return String(n)
  if (Array.isArray(n)) return n.map(flattenText).join('')
  if (isValidElement(n)) return flattenText((n.props as { children?: ReactNode }).children)
  return ''
}

/** Pull the code text + fence language out of a <pre><code class="language-x"> pair. */
function extractFence(children: ReactNode): { text: string; lang: string } | null {
  const child = Array.isArray(children) ? children[0] : children
  if (!isValidElement(child)) return null
  const props = child.props as { className?: string; children?: ReactNode }
  const lang = /language-([\w+-]+)/.exec(props.className ?? '')?.[1] ?? 'text'
  return { text: flattenText(props.children).replace(/\n$/, ''), lang }
}

/** GFM markdown in the Observatory theme; fenced code goes through CodeView. */
export default function MarkdownView({ text }: { text: string }) {
  return (
    <div className="md-view">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          pre({ children }) {
            const fence = extractFence(children)
            if (fence) return <CodeView text={fence.text} lang={fence.lang} />
            return <pre className="code">{children}</pre>
          },
          a({ children, href }) {
            return <a href={href} target="_blank" rel="noreferrer">{children}</a>
          },
        }}
      >
        {text}
      </Markdown>
    </div>
  )
}

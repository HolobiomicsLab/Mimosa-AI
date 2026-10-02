import { useEffect, useState } from 'react'

export interface ToastAction { label: string; onClick: () => void }
interface ToastInput { kind: 'error' | 'info'; text: string; actions?: ToastAction[] }
interface ToastMsg extends ToastInput { id: number }

let deliver: ((t: ToastInput) => void) | null = null
let pending: ToastInput[] = []
let nextId = 0

/** Fire a toast from anywhere; queued until a ToastHost mounts, never dropped. */
export function toast(input: ToastInput): void {
  if (deliver) deliver(input)
  else pending.push(input)
}

/** Fixed overlay rendering toasts; sticky until dismissed or an action is taken. */
export function ToastHost() {
  const [toasts, setToasts] = useState<ToastMsg[]>([])

  useEffect(() => {
    deliver = (input) => setToasts((cur) => [...cur, { ...input, id: ++nextId }])
    if (pending.length) { pending.forEach(deliver); pending = [] }
    return () => { deliver = null }
  }, [])

  const dismiss = (id: number) => setToasts((cur) => cur.filter((t) => t.id !== id))

  return (
    <div className="toast-host">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`}>
          <span className="toast-text">{t.text}</span>
          <div className="toast-actions">
            {t.actions?.map((a) => (
              <button key={a.label} onClick={() => { a.onClick(); dismiss(t.id) }}>{a.label}</button>
            ))}
            <button className="toast-x" aria-label="dismiss" onClick={() => dismiss(t.id)}>✕</button>
          </div>
        </div>
      ))}
    </div>
  )
}

import { useCallback, useEffect, useRef, useState } from 'react'
import type { LiveEvent } from './types'

/** Fetch-on-mount helper with loading/error state and a manual refetch. */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const run = useCallback(() => {
    let alive = true
    setLoading(true)
    fn()
      .then((d) => alive && (setData(d), setError(null)))
      .catch((e) => alive && setError(String(e)))
      .finally(() => alive && setLoading(false))
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(run, [run])
  return { data, error, loading, refetch: run }
}

/** Subscribe to the backend live WebSocket; returns latest event + connected flag. */
export function useLive(onEvent?: (e: LiveEvent) => void) {
  const [connected, setConnected] = useState(false)
  const [last, setLast] = useState<LiveEvent | null>(null)
  const cb = useRef(onEvent)
  cb.current = onEvent

  useEffect(() => {
    let ws: WebSocket | null = null
    let retry: ReturnType<typeof setTimeout>
    let closed = false

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/api/live`)
      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!closed) retry = setTimeout(connect, 2000)
      }
      ws.onmessage = (ev) => {
        try {
          const e = JSON.parse(ev.data) as LiveEvent
          setLast(e)
          cb.current?.(e)
        } catch { /* ignore malformed frames */ }
      }
    }
    connect()
    return () => { closed = true; clearTimeout(retry); ws?.close() }
  }, [])

  return { connected, last }
}

import type {
  RunSummary, RunDetail, Tree, Series, MemoryList, Timeline,
  StepDetail, CallDetail, WorkspaceScopes, WorkspaceListing,
  SetupInfo, SetupConfig, KeyStatus, McpHealth, RefineResult, ClassifyResult,
  LaunchInfo, RunMode,
} from './types'

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText} — ${path}`
    try {
      const err = (await res.json()) as { detail?: unknown }
      if (err.detail) detail = String(err.detail)
    } catch { /* keep the status-line message */ }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

const get = <T,>(path: string) => request<T>('GET', path)

export const api = {
  runs: () => get<RunSummary[]>('/runs'),
  run: (id: string) => get<RunDetail>(`/runs/${id}`),
  tree: (id: string) => get<Tree>(`/runs/${id}/tree`),
  series: (id: string) => get<Series>(`/runs/${id}/series`),
  memory: (id: string) => get<MemoryList>(`/runs/${id}/memory`),
  timeline: (id: string) => get<Timeline>(`/runs/${id}/memory/timeline`),
  step: (id: string, agent: string, index: number) =>
    get<StepDetail>(`/runs/${id}/memory/step?agent=${encodeURIComponent(agent)}&index=${index}`),
  call: (id: string, name: string) =>
    get<CallDetail>(`/runs/${id}/memory/call/${encodeURIComponent(name)}`),
  workspaceScopes: () => get<WorkspaceScopes>('/workspace/scopes'),
  workspaceFiles: (scope: string) =>
    get<WorkspaceListing>(`/workspace/${encodeURIComponent(scope)}/files`),

  // ── setup & launch ──
  setup: () => get<SetupInfo>('/setup'),
  saveConfig: (patch: Partial<SetupConfig>) =>
    request<{ applied: Partial<SetupConfig>; config: SetupConfig }>('PATCH', '/setup/config', patch),
  saveKey: (name: string, value: string) =>
    request<{ name: string; saved_to: string; keys: KeyStatus }>('POST', '/setup/keys', { name, value }),
  mcpHealth: () => get<McpHealth>('/setup/mcp'),
  refine: (objective: string, history: { question: string; answer: string }[]) =>
    request<RefineResult>('POST', '/assist/refine', { objective, history }),
  classify: (objective: string) =>
    request<ClassifyResult>('POST', '/assist/classify', { objective }),
  launch: (params: { objective: string; mode: RunMode; learn: boolean; judge: boolean }) =>
    request<LaunchInfo>('POST', '/launches', params),
  launches: () => get<LaunchInfo[]>('/launches'),
  launchStatus: (id: string) => get<LaunchInfo>(`/launches/${encodeURIComponent(id)}`),
  cancelLaunch: (id: string) => request<LaunchInfo>('POST', `/launches/${encodeURIComponent(id)}/cancel`),
}

/** Absolute URL for a run artifact (image/text served directly by the API). */
export const artifactUrl = (id: string, name: string) => `/api/runs/${id}/artifacts/${name}`

/** Absolute URL for a workspace file within a scope. */
export const workspaceFileUrl = (scope: string, path: string) =>
  `/api/workspace/${encodeURIComponent(scope)}/file?path=${encodeURIComponent(path)}`

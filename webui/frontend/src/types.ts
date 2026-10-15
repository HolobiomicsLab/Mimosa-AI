// Mirrors the Observatory backend JSON contract (see webui/backend/app).

export type RunStatus = 'completed' | 'running' | 'crashed' | 'error'
export type EvolutionKind = 'seed' | 'mutation' | 'crossover' | null

export interface RunSummary {
  id: string
  created_at: string | null
  goal: string | null
  status: RunStatus
  score: number | null
  cost_usd: number | null
  iteration: number | null
  evolution_kind: EvolutionKind
  parents: string[]
  wall_time_s: number | null
  is_single_agent: boolean
}

export interface Artifact {
  name: string
  filename: string
  kind: 'image' | 'code' | 'markdown' | 'text' | 'json'
  size: number
  empty: boolean
}

// Policy-BT verifier verdict (webui/backend/app/verification.py). Pair sides
// are relabeled `self` (the viewed run) / `opponent` by the backend.
export type PairOutcome = 'win' | 'loss' | 'tie'
export type PairChannel = 'execution' | 'code' | 'visual'

export interface PolicyCriterion {
  id: string
  stage: number
  description?: string
  direction: 'higher' | 'lower' | 'satisfy'
}

/** One adapter/visual evidence entry; `detail` (code) or `observation` (visual). */
export interface PairEvidence {
  side: 'self' | 'opponent'
  criterion?: string
  path?: string
  value?: unknown
  detail?: string
  observation?: string
}

export interface PolicyPair {
  opponent: string
  opponent_reward: number | null
  outcome: PairOutcome
  stage: number
  channel: PairChannel | null
  policy_version: string | null
  feedback: string | null
  opponent_feedback: string | null
  evidence: PairEvidence[]
  values: { self: Record<string, unknown>; opponent: Record<string, unknown> } | null
}

export interface ProcessRecord {
  argv: string[] | null
  returncode: number | null
  timed_out: boolean | null
  error: string | null
  duration_s: number | null
  stdout: string | null
  stderr: string | null
}

export interface ReproductionContract {
  script?: string
  installs_requirements?: boolean
  regenerates_outputs?: boolean
  generated_outputs?: string[]
  reason?: string
}

export interface ExecutionRecord {
  eligible: boolean | null
  status: string | null
  reason: string | null
  enabled: boolean | null
  attempted: boolean | null
  script: string | null
  returncode: number | null
  timed_out: boolean | null
  duration_s: number | null
  preserved_outputs: string[] | null
  symlink_gaps: string[] | null
  contract: ReproductionContract | null
  files: RerunFileDiff | null
  provision: ProcessRecord | null
  run: ProcessRecord | null
}

/** Workspace diff of the rerun; path lists, `unchanged` collapsed to a count. */
export interface RerunFileDiff {
  created?: string[]
  changed?: string[]
  removed?: string[]
  regenerated?: string[]
  missing_outputs?: string[]
  inventory_errors?: Record<string, unknown>[]
  unchanged_count: number
}

export interface PolicyBTVerification {
  reward: number | null
  arrival_reward: number | null
  tournament_version: number | null
  ratings: Record<string, number>
  execution: ExecutionRecord
  pairs: PolicyPair[]
  criteria: PolicyCriterion[] | null
}

export interface RunDetail extends RunSummary {
  original_task: string | null
  textual_gradient: string | null
  evaluation_text: string | null
  verification: PolicyBTVerification | null
  genotype: string | null
  evolution_prompt: string | null
  metrics: Record<string, unknown> | null
  artifacts: Artifact[]
  family_size?: number
  learning_mode?: boolean
}

export interface TreeNode {
  id: string
  iteration: number | null
  evolution_kind: EvolutionKind
  created_at: string | null
  score: number | null
  status: RunStatus
  is_focus: boolean
}
export interface TreeEdge { source: string; target: string; kind: EvolutionKind }
export interface Tree { focus: string; nodes: TreeNode[]; edges: TreeEdge[] }

export interface SeriesPoint {
  uuid: string
  iteration: number | null
  evolution_kind: EvolutionKind
  overall_score: number | null
  overall_score_uncapped: number | null
  qd_score: number | null
  novelty_score: number | null
  cumulative_cost_usd: number | null
  iteration_cost_usd: number | null
  wall_time_s: number | null
  on_error: boolean | null
}
export interface Series { focus: string; points: SeriesPoint[] }

export interface MemoryAgent {
  name: string
  kind: 'agent'
  steps: number
  errored: boolean
  start_time: number | null
  total_tokens: number
}
export interface MemoryCall {
  name: string
  kind: 'llm_call'
  model: string | null
  created: number | null
  cost_usd: number | null
  total_tokens: number | null
}
export interface MemoryList { run_id: string; agents: MemoryAgent[]; calls: MemoryCall[] }

export interface TimelineStep {
  agent: string
  index: number
  step_number: number | null
  start_time: number | null
  duration: number | null
  is_final_answer: boolean
  error_type: string | null
  code: string
  output_text: string
  observations: string
  tokens: { input_tokens?: number; output_tokens?: number; total_tokens?: number }
  order: number
}
export interface Timeline { run_id: string; steps: TimelineStep[] }

export interface StepDetail {
  agent: string
  index: number
  step_number: number | null
  timing: { start_time?: number; end_time?: number; duration?: number } | null
  is_final_answer: boolean
  error: { type?: string; message?: string } | null
  code: string
  output_text: string
  observations: string
  action_output: string | null
  tokens: Record<string, number> | null
  input_messages: { role: string; content: string }[]
}

export interface CallDetail {
  name: string
  model: string | null
  provider: string | null
  temperature: number | null
  reasoning_effort: string | null
  created: number | null
  cost_usd: number | null
  tokens: { prompt: number | null; completion: number | null; total: number | null }
  messages: { role: string; content: string }[]
  response: string
}

export interface WorkspaceFile {
  path: string
  size: number
  mtime: number
  kind: string
  priority: number
}
export interface WorkspaceListing {
  scope: string
  root: string
  files: WorkspaceFile[]
  auto_preview: string | null
}
export interface WorkspaceScopes {
  live: { available: boolean; path: string }
  snapshots: string[]
}

export interface UploadedFile { name: string; size: number; kind: string }
export interface UploadResult { saved: UploadedFile[] }

export interface LiveEvent {
  type: 'iteration_complete' | 'execution_complete' | 'tree_updated'
    | 'run_finished' | 'archive_appended' | 'workflow_crafted'
  run_id: string | null
  filename: string
}

// ── Phase 2: setup & launch ──

export interface SetupConfig {
  workspace_dir: string | null
  planner_llm_model: string | null
  workflow_llm_model: string | null
  smolagent_model_id: string | null
  capsule_namer_model: string | null
  judge_model: string | null
  reasoning_effort: string | null
  max_tokens: number | null
  learned_score_threshold: number | null
  max_learning_evolve_iterations: number | null
  export_astra: boolean | null
  _config_path: string
  _config_exists: boolean
  _workspace_exists: boolean
}

export interface KeyStatus {
  keys: { name: string; present: boolean; source: 'env' | 'dotenv' | null }[]
  any: boolean
}

export interface SetupInfo {
  config: SetupConfig
  keys: KeyStatus
  presets: Record<'orchestration' | 'agent' | 'judge', string[]>
  bridge: { available: boolean; error: string | null }
}

export interface McpHealth {
  reachable: boolean
  open_ports: { host: string; port: number }[]
  scanned: number
  note: string | null
}

/** Bridge results: `available` is false when the Mimosa venv is missing;
 * `degraded` is true when the LLM/parse failed but a usable fallback was returned. */
export interface RefineResult {
  ok: boolean
  available?: boolean
  error?: string
  result?: {
    is_clear: boolean
    question: string | null
    refined_prompt: string
    degraded?: boolean
    note?: string | null
  }
}

export interface ClassifyResult {
  ok: boolean
  available?: boolean
  error?: string
  result?: {
    mode: 'task' | 'goal'
    confidence: number | null
    reasoning: string | null
    suggested_label: string | null
    degraded?: boolean
    note?: string | null
  }
}

export type RunMode = 'task' | 'goal'

export interface ObjectiveHistoryEntry {
  objective: string
  mode?: string | null
  timestamp?: string | null
}

export interface ObjectiveHistoryResult {
  ok: boolean
  available?: boolean
  error?: string
  result?: { entries: ObjectiveHistoryEntry[] }
}

export interface LaunchInfo {
  id: string
  pid: number
  running: boolean
  returncode: number | null
  failed: boolean
  /** 'workflow_generation' = the crafting LLM produced no usable workflow. */
  failure_hint: 'workflow_generation' | 'crash' | null
  error_line: string | null
  objective: string
  mode: RunMode
  learn: boolean
  judge: boolean
  started_at: string
  log: string
  log_tail?: string
}

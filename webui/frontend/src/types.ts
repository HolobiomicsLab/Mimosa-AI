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

export type ClaimStatus = 'pass' | 'fail' | 'error' | 'unsure'

export interface EvaluationClaim {
  id: string
  importance: number
  rationale: string
  description: string
  relevant_files: string[]
  kind: string | null
  status: ClaimStatus | null
  score: number | null
  details: string | null
}

export interface RunDetail extends RunSummary {
  original_task: string | null
  textual_gradient: string | null
  evaluation_text: string | null
  evaluation_scores: Record<string, unknown> | null
  evaluation_claims: EvaluationClaim[] | null
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
    | 'step_appended' | 'llm_call_logged' | 'gradient_updated'
    | 'evaluation_updated' | 'astra_updated' | 'evaluation_capsule_updated'
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

// ── Provenance: the run's ASTRA capsule + independent evaluations ──

export interface AstraDecisionOption {
  label?: string
  description?: string
  excluded_reason?: string
}

export interface AstraDecision {
  label?: string
  rationale?: string
  default?: string | number | null
  options?: Record<string, AstraDecisionOption>
  /** The model that made this decision, when the exporter recorded it. */
  model?: string
}

export interface AstraUniverse {
  id?: string
  description?: string
  decisions?: Record<string, unknown>
}

export interface AstraCapsule {
  name: string | null
  description: string | null
  version: string | null
  decisions: Record<string, AstraDecision>
  outputs: unknown[]
  universes: AstraUniverse[]
}

export interface ProvenanceFlag {
  kind?: string
  artefact?: string
  detail?: string
}

export interface EvalVerdict {
  check_id?: string
  status?: string
  method?: string
  criterion?: string
  failure_kind?: string | null
  observed?: unknown
  expected?: unknown
  note?: string
  evidence?: string[]
  provenance_flags?: ProvenanceFlag[]
}

export interface JudgeVerdict {
  item_id?: string
  layer?: string
  verdict?: string
  rationale?: string
  confidence?: number | null
  evidence_cited?: string[]
}

export interface JudgeLayer {
  instrument?: Record<string, unknown>
  summary?: Record<string, unknown>
  refused?: { item_id?: string; reason?: string }[]
  verdicts?: JudgeVerdict[]
}

export interface RunEvaluation {
  source: string
  name: string | null
  criteria_source: string | null
  independent_of_subject: boolean | null
  summary: Record<string, unknown>
  verdicts: EvalVerdict[]
  workspace_flags: ProvenanceFlag[]
  target_conflicts: unknown[]
  judge: JudgeLayer | null
}

export interface Provenance {
  run_id: string
  astra: AstraCapsule | null
  /** Family members holding the capsule, when this run has none of its own. */
  family_capsules: string[]
  evaluations: RunEvaluation[]
}

// ── evolution replay + QD atlas ─────────────────────────────────────────────

/** One family member, annotated with everything the replay narrates. */
export interface EvolutionNode {
  id: string
  iteration: number | null
  evolution_kind: EvolutionKind
  created_at: string | null
  score: number | null
  status: RunStatus
  is_focus: boolean
  parents: string[]
  score_uncapped: number | null
  qd_score: number | null
  novelty_score: number | null
  iteration_cost_usd: number | null
  cumulative_cost_usd: number | null
  wall_time_s: number | null
  on_error: boolean | null
  claims: { passed: number; failed: number; error: number; unsure: number } | null
  gradient_snippet: string | null
  selection: {
    improvement_type?: string | null
    delta_reward?: number | null
    is_validated?: boolean | null
    confidence?: number | null
    admit_rejected?: boolean | null
  } | null
}

export interface FamilyEvolution {
  focus: string
  nodes: EvolutionNode[]
  edges: { source: string; target: string; kind: EvolutionKind }[]
}

/** One run projected onto the 2-D PCA of Mimosa's own QD behaviour space. */
export interface AtlasPoint {
  id: string
  x: number
  y: number
  score: number | null
  iteration: number | null
  evolution_kind: EvolutionKind | null
  family: number | null
  started_at: string | null
  cost: number | null
}

export interface AtlasData {
  points: AtlasPoint[]
  edges: { source: string; target: string }[]
  skipped: { id: string; reason: string }[]
  variance_explained: number[]
  n_dimensions: number
}

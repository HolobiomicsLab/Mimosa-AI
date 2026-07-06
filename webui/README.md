# Mimosa Observatory

A web frontend for **Mimosa-AI** — the self-evolving multi-agent system for
scientific research. Mimosa is otherwise CLI-only; the Observatory renders what
a run produces so you can watch and inspect evolution instead of reading logs.

Phases 1 and 2 of the agreed build order are done: **observability** (a
read-only view over the artifacts Mimosa writes to disk, plus a live feed
derived from filesystem changes) and **setup & launch** (the CLI onboarding as
a web flow: config, API keys, objective refinement, mode suggestion, and
subprocess-isolated run launching). Neither changes Mimosa's core. Phase 3
(in-process event bus, live control) builds on this — see *Roadmap* below.

```
webui/
  backend/   FastAPI read-only API over sources/workflows + sources/memory
  frontend/  React + Vite + TypeScript app (lineage tree, replay, workspace…)
```

## What it shows

For every evolution run under `sources/workflows/<uuid>/`:

- **Overview** — the task, the verifier evaluation (claim pass/fail), the
  textual gradient that steers the next mutation, and the evolution prompt.
- **Workflow** — the executed LangGraph diagram (`workflow_<uuid>.png`) next to
  the generated genotype source.
- **Lineage** — the evolution family as an interactive tree (reconstructed from
  each run's `lineage_*.json` parent links via connected components), coloured by
  score using Mimosa's own red→green scale, plus a reward/QD/novelty/cost curve
  built from `run_metrics.json` across the family. Click a node to open that run.
- **Replay** — every agent step across all `task_*.json` traces, merged and
  ordered by real `timing.start_time` (not filename order), with a per-step
  inspector (executed code, reasoning, observations, action output, errors,
  tokens, and the full model message history) and cards for each single-shot LLM
  call (prompt, response, cost, tokens).
- **Workspace** — the toolomics workspace and per-run `/tmp` snapshots, files
  ranked by a relevance heuristic (figures first, then fresh/large data).
- **Artifacts** — a raw browser over every file in the run dir.

Plus two pages that replace the CLI onboarding:

- **Setup** — API-key status and entry (values written to the same dotenv files
  the CLI uses, never echoed back), the model role slots (orchestration / agent
  / judge, mirroring the CLI's grouping), the toolomics workspace path, learning
  knobs, and an MCP port scan.
- **New run** — a three-step wizard: objective → LLM clarifier loop (same
  prompts as the CLI, executed in the Mimosa venv via the bridge) → goal-vs-task
  suggestion with manual override, learning/judge toggles, launch. Launched runs
  are detached subprocesses; the monitor polls status and a plain-text log tail,
  and can cancel the process group.

A live indicator reflects a WebSocket that emits semantic events
(`iteration_complete`, `execution_complete`, `tree_updated`, `run_finished`,
`archive_appended`, `workflow_crafted`) from filesystem changes; the run list
refetches on run-level events.

## Running it

**Backend** (Python ≥ 3.11, uses `uv`):

```bash
cd webui/backend
uv run uvicorn app.main:app --port 8848
```

By default it reads `~/Documents/CNRS/Mimosa-AI/sources/{workflows,memory}` and
the toolomics workspace. Override any location with env vars:

| Env var | Default | Meaning |
| --- | --- | --- |
| `MIMOSA_ROOT` | `/Users/mlg/Documents/CNRS/Mimosa-AI` | Mimosa checkout/install to observe |
| `MIMOSA_WORKFLOW_DIR` | `$MIMOSA_ROOT/sources/workflows` | per-run evolution artifacts |
| `MIMOSA_MEMORY_DIR` | `$MIMOSA_ROOT/sources/memory` | per-agent traces |
| `MIMOSA_WORKSPACE_DIR` | `…/toolomics/workspace` | shared toolomics workspace |
| `MIMOSA_SNAPSHOT_GLOB` | `/tmp/mimosa_run_*` | per-run workspace snapshots |
| `MIMOSA_CONFIG` | `$MIMOSA_ROOT/config_default.json`, else `~/.config/mimosa/config.json` | Mimosa config the setup page edits |
| `MIMOSA_PYTHON` | `$MIMOSA_ROOT/.venv/bin/python` | Python that can import Mimosa (refine/classify/launch bridge) |

**Frontend** (Node ≥ 20):

```bash
cd webui/frontend
npm install
npm run dev          # http://localhost:5173, proxies /api + /api/live to :8848
```

Point the dev proxy elsewhere with `MIMOSA_API=http://host:port npm run dev`.

## API surface

```
GET  /api/health
GET  /api/runs                              list + status/score/cost/kind
GET  /api/runs/{id}                         full detail (+ genotype, eval, gradient)
GET  /api/runs/{id}/tree                    evolution family graph (nodes+edges)
GET  /api/runs/{id}/series                  reward/QD/novelty/cost per iteration
GET  /api/runs/{id}/artifacts[/{name}]      list / serve an artifact file
GET  /api/runs/{id}/memory                  agents + LLM-call summaries
GET  /api/runs/{id}/memory/timeline         ordered compact agent steps
GET  /api/runs/{id}/memory/step?agent=&index=   full step detail
GET  /api/runs/{id}/memory/call/{name}      full LLM-call detail
GET  /api/archive                           shared QD archive feed
GET  /api/workspace/scopes                  live + snapshot scopes
GET  /api/workspace/{scope}/files[?…]       ranked file listing / file content
WS   /api/live                              filesystem-derived event stream

GET   /api/setup                            config + key status + presets + bridge health
PATCH /api/setup/config                     merge-patch the editable config subset
POST  /api/setup/keys                       upsert an API key into the dotenv files
GET   /api/setup/mcp                        TCP-probe the MCP discovery port range
POST  /api/assist/refine                    one clarifier round {objective, history}
POST  /api/assist/classify                  goal-vs-task suggestion {objective}
POST  /api/launches                         start a run {objective, mode, learn, judge}
GET   /api/launches[/{id}]                  list / poll (with log tail)
POST  /api/launches/{id}/cancel             SIGTERM→SIGKILL the process group
```

Everything above the break is read-only; the setup/launch block is the only
part that writes (config file, dotenv, spawned run processes).

## Design notes learned from Mimosa's internals

- **Each evolution iteration is its own run dir.** There is no per-iteration
  history inside one dir; the tree is the connected component of the parent
  graph across sibling dirs. The Observatory reconstructs this itself.
- **`state_result.json` is empty/absent until a run finishes** — used as the
  running/crashed sentinel. Evaluation scores inside it are stored as strings
  and are coerced by the backend.
- **Agent memory is written at agent-end, not per step**, and can be multi-MB
  (`task_builder.json` was 5 MB); the API returns compact step summaries and
  fetches full content per step, with an mtime cache.
- **The live workspace is wiped/restored between runs**; durable outputs are the
  `/tmp` snapshots, and mid-run files live in the remote MCP sandbox (visible
  only through agent `execute_command` observations in Replay).

## Roadmap

- **Phase 3 — Live control plane.** Add a small structured event bus inside
  Mimosa (planner steps, evolution iterations, per-agent steps via a smolagents
  step callback) that layers richer semantic events onto the same WebSocket
  channel — enabling the live planner-step panel and per-step streaming that the
  filesystem feed can only approximate.

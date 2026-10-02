# Mimosa Observatory

A web frontend for **Mimosa-AI** — the self-evolving multi-agent system for
scientific research. Mimosa is otherwise CLI-only; the Observatory renders what
a run produces so you can watch and inspect evolution instead of reading logs.

Phases 1 and 2 of the agreed build order are done: **observability** (a
read-only view over the artifacts Mimosa writes to disk, plus a live feed
derived from filesystem changes) and **setup & launch** (the CLI onboarding as
a web flow: config, API keys, objective refinement, input-file upload into
the workspace, mode suggestion, and subprocess-isolated run launching).
Neither changes Mimosa's core. Phase 3
(in-process event bus, live control) builds on this — see *Roadmap* below.

```
webui/
  backend/   FastAPI API over sources/workflows + sources/memory, setup & launch
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
- **Provenance** — the run rendered FROM its structured record rather than
  from its own reporting: the transparency exporter's ASTRA capsule
  (`runs_capsule/<uuid>/astra.yaml` — decisions with alternatives, rationale,
  and universes; a run without its own capsule links to the family members
  that have one) beside every independent `asb_eval` evaluation capsule
  (`eval_astra.yaml`) that names the run — executor verdicts over the ASB
  card's own criteria, workspace provenance flags, and the pinned-instrument
  judge layer, which is never merged into the executor score. Directories are
  overridable via `MIMOSA_CAPSULE_DIR` / `MIMOSA_EVAL_DIR` (defaults:
  `runs_capsule/`, `evaluations/`).
- **Evolution** — the family's search replayed as an animated lineage: nodes
  reveal in evolution order (play/pause/step/speed/scrub), the best-so-far
  badge chases the frontier, and the info panel narrates each run — score with
  delta vs parent, claim counts, QD/novelty, cost, the selection log, and the
  textual gradient that steered the next mutation
  (`GET /api/runs/{id}/evolution` joins all of it per node).
- **Artifacts** — a raw browser over every file in the run dir.

Everywhere text is shown, content renders by type (`frontend/src/render/`):
shiki syntax highlighting for code, a collapsible JSON tree, GFM markdown,
CSV/TSV table previews, ANSI-stripped logs — each with a raw toggle.

Two cross-run surfaces:

- **QD Atlas** (`/atlas`, `GET /api/atlas/{space}`) — every run PCA-projected
  to 2D with parent→child trails, colour by score/family/iteration, family
  filter, fleet time-replay, zoom/pan, and a per-family **trajectory mode**
  that steps point-by-point along a comet trail. Two spaces: `qd` (Mimosa's
  384-dim behaviour descriptor — task-level, so one family's runs coincide)
  and `genotype` (TF-IDF of the evolved workflow code, where within-family
  drift is visible).
- **Live activity** — the backend watches all four artifact roots
  (workflows, memory, `runs_capsule`, evaluations) and streams semantic
  events over the existing `/api/live` WebSocket (steps appended, gradient
  written, ASTRA capsule updated, …). The run page shows a live feed that
  diffs the ASTRA decision layer on every capsule update and auto-refreshes
  the workspace and provenance panels.

Plus two pages that replace the CLI onboarding:

- **Setup** — API-key status and entry (values written to the same dotenv files
  the CLI uses, never echoed back), the model role slots (orchestration / agent
  / judge, mirroring the CLI's grouping), the toolomics workspace path, learning
  knobs, and an MCP port scan.
- **New run** — a three-step wizard: objective → LLM clarifier loop (same
  prompts as the CLI, executed in the Mimosa venv via the bridge) → goal-vs-task
  suggestion with manual override, learning/judge toggles, launch. The objective
  step can also upload input files straight into the live workspace — the folder
  Mimosa snapshots as the run's initial state at launch — and lists/removes
  what's already there. Launched runs are detached subprocesses; the monitor
  polls status and a plain-text log tail, and can cancel the process group.

A live indicator reflects a WebSocket that emits semantic events
(`iteration_complete`, `execution_complete`, `tree_updated`, `run_finished`,
`archive_appended`, `workflow_crafted`) from filesystem changes; the run list
refetches on run-level events.

## Deployment

The Observatory is a **single-operator, localhost tool with no
authentication**. It's meant to run on the same machine as the Mimosa
checkout you're observing, reachable only from `localhost`. Don't expose it
on a shared or public network as-is.

### Prerequisites

- Backend: Python ≥ 3.11, [`uv`](https://docs.astral.sh/uv/).
- Frontend: Node ≥ 20.

### Install

```bash
cd webui/backend && uv sync
cd webui/frontend && npm install
```

### Run

Start both servers with the helper script (it installs dependencies on first
run, then runs the backend and frontend together; `Ctrl-C` stops both):

```bash
cd webui && ./deploy.sh            # open http://localhost:5173
./deploy.sh --check                # preflight + install only, don't start servers
```

Ports are overridable via `MIMOSA_BACKEND_PORT` / `MIMOSA_FRONTEND_PORT`. Or run
the two servers by hand, in separate shells:

```bash
# Backend — from webui/backend
uv run uvicorn app.main:app --host 127.0.0.1 --port 8848

# Frontend — from webui/frontend
npm run dev          # http://localhost:5173, proxies /api + /api/live to :8848
```

By default the backend reads `~/Documents/CNRS/Mimosa-AI/sources/{workflows,memory}`
and the toolomics workspace. The frontend dev server proxies `/api` (REST plus
the `/api/live` WebSocket) to the backend on `:8848`, so the app itself only
uses same-origin relative URLs. Point the proxy elsewhere with
`MIMOSA_API=http://host:port npm run dev`.

### Environment

Every variable is optional; unset ones fall back to the defaults below,
which assume a local Mimosa checkout at the given path.

| Env var | Default | Meaning |
| --- | --- | --- |
| `MIMOSA_ROOT` | `/Users/mlg/Documents/CNRS/Mimosa-AI` | Mimosa checkout/install to observe |
| `MIMOSA_WORKFLOW_DIR` | `$MIMOSA_ROOT/sources/workflows` | per-run evolution artifacts |
| `MIMOSA_MEMORY_DIR` | `$MIMOSA_ROOT/sources/memory` | per-agent traces |
| `MIMOSA_WORKSPACE_DIR` | `…/toolomics/workspace` | shared toolomics workspace |
| `MIMOSA_SNAPSHOT_GLOB` | `/tmp/mimosa_run_*` | per-run workspace snapshots |
| `MIMOSA_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | allowed dev frontend origins |
| `MIMOSA_CONFIG` | `$MIMOSA_ROOT/config_default.json`, else `~/.config/mimosa/config.json` | Mimosa config the setup page edits |
| `MIMOSA_PYTHON` | `$MIMOSA_ROOT/.venv/bin/python` | Python that can import Mimosa (refine/classify/launch bridge) |

### Observability vs. setup & launch

The observability endpoints (runs, tree, series, artifacts, memory, workspace
browsing, live feed) and the live-workspace upload need only the backend's own
dependencies — the backend never imports `torch`, `smolagents`, or Mimosa
itself.

The setup and launch endpoints are different: objective refinement, goal/task
classification, and launching runs shell out to the real Mimosa install via
`MIMOSA_PYTHON`. Those features additionally require a working Mimosa install
with its own `.venv` and provider API keys available (in the project `.env`
or `~/.config/mimosa/.env`). If `MIMOSA_PYTHON` doesn't resolve to a usable
interpreter, the setup page still lets you view and edit config, but refine,
classify, and launch are disabled and the UI says so.

### Production notes

There is currently no Dockerfile, docker-compose file, systemd unit, nginx
config, or CI in this repo for the Observatory. The commands above (running
`uvicorn` directly and `npm run dev`) are the only supported way to run it
today.

For a real deployment, build the frontend (`npm run build`, static assets
land in `webui/frontend/dist/`) and serve `dist/` from any static server or
CDN, with the backend reachable at the same origin through a reverse proxy
you provide (or point the built frontend at the backend's URL directly). The
backend does **not** currently serve the built frontend itself — there is no
static-file mount in `app.main`. None of this reverse-proxy/static-serving
infrastructure is shipped; it's on the operator to add, and the localhost/
single-operator/no-auth posture above still applies once you do.

See [`webui/backend/README.md`](./backend/README.md) and
[`webui/frontend/README.md`](./frontend/README.md) for backend- and
frontend-specific detail (tests, lint, build flags).

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
POST  /api/workspace/upload                 multipart run-input upload into the live workspace
DELETE /api/workspace/live/file?path=…      remove one live-workspace file
```

Everything above the break is read-only; the block below it is the only part
that writes (config file, dotenv, spawned run processes, live-workspace
uploads).

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

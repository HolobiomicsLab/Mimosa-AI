# AGENTS.md — Mimosa-AI

Guidance for AI coding agents working in this repository. Assume no prior
knowledge of the project.

## Project overview

**Mimosa-AI** is an open-source Python framework for **autonomous scientific
research** (Holobiomics Lab, CNRS — Apache 2.0, companion paper
arXiv:2603.28986). For each task it:

1. **Synthesizes a custom multi-agent workflow as plain Python source code**
   (no DSL, no YAML — "code-as-genotype"),
2. **Runs it in a sandbox** (Hugging Face SmolAgents `LocalPythonExecutor`
   with an AST allow-list, plus shared LangGraph state),
3. **Verifies** the run with the policy/BT verifier
   (`sources/evaluators/policy_bt_verifier/`, the default): each
   generation's workspace is snapshotted and must carry a reproduction
   contract — a `.sh` script that installs requirements and regenerates
   every output (missing or invalid → reward 0); an LLM reconnaissance
   inventories the artifacts, per-schema adapters normalize layouts,
   and one cached deterministic comparison policy decides every pair
   (vision-model tie-break when both sides have relevant images); the
   reward is a regularized Bradley–Terry fit over all pair verdicts,
   refreshed for every generation of the task session.
4. **Evolves** the workflow across generations with a Quality-Diversity
   inspired search (unstructured archive, `qd_score = (1−w)·quality + w·novelty`,
   code-embedding novelty via k-NN cosine distance, directive-driven
   mutation magnitude, crossover).

The mutator is steered by the verifier's `abstracted_textual_gradient`
(elimination-point-first feedback) — raw scores and scorer code stay
inside the verifier.

Five-layer architecture (details in `docs/concepts/architecture.md`):

| Layer | Component | Role |
|-------|-----------|------|
| 0 | Planner (`sources/core/planner.py`) | Goal → task decomposition (only in `--goal` mode) |
| 1 | ToolManager + Perspicacité client | MCP tool discovery on configured IP/port ranges; optional literature grounding |
| 2 | EvolutionEngine (`sources/core/evolution_engine.py`) | Workflow synthesis + QD evolution loop |
| 3 | WorkflowRunner (`sources/core/workflow_runner.py`) | Sandboxed execution of generated workflows |
| 4 | PolicyBTVerifierEvaluator (`sources/evaluators/policy_bt_verifier/`) | Verification driving the next mutation: snapshot + reproduction contract (`.sh` rerun in a clean venv) → LLM recon → adapters + one comparison policy → regularized Bradley–Terry reward over all pair verdicts |

### Verifier research program (TL;DR for successor agents)

The policy/BT verifier above is not a guess — it productionizes the offline E44–E48
architecture (fresh re-execution + deterministic digest + LLM policy tournament), which
was itself designed from a 14-experiment component research program (E12–E24 + E19c,
2026-09-23 → 09-25) in `experiments_verifiers/` (full table, codenames and one-liners in
its README; human report: `experiments_verifiers/REPORT_cycle2_E12-E19.html`). The hybrid
temporal-ladder verifier (E19/E19b) was the previous production channel; it stays in-tree
(`sources/evaluators/hybrid_verifier/`) but runs are made with `verifier_kind="policy_bt"`.
What was tried and learned, in order:

- **Weight learning is closed.** E12 ‘HAL’ ran the pre-registered falsifier for within-task weight
  fitting (FAIL, inside the ±4-task noise band) and E13 ‘NEO’ swept model capacity (every upgrade
  ≤ linear L3) — re-weighting the same claims cannot fix picking. L3 stays useful BETWEEN tasks
  only (budgeting/early-stop).
- **Pairwise comparators need a contract.** E18 ‘JOSHUA’ (LLM writes a script comparing both
  workspaces per pair) was the worst ranker (0.480) — its scripts extracted task-invariant values
  and invented unstable preference heuristics (E25 ‘CREED’ autopsy: decisive accuracy 0.44 below
  chance; 8/9 order-flips changed the deciding metric itself). Same-policy-per-side scoring
  (E19 ‘AVA’) fixed the symmetry; a fixed claim ladder fixed the ties.
- **Claims content is the lever.** Ranking limits come from WHAT is checked, not how it's
  compared: E19b ‘TERMINATOR’ proved content upgrades move feedback relevance (0.627 → 0.672) but
  not ranking; the E25 autopsy showed the quality signal (headline loss/AUC) was extracted in 3 of
  130 decisions.
- **The reward is a regularized Bradley–Terry fit over all pair verdicts.** Every evaluation
  refreshes the current rating of every generation of the task session; `arrival_reward` keeps
  the rating at arrival time. E19c 'RATCHET' measured raw win-rate ranking better than an
  unregularized BT strength fit on the frozen arena (0.742 vs 0.643, paired CI excludes 0) —
  the fitted-probability rating is the shipped policy_bt reward
  (`policy_bt_reward = "bradley_terry"`, the only allowed value; `"win_rate"` raises an
  error). Survival-vs-a-growing-field is still the signal: ineligible generations sit at
  reward 0.
- **The production channel is the policy/BT tournament (E44–E48 design).** Verifier-owned
  snapshot copies, gated by a reproduction contract (an LLM-assessed `.sh` script that
  installs requirements and regenerates outputs; missing/invalid → reward 0 and a gradient
  naming the fix), optional rerun with bash in a fresh venv with no API keys
  (`policy_bt_reexecute`, default on — not an OS sandbox; run logs copied to
  `.policy_bt_reexec/`), ONE LLM reconnaissance (complete inventory, bounded previews,
  explicit byte-page inspection), per-schema adapters plus ONE cached comparison policy run
  in a restricted interpreter subprocess (both orders + self-tie checks; E48 safeguards:
  static screen, runtime smoke, `policy_bt_repair_rounds` repairs), kimi per-pair visual
  tie-break when both sides have relevant images (`policy_bt_visual_channel`), and the
  E47 evidence-checked gradient. Persistence: a per-task-session registry
  (sha256(goal, session) keyed) stores snapshots/pairs/ratings/policy, so predecessor
  comparisons never re-read `/tmp` or the live workspace. The E44–E48 digest/metrics
  modules no longer exist — digest responsibilities live in `artifacts.py`/`recon.py`,
  pair logic in `policy.py`, gradients in `results.py`.
- **Feedback quality is separately measurable.** E16 ‘ANDERTON’ built a validated gradient-vs-gold
  judge (E16v2, two-axis calibration gates) — a standing metric for every future campaign; the
  legacy verifier's gradients hallucinated 65% of requirements vs the hybrid's ~25%.
- **Execution is a gate, not a judge.** E24 ‘TARS’: 29% of workspaces fail clean re-execution —
  use re-execution as an admission cap; its holdout channel is real but limited (5/19 groups).
- **Adversarial hardening is designed, not run**: E20 ‘OCEAN’ (sabotage validation), E21 ‘COBB’
  (prompt injection through artifacts), E22 ‘FRANK’ (deceptive gradients), E23 ‘MALCOLM’
  (Goodhart pressure test) — see `experiments_verifiers/HANDOFF_V2.md`.

Rules of thumb: pickers are saturated (8–9/19 vs oracle 12 — don't build more pickers); feedback
quality and between-task budgeting are where improvements land; any new verification idea must
beat the E19c numbers on the frozen arena (`experiments_verifiers/results/E19c_ratchet_*.md`)
before touching production.

## Tech stack

- **Language:** Python, **requires `>=3.11,<3.12`** (workflow runners
  separately default to Python 3.12, see `runner_default_python_version`).
- **Package/build manager:** `uv` (see `uv.lock`); build backend `hatchling`.
  Console script: `mimosa = main:cli_main`.
- **Key dependencies:** `smolagents[litellm,mlx-lm,telemetry,mcp]`, `langgraph`,
  `litellm` + `openai` (multi-provider LLM access), `fastmcp` (MCP client),
  `sentence-transformers` (local `all-MiniLM-L6-v2` genotype embeddings),
  `python-a2a`, `torch`, `matplotlib`/`imageio`/`pillow` (visualization),
  `pygame`, `psutil`.
- **External services expected at runtime** (not part of this repo):
  - **Toolomics** — companion platform exposing scientific tools as MCP
    servers; discovery scans the `discovery_addresses` ranges (default
    `0.0.0.0:5000–5200`). Any reachable MCP server works; Toolomics is not
    strictly required.
  - **Perspicacité-AI** — optional literature-grounding service, picked up
    automatically when running.
- **Optional web UI ("Observatory"):** `webui/` — FastAPI backend +
  React/Vite/TypeScript frontend, single-operator localhost tool with **no
  authentication** (never expose it publicly). Start with `cd webui && ./deploy.sh`.
- **Docs site:** MkDocs Material (`mkdocs.yml`, content in `docs/`).

## Repository layout

```
main.py                      # CLI entry point & mode dispatch (argparse → async main)
config.py                    # Config dataclass: defaults, JSON load, CLI overrides, path validation
config_default.json          # Template config — copy to my_config.json and edit
auto-install.sh              # Automated installer script
cleanup.sh                   # Deletes top-level generation folders of sources/workflows/ + tmp/;
                             # `./cleanup.sh <subfolder>` empties one campaign folder; -n = dry run
workflow_evolution_anim.py   # Interactive evolution-tree / trace replay viewer

sources/
  core/                      # Evolution engine, selection, variation, planner,
                             # factories, workflow runner, tools manager, LLM provider
  evaluators/                # Policy/BT verifier package (the verifier in use), hybrid
                             # verifier (previous channel), legacy claims verifier
                             # (deprecated), scenario/generic evaluators
  benchmark_evaluation/      # CSV batch mode, ScienceAgentBench/PaperBench grading,
                             # capsule evaluation, CodeBERT scorer, execution sandbox
  cli/                       # OnboardCLI (zero-arg wizard), EvaluationCLI, MemoryChatCLI, theming
  extensibility/             # Human (no-LLM manual) mode, text-to-speech hook
  modules/                   # Code TEMPLATES injected into generated workflows
                             # (state_schema.py, smolagent_factory.py) — do not lint
  prompts/                   # Markdown system prompts (workflow_v*.md, planner_*.md) — do not lint
  security/                  # Pre-flight package vetting (check_package.py)
  transparency/              # ASTRA export, decision extraction, memory traces, YAML writer
  utils/                     # paths (XDG handling), pricing, logging, notify (Pushover),
                             # workspace management, precheck, perspicacite client, plots
  workflows/                 # RUNTIME: generated workflow genotypes, per-UUID run folders
  memory/                    # RUNTIME: LLM call cache + memory traces (JSON)
  cache/                     # RUNTIME: e.g. cached OpenRouter pricing

tests/                       # pytest suite (mixed naming: test_*.py and *_test.py)
datasets/                    # Benchmark suites + scenarios/. ScienceAgentBench/ contains
                             #   datasets/, eval_programs/ (the ACTUAL graders), gold_programs/,
                             #   gold_results/ (gold artifacts), scoring_rubrics/
webui/                       # Observatory web UI (FastAPI backend + React frontend)
docs/                        # MkDocs documentation; DEVELOPER_GUIDE.md is the deep dive
runs_capsule/                # RUNTIME: archived final-workspace snapshots (per task)
tmp/                         # RUNTIME: WorkflowRunner temp dirs
run_notes/, logs/            # RUNTIME: per-campaign task records and log files
audits/                      # Audit reports (single-file offline HTML) + analysis scripts
CLA/                         # Contributor agreements (legal, not code)
```


## Build, install, and run commands

```bash
# Install deps (uv creates .venv from uv.lock)
uv sync

# Run from a checkout
uv run main.py                          # interactive onboarding wizard
uv run main.py --task "..."             # single task
uv run main.py --goal "..."             # planner mode (multi-step objective)
uv run main.py --task "..." --learn     # evolve across generations
uv run main.py --task "..." --single_agent   # single-agent baseline
uv run main.py --manual                 # no-LLM manual MCP tool shell
uv run main.py --papers datasets/x.csv --csv_runs_limit 10   # batch CSV eval
uv run main.py --science_agent_bench --csv_runs_limit 10     # benchmark
uv run main.py --memory_cli             # RAG chat over a finished run's memory

# Install as a standalone command (usable from any directory)
uv tool install .           # then: mimosa --task "..."

# Docs
uvx --with mkdocs-material mkdocs serve
```

Execution requires **at least one LLM API key** in `.env` (repo root, or
`~/.config/mimosa/.env` for installed runs): `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `HF_TOKEN`, or
`OPENROUTER_API_KEY`. Optional: `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_PRIVATE_KEY`
(telemetry), `PUSHOVER_USER`/`PUSHOVER_TOKEN` (notifications).

## Configuration

- `config.py` holds built-in defaults; `config_default.json` is the template
  to copy and edit. Load with `--config my_config.json`; many fields also have
  CLI override flags (`--workflow_dir`, `--runner_default_timeout`, …).
- Key fields: `workspace_dir` (must point at the Toolomics workspace),
  `discovery_addresses` (MCP scan ranges), `workflow_llm_model` /
  `smolagent_model_id` / `judge_model` / `planner_llm_model` (litellm-style
  model IDs), `verifier_kind` (`"policy_bt"`, the default in both
  `config.py` and the shipped `config_default.json` — the policy/BT
  knobs are the `policy_bt_*` fields), `learned_score_threshold` (early stop), and the QD knobs
  (`population_size`, `novelty_weight`, `novelty_k_neighbours`,
  `crossover_rate`, `max_learning_evolve_iterations`, …).
- **Repo checkouts vs installed runs** (`sources/utils/paths.py`): checkouts
  keep mutable state in-tree (`sources/workflows`, `sources/memory`, `tmp/`,
  `runs_capsule/`); installed `mimosa` runs use XDG dirs
  (`~/.config/mimosa`, `~/.local/share/mimosa`, `~/.cache/mimosa`). Presence
  of `pyproject.toml` next to the package is what identifies a checkout.
- The wheel excludes `sources/workflows`, `sources/memory`, `sources/cache`
  and force-includes `main.py` + `config.py`.

## Testing

- Framework: **pytest** (dev dependency group: `pytest>=9.1.1`).
- Test files live flat in `tests/` and use **both** `test_*.py` and
  `*_test.py` naming — match the existing convention of nearby tests when
  adding new ones. Tests are offline unit tests (mocked LLM/provider calls);
  they do not require API keys, MCP servers, or Toolomics.

```bash
uv run pytest tests/                    # all tests
uv run pytest tests/evaluator_test.py   # single file
```

- End-to-end "evaluation" is a separate, LLM- and benchmark-driven activity
  (`--papers`, `--science_agent_bench`), not part of the pytest suite. Run
  `./cleanup.sh` (or `./cleanup.sh <workflow_dir subfolder>`) before benchmark
  sweeps so cached workflows are not reused; never run it on a campaign you
  still need to audit.

## Auditing finished runs

Finished benchmark runs are **fully auditable** from the artifacts in this
repository — the verifier's reward, the benchmark's own grades, and the
physical outputs can be reconstructed and cross-checked. This section maps the
evidence, gives the audit procedure, and defines what a good audit report
looks like. Audit folders are **read-only evidence** — never modify, move, or
delete anything inside them.

Every campaign writes enough artifacts to reconstruct it end to end: what the
verifier rewarded, what the benchmark actually graded, and what the workflows
physically produced. Auditing means aligning those three records and exposing
every divergence. Never modify, move, or delete anything inside those
folders, and never "clean up" a run folder to make a report prettier —
they are the evidence.

> **Verifier artifact formats.** Runs are made with the policy/BT
> verifier (`verifier_kind="policy_bt"`, the default in `config.py` and
> `config_default.json`): `evaluation.txt` opens with the line
> `Policy-BT verifier` followed by a JSON object (reward, tournament
> version, the reproduction/execution record, the `input_provenance`
> check of the task inputs against the transfer manifest, and the PAIR VERDICTS —
> each pair tagged with its `policy_version`), the gradient lands in
> `textual_gradient.txt`, stage-level JSON records stream to
> `verifier_policy_bt.log`, and the machine-readable pair evidence is
> in `policy_bt_evidence.json` (same folder). The tournament state for
> a task session lives in
> `<workflow_dir>/.policy_bt/<sha256(goal, session)>/tournament.json`
> with one verifier-owned `snapshots/<uuid>/` copy per generation (plus
> `pristine_inputs/`, the task's input tables as the benchmark transfer gave
> them); its `bundle.criteria_guard` records why each frozen criterion was kept,
> revised or dropped (`criteria_guard` verdicts, `plan_revisions`,
> `dropped_criteria`, and `quality_gap` when a held-out quality criterion for
> a placeholder test label stayed missing).
> Older campaigns on disk were produced by earlier channels: the hybrid
> temporal-ladder verifier (stage headers and win-rate fields) and,
> before it, the legacy claims verifier (per-source claim verdicts with
> importance ratings and the 0.89 hard-fail cap, deprecated 2026-09-24).
> Identify the channel from the `evaluation.txt` header before
> interpreting a run — see `docs/concepts/evaluation-pipeline.md`.

### Evidence map

| Path | What it holds | Audit notes |
|------|---------------|-------------|
| `run_notes/run_<x>/<task>.json` | Per-task campaign record and the **ground truth**: `goal`, `capsule_name`, `evolved_workflows_uuids` (ordered, oldest → newest), `evolution_rewards` / `evolution_costs` (the verifier signal), and `ablations[]` — the benchmark's own grader re-scored per generation (`evolution_index`, `uuid`, `VER`, `VER_message`, `SR`, `SR_message`, `CBS`, `cost`, `status`, `source`). Totals: `total_eval`, `ver_success`, `sr_success`, `avg_cbs`, `max_judge_reward`, `git`. | `ablations` outranks the verifier reward — it is what the benchmark actually graded. `source` is `snapshot` (genotype re-graded) or `capsule` (final submission). `ver_success` / `sr_success` / `total_eval` semantics are undocumented: quote them, don't interpret them. |
| `sources/workflows/run_<x>/<uuid>/` | Per-generation evolution artifacts: `evaluation.txt` (verifier verdicts — policy/BT: reward + tournament version + execution record + PAIR VERDICTS; older runs: hybrid stage headers or legacy per-claim verdicts), `textual_gradient.txt` (policy/BT: the LATEST fold of this generation's gradient over all current pairs — what the NEXT mutation of this parent receives; earlier folds with time and `tournament_version` in `textual_gradient_history.jsonl`, the first one also in `state_result.json` `evaluation.verifier.arrival_textual_gradient`), `verifier_policy_bt.log` (per-stage JSON records), `policy_bt_evidence.json` (same report, machine-readable), `lineage_<id>.json` (kind + parents), `run_metrics.json` (`overall_score`, `on_error`, costs), `run_stdout.log` (raw workflow execution output, verbatim incl. ANSI escapes; on failed runs the error text; when this generation is later a parent without a verifier gradient, only its TAIL — the last 4096 characters (`RAW_LOG_TAIL_CHARS`) after a marker line — is the unverified `<raw_log>` the directive LLM or crossover prompt receives; the full file is not what the LLM saw), `workflow_genotype_<id>.py`, `evolution_prompt_*`, `state_result.json` (clean runs). | **Empty folder** (since 2026-10-14: only `run_stdout.log` starting `WORKFLOW_GENERATION_ERROR:`) = failed generation attempt. Since 2026-10-20 the run-notes slot records that folder's own uuid, or a fresh `generation_failed_<YYYYMMDD_HHMMSS>_<8hex>` folder when the factory made none; older notes record the literal placeholder `generation_failed`, whose shared folder keeps only the last failure's log. A round with `on_error: true` has no `evaluation.txt` and no gradient — read `run_stdout.log` for the run-time error (folders older than 2026-10-14 lack it; then the cause is not recorded anywhere — say so instead of guessing). |
| `runs_capsule/run_<x>/<capsule>/` | The final submitted workspace: final script(s), `pred_results/` artifacts, and **`evaluation_results.json`** — the official VER / SR / CBS / cost plus the full VER/SR messages that produced the task's benchmark score. | The capsule defines the task result, not the last generation folder. Read the prediction artifacts' headers/first rows; they ground every claim about what was delivered. |
| `/tmp/mimosa_run_<wshash>_<uuid>/` | The per-round **run workspace** — the files the workflow physically wrote that round (scripts, `pred_results/`, reports). Match rounds by uuid substring. | **Volatile**: `cleanup.sh`, reboots, and benchmark sandboxes remove them (`/tmp/mimosa_sandbox_*` are transient and usually already gone). Record listings, sizes, and output-file headers at inspection time — an audit is a dated snapshot. Artifact presence ≠ run success: errored rounds (`on_error: true`) can still leave complete outputs. |
| `datasets/ScienceAgentBench/eval_programs/*.py` (+ `gold_results/`, `gold_programs/`, `scoring_rubrics/`) | The **actual benchmark graders** and their gold artifacts. | The highest-value audit resource. `SR_message` formats map 1:1 to eval scripts — e.g. `overlap: X` → `antibioticsai_filter_eval.py`, a hardcoded `N/A` → `eval_mat_feature_select.py`, a long critique list → the `gpt4_visual_judge` vision path. Read the grader code and its gold file before interpreting ANY SR result; the message alone can be uninformative by design. |
| `audits/` | Audit reports and analysis scripts (e.g. `reward_signal_analysis/`). | New audit reports belong here. Exemplars to extend, not rewrite: `mimosa_verifier_steering_report.html` (campaign dossier) and the per-task `*_sr_explainer.html` deep dives. |


### Audit procedure (the loop that finds everything)
1. **Inventory** the campaign: list `run_notes/run_<x>/*.json`; extract goal,
   uuid list, rewards, ablations, capsule_name.
2. **Start investigation of why SR success/failure**; Find the final delivered capsule matching the `capsule_name` in  `run_capsule/run_<x>/<name>`; Find the real grader and gold reference file in `datasets/ScienceAgentBench/`. Compare the gold with what pred_results the agents produced, compare the gold script against the produced script.
3. **Understand deeply** what justify the `SR_message`, compare the reference files and produced files and identify the root failure cause.
4. **Understand this**: The final artefacts from the best rewarded workflow following the QD evolution are the one that will be evaluated and give the final verdict on VER/SR, but we also do ablations of per-evolution iteration results on VER/SR, these ablations are what the `ablations` field of `run_capsule/run_<x>/<name>` is for. If you need the per evolution index capsule backup in addition the ablation verdicts they are in ``/tmp/mimosa_run_<wshash>_<uuid>` (Confirm still available, `/tmp/` is sometimes cleaned up)
4. **Follow up with evolution investigation**: we need to understand how come the final artefact produced by the most evolved and best scoring workflow, therefore we now investigate per evolution-iteration produced artefacts and how aligned evolution was with winning VER/SR.
- 5. **Join evolution iterations** versus produced workflows and artefacts by **uuid**. `ablations[i].evolution_index` equals the position in `evolved_workflows_uuids` — verify this; runs exist with gaps.
- 6. For each evolved uuid **Find it's final deliverable**: per evolution iterations produced artefacts by agents in `/tmp/mimosa_run_<wshash>_<uuid>/`; compare the artefact script and pred_results with the gold script and gold results. Understand how it differs from the previous iteration (better? worst?)
- 7. Understand why the results artefacts differs from reading the artefacts themselves: open up `sources/workflows/run_<x>/<uuid>` for the previous and current workflow
8. **Read the evaluations** (`evaluation.txt`) of the previous and current iteration: which pairs were won/lost and on which decision rung, how did that change between the 2 evolution iterations ? does it drive toward or away from winning the task (according to gold) ? The evaluation will drive the gradient.
9. **Read the gradient** of round N-1 as the child received it. Policy-BT refolds a parent's gradient from all its current pairs every time the tournament changes: `textual_gradient.txt` is the LATEST fold, `state_result.json` `evaluation.verifier.arrival_textual_gradient` is the first one, and `textual_gradient_history.jsonl` lists every fold with an ISO `time` and `tournament_version` — pick the last entry before the child's `lineage_<id>.json` `created_at`; if none precedes it (generations evaluated before 2026-10-17 have no arrival entry), the child received `arrival_textual_gradient`. Did the mutation act on the diagnosis? Did the loop close (failure → gradient → repair)? Flag reward/benchmark misalignement
10. **Look at the directive**: extract and have a look at `evolution_prompt_20260920_181842_d8e1c063.md` directive field <directive></directive> for the workflow N (not N-1), it's the directive that produced the current workflow, is the directive aligned ?
10. **Diff the workflow**: diff workflow N-1 against workflow N, what change in the workflow were made to fix the issues ?
10. **Reflect on the broader picture**: did evolution help ? construct a table for every iterations ? Investigate the deep cause of misalignement. Flag win and encouraging behaviors.

### Writing the audit report

Reports are **single-file offline HTML** in `audits/` — no CDNs or external
assets, works by double-click, and a `@media print` rule expands every
collapsed section for PDF export. System fonts, restrained styling, dense
tables avoided. Required structure and rules:

- **TL;DR first**: three stat cards (the task / what evolution optimized /
  why SR passed or failed) plus a one-sentence answer. The reader gets the
  finding in 30 seconds, then descends.
- **Cite everything**: file path + JSON key or line number for every quote;
  paths and uuids rendered as copyable `<code>` elements.
- **Steering table**: one row per generation — reward | ablation VER/SR/CBS |
  mutation lineage | one-line verifier summary | one-line benchmark message —
  sortable, with rows clickable into a per-round **evidence triad**:
  ① verifier says (claims + gradient), ② benchmark says (VER/SR + verbatim
  messages), ③ workspace shows (`/tmp` listing, capsule artifacts).
- **Long quotes collapse**: one-line excerpt, click to expand the verbatim
  text (tracebacks trimmed to the exception line in tables; full text behind
  the expand).
- **Quantify divergences**: tabulate the reproduced grader metric per round
  (overlap, RMSE, AUC, judge score) next to the recorded verdicts.
- **Synthesis section**: counted facts only ("in N of M tasks …", recurring
  claim names, gradient headlines), followed by one clearly labeled
  observation.
- **State gaps, volatility, and the build date.** An audit is a snapshot of
  moving evidence; date it and list what was already gone.

The steering table aligns the verifier with the benchmark *per generation*.
A second report type answers the operator question directly: **between two
successive iterations, what did the loop actually do?** Build one for any run
with more than one generation per task, and merge it with the steering report
when both are wanted.

Structure — one card per consecutive slot pair (gen N → gen N+1), five labeled
blocks, in this order:

1. **① Textual gradient** — the gradient of the generation being improved as
   the child received it (policy/BT: the last `textual_gradient_history.jsonl`
   entry before the child's `lineage_<id>.json` `created_at`; older runs:
   `textual_gradient.txt`), quoted verbatim, with its file path.
   If that round is `on_error` it has no gradient (no `evaluation.txt` either):
   state that instead of substituting anything.
2. **② Directive** — the `<directive>...</directive>` block of the child's
   `evolution_prompt_<uuid>.md`, quoted verbatim. State absence explicitly
   (no block vs no prompt file).
3. **③ Verifier verdict Δ** — parsed `evaluation.txt` of both sides: claim
   counts (pass/fail/error/unsure) and overall score before → after, the
   claim-level flips (name, importance, status before → after, capped with a
   "+N more"), new/gone claims, plus reward Δ and benchmark ablation
   VER/SR/CBS/cost Δ with both messages (one-line excerpt, click-to-expand).
4. **④ Genotype diff** — unified diff of `workflow_genotype_<uuid>.py`
   (+/− line counts, truncated with a marker when long). Default base: the
   previous slot. If that slot failed, diff against the declared lineage
   parent, else the **last real genotype**, and LABEL which base was used and
   why; quote both file paths as copyable `<code>`.
5. **⑤ Outputs Δ + reference** — recursive size listing of both
   `/tmp/mimosa_run_*_<uuid>/` workspaces (added / removed / changed, sizes),
   diff of generated top-level scripts when both exist; for the final
   generation, a "final submission vs reference" block: capsule
   `evaluation_results.json` (VER/SR/CBS/cost + verbatim messages), capsule
   artifact listing, the grader's expected `pred_results/` paths and metric
   lines quoted from the eval program **with line numbers** and a present/
   MISSING check, artifact shapes (PNG pixel size, CSV rows/columns, line
   counts) explicitly labeled "my computation", and the gold program / rubric
   / dataset paths cited. Missing grader gold artifacts are a stated gap.

Also mandatory:

- **Show structural facts instead of staying silent.** If the mode makes the
  steering question vacuous, demonstrate it: one-shot runs have no mid-run
  gradients — quote every **terminal gradient** (written after the last
  genotype, never consumed) in an appendix card. Reseed chains (failed slots
  between real generations) are shown as transitions with the base labeled.
  A duplicated run-note gets a banner (state which keys differ/are identical);
  unattributed folders get their own lineage section; a task still running is
  reconstructed from disk along lineage edges, dated, with live verifier
  scores as disk observations and no invented verdicts.
- **Slot chip strip per task**: one chip per slot — reward + VER/SR
  color-coded, failed slots marked, each linking to its card; a "final" chip
  links to the reference block.
- **Interactive single-file HTML**, same constraints as above (no external
  assets, inline CSS/JS): global transition filters (all / with verdict flips /
  paid-while-SR-false / with code diff / failed slots) that hide empty task
  sections, expand-all/collapse-all, copy buttons on every path and uuid
  (clipboard API with `execCommand` fallback for `file://`), and a
  `@media print` rule that expands every collapsed block and hides controls.
- Same conduct as any audit: cite file path + JSON key or line number for
  every quote, label recomputed numbers "my computation", keep a dedicated
  gaps section, and never judge chemical correctness.

## Code style guidelines

- **Ruff** is the only linter/formatter. Config in `pyproject.toml`:
  line-length 88, 4-space indent, rules `E,F,UP,B,SIM,I` (pycodestyle,
  Pyflakes, pyupgrade, bugbear, simplify, isort), `fix = true`, `E501` and
  `SIM103` ignored. Run `ruff check` / `ruff format` before submitting.
- Ruff **excludes** `main.py`, `sources/modules` (workflow code templates),
  `sources/prompts`, `sources/workflows` (dynamically generated genotype
  files), `sources/memory`, and `tmp` — do not "fix" style in those; they
  are runtime artifacts or injected code, not hand-maintained sources.
- Google-style docstrings on public functions (pydocstyle convention is
  configured but the `D` ruleset is currently commented out).
- Config plumbing follows a strict pattern: default in `config.py` →
  overridable from the JSON config file → optional CLI flag override via
  `add_config_arguments` / `apply_config_overrides` in `main.py`.
- Keep PRs minimal and documented; the full checklist (docs updates, mermaid
  diagram regeneration, PR description sections) is in
  `docs/DEVELOPER_GUIDE.md`.

## Security considerations

- **Never commit secrets.** `.env`, `token.json`, and `credentials.json` at
  the repo root hold live API keys/credentials; treat them as local-only.
  Installed-run secrets belong in `~/.config/mimosa/.env`.
- Generated workflows execute **LLM-written code**. Execution goes through
  the SmolAgents sandbox (`LocalPythonExecutor` with an AST allow-list) in
  `sources/core/workflow_runner.py`, with resource limits
  (`runner_default_timeout`, `runner_default_max_memory_mb`,
  `runner_default_max_cpu_percent`). Do not bypass or weaken this sandbox.
  **Exception — verifier checks are NOT sandboxed**: the policy/BT
  verifier's LLM-written comparison policies
  (`sources/evaluators/policy_bt_verifier/policy.py`) run as plain
  subprocesses of the host interpreter (`sys.executable`, fresh empty
  temp cwd, PATH-only environment — no API keys; imports statically
  screened to a stdlib allowlist, reflective calls/dunders/write APIs
  banned, subprocess timeout). Its fresh re-execution channel
  (`policy_bt_verifier/reexec.py`) provisions a per-task local `./venv`
  and pip-installs the workspace's requirements under era pins. The
  previous hybrid channel runs its scorer scripts as host subprocesses
  the same way, and the deprecated legacy verifier pip-installs
  LLM-vetted recovery packages into the host environment (see
  `docs/concepts/evaluation-pipeline.md`).
- `sources/security/check_package.py` provides pre-flight dependency vetting
  (version blocklisting for known supply-chain CVEs, e.g. the litellm 1.82.8
  `.pth` attack — hence the `litellm>=1.77.7,<1.92` pin in the runner
  requirements).
- The web UI has **no authentication** and launches runs as local
  subprocesses — localhost single-operator use only.
- MCP discovery scans configured IP/port ranges; keep `discovery_addresses`
  restricted to trusted hosts.

## Contributing notes

- Non-trivial external contributions require signing the Individual CLA
  (enforced by the `CLA Check` GitHub workflow via CLA Assistant Lite);
  employment-related contributions may need the Employer Authorization.
  Documents are in `CLA/`; contact `dr20.spv@listes.cnrs.fr`.
- Before a PR: `pytest tests/` passes, ruff-clean, docstrings on public
  functions, relevant `docs/` pages updated (and `.mermaid` diagrams
  re-rendered if architecture changed).
- The canonical deep-dive for architecture, execution flow, and evaluation
  is `docs/DEVELOPER_GUIDE.md`; concept docs live under `docs/concepts/`.
  Note: a few paths in the developer guide are historical (e.g. it says
  `sources/evaluation/` — the actual directory is
  `sources/benchmark_evaluation/`); trust the repository over the docs.


## Skills to read

Always read your engineering-standards skill before doing any tasks.
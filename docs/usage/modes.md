# Execution modes

Mimosa-AI is a single entry point — `main.py` — that dispatches between
modes based on the flag you pass. This page summarizes what each does.

## Quick reference

| Mode flag | What it runs |
| --------- | ------------ |
| *(no args)* | Interactive **onboarding wizard** (`OnboardCLI`). |
| `--evaluation_cli` | Interactive **benchmark launcher** (`EvaluationCLI`). |
| `--task "…"` | One-shot single task. |
| `--goal "…"` | Planner → multi-task plan → evolve each. |
| `--manual` | Manual CLI to debug MCPs and exercise tools. |
| `--papers <csv>` | Batch run over a CSV of (paper, task) rows. |
| `--science_agent_bench` | ScienceAgentBench batch evaluation. |
| `--workflow_eval_mode` | Workflow-generation-quality evaluation only. |
| `--scenario <path>` | Run a scenario rubric instead of LLM-judged scoring. |

Add `--learn` to any of the goal / task / batch flags to enable iterative
learning. Add `--single_agent` for the single-agent baseline.

The full flag list — including overrides for config fields — is in the
[CLI reference](../reference/cli.md).

## Interactive onboarding (zero args)

```bash
uv run main.py
```

Drops you into `OnboardCLI` ([`sources/cli/onboard_cli.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/cli/onboard_cli.py)).
The wizard walks through workspace selection, model choice, key sanity
checks, then either kicks off a task or saves a config file for later.

**Recommended for first-time users** — everything below assumes you've
already configured Mimosa once.

## Task mode

```bash
uv run main.py --task "Train a multitask model on the Clintox dataset…" \
              --config my_config.json
```

What happens:

```
EvolutionEngine.start_workflow_evolution(goal)
  ├─ reset session archive
  ├─ first run: seed prompt OR template mutation
  └─ evolve_generation()                          # depth-first recursion
      ├─ orchestrate_workflow()
      │   ├─ Perspicacite grounding
      │   ├─ workflow_factory.craft_workflow()
      │   └─ workflow_runner.execute()            # sandbox
      ├─ WorkflowEvaluator.evaluate(verifier)
      ├─ SelectionPressure.validate_survivor()
      ├─ record_lineage()
      ├─ pick next parent
      ├─ choose mutation vs. crossover (~30 %)
      └─ recurse until threshold OR max_depth
  ↓
WorkspaceManager.restore_best(best_uuid)
```

Use task mode when you can describe the work in one sentence and don't need
the planner to break it down.

!!! tip "Goals as files"
    Both `--task` and `--goal` accept a file path *or* a literal string. If
    the argument is a readable file, its contents are loaded as the goal.

## Goal mode

```bash
uv run main.py --goal "Reproduce experiments from arXiv:2306.00306 and compare results."
```

Adds Layer 0 — the **Planner** ([`sources/core/planner.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/planner.py))
— before evolution:

```
Planner.start_planner(goal)
  ├─ generate multi-step plan
  ├─ human approval prompt
  └─ for each step:
      └─ start_workflow_evolution(step_task)
  ↓
LocalTransfer.transfer_workspace_files_to_capsule()
```

Each sub-step runs the same evolution loop as `--task`. Use goal mode for
**reproductions of papers** or for objectives that span multiple
operations.

## Manual mode

```bash
uv run main.py --manual
```

Drops you into `HumanMode` ([`sources/extensibility/human_mode.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/extensibility/human_mode.py)).
There's no LLM involvement — you call discovered MCPs by hand, useful for:

- Verifying a new Toolomics tool exposes the right interface.
- Debugging discovery (which servers are visible? on which port?).
- Probing the workspace state at a known point.

## Batch CSV / benchmark modes

For systematic evaluation on a CSV of tasks:

```bash
# Custom benchmark
uv run main.py --papers datasets/our_benchmark.csv --csv_runs_limit 10 --learn

# ScienceAgentBench
uv run main.py --science_agent_bench --csv_runs_limit 102 --learn
```

Concurrency is controlled by `config.max_concurrent_eval_tasks` (default
`1`) and `config.task_start_delay` between launches. See the dedicated
pages for each:

- [ScienceAgentBench](../science_agent_bench_evaluation.md)
- [PaperBench](../papers_bench_evaluation.md)
- [Custom benchmarks](../evaluation/custom-benchmarks.md)

## Scenario mode

```bash
uv run main.py --scenario datasets/scenarios/<scenario>.json
```

Loads a rubric file and runs the task with `ScenarioEvaluator` — useful
when you have a hard ground-truth assertion list and want pass/fail scoring
instead of LLM-judged scoring.

## Single-agent baseline

```bash
uv run main.py --task "…" --single_agent
```

Replaces the multi-agent factory with `SingleAgentFactory`. Fast and cheap,
but **cannot improve through learning** — included so benchmark comparisons
between single-agent and multi-agent modes are honest.

## Workflow-eval mode

```bash
uv run main.py --workflow_eval_mode --csv_runs_limit 50
```

Runs `WorkflowEval` ([`sources/evaluation/eval_workflow_generation.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/evaluation/eval_workflow_generation.py))
which scores the *quality of workflow generation itself* — useful when
iterating on prompt templates.

## See also

- [CLI reference](../reference/cli.md) — every flag, fully documented.
- [Iterative learning](learning.md) — `--learn` deep dive.
- [Workspace & audit trail](workspace.md) — where the run artefacts go.

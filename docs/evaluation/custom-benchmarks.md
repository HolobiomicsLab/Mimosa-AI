# Custom benchmarks

Run Mimosa-AI on your own CSV of tasks. Useful for internal benchmarks,
private datasets, or domain-specific reproductions.

## Minimum CSV schema

`--papers` mode expects a CSV with at least:

| Column | Required | Description |
| ------ | -------- | ----------- |
| `task` or `prompt` | yes | Task description that becomes Mimosa's goal. |
| `instance_id` | no | Stable identifier — used in capsule names. |
| `expected_output` | no | Reference output for soft-claim grounding. |
| `domain_knowledge` | no | Extra context fed to the workflow factory. |

(Mimosa is forgiving about column names — see
[`sources/utils/dataset.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/utils/dataset.py)
for the exact column mapping logic.)

## Running it

```bash
uv run main.py --papers datasets/my_benchmark.csv \
              --csv_runs_limit 20 \
              --config my_config.json
```

With iterative learning:

```bash
uv run main.py --papers datasets/my_benchmark.csv \
              --csv_runs_limit 20 \
              --learn \
              --config my_config.json
```

## What you'll get

Each row produces:

- A capsule in `runs_capsule/<capsule_name>/`.
- An `evaluation_results.json` per capsule with per-claim scores and cost.
- Workspace snapshots and lineage records under `sources/workflows/`.

An aggregate summary prints to the console at the end of the run.

## Concurrency

By default `--papers` runs one task at a time. Bump
`max_concurrent_eval_tasks` in your config if your API budget allows:

```json
{
  "max_concurrent_eval_tasks": 4,
  "task_start_delay": 15.0
}
```

`task_start_delay` staggers launches so concurrent runs don't all hit the
same MCP discovery + LLM-warmup spike.

## Scenario-based evaluation

If you have a hard ground-truth assertion list per task instead of free-form
expected outputs, write a **scenario file** (JSON) and run:

```bash
uv run main.py --scenario datasets/scenarios/my_scenario.json
```

The scenario evaluator scores against the rubric directly — no LLM judge.
See [`sources/evaluation/scenario_loader.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/evaluation/scenario_loader.py)
for the loader and [`sources/core/evaluators/scenario.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/evaluators/scenario.py)
for the scoring logic.

## Reset between runs

```bash
./cleanup.sh
```

Wipes `sources/workflows/`, the LLM cache, and `runs_capsule/`. Do this
before any benchmark you want to publish — without it, cached workflows
from prior runs can leak through similarity-based parent selection.

## See also

- [ScienceAgentBench](../science_agent_bench_evaluation.md) — reference batch.
- [PaperBench](../papers_bench_evaluation.md) — paper-replication batch.
- [Workspace & audit trail](../usage/workspace.md) — where the results live.

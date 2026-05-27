# CLI reference

Every command-line flag accepted by `main.py`, grouped by purpose.

## Synopsis

```bash
uv run main.py [--config <path>] <mode> [<options>]
```

If no mode is given, Mimosa drops into the interactive onboarding wizard.

## Mode flags

| Flag | Description |
| ---- | ----------- |
| `--task TASK` | Run one task without planning. `TASK` is a string or a path to a file containing the goal. |
| `--goal GOAL` | Run with the planner: decompose, get human approval, then evolve each step. Accepts a string or a file path. |
| `--manual` | Manual CLI (no LLM) for debugging MCPs and tools. |
| `--papers <CSV path>` | Batch evaluation on a CSV of (task, expected output) rows. |
| `--science_agent_bench` | Batch evaluation on `datasets/ScienceAgentBench.csv`. |
| `--workflow_eval_mode` | Workflow-generation-quality evaluation. |
| `--scenario <path>` | Use a scenario rubric instead of LLM-judge scoring. |
| `--evaluation_cli` | Interactive evaluation launcher for ScienceAgentBench. |

## Behaviour modifiers

| Flag | Description |
| ---- | ----------- |
| `--learn` | Enable iterative learning until `learned_score_threshold` is reached or `max_learning_evolve_iterations` is hit. |
| `--single_agent` | Use the single-agent factory (no learning, baseline for benchmarks). |
| `--disable_judge` | Skip the verifier (no scoring, no learning). |
| `--debug` | Verbose debug logging to console. |
| `--verbose` | Enable info-level logging to console. |
| `--csv_runs_limit N` | Cap on rows in batch / CSV modes (default `200`). |

## Config overrides

These override values from `--config`'s JSON or the defaults in
`config.py`. Useful for one-off runs.

| Flag | Overrides |
| ---- | --------- |
| `--config <path>` | Load JSON config from `<path>`. |
| `--workflow_dir <path>` | `config.workflow_dir`. |
| `--schema_code_path <path>` | `config.schema_code_path`. |
| `--smolagent_factory_code_path <path>` | `config.smolagent_factory_code_path`. |
| `--prompt_workflow_creator <path>` | `config.prompt_workflow_creator`. |
| `--runner_default_python_version <ver>` | `config.runner_default_python_version`. |
| `--runner_default_timeout <s>` | `config.runner_default_timeout`. |
| `--runner_default_max_memory_mb <mb>` | `config.runner_default_max_memory_mb`. |
| `--runner_default_max_cpu_percent <pct>` | `config.runner_default_max_cpu_percent`. |
| `--runner_temp_dir <path>` | `config.runner_temp_dir`. |
| `--pushover_token <token>` | `config.pushover_token`. |
| `--pushover_user <user>` | `config.pushover_user`. |
| `--max_evolve_iterations N` | `config.max_learning_evolve_iterations`. |

## Examples

=== "First-time setup"

    ```bash
    uv run main.py
    ```
    Launches the onboarding wizard.

=== "One task, no learning"

    ```bash
    uv run main.py \
      --task "Train a multitask model on the Clintox dataset to predict drug toxicity." \
      --config my_config.json
    ```

=== "One task, with learning"

    ```bash
    uv run main.py \
      --task "Train a multitask model on the Clintox dataset…" \
      --learn \
      --config my_config.json
    ```

=== "Reproduce a paper"

    ```bash
    uv run main.py \
      --goal "Reproduce experiments from arXiv:2306.00306." \
      --config my_config.json
    ```

=== "ScienceAgentBench batch"

    ```bash
    uv run main.py --science_agent_bench --csv_runs_limit 102 --learn
    ```

=== "Custom CSV"

    ```bash
    uv run main.py --papers datasets/my_benchmark.csv \
      --csv_runs_limit 20 \
      --learn \
      --config my_config.json
    ```

=== "Single-agent baseline"

    ```bash
    uv run main.py --science_agent_bench --csv_runs_limit 7 --single_agent
    ```

=== "Override runner limits"

    ```bash
    uv run main.py --task "…" \
      --runner_default_timeout 7200 \
      --runner_default_max_memory_mb 4096
    ```

## See also

- [Execution modes](../usage/modes.md) — what each mode actually does.
- [Configuration reference](configuration.md) — full config field list.
- [Configuration](../getting-started/configuration.md) — day-to-day config tips.

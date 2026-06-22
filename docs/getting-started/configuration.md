# Configuration

Mimosa reads its settings from a `Config` object built in [`config.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/config.py).
The interactive wizard writes one to `config_default.json`; for full control,
copy and edit your own:

```bash
cp config_default.json my_config.json
uv run main.py --task "…" --config my_config.json
```

This page walks through the config you'll care about day to day. For every
field with defaults and types, see the [Configuration reference](../reference/configuration.md).

## What you'll set first

| Field | Description |
| ----- | ----------- |
| `workspace_dir` | Path to the Toolomics workspace — Mimosa's shared filesystem. |
| `discovery_addresses` | IP + port ranges scanned for MCP servers. |
| `workflow_llm_model` | LLM that synthesizes multi-agent workflows. |
| `smolagent_model_id` | LLM used by execution agents inside each workflow. |
| `judge_model` | LLM that scores soft claims in the verifier. |
| `planner_llm_model` | LLM that decomposes goals into tasks (`--goal` mode only). |
| `learned_score_threshold` | Score that triggers early stop in `--learn` mode (default `0.9`). |
| `max_learning_evolve_iterations` | Hard cap on evolve iterations (default `20`). |

## Choosing models

LLM choice is the biggest lever you have. Some practical guidance:

=== "Workflow synthesis"

    The model in `workflow_llm_model` writes Python code that wires agents and
    tools together. It benefits from strong reasoning. Good picks:

    - `anthropic/claude-opus-4-5`
    - `openai/gpt-5.5`
    - `z-ai/glm-5`

=== "Execution agents"

    `smolagent_model_id` runs *inside* the generated workflow. Each subtask is
    fairly self-contained, so a cheaper/faster model works well:

    - `openrouter/deepseek/deepseek-v3.2`
    - `anthropic/claude-haiku-4-5`
    - `openai/gpt-5-mini`

=== "Judge"

    The judge needs decent reasoning but is only called once per generation. A
    capable medium-size model is usually enough:

    - `openai/gpt-5.5`
    - `anthropic/claude-opus-4-5`

!!! tip "OpenRouter routing"
    For OpenRouter models, Mimosa picks providers from `openrouter_provider`
    and refines per-model via a quantization-fidelity precheck. fp8/int4
    providers sometimes pass capability checks but corrupt escape sequences
    in generated code — Mimosa filters those out for you. See
    [Troubleshooting](../reference/troubleshooting.md).

## Workspace & paths

| Field | Default | Use |
| ----- | ------- | --- |
| `workspace_dir` | *(set per machine)* | Toolomics shared workspace. |
| `workflow_dir` | `sources/workflows` | Where generated workflows are saved. |
| `memory_dir` | `sources/memory` | Where LLM-call caches and traces live. |
| `runs_capsule_dir` | `runs_capsule/` | Per-run archive snapshots. |
| `runner_temp_dir` | `./tmp` | Sandbox scratch space. |

## Runner limits

The sandbox enforces resource caps per generated workflow:

| Field | Default | Purpose |
| ----- | ------- | ------- |
| `runner_default_python_version` | `3.12` | Python version inside the sandbox. |
| `runner_default_timeout` | `3600` | Per-run timeout (seconds). |
| `runner_default_max_memory_mb` | `1024` | RAM cap (MB). |
| `runner_default_max_cpu_percent` | `100` | CPU cap (%). |

These can be overridden on the command line for one-off runs:

```bash
uv run main.py --task "…" \
  --runner_default_timeout 7200 \
  --runner_default_max_memory_mb 4096
```

## Learning parameters

| Field | Default | Meaning |
| ----- | ------- | ------- |
| `learned_score_threshold` | `0.9` | Stop evolving when `overall_score` reaches this. |
| `max_learning_evolve_iterations` | `20` | Max generations before giving up. |

See [Iterative learning](../usage/learning.md) for the full evolution machinery.

## CLI overrides

A handful of config fields can be overridden directly on the command line —
useful for one-off changes without editing JSON. The full list is in
[CLI reference](../reference/cli.md).

## Saving and loading configs

`Config.dump(path)` writes the current config to JSON, and `Config.load(path)`
reads it back. The `--config <path>` flag loads at startup. Running `python
config.py` from the project root regenerates `config_default.json`.

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

    The judge model scores the soft claims in the verifier. During an
    evaluation run the verifier is one of the places where the most tokens get
    spent, so the model you choose here has a real effect on how much a run
    costs in the end.

    The judge needs decent reasoning, but it does not need the single most
    expensive flagship model you can find. A capable medium-size model is
    usually enough, and a medium-size model costs a lot less per token than a
    flagship one. As a rough guide, a mid-tier model such as a Sonnet-class
    Claude or `deepseek-v4-pro` is in the region of five times cheaper per
    token than a top-tier flagship such as an Opus-class Claude, and it still
    does a good job on the kind of judging the verifier asks for. Good picks
    that you call directly:

    - `anthropic/claude-sonnet-4-6`
    - `openai/gpt-5.5`

    If you run your models through OpenRouter, which is the common setup for
    the metabolomics evaluation, there are several capable-but-cheaper routes
    that also work well as a judge:

    - `openrouter/deepseek/deepseek-v4-pro`
    - `openrouter/z-ai/glm-5`
    - an OpenRouter MiniMax route (for example the latest `minimax` model)

    The exact version tag for a given model changes over time on OpenRouter,
    so it is worth checking what is currently offered and picking the most
    recent capable version of whichever family you prefer. Either way, the
    point is the same: a mid-tier model is plenty for the judge, and it costs
    much less than a flagship.

    So the simplest and biggest cost saving you can make for evaluation is to
    set `judge_model` to one capable-but-cheaper model and let the whole
    verifier run on it, rather than pointing it at the most expensive model
    you have.

    You might be tempted to go one step further and add a second, even cheaper
    model just for the small mechanical steps inside the verifier — for
    example dropping duplicate claims, or checking which Python packages a
    claim needs. In practice this is not worth doing. Those mechanical steps
    are only a small fraction of all the tokens the verifier uses, so moving
    them onto a separate cheap model saves almost nothing while making the
    configuration harder to follow. The calls that actually cost tokens, and
    that actually need good judgement — rating how important a claim is,
    choosing which files to look at, writing the small verifier script, and
    giving the final verdict — are better left on the one `judge_model` you
    already picked. Choosing a sensible `judge_model` is where essentially all
    of the saving comes from.

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

# Configuration reference

Full list of `Config` fields with defaults and types. Source:
[`config.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/config.py).

For typical setup, see [Configuration](../getting-started/configuration.md);
this page is the exhaustive reference.

## Workspace & discovery

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `workspace_dir` | `str` | `/home/.../Toolomics/workspace` | Toolomics shared workspace. |
| `discovery_addresses` | `list[AddressMCP]` | `[AddressMCP("0.0.0.0", 5000, 5100)]` | IP + port ranges scanned for MCP servers. |

`AddressMCP` is a dataclass with `ip: str`, `port_min: int`, `port_max: int`.
Ports must be in `[0, 65535]` and `port_min ≤ port_max`.

## LLM roles

| Field | Type | Default | Used by |
| ----- | ---- | ------- | ------- |
| `planner_llm_model` | `str` | `deepseek/deepseek-chat` | Layer 0 — task decomposition. |
| `workflow_llm_model` | `str` | `openai/gpt-5.5` | Workflow synthesis. |
| `smolagent_model_id` | `str` | `openrouter/deepseek/deepseek-v3.2` | Execution agents inside the sandbox. |
| `judge_model` | `str` | `openai/gpt-5.5` | Verifier soft-claim verdicts. |
| `capsule_namer_model` | `str` | `deepseek/deepseek-chat` | Generates human-readable capsule names. |
| `engine_name` | `str` | `litellm` | SmolAgents engine — keep as `litellm`. |
| `reasoning_effort` | `str` | `medium` | `minimal | low | medium | high` for models that support it. |
| `max_tokens` | `int` | `8192` | Token cap on LLM responses. |

## Prompts

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `prompt_planner` | `str` | `sources/prompts/planner_reproduction.md` | Planner system prompt. |
| `prompt_workflow_creator` | `str` | `sources/prompts/workflow_v11.md` | Workflow-generation prompt. |
| `prompt_smolagent` | `str` | `sources/prompts/smolagent_sys_prompt.md` | SmolAgent system prompt. |

## Learning

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `learned_score_threshold` | `float` | `0.9` | `--learn` stops when `overall_score` reaches this. |
| `max_learning_evolve_iterations` | `int` | `20` | Hard cap on evolve iterations. |
| `max_concurrent_eval_tasks` | `int` | `1` | Concurrent tasks in CSV / batch modes. |

## OpenRouter routing

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `openrouter_provider` | `list[str] \| None` | curated list | Global provider allow-list. |
| `openrouter_provider_by_model` | `dict[str, list[str]]` | `{}` | Per-model overrides (written by precheck). |
| `openrouter_quantizations_by_model` | `dict[str, list[str] \| None]` | `{}` | Per-model quantization filter (precheck). |
| `default_openrouter_quantizations` | `list[str]` | `["bf16", "fp16", "fp8"]` | Default safety filter — blocks int4/fp4. |

See [Troubleshooting → OpenRouter quantization](troubleshooting.md#openrouter-quantization).

## Paths

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `schema_code_path` | `str` | `sources/modules/state_schema.py` | Workflow state template. |
| `smolagent_factory_code_path` | `str` | `sources/modules/smolagent_factory.py` | SmolAgent factory template. |
| `runs_capsule_dir` | `str` | `runs_capsule/` | Capsule archive root. |
| `workflow_dir` | `str` | `sources/workflows` | Per-generation artefacts. |
| `memory_dir` | `str` | `sources/memory` | LLM cache + checklists. |

## Sandbox runner

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `runner_default_python_version` | `str` | `3.12` | Python in the sandbox. |
| `runner_default_timeout` | `int` | `3600` | Per-run timeout (s). |
| `runner_default_max_memory_mb` | `int` | `1024` | RAM cap (MB). |
| `runner_default_max_cpu_percent` | `int` | `100` | CPU cap (%). |
| `runner_temp_dir` | `str` | `./tmp` | Sandbox scratch root. |
| `runner_requirements` | `list[str]` | (pinned list) | Pip requirements installed in every sandbox. |

## Notifications

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `pushover_token` | `str \| None` | `$PUSHOVER_TOKEN` | Pushover API token. |
| `pushover_user` | `str \| None` | `$PUSHOVER_USER` | Pushover user key. |

## Methods worth knowing

```python
config.dump("my_config.json")     # Save current config to JSON.
config.load("my_config.json")     # Load config from JSON.
config.refresh_pricing()          # Force re-fetch of OpenRouter pricing.
config.openrouter_provider_for(model_id)
config.openrouter_quantizations_for(model_id)
config.validate_paths()           # Assert required paths exist.
config.create_paths()             # mkdir any runtime paths.
```

## See also

- [Configuration (getting started)](../getting-started/configuration.md) — what to actually set.
- [API keys & environment](../getting-started/env.md) — env-var side.
- [CLI reference](cli.md) — flags that override config fields.

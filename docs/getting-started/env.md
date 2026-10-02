# API keys & environment

Mimosa talks to LLMs through a unified provider abstraction
([`LLMProvider`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/llm_provider.py))
built on [LiteLLM](https://github.com/BerriAI/litellm). You pick *which*
provider via the model id in your config, and the corresponding API key
must be in the environment.

## The `.env` file

Mimosa loads variables from `.env` at the project root automatically via
`python-dotenv`. At least one key must be present:

```env
# Pick the ones you actually use ----------------------------------------------
ANTHROPIC_API_KEY=...       # Claude — recommended for workflow orchestration
OPENAI_API_KEY=...          # OpenAI (GPT-4o, GPT-5, …)
MISTRAL_API_KEY=...         # Mistral
DEEPSEEK_API_KEY=...        # DeepSeek
HF_TOKEN=...                # HuggingFace Inference Providers
OPENROUTER_API_KEY=...      # Any model via OpenRouter

# Optional --------------------------------------------------------------------
LANGFUSE_PUBLIC_KEY=...     # Trace visualisation
LANGFUSE_PRIVATE_KEY=...
PUSHOVER_TOKEN=...          # Phone notifications
PUSHOVER_USER=...
```

`validate_environment()` in `main.py` checks at least one of the LLM keys
exists and refuses to start otherwise.

## Which provider for which role?

You'll see four model-id slots in your config. They can all use the same
provider, but it's often cheaper and faster to mix.

| Slot | Reasoning depth | Typical good pick |
| ---- | --------------- | ----------------- |
| `workflow_llm_model` | High — designs the whole multi-agent topology. | `anthropic/claude-opus-4-5` |
| `smolagent_model_id` | Low/medium — executes a single subtask. | `openrouter/deepseek/deepseek-v3.2` |
| `judge_model` | Medium — soft-claim verdicts. | `openai/gpt-5.5` |
| `planner_llm_model` | Medium — task decomposition (`--goal` mode). | `deepseek/deepseek-chat` |

!!! tip "Reasoning effort"
    For models that support it (Claude, GPT-5), `Config.reasoning_effort`
    accepts `minimal | low | medium | high`. The default is `medium`. Bump it
    to `high` if Mimosa is producing weak workflows; drop it to `low` to cut
    cost on cheap subtasks.

## OpenRouter routing

OpenRouter brokers many providers. Mimosa supports per-model provider
selection driven by a precheck:

- `openrouter_provider` — global allow-list (defaults to a curated set).
- `openrouter_provider_by_model` — overrides per model id, written by the
  precheck.
- `openrouter_quantizations_by_model` — quantization filter per model id,
  also written by the precheck.

The default safety filter is `["bf16", "fp16", "fp8"]`, which blocks unsafe
`int4`/`fp4` routing. **fp8 and int4 providers can pass basic capability
checks while corrupting `\n` escapes in generated code** — Mimosa filters
those out via fidelity probes. See
[Troubleshooting](../reference/troubleshooting.md#openrouter-quantization).

## HuggingFace Inference Providers

Set `HF_TOKEN` and use model ids of the form
`huggingface/<provider>/<model>` (see LiteLLM docs for the exact syntax for
each provider).

## Local MLX (Apple Silicon)

For local inference on Apple Silicon, `smolagents[mlx-lm]` is already in the
dependency list. Use a model id of the form `mlx/<model>`; no API key needed.

## Optional: phone notifications

Mimosa can ping a Pushover client whenever a run finishes or improves.
Required:

```env
PUSHOVER_TOKEN=...
PUSHOVER_USER=...
```

Full setup at [Notifications](../usage/notifications.md).

## Optional: telemetry

Mimosa emits OpenTelemetry spans to a local Langfuse instance for trace
inspection. Required:

```env
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_PRIVATE_KEY=...
```

Full setup at [Telemetry](../usage/telemetry.md).

# Portable completion backend information flow

Mimosa selects a completion transport only from an explicit model prefix.
`codex-cli/<model>` maps to bridge backend `codex_cli`; `claude-cli/<model>`
maps to `claude_cli`. All other model identifiers retain the existing LiteLLM
path. CLI dispatch happens at the beginning of `LLMProvider.__call__`, before
disk-cache lookup and before the API retry loop, so a bridge failure cannot
fall through to a paid API provider.

For a CLI text role, `LLMProvider` builds protocol-v1 messages from its system
message and the caller's user prompt. It sends the requested backend, bare
model, text response format, configured authentication mode, reasoning effort,
and bounded timeout to `sources/core/completion_backends.py`. That client reads
only the explicit absolute `HARNESS_COMPLETION_BRIDGE` path, loads the trusted
module with `importlib`, and calls `complete(request)` once in-process. The
returned route fields must match the request and completed text must be
non-empty. The complete result envelope is retained in
`last_completion_metadata`. Temperature and max-token settings are omitted,
warned, and recorded as unsupported controls.

CLI completion records carry `cache_eligible: false`, and both cache layouts
skip that marker before comparing messages. Explicit `api_base`/`api_key_env`
routes also bypass cache reads and mark their records ineligible. Mimosa's
legacy cache identity contains only role and message context, so neither
portable opt-in route becomes cacheable until route identity is part of the
cache key.

`Config.api_base` and `Config.api_key_env` contain an endpoint and an environment
variable name, never a credential value. They persist through JSON. The endpoint
is applied only to explicit `openai/` API models, or to `claude-cli/` under
`api_key` authentication. HTTPS is required; URL user information, queries,
and fragments are rejected. Naming a missing key variable fails before
LiteLLM and does not fall back to a provider's ambient key variable.

Planner, workflow authoring, variation, judge/extraction, and capsule naming
construct `LLMConfig` with these route fields. Generated SmolAgent code receives
only the endpoint and environment-variable name, then resolves the value in its
runtime. CLI prefixes are rejected for every SmolAgent candidate before tools,
provider probes, or engine construction because this bridge slice supports text
completion without tools. Precheck records CLI text roles as metadata-only and
does not probe them through LiteLLM.

## Minimal configurations

Set the bridge path in the process environment before starting Mimosa:

```sh
export HARNESS_COMPLETION_BRIDGE=/absolute/path/to/harness_completion.py
```

The following JSON fragment uses native CLI subscriptions for text roles while
keeping the tool-capable SmolAgent on its existing API engine. The SmolAgent
provider's usual API credential must still be present in the environment.

```json
{
  "planner_llm_model": "codex-cli/gpt-6-astra",
  "workflow_llm_model": "claude-cli/claude-opus-5",
  "judge_model": "openrouter/z-ai/glm-5.3-flash",
  "capsule_namer_model": "claude-cli/claude-opus-5",
  "smolagent_model_id": "openrouter/z-ai/glm-5.3",
  "engine_name": "litellm",
  "harness_auth_mode": "subscription"
}
```

The ASB evaluation campaign keeps its judge on an open-weight OpenRouter model.
CLI authoring or solver success does not establish scientific acceptance.

For direct GLM access through z.ai's standard OpenAI-compatible API, bind the
endpoint to a named environment variable. This route does not use the CLI
bridge.

```sh
export ZAI_API_KEY='...'
```

```json
{
  "planner_llm_model": "openai/glm-5.3",
  "api_base": "https://api.z.ai/api/paas/v4/",
  "api_key_env": "ZAI_API_KEY"
}
```

For z.ai's Claude-compatible API, keep the explicit `claude-cli/` transport
and select API-key authentication. The bridge receives the variable name and
resolves its value from the environment; Mimosa does not put the value in the
request or generated source.

```sh
export ANTHROPIC_AUTH_TOKEN='...'
```

```json
{
  "workflow_llm_model": "claude-cli/glm-5.3",
  "harness_auth_mode": "api_key",
  "api_base": "https://api.z.ai/api/anthropic",
  "api_key_env": "ANTHROPIC_AUTH_TOKEN"
}
```

These endpoint examples require separately authorized API access. The direct
GLM routes were covered with offline fakes in this change and were not called
live. A coding-plan subscription is not represented here as generic API
entitlement.

# Troubleshooting

Common failures, what they mean, and how to fix them.

## Installation & startup

### `ModuleNotFoundError: dotenv` (or any other dep)

You ran `python main.py` directly instead of through `uv` or the venv.

```bash
# Fix 1: use uv
uv run main.py …

# Fix 2: activate the venv first
source .venv/bin/activate
python main.py …
```

### `No valid API key environment variable found`

Mimosa's `validate_environment()` requires at least one of
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `MISTRAL_API_KEY`,
`DEEPSEEK_API_KEY`, `HF_TOKEN`, or `OPENROUTER_API_KEY`. Put one in `.env`
at the repo root and try again. See
[API keys & environment](../getting-started/env.md).

### `Workspace directory not found`

The `workspace_dir` in your config doesn't exist. Either start Toolomics
(which creates it) or update the path:

```bash
uv run main.py --task "…" --config my_config.json
# edit my_config.json: "workspace_dir": "/path/that/exists"
```

## MCP discovery

### Mimosa finds zero tools

`ToolManager` scanned `discovery_addresses` and got no MCP handshakes back.
Check:

- Is **Toolomics running**? `curl http://localhost:5000/` should return
  something (or another port in your configured range).
- Are the IPs/ports in `discovery_addresses` correct? The default is
  `AddressMCP("0.0.0.0", 5000, 5100)`.
- Is a firewall blocking the port range?

### A tool I expect to be discoverable doesn't show up

Mimosa does **not** use tool tags. Every tool exposed by every reachable
MCP server is made available to every agent in the generated workflow.
If a tool is missing, it's one of:

- The MCP server hosting it isn't in `Config.discovery_addresses`. Check
  the IP / port range and that the server answers a handshake on
  `http://<ip>:<port>/mcp`.
- The tool is registered with ToolHive but the ToolHive code path is
  commented out in `tools_manager.py` (the default in this build). Use a
  network-MCP exposure of the same tool, or re-enable ToolHive
  discovery.
- The Toolomics service is in `config_<instance_id>.json` with
  `"enabled": false`. Flip it to `true` and re-run Toolomics' `./start.sh`.

Use `--manual` mode to list everything that *is* discoverable and
confirm the tool name actually appears.

## OpenRouter quantization

### Workflows fail with weird `\n` corruption in the code

**Root cause**: some OpenRouter providers (notably aggressive fp8 / int4
quants) pass capability checks but corrupt escape sequences in generated
Python — strings come back with `\\n` where you wanted `\n`, causing
syntax errors or silently wrong logic.

**Fix**: the precheck filters these out via fidelity probes (it doesn't
just test the model exists, it tests its escape handling).
`Config.default_openrouter_quantizations = ["bf16", "fp16", "fp8"]` blocks
int4/fp4 by default. If you're seeing escape corruption, run the precheck
explicitly:

```python
from sources.utils.precheck import PreCheck
from config import Config
PreCheck(Config()).run()
```

You can also tighten the filter to `["bf16", "fp16"]` to exclude fp8
entirely.

### Precheck refuses to find any provider

You may have whitelisted only a small set in `Config.openrouter_provider`
but none of them serves the model you asked for. Either:

- Widen the whitelist (the default includes 25+ providers), or
- Switch to a model with broader provider support.

## Runtime / execution

### Sandbox hits `runner_default_timeout`

A workflow ran for longer than the timeout (default `3600` s). Either:

- Bump the timeout: `--runner_default_timeout 7200`.
- Look at the workflow code in `sources/workflows/<uuid>/workflow_genotype_*.py`
  — sometimes a runaway loop or stuck tool call is the real issue.

### `OSError: [Errno 24] Too many open files`

Heavy concurrency in batch mode can exhaust file descriptors. Lower
`max_concurrent_eval_tasks`, or raise the OS limit:

```bash
ulimit -n 8192
```

### LangFuse / OTLP errors at startup

If you see `Failed to connect to OTLP collector` and you haven't set up
Langfuse:

- Just unset `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_PRIVATE_KEY` in your `.env`.
  Telemetry no-ops when they're missing.

If you *did* set them, check Langfuse is actually running
(`docker compose ps` in the Langfuse repo).

## Learning / evaluation

### `--learn` runs but score never increases

A few likely causes:

- **Verifier hard fails.** If every generation has a refuted hard claim,
  `hard_fail_capped` flips `true` and `overall_score` is capped at
  `_HARD_FAIL_CAP` (currently `0.7`). QD selection ranks on this capped
  score (`overall_score_uncapped` is still logged for analysis), so
  capped runs stay below uncapped improvements. You may have a
  structural mismatch between the task description and what the
  workflow can actually verify against.
- **Recurring failure code.** The `abstracted_prompt_gradient` is
  prefixed with a short code name (e.g. `FALLBACK_ECFP_CLASSIFIER`).
  If the same code recurs across generations, the loop is re-discovering
  the same failure mode — inspect `evaluation.txt` for the per-claim
  detail behind it.
- **Tool gap.** The agents may need a tool that Toolomics doesn't expose.
  Use `--manual` mode to confirm the tool you need is actually
  discoverable.

### Benchmark scores look too good

Run `./cleanup.sh` and re-run. Cached workflows from previous runs leak
through the disk-similarity fallback in `WorkflowSelector`, biasing the
benchmark optimistically.

## Other

### "I changed a prompt, but nothing's different"

LLM responses are cached. Wipe `sources/memory/` to invalidate the cache,
or change just enough of the prompt that the hash differs.

### "I want to debug what the orchestrator sent the model"

The exact prompt is at
`sources/workflows/<uuid>/evolution_prompt_<uuid>.md`. Open it.

## Still stuck?

Open an issue at
[github.com/HolobiomicsLab/Mimosa-AI/issues](https://github.com/HolobiomicsLab/Mimosa-AI/issues).
Include:

- The exact command.
- The full traceback (or last ~30 lines of the log).
- The contents of your config (redact API keys).
- The Mimosa commit hash (`git rev-parse HEAD`).

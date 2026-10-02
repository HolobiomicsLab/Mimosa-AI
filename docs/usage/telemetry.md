# Telemetry

Mimosa-AI emits OpenTelemetry traces for every LLM call, tool call, and
agent step. The recommended sink is a local [Langfuse](https://langfuse.com/)
instance — span-level inspection without sending data off your machine.

## Quick start

1. **Run Langfuse locally** with Docker Compose:

   ```bash
   git clone https://github.com/langfuse/langfuse.git
   cd langfuse
   docker compose up -d
   ```

2. **Get the keys** from the Langfuse UI (default `http://localhost:3000`)
   — Settings → API keys → "Create new keys".

3. **Add them to `.env`**:

   ```env
   LANGFUSE_PUBLIC_KEY=...
   LANGFUSE_PRIVATE_KEY=...
   ```

4. **Run Mimosa as usual.** Traces will appear in the Langfuse dashboard
   in real time.

## What you'll see

The dashboard renders each run as a hierarchical trace:

- **Workflow generation** — the LLM call that emitted the workflow source.
- **Per-iteration spans** — one per generation in `--learn` mode.
- **Agent steps** — every SmolAgent step inside the sandbox.
- **Tool calls** — each MCP call, with args, output, and latency.
- **Judge calls** — verifier soft-claim verdicts and per-claim executable
  scripts.

Span attributes include token counts, latency, model id, and cost. You can
filter by model, by tag, or by user.

## Without telemetry

If `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_PRIVATE_KEY` are absent, the
OpenTelemetry exporter no-ops. Mimosa runs identically — you just lose the
trace UI.

## Custom sinks

Mimosa uses the standard OpenTelemetry SDK plus the OTLP exporter, so any
OTLP-compatible backend works:

- **Jaeger** — set `OTEL_EXPORTER_OTLP_ENDPOINT` to your Jaeger collector.
- **Honeycomb / Datadog** — same env var, plus their auth headers.
- **A local file** — point the OTLP endpoint at an OTLP file collector.

See the [OpenTelemetry Python docs](https://opentelemetry.io/docs/languages/python/)
for the exact env-var configuration.

## Cost tracking

LLM costs come from the OpenRouter pricing client
([`sources/utils/pricing.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/utils/pricing.py)),
which fetches live pricing and caches it under
`sources/cache/openrouter_pricing.json`. Per-run cost is emitted as a span
attribute and aggregated in capsule `evaluation_results.json`.

## See also

- [API keys & environment](../getting-started/env.md) — env-var setup.
- [Notifications](notifications.md) — push-style real-time updates instead.
- [Transparency & replay](transparency.md) — disk-side alternatives.

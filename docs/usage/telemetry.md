# Telemetry

Mimosa-AI emits OpenTelemetry traces for agent steps and tool calls when
Langfuse keys are set. The recommended sink is a local [Langfuse](https://langfuse.com/)
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
   LANGFUSE_SECRET_KEY=...
   ```

4. **Run Mimosa as usual.** Traces will appear in the Langfuse dashboard
   in real time.

## What you'll see

The dashboard shows agent steps and tool calls for a run. The exact span set
depends on the instrumentor version.

## Without telemetry

If `LANGFUSE_PUBLIC_KEY` or `LANGFUSE_SECRET_KEY` is missing, Mimosa does not
install the instrumentor and exports nothing. Mimosa runs identically; you
just lose the trace UI.

## Custom sinks

Mimosa sets the OTLP endpoint itself. When both keys are set,
`sources/modules/smolagent_factory.py` sets `OTEL_EXPORTER_OTLP_ENDPOINT` to
`http://localhost:3000/api/public/otel`, overwriting any value already in the
environment. Other OTLP backends (Jaeger, Honeycomb, Datadog) need a code change.

## Cost tracking

LLM costs come from the OpenRouter pricing client
([`sources/utils/pricing.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/utils/pricing.py)),
which fetches live pricing and caches it under
`sources/cache/openrouter_pricing.json`.

## See also

- [API keys & environment](../getting-started/env.md) — env-var setup.
- [Notifications](notifications.md) — push-style real-time updates instead.
- [Transparency & replay](transparency.md) — disk-side alternatives.

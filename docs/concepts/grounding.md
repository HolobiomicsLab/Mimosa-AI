# Scientific grounding

Mimosa-AI is built for **research**, not chat. Workflow synthesis
benefits from grounding in real literature, not the model's prior. That
grounding comes from
[Perspicacité](https://github.com/HolobiomicsLab/Perspicacite-AI), an
optional companion AI.

> **Verifier note.** The default hybrid temporal-ladder verifier does
> **not** consult Perspicacité — its policy scorers are deterministic
> Python with no soft-claim path. The soft-claim grounding below
> belongs to the **deprecated legacy verifier**
> (`verifier_kind = "legacy"`, pre-2026-09-24) and to
> `GenericEvaluator`.

## What Perspicacité provides

Perspicacité serves two purposes:

1. **Workflow synthesis grounding** — when the factory drafts a workflow,
   it can ask Perspicacité for relevant literature on the task domain. The
   returned context is prepended to the workflow-creation prompt, biasing
   the synthesized agents toward methods that actually appear in the
   literature.
2. **Soft-claim grounding (legacy verifier only)** — when the deprecated
   [multi-source verifier](evaluation-pipeline.md#legacy-multi-source-verifier-deprecated)
   judges a soft claim (something it can't recompute), it asks
   Perspicacité for grounding evidence. The judge LLM then weighs the
   claim against that evidence rather than against the model's prior
   alone.

## Wiring

The [`PerspicaciteClient`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/utils/perspicacite_client.py)
is a thin HTTP client. Perspicacité runs on its own port and exposes a
small REST surface; Mimosa hits it whenever grounding is requested.

The [`grounding.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/evaluators/grounding.py)
adapter in the evaluator package wraps the client and exposes a uniform
"ask for evidence on this claim/topic" interface to the legacy verifier
and `GenericEvaluator` (the hybrid verifier does not call it).

## What happens if it's not running?

Both call sites fail gracefully:

- The workflow factory falls back to ungrounded synthesis (model prior only).
- The legacy verifier's soft-claim path returns `unsure` more often (the
  hybrid verifier is unaffected — it never calls Perspicacité).

You'll see warnings in the log but Mimosa keeps running. The whole point of
Perspicacité is to *improve* synthesis and scoring — it isn't a hard
dependency.

## When grounding matters most

- **Niche / under-resourced domains.** General-purpose LLMs hallucinate
  more confidently the further you go from their training distribution.
  Perspicacité narrows the prior to literature in your actual field.
- **Reproductions of published work.** When Mimosa is told "reproduce the
  experiments from paper X", giving the synthesizer access to the paper's
  methodology — not just its title — meaningfully improves first-attempt
  workflow quality.
- **Soft-claim adjudication (legacy verifier).** "Is this loss function
  appropriate for this data?" was a claim the legacy verifier's soft
  path had to adjudicate; grounding the judge in published practice was
  the next-best thing. The hybrid verifier avoids the problem by
  scoring only deterministic, measurable claims.

## Setup

See [Installation §4](../getting-started/installation.md#4-optional-start-perspicacite-for-scientific-grounding).

## See also

- [Tool discovery & MCP](tools-and-mcp.md) — the other half of "where does
  external knowledge come from".
- [Evaluation pipeline](evaluation-pipeline.md) — the hybrid verifier;
  the legacy soft-claim path is documented in its deprecated section.

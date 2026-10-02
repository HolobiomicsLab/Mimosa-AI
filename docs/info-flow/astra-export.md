# ASTRA post-run export — info flow

> **Reader's note.** This page is an info-flow audit, not a user-facing
> tutorial. It traces every variable that reaches the decision-extraction
> LLM and every variable that ends up in the emitted ASTRA YAML.
> For the spec itself, see <https://astra-spec.org/latest/specification/>.

## Why a post-run export

ASTRA captures *why* a scientific analysis is the way it is — the
methodological choices a reviewer would scrutinize. Mimosa's evolution loop
produces a single best workflow and runs an agent over it; the reasoning
behind each step lives in the smolagents memory trace, not in any declared
schema. The exporter walks the best run's memory after the evolution loop
finishes and surfaces those decisions into a standards-compliant YAML pair
written next to the analysis artefacts.

The export is **post-run only** — wiring decision tracking into the
evolution loop itself would touch every agent, prompt, and runner. After
the fact, with the trace already on disk, the operation is local and
non-fatal: a failed export must never break the evolution loop.

## Entry point

The exporter is invoked from
[`sources/core/evolution_engine.py`](../../sources/core/evolution_engine.py)
in `start_workflow_evolution`, immediately after
`workspace_mgr.restore_best(best_run.current_uuid)`. The call is gated on
`config.export_astra` (opt-in, off by default).

Output target: `<config.runs_capsule_dir>/<best_uuid>/astra.yaml` plus
`<config.runs_capsule_dir>/<best_uuid>/universes/best.yaml`. The best
run's UUID is used as the capsule subfolder name — stable, deterministic,
matches the `sources/memory/<uuid>` convention — and intentionally
independent of the LLM-named goal capsule that `LocalTransfer` produces
later.

Failure of the export is logged via `print_warn` and the engine continues.

## Variables that reach the LLM

The extraction prompt template lives at
[`sources/transparency/prompts/decision_extraction.md`](../../sources/transparency/prompts/decision_extraction.md).
For each surviving (post-prefilter) step, the prompt receives these
variables — and nothing else:

| Variable      | Source                                                    | Why it's needed                                                |
| ---           | ---                                                       | ---                                                            |
| `goal`        | `start_workflow_evolution(goal)` argument                 | Lets the LLM tell methodology from goal-irrelevant scaffolding |
| `step_index`  | Position of the step in the compacted trace               | Provenance — surfaces in `Decision.source_step`                 |
| `reasoning`   | `memory_trace.extract_output_text` — `model_output_message.content`, else `model_output` | The agent's natural-language justification, where the decision is voiced |
| `code`        | `memory_trace.extract_code` — `code_action`, else `tool_calls[*].function.arguments`, else a fenced block in the output | The concrete chosen option as executed |
| `observation` | `step.observations` truncated to 1200 chars               | Disambiguates whether the step succeeded or recovered           |

> **Shared extraction.** Code/reasoning extraction lives in
> [`sources/transparency/memory_trace.py`](../../sources/transparency/memory_trace.py),
> reused by both the exporter and the interactive memory chat
> (`sources/cli/memory_chat_cli.py`) so the two never disagree on "the code
> the agent ran". Note `code` reads `code_action` — **not** `action_output`,
> which is the step's *result*, not its source.

Variables that are deliberately **not** passed:

- `model_input_messages` — re-serialises the entire prior conversation on
  every step. Stripped in [`trace_compaction.compact_step`](../../sources/transparency/trace_compaction.py).
  This is the dominant token-cost saving, usually ≥80% of the raw file.
- `model` — the per-step model id stamped by `save_memories`
  (e.g. `openrouter/qwen/qwen3.7-plus`). It rides along on the compact step
  as provenance for the YAML/recipe outputs but never enters the prompt, so
  extraction cache keys are unaffected.
- Workspace file contents — the *names* of output files reach the YAML
  builder, but never the LLM. Decisions live in the reasoning, not the
  artefacts.
- Other runs from the evolution — only the best run's memory is read.

## Heuristic prefilter

`is_methodological_candidate` in `trace_compaction.py` drops steps whose
code is entirely matched by mechanical patterns: `import`, `pd.read_*`,
`open()`, `plt.*`, `subprocess.*`, `pip install`, `print(...)`. A step
survives if at least one non-mechanical line of code or non-empty
reasoning is present. This typically halves what reaches the LLM.

## Output construction

After extraction, `yaml_writer.build_analysis` populates the ASTRA
analysis dict:

| ASTRA field             | Source                                                |
| ---                     | ---                                                   |
| `version`               | Constant `0.1`                                        |
| `name`                  | `"Mimosa best run <best_uuid>"`                       |
| `description`           | Constant, indicates the post-run extraction provenance |
| `inputs[0]`             | `task_description` derived from the user `goal`       |
| `outputs[i]`            | Each top-level file in the artefacts directory — prefers `/tmp/mimosa_run_<session>_<uuid>` (the canonical snapshot), falls back to `config.workspace_dir`. See `AstraExporter._resolve_artefacts_dir`. OS junk (`.DS_Store`, `Thumbs.db`, `.gitkeep`), dotfiles, and our own `recipe.py` are dropped. |
| `outputs[i].decisions`  | All decision IDs surfaced from the trace               |
| `outputs[i].recipe.command` | `python recipe.py`. The run's executed code is reconstructed by `memory_trace.reconstruct_recipe` (ordered `code_action` of every step) and written to `runs_capsule/<uuid>/recipe.py`. ASTRA's `recipe.command` is a single shell command, so the multi-step transcript lives in the script and the command points at it. Falls back to a memory-trace pointer string when no code was recovered. |
| `decisions[d.id]`       | One entry per deduped surfaced decision               |
| `decisions[d.id].rationale` | LLM-extracted from the step's reasoning            |
| `decisions[d.id].default` | The realised option (`Decision.chosen_option_id`)   |
| `decisions[d.id].options` | Multi-option map: the chosen option plus any alternatives the agent explicitly weighed in the trace (never invented) |
| `decisions[d.id].model` | Per-step model id from the memory trace, when recorded — provenance extension beyond the spec; omitted for legacy traces |
| `extraction`            | Extraction-health block (see below) — provenance extension beyond the spec |

The companion universe file
(`universes/best.yaml`) is a flat `{decision_id: option_id}` map matching
the realised configuration.

## Extraction health

The extractor refuses to guess. A response whose `chosen_option_id` is
missing or unresolvable is **dropped and counted as malformed** unless the
response listed exactly one cleanly-parsed option (then that option is the
choice by construction; this also covers the legacy single-option shape).
Prose, truncated JSON, empty/None content, and schema violations are all
counted as `malformed`; LLM calls that raise are counted as `crashed`. Both
counters are logged as warnings, echoed by the exporter CLI, and written
into `astra.yaml`:

```yaml
extraction:
  steps_considered: <steps sent to the LLM>
  decisions_recorded: <deduped decisions kept>
  llm_call_failures: <crashed calls>
  malformed_responses: <unusable responses>
```

A capsule produced from a degraded extraction is therefore self-describing:
zero decisions with non-zero failure counters reads as "extraction broke",
never as "the run made no decisions".

## Deduplication

Decisions are keyed by `Decision.id`. If two steps surface the same
decision (e.g. the agent re-justifies a choice late in the trace), the
first occurrence wins for the core fields (label, rationale, chosen option,
provenance) while options surfaced by later re-justifications are unioned
in by `_merge_options`, so alternatives raised later aren't lost.

## Caching

Each per-step LLM call uses `LLMProvider` with `agent_name =
f"astra_decision_step_{i}"` and `memory_path = sources/memory/<best_uuid>`.
Cache hits are deterministic on the prompt content, so re-running the
export on the same best run incurs zero new LLM cost.

## What is NOT captured (yet)

- **Cross-decision dependencies** (`requires`, `incompatible_with`) —
  the trace doesn't encode them; would need a second LLM pass.
- **`tags`, `when`, `from`** — ASTRA optional fields; not used in v1.
- **`insights`** — ASTRA's prior-insight links; out of scope.

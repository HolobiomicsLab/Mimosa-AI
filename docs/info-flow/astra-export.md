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
| `reasoning`   | `step.model_output_message.content` (str or text chunks)  | The agent's natural-language justification, where the decision is voiced |
| `code`        | `step.action_output` (fallback: code block in reasoning)  | The concrete chosen option as executed                          |
| `observation` | `step.observations` truncated to 1200 chars               | Disambiguates whether the step succeeded or recovered           |

Variables that are deliberately **not** passed:

- `model_input_messages` — re-serialises the entire prior conversation on
  every step. Stripped in [`trace_compaction.compact_step`](../../sources/transparency/trace_compaction.py).
  This is the dominant token-cost saving, usually ≥80% of the raw file.
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
| `outputs[i]`            | Each top-level file in the restored workspace (excluding `astra.yaml`/`universes`) |
| `outputs[i].decisions`  | All decision IDs surfaced from the trace               |
| `outputs[i].recipe.command` | Pointer to the memory trace (Mimosa does not run as a single CLI command) |
| `decisions[d.id]`       | One entry per deduped surfaced decision               |
| `decisions[d.id].rationale` | LLM-extracted from the step's reasoning            |
| `decisions[d.id].options` | Single-option map keyed by `d.option_id` (Mimosa picks one option, doesn't enumerate alternatives) |

The companion universe file
(`universes/best.yaml`) is a flat `{decision_id: option_id}` map matching
the realised configuration.

## Deduplication

Decisions are keyed by `Decision.id`. If two steps surface the same
decision (e.g. the agent re-justifies a choice late in the trace), the
first occurrence wins and `source_step` records which step it came from.
This is a deliberate v1 simplification — multi-option enumeration is
deferred until reviewers actually ask for it.

## Caching

Each per-step LLM call uses `LLMProvider` with `agent_name =
f"astra_decision_step_{i}"` and `memory_path = sources/memory/<best_uuid>`.
Cache hits are deterministic on the prompt content, so re-running the
export on the same best run incurs zero new LLM cost.

## What is NOT captured (yet)

- **Cross-decision dependencies** (`requires`, `incompatible_with`) —
  the trace doesn't encode them; would need a second LLM pass.
- **Alternatives considered** (multi-option `options`) — the agent
  usually picks one; back-filling alternatives is honest only if grounded
  in the trace.
- **`tags`, `when`, `from`** — ASTRA optional fields; not used in v1.
- **`insights`** — ASTRA's prior-insight links; out of scope.

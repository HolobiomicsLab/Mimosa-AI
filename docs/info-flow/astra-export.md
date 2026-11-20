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

A failure in the outer call is printed as `print_err("Error in Astra Export: ...")` (`evolution_engine.py:991`) and the engine continues. Failures inside `_export_astra` are logged with `logger.warning` and `print_warn`.

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
| `observation` | Step observation, truncated to 1200 chars by `trace_compaction.compact_step` | Disambiguates whether the step succeeded or recovered |

> **Shared extraction.** Code/reasoning extraction lives in
> [`sources/transparency/memory_trace.py`](../../sources/transparency/memory_trace.py),
> reused by both the exporter and the interactive memory chat
> (`sources/cli/memory_chat_cli.py`) so the two never disagree on "the code
> the agent ran". `code` tries `code_action` first, then tool-call arguments, then a fenced block in the output. It never reads `action_output`, which is the step's *result*, not its source.

Variables that are deliberately **not** passed:

- `model_input_messages` — re-serialises the entire prior conversation on
  every step. Stripped in [`trace_compaction.compact_step`](../../sources/transparency/trace_compaction.py).
  This is the main token-cost saving. The module docstring (`trace_compaction.py:3-6`) estimates the compacted trace at under 20% of the raw size; the export does not measure it.
- `model` — the per-step model id stamped on each action step by the agent factories
  (e.g. `openrouter/qwen/qwen3.7-plus`). It rides along on the compact step
  as provenance for the YAML/recipe outputs but never enters the prompt, so
  extraction cache keys are unaffected.
- Workspace file contents — the *names* of output files reach the YAML
  builder, but never the LLM. Decisions live in the reasoning, not the
  artefacts.
- Other runs from the evolution — only the best run's memory is read.

## Heuristic prefilter

`is_methodological_candidate` in `trace_compaction.py` decides whether a step is sent to the LLM. A step with no code survives if its reasoning is non-empty. A step with code survives if at least one of its lines is not mechanical. Reasoning does not rescue a step whose code is entirely mechanical. The mechanical patterns (`trace_compaction.py:36-48`) are: `import` and `from ... import` lines; `os.listdir`, `os.path.*`, `os.makedirs`, `os.getcwd`; `pd.read_*`; `open(`; `plt.show/savefig/figure/subplot/plot/bar/scatter/hist/imshow`; `print(`; `subprocess.run/Popen/check_output`; `!pip install` lines. The reduction in steps sent is not measured here.

## Output construction

After extraction, `yaml_writer.build_analysis` populates the ASTRA
analysis dict:

| ASTRA field             | Source                                                |
| ---                     | ---                                                   |
| `version`               | Constant `0.1`                                        |
| `name`                  | `"Mimosa best run <best_uuid>"`                       |
| `description`           | Constant, indicates the post-run extraction provenance |
| `inputs[0]`             | `task_description` derived from the user `goal`       |
| `outputs[i]` | Each top-level file in the artefacts directory. The directory is the first match of `/tmp/mimosa_run_*_<uuid>` (`AstraExporter._SNAPSHOT_ROOT`), falling back to `config.workspace_dir`. See `AstraExporter._resolve_artefacts_dir`. OS junk (`.DS_Store`, `Thumbs.db`, `.gitkeep`), dotfiles, `astra.yaml`, `universes/`, and our own `recipe.py` are dropped. |
| `outputs[i].decisions`  | All decision IDs surfaced from the trace               |
| `outputs[i].recipe.command` | `python recipe.py`. The run's executed code is reconstructed by `memory_trace.reconstruct_recipe` (ordered output of `extract_code` for every step) and written to `runs_capsule/<uuid>/recipe.py`. ASTRA's `recipe.command` is a single shell command, so the multi-step transcript lives in the script and the command points at it. Falls back to a memory-trace pointer string when no code was recovered. |
| `decisions[d.id]`       | One entry per deduped surfaced decision               |
| `decisions[d.id].rationale` | LLM-extracted from the step's reasoning            |
| `decisions[d.id].default` | The realised option (`Decision.chosen_option_id`)   |
| `decisions[d.id].options` | Multi-option map: the chosen option plus any alternatives the agent explicitly weighed in the trace (never invented) |
| `decisions[d.id].model` | Per-step model id from the memory trace, when recorded — provenance extension beyond the spec; omitted for legacy traces |
| `extraction`            | Extraction-health block (see below) — provenance extension beyond the spec |

The companion file `universes/best.yaml` holds `id: best`, a description, and a `decisions` map of `{decision_id: chosen_option_id}` matching the realised configuration.

## Extraction health

The extractor refuses to guess. A response whose `chosen_option_id` is
missing or unresolvable is **dropped and counted as malformed** unless the
response listed exactly one cleanly-parsed option (then that option is the
choice by construction; this also covers the legacy single-option shape).
Prose, truncated JSON, empty/None content, and schema violations are all
counted as `malformed`; LLM calls that raise are counted as `crashed`. Both
counters are logged as warnings and written
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
f"astra_decision_step_{i}"`, `memory_path = <config.memory_dir>/<best_uuid>` (default `sources/memory/<best_uuid>`), and `use_cache=True` (`decision_extractor.py:216`).
A re-run of the export on the same best run should reuse cached responses instead of calling the LLM again. [INFERENCE: the cache-key derivation in `llm_provider.py` was not traced in this audit.]

## What is NOT captured (yet)

- **Cross-decision dependencies** (`requires`, `incompatible_with`) —
  the trace doesn't encode them; would need a second LLM pass.
- **`tags`, `when`, `from`** — ASTRA optional fields; not used in v1.
- **`insights`** — ASTRA's prior-insight links; out of scope.

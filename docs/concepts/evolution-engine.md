# Evolution engine

Mimosa-AI is a **self-evolving multi-agent system**. Rather than forcing
every task through a fixed pipeline, the engine composes a custom workflow
per task and refines it over generations. This page walks through what
actually happens inside the loop.

## Big picture

The [`EvolutionEngine`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/evolution_engine.py)
runs a **depth-first recursion** seeded from a Quality-Diversity (QD) archive:

```mermaid
flowchart TB
    Start([Task spec]) --> Reset[Reset session archive]
    Reset --> Seed{First iteration?}
    Seed -- yes --> SeedPrompt[Seed genome prompt<br/>or template mutation]
    Seed -- no --> Pick[Pick parents via QD-roulette<br/>fallback to disk similarity scan]
    SeedPrompt --> Orch[Orchestrate workflow<br/>LLM → sandbox]
    Pick --> Decide{Crossover ≈ 0.3?}
    Decide -- mutation --> Mut[Mutation prompt<br/>stagnation-scoped]
    Decide -- crossover --> Cross[Crossover prompt<br/>best-parent-first]
    Mut --> Orch
    Cross --> Orch
    Orch --> Eval[multi-source per-claim verifier<br/>reward + prompt gradient]
    Eval --> Admit{Validity: improvement<br/>or qd_score ≥ threshold?}
    Admit -- yes --> Archive[Admit to archive]
    Admit -- no --> Reject[Reject<br/>telemetry only]
    Archive --> Lin[Lineage]
    Reject --> Lin
    Lin --> Stop{Score ≥ threshold<br/>or depth reached?}
    Stop -- no --> Pick
    Stop -- yes --> Restore[Restore best workspace]
```

A more detailed view lives in the source diagram
[`docs/diagrams/evolution_loop.mermaid`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/docs/diagrams/evolution_loop.mermaid).

## Each recursive step

1. **Reset** the workspace to the initial state.
2. **Orchestrate** a workflow run (LLM writes Python → sandbox runs it).
3. **Snapshot** the workspace.
4. **Evaluate** — get `overall_score` and `reward_uncapped` (i.e.
   `overall_score_uncapped` in the persisted JSON) plus an
   `abstracted_prompt_gradient`.
5. **`validate_survivor()`** — admit to the archive when the candidate
   improves over baseline or clears `qd_score > admit_threshold`;
   capacity is curated by lowest-`qd_score` eviction.
6. **Select** next parent(s) and choose mutation vs. crossover.
7. **Recurse** until the score threshold is hit or `max_depth` is reached.

Termination:

- `overall_score > learned_score_threshold` (default `0.95`) in `--learn` mode, *or*
- `max_depth` reached (default `25`).

## Selection: Quality-Diversity (QD)

[`SelectionPressure`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/selection.py)
implements four strategies — `greedy`, `tournament`, `novelty`, and `qd`
(default). In QD mode:

- A **session archive** holds up to `population_size = 50` members.
- Each member has `qd_score = (1−w)·quality_norm + w·novelty_norm`, with
  `w = novelty_weight = 0.4`.
- Quality is sourced from `reward_uncapped` so the hard-fail cap doesn't
  flatten rank ordering.
- Novelty is k-NN distance (`k = 25`) in behaviour-descriptor space, where
  the descriptor is `[n_agents, n_edges, n_branches, prompt_chars]`
  extracted by [`code_features.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/code_features.py).
- Admission gate: candidate is admitted when it either improves over
  baseline by `min_improvement_threshold` or clears
  `qd_score > admit_threshold`. When the archive reaches capacity, the
  lowest-`qd_score` member is evicted.
- Parent draw applies an inverse-child-count penalty
  `÷ (1 + n_children_already)` to spread offspring across the archive.

When the archive is empty (cold start), [`WorkflowSelector`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/workflow_selection.py)
falls back to a **similarity-filtered disk scan** (`cosine ≥ 0.5` on MiniLM
embeddings of `original_task`, `score ≥ 0.05`) — this lets useful workflows
transfer across tasks.

## Variation: stagnation-driven mutation scope

[`VariationEngine`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/variation_engine.py)
assembles mutation and crossover prompts. There is no fixed phase
schedule by iteration progress; mutation boldness is a continuous
function of how much the population is repeating itself.

**Stagnation signal.** `_compute_stagnation()` takes the last 4
non-failure prompt gradients, computes their pairwise MiniLM cosine
similarity, and rescales the mean (`0.4` ≈ unrelated → `0`,
`0.8+` ≈ fully stagnated → `1`).

**Effective boldness.** `stagnation_effective = raw_stagnation · (1 −
parent_score)`. Near-winners stay protected from disruption even when
the population stagnates.

**Agent budget.** The current agent count grows toward
`max_possible_agents = 7` proportionally to `stagnation_effective`,
then a Beta-Binomial draw samples the actual count inside that window
(biased upward by stagnation). The seed generation samples agents from
`[1, 4]` with stagnation prior `0.5`.

**Scope band.** A single advisory line is added to the mutation prompt,
chosen by `stagnation_effective`:

| Stagnation effective | Mutation scope                                                         |
| -------------------- | ---------------------------------------------------------------------- |
| < 0.20               | prompt-only little tweak                                               |
| < 0.40               | prompt, handoff, tools — improve information flow                      |
| < 0.60               | significant redesign while keeping topology                            |
| < 0.80               | bold rewire — restructure or grow the agent set                        |
| ≥ 0.80               | complete rethink — discard inherited topology / prompts                |

The bands are advisory text steered to the LLM, not hard gates: the
LLM can still pick any topology. The hard control is the agent-count
budget passed in the same prompt block.

## Crossover

With probability ~0.3 per generation, two parents are combined instead of
one being mutated. The crossover prompt is **best-parent-first**: the
strongest parent's code structure leads, and weaker parents contribute
specific improvements rather than competing for the skeleton.

## Lineage & reproducibility

After each generation, [`lineage.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/lineage.py)
writes a sidecar record `sources/workflows/<uuid>/lineage_<uuid>.json`
with the parents and the operator used (`seed | mutation | crossover`).
Combined with `evolution_prompt_<uuid>.md` (the exact LLM prompt used),
runs are fully reproducible — same prompt, same code path, same seed
yields the same code.

## Visualisations

When the loop finishes, Mimosa emits two artifacts in
`sources/workflows/<uuid>/`:

- `reward_progress.png` — reward over iterations.
- `evolution_tree.png` — rendered lineage tree.

![Reward progress example](../images/evolve_example.png){ width="80%" }

## See also

- [Evaluation pipeline](evaluation-pipeline.md) — what the verifier actually returns.
- [Iterative learning](../usage/learning.md) — running the engine end-to-end.
- [Developer guide](../DEVELOPER_GUIDE.md) — code paths and class wiring.

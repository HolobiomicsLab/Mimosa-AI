# Mimosa V2 — Evolution Algorithm

A neuroevolution-lens reading of the current evolution loop. Maps the
existing code in [sources/core/](../sources/core/) onto the four canonical
components of an evolutionary algorithm — **representation**, **selection**,
**variation**, **evaluation** — flags where the implementation departs from
neuroevolution best practice, and points to the open V2 design questions.

---

## 1. The big picture

Mimosa evolves multi-agent **workflows**: Python programs that wire together
LLM agents, tools, and control flow into a graph that attempts to solve a
scientific task. A single evolutionary run looks like this:

```
                      ┌───────────────────────────────┐
   task spec  ─────►  │  install task-locked          │
                      │  verification checklist       │  (verifier.py L280)
                      └──────────────┬────────────────┘
                                     ▼
                      ┌───────────────────────────────┐
                      │  parent selection             │  (workflow_selection.py)
                      │  archive ?  → archive draw    │
                      │  cold start → disk sim. scan  │
                      └──────────────┬────────────────┘
                                     ▼
                ┌──────────────────────────────────────┐
                │  variation                           │  (variation_engine.py)
                │  mutation prompt  or  crossover      │
                │  + 5-axis perturbation       │
                │  + annealing/curriculum phase        │
                └──────────────┬───────────────────────┘
                               ▼
                ┌──────────────────────────────────────┐
                │  orchestrator → meta-agent writes    │  (orchestrator.py,
                │  workflow code → sandbox exec        │   workflow_runner.py)
                └──────────────┬───────────────────────┘
                               ▼
                ┌──────────────────────────────────────┐
                │  3-layer verifier                    │  (verifier.py)
                │  L1 abstracted diagnosis             │
                │  L2 task-locked checklist            │
                │  L3 independent cheat audit          │
                └──────────────┬───────────────────────┘
                               ▼
                ┌──────────────────────────────────────┐
                │  selection.validate_survivor()       │  (selection.py)
                │  → QD score, archive update          │
                └──────────────┬───────────────────────┘
                               ▼
                          next iteration
```

The whole loop lives in [EvolutionEngine.evolve_generation()](../sources/core/evolution_engine.py:359),
which recurses depth-first up to `max_depth` iterations. Despite being a
recursive single trajectory, the **archive** in
[SelectionPressure](../sources/core/selection.py:38) makes the *parent pool*
behave like a population of size up to 100.

---

## 2. Representation — the genotype/phenotype split

| Layer | Where | What |
|---|---|---|
| **Genotype** | `workflow_dir/<uuid>/workflow_genotype_<uuid>.py` | Full Python source of the workflow (agents, tools, control flow). Loaded by [`load_workflow_genotype_code`](../sources/core/evolution_engine.py:149). |
| **Phenotype** | `workflow_dir/<uuid>/state_result.json` + workspace artifacts | Execution trace: per-agent answers, intermediate files, final outputs. |
| **Fitness** | `overall_score` (verifier) on `WorkflowInfo` | Scalar in [0, 1], capped at `_HARD_FAIL_CAP = 0.7` when a hard claim refutes the run. |
| **Memory key** | `original_task` text, embedded with `all-MiniLM-L6-v2` | Cosine similarity ≥ 0.5 to retrieve cold-start parents from disk ([WorkflowSelector](../sources/core/workflow_selection.py:78)). |

**Substrate**: code-space (the most expressive of the abstraction levels
discussed in the V2 context). The mutator does not edit AST nodes — it
prompts an LLM to rewrite the whole Python source given the parent code,
diagnosis, and a sampled perturbation. This sits between Promptbreeder
(prompt-only) and DGM/ADAS (full code-space agents). It pays the price for
expressivity in sandboxed execution cost per iteration.

**Behavior descriptor**: an AST-parsed structural vector
`[n_agents, n_edges, n_branches, prompt_chars]`, each normalised by a
fixed scale so axes contribute comparably to k-NN Euclidean distance
([code_features.py](../sources/core/code_features.py),
called from [selection.py:_extract_behaviour_descriptor](../sources/core/selection.py)).
This replaces the original `[reward, cost, iteration_count]`, which was
collinear with fitness and made novelty collapse to "I am new because I
came later." The new descriptor is orthogonal to reward and tracks
**granularity** of the workflow (per §2.5 of
[math_lens.md](math_lens.md)). It does not yet capture **basin
identity** (prompt content) or **boundary discontinuity D** — see §4.3
of math_lens for the tiered upgrade path.

---

## 3. Selection — archive + parent draw

### 3.1 Strategy

`SelectionPressure` ([selection.py](../sources/core/selection.py)) supports
four strategies; the default wired in `EvolutionEngine.__init__` is **QD**
with:

- `population_size=100`
- `novelty_k_neighbours=5`
- `novelty_weight=0.4` (60% quality, 40% novelty)
- `min_improvement_threshold=0.01`

### 3.2 Two-stage selection

1. **Survivor validation** ([selection.py:81](../sources/core/selection.py:81))
   runs after each new run. In QD mode it computes a behavior descriptor,
   measures k-NN novelty against the archive, and combines quality with
   novelty into `qd_score`. Quality uses
   [`reward_uncapped`](../sources/core/workflow_info.py)
   (`max(0, base_mean + info_bonus − cheat_penalty)`) — the 0.7
   hard-fail cap is kept for admissibility decisions but stripped from
   the parent-draw signal so distinct refuted-but-improving runs stay
   rank-ordered. The `is_valid` flag is set if `relative_improvement >
   threshold` **or** `qd_score > admit_threshold` (default 0.3) — the
   latter makes the loop genuinely open-ended: regressions can survive
   when behaviorally novel. Admission to the archive is then gated by
   [`_try_admit`](../sources/core/selection.py) on
   `is_valid AND not Pareto-dominated on (reward_uncapped, novelty_score)
   by an existing member`. Rejections increment `_n_admit_rejected` and
   surface as a per-iteration `admit_rejected` flag on
   [`SelectionLog`](../sources/core/schema.py).
2. **Parent draw** ([selection.py:112](../sources/core/selection.py:112) →
   `select_parent` / `select_parents`) samples from the archive weighted by
   `qd_score` (a roulette-style pick). For crossover, the second parent is
   drawn from the residual pool without replacement.

### 3.3 Cold start

When the archive is empty (start of a session — `_archive = []` is reset in
`start_workflow_evolution` at [evolution_engine.py:291](../sources/core/evolution_engine.py:291)),
[`WorkflowSelector.select_parent_workflows`](../sources/core/workflow_selection.py:170)
falls back to a **similarity-filtered disk scan**: encode the task with
MiniLM, retrieve workflows with cosine ≥ 0.5 and score ≥ 0.05, then route
those through the same `SelectionPressure.select_parents` so the QD weighting
still applies. This is Mimosa's only cross-task transfer mechanism.

### 3.4 Crossover vs. mutation

Decided stochastically per iteration with `crossover_rate` ≈ 0.3–0.4
([evolution_engine.py:476](../sources/core/evolution_engine.py:476) calls
`select_parent_workflow` with no rate, so the default fires). Crossover
requires ≥ 2 parents post-rehydration; otherwise it silently falls back to
mutation.

### 3.5 Neuroevolution reading

| Principle (from the skill) | Mimosa today | Verdict |
|---|---|---|
| Archives beat incumbents | Archive of 100 with QD weighting + admit gate | ✅ Matches DGM/ADAS pattern; admit gate cuts archive bloat from unconditional appends. |
| QD is the default open-endedness scheme | Novelty + quality on AST-derived structural descriptor `[n_agents, n_edges, n_branches, prompt_chars]` | ⚠️ Descriptor is orthogonal to fitness but only covers granularity (math_lens §4.2); basin identity and boundary D are not yet captured. |
| Mode collapse is the default failure mode | perturbations + tried-strategy memory + admit gate Pareto check | ⚠️ Diversity injected at the prompt layer; population-layer dedup now via Pareto on `(reward_uncapped, novelty)`. |
| Selection bias = `qd_score` roulette | `random.choices(weights=qd)` over `reward_uncapped`-driven `qd_score` | ⚠️ No inverse-child-count term — high-`qd` parents can dominate offspring. |
| Cross-task transfer via similar tasks | Cosine ≥ 0.5 on MiniLM of `original_task` | ⚠️ Pure embedding similarity collapses behaviorally distinct solutions (the skill explicitly flags this as a V1 limitation). |

---

### 4 Annealing + complexity curriculum

[`_get_temperature_phase`](../sources/core/variation_engine.py:30)
returns one of six phase blocks based on
`progress = (iter / (max_iter-1)) ** (1 - 0.5 * score)`. The exponent makes
high-scoring runs progress *slower* (stay exploratory longer) and
low-scoring runs progress *faster* (commit to convergence earlier).

| Phase | Progress | Budget | Behavior |
|---|---|---|---|
| SEED | < 0.10 | 2–3 agents | Minimum-viable chain. No validators. |
| BOOTSTRAP | < 0.25 | 3–5 agents, sequential | Add ≤ 1 agent per iter. Sequential only. |
| DIVERGE | < 0.40 | 4–7 agents, branching/loops OK | Force qualitatively different topologies. |
| SCALE | < 0.60 | 5+ agents | Add one purposeful layer (validator/critic/fallback). |
| CONVERGE | < 0.80 | freeze count | Rewrite worst-agent prompt. No new agents. |
| POLISH | ≥ 0.80 | frozen | Micro-improvements only. |

This is a **complexity-earned** curriculum: agents must be justified by the
failures of simpler topologies. Pragmatically it lowers the per-iter
exploration cost early when debugging is cheapest.

### 4.3 Crossover

[`crossover_prompt`](../sources/core/variation_engine.py:258) sorts parents
best-first (LLM primacy bias inherits strong traits) and prompts for
*genuine recombination*, not parent-and-patch. Note: the rubric-blind
**abstracted diagnosis** (verifier Layer 1) is used in place of raw judge
output, so the recombinator doesn't learn rubric-shaped patterns.

### 4.4 Neuroevolution reading

| Principle | Mimosa today | Verdict |
|---|---|---|
| Mode-collapse mitigation | Tried-strategy dedup + 5-axis combinatorial sampling | ✅ At the prompt layer. ⚠️ No island migration, no MAP-Elites grid. |
| Abstraction level for long runs | Code-space rewrite | ✅ Matches DGM/ADAS for 20+ iter horizons. |
| Diversity inheritance under crossover | Best-first parent ordering + rubric-blind diagnosis | ✅ Smart anti-Goodhart move. ⚠️ Single-shot LLM recombination has no structural recombination operator — fully relies on the LLM to "do the right thing." |

---

## 5. Evaluation — the 3-layer verifier

[`VerifierEvaluator`](../sources/core/evaluators/verifier.py) is the
fitness function. It is structured explicitly as a **defense in depth
against Goodhart's law** (see recent commit `dd13315 feat (verifier): close
Goodhart leaks with 3-layer defense`).

### 5.1 Pipeline

```
state_result.json ─► extract claims ─► generate verifier scripts
                            │                       │
                            ▼                       ▼
                  task-locked checklist     sandboxed execution
                   (if installed)            per claim
                            │                       │
                            └────────► aggregate ◄──┘
                                         │
                                         ▼
                                 cheat penalty (Layer 3)
                                         │
                                         ▼
                              abstracted diagnosis (Layer 1)
                                         │
                                         ▼
                                 overall_score in [0, 1]
```

### 5.2 The three layers — and what each closes

**Layer 1 — Abstracted diagnosis** ([verifier.py:266](../sources/core/evaluators/verifier.py:266)).
After scoring, the judge writes a *rubric-blind* plain-language summary of
what the workflow failed to do. This is the only feedback signal the
mutator sees about the verifier — by design it does not leak which claims
were checked. Without this, the mutator quickly learns to **author its own
exam**: print `"loaded ClinTox dataset"` because that's the literal string
the verifier extracted last time.

**Layer 2 — Task-locked checklist**
([task_checklist.py](../sources/core/evaluators/task_checklist.py),
installed in [evolution_engine.py:566](../sources/core/evolution_engine.py:566)).
Built **once per task** from the task spec + literature grounding,
*before* any candidate runs. The verifier then derives its claim set from
this checklist instead of from agent narration. This closes the loop where
"the workflow that prints X" → "the verifier checks for X" → "all workflows
that know to print X pass."

**Layer 3 — Independent cheat detector**
([cheat_detector.py](../sources/core/evaluators/cheat_detector.py)). Reads
the task spec + the workflow's `.py` source, **never** the claim list or
evaluation output. Returns a penalty in [0, 1] subtracted from
`overall_score`, plus two-way split findings:
- *behavioral* — safe to feed back to mutator (says *what* went wrong at
  task level, not *how* it was detected).
- *mechanism* — for audit reports only; leaking mechanism teaches the
  mutator to hide cheats rather than avoid them.

### 5.3 Per-claim scoring

For each claim, the verifier generates a small Python script that:
- runs in the agents' workspace,
- opens files in `likely_relevant_files`,
- **recomputes** the asserted value from on-disk artefacts,
- emits one JSON line: `{"claim_id": ..., "status": "pass|fail|error", ...}`.

Two anti-cheat tripwires fire on the verifier itself
([_verifier_appears_to_cheat](../sources/core/evaluators/verifier.py:91)):
- Reject if any string literal of ≥ 80 chars overlaps a 60-char window of
  the execution text (the verifier is parsing the agent's answer back to
  itself — tautology).
- Reject if `likely_relevant_files` is non-empty but no I/O markers appear
  in the script. One retry with explicit feedback; then fall back to soft
  LLM verdict.

Soft (non-executable) claims get a 3-verdict LLM check anchored on
workspace previews + literature grounding (`pass`/`unsure`/`fail` mapped to
1.0/0.5/0.0).

### 5.4 Aggregation

[`_aggregate`](../sources/core/evaluators/verifier.py:1039):
```
overall = clamp(base_mean + info_bonus, 0, 1)
if any hard claim explicitly failed:
    overall = min(overall, 0.7)        # _HARD_FAIL_CAP
overall = max(0, overall - cheat_penalty)
```
where `info_bonus(n_hard_pass) = 0.15 · (1 - exp(-n_hard_pass / 4))` —
saturating reward for thoroughness, gameable spam yields nothing because
only **passing hard** claims count.

### 5.5 Neuroevolution reading

| Principle | Mimosa today | Verdict |
|---|---|---|
| Single-judge evaluation leaks signal | Per-claim ensemble (12–24 atomic checks) + cheat detector + per-task checklist | ✅ Best-in-class anti-Goodhart in the systems catalog. The 3-layer split is closer to "criticality injection + ensemble" than to single judge. |
| Debate/ensemble for robustness | No multi-judge debate; each claim is checked once | ⚠️ Could escalate finalists to k=3 ensemble per RewardBench-2 finding. |
| Scientific correctness is often delayed/unverifiable | Literature grounding via Perspicacite, hard cap at 0.7 for any refuted hard claim | ✅ Acknowledged. Hard cap is a sensible conservative move. |
| Self-evolving systems drift | Cheat detector independent epistemology; checklist fixed pre-evolution; rubric-blind feedback | ✅ Three independent epistemic surfaces is unusual in the literature and exactly what the alignment-drift heuristic recommends. |

---

## 6. Anti-Goodhart epistemic isolation

This is the conceptual core of V2 and worth pulling out. **Three feedback
surfaces, three independent epistemologies:**

| Surface | Sees | Hidden from |
|---|---|---|
| **Verifier (Layer 2)** | task spec + checklist + workspace artefacts | agent narration during verifier *generation* in checklist mode |
| **Cheat detector (Layer 3)** | task spec + workflow `.py` | claims, evaluation output, verifier scripts |
| **Mutator** | abstracted diagnosis (Layer 1) + behavioral cheat findings | raw claim list, raw judge logs, cheat *mechanism* findings |

The invariant: the mutator can learn to *do the task better* but cannot
learn to *satisfy the verifier without doing the task*, because the
artifacts that would teach it that — claim ids, check patterns, detection
mechanisms — are not in its loss signal.

The phrase to remember: the workflow is no longer **authoring its own exam**.

---

## 7. Where Mimosa V2 stands vs. the neuroevolution canon

Cross-referencing the skill's decision heuristics:

| Question | Skill default | Mimosa V2 today | Gap |
|---|---|---|---|
| Selection scheme | Archive + parent selection weighted by fitness × inverse-child-count | Archive + QD-weighted roulette + admit gate (Pareto on `(reward_uncapped, novelty)`); no child-count term yet | Minor — add child-count penalty to avoid over-mining the same parent. |
| Diversity mechanism | CVT-MAP-Elites over hybrid behavior descriptor | k-NN novelty with AST-derived structural descriptor | **Partial fix**. Descriptor left fitness-space; not yet MAP-Elites bins. |
| Behavior descriptors | Topology + execution features (graph complexity, tool diversity, …) | `[n_agents, n_edges, n_branches, prompt_chars]` from AST walk | **Partial fix**. Covers granularity only; basin identity (prompt content) and boundary D not captured — see math_lens §4.3 for tiered upgrade. |
| Judge | Criteria injection + k=3 ensemble; debate for finalists | Per-claim atomic check ensemble + cheat audit + checklist | Mostly ✅. Could add debate on borderline aggregates. |
| Memory key | Hybrid task embedding + behavior descriptors | Pure MiniLM cosine on task text | Inherits V1 limitation. Needs behavior-aware retrieval. |
| Mutation substrate | Code-space for long runs | Code-space | ✅. |
| Abstraction level | Topology + skill library, evolved jointly | Topology only; no skill library | **Open**. Skill library (Voyager/CASCADE) not present yet. |

### Strengths over the canon
- The **3-layer Goodhart defense** is more aggressive than anything in the
  systems catalog. The independent epistemology between verifier ⊥ cheat
  detector ⊥ mutator is a Mimosa-specific contribution.
- The **complexity-earned curriculum** with score-modulated annealing is a
  pragmatic improvement on flat exploration schedules.

### Most leveraged next moves (not implementing them here — pointers only)
1. **Upgrade the behavior descriptor.** Current AST descriptor covers
   granularity. The math_lens §4 framing argues the next axes to add are
   *basin identity* (prompt-content embedding via MiniLM + random
   projection — Tier 2) and eventually *boundary discontinuity D* via
   logprob surprisal (Tier 3). Drop `prompt_chars` when prompt-content
   axes land.
2. **MAP-Elites grid.** Bin the structural descriptor and keep one elite
   per cell instead of an unbounded-capacity archive evicting by
   `qd_score`. The admit gate already filters out dominated candidates;
   binning would replace the Pareto check with the canonical QD design.
4. **Parent-draw child-count penalty.** Divide `qd_score` by
   `(1 + n_children_already)` in `select_parent` to spread offspring.
5. **Skill library.** Persist code fragments (agents, tool-binding
   patterns) that pass the verifier with high `n_hard_pass`, exposed as
   reusable components to the mutator. This is the CASCADE/Voyager
   direction the skill flags as the abstraction-level frontier.

---

## 8. Code map (one-line per file)

- [evolution_engine.py](../sources/core/evolution_engine.py) — top-level
  evolutionary loop, recursion over generations, plotting, notifications.
- [selection.py](../sources/core/selection.py) — `SelectionPressure` +
  archive + survivor validation + admit gate (`_try_admit`,
  `_is_dominated`), four strategies (greedy, tournament, novelty, QD).
- [code_features.py](../sources/core/code_features.py) — AST-derived
  behaviour descriptor (`extract_code_features`) returning
  `[n_agents, n_edges, n_branches, prompt_chars]` normalised.
- [workflow_selection.py](../sources/core/workflow_selection.py) — parent
  retrieval; steady-state archive draw vs. cold-start disk scan with MiniLM
  similarity.
- [variation_engine.py](../sources/core/variation_engine.py) — prompt
  assembly for mutation/crossover; annealing curriculum.
- [orchestrator.py](../sources/core/orchestrator.py) — meta-agent → workflow
  factory → sandbox runner pipeline.
- [evaluators/verifier.py](../sources/core/evaluators/verifier.py) — per-claim
  verifier, aggregation, 3-layer defense glue.
- [evaluators/task_checklist.py](../sources/core/evaluators/task_checklist.py) —
  Layer 2 task-locked checklist built once per task hash.
- [evaluators/cheat_detector.py](../sources/core/evaluators/cheat_detector.py) —
  Layer 3 independent script audit.
- [evaluators/grounding.py](../sources/core/evaluators/grounding.py) —
  Perspicacite literature-grounding adapter used by both checklist build
  and verifier claim extraction.
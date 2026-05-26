# Mimosa V2 — Evolution Algorithm

A neuroevolution-lens reading of the current evolution loop. Maps the
code in [sources/core/](../sources/core/) onto the four canonical
components of an evolutionary algorithm — **representation**,
**selection**, **variation**, **evaluation** — flags where the
implementation departs from neuroevolution best practice, and points to
the open V2 design questions.

For visual companions:
- [Evolution loop diagram](diagrams/evolution_loop.mermaid) (PNG: [images/evolution_loop.png](images/evolution_loop.png))
- [Evaluation pipeline diagram](diagrams/evaluation_pipeline.mermaid) (PNG: [images/evaluation_pipeline.png](images/evaluation_pipeline.png))
- [Overall architecture diagram](diagrams/architecture_overall.mermaid) (PNG: [images/architecture_overall.png](images/architecture_overall.png))

---

## 1. The big picture

Mimosa evolves multi-agent **workflows**: Python programs that wire
together LLM agents, tools, and control flow into a graph that attempts
to solve a scientific task. A single evolutionary run looks like this:

```
                      ┌───────────────────────────────┐
   task spec  ─────►  │  install task-locked          │
                      │  verification checklist       │  (task_checklist.py)
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
                │  + phase-aware annealing schedule    │
                │  + rubric-blind diagnosis feedback   │
                └──────────────┬───────────────────────┘
                               ▼
                ┌──────────────────────────────────────┐
                │  orchestrator → meta-agent writes    │  (orchestrator.py,
                │  workflow code → sandbox exec        │   workflow_runner.py)
                └──────────────┬───────────────────────┘
                               ▼
                ┌──────────────────────────────────────┐
                │  3-layer verifier                    │  (evaluators/verifier.py)
                │  L1 abstracted diagnosis             │
                │  L2 task-locked checklist            │
                │  L3 independent cheat audit          │
                └──────────────┬───────────────────────┘
                               ▼
                ┌──────────────────────────────────────┐
                │  selection.validate_survivor()       │  (selection.py)
                │  → QD score, admit gate, archive     │
                └──────────────┬───────────────────────┘
                               ▼
                          next iteration
```

The whole loop lives in
[`EvolutionEngine.evolve_generation()`](../sources/core/evolution_engine.py),
which **recurses depth-first** up to
`config.max_learning_evolve_iterations` (default **25**). Despite being a
recursive single trajectory, the **session archive** in
[`SelectionPressure`](../sources/core/selection.py) makes the parent
pool behave like a population of up to **50** members.

---

## 2. Representation — the genotype/phenotype split

| Layer | Where | What |
|---|---|---|
| **Genotype** | `workflow_dir/<uuid>/workflow_genotype_<uuid>.py` | Full Python source of the workflow (agents, tools, control flow). Loaded by [`load_workflow_genotype_code`](../sources/core/evolution_engine.py). |
| **Phenotype** | `workflow_dir/<uuid>/state_result.json` + workspace artefacts | Execution trace: per-agent answers, intermediate files, final outputs. |
| **Fitness (admit signal)** | `overall_score` on `WorkflowInfo` | Scalar in [0, 1], capped at `_HARD_FAIL_CAP = 0.94` when a hard claim refutes the run. |
| **Fitness (rank signal)** | `reward_uncapped` = `base_mean + info_bonus − cheat_penalty` | Uncapped quality used by the QD archive so distinct refuted-but-improving runs stay rank-ordered. |
| **Memory key** | `original_task` text, embedded with `all-MiniLM-L6-v2` | Cosine similarity ≥ 0.5 to retrieve cold-start parents from disk ([`WorkflowSelector`](../sources/core/workflow_selection.py)). |
| **Lineage** | `workflow_dir/<uuid>/lineage_<uuid>.json` | `{parents, evolution_kind, iteration, goal_snippet}` — written by [`record_lineage`](../sources/core/lineage.py); consumed by [`evolution_tree.py`](../sources/utils/evolution_tree.py) to render the tree PNG. |

**Substrate**: code-space (the most expressive of the abstraction levels
discussed in the V2 context). The mutator does not edit AST nodes — it
prompts an LLM to rewrite the whole Python source given the parent code,
diagnosis, and a phase-gated schedule. This sits between Promptbreeder
(prompt-only) and DGM/ADAS (full code-space agents). It pays the price
for expressivity in sandboxed execution cost per iteration.

**Behaviour descriptor**: an AST-parsed structural vector
`[n_agents, n_edges, n_branches, prompt_chars]`, each normalised by a
fixed scale `(10, 10, 10, 5000)` so axes contribute comparably to k-NN
Euclidean distance
([`code_features.py`](../sources/core/code_features.py), called from
[`selection.py::_extract_behaviour_descriptor`](../sources/core/selection.py)).
This replaces the original `[reward, cost, iteration_count]`, which was
collinear with fitness and made novelty collapse to *"I am new because
I came later."* The new descriptor is orthogonal to reward and tracks
**granularity** of the workflow (per §2.5 of
[math_lens.md](math_lens.md)). It does not yet capture **basin
identity** (prompt content) or **boundary discontinuity D** — see §4.3
of math_lens for the tiered upgrade path.

---

## 3. Selection — archive + parent draw

### 3.1 Strategy

`SelectionPressure` ([selection.py](../sources/core/selection.py))
supports four strategies; the default wired in `EvolutionEngine.__init__`
is **QD** with:

- `population_size = 50`
- `novelty_k_neighbours = 25`
- `novelty_weight = 0.4` (60 % quality, 40 % novelty)
- `min_improvement_threshold = 0.01`
- `admit_threshold = 0.3` (selection.py default)
- `initial_population = 2` — first two runs are always seeded; archive
  draw kicks in from iteration 3 onward.

### 3.2 Two-stage selection

1. **Survivor validation**
   ([`selection.py::_validate_open_ended`](../sources/core/selection.py))
   runs after each new run. In QD mode it computes a behaviour
   descriptor, measures k-NN novelty against the archive, and combines
   quality with novelty into `qd_score`. Quality uses `reward_uncapped`
   — the 0.94 hard-fail cap is kept for admissibility decisions but
   stripped from the parent-draw signal so distinct refuted-but-improving
   runs stay rank-ordered. The `is_valid` flag is set if
   `relative_improvement > threshold` **or** `qd_score > admit_threshold`
   — the latter makes the loop genuinely open-ended: regressions can
   survive when behaviourally novel. Admission to the archive is then
   gated by [`_try_admit`](../sources/core/selection.py) on
   `is_valid AND not Pareto-dominated on (reward_uncapped, novelty_score)
   by an existing member` (with ε-bands). Rejections increment
   `_n_admit_rejected` and surface as a per-iteration `admit_rejected`
   flag on [`SelectionLog`](../sources/core/schema.py).

2. **Parent draw** (`select_parent` / `select_parents`) samples from the
   archive weighted by
   `qd_score ÷ (1 + n_children_already)` — the inverse-child-count
   penalty implements §7 leveraged-move #4 from this document, spreading
   offspring instead of letting a single high-qd ancestor monopolise the
   stream. For crossover, the second parent is drawn from the residual
   pool without replacement.

### 3.3 Cold start

When the archive is empty (start of a session —
`_archive = []` is reset in `start_workflow_evolution` at
[evolution_engine.py:259](../sources/core/evolution_engine.py:259)),
[`WorkflowSelector.select_parent_workflows`](../sources/core/workflow_selection.py)
falls back to a **similarity-filtered disk scan**: encode the task with
MiniLM, retrieve workflows with cosine ≥ 0.5 and score ≥ 0.05, then
route those through the same `SelectionPressure.select_parents` so the
QD weighting still applies. This is Mimosa's only cross-task transfer
mechanism.

### 3.4 Crossover vs. mutation

Decided stochastically per iteration. `EvolutionEngine.select_parent_workflow`
forwards `crossover_rate = 0.4` to
`workflow_selection.select_parent_workflows`, which fires crossover when
`len(candidates) >= 2 AND random() < crossover_rate`. Crossover requires
≥ 2 parents post-rehydration; otherwise it silently falls back to
mutation. The first `initial_population = 2` iterations are forced into
the SEED branch even if parents are available — this guarantees a
non-trivial archive before mutation/crossover can fire.

### 3.5 Neuroevolution reading

| Principle (from the skill) | Mimosa today | Verdict |
|---|---|---|
| Archives beat incumbents | Archive of 50 with QD weighting + admit gate | ✅ Matches DGM/ADAS pattern; admit gate cuts archive bloat from unconditional appends. |
| QD is the default open-endedness scheme | Novelty + quality on AST-derived structural descriptor `[n_agents, n_edges, n_branches, prompt_chars]` | ⚠️ Descriptor is orthogonal to fitness but only covers granularity (math_lens §4.2); basin identity and boundary D are not yet captured. |
| Mode collapse is the default failure mode | Phase-aware annealing + admit gate Pareto check + inverse-child-count penalty | ✅ Diversity injected at the prompt and population layers. ⚠️ No island migration, no MAP-Elites grid. |
| Selection bias = `qd_score` roulette | `random.choices(weights=qd / (1+n_children))` over `reward_uncapped`-driven `qd_score` | ✅ Inverse-child-count term landed. |
| Cross-task transfer via similar tasks | Cosine ≥ 0.5 on MiniLM of `original_task` | ⚠️ Pure embedding similarity collapses behaviourally distinct solutions (V1 limitation). |

---

## 4 Variation — annealing + complexity curriculum

[`_get_temperature_phase`](../sources/core/variation_engine.py)
returns one of five phase blocks based on
`progress = (iter / (max_iter-1)) ** (1 - α·score)` with `α = 0.5`. The
exponent makes high-scoring runs progress *slower* (stay exploratory
longer) and low-scoring runs progress *faster* (commit to convergence
earlier).

| Phase | Progress | Budget | Behaviour |
|---|---|---|---|
| **SEED** | < 0.25 | 1–2 agents | Strongest possible single-agent workflow with relevant tools. |
| **ANCHOR** | < 0.50 | 1–3 agents | **Largest iteration budget.** Rewrite agent prompts or split verification/steps. Each variant changes ONE thing from the previous best. |
| **DECOMPOSE** | < 0.65 | 2–5 agents | Topology, prompt, handoff format. See if multi-agent shape outperforms a well-prompted single agent. |
| **ENGAGE** | < 0.85 | 2–6 agents | Tighten the workflow you have; remove agents that aren't earning their place. Sharpen solver/executor prompts. |
| **POLISH** | ≥ 0.85 | frozen | One prompt fix per iteration, targeting the single most concrete failure. No topology change. |

This is a **complexity-earned** curriculum: agents must be justified by
the failures of simpler topologies. Pragmatically it lowers the
per-iter exploration cost early when debugging is cheapest. The
ordering — SEED → ANCHOR (prompt) → DECOMPOSE (topology) — is the new
"prompt before topology" EV-ranked schedule (previous V2 versions ran
topology earlier).

### 4.3 Crossover

[`crossover_prompt`](../sources/core/variation_engine.py) sorts parents
best-first (LLM primacy bias inherits strong traits) and prompts for
*genuine recombination*, not parent-and-patch. The rubric-blind
**abstracted diagnosis** (verifier Layer 1) is used in place of raw
judge output, so the recombinator doesn't learn rubric-shaped patterns.

### 4.4 Neuroevolution reading

| Principle | Mimosa today | Verdict |
|---|---|---|
| Mode-collapse mitigation | Phase-aware annealing + inverse-child-count penalty + Pareto admit gate | ✅ At the prompt *and* population layer. ⚠️ No island migration, no MAP-Elites grid. |
| Abstraction level for long runs | Code-space rewrite | ✅ Matches DGM/ADAS for 25-iter horizons. |
| Diversity inheritance under crossover | Best-first parent ordering + rubric-blind diagnosis | ✅ Smart anti-Goodhart move. ⚠️ Single-shot LLM recombination has no structural recombination operator — fully relies on the LLM to "do the right thing." |

---

## 5. Evaluation — the 3-layer verifier

[`VerifierEvaluator`](../sources/core/evaluators/verifier.py) is the
fitness function. It is structured explicitly as a **defense in depth
against Goodhart's law**.

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
                                 reward_uncapped (no cap)
```

See [evaluation_pipeline.mermaid](diagrams/evaluation_pipeline.mermaid)
for the rendered version.

### 5.2 The three layers — and what each closes

**Layer 1 — Abstracted diagnosis** ([verifier.py](../sources/core/evaluators/verifier.py)).
After scoring, the judge writes a *rubric-blind* plain-language summary
of what the workflow failed to do. This is the only feedback signal the
mutator sees about the verifier — by design it does not leak which
claims were checked. Without this, the mutator quickly learns to
**author its own exam**: print `"loaded ClinTox dataset"` because that's
the literal string the verifier extracted last time.

**Layer 2 — Task-locked checklist**
([task_checklist.py](../sources/core/evaluators/task_checklist.py),
installed by [`EvolutionEngine._install_task_checklist`](../sources/core/evolution_engine.py)).
Built **once per task hash** from the task spec + literature grounding,
*before* any candidate runs, and cached at
`workflow_dir/_task_checklists/task_checklist_<task_hash>.json`. The
verifier then derives its claim set from this checklist instead of from
agent narration. This closes the loop where
"the workflow that prints X" → "the verifier checks for X" → "all
workflows that know to print X pass."

**Layer 3 — Independent cheat detector** (inside `verifier.py`). Reads
the task spec + the workflow's `.py` source, **never** the claim list
or evaluation output. Returns a penalty in [0, 1] subtracted from
`overall_score`, plus two-way split findings:
- *behavioural* — safe to feed back to mutator (says *what* went wrong
  at task level, not *how* it was detected).
- *mechanism* — for audit reports only; leaking mechanism teaches the
  mutator to hide cheats rather than avoid them.

### 5.3 Per-claim scoring

For each claim, the verifier generates a small Python script that:
- runs in the agents' workspace,
- opens files in `likely_relevant_files`,
- **recomputes** the asserted value from on-disk artefacts,
- emits one JSON line: `{"claim_id": ..., "status": "pass|fail|error", ...}`.

Two anti-cheat tripwires fire on the verifier itself
(`_verifier_appears_to_cheat`):
- Reject if any string literal of ≥ 80 chars overlaps a 60-char window
  of the execution text (the verifier is parsing the agent's answer
  back to itself — tautology).
- Reject if `likely_relevant_files` is non-empty but no I/O markers
  appear in the script. One retry with explicit feedback; then fall
  back to soft LLM verdict.

Soft (non-executable) claims get a 3-verdict LLM check anchored on
workspace previews + literature grounding (`pass`/`unsure`/`fail` mapped
to `1.0 / 0.5 / 0.0`).

### 5.4 Aggregation

[`_aggregate`](../sources/core/evaluators/verifier.py):
```
overall = clamp(base_mean + info_bonus, 0, 1)
if any hard claim explicitly failed:
    overall = min(overall, 0.94)         # _HARD_FAIL_CAP
overall = max(0, overall - cheat_penalty)
```
where
```
info_bonus(n_hard_pass) = 0.15 · (1 - exp(-n_hard_pass / 8))
```
(`_INFO_BONUS_ALPHA = 0.15`, `_INFO_BONUS_BETA = 8`) — saturating reward
for thoroughness; gameable spam yields diminishing returns because only
**passing hard** claims count.

The 0.94 cap is intentionally close to 1.0 (and well above the
`learned_score_threshold = 0.95` floor it would block); it tags runs
that have hard refutations without flattening them down to a tier where
the mutator can't tell the difference between *"close but cheats"* and
*"close but legitimately almost there."*

### 5.5 Neuroevolution reading

| Principle | Mimosa today | Verdict |
|---|---|---|
| Single-judge evaluation leaks signal | Per-claim ensemble (typ. 5–20 atomic checks) + cheat detector + per-task checklist | ✅ Best-in-class anti-Goodhart in the systems catalog. The 3-layer split is closer to "criticality injection + ensemble" than to single judge. |
| Debate/ensemble for robustness | No multi-judge debate; each claim is checked once | ⚠️ Could escalate finalists to k=3 ensemble per RewardBench-2 finding. |
| Scientific correctness is often delayed/unverifiable | Literature grounding via Perspicacite, hard cap at 0.94 for any refuted hard claim | ✅ Acknowledged. Hard cap is a sensible conservative move. |
| Self-evolving systems drift | Cheat detector independent epistemology; checklist fixed pre-evolution; rubric-blind feedback | ✅ Three independent epistemic surfaces is unusual in the literature and exactly what the alignment-drift heuristic recommends. |

---

## 6. Anti-Goodhart epistemic isolation

This is the conceptual core of V2 and worth pulling out. **Three
feedback surfaces, three independent epistemologies:**

| Surface | Sees | Hidden from |
|---|---|---|
| **Verifier (Layer 2)** | task spec + checklist + workspace artefacts | agent narration during verifier *generation* in checklist mode |
| **Cheat detector (Layer 3)** | task spec + workflow `.py` | claims, evaluation output, verifier scripts |
| **Mutator** | abstracted diagnosis (Layer 1) + behavioural cheat findings | raw claim list, raw judge logs, cheat *mechanism* findings |

The invariant: the mutator can learn to *do the task better* but cannot
learn to *satisfy the verifier without doing the task*, because the
artifacts that would teach it that — claim ids, check patterns,
detection mechanisms — are not in its loss signal.

The phrase to remember: the workflow is no longer **authoring its own
exam**.

---

## 7. Where Mimosa V2 stands vs. the neuroevolution canon

Cross-referencing the skill's decision heuristics:

| Question | Skill default | Mimosa V2 today | Gap |
|---|---|---|---|
| Selection scheme | Archive + parent selection weighted by fitness × inverse-child-count | Archive + QD-weighted roulette + inverse-child-count penalty + admit gate (Pareto on `(reward_uncapped, novelty)`) | ✅ child-count term has landed. |
| Diversity mechanism | CVT-MAP-Elites over hybrid behaviour descriptor | k-NN novelty with AST-derived structural descriptor | **Partial fix**. Descriptor left fitness-space; not yet MAP-Elites bins. |
| Behaviour descriptors | Topology + execution features (graph complexity, tool diversity, …) | `[n_agents, n_edges, n_branches, prompt_chars]` from AST walk | **Partial fix**. Covers granularity only; basin identity (prompt content) and boundary D not captured — see math_lens §4.3 for tiered upgrade. |
| Judge | Criteria injection + k=3 ensemble; debate for finalists | Per-claim atomic check ensemble + cheat audit + checklist | Mostly ✅. Could add debate on borderline aggregates. |
| Memory key | Hybrid task embedding + behaviour descriptors | Pure MiniLM cosine on task text | Inherits V1 limitation. Needs behaviour-aware retrieval. |
| Mutation substrate | Code-space for long runs | Code-space | ✅. |
| Abstraction level | Topology + skill library, evolved jointly | Topology only; no skill library | **Open**. Skill library (Voyager/CASCADE) not present yet. |

### Strengths over the canon
- The **3-layer Goodhart defense** is more aggressive than anything in
  the systems catalog. The independent epistemology between
  verifier ⊥ cheat detector ⊥ mutator is a Mimosa-specific contribution.
- The **complexity-earned curriculum** with score-modulated annealing
  (now in the SEED → ANCHOR → DECOMPOSE → ENGAGE → POLISH form) is a
  pragmatic improvement on flat exploration schedules.
- The **inverse-child-count penalty** in `select_parent` materially
  prevents single-parent monopolies that early QD runs exhibited.

### Most leveraged next moves (pointers only, not implementing here)
1. **Upgrade the behaviour descriptor.** Current AST descriptor covers
   granularity. The math_lens §4 framing argues the next axes to add
   are *basin identity* (prompt-content embedding via MiniLM + random
   projection — Tier 2) and eventually *boundary discontinuity D* via
   logprob surprisal (Tier 3). Drop `prompt_chars` when prompt-content
   axes land.
2. **MAP-Elites grid.** Bin the structural descriptor and keep one elite
   per cell instead of an unbounded-capacity archive evicting by
   `qd_score`. The admit gate already filters out dominated candidates;
   binning would replace the Pareto check with the canonical QD design.
3. **k=3 judge debate on borderline aggregates.** Escalate runs whose
   `overall_score` sits in `[admit_threshold − ε, admit_threshold + ε]`
   to a 3-judge ensemble per RewardBench-2 finding.
4. **Skill library.** Persist code fragments (agents, tool-binding
   patterns) that pass the verifier with high `n_hard_pass`, exposed as
   reusable components to the mutator. This is the CASCADE/Voyager
   direction the skill flags as the abstraction-level frontier.
5. **Behaviour-aware retrieval.** Replace pure MiniLM cosine on
   `original_task` with a hybrid key that mixes task embedding and
   archived behaviour descriptors so cross-task transfer doesn't
   collapse behaviourally distinct solutions.

---

## 8. Code map (one line per file)

- [evolution_engine.py](../sources/core/evolution_engine.py) — top-level
  evolutionary loop, depth-first recursion over generations, plotting,
  Pushover notifications, workspace lifecycle.
- [selection.py](../sources/core/selection.py) — `SelectionPressure`
  + archive + survivor validation + admit gate (`_try_admit`,
  `_is_dominated`), four strategies (greedy, tournament, novelty, QD),
  inverse-child-count penalty.
- [code_features.py](../sources/core/code_features.py) — AST-derived
  behaviour descriptor (`extract_code_features`) returning
  `[n_agents, n_edges, n_branches, prompt_chars]` normalised.
- [workflow_selection.py](../sources/core/workflow_selection.py) —
  parent retrieval; steady-state archive draw vs. cold-start disk scan
  with MiniLM similarity; on-disk child-count for the parent-draw
  penalty.
- [variation_engine.py](../sources/core/variation_engine.py) — prompt
  assembly for mutation/crossover; phase-aware annealing
  (SEED → ANCHOR → DECOMPOSE → ENGAGE → POLISH).
- [orchestrator.py](../sources/core/orchestrator.py) — Perspicacite
  grounding → workflow factory → sandbox runner pipeline.
- [lineage.py](../sources/core/lineage.py) — `seed | mutation |
  crossover` records persisted per workflow folder; consumed by the
  evolution-tree visualiser.
- [evaluators/verifier.py](../sources/core/evaluators/verifier.py) —
  per-claim verifier, aggregation, 3-layer defense glue.
- [evaluators/task_checklist.py](../sources/core/evaluators/task_checklist.py) —
  Layer 2 task-locked checklist built once per task hash.
- [evaluators/grounding.py](../sources/core/evaluators/grounding.py) —
  Perspicacite literature-grounding adapter used by both checklist
  build and verifier claim extraction.
- [evaluators/evaluator.py](../sources/core/evaluators/evaluator.py) —
  `WorkflowEvaluator` facade routing to generic / scenario / verifier
  backends.

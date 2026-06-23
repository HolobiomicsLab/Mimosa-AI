# Part 8 — Source A–E effectiveness audit for SAB-pass

One Opus subagent classified every cached claim across 3 tasks × 5 sources (~120 claims total) as **SR-aligned** (verifying the claim directly informs SR pass/fail), **SR-adjacent** (necessary-but-not-sufficient), or **SR-orthogonal** (claim and SR are independent). Source full prompts: `sources/evaluators/verifier_claim_sources.py:131-429`. Per-source totals across the three tasks:

| | A literature | B goal | **C math** | D reprod | E stats |
|---|---:|---:|---:|---:|---:|
| SR-aligned | 2 | 3 | **4** | 1 | 3 |
| SR-adjacent | 10 | 10 | 11 | 8 | 14 |
| SR-orthogonal | 13 | 13 | **1** | 14 | 12 |
| **% aligned** | 8 % | 12 % | **25 %** | **4 %** | 10 % |
| **% orthogonal** | 52 % | 50 % | **6 %** | **61 %** | 41 % |
| verdict | mostly-orthogonal | mixed (SR-killers hide here) | **effective** | **dead** | mostly-orthogonal |

## Per-source diagnosis

- **Source C (math invariants)** — the **only** source that consistently lands SR-aligned claims. `material_ids_match_test_set`, `output_columns_correct`, `smiles_sequence_match` are literal SR checks. Cannot reach value-level gold-equality (gold not in workspace) but covers schema/cardinality/finiteness. **Grow, seed templates, prioritise.**

- **Source D (reproducibility)** — `requirements.txt`/`pinned-deps`/`workspace-not-cluttered` are SR-irrelevant. The LLM consistently pads with 6–9 claims (vs the prompt's 3 "ALLOWED claim shapes") by **duplicating Source A/B content** — those duplicates are weaker than A/B's originals. Penalised clintox's SR-passing data-correctness in the wrong direction. **Remove or cap at 3 claims, importance ≤ 3.**

- **Source B (goal)** — contains the single best SR-detecting claim across the corpus: `output_columns_exact` on clintox caught the `_probability` suffix bug exactly. But B is diluted by orthogonal concept-paraphrases (`uses_deepchem`, `trains_multitask_model`). The "concept claims sit ALONGSIDE exact-name claims" rule at lines 232-236 doubles the budget at the cost of signal. **Invert: when DATASET PREVIEW exists, exact-column claim MUST be first and concept-claim MUST be dropped.**

- **Source E (statistical)** — has actively **anti-correlated** claims on imbalanced classification: `beats_majority_baseline_fda` on clintox FAILS on competent calibrated models because FDA_APPROVED majority class is 1.0 (no accuracy beats 1.0) — but SR uses AUC, not accuracy. The sentinel-detection block is excellent and SR-positive (it was the only thing that caught the `-999` placeholder bug in prior runs). **Re-prompt: drop the majority-class-accuracy rule, keep the sentinel block.**

- **Source A (literature)** — `MANDATORY GOAL CLAIM` (lines 161-167) is the only SR-friendly slot, and works (bulk_mod's `deliverable_and_performance` is A's one clear SR-aligned claim). The other ~7 A claims per task drift to method names SR ignores. **Force outcome-bar-only emission when literature names a bar; otherwise emit just the deliverable claim and stop.**

## Per-best-workflow effectiveness (signal that actually reached the optimiser)

| | clintox best (0.821, SR=F) | mat_diff best (0.952, SR=F) | bulk_mod best (0.760, SR=T-pred) |
|---|---|---|---|
| Source that caught the actual SR-killer | **B** (`output_columns_exact` FAIL with the exact `_prob` suffix in details) | **none** — all gold-comparison-related claims passed; nothing in any source caches a "feature values round-equal gold" check because they cannot (gold isn't in workspace) | n/a (predicted pass) |
| Was the FAIL importance-weighted enough to halt selection? | **No** — 0.821 ≥ 0.8 threshold, selection halted, SR=False slipped through | n/a | n/a |
| Anti-correlated FAILs that depressed reward wrongly | E's `beats_majority_baseline_*` (2 FAILs on legitimately calibrated model) | none | E's `model_uses_material_features_not_ids` + `no_target_leakage_in_features` (FALSE positives — see Part 7 §3) |

The SR-killer claim **existed** on clintox but was outvoted by 13 passing siblings. Source B's signal was correct but diluted.

## Concrete edits (rank-ordered by expected SR-correlation lift)

| # | file:line | edit | expected impact |
|---|---|---|---|
| 1 | `verifier_claim_sources.py:317-359` | Remove or drastically shrink Source D — keep one claim emitted only when workspace > 50 files; drop manifest/pinning entirely | claim pool drops 4-9 orthogonal items/task; SR-aligned share rises 10% → 15% |
| 2 | `verifier_claim_sources.py:232-236` | Invert Source B's "alongside" rule into "first": when DATASET PREVIEW exists, exact-column claim MUST be first and concept-paraphrases MUST be dropped | makes `output_columns_exact`-style claim deterministic on every task with a preview; would have caught clintox at iter 1 |
| 3 | `verifier_claim_sources.py:260-314` | Grow Source C; append a PRIMARY-KEY claim template: "for each (input_path, output_path) where output is per-row, emit one claim that the primary-key column of output equals input's as a list, in order" | C's SR-aligned share goes 25% → 50%; deterministic capture of clintox SMILES-order and bulk_mod material_id-order |
| 4 | `verifier_claim_sources.py:384-388` | Replace "accuracy > 0.55 on balanced binary" with "use AUC-ROC ≥ 0.6 if probabilities present and classes imbalanced; accuracy otherwise" | removes the 2 anti-correlated FAILs that dragged clintox's E pass rate to 7/12 |
| 5 | `verifier_claim_sources.py:161-167, 172` | Force Source A to emit only the deliverable claim (and outcome-bar claim when literature names one), stopping there; set `target_min=1, target_max=2` | A shrinks from ~8 → ~2 claims/task; signal-to-noise jumps |

**Net of (1)+(2)+(3):** per-task claim count drops 38 → 22, SR-aligned share rises 12 % → 35 %, SR-killer claims become deterministic seeds rather than lucky draws. Should let mat_diff iter-9-class (0.952 with SR=False) flip to either much higher (caught the value-equality problem) or much lower (the deterministic SR-killer fires) — *either way* the verifier↔SR Pearson r > 0.

## Limitations (the agent's honest caveats)

- **N=3 tasks**, binomial 95 % CI on the per-source alignment rates is ±15 pp. Treat the leader/laggard ordering as well-grounded; treat absolute numbers as point estimates only.
- **1 of 3 tasks (bulk_mod) had no SR ground truth** — the SR=True prediction is inference, not measurement. Bulk_mod's contribution to the per-source counts is conditional on that holding.
- **Value-level gold-equality cannot be tested by any prompt edit** — mat_diff SR requires per-cell `pred[col] == gold[col]`, and gold isn't in the workspace. A separate "gold-shadow" evaluator is needed for that class of task; no source-prompt change reaches it.
- **The A/J/O judgements are the agent's, not measured.** A rigorous version: label a few hundred (claim, workflow) pairs and compute Phi-coefficient with SR. Three best-workflow snapshots aren't enough.
- **Recommendations are extrapolations.** Whether re-prompting actually moves verifier↔SR correlation requires re-running the evolution loop with the edits and observing whether the LLM mutator responds to the now-dominant signal.

## Files

- Agent's full report (with per-claim classification tables): output preserved in `/tmp/claude-1000/.../tasks/aaaa44cac7a6774ce.output`
- Cache files surveyed: `sources/workflows/run_1/_verifier_tmp/claim_cache_{edc4def39a21daa0,bcc62483f627f138,cbf1704cc7aae656}_source_{a..e}.json` (15 files)
- Best-workflow evaluations: `sources/workflows/run_1/{20260623_092931_faff3fe1,20260623_152043_bb682237,20260623_192501_970cdd18}/evaluation.txt`
- SR scripts compared against: `datasets/ScienceAgentBench/eval_programs/{clintox_nn_eval,eval_mat_feature_select,eval_bulk_modulus}.py`

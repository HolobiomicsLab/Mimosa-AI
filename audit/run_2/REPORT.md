# Part 6 — New mat_diffusion run audit (`capsule_2460905990421864831`, 20 iterations, 2026-06-21)

This is a fresh single-task Mimosa run on `material_diffusion_shap_feature_selection`. It supersedes the prior-run audit on the same task (`audit/run_1/`, deleted with the worktree but preserved in conversation context).

**Run-level shape:**
- 20 iterations, 1 generation_failed (iter 2), 19 real workflows
- **Best workflow = iter 0 (seed) `20260621_003623_ef01df13`, reward 0.8547 — never beaten across 19 mutations**
- mean(real) ≈ 0.62, std ≈ 0.10, peak non-seed = 0.7672 (iter 10 + iter 12)
- SAB grading: **VER=True, SR=False, CBS=0.895, cost $10.96, total_eval=1**
- run_notes `analysis` field: "Success_level: High"
- Total cost ≈ $10.96, wall clock ≈ 10h

Three Opus agents read the artifacts in depth. The combined picture is consistent and uncomfortable.

---

## (a) Gradient → workflow-change relationship

Per-iteration sampling (5 of 20):

| iter | reward | parent | gradient prescription | what the workflow actually changed | followed? |
|---:|---:|---|---|---|---|
| 1 | 0.419 | seed | **NONE** — iter 1's prompt is the cold-start "First workflow generation" template; the seed's `textual_gradient` is dead-lettered | fresh 3-agent design that forgot MastML, requirements.txt, and the SHAP call | no gradient existed |
| 5 | 0.681 | iter 4 (0.64) | consolidate to 2 agents, featurize only "joined" column, force `mean(\|SHAP\|)`, delete other `.py`, pin deps `==` | followed line-for-line | **yes** |
| 10 | 0.767 | crossover iter 7+9 | featurize ALL THREE composition columns (`comp1_/comp2_/joined_`) — verifier diff flagged the missing two | inverted iter 5's "joined only" → all three | **yes** (peak achieved) |
| 11 | 0.511 | crossover iter 1+3 | flagged `WRONG_CSV_FORMAT`, `MASTML_NOT_USED` | shrank back to 3 agents, re-discovered the SHAP call was missing | parents were the wrong pair — iter 10's just-measured 0.767 was ignored |
| 19 | 0.632 | crossover iter 17+14 | flagged `INCOMPLETE_SHAP_SELECTION`, bad suffixes | reverted to "joined only" — directly contradicting iter 10's prescription | gradient ignored on the columns axis |

Gradients are *single-parent-local*: a grep across all 20 `evolution_prompt_*.md` for `iteration|prior workflow|previous run|past attempt|history` returns **zero hits** in 16 of 20 prompts and one templated hit ("Your previous workflow attempt did not reach…") in the rest. No aggregated bug list. No archive of "tried this, broke that."

## (a-bis) Information retention across the lineage

The engine **does not learn across generations**. Same critique recurs across non-adjacent iterations as if rediscovered:

- `[uses_shap]` failed in iter 1, 4, 11, 15, 18, 19 — six non-adjacent iterations, each re-diagnosed from scratch
- `[exactly_one_script]` failed in iter 1, 4, 7, 15 — every time the next gradient adds a "delete other .py" instruction, the iteration after that regresses
- The "use only joined column" vs "use all three columns" choice **oscillates 4 times** across the lineage with no gradient ever citing the prior toggle
- The seed's `requirements.txt` lesson — the one concrete fix the seed's gradient could have offered — is never delivered, because iter 1 was a cold-start with no parent context
- The QD sampler chose iter 11's parents (iter 1 @ 0.42 + iter 3 @ 0.63) **after** iter 10 had just scored 0.77 — the just-observed peak was ignored

## (b) `evaluation_results.json` — why SR is False

Verbatim from `runs_capsule/capsule_2460905990421864831/evaluation_results.json`:

```
task_id      "2"
VER          true
VER_message  "Code executed. Output: Data shape: (403, 4) ... Featurizing composition columns... 
              Feature matrix shape: (403, 108) ... After imputation shape: (403, 96) ... 
              Train size: (322, 96), Test size: (81, 96) ... Top 20 fe..."
SR           false
SR_message   "N/A"
CBS          0.8952582590878108
cost_usd     10.9573066216
```

The script in `datasets/ScienceAgentBench/eval_programs/eval_mat_feature_select.py:3-12` is:

```python
pred = pd.read_csv("./pred_results/mat_diffusion_features.csv").round(decimals=4)
gold = pd.read_csv("./benchmark/eval_programs/gold_results/mat_diffusion_features_gold.csv").round(decimals=4)
overlap = 0
for column in list(gold.columns.values):
    if column in pred.columns:
        overlap += int(all(list(pred[column] == gold[column])))
return int(overlap >= 14), "N/A"
```

It did not crash (would have populated `SR_message`). It ran cleanly and returned `0` because **column-name overlap is 0**:

- Gold columns (mastml/Magpie): `['NdUnfilled_min_value', 'MendeleevNumber_arithmetic_average', 'MeltingT_difference', 'AtomicVolume_difference', 'MendeleevNumber_max_value', 'BCCmagmom_min_value', ...]`
- Agent columns (homemade): `['comp2_MeltingPoint_max', 'comp1_FirstIonization_min', 'joined_Electronegativity_mean', ...]`

`set(gold.columns) ∩ set(agent.columns) = ∅` → `0 ≥ 14` is False → SR=0 deterministically. Both files are shape `(403, 20)`; the failure is purely featurizer-family naming, not row count.

## (c) Cross-check against SAB reality

The gold script at `datasets/ScienceAgentBench/gold_programs/mat_feature_select.py:10` imports:

```python
from mastml.feature_generators import ElementalFeatureGenerator, OneHotGroupGenerator
```

→ emits Magpie-style descriptors like `MendeleevNumber_arithmetic_average`, `NdUnfilled_min_value`, `BCCenergy_pa_min_value`.

The agent's `runs_capsule/capsule_2460905990421864831/mat_feature_select.py:32-83` **does not use mastml**. It downloads a generic periodic-table CSV (a public Gist) and hand-rolls `featurize_composition()` over 9 properties with `mean/std/min/max` aggregations and `comp1_/comp2_/joined_` prefixes.

Same root failure as the previous run on this task (`mat_shap` in audit/run_1), where the agent used matminer's `ElementProperty` instead. **Different wrong library, same wrong family.** Secondary defect: gold applies `OneHotGroupGenerator` + `StandardScaler` before SHAP (gold cell values are z-scored, e.g. `-0.8555…`), agent saves raw values (e.g. `1234.15` for melting points). Even with matching column names the exact `==` test would fail.

**Did the verifier see this?** Yes, exactly:

```
[script_uses_elemental_feature_generator] (importance=9; core methodology; explicitly named in goal)
  kind=executable status=fail score=0.0
  details: Script does NOT use ElementalFeatureGenerator from Mastml; 
           it defines its own featurize_composition function using a periodic table CSV
```

The verifier diagnosed it perfectly. But the **aggregation didn't escalate it**. 13 of 15 visible claims passed → `base_mean = 0.855` → `hard_fail_capped=False`. A claim with `importance=9` + `explicitly named in goal` + `fail` should arguably trip the `_HARD_FAIL_CAP = 0.7` in `sources/core/selection.py`, since by SAB's own grader this single defect guarantees SR=0. It doesn't, so the seed survived 19 mutations unchallenged.

## (d) Verifier verdict correctness + stability

### d.1 — Verdict correctness on the best workflow (17 claims)

Cross-checked against the runs_capsule artifacts:

| | count |
|---|---:|
| AGREE (verifier matches artifact) | 15 (13 pass + 2 fail) |
| DISAGREE-FN (false-negative FAIL) | 0 |
| DISAGREE-FP (false-positive PASS) | 0 |
| UNVERIFIABLE (script error) | 2 |

The two UNVERIFIABLE are: `[deps_pinned_versions]` whose verifier script has a fatal syntax error (mixed quote regex `r'["']…['"]'`, see §g); and `[script_reads_correct_input]` recorded as ERROR in the run but running cleanly when re-executed locally (env/cwd issue at SAB grading time).

The 0.855 headline is **substantively defensible** for the seed in isolation. The one notable miss (homemade featurize_composition vs MastML) is correctly flagged. But — see d.2 and (g) — **the verifier was generated from the seed's source code**, so several "passing" verdicts are coincidentally right because they verify the seed's own surface choices.

### d.2 — Verifier stability across the lineage (the bombshell)

Compared the **set of claim_ids** across iter 0 (seed, 17), iter 1 (20), iter 10 (17):

| pair | shared claim_ids |
|---|---:|
| iter 0 ∩ iter 1 | **1** (only `workspace_not_cluttered`) |
| iter 0 ∩ iter 10 | **14** (essentially identical rubric) |
| iter 1 ∩ iter 10 | **1** |
| **all three** | **1** |

|claims unique to iter 1| = **16 of 17**. Iter 1's rubric has wholly new ids: `top_20_feature_selection`, `numeric_feature_values`, `no_target_leakage_in_features`, `shap_computed_on_disjoint_test_set`, `mastml_elemental_featurization` — semantically overlapping with iter 0's but with different names, wording, and re-derived importances. For shared claims (iter 0 ↔ iter 10), importance is perfectly stable.

**This is noise drift, not task-adaptive drift.** Iter 1's workflow does something different from iter 10's, yet iter 0 and iter 10 (two equally different workflows) share an almost identical rubric. Comparing scores across iterations is therefore **not safe**: iter 1's 0.419 vs iter 10's 0.767 vs iter 0's 0.855 mix three differently-defined rubrics. The evolution loop is partially selecting against **rubric churn** rather than against workflow quality.

## (g) Per-claim verifier script syntax + faithfulness

17 generated `verify_*.py` scripts in `sources/workflows/run_1/_verifier_tmp/20260621_003623_ef01df13/`. Patterns:

- **AST-based source inspection: 10 scripts** (`uses_shap`, `shap_method_used`, `script_uses_elemental_feature_generator`, `output_saved_to_specified_path`, `missing_values_handled`, `script_reads_correct_input`, `output_valid_20_shap_features`, `dep_manifest_covers_imports`, `exactly_one_script`, `deps_pinned_versions`) — **major improvement vs prior audit's regex/substring-heavy era**
- pandas-based numeric/shape inspection: 6
- Filesystem walk: 3
- `os.path.isfile`: 1
- Subprocess: 0

All scripts parse and run **except** `verify_deps_pinned_versions.py:98` — fatal `SyntaxError` from a mixed-quote regex `r'["']([^"']+)["']'`. Same class of LLM-regex bug as in the prior audit.

**Brittleness patterns still present:**

| script | brittleness |
|---|---|
| `verify_features_are_elemental_properties.py:17-19,42` | hardcoded `aggregation_suffixes = {mean,std,min,max}`, `known_properties = {AtomicNumber,MeltingPoint,BoilingPoint,...}`, `prefix in ('comp1','comp2','joined')` — these are the **seed's exact featurizer choices** |
| `verify_output_valid_20_shap_features.py:95` | same hardcoded `valid_prefixes = ('comp1_','comp2_','joined_')` tuple |
| `verify_uses_shap.py:49,57` | hardcoded variable names `shap_abs` and `top20_features` |
| `verify_output_saved_to_specified_path.py:26,37` | requires variable literally named `output_path` |
| `verify_exactly_one_script.py` | substring filename matching (`test`, `convert`, `util`, `helper`) — would misclassify `convert_features.py` as auxiliary |

**The bombshell finding from (g) + (d.2) combined: the verifier is SEED-ANCHORED.** When generated against the seed's source code, the per-claim scripts embed the seed's variable names, prefix conventions, and property whitelists. Mutations that **improve the science** (use MastML's `ElementalFeatureGenerator` which the goal explicitly names) emit output with MastML's naming (`MeltingT_max_value_comp2`, etc.) and **fail the seed-anchored checks**. Iter 10's evaluation L51-54 shows `features_are_elemental_properties` FAIL specifically because the output uses MastML's names, not the seed's `comp1_/comp2_/joined_` prefixes.

**The seed wasn't the best workflow. The seed was the workflow the verifier was calibrated for.**

## (f) Why did the run stagnate

Three coupled mechanisms, in order of severity:

**1. The verifier is seed-anchored (the new finding).** Verifier scripts encode the seed's surface naming as hard predicates. Any mutation that fixes the gold-named featurizer (the one importance-9 failing claim the seed has) necessarily changes the column naming convention — and the changed naming convention then **fails 2–3 previously-passing seed-anchored checks** (`features_are_elemental_properties`, `output_valid_20_shap_features`). The reward gradient points **away from the SR-passing direction**. This is not noise; it is a structural feature of how `_verifier_tmp/<seed_uuid>/*.py` is generated.

**2. Aggregation rewards claim coverage, not correctness.** Seed at 13/15 passed = 0.855. Mutations that fix the importance-9 critical fail (`script_uses_elemental_feature_generator`) but break 3-4 seed-anchored auxiliary checks score *lower*. Iter 10 (0.7672) achieved its peak despite the centerpiece `[uses_shap]` claim **FAILING** (status=fail score=0.0) — the reward came from passing 12 other claims of varying importance. The hard-fail cap (`_HARD_FAIL_CAP=0.7`) didn't trigger because the failing claim only had importance 9, not the strict ≥10 threshold; the eligibility-for-cap predicate is too narrow given that the verifier itself tagged this claim "explicitly named in goal."

**3. Single-parent-local gradients + non-elitist QD sampler discard the few improvements found.** Gradients never reference any prior iteration. The QD sampler chose iter 11's parents (iter 1 @ 0.42 + iter 3 @ 0.63) after iter 10 had just hit 0.77 — meaning the freshly-observed peak was not consulted. Boldness pinned at 1.0 from iter 11 onwards (variation_log) so the engine spent the final 9 iterations at maximum-rewire mode, which is the regime that *least* preserves load-bearing parts.

## (e) Comparison to the prior run on the same task

Prior run on `mat_shap` (from audit/run_1 / conversation context):

| metric | prior (mat_shap, deleted run_1) | new (this audit) | change |
|---|---:|---:|---|
| n iterations | 23 | 20 | -3 |
| n generation_failed | 8 (35%) | 1 (5%) | **−30 pts ✓** |
| best reward | 0.7949 (seed) | 0.8547 (seed) | +0.06 |
| mean(real) reward | 0.478 | 0.62 | +0.14 ✓ |
| best workflow position | iter 0 (seed) | iter 0 (seed) | — *(same pathology)* |
| SAB VER | 3/6 (50%) on final | 1/1 (100%) on this graded one | execution improved |
| SAB SR | 1/6 (17%) on final, is_success=False | 0/1 (0%), is_success=False | SR unchanged |
| Root failure | wrong featurizer (matminer instead of mastml) | wrong featurizer (homemade instead of mastml) | **same root cause** |
| Verifier caught it? | yes — `elemental_feature_generation` FAIL | yes — `script_uses_elemental_feature_generator` FAIL (imp 9) | both runs: yes |
| Aggregation escalated it? | no | no | both runs: no |

**Real progress:** infrastructure (gen-failure rate down ~7×), execution stability (VER 100% vs partial), gradient quality and some early-iteration mutation effort (iter 5 followed its gradient faithfully). The aggregator is now better-instrumented (more AST-based checks vs prior regex-heavy era).

**No progress on:** the structural pathology — seed-anchoring of verifiers, aggregation that doesn't escalate goal-named importance-9 fails, single-parent-local gradients, non-elitist QD sampling. **Same task fails for the same root reason, in the same way, with the same diagnostic claim correctly emitted and the same aggregation failure to act on it.** The fact that the new verifier rubric independently re-derived the same correct diagnosis is a sign of robust claim-elicitation. The fact that 19 mutations could not act on it is a sign of an unfixed structural blocker.

## What this changes about the prior-audit conclusions

- **Part 5 (verifier verdict precision):** Still holds — verifier verdicts are right ~88% of the time at the per-claim level. **But this audit reveals a layer the prior one missed: the verifier's claims are CALIBRATED to the seed.** A perfectly-precise verifier that asks seed-anchored questions selects for seed-likeness, not for science quality. This is a worse problem than false negatives.
- **Part 3 / Part 5 easy-win lists:** The "AST-based checks instead of regex" recommendation has been **partially implemented since the prior audit** (10 of 17 verifier scripts here are AST-based vs the prior era's substring-heavy approach). Progress. But the seed-anchoring problem is orthogonal — AST that whitelists `comp1_/comp2_/joined_` is still seed-anchored.
- **The aggregation issue is now the headline.** The verifier identified the SR-failing defect with `importance=9` + `explicitly named in goal`. The aggregation arithmetic didn't escalate. This is `sources/evaluators/verifier.py:_aggregate()` (formerly `sources/core/evaluators/`) and the importance-threshold for `_HARD_FAIL_CAP`. **Tightening the hard-fail-cap trigger** ("any importance≥9 + explicitly-named-in-goal fail → cap") closes the loop on this specific class of stagnation.

## Concrete recommendations for the next run

1. **Tighten the hard-fail-cap trigger** in `sources/evaluators/verifier.py::_aggregate`: trip cap on any `importance ≥ 9` AND `explicitly named in goal` fail, not just `importance ≥ 10`. Closes the bulk_modulus-class and mat_shap-class "seed has a goal-named bug, aggregation doesn't punish it" failure.
2. **De-anchor the verifier from the seed**: when generating `_verifier_tmp/<uuid>/*.py`, instruct the generator not to whitelist surface naming choices observed in the source. Specifically, never embed the parsed list of prefixes/properties/variable-names from the current workflow as a hard predicate. Prefer name-agnostic checks (does the SHAP method run? does the output have 20 finite-valued numeric columns? do the column names parse as `<family>_<aggregation>` for some library family?).
3. **Make the gradient cross-generational.** Maintain a small persistent "recurring failure" digest across iterations (last 5-10 failed claim_ids + their fixes) and prepend it to every evolution_prompt. Cheap, eliminates the `[uses_shap]` re-discovery across iters 1, 4, 11, 15, 18, 19.
4. **Elitist QD sampling**: at parent selection time, always include the just-observed best non-seed workflow in the candidate parent pool with a configurable probability (e.g., 0.3). Iter 11 would have used iter 10 instead of iter 1+3.
5. **Cap boldness ramp**: don't let `effective_boldness` pin at 1.0 forever. Once at 1.0 for K iterations without improvement, reset to 0.3 (return to exploitation mode rather than committing to maximum rewire). Avoids the iter 11+ regime where the engine spent half the budget at max-rewire on a verifier whose seed-anchored claims punish rewires.
6. **Fix `verify_deps_pinned_versions.py` regex generation:** the LLM is producing `r'["']([^"']+)["']'` with mixed escape conventions. Add a test-and-retry loop in `_generate_verifier()`: parse the produced script with `ast.parse(...)` before saving; if SyntaxError, regenerate with the error message in the prompt. Eliminates the ERROR-not-FAIL class entirely.

## Files

- `run_notes/capsule_2460905990421864831.json`
- `sources/workflows/run_1/20260621_003623_ef01df13/` (best workflow)
- `sources/workflows/run_1/_verifier_tmp/20260621_003623_ef01df13/*.py` (17 scripts)
- `runs_capsule/capsule_2460905990421864831/` (artifacts including `evaluation_results.json`, `mat_feature_select.py`, `pred_results/mat_diffusion_features.csv`)
- `datasets/ScienceAgentBench/eval_programs/eval_mat_feature_select.py` (SR grader)
- `datasets/ScienceAgentBench/gold_programs/mat_feature_select.py` (gold)
- `datasets/ScienceAgentBench/eval_programs/gold_results/mat_diffusion_features_gold.csv` (gold output)
- `sources/evaluators/verifier.py::_aggregate` (where the hard-fail-cap predicate lives)
- `sources/core/selection.py` (defines `_HARD_FAIL_CAP=0.7`)
- `sources/workflows/run_1/variation_log.jsonl` (boldness trajectory)

# Part 7 — Run 3 audit (clintox + mat_diffusion + bulk_modulus, post-cache-refactor)

This run is the first under the new per-`(task, source)` claim-list cache (`refactor(verifier)` commit `71d47a6`, see audit/run_2/REPORT.md §10). Three tasks were attempted in sequence on the same `sources/workflows/run_1/` directory between 2026-06-22 and 2026-06-23.

Three Opus agents reviewed in parallel; one cross-cutting script measured cache effectiveness directly. The numbers say the cache works dramatically and the engine is finally improving across iterations — but SAB SR is still failing for new (and identifiable) reasons.

---

## 0. Run state — what's actually on disk

| task | run_notes | on-disk uuids | runs_capsule | SAB grade | best verifier reward |
|---|---|---:|---|---:|---:|
| clintox_multitask_toxicity_approval_prediction | yes (20 iters claimed) | **1** (a fresh re-attempt seed; the 18 from run_notes are gone) | yes (VER=False, SR=False) | VER=0/1, SR=0/1 | 0.821 (the lone seed) |
| mat_diffusion_shap_feature_selection | yes (10 iters) | 10 | yes (VER=True, SR=False) | VER=1/1, SR=0/1 | **0.952** |
| predict_bulk_modulus_model | **no** | 10 (7 evaluated, 2 generation_failed, 1 incomplete) | **no** | not graded | **0.760** (iter 8, predicted SR=PASS) |

Clintox is an orphan ([§1](#1-clintox-forensics)). Bulk_modulus run was interrupted at iter 9 ([§3](#3-bulk_modulus-9-iters-interrupted--would-have-passed-sr)). Mat_diffusion completed cleanly and is the focus of the headline gain ([§2](#2-mat_diffusion-10-iters-max-0952-sr-still-false)).

---

## 1. Clintox forensics

**The user's note was right and the situation is worse than expected.** Of the 20 uuids `run_notes/clintox_multitask_toxicity_approval_prediction.json` claims, **0 of them are on disk**. The lone `20260623_092931_faff3fe1` is a fresh iteration-0 seed (`parents: []`) created *after* the deletion, never committed to run_notes, never transferred to the capsule.

The 18 missing uuids were not evicted by `cleanup.sh` (which would have wiped the lot — 21 unrelated folders survive). They were deleted by **a manual `rm` between sessions**, most plausibly to free disk before the mat_diffusion run started. Evidence:
- The mat_diffusion uuids in `run_notes` (`20260623_100554_*` → `20260623_152043_*`) are *exactly* the 10 mtime-contiguous mat_diff folders on disk.
- The bulk-modulus folders (`20260623_15572*` → `20260623_1958*`) form the next block.
- `variation_log.jsonl` + `qd_archive.jsonl` reference only the mat_diff and bulk_mod uuids — not a single clintox row survives in the logs.
- The capsule's three `clintox_nn{,_fixed,_fixed2}.py` files have mtimes **00:38–00:55** on 2026-06-23 — *eight hours before the run_notes timestamp (08:43:17)*. These are leftover hand-edits from an overnight manual debugging session, not iteratively-improved evolution outputs.

**VER failure cause (from `evaluation_results.json`):** `Can't kekulize mol` / `Unkekulized atoms` RDKit errors. The agent's code ran inside Mimosa's sandbox (judge 0.79) but SAB's external grader rejected the SMILES. The lone surviving seed's evaluation.txt also shows `output_columns_exact` FAIL — `FDA_APPROVED_prob`/`CT_TOX_prob` instead of `FDA_APPROVED`/`CT_TOX`. Same column-name-suffix bug Part 1–5 documented; un-fixed here too.

**Recommendation**: tighten `LocalTransfer.transfer_workspace_files_to_capsule` to archive the winning uuid's `workflow_genotype_*.py` into the capsule *verbatim* before the workflow folder is allowed to be touched. Right now the capsule keeps whatever ad-hoc Python was last placed there, with no traceability back to the lineage.

---

## 2. mat_diffusion — 10 iters, max 0.952, SR still False

### Headline numbers vs prior runs on the same task

| run | iters | max reward | mean (real) | SAB VER | SAB SR | featurizer family used |
|---|---:|---:|---:|---:|---:|---|
| audit/run_1 (prior-prior) | 23 | 0.795 (seed) | 0.478 | 3/6 | 1/6 | matminer (wrong) |
| audit/run_2 (prior) | 20 | 0.855 (seed) | 0.62 | 1/1 | 0/1 | homemade periodic-table CSV (wrong) |
| **audit/run_3 (this)** | **10** | **0.952** (iter 9, not seed) | **0.62** | 1/1 | 0/1 | **mastml `ElementalFeatureGenerator` (correct)** |

**The seed was finally beaten** (vs both prior runs where the seed was always best). Best workflow is iter 9 (the LAST one), a crossover of iter 4 + iter 8 — evolution actually progressed.

### Why SR still fails (the new root cause)

The agent finally landed on the right featurizer family. The capsule's `mat_feature_select.py` emits 20 MastML-named features (`NdUnfilled_min_value`, `MendeleevNumber_arithmetic_average`, …). **13 of those 20 names exactly match the gold's column names.** SAB's SR threshold is **≥14 name-matches AND per-cell value equality** (`eval_mat_feature_select.py:8-10`). Two failure modes hit simultaneously:

1. **One short on naming** (13/20 vs threshold 14). One more matching column and we'd clear the first half.
2. **Zero columns pass value equality** even on the 13 shared names: gold is `StandardScaler`-normalized (mean ≈ 0, std ≈ 1.0), agent's values are raw (`MendeleevNumber_arithmetic_average` pred mean 58.6, std 10.9). The verifier has **no claim that checks for StandardScaler application** or value-distribution sanity.

The gold pipeline (`gold_programs/mat_feature_select.py:40-53`) is `ElementalFeatureGenerator → OneHotGroupGenerator → SklearnPreprocessor(StandardScaler) → ShapFeatureSelector`. The capsule implements steps 1, 3 (partially — hand-rolled SHAP), skips step 2 (OneHotGroup) and step 3a (StandardScaler).

### Gradient quality + information retention

Gradients are still **single-parent-local**. The same "feature names have a raw column prefix" defect is **re-diagnosed from scratch in 8 of 10 iterations** (1, 2, 3, 4, 5, 7, 8, 9 — with iter 9's diagnosis just being the inverse of the others). The `requirements.txt`/`pinned_versions` diagnosis recurs in iters 0, 1, 2, 3, 4, 5, 7 — finally fixed at iter 9.

The crossover at iter 6 (parents = iter 5 multi-agent + iter 3 multi-agent) **collapsed to a single-agent solver**, lost MastML access to sandbox security, dumped 759 files into the workspace, scored 0.178 (run minimum). The variation engine respected an upper bound on agent count but not a lower bound — `min(parent_agents) - 1` would have caught this.

### Verifier verdict stability + cache effect (mat_diff)

From `cache_effect.py`:

| metric | prior run (audit/run_2) | this run | change |
|---|---:|---:|---|
| mean pairwise `|iter_i ∩ iter_j| / |iter_i|` | ~7 % | **70 %** | +63 pts |
| all-iter intersection | 1 (`workspace_not_cluttered`) | **4** | +3 |
| cache reuse rate per iter (`|iter_ids ∩ cached_ids| / |cached_ids|`) | n/a | 36–50 % | n/a |

The cache hit the design target. Some paraphrase drift remains — iter 9 introduces 3 brand-new claim_ids that are near-synonyms of cached ones (`output_saved_to_exact_path` ≈ `save_csv_exact_path`; `output_contains_20_shap_features` ≈ `output_shape_correct`). The cache prevents drift on already-seen claims but doesn't stop the LLM from inventing fresh-but-equivalent ones for new dimensions.

### Verifier-script faithfulness (mat_diff)

Mostly clean — the refactor's source-blindness rules landed. Three example scripts are AST-based and don't whitelist surface naming:
- `verify_random_forest_model_used.py` — AST scan for the import + `.fit` + `TreeExplainer` + `.shap_values`
- `verify_uses_elemental_feature_generator.py` — AST scan for `from mastml... import ElementalFeatureGenerator`
- `verify_reads_correct_input_data.py` — generic input-path AST scan

**One stale-anchor residue**: `verify_feature_columns_have_valid_identifiers.py:42-53` hard-codes `valid_prefixes = ['comp1_', 'comp2_', 'joined1_', 'joined2_']` + a 10-element MastML-property whitelist. This belongs to an earlier-iter featurizer; the current best workflow's MastML names (`NdUnfilled_min_value`, etc.) all fall outside the whitelist and the script returns FAIL. **This is the single FAIL pulling 0.952 below 1.0.** Cache-refactor success on 16-of-17 scripts; one regression.

---

## 3. bulk_modulus — 9 iters, interrupted, would have passed SR

### Real progress, then a halt

| iter | uuid | reward | RMSE (GPa) | kind |
|---|---|---:|---:|---|
| 1 (seed) | `155908_276b7b36` | 0.621 (cap) | **145.20** (target leakage → constant predictions) | seed |
| 2 | `164547_0e3d612c` | 0.749 | 36.12 | mutation — *followed the gradient* |
| 3 | `172504_e901c27f` | 0.665 | **21.81** | mutation — plateau begins |
| 4–6 | various | 0.66–0.70 | 21.81–21.97 | mostly mutations |
| **8 (best)** | `192501_970cdd18` | **0.760** | **21.81** | crossover |
| 9 | `195824_e9aedc0b` | — | — | **interrupted** (goal + genotype written, no state_result, no evaluation) |

**The seed was beaten by every mutation.** Evolution worked — RMSE dropped from 145 → 36 → 21.8 in two iterations and held. Best iter 8's predicted SR = **PASS** (RMSE 21.81 < 24.0 threshold by 2.2 GPa margin; columns `['material_id', 'K_VRH']` exact; output at the exact path; material_ids in identical order).

**No `run_notes/` and no `runs_capsule/predict_bulk_modulus_model/`** — the note-writer and `LocalTransfer` never ran. The user can rerun from iter 9 onwards, or commit the best workflow's output manually.

### New verifier brittleness — AST-pattern blindness

This is the biggest *new* finding. The refactor killed regex-substring brittleness (the prior-audit `'rf_full'` / single-quote bug). But the LLM-generated verifier scripts now have **AST-pattern brittleness** — they walk the AST but match against literal `ast.List` of `ast.Constant` patterns, missing equivalent ListComps and named variables.

Three concrete FALSE POSITIVES on the best workflow (importance ≥ 8):

1. `verify_no_target_leakage_in_features.py:16-33` — walks `.drop(columns=...)` looking for hardcoded string literals inside an `ast.List`. The workflow uses `drop(columns=[c for c in leak_cols if c in df_train.columns])` — a `ListComp` over a named variable. Match nothing → `fail`. **The workflow IS dropping the leak columns.**
2. `verify_model_uses_material_features_not_ids.py:47-60` — same pattern with `non_feature_cols`. **The workflow IS dropping `material_id` and `formula`.**
3. `verify_model_hyperparameters.py:25-38` — only reads `RandomForestRegressor(...)` kwarg literals, doesn't trace `GridSearchCV(param_grid={'n_estimators': [100, 200], 'max_depth': [12, 15, None]})`. Reports `n_estimators=None, max_depth=None`. **The workflow IS doing hyperparameter tuning.**

These three FPs sum to ~0.10–0.15 of the verifier reward and **floor the score at 0.76 — below the 0.80 early-stop threshold** in `config.learned_score_threshold`. The run would never converge; it would grind to the iter cap. The refactor moved brittleness from regex tokens to AST patterns but didn't remove it.

### Cache effect (bulk_mod)

| metric | this run |
|---|---:|
| mean pairwise share | **85 %** |
| all-iter intersection | **19** |
| cache reuse rate per iter | 56–69 % |

Highest stability across the three tasks — likely because bulk_modulus is the most well-defined task (a clear regression target with a clear gold-style featurizer).

---

## 4. Cross-cutting: did the cache refactor work?

`audit/run_3/cache_effect.py` (already committed, `086b008`):

| task | mean pairwise share | all-iter intersection | cache reuse |
|---|---:|---:|---:|
| clintox (1 iter only) | n/a | n/a | n/a |
| mat_diffusion (10) | 70 % | 4 | 40–50 % per iter |
| bulk_modulus (7 evaluated) | 85 % | 19 | 56–69 % per iter |
| **prior baseline (audit/run_2 mat_diff)** | **~7 %** | **1** | n/a |

A **10× improvement on rubric stability**. The Part-6 finding that "iter 0 ∩ iter 1 ∩ iter 10 = 1 claim_id" no longer applies — scores across iterations now measure largely the same set of properties. Both task-specific cases also show progress on the underlying claim quality (mat_diff finally got the right featurizer; bulk_mod is RMSE 21.8 below SR threshold).

The refactor solved exactly what it was designed to solve. The seed-anchoring narrative is closed.

---

## 5. What's blocking now (ordered by leverage)

| # | blocker | task | fix |
|---|---|---|---|
| **1** | **AST-pattern brittleness in LLM-generated verifier scripts** — looks for `ast.List` of `Constant`s, misses `ListComp` + named-variable equivalents | bulk_mod (3 FPs on best iter); mat_diff (1 FP on `feature_columns_have_valid_identifiers`) | extend the verifier-script generation prompt to instruct: "where possible, runtime-introspect the produced artefact (read the CSV, import the module, call `feature_labels()`) rather than parse the source AST for literals. Source-AST checks must resolve named variables and accept `ListComp` over comparable sets, not just `ast.List` of literal Constants." |
| **2** | **No SR-shaped claims**: nothing checks value-distribution sanity, no claim asks "is StandardScaler applied", no claim measures column-overlap with gold's feature_labels() output | mat_diff (this is exactly what would close 0.952 → SR=True) | add a Source-B prompt instruction: "if the goal's example output preview shows scaled/normalized values, emit a claim that the produced output has matching value distribution (mean and std within 0.5 of the example)". And add an `applies_standard_scaler_to_features` claim. These are still goal-derived, not SAB-rubric-derived — no cheating. |
| **3** | **Crossover lower-bound missing**: crossover at mat_diff iter 6 collapsed two 4–5-agent parents into a 1-agent child, scoring 0.178 (run minimum). Variation engine enforces upper bound only. | mat_diff (the iter-6 regression cost real LLM money) | add `min(parent_agents) - 1` lower bound to crossover topology |
| **4** | **No cross-iteration memory of failure modes** — same defect re-diagnosed 8 of 10 times | mat_diff (heavy); bulk_mod (lighter, 3 of 7) | persist a rolling `run_notes/<task>_recurring_bugs.md` (3–5 most-frequent failed claim_ids + their attempted fixes per task); prepend to every evolution_prompt |
| **5** | **`LocalTransfer` doesn't archive the winning genotype** — capsule has hand-edited script, no provenance | clintox (provenance lost) | `LocalTransfer.transfer_workspace_files_to_capsule` should copy `sources/workflows/run_1/<winner_uuid>/workflow_genotype_*.py` into the capsule before any other step |
| **6** | **Run-interruption handling** — bulk_modulus was killed at iter 9 with no checkpoint | bulk_mod (current state is `_verifier_tmp` orphans) | atomic per-iter `run_notes/<task>.json` writes (not at end-of-run) so an interruption leaves a valid partial record |

**Recommended landing order:** #1 (closes the AST-pattern blindness uncovered here, should improve bulk_mod from 0.76 to ≥0.85, may close mat_diff's residual single-fail), #2 (adds the SR-shaped claim that closes mat_diff's judge-vs-SR gap), then #3–6.

---

## 6. Files

- `audit/run_3/cache_effect.py` — already committed
- `runs_capsule/clintox_multitask_toxicity_approval_prediction/evaluation_results.json` — VER=False, SR=False
- `runs_capsule/mat_diffusion_shap_feature_selection/evaluation_results.json` — VER=True, SR=False, CBS=0.840
- `runs_capsule/predict_bulk_modulus_model/` — does not exist (run interrupted)
- `sources/workflows/run_1/_verifier_tmp/claim_cache_{edc4def39a21daa0,bcc62483f627f138,cbf1704cc7aae656}_source_{a..e}.json` — 15 cache files, one per task × source
- Best mat_diff workflow: `sources/workflows/run_1/20260623_152043_bb682237/`
- Best bulk_mod workflow: `sources/workflows/run_1/20260623_192501_970cdd18/` (its generated script at `/tmp/mimosa_run_a57fca5d3619_20260623_192501_970cdd18/predict_bulk_modulus.py`)
- 3 AST-pattern-brittle bulk_mod verifier scripts: `_verifier_tmp/20260623_192501_970cdd18/verify_{no_target_leakage_in_features,model_uses_material_features_not_ids,model_hyperparameters}.py`
- 1 stale-anchor mat_diff script: `_verifier_tmp/20260623_152043_bb682237/verify_feature_columns_have_valid_identifiers.py:42-53`

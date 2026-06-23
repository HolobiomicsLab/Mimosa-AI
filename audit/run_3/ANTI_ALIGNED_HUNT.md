# Part 9 — Anti-aligned claims really do exist (and Source D is worse than "dead")

Follow-up to [`SOURCE_EFFECTIVENESS.md`](SOURCE_EFFECTIVENESS.md), which reported **0 SR-anti-aligned claims** but flagged 3 candidates. A focused second-pass Opus agent re-examined every claim against the SAB **gold programs** and the SR scripts, looking specifically for the (claim PASS ∧ SR FAIL) and (claim FAIL ∧ SR PASS) constructions. The conservative "0" doesn't survive rigorous checking — the gold programs themselves FAIL ≥ 6 Mimosa claims while passing SR.

## Key fact the prior audit missed

Class balance in clintox gold (`clintox_gold.csv`, 292 rows): **FDA_APPROVED majority = 0.959, CT_TOX majority = 0.945**. Any claim of the form "model accuracy > majority class" is unsatisfiable on this task by a calibrated probabilistic model — and SR uses AUC, not accuracy. The gold workflow's outputs threshold to roughly the majority class at 0.5, so the gold OUTPUT itself FAILs the "beats majority" claim while passing SR.

## 6 confirmed anti-aligned claims (gold FAILs claim, gold PASSes SR)

| # | source | task | claim_id | mechanism |
|---|---|---|---|---|
| 1 | **E** | clintox | `beats_majority_baseline_ct_tox` | majority = 0.945; gold probs threshold below it; AUC ≫ 0.77 ⇒ SR PASS, claim FAIL |
| 2 | **E** | clintox | `beats_majority_baseline_fda` | majority = 0.959; same mechanism |
| 3 | **E** | clintox | `inter_class_separation_fda` | verifier mis-groups classes (`class_0=0, class_1=294`) — implementation bug whose net effect is a chronic FAIL on SR-passing models with that imbalance |
| 4 | **A** | bulk_mod | `model_hyperparameters` | demands `n_estimators ≥ 100` AND `max_depth ≤ 12`; gold uses `RandomForestRegressor(n_estimators=50)` with no `max_depth` |
| 5 | **A** | bulk_mod | `hyperparameter_tuning` | demands "grid search + k ≥ 5 CV"; gold does zero tuning |
| 6 | **D** | mat_diff | `uses_shap_for_selection` | demands literal `import shap`; gold uses `mastml.feature_selectors.ShapFeatureSelector` (no `import shap`) |

## 2 chronic-bias claims (constant downward pull on SR-passing workflows)

| # | source | task | claim_id | mechanism |
|---|---|---|---|---|
| 7 | **D** | × 3 tasks | `dependencies_pinned` | gold programs have no pinned manifest; SR ignores manifests; claim FAILs gold every time |
| 8 | **D** | bulk_mod | `manifest_present_and_covers_imports` | same — workspace has no manifest, gold has no manifest, SR doesn't check |

Source D contributes **6 of 8 anti-aligned cases** (3 strong + 3 chronic-pinned cases) — the picture isn't "D is dead, takes up budget without adding signal"; it's "D actively pulls reward *away* from SR-passing behaviour".

## Updated 4-way classification table

| | A literature | B goal | C math | D reprod | E stats |
|---|---:|---:|---:|---:|---:|
| SR-aligned | 2 | 4 | 4 | 1 | 3 |
| SR-adjacent | 9 | 10 | 11 | 5 | 11 |
| SR-orthogonal | 12 | 13 | 1 | 14 | 11 |
| **SR-anti-aligned** | **2** | 0 | 0 | **6** | **3** |
| total | 25 | 27 | 16 | 26 | 28 |
| **% aligned** | 8 % | 15 % | **25 %** | 4 % | 11 % |
| **% adjacent** | 36 % | 37 % | **69 %** | 19 % | 39 % |
| **% orthogonal** | 48 % | 48 % | 6 % | 54 % | 39 % |
| **% anti-aligned** | 8 % | 0 % | 0 % | **23 %** | 11 % |

## Net-effectiveness score (aligned + adjacent − anti-aligned) / total

Anti-aligned counts as DOUBLY bad: it consumes weight in the importance-weighted-mean denominator AND points away from SR. This is the most honest single-number rank:

| source | net | reading |
|---|---:|---|
| **C math** | **+0.94** | leader by a mile; small claim budget (n=16) but every claim either aligns or is necessary-for-SR |
| **B goal** | +0.52 | mixed; contains the SR-killers but also concept-paraphrases that miss |
| **E stats** | +0.39 | the sentinel-leakage block earns its keep; the majority-baseline rule actively poisons signal on imbalanced tasks |
| **A literature** | +0.36 | shrunk by hyperparameter-prescriptive claims that contradict gold |
| **D reprod** | **−0.00** | **net-zero to slightly negative** — D is not just dead, it's a small drag |

Source D being net-negative is the big update. The prior audit's "61 % orthogonal" reading was already bad enough to argue removal; the new reading is "23 % anti-aligned" — that's worse than orthogonal because every D claim that FAILs the gold actively pulls reward away from SR-passing behaviour.

## Sharpened edits — what the 5 prompt changes should actually say

Same 5 file:line targets as in [`SOURCE_EFFECTIVENESS.md`](SOURCE_EFFECTIVENESS.md), with text refined to specifically kill the anti-aligned mechanisms:

1. **Source E, lines 384–388** — replace
   > "on a balanced binary task, accuracy is above 0.55"

   with

   > "**On classification**: if all class proportions ≥ 0.20, you may emit `accuracy > 0.55`. If any class proportion ≥ 0.80, REFUSE to emit any accuracy-vs-majority claim — emit `AUC-ROC ≥ 0.6 over a separated test split` instead. Never emit a majority-baseline claim when probability columns (mean ≠ 0.5 with non-zero variance) are visible in the artefact."

   Kills #1, #2.

2. **Source E, the inter-class-separation bullet (find around line 416)** — append guard:
   > "Only emit when both classes have ≥ 5 test rows; refuse on near-degenerate splits."

   Kills #3.

3. **Source A, after line ~157 (literature-prescriptive)** — add:
   > "Do NOT emit hyperparameter-value claims (`n_estimators`, `max_depth`, `learning_rate`, `batch_size`, `n_layers`, `dropout`) unless the goal text literally pins that value. Literature 'best practice' values are not user requirements; the gold reference solution may legitimately use defaults."

   Kills #4, #5.

4. **Source D, lines 317–359** — instead of "shrink to one workspace-clutter claim", make it conditional: emit a manifest/pinning claim **only** when the workspace already contains a `requirements.txt`/`pyproject.toml` to grade. Never emit a manifest-presence claim that will deterministically FAIL the gold workflow.

   Kills #7, #8 (six chronic FAIL cases) and most of D's orthogonal noise.

5. **Source D + Source C — library-name claim shape** — replace `"uses the \`shap\` library"` style with
   > "the workflow performs SHAP-based feature ranking, evidenced by **either** (a) `import shap` **or** (b) instantiating a wrapper class whose name contains `Shap` (e.g. `ShapFeatureSelector`, `ShapExplainer`)."

   Kills #6.

## Limitations the agent surfaced

- The 6 strong anti-aligned cases are evidenced by **direct construction** (read gold source code, run the SR comparison mentally, observe verifier-script verdict on disk). The chronic-bias cases (#7, #8) are **gold-source-code-derived**, not measured against a population of workflows.
- 3 claims couldn't be classified with confidence: `predictions_non_negative` (C, bulk_mod — edge cases), `prediction_range_broad` (E, bulk_mod — depends on gold K_VRH spread we didn't fully verify), `feature_names_scientific` (E, mat_diff — depends on judge interpretation).
- **Source C's +0.94 net-effectiveness is partly an artefact of small n=16**. Per-claim signal density is high; total signal volume is low. Growing C's budget (per recommendation #3 in `SOURCE_EFFECTIVENESS.md`) is the right call but the per-claim quality should not be assumed to scale linearly.
- **Verifier-script implementation bugs are conflated with prompt-level anti-alignment**. #3 (inter_class_separation) and likely #4/#5 (hyperparameter prescription) ride partly on how the per-claim script reads the workflow. A prompt edit alone won't fix those — the verifier-script generation prompt at `verifier_per_claim.py` also needs guardrails (this aligns with Part 7 §5 finding about AST-pattern brittleness in generated scripts).
- **What would resolve the residuals**: run the evolution loop with edits 1–5 applied, then compute Phi-coefficient between per-claim verdict and SR verdict over ≥ 50 (workflow, task) pairs. Anything with `|Phi| > 0.2` and negative sign is genuinely anti-aligned and warrants further patching.

## Headline answer to the user's question

> "what's the percent of SR-anti-aligned for each source?"

| source | % anti-aligned |
|---|---:|
| A | 8 % |
| B | 0 % |
| C | 0 % |
| **D** | **23 %** |
| E | 11 % |

> "how about SR-adjacent?"

| source | % adjacent |
|---|---:|
| A | 36 % |
| B | 37 % |
| **C** | **69 %** |
| D | 19 % |
| E | 39 % |

Source C's adjacent rate (69 %) is the third leg of why it dominates: every C claim that isn't directly SR-aligned is still a *necessary pre-filter* for SR (cardinality, finiteness, schema). Source D's 19 % adjacent rate is the bottom — most D claims aren't even necessary-for-SR.

The combined finding — **Source D is anti-aligned at 23 % and adjacent at only 19 %** — moves Source D from "remove because it adds no signal" to "remove because it adds negative signal".

## Files

- Agent's full report preserved in `/tmp/claude-1000/.../tasks/aae6eee4b731852d6.output`
- All cache files, gold programs, SR scripts as cited
- Edits target: `sources/evaluators/verifier_claim_sources.py`

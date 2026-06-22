# Part 7 — In-progress clintox run audit (post-71d47a6 cache refactor)

Run state (snapshot at audit time):
- Task: `clintox_multitask_deepchem_ecfp` (claim-cache hash `edc4def39a21daa0`)
- 12 workflow folders under `sources/workflows/run_1/`
- **Two sessions** (chain 1 on 2026-06-21 evening, iters 0–6; chain 2 on 2026-06-22, iters 0–4)
- No `runs_capsule/<clintox>/` yet — SAB grading hasn't fired
- Best verifier reward so far: **0.7006** at `20260622_143308_f7d794db` (chain-2 iter 1, tagged `kind="seed"`)
- Five `claim_cache_edc4def39a21daa0_source_{a..e}.json` files in `_verifier_tmp/` — the new continuity mechanism is active

Three Opus agents read the artifacts. Headline split: the **verifier-cache refactor delivered exactly what was promised**, but **the SR-blocking bug persists** for a different root cause — information flow from verifier verdicts back to the variation engine is broken.

---

## (a) Gradient → workflow-change relationship

| iter | uuid | kind | score | gradient prescription | what changed | followed? |
|---:|---|---|---:|---|---|---|
| chain-1 iter 2 | `205714_05a91c3a` | mutation | 0.5455 | tweak agent prompt: exact column names `FDA_APPROVED`, `CT_TOX` (no `_prob`); dedup train↔test SMILES | 5 surgical edits in `instruct_solver`; topology unchanged | **yes** |
| chain-1 iter 5 | `213449_846d2aab` | crossover | **0.6081 (chain-1 peak)** | "do not pick one parent and patch it — genuinely recombine; HARD CAP: agent count = max(parents)" — both parents had 1 agent | jumped to **4 agents** with conditional-edge fan-out | **no — hard cap violated**, but it worked |
| chain-2 iter 1 | `143308_f7d794db` | "seed" | **0.7006 (run best)** | header is literally `## First workflow generation` — **no parent code, no gradient** | 3-agent build (validator + diagnostician); re-introduces the `_prob` suffix in the prompt | n/a — cold start |
| chain-2 iter 2 | `152505_a7f07456` | mutation | 0.5229 | directive: "use columns `FDA_APPROVED`, `CT_TOX` (not `FDA_APPROVED_prob`)" — **textual_gradient.txt for same iter says the OPPOSITE**: `MISSING_PROB_COLUMNS: ... required columns FDA_APPROVED_prob, CT_TOX_prob` | major rewrite of 3 agent prompts; column-fix landed but sklearn fallback crept in | partially — directive obeyed, but "tweak-only" rule broken |

The chain-1 iter 5 violation (1→4 agents despite a hard cap) is interesting: the LLM ignored the rule and was rewarded. The chain-2 iter 2 contradiction inside the verifier's own outputs is more concerning — the textual-gradient summarizer (built by `_build_abstractec_textual_gradient` in `verifier.py`) hallucinated the polarity of the bug.

## (a-bis) Information retention across the lineage

**No prompt references any iteration beyond the immediate parent.** Grep across all 10 evolution prompts for `iteration|prior workflow|past attempt|history` finds zero substantive hits.

Concrete examples of myopia:

1. **`TRAIN_TEST_LEAKAGE` recurs 6 times non-adjacent** — chain-1 iters 0, 1, 2 + chain-2 iters 0, 1, 2 — each gradient flags the same 2-SMILES overlap as if newly discovered.
2. **Column-name `_prob` ping-pong** — chain-1 iter 1's directive removed `_prob`; chain-2 iter 1 (cold seed) re-introduced `_prob` at line 25 of its genotype; chain-2 iter 2's directive removed it again. Same lesson re-learned 4 iters later.
3. **Cross-run ancestry phantom** — chain-2 iter 0's `lineage.parents` points to `20260512_162504_70ccefbf` (May 12, prior run, NOT on disk in `run_1/`). The mutation prompt's `Previous workflow code` block is **empty** (`<python>\n\n</python>`). The mutation engine produced output without ever seeing the parent code — a degenerate mutation that is structurally a cold seed.
4. **Chain-2 iter 1 bypassed the archive entirely** — `qd_archive.jsonl` has all 6 chain-1 entries (including chain-1 peak `846d2aab` at 0.6081); chain-2 iter 1 was generated from the bare workflow_v10 synthesis prompt, not from any archive parent. The multi-agent topology re-appears because it's baked into the synthesis template.
5. **Verifier self-contradiction within one iter** — chain-2 iter 2's `evolution_prompt_…md:192-194` says "do NOT use `_prob`"; its sibling `textual_gradient.txt:1-2` says the opposite. Two channels, opposite directions, no arbitration.

The score trajectory:

```
chain 1:  0.18 → 0.55 → 0.55 → ? → 0.58 → 0.61(peak) → ?
chain 2:                       0.60 → 0.70(best) → 0.52 → 0.70 → ?
```

Two parallel hill-climbs that **don't propagate gains across the session boundary**. Chain 1 ground from 0.18 to 0.61; chain 2 jumped to 0.70 on its first two attempts — not because the QD archive transferred learned mutations but because the workflow_v10 synthesis template has already absorbed the multi-agent topology chain 1 had to discover. The QD archive carries **phenotype descriptors but not source code**, so cross-run continuity is informationally void.

`variation_log.jsonl` shows `iters_since_improvement=0, plateau=0.0, effective_boldness=0.0` for everything except chain-2 iter 4 (`plateau=0.33, effective_boldness=0.26`). The plateau detector fires once, too late to redirect anything.

---

## (b) Best workflow + predicted SR outcome (no clintox capsule yet)

Best: `20260622_143308_f7d794db`, `overall_score = 0.7006`, `hard_fail_capped=False`. The agent's actual clintox script + CSV are preserved at `/tmp/mimosa_run_2b051da359a2_20260622_143308_f7d794db/` (the verifier's read-side mirror, persisted from the sandbox).

**CSV header (verbatim)**: `smiles,FDA_APPROVED_prob,CT_TOX_prob`
**First 3 rows**:
```
Cc1c(cccc1O)C(=O)N[C@@H](CSc2ccccc2)[C@@H](C[NH+]3C[C@H]4CCCC[C@H]4C[C@H]3C(=O)NC(C)(C)C)O,0.99999803,2.0722648e-06
c1ccc(cc1)CN(CC2=[NH+]CCN2)c3ccccc3,0.9998837,0.00014147219
c1ccc(cc1)C(=O)NCC(=O)[O-],0.9999679,3.6463476e-05
```
**Row count**: 292 (matches gold).

**Gold header**: `smiles,FDA_APPROVED,CT_TOX`. **Mismatch on probability columns — `_prob` suffixes.**

**SR prediction: FAIL — deterministic `KeyError`.** SAB's `eval_programs/clintox_nn_eval.py:11` does:

```python
metric = roc_auc_score(gold[['FDA_APPROVED', 'CT_TOX']], pred[['FDA_APPROVED', 'CT_TOX']])
```

The indexer raises `KeyError: "['FDA_APPROVED', 'CT_TOX'] not in index"`. `func_correctness` is never reached. `data_correctness` (smiles order) would actually pass — the SMILES column is correct. So SR=0 deterministically, **identical failure mode to the prior audit's run 1 on clintox.** The agent's `clintox_nn.py` line 25 hard-codes `"FDA_APPROVED_prob"` and `"CT_TOX_prob"` in the output DataFrame builder; the bug is right there in source.

VER would pass (script runs, file produced, valid probabilities). CBS likely moderate (same `MultitaskClassifier`, same ECFP, same `n_features=1024`, same `learning_rate=0.001`, slightly different layer sizes [512, 256] vs gold [1000]).

## (c) Cross-check against SAB ground truth

- **Featurizer**: gold uses `dc.feat.CircularFingerprint(radius=2, size=1024)`; agent uses `CircularFingerprint(radius=2, size=1024, smiles=True)`. **Same family, same hyperparameters.** Verifier `[ecfp_featurization_used]` passes.
- **Model class**: gold `MultitaskClassifier(n_tasks=2, n_features=1024, layer_sizes=[1000], dropouts=[0.25], lr=0.001, batch_size=50)`; agent same class + same `n_tasks`, same `n_features`, same `lr`, similar dropout — only the layer topology differs (`[512, 256]` vs `[1000]`). Verifier `[multitask_classifier_used]` passes.
- **Column-name bug**: same root cause as audit/run_1's clintox finding. Verifier `[output_columns_exact]` (importance 6) FAILS with verdict "Expected `['smiles','FDA_APPROVED','CT_TOX']` but got `['smiles','FDA_APPROVED_prob','CT_TOX_prob']`".
- **The new claim_cache CORRECTLY seeded the exact-columns claim.** From `_verifier_tmp/claim_cache_edc4def39a21daa0_source_b.json`:

  > `[output_columns_exact]` "`pred_results/clintox_test_pred.csv` has columns exactly equal to `['smiles', 'FDA_APPROVED', 'CT_TOX']` in that order (same names, same order, no added suffixes like `_prob`/`_score`, no renames)."

  This claim is being re-emitted and re-failed every iteration. The check WORKS. The fix didn't propagate into the workflow because:
  - importance 6 is too low to trigger the hard-fail cap (threshold ≥ 10) — the score is 0.70, not 0.50
  - in chain-2 iter 2 the textual_gradient builder *hallucinated the polarity* and told the variation engine to ADD `_prob` not remove it (see `textual_gradient.txt:1-2`)
  - subsequent iterations still got it right in the *directive* block but evidently the agent code re-introduced it anyway

---

## (d) Verifier guidance + stability — the headline win

Sampled 4 workflows in chronological order: chain-1 seed (0.18), chain-1 peak (0.61), chain-2 best (0.70), chain-2 iter-3 (0.70).

### d.1 — claim_id continuity (vs prior audit)

| | prior mat_diffusion audit (Part 6) | this clintox audit |
|---|---:|---:|
| |all-shared across N iters| | **1** (N=3, only `workspace_not_cluttered`) | **9** (N=4) |
| Sources A claim_ids in shared set | 0 | 1 (`auc_roc_evaluation`) |
| Sources B claim_ids in shared set | 0 | 2 (`script_saves_to_correct_path`, `two_binary_classification_heads`) |
| Source D | 1 | 3 |
| Importance stability across shared claims | unmeasured / unstable | 4 of 5 audited claims have a **single** importance value |

The seed workflow's 12 claim_ids are a **strict subset** of the best workflow's 21. Drift across all 4 workflows is **1.4 %** (one LLM-invented id: `uses_clintox_dataset` in chain-2 iter 3). 100 % cache compliance on 3 of the 4 sampled workflows.

### d.2 — Per-claim verifier scripts are no longer seed-anchored

Spot-checked 3 scripts in `_verifier_tmp/20260622_143308_f7d794db/`:

- `verify_output_columns_exact.py`: reads CSV with pandas, compares `df.columns` to the goal-literal expected list. No hardcoded variable names. Workflow-source-blind.
- `verify_script_saves_to_correct_path.py`: `ast.parse` of `clintox_nn.py`, walks `ast.Constant` looking for the literal path string `'pred_results/clintox_test_pred.csv'`. AST-based.
- `verify_ecfp_featurization_used.py`: `ast.parse`, walks `ImportFrom`/`Call` for `CircularFingerprint` + any `.featurize()` call. AST-based, no hardcoded variable names.

None reference seed-internal identifiers, prefix sets, or hardcoded SMILES — a clean departure from the mat_diffusion-era regime where `verify_uses_shap.py` hardcoded `target.id == "shap_abs"`.

### d.3 — Verdict quality on the best workflow

`evaluation.txt:1` → `Claims: 21  pass=13  fail=8  error=0  unsure=0`. **Highest-importance fail = 9** (`script_saves_to_correct_path`), **below the hard-fail-cap threshold of 10**, which is why `hard_fail_capped=False` and the score stayed at 0.70.

The 8 failing claims include the two SR-killers:
- `output_columns_exact` (imp 6) — the `_prob` suffix bug
- `script_saves_to_correct_path` (imp 9) — script doesn't contain the literal path string `pred_results/clintox_test_pred.csv` (it's built via `os.path.join` or a variable)

Plus: `auc_roc_evaluation` (imp 4 — script doesn't compute AUC), `deps_manifest_covers_imports` (imp 8 — no manifest), `deps_pinned` (imp 8), `model_beats_trivial_baseline` (imp 7 — CT_TOX model_acc=0.04 vs baseline 1.0; class imbalance), `no_train_test_leakage` (imp 6 — 2 SMILES overlap), `workspace_not_cluttered` (imp 2).

The verifier diagnoses are correct.

### Cache redundancy

The claim_cache union has **41 distinct ids** with semantic overlap. 5 different ids in the cache for "uses CircularFingerprint": `ecfp_featurization` (A), `ecfp_featurization_used` (E), `featurization_uses_ecfp` (D), `script_uses_ecfp` (C), `uses_ecfps_featurization` (B). Each iteration emits ~21 claims — only one of the synonym cluster wins per iteration. Worth deduplicating at seed time so the post-cache LLM doesn't have to arbitrate between five spellings of the same intent.

---

## Synthesis — what worked, what's still broken

### What the 71d47a6 refactor delivered (the wins)

- **Claim-id namespace stable across the QD archive.** Shared-across-4 = 9 vs prior 1-of-3. Importance values stable for 4 of 5 audited shared claims. The "noise drift" pathology from the mat_diffusion audit is closed.
- **Per-claim verifier scripts are no longer seed-anchored.** AST-based, source-blind, no hardcoded variable names. The byte-identical-reuse failure mode that contaminated the mat_diffusion audit can no longer happen.
- **The SR-failing column-name claim WAS seeded correctly** in `claim_cache_..._source_b.json` and is re-emitted + re-failed on every workflow. The verifier's information about the bug is stable and accurate.

### What's still broken (where SR-pass is now blocked)

1. **Information flow verifier→variation_engine is leaky.** Three concrete failures observed:
   - **Importance ceiling.** The SR-killing `output_columns_exact` claim has importance 6. The hard-fail cap fires at importance ≥ 10. Even the `script_saves_to_correct_path` claim at importance 9 doesn't trip it. The reward 0.70 doesn't *signal* that an SR-blocking bug is present — only that "some claims failed." Suggestion: bump literal-deliverable goal claims (paths + schemas the goal text spells out) to importance 10 by default, or change the cap predicate to `importance ≥ 9 AND source == 'b' AND status == 'fail'`.
   - **Textual-gradient builder hallucinates polarity.** Chain-2 iter 2's `textual_gradient.txt` says "MISSING_PROB_COLUMNS: required columns FDA_APPROVED_prob, CT_TOX_prob" — the OPPOSITE direction from the verifier's actual finding. The function at `verifier.py::_build_abstractec_textual_gradient` is summarizing a report that says "remove `_prob`" into a diagnosis that says "add `_prob`." That's an LLM call going wrong. Suggestion: add a polarity-consistency check post-generation that re-reads the diagnosis against the failing-claim descriptions and rejects on contradiction.
   - **Cross-session continuity is empty.** Chain-2 iter 0's lineage points to a parent UUID from a *prior run* (`20260512_162504_70ccefbf`) that isn't on disk in `run_1/`; the mutation prompt's `<python>` block is empty. The QD archive carries phenotype descriptors but not source code, so cross-run ancestry is informationally void. Suggestion: persist the parent's genotype source bytes alongside the archive entry, OR have the variation engine refuse to mutate when the parent code isn't reachable (force a fresh seed instead of a degenerate mutation).

2. **The variation engine doesn't reference > 1 ancestor.** Recurring failures (TRAIN_TEST_LEAKAGE 6 times non-adjacent; `_prob` ping-pong 4-iter cycle) keep being rediscovered. The Part-6 recommendation to maintain a small "recurring failure" digest across iterations and prepend it to every evolution_prompt would close this. Hasn't landed yet.

3. **Cache redundancy lowers the seeding signal.** 5 different cached ids for "uses CircularFingerprint"; ~50 % of cached ids are dropped per iter as synonyms. This isn't broken per se (the LLM correctly picks one winner each iter), but a seed-time dedup would compress the cache and let the per-iter LLM spend less context on arbitration.

4. **Crossover hard cap is being violated.** Chain-1 iter 5 jumped 1→4 agents in a crossover where the cap said max(1, 1) = 1. The cap is advisory, not enforced. It worked out in that one case (chain-1 peak), but it's an open loophole. Variation-engine-side fix.

### Bottom line

**The verifier-side fix landed cleanly and works as designed.** The mat_diffusion-era seed-anchoring local-optimum trap is gone. The verifier now sees a stable claim_id namespace across the QD archive, and the per-claim scripts are workflow-source-blind.

**But SR still doesn't pass on clintox**, for a different reason now: even though the verifier correctly diagnoses the `_prob` suffix bug on every iteration, the **information doesn't propagate back to the variation engine strongly enough** to drive the next mutation to fix it. Three concrete leaks identified above. The work has shifted from "fix the verifier" to "fix the information channel between verifier and variation engine." That's a smaller, more tractable next step.

---

## Files

- `sources/workflows/run_1/_verifier_tmp/claim_cache_edc4def39a21daa0_source_{a..e}.json` — the new continuity cache (seeded by chain-1 iter 0 at 2026-06-21 20:21–20:30)
- `sources/workflows/run_1/20260622_143308_f7d794db/{evaluation.txt, state_result.json, run_metrics.json}` — best workflow
- `sources/workflows/run_1/20260622_152505_a7f07456/{evolution_prompt_…md, textual_gradient.txt}` — the verifier-self-contradiction example
- `sources/workflows/run_1/qd_archive.jsonl` — archive carries phenotype descriptors only, no source bytes
- `sources/evaluators/verifier.py::_build_abstractec_textual_gradient` — the LLM call that hallucinated polarity
- `datasets/ScienceAgentBench/eval_programs/clintox_nn_eval.py:11` — the `KeyError` site
- `/tmp/mimosa_run_2b051da359a2_20260622_143308_f7d794db/clintox_nn.py:25` — where the `_prob` bug is hard-coded in the agent's script

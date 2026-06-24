# Audit prompt v2 — Mimosa evolution on ScienceAgentBench

> Improved from the v1 prompt after 9 iterations of audit (`audit/run_{1,2,3}/`). Use this in a fresh session — the prior audits are committed and the agent can read them as ground truth on prior findings.

## Task

Audit a Mimosa evolution run on **ScienceAgentBench (SAB)** tasks. Produce a single comprehensive, visualisation-heavy report answering: *does evolving multi-agent workflows for science work, why or why not, and what should change?* Read your `engineering-standards` skill before writing any code. Delegate parallelisable parsing and per-task analysis to sub-agents (Opus when the work is non-trivial); reserve the main thread for synthesis and cross-cutting checks.

## Read these in order before doing anything

1. **`audit/run_3/REPORT.md`, `SOURCE_EFFECTIVENESS.md`, `ANTI_ALIGNED_HUNT.md`** — the existing audit synthesis. Treat as confirmed prior findings; you do not need to re-derive them. Use them to calibrate what's new in this run vs what's already documented.
2. **`audit/run_3/cache_effect.py`** — the canonical script for the per-(task, source) claim-id stability metric. Re-run it or extend it; don't rewrite from scratch.
3. **Project docs** on Mimosa architecture, evolution engine, evaluation pipeline. Come away knowing: (a) workflows are evolved per-task via a Quality-Diversity archive with stagnation-driven mutation scope; (b) verifier reward is a self-referential pressure signal, **not** ground truth; (c) SAB grading is a separate system that compares the workflow's output against author-provided gold files; (d) since commit `71d47a6` the verifier per-claim cache has been **removed** — only a **per-(task, source) claim-LIST cache** persists, keyed by `sha256(goal)[:16]`, at `sources/workflows/<run>/_verifier_tmp/claim_cache_<task_key>_source_{a..e}.json`.
4. **ScienceAgentBench**: read `datasets/ScienceAgentBench/README.md` + sample `eval_programs/*.py` + `gold_programs/*.py` to internalise that **SR is a hard threshold on one artefact per task** (numeric or GPT-4V plot judge), grading specific gold-author choices that are usually NOT in the goal text.

## Critical framing — do not violate

- **Verifier reward measures whether the workflow's claims survive Mimosa's *own* verifier. It is not evidence the science is correct.** Every claim about "does evolution work" MUST be reported on three axes: (a) verifier reward (`overall_score` / `max_judge_reward`), (b) independent SAB grade (`ver_success` / `sr_success` / `is_success`), (c) predicted-SR-pass from inspecting the artefact directly (gold-comparison done in your head for numeric tasks; visual inspection for plot tasks). The (a)↔(b) divergence is the headline finding, not a footnote.
- The `analysis` prose field in `run_notes/*.json` is the LLM judge's commentary and is **not independent** of the verifier signal. Do not cite it as corroboration.
- The correct claim-source taxonomy is **A literature / B goal / C math invariants / D reproducibility / E statistical non-triviality**. The old "C narration" / "D math" / "E reproducibility" mapping was wrong (Part 5 correction). Source F is defined in docs but never emitted; ignore it.
- **`runs_capsule/<capsule_name>/` holds the BEST workflow's artefacts, not the final.** If you audit "the final workflow" you'll get the wrong story for ~50 % of tasks (see Part 5 of audit/run_1).

## Data layout (three sources, must be joined)

1. **`sources/workflows/<run>/<uuid>/`** — per-genome artefacts. Each folder has `lineage_<uuid>.json` (parents + `evolution_kind`: `seed | mutation | crossover`), `goal_<uuid>.txt`, `evolution_prompt_<uuid>.md`, `workflow_genotype_<uuid>.py`, `system_prompt_<uuid>.md`, `state_result.json`, `evaluation.txt` (per-claim verdicts), `run_metrics.json`. Also at the root: `qd_archive.jsonl`, `variation_log.jsonl`, `_verifier_tmp/<uuid>/verify_*.py` (per-claim verifier scripts dumped as a runner side-effect — useful for AST-brittleness audit), `_verifier_tmp/claim_cache_<task_key>_source_<label>.json` (5 files per task — the new continuity layer).

2. **`run_notes/<capsule_name>.json`** — task-level rollup. **Where the SAB grades live.** Key fields: `evolved_workflows_uuids` (ordered, may contain `"generation_failed"` sentinels), `evolution_rewards` (index-aligned), `evolution_costs`, `ver_success` / `sr_success` / `total_eval` / `is_success`, `max_judge_reward`, `avg_cbs`. Treat `evolution_avg_reward` with suspicion — it includes zeros from `generation_failed` and is misleading. **Reconcile against on-disk reality**: `rm` between sessions can delete workflow folders without updating `run_notes` (Part 7 §1 — clintox lost 18 of 20 referenced uuids).

3. **`runs_capsule/<capsule_name>/`** — the BEST workflow's artefacts as transferred by `LocalTransfer`. Contains the agent's `.py` script (sometimes multiple versions if there was hand-editing — check mtimes), input data, `pred_results/*`, `evaluation_results.json` (SAB's per-task VER/SR/CBS/cost output). For plot tasks, the agent's saved PNG lives here too — **open it and the gold PNG in `datasets/ScienceAgentBench/eval_programs/gold_results/` side-by-side**.

## Step 0 — Build the index from `run_notes`, then reconcile against disk

For each task:
- `evolved_workflows_uuids` is the authoritative iteration → uuid spine, index-aligned with `evolution_rewards` / `evolution_costs`. Do NOT re-derive iteration order from folder mtimes.
- **Separate `"generation_failed"` / reward = 0.0 sentinels from genuinely low-scoring workflows.** Failed generations are infrastructure noise (orchestrator crash); recompute means over real workflows only.
- **Cross-check on-disk reality**: for each uuid in `run_notes`, verify `sources/workflows/<run>/<uuid>/` exists. Missing folders are evidence of inter-session deletion (Part 7); flag them explicitly.
- **Check for run completion**: if the last evolved uuid has no `state_result.json` or `evaluation.txt`, the run was interrupted (Part 7 §3 — bulk_modulus iter 9 case). Note this; don't grade interrupted iters as failures.
- **Check for capsule provenance**: `diff` the capsule's `.py` script against `workflow_genotype_<best_uuid>.py` for the highest-`max_judge_reward` uuid. If they differ, `LocalTransfer` is dropping ad-hoc files — flag this as a provenance gap.

Emit the index as a table: `task → ordered (iteration, uuid, operator, reward, ev_present, capsule_present) → SAB VER/SR/CBS/is_success`.

## Analysis — answer in this gated order

### TIER 0 — Can we trust the instrument?

- **Per-claim verdict precision**: for the best workflow per task, manually cross-check every claim in `evaluation.txt` against the actual artefact in `runs_capsule/`. Tally AGREE / DISAGREE-FN / DISAGREE-FP / UNVERIFIABLE. Part 5 baseline was 88 % AGREE, 8 % FN (false-negative FAIL from brittle pattern-matching), 0.5 % FP. Has the rate moved?
- **AST-pattern brittleness** in `_verifier_tmp/<best_uuid>/verify_*.py`: for each script, parse with `ast.parse` and detect: (a) `Compare` whose left ends in `.id`/`.attr`/`.name` and right is a string `Constant` (the `target.id == 'foo'` pattern), (b) `Set` or `Tuple` of ≥ 5 string-only Constants (hardcoded whitelist), (c) regex literals with mixed quotes (the syntax-bug class), (d) hardcoded `.R` / `.cpp` extensions when the workflow is Python. Tally per source. Part 7 §3 baseline: ≥ 3 importance-≥ 8 false-positive FAILs per task from this class.
- **Claim-cache stability** (the post-refactor metric): run `audit/run_3/cache_effect.py` on the new data. Mean pairwise share, all-iter intersection, cache reuse rate. Part 7 baseline: 70 % (mat_diff) / 85 % (bulk_mod).
- **Gradient leakage**: does `evolution_prompt_<uuid>.md` contain literal claim_id strings from prior `evaluation.txt`? Sample 8 prompts per task. Part 1 baseline: ~0 leaks.

### TIER 1 — Does evolution improve anything?

- Per-task: seed reward, best reward, Δ(seed → best), mean(real). Compare to the audit-run-3 baselines per task in `audit/run_3/REPORT.md` (mat_diff: 0.708 → 0.952; bulk_mod: seed beaten by every mutation, 145 → 21.8 RMSE).
- Operator attribution from `variation_log.jsonl`: mutation vs crossover Δreward, scope band (`EXPLOITATION`/`ALIGNMENT`/`ADAPTATION`/`EXPLORATION`/`RE-SPECIATION`) × Δreward. Note which bands never fire.
- **Gradient → workflow-change fidelity**: for 5 sampled iters, read the gradient block in `evolution_prompt_*.md` and diff the genotype against parent. Did the workflow follow the gradient? Part 7 §a baseline: when gradients exist they're followed; the failure mode is *no gradient* (cold-start prompts on iter 1) and *single-parent-locality* (gradients never reference any prior iter).

### TIER 2 — Is the search behaving?

- **QD descriptor coverage**: 384-D MiniLM embedding per workflow, PCA-2D trail per task. Confirm tasks form distinct clusters (no collapse) but also don't explore beyond the immediate seed neighbourhood.
- **Stagnation → boldness loop**: from `variation_log.jsonl` track `effective_boldness` vs iteration and `iters_since_improvement`. Does boldness reset, or does it pin at 1.0 forever once plateaued? Part 7 baseline: pinned at 1.0 from iter 11 onwards on mat_diffusion (prior run), the highest-rewire regime — which is the regime that LEAST preserves load-bearing parts.
- **QD elitist bias**: when crossover fires, look at the parent pair's rewards. Does the sampler include the recent best, or does it ignore it? Part 7 baseline: ignored on mat_diff (iter-11 crossover used iter 1 + iter 3 parents after iter 10 hit a peak).
- **Cross-iteration memory**: search all `evolution_prompt_*.md` for references to prior iterations (`grep -E "iteration|prior workflow|previous run|past attempt"`). Baseline: 0–4 hits per 20 prompts.

### TIER 3 — What is the verifier verifying?

- Per task: claim-source mix (A literature / B goal / C math / D reproducibility / E statistical) by count and by importance-weighted total. The audit-run-3 baselines per task are in `audit/run_3/SOURCE_EFFECTIVENESS.md`.
- Hard-fail-cap rate per task (`ev_hard_fail_capped`). Per `_HARD_FAIL_CAP` in `verifier.py` and `_HARD_FAIL_IMPORTANCE`. Does the cap fire when it should? (Part 6 finding: a goal-named importance-9 fail didn't trip the cap because the predicate is `≥ 10`.)
- Executable vs soft claim ratio per task.

### TIER 4 — Per-source SR-effectiveness (new tier — Part 8/9)

For each of the 5 sources, classify every cached claim into 4 buckets against SR ground truth (the gold programs at `datasets/ScienceAgentBench/gold_programs/<task>.py` and the SR scripts at `eval_programs/<task>_eval.py`):

- **SR-aligned**: claim PASS ⟺ SR PASS.
- **SR-adjacent**: claim FAIL ⟹ SR FAIL (necessary, not sufficient).
- **SR-orthogonal**: claim and SR move independently.
- **SR-anti-aligned**: P(SR pass | claim pass) < P(SR pass | claim fail). **Construct two scenarios per candidate**: (a) the workflow that satisfies the claim and fails SR; (b) the workflow that violates the claim and passes SR. The cleanest evidence: **does the SAB gold itself FAIL the claim?** If yes, anti-aligned.

Report per-source totals + net-effectiveness `(aligned + adjacent − anti-aligned) / total`. Part 9 baseline: C math = +0.94 (leader), D reproducibility = −0.00 (net-negative due to 23 % anti-aligned). Has this changed if prompt edits have landed?

### TIER 5 — Operational health

- **Run-state forensics**: were workflow folders deleted between sessions? Are capsule scripts hand-edited (mtimes earlier than the run start)? Did `LocalTransfer` archive the winning genotype verbatim? Was the run interrupted?
- **Cost per quality**: cumulative cost vs verifier reward vs SAB SR outcome. ROI per iteration.

## Required visualisations

Replace the v1 list with these — they map to actual findings:

1. **Three-axis scatter (per task)**: verifier reward × SAB SR pass/fail × predicted-SR-pass-from-artefact (from your manual gold comparison). Colour by task. One point per BEST workflow per task — but additionally one ghost-point per FINAL workflow when best ≠ final. **Headline figure, place first.**
2. **Per-source SR-alignment stacked bar**: rows = sources A-E, segments = aligned / adjacent / orthogonal / anti-aligned. Mark Source D as net-negative if applicable. Place second.
3. **AST-brittleness scan per verifier script**: rows = scripts in `_verifier_tmp/<best_uuid>/`, columns = brittleness type (hardcoded-string-compare, literal whitelist ≥ 5, mixed-quote regex, wrong-language ext). Cells = count. Highlights which scripts will false-fail on cosmetic code changes.
4. **Claim-id stability heatmap** (the cache effect): rows = iterations, columns = unique claim_ids ordered by first-appearance, cells = pass/fail/missing. Picture is now mostly stable bands; pre-refactor baseline was speckled noise. Annotate the all-iter-intersection count and mean pairwise share.
5. **Gradient-vs-mutation diff timeline**: per task, one panel per iteration. For each: did the gradient name a defect that the next workflow's code-diff actually addressed? Bar chart of "gradient followed" / "gradient ignored" / "no gradient (cold start)" / "gradient was misleading".
6. **Information-retention waterfall**: per task, x = iteration, y = failure-code names sorted by first-appearance. Bands show when each code persists. Reveals "rediscovered every iter" vs "fixed once" vs "appears in a single iter".
7. **Crossover parent-quality scatter**: per crossover, x = parent_1 reward, y = parent_2 reward, colour = child reward, marker size = child reward improvement. Detects "QD samples low-quality parents while higher-quality ones exist".
8. **Boldness ratchet**: per task, x = iteration, two lines: `effective_boldness` (from variation_log) and `iters_since_improvement`. Annotate the iteration where boldness first hits 1.0 and whether it resets.
9. **Per-task SR root-cause spectrum**: horizontal bar per task showing the SR-failure mechanism contributors (column-name mismatch / wrong-featurizer / missing-preprocessing-step / threshold-miss / plot-visual-mismatch / placeholder-leakage / interrupted-run). Discrete categories; total height per bar = "magnitude of gap to SR pass" (e.g. number of column-names short, percent over RMSE threshold).
10. **Cost-per-quality**: x = cumulative cost (\$), y = best-so-far verifier reward. Mark SAB grading attempts as dots; colour by SR pass/fail. Reveals whether more cost buys more SR (probably not) vs more verifier reward (yes, slightly).

Drop from v1: the lineage tree (cluttered, low information per pixel), the operator × scope-band Δreward heatmap (boldness ratchet covers it more cleanly), and the QD coverage trail (the 384-D PCA is uninformative — replace with the brittleness scan).

## Deliverable

A single report (Markdown), lead with:

1. The index table — one row per task — with the three-axis verdict (verifier reward / SAB SR / predicted-from-artefact).
2. The reward ↔ SAB correlation verdict with Pearson r (and the disclaimer that r is on N=3-7 tasks).
3. **A one-paragraph answer to: "does evolving multi-agent workflows for science work, and how do we know?"** Reference prior baselines from `audit/run_3/REPORT.md` — say *what's moved* since.

Then the tiered analysis with figures inline.

## Things to flag explicitly (don't bury)

- Every place a conclusion rests on the verifier signal alone without artefact-level corroboration.
- Every anti-aligned claim found (Part 9 catalogue: `beats_majority_baseline_*` on imbalanced binary, `model_hyperparameters` / `hyperparameter_tuning` on bulk_mod when goal doesn't pin them, `dependencies_pinned` and `manifest_present_*` on any task without a manifest in the workspace, library-name claims that demand literal `import X`).
- Every AST-brittleness pattern in verifier scripts (the prior audit identified `ast.List` of `Constant` whitelists, hardcoded variable names, mixed-quote regex literals).
- Every claim-cache file whose seeded list looks wrong (e.g. emits a hyperparameter-prescription claim, an `import shap`-literal claim, a majority-baseline-accuracy claim).
- Generation-failure rate as infrastructure health, separate from search quality.
- Run-state forensics (deleted folders, hand-edited capsules, interrupted runs).

## What NOT to do

- Don't re-run `audit/run_3/cache_effect.py` from scratch — extend it.
- Don't audit "the final workflow" when the capsule was built from the best. Match the agent's interpretation.
- Don't propose `inject the SAB scoring rubric JSON as claims` — that's reading the answer key (Part 4 cheating-concern). The legitimate version is strengthening Source B to extract goal-derived literal identifiers more aggressively.
- Don't write an `audit/run_N+1/REPORT.md` that duplicates prior structure verbatim. Lead with what's NEW. Reference prior parts by `audit/run_X/FOO.md` rather than re-deriving.
- Don't propose more than 5 concrete prompt edits in any single delivery — pick the highest-leverage ones, and tie each to a specific anti-alignment or brittleness pattern in the data.

## Sub-agent dispatch pattern that worked across the prior audits

- **Parsing / index building**: do yourself with Python — one script, ~150 LOC; commit it.
- **Per-task verifier-vs-artefact audit**: one Opus agent per group of 2-4 tasks (`general-purpose` with `model: opus`). Give it the SR script, gold program, gold output, best uuid, the cache files. Ask for the AGREE/FN/FP/UNVERIFIABLE table.
- **Per-source effectiveness audit**: one Opus agent that reads all 15 cache files + 3 SR scripts. Cross-validate with a second focused agent that does the anti-aligned hunt against gold programs.
- **AST-brittleness scan**: one Opus agent that walks `_verifier_tmp/<best_uuid>/*.py` for each task and applies the 4-bucket classification.
- **Visualisation building**: do yourself in one Python script (`audit/run_N/visualize.py`). Don't spawn agents for matplotlib — it's faster to write the code directly.

Save intermediate outputs to `audit/run_N/data/*.csv` and figures to `audit/run_N/figs/*.png` so they're reproducible and committable.

## Self-anchored success criteria

The audit is done when you can answer all of:
1. Where is the verifier reward right but SR wrong (and by what mechanism)?
2. Where is the verifier reward wrong (false-negative or false-positive on the artefact)?
3. Which source is currently the bottleneck on SR alignment, and which concrete prompt edit (file:line + exact text) would close the most anti-aligned claims?
4. Did the engine improve over prior runs on the same task, and what mechanism explains the change (gradient quality / claim stability / aggregation arithmetic / cache effect)?
5. What's the highest-leverage single change (one PR-sized edit) for the next run?

Each answer should cite specific file paths and line numbers.

— end of prompt v2 —

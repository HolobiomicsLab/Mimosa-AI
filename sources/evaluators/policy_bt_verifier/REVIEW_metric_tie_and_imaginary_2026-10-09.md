# Review — metric-tie reporting, fuzzy metric matching, imaginary target gradient (2026-10-09)

Reviewed: `PLAN_metric_tie_and_imaginary_2026-10-09.md` (GLM-5.3 implementation
record) and the code it describes:
- `sources/evaluators/policy_bt_verifier/gradient.py` (`fold_gradient`)
- `sources/evaluators/policy_bt_verifier/policy.py` (`V3_PROMPT`)
- `sources/evaluators/policy_bt_verifier/evaluator.py` (`_build_imaginary_gradient`, `_fuzzy_metric_lookup`, stage 8b)
- `config.py` / `config_default.json` (`policy_bt_imaginary_gradient_rate`)
- `tests/policy_bt_verifier_test.py::TestImaginaryGradient`

Evidence used: the HP-folding run of 2026-10-08 evening (8 generations,
`sources/workflows/20261008_183005_6d3fed0c` → `…193246_4fc65d93`, registry
`sources/workflows/_verifier_tmp/policy_bt_registry_82c9c2036442baba.json`).
All probes below are local replays on the stored digests (no LLM, no reruns).
Numbers I computed are labelled *(my computation)*.

Status of this document: **review + redesign plan. No code was changed.**

---

## 0. TL;DR

1. **The three problems GLM identified are real.** A: the gradient said "NOT ASSESSED" when metrics were in fact compared and tied. B: metric key names drift across generations. C: the search has no pressure to go beyond its first plausible answer. **The three fixes are built on the wrong substrate.**
   - Fix A regex-parses LLM-written English.
   - Fix B adds a second, different fuzzy matcher next to the policy's own.
   - Fix C writes *fabricated measurements* into the channel the mutator is told to trust.
2. **Fix A misses its own motivating case.** On the real tie feedback from the HP runs (`minimum_energy -2 vs -2`), the regex returns **False**, so the gradient still says NOT ASSESSED. On non-metrics (`fresh deliverables 12 vs 12`, `wall_time_s 0.10 vs 0.20`) it returns **True**. It is wrong in both directions *(my computation)*.
3. **Fix B is half missing, and the half that exists picks wrong values.**
   - The `METRIC-KEY MATCHING` prompt rule the plan describes is **not in `policy.py`**.
   - `_fuzzy_metric_lookup` returned a value from a *different quantity* in 6 of the 28 lookups that resolved on the real 10-08 digests. Example: `hh_contact_count` resolved to `47` (the optimum count) in 3 generations *(my computation)*.
4. **Fix C has two hard defects and three design defects.**
   - Hard defect: the target arithmetic moves *lower-is-better* negative metrics the wrong way (`minimum_energy −2` → "push DOWN toward −1.5").
   - Hard defect: on bounded metrics it produces impossible targets (AUC 0.95 → 1.19–1.9).
   - Design: it **replaces** the real gradient, including real decisive losses.
   - Design: it fires at random, not on a plateau.
   - Design: it is invisible to audits, because `imaginary_used` is never persisted and `textual_gradient.txt` holds the fabricated text.
5. **Proposed redesign.** The verifier computes one deterministic **metric ledger** per generation (recon-supplied aliases + a strict resolver). The policy, the gradient fold and any exploration logic all read it. **Exploration moves out of the verifier** into the variation engine as an *honest, labelled* "frontier probe", triggered by the plateau state the stop controller already tracks. The verifier keeps one job: honest measurement.
6. **Plan document accuracy.** It says "COMPLETE — 79 tests passing, ruff clean". The tests do pass (79/79). Ruff fails with 5 errors, and one of the three fixes is absent.

---

## 1. Context — why these fixes exist

- The policy_bt verifier compares each new generation with every predecessor using one LLM-written deterministic `compare(dA, dB)` per task. The reward is the win-rate against predecessors. The textual gradient folds the pairwise feedbacks and steers the next mutation, through the variation engine's directive LLM.
- Since 2026-10-08 the engine stops on a **champion plateau** (`sources/core/stopping.py`). A generation becomes champion only by beating the current champion head-to-head. The run stops after `plateau_patience=3` generations without a new champion, once at least `min_learning_evolve_iterations=6` have run.
- In the 10-08 evening run it worked as designed. Champion `3a1b2128` took over at gen 6 on a real contract difference: `requirement.txt` was present in it and missing in `6d3fed0c`/`cf812205`. The run stopped at gen 9: `stop_reason="plateau: champion 3a1b2128 unchanged for 3 generations"` (`sources/workflows/20261008_193246_4fc65d93/run_metrics.json`).
- **The scientific picture.** Every generation that produced results reported `minimum_energy = −2`, 47 D4-unique optima, 376 raw optima and 16 268 SAWs. I brute-forced the 10-mer: the true minimum is −2, so the task was solved at gen 1 *(my computation, 2026-10-07)*. Because every generation got the same answer, all the pairs tied, the gradients carried little information, and the operator's concern follows: *on a task with headroom, the loop would also lock onto its first plausible answer.*

That concern is legitimate. The disagreement below is about **mechanism**, not intent.

---

## 2. Findings — the current implementation

### 2.1 Fix A — "METRICS COMPARED AND TIED" by regex (`gradient.py:143-186`)

**What it does.** It joins all tie feedback strings and searches them with
`[-]?\d+[.]\d+\s+vs\s+[-]?\d+[.]\d+|[-]?\d{2,}\s+vs\s+[-]?\d{2,}`. On a match it prints
`METRICS COMPARED AND TIED — predecessors report the same values; the search has converged on this metric.`; otherwise it prints the coverage footer.

**Problems.**

| # | Problem | Evidence |
|---|---|---|
| A1 | Misses single-digit integers, which is exactly the HP case | Real feedback from the 10-08 registry, `"…minimum_energy -2 vs -2 (+0.0% delta); num_hh_contacts 2 vs 2…"` → **no match** *(my computation)* |
| A2 | Matches non-metrics | `"fresh deliverables 12 vs 12"` → match; `"wall_time_s 0.10 vs 0.20"` → match (not even a tie value) *(my computation)* |
| A3 | Throws the information away | It prints "converged on this metric" without naming the metric or the value. The mutator learns less than the raw feedback already said. |
| A4 | Wrong layer | The fold reverse-engineers structure from English written by an LLM-authored policy that *had* the numbers. Its format changes every time recon refresh rewrites the policy (gens 3/6/11), so any regex is permanently fragile. |
| A5 | The log contradicts the fold | `evaluator.py:1224-1226` logs `coverage_footer_appended = coverage and (ties or not pair_records)`, the old condition, and now reports `True` when the fold printed the TIED line instead. |
| A6 | Cosmetic | `gradient.py:143-149` has a duplicated, truncated comment block ("All-decisive folds … mean every pair" followed by the start of a second copy). |

### 2.2 Fix B — fuzzy metric matching

**B-prompt (`policy.py`).** The plan's §3 says a `METRIC-KEY MATCHING` block was added after `VALUE-RUNG CONSTRAINT`. **It is not there.** A grep for `METRIC-KEY`, `fuzzy`, `synonym`, `SEARCH all`, `DESCRIPTIVE` finds nothing. Only the `TIE FEEDBACK (binding)` rule landed (`policy.py:154-158`); that rule is good and should stay.

**B-helper (`evaluator.py:376-408`, `_fuzzy_metric_lookup`).**
- It tokenises the recon key (`results.json.optimal_conformation.hh_contact_count` → `{results, json, optimal, conformation, hh, contact}` after stop-word removal), scores every json_metrics key by token overlap, and returns the first maximum.
- The threshold is `overlap ≥ 1`.

Replayed on the 8 real digests against the recon's 4 primary metrics *(my computation)*:

| Generation | `hh_contact_count` | `number_of_optimal_conformations` | `total_self_avoiding_walks_enumerated` |
|---|---|---|---|
| 6d3fed0c | 2 ✓ | 47 ✓ | 16268 ✓ |
| cf812205 | **47 ✗** (picked an optimum count) | 47 ✓ | **94** (first-step-E convention, not total SAWs) |
| 396c629f | 2 ✓ | **2 ✗** (picked `…number_of_h_h_contacts`) | 16268 ✓ |
| 32ed726e | 2 ✓ | 47 ✓ | 16268 ✓ |
| 196a4ef1 | None (empty digest) | None | None |
| 3a1b2128 | 2 ✓ | 47 ✓ | 16268 ✓ |
| bed7e27b | **47 ✗** | 47 ✓ | None |
| 4fc65d93 | **47 ✗** | 47 ✓ | **376 ✗** (raw optimal walks, not total SAWs) |

Root causes:
- **Overlap ≥ 1 is far too permissive.** Two generic tokens (`optimal`, `conformation`) outvote the one token that identifies the quantity (`contact`).
- **Ties are broken by dict iteration order**, so file order (alphabetical: `independent_verify_result.json` before `results.json`) decides. That means verifier copies, stale files or helper outputs can win.
- **No freshness preference** (`deliverables[].produced_this_run` is ignored), **no file hint** (the `results.json.` prefix of the recon key is tokenised into noise), and **no exclusion of verification echoes** (`independent_*`, `reported_*`, `solver_claimed.*`).
- **`log_metrics` are never searched.**
- **It always returns a number** and never says "ambiguous". A wrong value then becomes the "current" value in an imaginary gradient.

**The architectural problem behind B.** There are now **three independent metric-comparison implementations** with different semantics:
1. the LLM-written `compare()`, rewritten at every recon refresh;
2. the fold's regex (A);
3. `_fuzzy_metric_lookup` (B, used by C).

They can disagree: the policy can decide on quantity X while the gradient talks about quantity Y. Nothing reconciles them, and only the first is smoke-tested.

**A stricter lexical resolver is not enough on its own.** I prototyped one: query-token coverage ≥ 66%, verification keys excluded, ambiguity reported instead of guessed. It removed the wrong matches but left most drifted keys **unresolved** (`num_raw_optimal_walks_first_step_plus_x`, `optimal_count_modulo_lattice_symmetry`, `raw_saw_count.count` …) *(my computation)*. Key drift between LLM-written workflows is semantic, not lexical. Something that *sees all digests* — the recon — has to declare the aliases.

### 2.3 Fix C — imaginary target gradient (`evaluator.py:315-373`, stage 8b `:1249-1280`)

**What it does.** With probability 0.25:
1. pick a random recon primary metric;
2. fuzzy-look up its current value;
3. multiply by a factor in [1.25, 2.0] (or `2 − factor` for "lower");
4. **replace** `gradient_text` with `"GEN … — LOST to a stronger hypothetical approach … current X vs achievable ~Y … consider a fundamentally different algorithm"`.

Rewards and verdicts are untouched.

**Hard defects (wrong output on real metric types):**

| # | Case | What it produces *(my computation)* |
|---|---|---|
| C1 | Lower-is-better, negative (HP `minimum_energy −2`) | target = −2 × (2 − f) ∈ **[−1.5, 0]** → "push minimum_energy DOWN toward −1.5", which is *higher* energy. The direction is wrong. |
| C2 | Higher-is-better at 0 (`n_contacts 0`, `accuracy 0`) | target 0 → a "push UP toward 0" gradient |
| C3 | Bounded metric (AUC/F1/accuracy 0.95) | target **1.19–1.9**: impossible |
| C4 | Integer / discrete metric | non-integer targets (`hh_contact_count` 2 → "~3.1") |
| C5 | Invariants / reported facts (degeneracy 47, total SAWs 16 268) | asks for *more* optimal conformations or *more* self-avoiding walks than exist. The recon itself labels these "equality expected, not true optimization direction" (registry `recon.config.primary_metrics[2].notes`). |
| C6 | Wrong current value (from B) | the "current" value can be another quantity entirely (table above) |

**Design defects:**

| # | Problem | Why it matters |
|---|---|---|
| C7 | **Fabricated facts in a trusted channel.** The directive LLM's system prompt says *"The diagnosis is a summary of deterministic ground truth verification … The diagnosis is trusted"* (`sources/core/variation_engine.py:330-331`). | A fake "LOST" with invented numbers is treated as measured. The likeliest response to "achievable ~−3.4" on a task whose true optimum is −2 is a workflow that *reports* −3.4. This is reward hacking pressure built into the loop. The research program designed adversarial tests for exactly this (E22 'FRANK', deceptive gradients; E23 'MALCOLM', Goodhart) and never ran them; this feature ships the attack. |
| C8 | **Replaces, not augments.** `gradient_text = imaginary` (`evaluator.py:1266`). | 25% of the time a real decisive loss, contract violation or crash-class finding disappears from the gradient. |
| C9 | **Unconditional random trigger.** | It fires on gen 1 (0 pairs), on generations that just *won* decisively, and during improving streaks. The plan's own rationale is "when all generations tie", but nothing checks that. |
| C10 | **Unreproducible.** It uses the unseeded global `random` (module import plus a second local `import random as _rnd`). | Two runs of the same campaign get different gradients. Offline replays of the stored policy no longer reproduce what the mutator saw. |
| C11 | **Invisible to audits.** `imaginary_used` is computed and never persisted (ruff F841). | `textual_gradient.txt` and `state_result.evaluation.verifier.abstracted_textual_gradient` hold the synthetic text. The AGENTS.md audit procedure ("Read the gradient … did the mutation act on the diagnosis?") would misattribute a fabricated diagnosis to the verifier. Only `verifier_policy_bt.log` has a `GRADIENT_IMAGINARY` line. |
| C12 | **Logs lie.** | After the swap, the `GRADIENT` stage line still reports refiner `issues/missing/corr` for a gradient that was thrown away. `GRADIENT_REFINED` logs the synthetic text as "final gradient" with `refiner_used: true`. |
| C13 | **Wrong layer.** | Exploration is a *search-policy* decision. The variation engine already owns the search state (`_search_state_block`, `iters_since_improvement`, plateau level — `variation_engine.py:161-255`), and the stop controller owns `since_change`. The verifier should never need to know the search is stuck. |
| C14 | **Bad interaction with the plateau stop.** | With p = 0.25 and patience 3, the chance that *no* exploration push happens in a plateau window is 0.75³ ≈ 0.42 *(my computation)*. Close to half the time the run stops without trying anything bolder. A state-triggered probe can *guarantee* one exploration attempt before the stop. |
| C15 | **The HP task is the wrong place to validate it.** | Its optimum is reached and provable. Exploration there can only waste budget or invite fabrication. Validation needs a task with real headroom (§4.4). |

### 2.4 What GLM got right

- The diagnosis of A/B/C is correct, and C is a valid operator concern.
- The `TIE FEEDBACK (binding)` prompt rule (`policy.py:154-158`): requiring tied values in the feedback is the right instinct.
- Leaving rewards and verdicts untouched by the imaginary gradient.
- Skipping the refiner for synthetic text.
- The config knob is plumbed (`config.py` default + `jsonify` + `from_json`, `config_default.json`).

---

## 3. Smaller bugs (fix regardless of redesign)

| # | Bug | Location | Fix |
|---|---|---|---|
| S1 | Ruff fails (5 errors). The plan claims "ruff clean". | `evaluator.py:27` I001 (`import random` out of order); `evaluator.py:353` SIM108; `evaluator.py:1267` F841 `imaginary_used`; `tests/policy_bt_verifier_test.py:1774` I001; `sources/core/evolution_engine.py:182` F821 `"Config"` undefined (from the stop pass) | Sort imports; persist the flag (or delete the code, §4.3); add `if TYPE_CHECKING: from config import Config` in `evolution_engine.py` |
| S2 | The `METRIC-KEY MATCHING` prompt rule described in the plan is absent | `policy.py:99-194` | Superseded by §4.1 (the policy reads the ledger); do not add the fuzzy rule |
| S3 | Duplicated/garbled comment | `gradient.py:143-149` | Rewrite as one comment |
| S4 | Mis-indented comment under the knob | `config.py:185-188` | Indent |
| S5 | `coverage_footer_appended` log uses the old condition | `evaluator.py:1224-1226` | Have the fold return what it appended (`fold_gradient(...) -> (text, meta)`) and log that |
| S6 | `from_json` uses the literal `0.25` instead of `self.policy_bt_imaginary_gradient_rate`, and does no range validation | `config.py:677-679` | Follow the file's pattern: default from `self`, validate `0 ≤ rate ≤ 1` |
| S7 | Duplicate import: module-level `import random` plus local `import random as _rnd` | `evaluator.py:31`, `:337` | One seeded `random.Random` instance (§4.3) |
| S8 | Plan document status contradicts itself ("3 failing" in §3/§4 vs "COMPLETE" in the header and §5) | `PLAN_metric_tie_and_imaginary_2026-10-09.md` | Mark it superseded by this review |
| S9 | The test monkeypatches the global `random.choice` | `tests/policy_bt_verifier_test.py:1959-1967` | Inject an RNG (§4.3); use pytest `monkeypatch` if anything |
| S10 | No test covers a wrong match; the fuzzy test passes on happy keys only | `TestImaginaryGradient` | §5 adds the real-drift fixture |

---

## 4. Redesign

Principles:
- **One source of truth for metric values.**
- **The verifier measures; the search explores.**
- **Nothing synthetic is ever presented as measured.**
- **Everything is deterministic, persisted and replayable.**

### 4.1 Metric ledger (replaces Fix B, and feeds A and C)

**Recon schema extension** (`recon.py` `RECON_PROMPT`, `ensure_shape`). Each `primary_metrics[]` entry gains:

```json
{
  "id": "min_energy",                       // stable snake_case id, used everywhere
  "key": "minimum energy of the conformation",
  "direction": "lower",                     // higher | lower | equal
  "role": "objective",                      // objective | invariant | report
  "aliases": [                              // EXACT digest keys seen in the slots
    {"file": "results.json", "key": "minimum_energy"},
    {"file": "results.json", "key": "min_energy"}
  ],
  "bounds": null,                           // only if derivable from the goal (e.g. [0,1] for AUC)
  "integer": true,
  "convention": "",                         // e.g. "full directed walks, not first-step-fixed"
  "comparable_contexts": [...], "discounted_contexts": [...], "notes": "..."
}
```

- `role` decides what may ever be pushed. Only `objective` metrics are optimisation targets. `invariant` covers facts every correct answer shares (degeneracy 47, total SAWs 16 268). `report` covers descriptive values.
- `aliases` come from the recon, which already sees every digest (`recon.build_prompt`). Each refresh at gens 3/6/11 *adds* aliases for newly drifted keys. That is the semantic matching lexical code cannot do.

**Deterministic resolver** — new module `sources/evaluators/policy_bt_verifier/metrics.py`:

```python
def resolve_primary_values(digest, primary_metrics) -> dict[str, dict]:
    """{id: {"status": "resolved"|"ambiguous"|"missing",
             "value": float|None, "source": "file:key"|None,
             "fresh": bool|None, "candidates": [...]}}"""
```

Resolution order per metric:
1. Exact alias hits, preferring a file that is `produced_this_run=True`, then a non-verification file (not `independent_*`/`verif*`/`check*`), then the alias order.
2. Strict lexical fallback only when no alias hits: every core query token must be present, verification-echo keys are excluded, and `log_metrics` are searched too.
3. Several candidates with **different values** → `ambiguous`, with the candidates listed. Never guess.
4. Nothing → `missing`.

Where it runs:
- Once per generation at digest time: `digest["primary_values"]` is stored in the registry next to the digest.
- Its results are stored with the `recon_version` they were computed under. When a refresh changes the aliases, predecessors' ledgers are re-resolved. This is deterministic, free and needs no LLM.

**Policy contract** (`V3_PROMPT`):
- R3/R5 value rungs compare `primary_values[id].value` **only when both sides are `resolved`**, using `direction`.
- `invariant` and `report` metrics are never decisive.
- `deciding` must be `primary_values[<id>]`.
- Delete the free-form "search json_metrics" behaviour.
- `POLICY_SHA` bumps automatically.
- Benefit: the LLM-written code shrinks to "read pre-resolved values", which is far easier to smoke-test, and the policy, the fold and exploration now see the *same* numbers.

**Log line** (new): `METRICS resolved=3 ambiguous=1 missing=0 (min_energy=-2 [results.json:minimum_energy] …)`.

### 4.2 Fold from data, not regex (replaces Fix A)

`fold_gradient(uuid, pair_records, coverage_footer, ledger=None)`, where `ledger = {"current": primary_values, "preds": {uuid: primary_values}, "metrics": primary_metrics}`. It adds a deterministic section:

```
METRICS (measured, vs 7 predecessors):
- min_energy (lower is better, objective): -2 here; predecessors -2 ×7 → tied with all
- hh_contacts (higher is better, objective): 2 here; predecessors 2 ×6, missing ×1 → tied
- n_optimal_d4 (invariant): 47 here; predecessors 47 ×6, ambiguous ×1
NOT ASSESSED: total_saws — ambiguous in this generation (candidates:
  results.json:raw_optimal_walk_count.count=376, results.json:raw_saw_count.count=16268)
```

- `NOT ASSESSED` lists only metrics whose status is `missing`/`ambiguous`, *with the reason*. The recon coverage footer stays only for deliverables the ledger cannot cover.
- The regex is deleted.
- The fold returns `(text, meta)` with `meta = {"tied": [...], "not_assessed": [...], "footer_appended": bool}` for the log (fixes S5).
- `refined_text` keeps the `METRICS` block like it keeps the header (`gradient.py:304-329`), so the refiner cannot drop measured values.

### 4.3 Exploration: a "frontier probe" in the variation engine (replaces Fix C)

The intent is kept; the mechanism changes. The verifier no longer fabricates anything.

**Trigger** (deterministic, state-based). Evaluated in the engine after evaluation, when the next prompt is built (`evolution_engine.py:841-873`):
- the champion is unchanged for `≥ explore_after` generations (default `patience − 1`, so **at least one probe is guaranteed before the plateau stop**), **or**
- the last `k` (default 2) gradients were all-tie on every `objective` metric (read from the ledger in `state_result.evaluation.verifier.metric_summary`, a new compact field),
- and optionally a low base rate `explore_base_rate` (default 0) drawn from a `random.Random(seed)` owned by the engine and seeded from config. That makes runs replayable.

**Content.** It is an honest, labelled block passed to the directive LLM *alongside* the real diagnosis, never instead of it:

```
<exploration>
EXPLORATION PROBE — search policy, NOT a measured loss.
State: champion 3a1b2128 unchanged for 2 generations; all 7 pairs tied on
every objective metric (min_energy = -2 everywhere, method: exhaustive
enumeration in 6/7 generations).
The verifier cannot tell whether -2 is optimal. Propose ONE bolder change
that would either improve min_energy or independently confirm the bound
(e.g. a different algorithm, a proof/bound argument, a stronger
independent validation). Do not change reported numbers without a method
that produces them.
</exploration>
```

- **No invented target numbers by default.** A numeric target is allowed only for `role=objective` metrics *and* when `bounds` are goal-derivable, or when the metric's own history shows a trend. In that case use sign-aware arithmetic: `target = current ∓ max(|current|·δ, step)` in the improving direction, clipped to `bounds`, rounded when `integer`. Never multiply a signed value.
- **Directive system prompt** (`variation_engine.py:325-338`) gains one line: *"An `<exploration>` block, when present, is a search-policy request, not a measured fact; the `<diagnosis>` remains the only measured evidence."*
- **Persisted:** `variation_state.exploration = {"triggered": bool, "reason": "...", "metrics": [...]}`, written to `run_metrics.json` and `variation_log.jsonl` through the existing `last_variation_state` path. `textual_gradient.txt` stays 100% verifier output.
- **Removed:** `policy_bt_imaginary_gradient_rate`, `_build_imaginary_gradient`, `_fuzzy_metric_lookup` and stage 8b. New knobs follow the repo pattern (`config.py` → JSON → CLI): `explore_after_plateau: int | None` (None = `patience − 1`) and `explore_base_rate: float = 0.0`. Both apply to *any* verifier, because exploration is now verifier-agnostic.

Why this is better than the 25% coin:

| Property | Imaginary gradient (current) | Frontier probe |
|---|---|---|
| Fires when stuck | ~58% chance per plateau window *(my computation)* | guaranteed before stop |
| Fires when improving | 25% | never (unless base rate set) |
| Real losses preserved | no (replaced) | yes |
| Fabricated numbers | yes, often wrong direction/impossible | none by default; sign/bound-safe when used |
| Auditable | no | yes (persisted flag + reason) |
| Reproducible | no | yes (seeded) |
| Works for hybrid verifier | no (policy_bt only) | yes |

### 4.4 Validate on a task with headroom

The HP 10-mer cannot show whether exploration helps, because gen 1 already finds the optimum. Proposed benchmark:
- the classic **2D HP 20-mer `HPHPPHHPHPPHPHHPPHPH`**, literature minimum E = −9;
- greedy or Monte Carlo first attempts typically land at −7 or −8, and exhaustive enumeration is borderline at 20 residues, so the method choice matters.

Arms, same seeds, 12 generations, plateau stop on:
1. no exploration;
2. the current imaginary gradient (to measure the harm or benefit honestly);
3. the frontier probe.

Outcomes:
- best *recomputed* energy (an offline script recomputes energy from the reported conformation, so self-reports are never trusted);
- generations to reach the best value;
- **fabrication incidents**: reported energy ≠ energy recomputed from the reported coordinates;
- stop reason.

Expected signal: arm 2 shows fabrication incidents, arm 3 does not, and arm 3 ≥ arm 1 on best energy. That is the evidence to keep or drop the feature.

---

## 5. Tests (each defends a contract that a plausible bug would break)

**Metric ledger (`metrics.py`)**
1. *Alias hit beats lexical:* the recon alias `results.json:min_energy` resolves even when a key with more overlapping tokens exists elsewhere.
2. *Freshness preference:* the same alias exists in a fresh and a stale file → the fresh value wins.
3. *Verification echoes are skipped:* `independent_verify_result.json:minimum_energy` is not chosen when `results.json` has the alias.
4. *Ambiguity is reported, never guessed:* two candidates with different values → `status="ambiguous"` and both candidates listed.
5. **Real-drift regression fixture:** freeze the json_metrics of digests `cf812205`, `396c629f` and `4fc65d93` from the 10-08 registry into the test. `hh_contact_count` must **not** resolve to 47, `number_of_optimal_conformations` must **not** resolve to 2, and `total_saws` must **not** resolve to 376. (Each assertion fails on today's `_fuzzy_metric_lookup`.)
6. *Recon refresh re-resolves:* a new alias added in recon v2 resolves a predecessor that was `missing` under v1.

**Fold**
7. **Real tie string regression:** the HP feedback `"minimum_energy -2 vs -2 …; num_hh_contacts 2 vs 2"` plus a ledger with tied values → the `METRICS` section lists `min_energy … tied` and does **not** print NOT ASSESSED for it. (Fails today: A1.)
8. *Non-metric numbers are ignored:* tie feedback `"fresh deliverables 12 vs 12"` with no ledger values → no "tied" claim.
9. *Ambiguous metrics are listed with their reason* under NOT ASSESSED.
10. *Refined text keeps the METRICS block* (like the header today).
11. *Log meta matches the text:* `footer_appended` in the meta equals whether the footer line is present (S5).

**Exploration (variation engine + engine)**
12. *Plateau trigger:* the controller reports `since_change == patience − 1` → the probe triggers; `since_change == 0` → it does not.
13. *Guaranteed before stop:* simulate tie, tie, tie with patience 3 → at least one generation's prompt contains `<exploration>` before `should_stop()` returns True.
14. *Never replaces the diagnosis:* the prompt still contains the verbatim `<diagnosis>` text.
15. *Labelled:* the block contains "NOT a measured loss".
16. *Sign/bounds safety (if a numeric target is enabled):* lower-is-better −2 → target < −2; AUC 0.95 with bounds [0,1] → target ≤ 1.0; an integer metric → integer target; `role=invariant` → no target at all.
17. *Determinism:* same seed + same history → identical trigger sequence.
18. *Persistence:* `run_metrics.json["variation_state"]["exploration"]["triggered"]` reflects the trigger.

**Policy contract**
19. The `V3_PROMPT` text names `primary_values`, and `POLICY_SHA` changes when it does.
20. Smoke: a policy that reads `primary_values` passes on digests where some metrics are `missing`. This is the branch-only-error class from 10-07.

---

## 6. Ordered work plan

1. **Hygiene (S1–S10).** Ruff clean, comments fixed, `coverage_footer_appended` fixed, the plan document marked superseded. *Small; do first.*
2. **Metric ledger (§4.1).** `metrics.py`, recon schema + prompt (`aliases`, `role`, `id`, `bounds`, `integer`), digest attachment, re-resolution on recon refresh, the `METRICS` log line. Tests 1–6.
3. **Policy contract (§4.1).** `V3_PROMPT` reads `primary_values`. Tests 19–20.
4. **Fold from data (§4.2).** Delete the regex; the fold returns `(text, meta)`; `refined_text` keeps the `METRICS` block. Tests 7–11.
5. **Remove the imaginary gradient** (code, knob, tests). Add the `metric_summary` field to the verifier scores.
6. **Frontier probe (§4.3).** Trigger in the engine from `StopController` state plus the metric summary; `<exploration>` block in `mutation_prompt`/`crossover_prompt`; one line in the directive system prompt; knobs; persistence. Tests 12–18.
7. **Validation campaign (§4.4).** HP 20-mer, 3 arms. Report fabrication incidents and best recomputed energy.
8. **Docs.** `PLAN.md` gradient section; `docs/concepts/evolution-engine.md` (exploration + plateau); AGENTS.md note that `textual_gradient.txt` is verifier-only output.

Open questions for the operator:
- Is a numeric target in the probe ever wanted (only for bounded objective metrics), or text-only by default? *Recommended: text-only by default.*
- Should the probe also be enabled for the hybrid verifier? *Recommended: yes; it is verifier-agnostic by design.*

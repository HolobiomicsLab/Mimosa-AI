# Policy-BT Verifier — Reverse-Engineered Design Summary

> Reverse-engineered from the production code (`sources/evaluators/policy_bt_verifier/`,
> all 11 modules) and the offline research program (`experiments_verifiers/`,
> E14/E15/E18/E19/E19c/E24/E25/E26/E44–E48 harness + results of record).
> Written 2026-10-13 so the design can be understood — and rewritten better —
> from scratch. This file replaces the stale `PLAN.md` reference in
> `__init__.py` (that file no longer exists).

## 1. The idea

**Don't let an LLM judge workspaces — let it write, once per task, a small
deterministic Python "policy" that judges any *pair* of workspaces by their
digests; then reward each generation by its win-rate against all its
predecessors in that tournament.**

The LLM contributes *task understanding* (what the goal demands, which metric
names matter); the harness contributes *measurement* (fresh re-execution
facts, deterministic file statistics) and *symmetry* (same policy, same
digest schema, both argument orders). Gold labels never enter anywhere —
this is an in-loop evolution pressure signal, not a grader.

## 2. Why it looks like this — the measured lineage

Every load-bearing decision is the port of a measured result:

| Finding (experiment) | Design consequence |
|---|---|
| **E14 MAXIMUS** — arena + tournament framing | Generations compete pairwise within a task's evolution order, not against a fixed rubric |
| **E15 KATNISS** — LLM preference tournament: 0.643, worst feedback, $26 | No naked LLM-as-judge pairs — too costly, unstable |
| **E18 JOSHUA / E25 CREED** — LLM *writes the judge as code*: 0.480, worst ranker; scripts extracted task-invariant values, invented unstable heuristics; retrial with a decision contract reached 0.633 | Keep code-as-judge, but (a) feed it a **rich per-task digest** (not raw paths), (b) **recon** supplies task-specific metric knowledge, (c) bind it to a decision **ladder + contract** (what may be decisive) |
| **E19 AVA** — same-policy-per-side scoring fixed symmetry | One shared comparator applied to both sides; antisymmetry enforced at runtime |
| **E19c RATCHET** — arrival-moment **win-rate 0.742 = best ranker**; Bradley–Terry fit measured *worse* (0.643) | Reward = raw win-rate vs predecessors; history never rewritten; BT kept only as optional mode |
| **E24 TARS** — 29% of workspaces fail clean re-execution | Re-execution is core evidence: verifier-owned sandbox re-runs every generation; crash/timeout rung decides pairs before any content comparison; `fresh_artifacts` distinguishes stale residue |
| **E26 BUMBLEBEE** — kimi per-workspace visual scoring 0.833 on figure pairs | Dedicated **visual channel** for figure-dominant tasks (digest figures = dims only; a vision LLM compares the actual PNGs) |
| **E16v2 ANDERTON / E33 TRUMAN / E41-CAGE** — gradient instruments, saturation, ~11% label noise; E43: verifier *invented* wrong schema from goal text, never confronted it with a contract | Gradient is a faithful transcript of measured facts (no editorializing); `contract_files` confronts goal-named deliverables; **SchemaError** loop refreshes knowledge instead of silently tying |
| **E44 KUMITE** (2026-10-02) — the prototype: re-exec ledger → digest → recon → policy script → BT replay. Honest negative G1 0.704/0.687 pooled, antisymmetry 1.000; ranking between E19c and E19 at ~1/50 of E15's cost; v1 gradients 0.188 relevance | The whole pipeline skeleton; "measured provenance of every design decision" (the now-missing `PLAN.md`) |
| **E45 SPARTAN** — adversarial probe exposed E44's relevance gain as a *naming artifact* (placebo footers scored 0.62 > real 0.51); v2b assessment instrument; **transfer FAILS without recon** (generic arm −0.135) | Substance-only policy prompt (V3): decisive-rung floor, verbatim-goal-only contract matching (exact token equality), ban on peer-consensus constants, value-determinant rung; **recon is load-bearing, not optional** |
| **E46** — kimi fused into BT: visual decisiveness 0 → 0.75 | Hybrid channel routing (figure-dominance test on the digest) |
| **E47 REFINER** — LLM gradient refinement relevance-neutral (−0.005) | Gradient stays template-assembled from verdict feedbacks; no LLM rewrite step |
| **E48 FIXES** — 5 audited fixes re-run: coverage 19/19, non-visual 0.7482, antisym 1.000 | The reliability rails: real repair loop with error feedback, **runtime smoke on real digest pairs** (a policy crashing on one generation's shape must fail acceptance), token-cap escalation + truncation sentinel, kimi parse retry |

Net position: **best non-visual offline arm (0.748–0.75) near the shipped
hybrid's online anchors (E19c 0.742), at ~$0.02–0.06/task**, with
antisymmetry perfect by construction.

## 3. Architecture (production port)

One evaluation of one generation = a fixed stage pipeline
(`evaluator.py`; stage line
REEXEC→DIGEST→RECON→POLICY→PAIRS→REWARD→GRADIENT→DONE):

```
            ┌──────────────────── verifier-owned sandbox ────────────────────┐
workspace ─▶│ REEXEC (reexec.py): copy → venv → pip install → run run.sh/py,  │
            │ process-group kill on timeout → LEDGER (exit, wall, timeout,    │
            │ error_signature, fresh_artifacts, log_tail, manifest hash)      │
            └──────────────┬─────────────────────────────────────────────────┘
                           ▼
            DIGEST (digest.py): deterministic gold-free projection of the
            sandbox copy + ledger → {exec, contract_files, deliverables,
            tables(+id_order_match), arrays, figures, json_metrics, json_flags,
            inputs, train_used} — sorted, capped, rounded; infra filtered
                           ▼
            RECON (recon.py): ONE LLM call per task (cached in registry):
            goal + digests → TASK CONFIG {primary_metrics (id/aliases/
            direction/role/bounds), tie_policy, coverage_footer}
                           ▼
            METRIC LEDGER (metrics.py): deterministic resolver per generation:
            json_metrics alias hits → LLM text extraction from log_tail →
            lexical fallback → writes primary_values ONTO the digest
                           ▼
            POLICY (policy.py): ONE LLM call per task per recon-version:
            goal + config + digest SCHEMA (never values) → compare(dA,dB)
            source → AST static screen → runtime smoke on REAL digests
            (both orders, self-tie) → repair loop → cached (POLICY_SHA)
                           ▼
            PAIRS (verdicts.py): vs every predecessor — crash rung first
            (ledger facts), else code_pair_verdict: policy runs BOTH orders
            in PATH-only subprocesses (stdin digests, stdout JSON verdict);
            mirrored winner required, else instrument-noise tie; verdict
            cache keyed policy_sha|a|b; per-pair artifact diff logged
                           ▼   (figure-dominant tasks: kimi visual channel
            instead — largest pred PNGs per side, cached)
            SCHEMA-ERROR REFRESH: both-order SchemaError → rebuild recon from
            last-N digests → rewrite policy → re-evaluate flagged pairs
                           ▼
            REWARD (verdicts.py): win-rate vs predecessors (ties ½), arrival-
            moment, history never rewritten; optional BT fit
                           ▼
            GRADIENT (gradient.py): fold of pair feedbacks — measured facts
            with values, metrics-from-ledger first, contract restated in the
            goal's own words → textual_gradient.txt + evaluation artifacts
                           ▼
            REGISTRY (registry.py): per-goal JSON — generations (uuid, digest,
            ledger, reward), verdict/policy/recon caches, session tags
```

### How the policy consumes the digest (the R1–R5 ladder)

The policy is an LLM-written pure function `compare(dA, dB) -> verdict`.
The writer sees goal + recon config + the digest *schema documentation*
(never actual values); the runtime sees the two digests on stdin of a
PATH-only subprocess. The prompt binds it to a decision ladder:

- **R1 crash** — `exec.exit_code/timed_out/error_signature` (decisive).
- **R2 freshness** — `contract_files[].present`, `deliverables[].produced_this_run`
  as *contract* only; raw fresh-count differences are never decisive.
- **R3 value determinants** — `tables[].id_order_match`, distribution stats
  vs `inputs.train_csvs`, degeneracy (`n_unique`).
- **R4 schema** — columns/rows vs the goal contract.
- **R5 metrics** — **only** `digest["primary_values"]` (the pre-resolved
  metric ledger); the policy is forbidden from rummaging `json_metrics`
  itself; flags/shape markers are caveat-notes only.

`compare(d,d)` must tie; every pair runs both orders; disagreement or error
degrades to a labeled tie (instrument noise, never a coin flip).

### The five hard contracts

Violating any of these degenerates the design back to E18:

1. **Gold firewall** — no labels/reference outputs anywhere upstream of the
   benchmark.
2. **Symmetry** — one policy per task, digests computed by the same code,
   both orders must mirror.
3. **Determinism/persistence** — digests are byte-stable; the registry
   stores them so old generations stay comparable; history is append-only.
4. **Measured-facts-only feedback** — every gradient line states values
   with provenance; naming-without-assessing scores zero (E45's lesson);
   contract claims must match goal text by exact token equality.
5. **Knowledge-gap signalling** — unknown digest structure raises
   `SchemaError` → refresh loop, never a silent tie (E43's lesson).

## 4. What to do better if rewriting from scratch

The measured weaknesses and the 2026-10-13 code review point at the same
levers:

- **The digest is the bottleneck, not the comparator.** Picking is
  saturated everywhere (8–11/19 vs oracle 12); ranking headroom lives in
  *what evidence exists*. Fix the codec first: suffix-aware table parsing
  (`.tsv` is currently misparsed with comma sep), goal-filename extraction
  robust to punctuation/spaced names, ONE test-file rule (the token rule
  and the `*test*` glob currently contradict each other), recursion guards
  in JSON flattening, string-leaf float coercion, size caps on `.npy` /
  sha reads. A schema-first definition (declare surfaces, then implement)
  beats the accreted regexes.
- **Text metrics are half-landed**: the LLM extractor sees only the last 20
  log lines (`exec.log_tail`); the digest reads text files then discards
  their content (`metric_lines` is a deprecated stub). Feed the extractor
  curated file content — the machinery exists, the wiring dropped it.
- **Recon aliases drift** across generations; a schema-evolution memory
  (key migration table in the registry) would beat alias re-resolution.
- **Reward**: keep win-rate (E19c), but the E41 noise ceiling (~11%) says
  don't over-invest in ranking; invest in gradient faithfulness and
  between-task budgeting instead.
- **Keep**: both-orders execution, smoke-on-real-digests, arrival-moment
  history, SchemaError loop, PATH-only subprocesses,
  freshness-as-contract-not-count. These are the parts with measured,
  replicated justification.

# Failure-fingerprint descriptor — info flow

> **Reader's note.** This page is an info-flow audit, not a user-facing
> tutorial. It traces every variable that feeds the QD behaviour descriptor
> from its source to the point where novelty distance is computed.
> If you want to *understand* the descriptor, start with
> [`concepts/evolution-engine.md`](../concepts/evolution-engine.md) and
> [`concepts/evaluation-pipeline.md`](../concepts/evaluation-pipeline.md).

## Why a separate descriptor

QD selection compares candidates with two scalars:

- **quality** — `reward_uncapped`, i.e. how well the workflow scored.
- **novelty** — k-NN distance in *behaviour-descriptor space* to the
  rest of the archive.

The descriptor must be **orthogonal to quality**. If it isn't, novelty
becomes a noisy restatement of quality and QD collapses back into greedy
search. The previous structural descriptor
(`[n_agents, n_edges, n_branches, prompt_chars]`) was nominally
orthogonal but barely co-varied with outcomes — different DAG shapes did
not predict different basins. The failure fingerprint replaces it.

## The signal: per-source pass rates

The verifier extracts claims from six independent vantage points:

| Letter | Vantage |
| --- | --- |
| A | Literature / methodology bar |
| B | User goal / deliverable fidelity |
| C | Agent narration truthfulness (recompute-from-disk) |
| D | Math invariants |
| E | Computational reproducibility / CS practice |
| F | Statistical fingerprint |

Each claim is verified to one of `pass`, `fail`, `error`, `unsure`.

**Per source**, the fingerprint takes the pass rate:

```
pass_rate[s] = (# claims with status == "pass" and source == s) / (# claims with source == s)
```

`fail`, `error`, and `unsure` all count as non-passes — a measurement
error is still a non-pass from the optimiser's perspective, and quality
already penalises errors via `quality_norm`.

## Centering: the quality firewall

Per source pass rates **alone** would leak quality into novelty (an
all-pass run would sit at `[1,1,1,1,1,1]`, an all-fail one at
`[0,0,0,0,0,0]`, and their Euclidean distance would be large). To strip
quality out, we subtract the mean pass rate across present sources from
every entry:

```
mean_present = mean(pass_rate[s] for s in SOURCES if presence_mask[s])
vector[s]    = pass_rate[s] - mean_present     if presence_mask[s]
             = 0                               otherwise
```

The descriptor now encodes the *profile shape* of which sources fail
relative to the others — NOT the overall quality level. An all-pass run
and an all-fail run **both** yield the zero profile. This is the
*quality firewall* — and it is asserted in
[`tests/failure_fingerprint_test.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/tests/failure_fingerprint_test.py)
by `test_all_pass_yields_zero_profile` and `test_all_fail_yields_zero_profile`.

A run that fails source A but passes the others looks *very different*
from a run that fails source D but passes the others. Both might be
mediocre on quality, and both deserve a seat in the archive because each
one explores a different basin of failure modes.

## Sources of variables

```
                              ┌────────────────────────────────────┐
   workflow run               │ VerifierEvaluator.evaluate()       │
   ──────────────►            │   per_claim = list of:             │
   uuid, code,                │     {"claim": {"source": "source_X"},
   execution_text             │      "status": "pass" | "fail" | ..}
                              └─────────────┬──────────────────────┘
                                            │
                                            ▼
                              ┌────────────────────────────────────┐
                              │ failure_fingerprint                │
                              │   .compute_failure_fingerprint     │
                              │     -> {vector, presence_mask,     │
                              │         pass_rates}                │
                              └─────────────┬──────────────────────┘
                                            │
                                            │ persisted in scores dict
                                            ▼
                              ┌────────────────────────────────────┐
                              │ state_result.json                  │
                              │   evaluation.verifier              │
                              │     .failure_fingerprint           │
                              │       .vector       (list[float], len 6)
                              │       .presence_mask (list[float], len 6)
                              │       .pass_rates    (list[float], len 6)
                              └─────────────┬──────────────────────┘
                                            │
                                            │ EvolutionEngine reads via
                                            │ WorkflowInfo.state_result
                                            ▼
                              ┌────────────────────────────────────┐
                              │ IndividualRun.state_result         │
                              └─────────────┬──────────────────────┘
                                            │
                                            ▼
                              ┌────────────────────────────────────┐
                              │ SelectionPressure                  │
                              │   ._extract_behaviour_descriptor   │
                              │     → failure_fingerprint_from_    │
                              │        state_result(run.state_result)
                              │     → falls back to                │
                              │        neutral_fingerprint() when  │
                              │        verifier hasn't written one │
                              └─────────────┬──────────────────────┘
                                            │
                                            ▼
                              ┌────────────────────────────────────┐
                              │ k-NN novelty in 6-D space          │
                              │   PopulationMember.behaviour_      │
                              │   descriptor                       │
                              └────────────────────────────────────┘
```

## Variable inventory (audit table)

| Symbol | Type | Set by | Read by | Notes |
| --- | --- | --- | --- | --- |
| `per_claim[i]["claim"]["source"]` | str | `verifier_claims._extract_claims` | `compute_failure_fingerprint` | Normalised to `a`..`f` via `_source_letter`. |
| `per_claim[i]["status"]` | str | `verifier_per_claim._verify_claim` | `compute_failure_fingerprint` | One of `pass`/`fail`/`error`/`unsure`. |
| `pass_rates` | `list[float]` (len 6) | `compute_failure_fingerprint` | Persisted; debugging only. | Neutral 0.5 when source absent. |
| `presence_mask` | `list[float]` (len 6) | `compute_failure_fingerprint` | Persisted; debugging only. | `1.0` iff source emitted ≥1 claim. |
| `vector` | `list[float]` (len 6) | `compute_failure_fingerprint` | `_extract_behaviour_descriptor` → k-NN | The centered descriptor; **the QD signal**. |
| `state_result.evaluation.verifier.failure_fingerprint` | dict | `VerifierEvaluator.evaluate` / `_short_circuit_failed_run` | `failure_fingerprint_from_state_result` | Persisted in `state_result.json`. |
| `IndividualRun.state_result` | dict | `EvolutionEngine._evaluate_and_calculate_cost` | `SelectionPressure._extract_behaviour_descriptor` | Carries the fingerprint into selection. |
| `PopulationMember.behaviour_descriptor` | `list[float]` (len 6) | `SelectionPressure._validate_open_ended` | `_compute_novelty` | Snapshot of the fingerprint at admission time. |
| `qd_score` | float | `SelectionPressure._validate_open_ended` and `_refresh_member_metrics` | parent draw, archive eviction | `(1 − w)·quality_norm + w·novelty_norm`; quality and novelty are *additive*, never multiplied. |

## Failure modes the audit checks

- **No claims at all** (verifier short-circuit): the run gets the zero
  vector and zero presence mask. Distance to peers stays finite; the
  cold run isn't pushed to admit or reject solely on novelty.
- **Source F absent in practice** (the current code wires sources A–E):
  `presence_mask[5]` stays `0.0`; the centering step ignores that axis;
  distance lookups remain well-defined.
- **Unknown source label**: dropped silently by `_source_letter`. No
  out-of-band claim can pollute a bucket.
- **Mismatched descriptor length across archive members**: only happens
  during a schema upgrade; `_euclidean` returns `+inf` as a sentinel and
  `_novelty_range` clamps the normalisation. We recommend draining the
  archive when the dim changes — the descriptor dim is now a stable 6.

## Invariants (asserted)

- `len(vector) == DESCRIPTOR_DIM == 6`.
- All-pass and all-fail yield `[0]*6` (`test_all_pass_yields_zero_profile`, `test_all_fail_yields_zero_profile`).
- Two runs with the same per-source pass rates yield the **same** vector,
  regardless of overall reward — distance is `0`, so the second is treated
  as redundant.
- Two runs with opposite shape (a-pass-b-fail vs a-fail-b-pass) yield
  vectors that mirror each other through the origin
  (`test_distinct_profiles_yield_nonzero_distance`).

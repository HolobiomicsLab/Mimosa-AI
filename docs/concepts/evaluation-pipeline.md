# Evaluation pipeline

Mimosa scores each workflow run with a **three-layer verifier** designed to
be Goodhart-resistant: the mutator can only see a coarse diagnosis, not the
rubric. This prevents the loop from learning to game whatever signal it's
optimized against.

![Evaluation pipeline](../images/evaluation_pipeline.png){ width="100%" }

## The three layers

### Layer 1 — Per-claim verification

The judge first extracts **claims** from the run's `state_result.json`
(numerical results, file existence statements, comparisons, etc.) and
classifies each as executable or soft.

=== "Executable claims"

    An LLM writes a small verifier script. It opens workspace files and
    *recomputes* the asserted value, then compares to the agent's answer.

    **Anti-tautology tripwires:**

    - Literal/output overlap ≥ 80 characters rejects scripts that just
      re-emit the agent's own text.
    - The script must touch real I/O markers, not parse the answer string
      back to itself.

    Each executable claim returns `pass / fail` → `1.0 / 0.0`.

=== "Soft claims"

    Things that can't be recomputed (e.g. "this approach is appropriate
    for X"). An LLM verdict is rendered against workspace previews and
    literature grounding, with three possible outcomes:

    - `pass` → `1.0`
    - `unsure` → `0.5`
    - `fail` → `0.0`

### Layer 2 — Cheat detector

A separate LLM call reads **only the task spec and the workflow source**
(*not* the result). Findings are split:

- **Behavioural** — e.g. "the workflow hard-codes the expected answer".
  Safe to surface in the mutator's diagnosis.
- **Mechanism** — e.g. "this exact pattern was used to bypass the test".
  Audit-only — leaking it would teach the mutator to hide cheats.

The penalty applied to `overall` is `cheat_penalty ∈ [0, 1]`.

### Layer 3 — Abstracted diagnosis

A rubric-blind, plain-language summary of *what went wrong*. This is the
**only** verifier signal the mutator sees. It contains no numerical scores
and no rubric-level detail.

## Aggregation

```python
overall = clamp(base_mean + info_bonus, 0, 1)
if any_hard_claim_refuted:
    overall = min(overall, 0.94)            # _HARD_FAIL_CAP
overall = max(0, overall - cheat_penalty)
```

where

```python
info_bonus(n_hard_pass) = 0.15 * (1 - exp(-n_hard_pass / 8))
```

— a saturating reward for thoroughness. The hard-fail cap means *any*
falsified hard claim limits `overall` to ≤ 0.94, regardless of how many
soft claims pass.

Internally the engine also keeps `reward_uncapped` — the same score without
the 0.94 cap — so QD rank ordering doesn't flatten under hard fails.

## Why three layers?

| Concern | Mitigation |
| ------- | ---------- |
| The judge LLM repeats the agent's claims verbatim. | Per-claim executable verifiers with anti-tautology tripwires. |
| The mutator learns to hide cheats. | Mechanism findings are audit-only, never fed back. |
| The mutator over-fits to a numeric rubric. | Only abstracted diagnosis is returned — no rubric-level signal. |
| The hard-fail cap collapses ranking among failed runs. | `reward_uncapped` keeps QD ordering meaningful. |
| Soft-claim judges hallucinate. | Verdicts are grounded in workspace previews and Perspicacité. |

## Other evaluator backends

The default is `VerifierEvaluator`, but you can swap it via
`WorkflowEvaluator`:

| Backend             | File                                                                                                                                            | Use                                       |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `VerifierEvaluator` | [`verifier.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/evaluators/verifier.py)                                       | **Default** — 3-layer per-claim defense.  |
| `GenericEvaluator`  | [`generic.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/evaluators/generic.py)                                         | Legacy 4-criterion LLM judge.             |
| `ScenarioEvaluator` | [`scenario.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/evaluators/scenario.py)                                       | Rubric / assertion-based scoring.         |
| Perspicacité grounding | [`grounding.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/evaluators/grounding.py)                                  | Adapter used by the verifier.             |
| `BullshitDetector`  | [`bs_detection.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/evaluators/bs_detection.py)                               | Numerical-fraud penalty.                  |

## See also

- [Evolution engine](evolution-engine.md) — how scores drive selection.
- [Scientific grounding](grounding.md) — how Perspicacité is wired in.
- [Developer guide](../DEVELOPER_GUIDE.md) — class layout under `evaluators/`.

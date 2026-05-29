# Evaluation pipeline

Mimosa scores each workflow run with a **multi-source, per-claim verifier**
designed to be Goodhart-resistant: the mutator can only see a coarse
prompt gradient, never the rubric. This prevents the loop from learning to
game whatever signal it's optimized against.

![Evaluation pipeline](../images/evaluation_pipeline.png){ width="100%" }

## The pipeline

The verifier runs four stages per workflow run:

1. **Multi-source claim extraction** — six independent prompts each look at
   the run from a different vantage point and emit success-polarity claims.
2. **Per-claim verification** — for each claim, the judge either writes a
   small Python script that recomputes the asserted value from workspace
   files, or renders a soft LLM verdict when no deterministic check exists.
3. **Aggregation** — per-claim scores combine into `overall_score` with a
   saturating thoroughness bonus and a hard-fail cap.
4. **Abstracted prompt gradient** — a rubric-blind, plain-language summary
   of what went wrong, the **only** signal that reaches the mutator.

## The six claim sources

Each source is a different definition of "success" — claims from each are
merged and verified by the same per-claim machinery downstream.

| Source | Vantage | What it asks |
| ------ | ------- | ------------ |
| **A — Literature** | Peer-reviewed practice (via [Perspicacité](grounding.md)) | Does the science meet the methodology bar the field would demand? |
| **B — User goal** | The literal goal text | Did the agents deliver the deliverable the user spelled out (format, columns, bars, scope)? |
| **C — Agent narration** | The workflow's self-report | Are the agents telling the truth? Can claimed numbers / artefacts be reproduced from disk? |
| **D — Math invariants** | The type of object produced | Do the artefacts satisfy mathematical sanity properties (probabilities in [0,1], symmetry, no NaN, shape consistency, conservation)? |
| **E — Computational reproducibility** | An independent computer scientist re-running this | Are the **non-negotiable** CS practices present (declared deps cover used imports, no absolute paths, seeds on stochastic ops, entrypoint)? |
| **F — Statistical fingerprint** | A skeptical statistician | Is the result *real*, not vacuous? Does it beat a baseline, avoid degenerate predictions, show no leakage signatures? |

A few important constraints on these sources, enforced via the shared
`_CLAIM_RULES_BLOCK`:

- **Positive polarity** — every claim asserts a *success condition*. A
  workflow that produced nothing fails the claim "produced the deliverable",
  not passes "the answer is empty".
- **Discrimination test** — bare file-existence claims are *never* `hard`.
  An artifact claim only counts when chained to a functional property
  (e.g. "manifest covers the imports actually used by produced code").
- **Source E exclusions** — Source E explicitly forbids claims about
  README / docstrings / tests / code style / type hints / logging /
  module layout. It verifies CS *practice*, not engineering aesthetics.

## Per-claim verification

=== "Executable claims"

    An LLM writes a small verifier script. It opens workspace files and
    *recomputes* the asserted value, then emits a single JSON line.
    Scripts run in the agents' workspace with `numpy`, `pandas`, `scipy`,
    and `scikit-learn` pre-installed by the verifier on first use.

    **Anti-tautology tripwires:**

    - Scripts must touch real I/O markers, not parse the agent's answer
      string back to themselves.
    - Embedding the workflow output as a string literal and comparing it
      to itself is rejected.

    Each executable claim returns `pass / fail / error` → `1.0 / 0.0 /
    excluded from the mean`.

=== "Soft claims"

    Things that can't be recomputed (e.g. "this approach is appropriate
    for X"). An LLM verdict is rendered against workspace previews and
    literature grounding, with three outcomes:

    - `pass` → `1.0`
    - `unsure` → `0.5`
    - `fail` → `0.0`

## Aggregation

```python
base_mean = mean(score for each non-error claim)
bonus     = α · (1 − exp(−n_hard_pass / β))      # α=0.05, β=8
pre_cap   = clamp(base_mean + bonus, 0, 1)

overall_score = min(pre_cap, hard_fail_cap) if any_hard_claim_refuted else pre_cap
```

- `α = _INFO_BONUS_ALPHA = 0.05`, `β = _INFO_BONUS_BETA = 8.0` —
  saturating reward for thoroughness, conditioned on *passing hard*
  claims so trivial or failed claims contribute nothing.
- `hard_fail_cap = _HARD_FAIL_CAP = 0.99` — currently set permissively
  to keep the evolutionary signal smooth; a refuted hard claim still
  flags `hard_fail_capped = True`.
- The engine separately keeps `overall_score_uncapped` (pre-cap) so QD
  rank ordering doesn't flatten under hard fails.

## Abstracted prompt gradient

After aggregation the verifier composes a single-sentence diagnosis (the
**prompt gradient**) from the per-claim report and recent history of
similar runs. It is deliberately *rubric-blind*:

- It does not name specific claims or scores.
- It encodes failure modes by short code names (e.g. `FALLBACK_ECFP_CLASSIFIER`)
  so recurring patterns can be tracked across generations.
- It is the **only** verifier output that reaches the mutator.

## Cheat detector (currently disabled)

A standalone cheat detector pass over the agents' produced source code
existed in earlier versions and is kept in the codebase but is currently
**disabled** (`cheat = None`) pending a rewrite. The aggregation pipeline
still supports a `cheat_penalty` field for when it is re-enabled. The
behavioral anti-cheat pressure today comes from:

- Source C's recompute-from-disk verifiers.
- The "Used fallback" claim type, whose score is *inverted* — a passing
  fallback check counts as 0, a failing one as 1.
- The anti-tautology tripwires in the per-claim verifier generator.

## Why this design?

| Concern | Mitigation |
| ------- | ---------- |
| The judge LLM repeats the agent's claims verbatim. | Per-claim executable verifiers with anti-tautology tripwires. |
| The mutator over-fits to a numeric rubric. | Only the abstracted prompt gradient is returned — no rubric-level signal. |
| The hard-fail cap collapses ranking among failed runs. | `overall_score_uncapped` keeps QD ordering meaningful. |
| One vantage point misses the failure. | Six independent sources, claims merged. |
| Cosmetic hygiene gets gamed as "quality". | Source E only verifies non-negotiable CS practice; docs/tests/style are forbidden. |
| Artifact existence checks reward "moved files around". | `_CLAIM_RULES_BLOCK` forbids bare-existence as `hard`; max 2 soft artifact claims. |
| Soft-claim judges hallucinate. | Verdicts are grounded in workspace previews and Perspicacité. |

## Other evaluator backends

The default is `VerifierEvaluator`, but you can swap it via
`WorkflowEvaluator`:

| Backend             | File                                                                                                                                            | Use                                       |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `VerifierEvaluator` | [`verifier.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/evaluators/verifier.py)                                       | **Default** — multi-source per-claim.     |
| `GenericEvaluator`  | [`generic.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/evaluators/generic.py)                                         | Legacy 4-criterion LLM judge.             |
| `ScenarioEvaluator` | [`scenario.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/evaluators/scenario.py)                                       | Rubric / assertion-based scoring.         |
| Perspicacité grounding | [`grounding.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/evaluators/grounding.py)                                  | Adapter used by Source A.                 |
| `BullshitDetector`  | [`bs_detection.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/evaluators/bs_detection.py)                               | Numerical-fraud penalty.                  |

## See also

- [Evolution engine](evolution-engine.md) — how scores drive selection.
- [Scientific grounding](grounding.md) — how Perspicacité is wired in.
- [Developer guide](../DEVELOPER_GUIDE.md) — class layout under `evaluators/`.

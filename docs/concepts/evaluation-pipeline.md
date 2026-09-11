# Evaluation pipeline

> This page describes the **judge** that runs after every workflow
> execution — the [`VerifierEvaluator`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/evaluators/verifier.py)
> in `sources/core/evaluators/`. It is the **pressure signal that drives
> workflow evolution**, not a benchmark grader.
>
> If you are looking for ScienceAgentBench or PaperBench grading — those
> are different systems that compare the workflow's output against
> author-provided ground-truth files. See
> [ScienceAgentBench evaluation](../science_agent_bench_evaluation.md)
> and [PaperBench evaluation](../papers_bench_evaluation.md).

Mimosa scores each workflow run with a **multi-source, per-claim
verifier**. The verifier writes and executes **deterministic Python
programs** in the workspace to confirm what the agents claim they did,
across five independent vantage points (literature, user goal, agent
narration, math invariants, statistical
fingerprint). The single signal that flows back to the mutator is a short
**prompt gradient** that summarizes failure modes without leaking the
verified claims themselves.

The same per-claim verdicts are also projected into a 6-dim **failure
fingerprint** — a centered vector of per-source pass rates that records
*how* a run failed. It is persisted as a diagnostic but is **not** the QD
behaviour descriptor: novelty is measured in genotype-embedding space
(see [Selection](evolution-engine.md#behaviour-descriptor-genotype-embedding)).

![Evaluation pipeline](../images/evaluation_pipeline.png){ width="100%" }

## The pipeline

The verifier runs four stages per workflow run:

1. **Multi-source claim extraction** — five independent prompts each look at
   the run from a different vantage point and emit success-polarity claims.
2. **Per-claim verification** — for each claim, the judge either writes a
   small Python program that recomputes the asserted value from workspace
   files (the default path), or renders a soft LLM verdict when no
   deterministic check is possible.
3. **Aggregation** — per-claim scores combine into `overall_score` as an
   importance-weighted mean with a hard-fail cap.
4. **Prompt gradient** — a plain-language summary of failure modes, the
   **only** signal that reaches the mutator. It does not name claims,
   scores, or sources, so the mutator cannot turn the verified-claim
   vocabulary back into a rubric to optimize against.

## The five claim sources

Each source is a different definition of "success" — claims from each are
merged and verified by the same per-claim machinery downstream.

| Source | Vantage | What it asks |
| ------ | ------- | ------------ |
| **A — Literature** | Peer-reviewed practice (via [Perspicacité](grounding.md)) | Does the science meet the methodology bar the field would demand? |
| **B — User goal** | The literal goal text | Did the agents deliver the deliverable the user spelled out (format, columns, bars, scope)? |
| **C — Agent narration** | The workflow's self-report | Are the agents telling the truth? Can claimed numbers / artefacts be reproduced from disk? |
| **D — Math invariants** | The type of object produced | Do the artefacts satisfy mathematical sanity properties (probabilities in [0,1], symmetry, no NaN, shape consistency, conservation)? |
| **F — Statistical fingerprint** | A skeptical statistician | Is the result *real*, not vacuous? Does it beat a baseline, avoid degenerate predictions, show no leakage signatures? |

A few important constraints on these sources, enforced via the shared
`_CLAIM_RULES_BLOCK`:

- **Positive polarity** — every claim asserts a *success condition*. A
  workflow that produced nothing fails the claim "produced the deliverable",
  not passes "the answer is empty".
- **Discrimination test** — bare file-existence claims are *never* `hard`.
  An artifact claim only counts when chained to a functional property
  (e.g. "predictions.csv contains a valid probability for every row").

## Per-claim verification

The verifier prefers a deterministic Python program for every claim and
only falls back to an LLM verdict when no executable check is possible.

=== "Executable claims (preferred)"

    An LLM writes a small Python program *per claim*. The program opens
    workspace files and **recomputes the asserted value** from disk,
    then emits a single JSON line. Programs run in the agents'
    workspace with `numpy`, `pandas`, `scipy`, and `scikit-learn`
    pre-installed by the verifier on first use.

    Because the check is deterministic Python touching the same files
    the agents produced, the verdict does not depend on the judge LLM
    re-believing the agent's narration.

    **Anti-tautology tripwires:**

    - Programs must touch real I/O markers, not parse the agent's answer
      string back to themselves.
    - Embedding the workflow output as a string literal and comparing it
      to itself is rejected.

    Each executable claim returns `pass / fail / error` → `1.0 / 0.0 /
    excluded from the mean`.

=== "Soft claims (fallback)"

    Things that can't be recomputed from disk (e.g. "this approach is
    appropriate for X"). An LLM verdict is rendered against workspace
    previews and literature grounding, with three outcomes:

    - `pass` → `1.0`
    - `unsure` → `0.5`
    - `fail` → `0.0`

## Aggregation

```python
base_mean = importance_weighted_mean(score for each non-error claim)
pre_cap   = clamp(base_mean, 0, 1)

overall_score = min(pre_cap, hard_fail_cap) if any_hard_claim_refuted else pre_cap
```

- Per-claim weights come from the rater-assigned importance (1–10), so an
  importance-10 deliverable claim moves the score ~5× more than a
  low-importance hygiene claim.
- `hard_fail_cap = _HARD_FAIL_CAP = 0.7`; a refuted hard claim still
  flags `hard_fail_capped = True`.
- The engine still records `overall_score_uncapped` (pre-cap) for
  analysis, but QD ranks on the capped `overall_score` — a run that
  refuted a hard claim cannot top the archive on its other claims alone.

## Failure fingerprint (diagnostic)

The verifier also projects the same per-claim verdicts into a **failure
fingerprint**: a centered vector of per-source pass rates that records
*how* a candidate fails, not *whether* it failed.

```python
# Per source letter (a–g, fixed length — absent sources get a neutral value).
pass_rate[s] = passes[s] / total[s]              if total[s] > 0  else 0.5
presence[s]  = 1.0                                if total[s] > 0  else 0.0
# Center so the vector encodes profile shape, not quality level.
mean_present = mean(pass_rate[s] for s where presence[s] == 1)
vector[s]    = pass_rate[s] - mean_present       if presence[s] == 1
             = 0                                  otherwise
```

Centering means an all-pass run and an all-fail run both collapse to the
zero profile (asserted by `test_all_pass_yields_zero_profile` and
`test_all_fail_yields_zero_profile`), so the vector captures the *shape*
of which sources fail rather than the overall quality level.

The fingerprint is computed by
[`compute_failure_fingerprint`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/failure_fingerprint.py)
at the end of `VerifierEvaluator.evaluate()` and persisted under
`state_result.json` → `evaluation.verifier.failure_fingerprint`.

> **Diagnostic only — not the novelty descriptor.** The QD behaviour
> descriptor is the **genotype embedding** of the workflow's source code,
> and novelty is cosine distance in that space (see
> [Selection](evolution-engine.md#behaviour-descriptor-genotype-embedding)).
> `SelectionPressure` no longer reads the failure fingerprint; it remains
> persisted so failure profiles can be inspected offline.
Full info-flow audit of the QD behaviour descriptor:
[`docs/info-flow/genotype_embedding.md`](../info-flow/genotype_embedding.md).

## Prompt gradient

After aggregation the verifier composes a single-sentence diagnosis (the
**prompt gradient**) from the per-claim report and recent history of
similar runs. It is the **only** verifier output that reaches the
mutator, and it is written so it does not leak the verified claims back:

- It does not name specific claims, scores, or which of the five sources
  raised the issue.
- It encodes failure modes by short code names (e.g.
  `FALLBACK_ECFP_CLASSIFIER`) so recurring patterns can be tracked across
  generations without spelling out the underlying check.

The intent is informational, not punitive: the mutator learns *what
direction to push the workflow next* without being handed a vocabulary
it can over-fit against.

## Behavioral pressure against shortcut workflows

Pressure against shortcut or fabricated workflows comes from the
verifier pipeline itself:

- Source C's recompute-from-disk verifiers.
- The "Used fallback" claim type, whose score is *inverted* — a passing
  fallback check counts as 0, a failing one as 1.
- The anti-tautology tripwires in the per-claim verifier generator.

## Why this design?

| Concern | Mitigation |
| ------- | ---------- |
| The judge LLM repeats the agent's claims verbatim. | Per-claim deterministic Python recomputation, with anti-tautology tripwires. |
| The mutator over-fits to a numeric rubric. | Only the prompt gradient is returned — and it does not name the verified claims. |
| The hard-fail cap collapses ranking among failed runs. | Deliberate: QD ranks on the capped `overall_score`, with ties at the cap broken by novelty; `overall_score_uncapped` is still logged for analysis. |
| One vantage point misses the failure. | Five independent sources, claims merged. |
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

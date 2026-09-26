"""Hybrid verifier (E35 PRIME, v3) — the default Mimosa verification channel.

Verifier v3 combines every measured win of the E19..E35 experiment
series into one production channel:

- **Content lever (E30)** — the extraction prompt demands >=2
  ``core_computation`` claims (the goal's computation end-to-end: named
  inputs -> core transformation with goal-stated parameters -> deliverable
  produced from it) and >=2 ``method_identity`` claims (exact named
  method/model/featurizer on goal-named data; lookalike=0, import is not
  identity), plus the E16 key_missing fidelity families
  (``output_schema`` / ``prediction_sanity`` / ``deliverable_path``).
  The mandate is enforced by ``claims.validate_claims`` with a retry
  budget and disclosed when the model's best effort still falls short.
- **T1 firewall (E29)** — claims must reference workspace-verifiable
  properties only; a batched LLM screen (one judge call per task's
  claims, temperature 0) rejects claims hard-coding numeric answer
  constants the goal does not state — structural constants (ranges,
  sentinels, bin/grid counts) judged verifiable — and the extraction is
  re-issued with feedback.
- **Visual rung, early (E26)** — figure tasks (>=50% image deliverables)
  get 3-6 goal-anchored visual claims extracted once by the vision model
  (kimi-k3, <=2 grounding figures, claims FROZEN before scoring) and
  every generation's <=3 largest figures are scored 0-1 per claim with
  the same rubric on both sides. The visual stage sits BEFORE
  ``script`` on the ladder (E26: visual-early 0.833 vs 0.646 as 4th
  rung). Config: ``hybrid_verifier_visual_rung`` (default True).
- **Execution gate (E24)** — the generation's clean-room re-execution
  status caps the reward after scoring: crash / no-entry / timeout -> 0.0,
  divergent -> 0.5, clean recovery -> no cap. Config:
  ``hybrid_verifier_execution_gate`` (default True). Status data comes
  from a lookup (frozen E24 ledger by default); a live re-execution
  check plugs in through the same layer interface.
- **Reward (E19c)** — raw win-rate vs every previous generation of the
  task (ties = 0.5; first generation falls back to the mean claim
  score), measured best on the frozen arena (0.742 online vs BT 0.643).
- **Gradient, V5 (E19b/E19c/E32)** — decisive lost claims lead (each
  row: goal-anchored requirement + measured score + scorer evidence),
  then decisive wins, execution facts, visual evidence, the dead-claims
  report. NO elimination-point framing (E19c measured it significantly
  worse: 0.553 vs 0.672 relevance); every comparative row is
  goal-anchored (E32).

Base pipeline (E19/E19b): per task, ONE LLM call extracts the temporal
ladder of key claims from the goal + workspace inventory (``script`` ->
``log`` -> ``result``); ONE LLM call per claim writes a self-contained
deterministic Python scorer that grades ANY single workspace on a
continuous 0..1 scale; scorers are reused verbatim across generations.
A per-task label-free registry tracks every claim's score vector:
zero-variance claims are dropped and replaced by refined claims. A
one-shot format digest (``digest``) samples deliverable files
deterministically so the policy-writing LLM sees real formats. The
default aggregation mode ``temporal`` decides each pairwise winner at
the EARLIEST ladder stage where the two generations differ.

Measured results behind the design (see
``experiments_verifiers/results/E35_prime_2026-09-26.md``): gradient
relevance 0.690 (best measured on this population, e35 - e30
+0.056 [+0.003,+0.118]); execution gate +0.049 raw -> gated online;
deterministic and symmetric by construction.

Module layout
-------------
- ``inventory``  — workspace inventory with cheap content signals
  (CSV rows x cols, PNG dims, file kinds) + union across generations +
  figure-task detection (E26).
- ``claims``     — temporal-ladder claim extraction and refinement
  prompts + validation (stage tagging, temporal order, generic-claim
  drop, count enforcement, the E30 composition mandate, the E29 T1
  firewall screen, id management).
- ``digest``     — one-shot deterministic format digests of the task's
  deliverable files (seeded head/middle/tail slices), cached per task.
- ``scorers``    — policy-script generation prompts (ladder framing +
  digests), static policy screening, output parsing, and the
  pinned-subprocess contract.
- ``registry``   — per-task JSON registry: claim set (with stage /
  temporal_index, migrated on load), cached scorer scripts, format
  digests, per-generation score vectors, union inventory.
- ``aggregation``— zero-variance filter; temporal-elimination pairwise
  comparison with a CONFIGURABLE stage order (``STAGES`` for text
  tasks; ``FIGURE_STAGES`` = visual -> script -> log -> result for
  figure tasks); reward computation (win-rate default, Bradley–Terry
  optional).
- ``gradient``   — V5 decisive-first gradient assembly +
  ``evaluation.txt`` rendering with stage headers.
- ``layers``     — the ``EvidenceLayer`` extension interface; the three
  v3 layers: ``ClaimsEvidenceLayer`` (claims + deterministic scorers),
  ``VisualEvidenceLayer`` (E26 visual rung) and ``ExecutionGateLayer``
  (E24 execution gate).
- ``evaluator``  — ``HybridVerifierEvaluator``, the ``BaseEvaluator``
  subclass wired into the evaluation facade.

Extending with new evidence layers
----------------------------------
The aggregator is layer-agnostic: it consumes ``(claim_id, score)``
pairs keyed only by id. To add an evidence layer:

1. Implement the ``EvidenceLayer`` protocol from ``layers``
   (``collect(context) -> list[LayerScore]`` plus the optional
   ``revise`` and ``gate(context, collected) -> float | None`` cap
   hook).
2. Pass ``extra_layers=[...]`` (or override ``layers``) when
   constructing ``HybridVerifierEvaluator``.

No aggregation, registry, reward, or gradient code needs to change: the
new layer's ``LayerScore`` entries merge into the same score matrix, and
its ``gate`` return value caps the final reward
(``reward = min(reward, *caps)``); a layer exposing ``last_facts``
feeds the gradient's execution-facts section.
"""

from .aggregation import claim_stats, decisive, winner_of
from .evaluator import HybridVerifierEvaluator
from .layers import (
    ClaimsEvidenceLayer,
    EvidenceLayer,
    ExecutionGateLayer,
    LayerContext,
    LayerScore,
    VisualEvidenceLayer,
)
from .registry import TaskRegistry

__all__ = [
    "ClaimsEvidenceLayer",
    "EvidenceLayer",
    "ExecutionGateLayer",
    "HybridVerifierEvaluator",
    "LayerContext",
    "LayerScore",
    "TaskRegistry",
    "VisualEvidenceLayer",
    "claim_stats",
    "decisive",
    "winner_of",
]

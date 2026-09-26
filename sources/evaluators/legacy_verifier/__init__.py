"""[DEPRECATED 2026-09-24] Legacy per-claim verifier (pre-E19 design).

Superseded by ``sources.evaluators.hybrid_verifier`` (E19/E19b design).
Nothing in this package is wired into the default evaluation path anymore;
it is kept for reference and research replay, so do not extend it.

The package bundles the five modules of the old flat layout:

- ``verifier``             — ``VerifierEvaluator`` orchestrator (public class).
- ``verifier_claims``      — claim extraction (five sources) + importance rating.
- ``verifier_claim_sources`` — claim-extraction source prompts + dispatch table.
- ``verifier_per_claim``   — verifier-script generation, sandbox execution, scoring.
- ``verifier_workspace``   — workspace listing, file previews, grounding cache.

Import submodules directly (``from sources.evaluators.legacy_verifier import
verifier_per_claim``) for anything beyond the evaluator class itself.
"""

from .verifier import VerifierEvaluator

__all__ = ["VerifierEvaluator"]

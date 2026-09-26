"""Evaluator package for the Mimosa workflow evaluation system.

Historically this folder had no ``__init__.py`` (namespace package) and each
evaluator lived as a flat module; the 2026-09-26 refactor moved every
evaluator into its own subpackage. Current layout:

- ``base``            — shared ``BaseEvaluator`` + payload-parsing helpers.
- ``grounding``       — literature-grounding helper shared by evaluators.
- ``evaluator``       — the ``WorkflowEvaluator`` facade routing to channels.
- ``generic``         — generic 4-axis LLM judge evaluator.
- ``scenario``        — scenario-rubric evaluator.
- ``hybrid_verifier`` — the DEFAULT verification channel (E19/E19b design).
- ``legacy_verifier`` — the deprecated pre-E19 per-claim verifier.

The re-exports below are deliberately LAZY (PEP 562 ``__getattr__``): an
``import sources.evaluators`` stays cheap, never pulls the deprecated legacy
chain, and cannot re-open the ``scenario`` -> ``scenario_loader`` import
cycle documented in ``sources/benchmark_evaluation/__init__.py``. Prefer the
explicit module paths (e.g. ``from sources.evaluators.evaluator import
WorkflowEvaluator``) in new code; these aliases exist for convenience only.
"""

from __future__ import annotations

import importlib

_REEXPORTS: dict[str, str] = {
    "WorkflowEvaluator": ".evaluator",
    "GenericEvaluator": ".generic",
    "ScenarioEvaluator": ".scenario",
    "HybridVerifierEvaluator": ".hybrid_verifier",
    "VerifierEvaluator": ".legacy_verifier",
}


def __getattr__(name: str) -> object:
    """Lazily resolve the package-level evaluator re-exports."""
    try:
        target = _REEXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    return getattr(importlib.import_module(target, __package__), name)


def __dir__() -> list[str]:
    """Complete ``dir()`` with the lazy re-export names."""
    return sorted(set(globals()) | set(_REEXPORTS))


__all__ = list(_REEXPORTS)

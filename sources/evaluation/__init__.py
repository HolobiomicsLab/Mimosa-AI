"""Evaluation sub-package for Mimosa-AI.

Provides tools for assessing workflow outputs, scoring generated code,
detecting numerical inconsistencies, and running benchmark datasets.

This module deliberately exposes NO package-level re-exports. The previous
re-export block pulled symbols from ``sources.core.evaluators.evaluator``,
which itself imports ``sources.core.evaluators.scenario`` — and ``scenario``
imports ``sources.evaluation.scenario_loader``, which re-enters this file.
That cycle broke any test collection that touched ``sources.core``.

Every caller in the codebase already uses fully-qualified submodule paths
(e.g. ``from sources.evaluation.scenario_loader import ScenarioLoader``), so
removing the re-exports is purely cycle-breaking — no public API regression.
"""

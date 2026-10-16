"""Scenario evaluator package — rubric-based workflow evaluation.

Moved here (2026-09-26) from the flat ``sources/evaluators/`` layout:
``scenario`` holds ``ScenarioEvaluator``, which loads scenario rubrics from
disk (legacy assertion format or ScienceAgentBench format) and scores a
workflow against them with an LLM judge.
"""

from .scenario import ScenarioError, ScenarioEvaluator

__all__ = [
    "ScenarioError",
    "ScenarioEvaluator",
]

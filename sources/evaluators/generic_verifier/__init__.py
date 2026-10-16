"""Generic evaluator package — 4-axis LLM judgment of workflow runs.

Moved here (2026-09-26) from the flat ``sources/evaluators/`` layout:

- ``generic``       — ``GenericEvaluator``: goal alignment, agent
  collaboration, output quality and answer plausibility, each scored by an
  independent LLM judge, plus the optional numerical bullshit penalty.
- ``bs_detection``  — ``BullshitDetectorNumerical``, the memory-review fraud
  detector behind the penalty (deprecated prototype status of its own).
"""

from .generic import GenericEvaluator, LLMEvaluationError, ScoreExtractionError

__all__ = [
    "GenericEvaluator",
    "LLMEvaluationError",
    "ScoreExtractionError",
]

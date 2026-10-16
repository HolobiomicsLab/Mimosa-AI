"""Full-oracle reward (``gold_feedback_reward=True`` only)."""

from __future__ import annotations

from typing import Any


def oracle_reward(grade: dict[str, Any]) -> float:
    """Benchmark-grader reward in [0, 1] for one graded generation.

    ``SR`` true gives 1.0 (reaches the 0.9 early-stop threshold). A clean
    re-execution with SR false gives ``0.5 * CBS`` (CodeBERT similarity to
    the gold program). A failed re-execution gives 0.0.

    Args:
        grade: Normalised grade with ``status="evaluated"``.

    Returns:
        The reward, rounded to 4 decimals.
    """
    if grade.get("SR"):
        return 1.0
    if grade.get("VER"):
        cbs = float(grade.get("CBS") or 0.0)
        return round(max(0.0, min(1.0, 0.5 * cbs)), 4)
    return 0.0

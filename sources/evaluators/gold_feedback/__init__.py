"""Gold-feedback (ORACLE / BENCHMARK-LEAKING) verifier control.

``verifier_kind = "gold"`` wraps the hybrid verifier and replaces the
textual gradient with the benchmark grader's own feedback (VER / SR
messages against the gold reference). Optional full-oracle mode
(``gold_feedback_reward = True``) also takes the reward from the grader.

This mode leaks the benchmark. It is an upper-bound research control
(E43 causal feedback experiment) and must never be used for reported
benchmark scores.

Modules:
    evaluator: ``GoldFeedbackEvaluator`` (composition: hybrid + grader).
    grader: task context, private workspace copy, shared grading call.
    format: gold-gradient text (E33 format, <= 2000 chars).
    reward: full-oracle reward.
    leak: the leakage warning text.
"""

from .evaluator import GoldFeedbackEvaluator
from .format import MAX_GRADIENT_CHARS, build_gold_gradient
from .leak import LEAK_WARNING

__all__ = [
    "GoldFeedbackEvaluator",
    "LEAK_WARNING",
    "MAX_GRADIENT_CHARS",
    "build_gold_gradient",
]

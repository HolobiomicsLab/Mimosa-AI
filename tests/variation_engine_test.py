"""Tests for VariationEngine plateau counter, success-rate signal and adaptive step size."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from sources.core.variation_engine import VariationEngine


class _FakeConfig:
    """Minimal stand-in for the real Config, just enough to satisfy
    VariationEngine.setup_llm without touching any network/provider code."""

    workflow_llm_model = "openai/gpt-4o-mini"
    reasoning_effort = "low"
    max_tokens = 8192

    def openrouter_provider_for(self, model):
        return None

    def openrouter_quantizations_for(self, model):
        return None


def _make_engine():
    return VariationEngine(_FakeConfig())


# ── _iters_since_improvement ──────────────────────────────────────────────


def test_iters_since_improvement_empty_history_is_zero():
    ve = _make_engine()
    assert ve._iters_since_improvement() == 0


def test_iters_since_improvement_no_scored_entries_is_zero():
    ve = _make_engine()
    for _ in range(4):
        ve.record_offspring_gradient("crash", is_failure=True)
    ve.record_offspring_gradient("no scores attached")  # child_score = None
    assert ve._iters_since_improvement() == 0


def test_iters_since_improvement_counts_consecutive_non_improvers():
    ve = _make_engine()
    ve.record_offspring_gradient("up",   child_score=0.30, best_before=0.20)
    ve.record_offspring_gradient("flat", child_score=0.30, best_before=0.30)
    ve.record_offspring_gradient("flat", child_score=0.30, best_before=0.30)
    ve.record_offspring_gradient("flat", child_score=0.30, best_before=0.30)
    assert ve._iters_since_improvement() == 3


def test_iters_since_improvement_resets_after_strict_improvement():
    ve = _make_engine()
    for _ in range(4):
        ve.record_offspring_gradient("flat", child_score=0.5, best_before=0.5)
    ve.record_offspring_gradient("up", child_score=0.6, best_before=0.5)
    assert ve._iters_since_improvement() == 0


def test_iters_since_improvement_skips_failures_and_none_entries():
    """Failures and unscored entries are transparent: they neither break nor
    extend the streak. The streak walks through them as if they weren't there."""
    ve = _make_engine()
    ve.record_offspring_gradient("flat", child_score=0.5, best_before=0.5)
    ve.record_offspring_gradient("crash", is_failure=True)
    ve.record_offspring_gradient("nul",   child_score=None, best_before=0.5)
    ve.record_offspring_gradient("flat", child_score=0.5, best_before=0.5)
    assert ve._iters_since_improvement() == 2


def test_iters_since_improvement_tie_is_not_an_improvement():
    """``c == b`` is a tie, not a strict improvement: the streak must NOT
    reset on equality."""
    ve = _make_engine()
    ve.record_offspring_gradient("flat", child_score=0.5, best_before=0.5)
    ve.record_offspring_gradient("flat", child_score=0.5, best_before=0.5)
    assert ve._iters_since_improvement() == 2


# ── _compute_success_rate (kept; unchanged contract) ──────────────────────


def test_compute_success_rate_none_when_no_scored_history():
    ve = _make_engine()
    ve.record_offspring_gradient("only a gradient, no score")
    assert ve._compute_success_rate() is None


def test_compute_success_rate_counts_strict_improvements():
    ve = _make_engine()
    # 2 improvements out of 4 ⇒ success_rate = 0.5
    ve.record_offspring_gradient("a", child_score=0.30, best_before=0.20)
    ve.record_offspring_gradient("b", child_score=0.30, best_before=0.30)  # tie ≠ improvement
    ve.record_offspring_gradient("c", child_score=0.40, best_before=0.30)
    ve.record_offspring_gradient("d", child_score=0.40, best_before=0.40)  # tie ≠ improvement
    assert ve._compute_success_rate() == 0.5


def test_compute_success_rate_excludes_failures():
    ve = _make_engine()
    ve.record_offspring_gradient("ok", child_score=0.50, best_before=0.40)
    ve.record_offspring_gradient(
        "crash", is_failure=True, child_score=0.99, best_before=0.40
    )
    ve.record_offspring_gradient("ok", child_score=0.55, best_before=0.50)
    assert ve._compute_success_rate() == 1.0

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")
    print("All variation_engine_test passed.")

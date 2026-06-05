"""Tests for VariationEngine stagnation signal and adaptive step size."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from sources.core.variation_engine import VariationEngine


def test_prompt_gradient_similarity_bounds_and_identity():
    ve = VariationEngine()
    sim_same = ve._prompt_gradient_similarity("foo bar baz", "foo bar baz")
    assert 0.99 <= sim_same <= 1.0 + 1e-6
    assert ve._prompt_gradient_similarity("", "anything") == 0.0
    assert ve._prompt_gradient_similarity("foo", "") == 0.0
    assert ve._prompt_gradient_similarity("", "") == 0.0


def _stub_similarity(ve: VariationEngine, fn) -> None:
    ve._prompt_gradient_similarity = fn.__get__(ve, type(ve))  # type: ignore[attr-defined]


def test_stagnation_empty_history_is_zero():
    ve = VariationEngine()
    assert ve._compute_stagnation() == 0.0


def test_stagnation_single_entry_is_zero():
    ve = VariationEngine()
    ve.record_offspring_gradient("only one entry")
    assert ve._compute_stagnation() == 0.0


def test_stagnation_high_when_offspring_gradients_repeat():
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 1.0)
    for _ in range(4):
        ve.record_offspring_gradient("same failure mode every time")
    assert ve._compute_stagnation() == 1.0


def test_stagnation_low_when_offspring_gradients_diverge():
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 0.4)
    for g in ("a", "b", "c", "d"):
        ve.record_offspring_gradient(g)
    assert ve._compute_stagnation() < 1e-9


def test_stagnation_window_only_considers_recent():
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 1.0 if a == b else 0.4)
    ve.record_offspring_gradient("ancient_a")
    ve.record_offspring_gradient("ancient_b")
    for _ in range(4):
        ve.record_offspring_gradient("recent_stuck")
    assert ve._compute_stagnation() == 1.0


def test_failure_tagged_gradients_are_filtered_from_stagnation():
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 1.0)
    for _ in range(4):
        ve.record_offspring_gradient("anything", is_failure=True)
    assert ve._compute_stagnation() == 0.0


def test_failure_entries_persist_in_history_for_audit():
    ve = VariationEngine()
    ve.record_offspring_gradient("crash", is_failure=True)
    ve.record_offspring_gradient("real diag")
    assert ve.prompt_gradient_history == [("crash", True), ("real diag", False)]


def test_step_size_damped_when_recent_offspring_improve():
    """Rechenberg 1/5 rule: with success_rate above threshold, even high
    gradient stagnation should yield a small mutation scope."""
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 1.0)
    prev_best = 0.5
    for inc in (0.05, 0.07, 0.09, 0.11, 0.13):
        ve.record_offspring_gradient(
            "repeating diagnosis text",
            child_score=prev_best + inc,
            best_before=prev_best,
        )
        prev_best += inc
    block = ve._get_prompt_step_size(parent_score=0.97)
    assert "prompt-only little tweak" in block, block


def test_step_size_unleashed_when_recent_offspring_stuck_despite_high_parent():
    """Plateau case: high parent_score + zero improvements + repeating
    gradient must NOT lock the search into 'tiny tweak' mode. This is the
    regression test for the 0.92 plateau bug — before the 1/5-rule rewrite,
    the old (1 − parent_score) damping returned "prompt-only little tweak"
    here. The principled fix escalates scope."""
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 1.0)
    for _ in range(5):
        ve.record_offspring_gradient(
            "DATA_LEAKAGE: same diagnosis again",
            child_score=0.92,
            best_before=0.92,
        )
    block = ve._get_prompt_step_size(parent_score=0.92)
    assert "tweak" not in block, block
    assert any(tag in block for tag in ("rewire", "rethink", "redesign")), block


def test_step_size_unleashed_when_parent_score_low_and_stuck():
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 1.0)
    for _ in range(5):
        ve.record_offspring_gradient(
            "repeating failure",
            child_score=0.10,
            best_before=0.10,
        )
    block = ve._get_prompt_step_size(parent_score=0.10)
    assert ("rewire" in block) or ("rethink" in block), block


def test_step_size_parent_score_clipped_to_unit_interval():
    """Out-of-range parent_score must be clipped to [0, 1] before use; the
    boldness value reported in the prompt stays a percentage in [0, 100]."""
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 1.0)
    for _ in range(4):
        ve.record_offspring_gradient("repeating failure")
    # Clipping behaviour: parent_score=1.5 must be treated as 1.0, not
    # crash and not produce a negative or > 100 % boldness in the output.
    block = ve._get_prompt_step_size(parent_score=1.5)
    import re
    m = re.search(r"Boldness:\s*([\d.]+)%", block)
    assert m is not None, f"no boldness reported: {block}"
    boldness_pct = float(m.group(1))
    assert 0.0 <= boldness_pct <= 100.0, boldness_pct


def test_compute_success_rate_none_when_no_scored_history():
    ve = VariationEngine()
    ve.record_offspring_gradient("only a gradient, no score")
    assert ve._compute_success_rate() is None


def test_compute_success_rate_counts_strict_improvements():
    ve = VariationEngine()
    # 2 improvements out of 4 ⇒ success_rate = 0.5
    ve.record_offspring_gradient("a", child_score=0.30, best_before=0.20)
    ve.record_offspring_gradient("b", child_score=0.30, best_before=0.30)  # tie ≠ improvement
    ve.record_offspring_gradient("c", child_score=0.40, best_before=0.30)
    ve.record_offspring_gradient("d", child_score=0.40, best_before=0.40)  # tie ≠ improvement
    assert ve._compute_success_rate() == 0.5


def test_compute_success_rate_excludes_failures():
    ve = VariationEngine()
    ve.record_offspring_gradient("ok", child_score=0.50, best_before=0.40)
    ve.record_offspring_gradient(
        "crash", is_failure=True, child_score=0.99, best_before=0.40
    )
    ve.record_offspring_gradient("ok", child_score=0.55, best_before=0.50)
    assert ve._compute_success_rate() == 1.0


def test_mutation_prompt_does_not_touch_gradient_history():
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 0.4)

    class _FakeWfInfo:
        overall_score = 0.5
        abstracted_prompt_gradient = "FAKE_DIAG_should_not_be_recorded"
        state_result = None

    before = len(ve.prompt_gradient_history)
    ve.mutation_prompt(
        goal="g",
        wf_info=_FakeWfInfo(),
        genotype="# code",
        run_stderr="",
        iteration_count=0,
        max_iterations=10,
    )
    assert len(ve.prompt_gradient_history) == before


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")
    print("All variation_engine_test passed.")

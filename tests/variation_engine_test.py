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


def test_step_size_damped_by_high_parent_score():
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 1.0)
    for _ in range(4):
        ve.record_offspring_gradient("INCONSISTENT_MULTITASK_SPLIT repeating")
    block = ve._get_prompt_step_size(parent_score=0.97)
    assert "prompt-only little tweak" in block, block


def test_step_size_unleashed_when_parent_score_low():
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 1.0)
    for _ in range(4):
        ve.record_offspring_gradient("INCONSISTENT_MULTITASK_SPLIT repeating")
    block = ve._get_prompt_step_size(parent_score=0.10)
    assert ("rewire" in block) or ("rethink" in block), block


def test_step_size_parent_score_clipped_to_unit_interval():
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 1.0)
    for _ in range(4):
        ve.record_offspring_gradient("repeating failure")
    block = ve._get_prompt_step_size(parent_score=1.5)
    assert "prompt-only little tweak" in block


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

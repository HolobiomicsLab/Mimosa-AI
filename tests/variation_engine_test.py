"""Tests for VariationEngine — stagnation signal + adaptive step size.

Covers:
- Cosine similarity contract on MiniLM-embedded prompt gradients.
- Stagnation computation over offspring gradients with failure-sentinel filter.
- `_get_prompt_step_size` damping by parent score (near-winners stay protected).
- `record_offspring_gradient` is the only state-mutating entry point for the
  stagnation signal (mutation_prompt no longer touches the history).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from sources.core.variation_engine import VariationEngine


# ── Similarity contract (one real-embedder test) ───────────────────────────


def test_prompt_gradient_similarity_bounds_and_identity():
    ve = VariationEngine()
    # Identical text → cosine ≈ 1.0 (allow tiny float wobble).
    sim_same = ve._prompt_gradient_similarity("foo bar baz", "foo bar baz")
    assert 0.99 <= sim_same <= 1.0 + 1e-6
    # Empty strings short-circuit to 0.0 without invoking the embedder.
    assert ve._prompt_gradient_similarity("", "anything") == 0.0
    assert ve._prompt_gradient_similarity("foo", "") == 0.0
    assert ve._prompt_gradient_similarity("", "") == 0.0


# ── Stagnation logic (similarity stubbed for speed + determinism) ──────────


def _stub_similarity(ve: VariationEngine, fn) -> None:
    """Bypass MiniLM by patching the instance similarity method."""
    ve._prompt_gradient_similarity = fn.__get__(ve, type(ve))  # type: ignore[attr-defined]


def test_stagnation_empty_history_is_zero():
    ve = VariationEngine()
    assert ve._compute_stagnation() == 0.0


def test_stagnation_single_entry_is_zero():
    ve = VariationEngine()
    ve.record_offspring_gradient("only one entry")
    assert ve._compute_stagnation() == 0.0


def test_stagnation_high_when_offspring_gradients_repeat():
    """Identical offspring gradients → raw cosine ~1.0 → max stagnation."""
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 1.0)
    for _ in range(4):
        ve.record_offspring_gradient("same failure mode every time")
    assert ve._compute_stagnation() == 1.0


def test_stagnation_low_when_offspring_gradients_diverge():
    """Distinct gradients (pairwise sim at MiniLM baseline) → ~0 stagnation."""
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 0.4)  # MiniLM unrelated baseline
    for g in ("a", "b", "c", "d"):
        ve.record_offspring_gradient(g)
    assert ve._compute_stagnation() < 1e-9


def test_stagnation_window_only_considers_recent():
    """Old diverse entries must not dilute a recent stuck streak."""
    ve = VariationEngine()
    sims = {("ancient_a", "ancient_b"): 0.4}  # baseline; ignored once window slides
    _stub_similarity(ve, lambda self, a, b: 1.0 if a == b else 0.4)
    ve.record_offspring_gradient("ancient_a")
    ve.record_offspring_gradient("ancient_b")
    for _ in range(4):
        ve.record_offspring_gradient("recent_stuck")
    # Default window=4 → all four entries are identical → stagnation = 1.0
    assert ve._compute_stagnation() == 1.0


# ── Fix #3: failure sentinels must NOT peg stagnation ──────────────────────


def test_failure_tagged_gradients_are_filtered_from_stagnation():
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 1.0)
    sentinel = "workflow code failed to generate or execute; ensure code runs"
    for _ in range(4):
        ve.record_offspring_gradient(sentinel, is_failure=True)
    # Even with cosine pinned at 1.0, failure entries get filtered out → 0.
    assert ve._compute_stagnation() == 0.0


def test_failure_entries_persist_in_history_for_audit():
    """is_failure is a filter for the stagnation signal, not a drop."""
    ve = VariationEngine()
    ve.record_offspring_gradient("crash", is_failure=True)
    ve.record_offspring_gradient("real diag")
    assert len(ve.prompt_gradient_history) == 2
    assert ve.prompt_gradient_history[0] == ("crash", True)
    assert ve.prompt_gradient_history[1] == ("real diag", False)


# ── Fix #2: parent score damps boldness near a winner ──────────────────────


def test_step_size_damped_by_high_parent_score():
    """A 0.97 parent under maxed-out raw stagnation should still get a tweak."""
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 1.0)
    for _ in range(4):
        ve.record_offspring_gradient("INCONSISTENT_MULTITASK_SPLIT repeating")
    block = ve._get_prompt_step_size(parent_score=0.97)
    assert "prompt-only tweak" in block, block


def test_step_size_unleashed_when_parent_score_low():
    """Same raw stagnation, low parent score → bold mutation permitted."""
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 1.0)
    for _ in range(4):
        ve.record_offspring_gradient("INCONSISTENT_MULTITASK_SPLIT repeating")
    block = ve._get_prompt_step_size(parent_score=0.10)
    assert ("rewire" in block) or ("rethink" in block), block


def test_step_size_parent_score_clipped_to_unit_interval():
    """parent_score outside [0,1] must not break the damper."""
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 1.0)
    for _ in range(4):
        ve.record_offspring_gradient("repeating failure")
    # Should not raise; should behave as if parent_score=1.0 (full damp).
    block = ve._get_prompt_step_size(parent_score=1.5)
    assert "prompt-only tweak" in block


# ── Fix #1: mutation_prompt must NOT mutate the gradient history ──────────


def test_mutation_prompt_does_not_touch_gradient_history():
    """History is fed by the engine via record_offspring_gradient only."""
    ve = VariationEngine()
    _stub_similarity(ve, lambda self, a, b: 0.4)

    class _FakeWfInfo:
        overall_score = 0.5
        abstracted_prompt_gradient = "FAKE_DIAG_should_not_be_recorded"
        state_result = None

    before = len(ve.prompt_gradient_history)
    _ = ve.mutation_prompt(
        goal="g",
        wf_info=_FakeWfInfo(),
        genotype="# code",
        run_stderr="",
        iteration_count=0,
        max_iterations=10,
    )
    after = len(ve.prompt_gradient_history)
    assert after == before, (
        f"mutation_prompt must not append to history; "
        f"grew from {before} to {after}"
    )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")
    print("All variation_engine_test passed.")

"""Tests for VariationEngine stagnation-aware phase regression.

Covers the new closed-loop signal: pairwise diagnosis similarity → Beta-sampled
progress regression → earlier phase prompt when the LLM-mutator is cycling.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from sources.core.variation_engine import VariationEngine


# ── _diagnosis_similarity ─────────────────────────────────────────────────


def test_diagnosis_similarity_identical_is_one():
    ve = VariationEngine()
    assert ve._diagnosis_similarity("foo bar baz", "foo bar baz") == 1.0


def test_diagnosis_similarity_disjoint_is_zero():
    ve = VariationEngine()
    assert ve._diagnosis_similarity("foo bar", "baz qux") == 0.0


def test_diagnosis_similarity_partial_overlap_half():
    ve = VariationEngine()
    # {foo, bar, baz, qux} ∩ {foo, bar, zap, zip} = {foo, bar} (size 2)
    # union size 6 → 2/6 ≈ 0.333
    sim = ve._diagnosis_similarity("foo bar baz qux", "foo bar zap zip")
    assert abs(sim - 2 / 6) < 1e-6


def test_diagnosis_similarity_handles_empty():
    ve = VariationEngine()
    assert ve._diagnosis_similarity("", "foo") == 0.0
    assert ve._diagnosis_similarity("foo", "") == 0.0
    assert ve._diagnosis_similarity("", "") == 0.0


def test_diagnosis_similarity_is_case_insensitive():
    ve = VariationEngine()
    assert ve._diagnosis_similarity("Foo BAR", "foo bar") == 1.0


# ── _compute_stagnation ───────────────────────────────────────────────────


def test_stagnation_empty_history_is_zero():
    ve = VariationEngine()
    assert ve._compute_stagnation() == 0.0


def test_stagnation_single_entry_is_zero():
    ve = VariationEngine()
    ve.diagnosis_history.append("only one entry")
    assert ve._compute_stagnation() == 0.0


def test_stagnation_high_when_diagnoses_cluster():
    ve = VariationEngine()
    diag = "verification failed assert claim X not supported in workspace"
    ve.diagnosis_history.extend([diag] * 5)
    assert ve._compute_stagnation() > 0.95


def test_stagnation_low_when_diagnoses_diverse():
    ve = VariationEngine()
    ve.diagnosis_history.extend([
        "verification failed claim X",
        "tool call returned empty result",
        "import error matplotlib missing",
        "agent emitted no answer step 2",
        "syntax error in generated code",
    ])
    assert ve._compute_stagnation() < 0.3


def test_stagnation_window_only_considers_recent():
    """Old diverse diagnoses shouldn't dilute a recent stuck streak."""
    ve = VariationEngine()
    ve.diagnosis_history.extend([
        "ancient diverse alpha",
        "ancient diverse beta",
        "ancient diverse gamma",
    ])
    ve.diagnosis_history.extend(["recent stuck mode"] * 5)
    # default window=5 picks up only the stuck ones
    assert ve._compute_stagnation(window=5) > 0.95


# ── _sample_phase_regression ──────────────────────────────────────────────


def test_regression_zero_when_no_stagnation():
    ve = VariationEngine()
    assert ve._sample_phase_regression(0.0) == 0.0


def test_regression_bounded_by_max():
    ve = VariationEngine()
    np.random.seed(0)
    for _ in range(100):
        r = ve._sample_phase_regression(0.99, max_regression=0.45)
        assert 0.0 <= r <= 0.45


def test_regression_mean_tracks_stagnation():
    """E[Beta(α, β)] = stagnation → mean of N samples ≈ stagnation·max_regression."""
    ve = VariationEngine()
    np.random.seed(42)
    samples = [ve._sample_phase_regression(0.8) for _ in range(500)]
    mean = sum(samples) / len(samples)
    # target: 0.8 × 0.45 = 0.36, allow ±0.06 noise for n=500
    assert 0.30 < mean < 0.42, f"expected ~0.36, got {mean:.3f}"


# ── _get_temperature_phase wiring ─────────────────────────────────────────


def test_phase_emits_stagnation_hint_when_stuck():
    ve = VariationEngine()
    ve.diagnosis_history.extend(["stuck same failure same"] * 6)
    np.random.seed(0)
    block = ve._get_temperature_phase(iteration_count=30, max_iterations=35)
    assert "stagnation=" in block


def test_phase_no_hint_when_not_stuck():
    ve = VariationEngine()
    block = ve._get_temperature_phase(iteration_count=2, max_iterations=35)
    assert "stagnation=" not in block


def test_phase_regression_pushes_to_earlier_phase():
    """High stagnation late in run should sometimes regress out of POLISH."""
    ve = VariationEngine()
    ve.diagnosis_history.extend(["identical failure mode"] * 6)
    np.random.seed(1)
    # iteration 30 / 35 → progress ≈ 0.88 → POLISH (≥ 0.85)
    # With stagnation ≈ 1.0 and Beta mean = 0.45, we expect frequent regression
    saw_earlier_phase = False
    for seed in range(30):
        np.random.seed(seed)
        block = ve._get_temperature_phase(iteration_count=30, max_iterations=35)
        if "POLISH" not in block:
            saw_earlier_phase = True
            break
    assert saw_earlier_phase, "high stagnation should sometimes escape POLISH"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")
    print("✅ All variation_engine_test passed.")

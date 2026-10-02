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


# ── record_offspring_gradient: history bookkeeping unchanged ──────────────


def test_failure_entries_persist_in_history_for_audit():
    ve = _make_engine()
    ve.record_offspring_gradient("crash", is_failure=True)
    ve.record_offspring_gradient("real diag")
    assert ve.textual_gradient_history == [("crash", True), ("real diag", False)]


# ── _get_prompt_step_size routing ─────────────────────────────────────────


def test_step_size_damped_when_recent_offspring_improve():
    """When success_rate ≥ 0.80, boldness collapses regardless of plateau."""
    ve = _make_engine()
    prev_best = 0.5
    for inc in (0.05, 0.07, 0.09, 0.11, 0.13):
        ve.record_offspring_gradient(
            "PROGRESS " + str(inc),
            child_score=prev_best + inc,
            best_before=prev_best,
        )
        prev_best += inc
    block = ve._get_prompt_step_size(parent_score=0.97)
    assert ve.last_variation_state["effective_boldness"] < 0.35, ve.last_variation_state
    assert "Slight mutation" in block, block


def test_step_size_plateau_escalates_but_hysteresis_blocks_respeciation():
    """Plateau at high parent_score: scope must lift past EXPLOITATION but
    must not reach RE-SPECIATION on a 5-iteration streak — the hysteresis
    gate requires iters_since_improvement ≥ 8."""
    ve = _make_engine()
    for _ in range(5):
        ve.record_offspring_gradient(
            "DATA_LEAKAGE: same diagnosis again",
            child_score=0.92,
            best_before=0.92,
        )
    block = ve._get_prompt_step_size(parent_score=0.92)
    state = ve.last_variation_state
    assert state["effective_boldness"] >= 0.35, state
    assert state["effective_boldness"] < 0.90, state
    assert state["respeciation_gate_open"] is False, state
    assert "Bolder mutation" not in block, block


def test_step_size_low_parent_and_stuck_lifts_scope():
    ve = _make_engine()
    for _ in range(5):
        ve.record_offspring_gradient(
            "repeating failure",
            child_score=0.10,
            best_before=0.10,
        )
    ve._get_prompt_step_size(parent_score=0.10)
    state = ve.last_variation_state
    # 5 consecutive non-improvers + success_rate = 0.0 must escape EXPLOITATION.
    assert state["effective_boldness"] >= 0.35, state


def test_hysteresis_gate_opens_after_eight_consecutive_non_improvers():
    """RE-SPECIATION fires only when iters_since_improvement ≥ 8 AND
    success_rate ∈ {None, 0.0}."""
    ve = _make_engine()
    for _ in range(8):
        ve.record_offspring_gradient(
            "stuck", child_score=0.5, best_before=0.5,
        )
    block = ve._get_prompt_step_size(parent_score=0.5)
    state = ve.last_variation_state
    assert state["respeciation_gate_open"] is True, state
    assert "Bolder mutation" in block, block


def test_cold_start_boldness_capped_at_thirty_percent():
    """No scored offspring → success_rate is None → effective ≤ 0.3 · plateau."""
    ve = _make_engine()
    for _ in range(20):
        ve.record_offspring_gradient("just a diagnosis, no score")
    ve._get_prompt_step_size(parent_score=0.5)
    state = ve.last_variation_state
    assert state["success_rate"] is None, state
    assert state["effective_boldness"] <= 0.3 + 1e-9, state


def test_step_size_parent_score_clipped_to_unit_interval():
    """Out-of-range parent_score must be clipped to [0, 1]; the reported
    boldness percentage stays inside [0, 100]."""
    ve = _make_engine()
    for _ in range(4):
        ve.record_offspring_gradient(
            "stuck", child_score=0.5, best_before=0.5,
        )
    block = ve._get_prompt_step_size(parent_score=1.5)
    import re
    m = re.search(r"Boldness:\s*([\d.]+)%", block)
    assert m is not None, f"no boldness reported: {block}"
    boldness_pct = float(m.group(1))
    assert 0.0 <= boldness_pct <= 100.0, boldness_pct


def test_mutation_prompt_does_not_touch_gradient_history():
    ve = _make_engine()

    class _FakeWfInfo:
        overall_score = 0.5
        abstracted_textual_gradient = "FAKE_DIAG_should_not_be_recorded"
        state_result = None

    before = len(ve.textual_gradient_history)
    ve.mutation_prompt(
        goal="g",
        wf_info=_FakeWfInfo(),
        genotype="# code",
        run_stderr="",
        iteration_count=0,
        max_iterations=10,
    )
    assert len(ve.textual_gradient_history) == before


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")
    print("All variation_engine_test passed.")

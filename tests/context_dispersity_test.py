"""Tests for the context-dispersity selection penalty.

The penalty rewards workflows that spread context across agents over
workflows that pile it onto one, without imposing a boundary on the agent.
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from sources.core.selection import (
    SelectionPressure,
    _agent_context_lengths,
    _context_dispersity,
)
from sources.utils.agent_context import read_agent_context_lengths


# ── dispersity measure ────────────────────────────────────────────────────


def test_dispersity_is_zero_for_evenly_shared_context():
    assert _context_dispersity([1000, 1000, 1000]) == 0.0


def test_dispersity_is_zero_below_two_agents():
    """A single agent carries no dispersion information."""
    assert _context_dispersity([900_000]) == 0.0
    assert _context_dispersity([]) == 0.0


def test_dispersity_grows_when_one_agent_hoards_context():
    balanced = _context_dispersity([500, 500, 500, 500])
    skewed = _context_dispersity([10, 10, 10, 900])
    assert skewed > balanced
    assert 0.0 <= skewed <= 1.0


def test_total_concentration_reaches_one():
    """One agent holding everything is the maximum of the measure."""
    assert _context_dispersity([1000, 0, 0, 0]) == pytest.approx(1.0)
    assert _context_dispersity([1000, 1, 1, 1]) == pytest.approx(1.0, abs=0.01)


def test_dispersity_keeps_a_gradient_in_the_severe_regime():
    """Severe imbalances must stay distinguishable rather than all clipping to 1."""
    severe = _context_dispersity([10, 10, 10, 900])
    worse = _context_dispersity([1, 1, 1, 900])
    assert severe < worse <= 1.0


def test_dispersity_is_scale_free_and_ignores_absolute_size():
    """Documented limitation: equally huge contexts are not penalised."""
    assert _context_dispersity([900_000] * 4) == 0.0
    assert _context_dispersity([10, 10, 10, 900]) == pytest.approx(
        _context_dispersity([1000, 1000, 1000, 90_000])
    )


def test_dispersity_ignores_non_positive_and_non_int_entries():
    run = SimpleNamespace(agent_context_lengths=[100, 0, -5, "x", 300])
    assert _agent_context_lengths(run) == [100, 300]


def test_agent_context_lengths_missing_attribute_is_empty():
    assert _agent_context_lengths(SimpleNamespace()) == []


# ── penalty applied to qd_score ───────────────────────────────────────────


def _pressure(dispersity_lambda: float) -> SelectionPressure:
    return SelectionPressure(
        config={},
        strategy="qd",
        novelty_weight=0.25,
        context_dispersity_lambda=dispersity_lambda,
    )


def test_zero_lambda_leaves_qd_score_unchanged():
    """Default configuration must not change any score."""
    pressure = _pressure(0.0)
    without = pressure._compose_qd_score(0.8, 0.5, 100, 1.0, 0.0)
    with_dispersity = pressure._compose_qd_score(0.8, 0.5, 100, 1.0, 0.9)
    assert without == with_dispersity


def test_positive_lambda_penalises_concentrated_context():
    pressure = _pressure(0.2)
    balanced = pressure._compose_qd_score(0.8, 0.5, 100, 1.0, 0.0)
    concentrated = pressure._compose_qd_score(0.8, 0.5, 100, 1.0, 1.0)
    assert concentrated < balanced
    assert balanced - concentrated == pytest.approx(0.2)


def test_penalty_is_disabled_unless_the_lambda_is_passed():
    """Callers that never heard of the knob (including dict configs) stay unpenalised."""
    assert SelectionPressure(config={}, strategy="qd").dispersity_lambda == 0.0


# ── reading lengths from agent memory ─────────────────────────────────────


def _write_agent(folder: Path, name: str, steps: list) -> None:
    (folder / name).write_text(json.dumps(steps))


def test_reader_takes_the_final_step_input_tokens_per_agent():
    with tempfile.TemporaryDirectory() as root:
        run = Path(root) / "uuid-1"
        run.mkdir()
        _write_agent(run, "task_a.json", [
            {"token_usage": {"input_tokens": 100}},
            {"token_usage": {"input_tokens": 450}},
        ])
        _write_agent(run, "task_b.json", [{"token_usage": {"input_tokens": 60}}])

        assert read_agent_context_lengths(root, "uuid-1") == [450, 60]


def test_reader_skips_unreadable_and_untokened_agents():
    with tempfile.TemporaryDirectory() as root:
        run = Path(root) / "uuid-1"
        run.mkdir()
        _write_agent(run, "task_ok.json", [{"token_usage": {"input_tokens": 10}}])
        _write_agent(run, "task_none.json", [{"token_usage": {}}])
        (run / "task_broken.json").write_text("{not json")
        (run / "verifier_x.json").write_text(json.dumps([{"token_usage": {"input_tokens": 999}}]))

        assert read_agent_context_lengths(root, "uuid-1") == [10]


def test_reader_returns_empty_for_missing_run():
    with tempfile.TemporaryDirectory() as root:
        assert read_agent_context_lengths(root, "absent") == []
        assert read_agent_context_lengths("", "uuid") == []


if __name__ == "__main__":
    test_dispersity_is_zero_for_evenly_shared_context()
    test_dispersity_grows_when_one_agent_hoards_context()
    test_zero_lambda_leaves_qd_score_unchanged()
    test_positive_lambda_penalises_concentrated_context()
    test_reader_takes_the_final_step_input_tokens_per_agent()
    print("context dispersity tests passed")

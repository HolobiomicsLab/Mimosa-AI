"""Speaking a step's result must not be able to fail the step.

Observed on the p_iimn ASB run of 2026-08-22 (uuid ``20260822_183340_88cdd1fe``):
the step scored 0.760, wrote ``reproduction_iimn.md``, and exported its ASTRA
capsule — then the run died with

    ❌ Critical error in step execution: unhashable type: 'slice'

and was reported as a 0% success rate with a non-zero exit. The cause was the
TTS summary line, which sliced each element of ``final_answers``. Those
elements are dicts (``{"status": ..., "approach": ...}``), not strings, so
``x[:128]`` indexed a dict with a slice.

Two independent guarantees are needed, and this file asserts both: coerce
before slicing (as every other consumer of ``answers`` already does), and keep
narration out of the step's success path entirely.
"""

import logging
from types import SimpleNamespace

import pytest

from sources.core.planner import Planner


# The exact shape recovered from that run's state_result.json.
REAL_ANSWERS = [
    {
        "status": "SUCCESS",
        "approach": "Systematic verification strategy combining literature retrieval "
                    "and line-by-line checking of the reproduction document.",
        "evidence": ["reproduction_iimn.md"],
    }
]


class _RecordingTTS:
    def __init__(self, explode: bool = False):
        self.spoken: list[str] = []
        self._explode = explode

    def speak(self, text, voice_index=0):
        if self._explode:
            raise RuntimeError("audio device gone")
        self.spoken.append(text)


def _planner(tts):
    """A Planner with only what narration touches — no LLM, no workspace."""
    planner = Planner.__new__(Planner)
    planner.tts = tts
    planner.logger = logging.getLogger("planner_narration_test")
    return planner


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------

def test_structured_answers_do_not_raise():
    """The regression: dict answers used to raise on the slice."""
    tts = _RecordingTTS()
    _planner(tts)._narrate_step_completion("discovery", 0.76, 12.0, REAL_ANSWERS)
    assert len(tts.spoken) == 1
    assert "0.76" in tts.spoken[0]
    assert "SUCCESS" in tts.spoken[0]  # the dict was rendered, not dropped


def test_the_bare_slice_is_gone_from_the_source():
    """Guard the specific expression, so it cannot be reintroduced verbatim."""
    import inspect
    src = inspect.getsource(Planner._narrate_step_completion)
    assert "str(x)[:128]" in src
    assert "[x[:128]" not in src


def test_each_answer_is_truncated_after_coercion():
    tts = _RecordingTTS()
    long_answer = {"status": "SUCCESS", "detail": "x" * 5000}
    _planner(tts)._narrate_step_completion("s", 1.0, 0.0, [long_answer])
    # 128 chars of the coerced dict, and nothing like the full 5000.
    assert len(tts.spoken[0]) < 400


@pytest.mark.parametrize("answers", [
    REAL_ANSWERS,
    ["a plain string answer"],
    [{"a": 1}, "mixed", 42, None, ""],
    [],
    None,
])
def test_no_answer_shape_can_raise(answers):
    _planner(_RecordingTTS())._narrate_step_completion("s", 0.5, 1.0, answers or [])


def test_empty_answers_are_narrated_explicitly():
    tts = _RecordingTTS()
    _planner(tts)._narrate_step_completion("s", 0.0, 0.0, [])
    assert "No answers produced" in tts.spoken[0]


# ---------------------------------------------------------------------------
# Narration is cosmetic — it must never propagate
# ---------------------------------------------------------------------------

def test_a_failing_tts_engine_does_not_fail_the_step(caplog):
    with caplog.at_level(logging.ERROR):
        _planner(_RecordingTTS(explode=True))._narrate_step_completion("s", 0.9, 3.0, REAL_ANSWERS)
    # Swallowed, but not silently: the traceback is on the record.
    assert "TTS narration failed" in caplog.text
    assert "audio device gone" in caplog.text


def test_narration_is_skipped_when_tts_is_disabled():
    _planner(None)._narrate_step_completion("s", 0.9, 3.0, REAL_ANSWERS)


def test_run_attempts_delegates_instead_of_inlining_the_summary():
    """The step body must not rebuild the summary string itself."""
    import inspect
    src = inspect.getsource(Planner.run_attempts)
    assert "_narrate_step_completion" in src
    assert "[:128]" not in src


def test_narration_runs_after_the_step_result_is_recorded():
    """Score and cost are on the step before anything is spoken."""
    import inspect
    src = inspect.getsource(Planner.run_attempts)
    assert src.index("step.score = attempt_score") < src.index("_narrate_step_completion")

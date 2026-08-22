"""An unattended run must not ask a question it cannot hear the answer to.

On the p_iimn run of 2026-08-22 the planner reached step 4 of 6, found a
dependency it believed unsatisfied, and called ``request_user_exit`` — which
printed "Continue ? (y(yes)/n(no))" into a redirected stdout and died with

    ❌ Planner: Execution failed: EOF when reading a line

The message names neither the question nor the step. It was the last blocking
``input()`` on the benchmark path; ``pricing.py`` and
``csv_mode._prompt_with_default`` had already been made headless-safe.

The dependency was not really unsatisfied. ``data_acquisition`` declared
``/workspace/data/`` as its output and wrote nine files into it, but the
workspace scan returns files, so a directory could never match — the step was
permanently "missing outputs" and blocked everything downstream. Both defects
are covered here: the question that could not be asked, and the reason it was
asked at all.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from sources.core.planner import Planner, UserInterventionRequired
from sources.core.schema import PlanStep, Task, TaskStatus


class _Notifier:
    def __init__(self):
        self.sent = []

    def send_message(self, msg, title=None, priority=0):
        self.sent.append((title, msg))


def _planner(workspace: str = "/tmp", history=None) -> Planner:
    p = object.__new__(Planner)
    p.workspace_path = workspace
    p.task_history = history or []
    p.notifier = _Notifier()
    import logging
    p.logger = logging.getLogger("headless_planner_test")
    return p


# ── the prompt ───────────────────────────────────────────────────────────────

def test_a_non_tty_run_raises_the_question_instead_of_reading_stdin():
    p = _planner()
    with patch("sys.stdin.isatty", return_value=False), \
         patch("builtins.input", side_effect=AssertionError("stdin must not be read")):
        with pytest.raises(UserInterventionRequired) as err:
            p.request_user_exit("Cannot execute step 'x' — missing dependencies: ['y']")

    assert "missing dependencies" in str(err.value), "the question must survive in the error"
    assert "not a TTY" in str(err.value)


def test_the_operator_is_still_notified_before_the_run_gives_up():
    p = _planner()
    with patch("sys.stdin.isatty", return_value=False):
        with pytest.raises(UserInterventionRequired):
            p.request_user_exit("blocked on step 4")
    assert p.notifier.sent and "blocked on step 4" in p.notifier.sent[0][1]


def test_it_raises_rather_than_exiting_so_the_harness_still_reports():
    """SystemExit from inside the planner would skip the CSV summary."""
    p = _planner()
    with patch("sys.stdin.isatty", return_value=False):
        with pytest.raises(UserInterventionRequired):
            p.request_user_exit("blocked")
        # Specifically NOT SystemExit.
        try:
            p.request_user_exit("blocked")
        except SystemExit:  # pragma: no cover - would be the regression
            pytest.fail("request_user_exit must not exit the process on a non-TTY")
        except UserInterventionRequired:
            pass


@pytest.mark.parametrize("answer,continues", [("y", True), ("yes", True), ("YES", True)])
def test_an_interactive_yes_still_continues(answer, continues):
    p = _planner()
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", return_value=answer):
        assert p.request_user_exit("carry on?") is None


def test_an_interactive_no_still_exits():
    p = _planner()
    with patch("sys.stdin.isatty", return_value=True), patch("builtins.input", return_value="n"):
        with pytest.raises(SystemExit):
            p.request_user_exit("stop?")


# ── directory outputs ────────────────────────────────────────────────────────

def _step(expected_outputs: list[str]) -> PlanStep:
    return PlanStep(name="s", goal_context="c", task="t", cost=0, score=0.0,
                    expected_outputs=expected_outputs)


def test_a_declared_directory_is_satisfied_by_the_files_inside_it():
    with tempfile.TemporaryDirectory() as ws:
        (Path(ws) / "data").mkdir()
        (Path(ws) / "data" / "MSV000080492_features.csv").write_text("a\n1\n")
        ok, missing = _planner(ws)._verify_expected_outputs(_step(["/workspace/data/"]))
    assert ok is True and missing == []


def test_a_declared_directory_that_stayed_empty_is_still_missing():
    with tempfile.TemporaryDirectory() as ws:
        (Path(ws) / "elsewhere.txt").write_text("noise")
        ok, missing = _planner(ws)._verify_expected_outputs(_step(["/workspace/data/"]))
    assert ok is False and missing == ["/workspace/data/"]


def test_a_nested_file_satisfies_the_parent_directory():
    with tempfile.TemporaryDirectory() as ws:
        (Path(ws) / "data" / "MSV000080492_raw").mkdir(parents=True)
        (Path(ws) / "data" / "MSV000080492_raw" / "LIMITATION.txt").write_text("blocked")
        ok, _ = _planner(ws)._verify_expected_outputs(_step(["/workspace/data/"]))
    assert ok is True


def test_a_directory_and_a_file_output_are_checked_independently():
    """The real declaration: ['/workspace/data/', '/workspace/data/inventory.csv']."""
    with tempfile.TemporaryDirectory() as ws:
        (Path(ws) / "data").mkdir()
        (Path(ws) / "data" / "features.csv").write_text("a\n1\n")
        ok, missing = _planner(ws)._verify_expected_outputs(
            _step(["/workspace/data/", "/workspace/data/dataset_inventory.csv"]))
    assert ok is False
    assert missing == ["/workspace/data/dataset_inventory.csv"], \
        "the directory is satisfied; only the named file is missing"


def test_file_outputs_are_unaffected_by_the_directory_rule():
    with tempfile.TemporaryDirectory() as ws:
        (Path(ws) / "reproduction_iimn.md").write_text("x")
        ok, _ = _planner(ws)._verify_expected_outputs(_step(["/workspace/reproduction_iimn.md"]))
    assert ok is True


@pytest.mark.parametrize("declared", ["/workspace/data/", "data/", "\\\\workspace\\\\data\\\\"])
def test_directory_spellings(declared):
    with tempfile.TemporaryDirectory() as ws:
        (Path(ws) / "data").mkdir()
        (Path(ws) / "data" / "f.csv").write_text("a\n1\n")
        ok, _ = _planner(ws)._verify_expected_outputs(_step([declared]))
    assert ok is True

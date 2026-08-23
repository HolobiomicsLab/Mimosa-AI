"""A batch run whose rows all failed must not report success.

``run_single_thread_eval_loop`` catches per-row exceptions and continues, which
is correct for a batch — one bad row should not abandon the rest — and records
``success_level: "Error"`` in ``execution_history``. Nothing downstream read
that, so the process exited 0.

Observed: a task that logged ``Error in csv row 1: Planner: Execution failed``
twice still exited 0. Any harness reading the exit code scores that as a pass.
"""

import asyncio
from types import SimpleNamespace

import pytest

from sources.benchmark_evaluation.csv_mode import CsvEvaluationMode


def _mode(history):
    """A CsvEvaluationMode with a pre-seeded history and no __init__ side effects."""
    mode = object.__new__(CsvEvaluationMode)
    mode.execution_history = history
    return mode


def test_no_rows_means_no_errors():
    assert _mode([]).errored_rows == []


def test_all_successful_rows_report_no_errors():
    history = [
        {"iteration": 1, "success_level": "Success"},
        {"iteration": 2, "success_level": "Partial"},
    ]
    assert _mode(history).errored_rows == []


def test_errored_rows_are_collected():
    history = [
        {"iteration": 1, "success_level": "Success"},
        {"iteration": 2, "success_level": "Error", "key_insight": "Planner failed"},
        {"iteration": 3, "success_level": "Error", "key_insight": "timed out"},
    ]
    errored = _mode(history).errored_rows
    assert len(errored) == 2
    assert [d["iteration"] for d in errored] == [2, 3]


def test_cached_rows_are_not_errors():
    """Cached rows are skipped work, not failures."""
    history = [{"iteration": 1, "success_level": "Cached"}]
    assert _mode(history).errored_rows == []


def test_papers_mode_exits_non_zero_when_a_row_failed(monkeypatch):
    import main

    class _Papers:
        def __init__(self, *a, **kw):
            self.execution_history = [{"iteration": 1, "success_level": "Error"}]

        async def start_evaluation(self, **kw):
            return None

        @property
        def errored_rows(self):
            return [d for d in self.execution_history
                    if d.get("success_level") == "Error"]

    monkeypatch.setattr(main, "CsvEvaluationMode", _Papers)
    args = SimpleNamespace(csv_runs_limit=1, papers="x.csv", learn=False,
                           single_agent=False)

    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(main.papers_mode(args, object()))
    assert excinfo.value.code == 1


def test_papers_mode_returns_normally_when_every_row_succeeded(monkeypatch):
    import main

    class _Papers:
        def __init__(self, *a, **kw):
            self.execution_history = [{"iteration": 1, "success_level": "Success"}]

        async def start_evaluation(self, **kw):
            return None

        @property
        def errored_rows(self):
            return []

    monkeypatch.setattr(main, "CsvEvaluationMode", _Papers)
    args = SimpleNamespace(csv_runs_limit=1, papers="x.csv", learn=False,
                           single_agent=False)

    asyncio.run(main.papers_mode(args, object()))  # must not raise

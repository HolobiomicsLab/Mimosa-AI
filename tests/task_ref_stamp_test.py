"""Unit tests for csv_mode's task_ref.json stamping.

The stamp restores the run -> benchmark-task join (ASB Challenge/TaskID CSV
columns) that csv_mode used to discard. Instances are built with __new__ so
no engine/planner (and no LLM stack beyond the import) is constructed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.benchmark_evaluation.csv_mode import CsvEvaluationMode

_ROW = {
    "Title": "Paper X",
    "URLS": "/Users/someone/private/paper.pdf",  # must NEVER be copied
    "Prompt": "Reproduce the experiments.",
    "Challenge": "challenge_03",
    "TaskID": "task_007",
}


def _mode(workflow_dir: Path | None) -> CsvEvaluationMode:
    mode = CsvEvaluationMode.__new__(CsvEvaluationMode)
    mode.config = SimpleNamespace(
        workflow_dir=str(workflow_dir) if workflow_dir else None
    )
    mode.logger = logging.getLogger("task_ref_test")
    return mode


def test_stamps_task_ref_into_every_existing_run_dir(tmp_path: Path) -> None:
    for run_uuid in ("run-a", "run-b"):
        (tmp_path / run_uuid).mkdir()
    mode = _mode(tmp_path)

    mode._stamp_task_ref(_ROW, 4, ["run-a", "run-b", "run-a"])  # dup collapses

    for run_uuid in ("run-a", "run-b"):
        stamped = json.loads((tmp_path / run_uuid / "task_ref.json").read_text())
        assert stamped == {
            "challenge": "challenge_03", "task_id": "task_007", "csv_row": 4,
        }


def test_never_copies_the_urls_column(tmp_path: Path) -> None:
    (tmp_path / "run-a").mkdir()
    _mode(tmp_path)._stamp_task_ref(_ROW, 0, ["run-a"])
    content = (tmp_path / "run-a" / "task_ref.json").read_text()
    assert "/Users/" not in content and "paper.pdf" not in content


def test_rows_without_asb_columns_are_not_stamped(tmp_path: Path) -> None:
    (tmp_path / "run-a").mkdir()
    row = {"Title": "Plain paper", "URLS": "http://x", "Prompt": "p"}
    _mode(tmp_path)._stamp_task_ref(row, 0, ["run-a"])
    assert not (tmp_path / "run-a" / "task_ref.json").exists()


def test_missing_run_dir_is_logged_never_fatal(tmp_path: Path, caplog) -> None:
    (tmp_path / "run-a").mkdir()
    mode = _mode(tmp_path)
    with caplog.at_level(logging.WARNING, logger="task_ref_test"):
        mode._stamp_task_ref(_ROW, 1, ["vanished-run", "run-a"])
    assert "vanished-run" in caplog.text
    assert (tmp_path / "run-a" / "task_ref.json").exists()


def test_no_workflow_dir_configured_is_a_silent_no_op(tmp_path: Path) -> None:
    _mode(None)._stamp_task_ref(_ROW, 0, ["run-a"])  # must not raise


def test_run_uuids_from_collects_evolution_runs_in_order() -> None:
    runs = [SimpleNamespace(current_uuid="u1"), SimpleNamespace(current_uuid="u2"),
            SimpleNamespace(current_uuid=None)]
    assert CsvEvaluationMode._run_uuids_from(runs=runs) == ["u1", "u2"]


def test_run_uuids_from_collects_planner_task_runs_and_final_uuid() -> None:
    tasks = [SimpleNamespace(
        evolve_runs=[SimpleNamespace(current_uuid="u1")],
        final_uuid="u1",
    ), SimpleNamespace(evolve_runs=None, final_uuid="u2")]
    # Duplicates survive here; the stamping loop dedupes with dict.fromkeys.
    assert CsvEvaluationMode._run_uuids_from(tasks=tasks) == ["u1", "u1", "u2"]


def test_run_uuids_from_handles_nothing() -> None:
    assert CsvEvaluationMode._run_uuids_from() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""Tests for the run-notes cache restore in CsvEvaluationMode.

Covers the explicit ``restore_notes_file`` path added for the evaluation
CLI: an operator-picked JSON (from ``run_notes/evaluations/``) must be
normalised (``final_results`` summary shape → flat per-task shape) and
restored into ``execution_history`` without any interactive prompt, while
``restore_cache=False`` and the automatic-scan behaviour stay unchanged.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.benchmark_evaluation import csv_mode
from sources.benchmark_evaluation.csv_mode import CsvEvaluationMode

MODEL = "openrouter/deepseek/deepseek-v3.2"


def _bare_evaluator() -> CsvEvaluationMode:
    """CsvEvaluationMode instance without running the heavy __init__."""
    evaluator = object.__new__(CsvEvaluationMode)
    evaluator.logger = logging.getLogger("tests.csv_mode_cache_restore")
    evaluator.config = SimpleNamespace(smolagent_model_id=MODEL)
    evaluator.execution_history = []
    evaluator.restore_notes_file = None
    return evaluator


def _write_notes(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _evaluation_summary() -> dict:
    """Shape written by EvaluationCLI into run_notes/evaluations/*.json."""
    return {
        "smolagent_model_id": MODEL,
        "eval_mode": "iterative",
        "status": "completed",
        "final_results": {
            "steps_evaluated": 10,
            "ver_success": 8,
            "ver_total": 9,
            "sr_success": 3,
            "sr_total": 9,
            "avg_cbs": 0.926,
            "total_cost": 17.94,
        },
    }


def _per_task_notes() -> dict:
    """Shape written by csv_mode into run_notes/<run>/<task>.json."""
    return {
        "model": MODEL,
        "total_eval": 4,
        "ver_success": 2,
        "sr_success": 1,
        "avg_cbs": 0.5,
        "total_cost": 2.0,
    }


def test_load_run_notes_from_file_normalises_evaluation_summary(tmp_path):
    path = _write_notes(
        tmp_path / "20260915_model_iterative.json", _evaluation_summary()
    )
    evaluator = _bare_evaluator()

    notes = evaluator._load_run_notes_from_file(path)

    assert notes is not None
    assert notes["total_eval"] == 9  # ver_total, not steps_evaluated
    assert notes["ver_success"] == 8
    assert notes["sr_success"] == 3
    assert notes["avg_cbs"] == 0.926
    assert notes["total_cost"] == 17.94
    assert notes["model"] == MODEL


def test_load_run_notes_from_file_keeps_flat_per_task_shape(tmp_path):
    path = _write_notes(tmp_path / "task.json", _per_task_notes())
    evaluator = _bare_evaluator()

    notes = evaluator._load_run_notes_from_file(path)

    assert notes is not None
    assert notes["total_eval"] == 4
    assert notes["ver_success"] == 2
    assert notes["model"] == MODEL


def test_load_run_notes_from_file_without_stats_returns_none(tmp_path):
    path = _write_notes(tmp_path / "empty.json", {"status": "running"})
    evaluator = _bare_evaluator()

    assert evaluator._load_run_notes_from_file(path) is None


def test_load_run_notes_from_file_unreadable_returns_none(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    evaluator = _bare_evaluator()

    assert evaluator._load_run_notes_from_file(path) is None


def test_load_run_notes_from_file_model_mismatch_still_loads(tmp_path):
    data = _evaluation_summary()
    data["smolagent_model_id"] = "other/model"
    path = _write_notes(tmp_path / "mismatch.json", data)
    evaluator = _bare_evaluator()

    notes = evaluator._load_run_notes_from_file(path)

    assert notes is not None  # explicit choice: warn but restore


def test_maybe_restore_cache_explicit_file_restores_without_prompt(
    tmp_path, monkeypatch
):
    path = _write_notes(tmp_path / "summary.json", _evaluation_summary())
    evaluator = _bare_evaluator()
    evaluator.restore_notes_file = path

    def _no_auto_scan():
        raise AssertionError("automatic scan must not run when a file is set")

    monkeypatch.setattr(evaluator, "_load_previous_run_notes", _no_auto_scan)

    asyncio.run(evaluator._maybe_restore_cache(True))

    assert len(evaluator.execution_history) == 9
    assert sum(1 for e in evaluator.execution_history if e["VER"]) == 8
    assert sum(1 for e in evaluator.execution_history if e["SR"]) == 3


def test_maybe_restore_cache_explicit_file_refused_when_false(tmp_path):
    path = _write_notes(tmp_path / "summary.json", _evaluation_summary())
    evaluator = _bare_evaluator()
    evaluator.restore_notes_file = path

    asyncio.run(evaluator._maybe_restore_cache(False))

    assert evaluator.execution_history == []


def test_maybe_restore_cache_auto_detect_restores(tmp_path, monkeypatch):
    evaluator = _bare_evaluator()
    monkeypatch.setattr(
        evaluator, "_load_previous_run_notes", lambda: _per_task_notes()
    )

    asyncio.run(evaluator._maybe_restore_cache(True))

    assert len(evaluator.execution_history) == 4


def test_maybe_restore_cache_auto_detect_prompts_when_unset(tmp_path, monkeypatch):
    evaluator = _bare_evaluator()
    monkeypatch.setattr(
        evaluator, "_load_previous_run_notes", lambda: _per_task_notes()
    )

    async def _decline(prompt, default="y"):
        return "n"

    monkeypatch.setattr(csv_mode, "_prompt_with_default", _decline)

    asyncio.run(evaluator._maybe_restore_cache(None))

    assert evaluator.execution_history == []

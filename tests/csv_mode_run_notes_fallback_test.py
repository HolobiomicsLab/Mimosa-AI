#!/usr/bin/env python3
"""Regression tests for the excluded-task lineage fallback in run notes.

Bug: when the current task was infra-excluded (``VER is None``),
``CsvEvaluationMode._save_run_notes`` fell back to ``sab_runs[-1]`` — the
previously *evaluated* task's execution data — and copied its runs, uuids,
rewards and ablations into the excluded task's note (proven: WaterQuality's
note carried tin_tungsten's ablations byte-identically).

Fix: when ``current_execution_data`` is provided it ALWAYS defines the
per-task note fields, excluded or not, so an excluded task records its own
lineage and ablations. The ``sab_runs[-1]`` fallback only triggers when no
execution data is given at all (legacy callers), and then logs a loud
warning naming the task so silent foreign-lineage copies can never happen
again.

All tests are offline: they build a bare ``CsvEvaluationMode`` (no
``__init__``) with a temporary ``run_notes_dir`` and never touch the LLM,
MCP servers or the network.
"""

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.benchmark_evaluation.csv_mode import CsvEvaluationMode
from sources.core.schema import IndividualRun

MODEL = "openrouter/deepseek/deepseek-v3.2"
LOGGER_NAME = "tests.csv_mode_run_notes_fallback"


def _bare_evaluator(run_notes_dir: Path) -> CsvEvaluationMode:
    """CsvEvaluationMode instance without running the heavy __init__."""
    evaluator = object.__new__(CsvEvaluationMode)
    evaluator.logger = logging.getLogger(LOGGER_NAME)
    evaluator.config = SimpleNamespace(smolagent_model_id=MODEL)
    evaluator.execution_history = []
    evaluator.run_notes_dir = run_notes_dir
    evaluator._start_row = 0
    # Keep the test hermetic: no git subprocess, deterministic note content.
    evaluator._get_git_info = lambda: {"commit": None, "branch": None, "dirty": None}
    return evaluator


def _make_runs(uuids: list[str], rewards: list[float]) -> list[IndividualRun]:
    """IndividualRun list; costs are cumulative like the evolution engine's."""
    return [
        IndividualRun(
            goal="g",
            prompt="p",
            current_uuid=u,
            reward=r,
            iteration_count=i,
            cost=0.1 * (i + 1),
        )
        for i, (u, r) in enumerate(zip(uuids, rewards, strict=True))
    ]


def _evaluated_data(capsule: str, uuids: list[str], rewards: list[float]) -> dict:
    """Execution data shape written by _evaluate_with_science_agent_bench."""
    return {
        "iteration": 1,
        "goal": f"goal for {capsule}",
        "status": "evaluated",
        "VER": True,
        "VER_message": "ver ok",
        "SR": True,
        "SR_message": "sr ok",
        "CBS": 0.8,
        "eval_cost": 1.5,
        "runs": _make_runs(uuids, rewards),
        "success_level": "Success",
        "ablations": [
            {
                "evolution_index": i,
                "uuid": u,
                "VER": True,
                "SR": True,
                "CBS": 0.8,
                "cost": 0.1,
                "status": "evaluated",
                "source": "snapshot",
            }
            for i, u in enumerate(uuids)
        ],
    }


def _excluded_data(capsule: str, uuids: list[str], rewards: list[float]) -> dict:
    """Execution data shape of an infra-excluded task (VER is None).

    ``runs`` is still set (both branches of _evaluate_with_science_agent_bench
    set it) and ``ablations`` holds the excluded-status entries computed by
    _evaluate_snapshot_ablations — the honest outcome for this task.
    """
    return {
        "iteration": 2,
        "goal": f"goal for {capsule}",
        "status": "excluded",
        "infra_error": "dataset file missing",
        "VER": None,
        "SR": None,
        "CBS": None,
        "eval_cost": 0.2,
        "runs": _make_runs(uuids, rewards),
        "success_level": "Excluded",
        "ablations": [
            {
                "evolution_index": i,
                "uuid": u,
                "VER": None,
                "SR": None,
                "CBS": None,
                "cost": 0.0,
                "status": "excluded",
                "source": "snapshot",
                "infra_error": "dataset file missing",
            }
            for i, u in enumerate(uuids)
        ],
    }


def _read_note(run_notes_dir: Path, capsule: str) -> dict:
    return json.loads((run_notes_dir / f"{capsule}.json").read_text("utf-8"))


def test_excluded_task_records_own_lineage(tmp_path, caplog):
    """Concurrent mode: an excluded task must record ITS OWN uuids/ablations.

    Regression for the proven corruption (WaterQuality's note carrying
    tin_tungsten's ablations): the excluded task's data is provided, so it
    must win over the previously evaluated task's.
    """
    prev = _evaluated_data(
        "tin_tungsten",
        ["20260902_102930_aaaaaaaa", "20260902_103100_bbbbbbbb"],
        [0.2, 0.9],
    )
    excluded = _excluded_data(
        "WaterQuality",
        ["20260902_110000_cccccccc", "20260902_110500_dddddddd"],
        [0.0, 0.1],
    )
    evaluator = _bare_evaluator(tmp_path)
    evaluator.execution_history = [prev]  # concurrent mode: current not yet in history

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        evaluator._save_run_notes(
            "WaterQuality",
            excluded["goal"],
            10.0,
            current_execution_data=excluded,
        )

    note = _read_note(tmp_path, "WaterQuality")
    own_uuids = ["20260902_110000_cccccccc", "20260902_110500_dddddddd"]
    prev_uuids = ["20260902_102930_aaaaaaaa", "20260902_103100_bbbbbbbb"]

    # (a) The note records the excluded task's OWN lineage and ablations.
    assert note["evolved_workflows_uuids"] == own_uuids
    assert note["evolution_rewards"] == [0.0, 0.1]
    assert note["evolution_iterations"] == 2
    assert note["capsule_name"] == "WaterQuality"
    assert note["task_cost"] == 0.2
    assert all(entry["status"] == "excluded" for entry in note["ablations"])
    assert [entry["uuid"] for entry in note["ablations"]] == own_uuids

    # (b) The previous task's data is never copied.
    assert note["evolved_workflows_uuids"] != prev_uuids
    assert all(u not in note["evolved_workflows_uuids"] for u in prev_uuids)
    assert note["ablations"] != prev["ablations"]

    # Excluded tasks stay out of the aggregate counters.
    assert note["total_eval"] == 1
    assert note["ver_success"] == 1
    assert note["sr_success"] == 1

    # Providing the data must not trigger the legacy-fallback warning.
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_excluded_task_in_history_records_own_lineage(tmp_path, caplog):
    """Sequential mode: same invariant when the excluded task was already
    appended to execution_history (the caller now passes its data)."""
    prev = _evaluated_data("tin_tungsten", ["20260902_102930_aaaaaaaa"], [0.9])
    excluded = _excluded_data("WaterQuality", ["20260902_110000_cccccccc"], [0.0])
    evaluator = _bare_evaluator(tmp_path)
    evaluator.execution_history = [prev, excluded]

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        evaluator._save_run_notes(
            "WaterQuality",
            excluded["goal"],
            10.0,
            current_execution_data=excluded,
        )

    note = _read_note(tmp_path, "WaterQuality")
    assert note["evolved_workflows_uuids"] == ["20260902_110000_cccccccc"]
    assert note["ablations"][0]["uuid"] == "20260902_110000_cccccccc"
    assert note["ablations"][0]["status"] == "excluded"
    assert note["total_eval"] == 1  # the excluded task is not counted
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_evaluated_task_note_unchanged_when_data_passed(tmp_path):
    """Normal (non-excluded) path: passing the data must not change the note."""
    evaluated = _evaluated_data(
        "clintox_nn",
        ["20260902_102930_aaaaaaaa", "20260902_103100_bbbbbbbb"],
        [0.2, 0.9],
    )
    evaluator = _bare_evaluator(tmp_path)
    evaluator.execution_history = [evaluated]

    evaluator._save_run_notes(
        "clintox_nn",
        evaluated["goal"],
        5.0,
        current_execution_data=evaluated,
    )
    note_with_data = _read_note(tmp_path, "clintox_nn")

    evaluator._save_run_notes("clintox_nn", evaluated["goal"], 5.0)
    note_legacy = _read_note(tmp_path, "clintox_nn")

    per_task_keys = [
        "capsule_name",
        "ver_success",
        "sr_success",
        "avg_cbs",
        "total_cost",
        "is_success",
        "task_cost",
        "max_judge_reward",
        "evolution_iterations",
        "evolved_workflows_uuids",
        "evolution_rewards",
        "evolution_costs",
        "evolution_total_cost",
        "evolution_avg_reward",
        "evolution_avg_cost",
        "ablations",
    ]
    assert note_with_data["evolved_workflows_uuids"] == [
        "20260902_102930_aaaaaaaa",
        "20260902_103100_bbbbbbbb",
    ]
    for key in per_task_keys:
        assert note_with_data[key] == note_legacy[key], key


def test_legacy_fallback_without_execution_data_warns(tmp_path, caplog):
    """(c) Legacy path: no execution data → sab_runs[-1] is used (backward
    compatibility) BUT a loud warning naming the task is logged."""
    prev = _evaluated_data("tin_tungsten", ["20260902_102930_aaaaaaaa"], [0.9])
    evaluator = _bare_evaluator(tmp_path)
    evaluator.execution_history = [prev]

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        evaluator._save_run_notes("WaterQuality", "goal for WaterQuality", 10.0)

    note = _read_note(tmp_path, "WaterQuality")
    # Backward-compatible behavior: the last evaluated run's lineage is kept.
    assert note["evolved_workflows_uuids"] == ["20260902_102930_aaaaaaaa"]
    assert note["ablations"] == prev["ablations"]

    # ... but it can never happen silently again.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "WaterQuality" in warnings[0].getMessage()
    assert "sab_runs" not in warnings[0].getMessage()  # operator-readable message

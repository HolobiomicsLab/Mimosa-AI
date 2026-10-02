#!/usr/bin/env python3
"""
Regression tests for the snapshot-ablation index mapping in CsvEvaluationMode.

Bug: ``_evaluate_snapshot_ablations`` recovered the run uuid from a snapshot
dir name with ``name.rsplit("_", 1)[-1]``. Run uuids contain underscores
(``YYYYMMDD_HHMMSS_<8hex>``, see sources/core/workflow_factory.py), so only
the trailing 8-hex fragment was recovered. It never matched
``run.current_uuid``, every snapshot fell into the orphan branch and was
numbered sequentially from ``len(runs)`` (e.g. idx 7..13), and the best run's
snapshot was re-evaluated instead of reusing the capsule metrics.

Fix: strip the known ``mimosa_run_<session_id>_`` prefix (session ids are
pure hex, no underscores) instead of splitting on "_".

Snapshot dirs are globbed from the real /tmp (hardcoded in csv_mode.py), so
tests create dirs there under a random session id and always clean up.
"""

import logging
import os
import shutil
import sys
import uuid as uuid_mod
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.benchmark_evaluation import csv_mode
from sources.benchmark_evaluation.csv_mode import CsvEvaluationMode
from sources.core.schema import IndividualRun


def _bare_evaluator() -> CsvEvaluationMode:
    """CsvEvaluationMode instance without running the heavy __init__."""
    evaluator = object.__new__(CsvEvaluationMode)
    evaluator.logger = logging.getLogger("tests.csv_mode_ablation_index")
    return evaluator


class _FakeCapsuleEvaluator:
    """Records construction and returns deterministic eval results."""

    calls: list[Path] = []

    def __init__(self, capsule_path, task_data, sab_loader, api_cost):
        self.capsule_path = Path(capsule_path)
        self.api_cost = api_cost
        _FakeCapsuleEvaluator.calls.append(self.capsule_path)

    def evaluate_all(self):
        return {
            "VER": (True, "ok"),
            "SR": (False, "not reproduced"),
            "CBS": 0.5,
            "cost": 0.01,
        }


def _make_session() -> str:
    return uuid_mod.uuid4().hex[:12]


def _make_snapshots(session_id: str, uuids: list[str]) -> list[Path]:
    dirs = []
    for u in uuids:
        d = Path(f"/tmp/mimosa_run_{session_id}_{u}")
        d.mkdir(parents=True)
        (d / "dummy.txt").write_text("x")
        dirs.append(d)
    return dirs


def _cleanup(session_id: str) -> None:
    for p in Path("/tmp").glob(f"mimosa_run_{session_id}_*"):
        shutil.rmtree(p, ignore_errors=True)


def _make_runs(uuids: list[str], rewards: list[float]) -> list[IndividualRun]:
    # IndividualRun.cost is cumulative across iterations.
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


def test_snapshot_indices_match_run_positions(monkeypatch):
    """Every snapshot must map to its true 0-based evolution index."""
    monkeypatch.setattr(csv_mode, "CapsuleEvaluator", _FakeCapsuleEvaluator)
    _FakeCapsuleEvaluator.calls = []
    session_id = _make_session()
    uuids = [
        "20260902_102930_aaaaaaaa",
        "20260902_103100_bbbbbbbb",
        "20260902_103500_cccccccc",
    ]
    try:
        _make_snapshots(session_id, uuids)
        runs = _make_runs(uuids, rewards=[0.2, 0.9, 0.5])
        execution_data = {"VER": True, "SR": True, "CBS": 0.9, "status": "evaluated"}

        ablations = _bare_evaluator()._evaluate_snapshot_ablations(
            session_id,
            row={},
            runs=runs,
            sab_loader=None,
            execution_data=execution_data,
        )

        assert [e["evolution_index"] for e in ablations] == [0, 1, 2]
        assert [e["uuid"] for e in ablations] == uuids
        # Per-iteration costs are deltas of the cumulative run costs.
        assert [e["cost"] for e in ablations] == pytest.approx([0.1, 0.1, 0.1])
    finally:
        _cleanup(session_id)


def test_best_snapshot_reuses_capsule_metrics(monkeypatch):
    """The best run's snapshot must not be re-evaluated."""
    monkeypatch.setattr(csv_mode, "CapsuleEvaluator", _FakeCapsuleEvaluator)
    _FakeCapsuleEvaluator.calls = []
    session_id = _make_session()
    uuids = [
        "20260902_102930_aaaaaaaa",
        "20260902_103100_bbbbbbbb",
        "20260902_103500_cccccccc",
    ]
    try:
        dirs = _make_snapshots(session_id, uuids)
        # Best run = highest reward: index 1.
        runs = _make_runs(uuids, rewards=[0.2, 0.9, 0.5])
        execution_data = {
            "VER": True,
            "VER_message": "capsule ok",
            "SR": True,
            "SR_message": "capsule reproduced",
            "CBS": 0.9,
            "status": "evaluated",
        }

        ablations = _bare_evaluator()._evaluate_snapshot_ablations(
            session_id,
            row={},
            runs=runs,
            sab_loader=None,
            execution_data=execution_data,
        )

        best = next(e for e in ablations if e["uuid"] == uuids[1])
        assert best["source"] == "capsule"
        assert best["evolution_index"] == 1
        assert (best["VER"], best["SR"], best["CBS"]) == (True, True, 0.9)
        # The evaluator ran for the other two snapshots only.
        evaluated = set(_FakeCapsuleEvaluator.calls)
        assert evaluated == {dirs[0], dirs[2]}
        others = [e for e in ablations if e["uuid"] != uuids[1]]
        assert all(e["source"] == "snapshot" for e in others)
    finally:
        _cleanup(session_id)


def test_generation_failed_uuid_parsed_in_full(monkeypatch):
    """``generation_failed`` must survive parsing (not truncated to 'failed')."""
    monkeypatch.setattr(csv_mode, "CapsuleEvaluator", _FakeCapsuleEvaluator)
    _FakeCapsuleEvaluator.calls = []
    session_id = _make_session()
    uuids = ["generation_failed", "20260902_103100_bbbbbbbb"]
    try:
        _make_snapshots(session_id, uuids)
        runs = _make_runs(uuids, rewards=[0.0, 0.9])
        execution_data = {"VER": True, "SR": True, "CBS": 0.9, "status": "evaluated"}

        ablations = _bare_evaluator()._evaluate_snapshot_ablations(
            session_id,
            row={},
            runs=runs,
            sab_loader=None,
            execution_data=execution_data,
        )

        assert [e["evolution_index"] for e in ablations] == [0, 1]
        assert ablations[0]["uuid"] == "generation_failed"
        assert ablations[1]["source"] == "capsule"
    finally:
        _cleanup(session_id)


def test_unknown_snapshot_stays_orphan(monkeypatch):
    """A snapshot whose uuid matches no run keeps the orphan fallback."""
    monkeypatch.setattr(csv_mode, "CapsuleEvaluator", _FakeCapsuleEvaluator)
    _FakeCapsuleEvaluator.calls = []
    session_id = _make_session()
    uuids = ["20260902_102930_aaaaaaaa", "20260902_103100_bbbbbbbb"]
    orphan = "20260902_109999_zzzzzzzz"
    try:
        _make_snapshots(session_id, uuids + [orphan])
        runs = _make_runs(uuids, rewards=[0.9, 0.2])
        execution_data = {"VER": True, "SR": True, "CBS": 0.9, "status": "evaluated"}

        ablations = _bare_evaluator()._evaluate_snapshot_ablations(
            session_id,
            row={},
            runs=runs,
            sab_loader=None,
            execution_data=execution_data,
        )

        orphan_entry = next(e for e in ablations if e["uuid"] == orphan)
        assert orphan_entry["evolution_index"] == len(runs)
        assert orphan_entry["cost"] == 0.0
        assert sorted(e["evolution_index"] for e in ablations) == [0, 1, 2]
    finally:
        _cleanup(session_id)

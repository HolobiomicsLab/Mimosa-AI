"""Tests for reading a Policy-BT verifier evaluation.txt into the Verification view."""

from __future__ import annotations

import json

import pytest
from app import store
from app.verification import REPORT_HEADER, read_policy_bt_report

OLD, NEW = "20261013_205210_06608c16", "20261013_210631_3b2c4c8e"

CODE_PAIR = {
    "winner": "a",
    "stage": 5,
    "feedback_a": "Your run reports min_energy -9; you win at stage 5.",
    "feedback_b": "Your run reports min_energy -8; search for -9.",
    "evidence": [
        {
            "criterion": "min_energy",
            "path": "energy.txt",
            "value": -9,
            "detail": "Reported minimum energy",
            "side": "a",
        }
    ],
    "channel": "code",
    "normalized_a": {"values": {"min_energy": -9}, "evidence": []},
    "normalized_b": {"values": {"min_energy": -8}, "evidence": []},
    "a": OLD,
    "b": NEW,
    "policy_version": "29b46f67",
}

EXECUTION = {
    "eligible": True,
    "reason": "Required shell script completed and declared outputs were regenerated.",
    "status": "succeeded",
    "script": "run.sh",
    "contract": {"script": "run.sh", "generated_outputs": ["energy.txt"]},
    "returncode": 0,
    "files": {"created": [], "changed": ["energy.txt"], "unchanged": ["a.py", "b.py"]},
    "run": {
        "argv": ["/bin/bash", "run.sh"],
        "returncode": 0,
        "stdout": {"path": "/x/stdout.log", "tail": "done\n"},
        "stderr": None,
    },
    "provision": None,
}


def _write_run(root, uuid, report_pairs, rewards=None):
    run = root / uuid
    run.mkdir()
    report = {
        "reward": 0.25,
        "tournament_version": 3,
        "execution": EXECUTION,
        "pairs": report_pairs,
    }
    (run / "evaluation.txt").write_text(
        f"{REPORT_HEADER}\n{json.dumps(report, indent=2)}", encoding="utf-8"
    )
    if rewards is not None:
        verifier = {
            "verifier_kind": "policy_bt",
            "reward": rewards[uuid],
            "arrival_reward": 0.25,
            "tournament_version": 4,
            "tournament_rewards": rewards,
        }
        (run / "state_result.json").write_text(
            json.dumps({"evaluation": {"verifier": verifier}}), encoding="utf-8"
        )
    return run


@pytest.fixture()
def workflows(tmp_path, monkeypatch):
    monkeypatch.setenv("MIMOSA_WORKFLOW_DIR", str(tmp_path))
    store.get_settings.cache_clear()
    yield tmp_path
    store.get_settings.cache_clear()


def test_pair_is_seen_from_the_viewed_side(workflows):
    rewards = {OLD: 0.75, NEW: 0.65}
    _write_run(workflows, OLD, [CODE_PAIR], rewards)
    _write_run(workflows, NEW, [CODE_PAIR], rewards)

    winner = read_policy_bt_report(workflows / OLD)["pairs"][0]
    loser = read_policy_bt_report(workflows / NEW)["pairs"][0]

    assert (winner["outcome"], loser["outcome"]) == ("win", "loss")
    assert loser["opponent"] == OLD and loser["opponent_reward"] == 0.75
    assert loser["feedback"] == CODE_PAIR["feedback_b"]
    assert loser["opponent_feedback"] == CODE_PAIR["feedback_a"]
    assert loser["values"] == {
        "self": {"min_energy": -8},
        "opponent": {"min_energy": -9},
    }
    assert [e["side"] for e in loser["evidence"]] == ["opponent"]
    assert [e["side"] for e in winner["evidence"]] == ["self"]


def test_live_rating_overrides_arrival_reward(workflows):
    _write_run(workflows, NEW, [CODE_PAIR], {OLD: 0.75, NEW: 0.65})
    report = read_policy_bt_report(workflows / NEW)
    assert report["reward"] == 0.65
    assert report["arrival_reward"] == 0.25
    assert report["tournament_version"] == 4


def test_report_reward_used_without_state_result(workflows):
    _write_run(workflows, NEW, [CODE_PAIR])
    report = read_policy_bt_report(workflows / NEW)
    assert report["reward"] == 0.25
    assert report["pairs"][0]["opponent_reward"] is None


def test_execution_gate_pair_drops_execution_evidence(workflows):
    gate = {
        "winner": "b",
        "stage": 0,
        "feedback_a": "Missing reproducibility .sh script.",
        "feedback_b": "Required shell script completed.",
        "channel": "execution",
        "evidence": [
            {"side": "a", "execution": {"eligible": False}},
            {"side": "b", "execution": {"eligible": True}},
        ],
        "a": OLD,
        "b": NEW,
        "policy_version": None,
    }
    _write_run(workflows, NEW, [gate])
    pair = read_policy_bt_report(workflows / NEW)["pairs"][0]
    assert pair["outcome"] == "win"
    assert pair["evidence"] == []
    assert pair["values"] is None


def test_execution_summary_collapses_unchanged_files(workflows):
    _write_run(workflows, NEW, [])
    execution = read_policy_bt_report(workflows / NEW)["execution"]
    assert execution["files"] == {
        "created": [],
        "changed": ["energy.txt"],
        "unchanged_count": 2,
    }
    assert execution["run"]["stdout"] == "done\n"
    assert execution["provision"] is None


def test_criteria_come_from_the_session_registry(workflows):
    _write_run(workflows, NEW, [])
    session = workflows / ".policy_bt" / "abc123"
    (session / "snapshots" / NEW).mkdir(parents=True)
    criteria = [
        {"id": "min_energy", "stage": 2, "direction": "lower"},
        {"id": "outputs_present", "stage": 1, "direction": "satisfy"},
    ]
    (session / "tournament.json").write_text(
        json.dumps({"bundle": {"criteria": criteria}}), encoding="utf-8"
    )
    ids = [c["id"] for c in read_policy_bt_report(workflows / NEW)["criteria"]]
    assert ids == ["outputs_present", "min_energy"]


def test_other_verifier_formats_return_none(workflows):
    run = workflows / NEW
    run.mkdir()
    (run / "evaluation.txt").write_text("Verifier Evaluation\n====\n", encoding="utf-8")
    assert read_policy_bt_report(run) is None
    assert read_policy_bt_report(workflows / OLD) is None


def test_run_detail_exposes_verification(workflows):
    _write_run(workflows, NEW, [CODE_PAIR])
    detail = store.run_detail(NEW)
    assert detail["verification"]["pairs"][0]["outcome"] == "loss"

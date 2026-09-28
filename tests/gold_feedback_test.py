#!/usr/bin/env python3
"""Offline tests for the gold-feedback (ORACLE / BENCHMARK-LEAKING) verifier.

No API key, no MCP server, no sandbox venv: the inner hybrid verifier and
the benchmark grader are replaced by fakes. Covers config plumbing, facade
selection, gradient assembly/compaction, gradient-only vs full-oracle
reward, fallback paths, private-copy grading and the behaviour-preserving
refactor of the snapshot-ablation grading call.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
import types
import uuid as uuid_mod
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from sources.benchmark_evaluation import csv_mode, snapshot_grading  # noqa: E402
from sources.benchmark_evaluation.csv_mode import CsvEvaluationMode  # noqa: E402
from sources.core.schema import IndividualRun  # noqa: E402
from sources.core.workflow_info import WorkflowInfo  # noqa: E402
from sources.evaluators import evaluator as evaluator_mod  # noqa: E402
from sources.evaluators import hybrid_verifier as hybrid_pkg  # noqa: E402
from sources.evaluators.gold_feedback import (  # noqa: E402
    LEAK_WARNING,
    MAX_GRADIENT_CHARS,
    GoldFeedbackEvaluator,
    build_gold_gradient,
)
from sources.evaluators.gold_feedback import format as fmt  # noqa: E402
from sources.evaluators.gold_feedback.reward import oracle_reward  # noqa: E402

ROW = {
    "instance_id": "7",
    "gold_program_name": "clintox_nn.py",
    "output_fname": "pred_results/clintox_test_pred.csv",
    "eval_script_name": "clintox_nn_eval.py",
}
HYBRID_GRADIENT = "HYBRID GRADIENT: claim C3 failed at stage result"
TRACEBACK = (
    "Generated code failed with code 1: Traceback (most recent call last):\n"
    + "".join(
        f'  File "/x/lib/mod{i}.py", line {i}, in f{i}\n    call()\n'
        for i in range(120)
    )
    + "ValueError: Found input variables with inconsistent numbers of samples: [292, 291]\n"
)
FIGURE_MSG = str(
    [
        "The first figure and the second figure differ. " * 40 + "\n[FINAL SCORE]: 50",
        "The generated plot misses the legend. " * 40 + "\n[FINAL SCORE]: 40",
        "Colours differ from the ground truth. " * 40 + "\n[FINAL SCORE]: 55",
    ]
)


def _mutator_view(text: str) -> str:
    """What variation_engine feeds the directive LLM (worst case: old clip)."""
    return text.strip().replace("_", " ")[:2048]


# --------------------------------------------------------------- fixtures


class _FakeHybrid:
    """Writes state_result + sidecars exactly like the hybrid verifier."""

    def __init__(
        self,
        workflow_dir: Path,
        workspace: Path,
        reward=0.42,
        fallback=None,
        short=False,
    ):
        self.workflow_dir = workflow_dir
        self.workspace_dir = workspace
        self.reward = reward
        self.fallback = fallback
        self.short = short
        self.n_pairs = 0
        self.calls = []

    def evaluate(self, uuid):
        self.calls.append(uuid)
        block = {
            "overall_score": 0.0 if self.short else self.reward,
            "overall_score_uncapped": 0.0 if self.short else 0.61,
            "reward_fallback": "short_circuit" if self.short else self.fallback,
            "abstracted_textual_gradient": HYBRID_GRADIENT,
            "abstractec_textual_gradient": HYBRID_GRADIENT,
            "n_pairs": self.n_pairs,
            "n_wins": self.n_pairs,
        }
        if self.short:
            block["skipped_reason"] = "workflow_generation_or_execution_failed"
        d = self.workflow_dir / uuid
        d.mkdir(parents=True, exist_ok=True)
        path = d / "state_result.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        state.setdefault("evaluation", {})["verifier"] = block
        path.write_text(json.dumps(state))
        (d / "evaluation.txt").write_text("Hybrid Verifier Evaluation\nreport body\n")
        (d / "textual_gradient.txt").write_text(HYBRID_GRADIENT)
        return {"uuid": uuid, "claims": [], **block}


class _FakeGrader:
    """Stands in for snapshot_grading.grade_directory."""

    def __init__(self, result=None, exc=None, write_outputs=True):
        self.result = result
        self.exc = exc
        self.write_outputs = write_outputs
        self.calls = []

    def __call__(self, directory, task_row, sab_loader):
        directory = Path(directory)
        self.calls.append(
            {
                "dir": directory,
                "row": task_row,
                "loader": sab_loader,
                "files": sorted(p.name for p in directory.rglob("*")),
            }
        )
        if self.write_outputs:  # mimics run_generated_code's copy-back
            (directory / "pred_results").mkdir(exist_ok=True)
            (directory / "pred_results" / "regenerated.csv").write_text("a,b\n")
        if self.exc:
            raise self.exc
        return dict(self.result)


def _evaluated(
    ver=True,
    ver_msg="Code executed. Output: done",
    sr=False,
    sr_msg="overlap: 0.0134",
    cbs=0.6,
):
    return {
        "VER": ver,
        "VER_message": ver_msg,
        "SR": sr,
        "SR_message": sr_msg,
        "CBS": cbs,
        "cost": 0.0,
        "status": "evaluated",
    }


def _setup(
    tmp_path,
    *,
    row=ROW,
    reward_flag=False,
    short=False,
    grader=None,
    fallback=None,
    timeout_s=1800,
    uuid="20260928_120000_abcdef12",
):
    workflow_dir = tmp_path / "workflows"
    workspace = tmp_path / "workspace"
    (workspace / "pred_results").mkdir(parents=True, exist_ok=True)
    (workspace / "clintox_nn.py").write_text("print('hi')\n")
    (workspace / "pred_results" / "clintox_test_pred.csv").write_text(
        "smiles,p\nC,0.1\n"
    )
    run = workflow_dir / uuid
    run.mkdir(parents=True)
    if not short:
        (run / f"workflow_genotype_{uuid}.py").write_text("# genotype\n")
        (run / "state_result.json").write_text(
            json.dumps({"goal": "g", "answers": ["x"]})
        )
    config = types.SimpleNamespace(
        workflow_dir=str(workflow_dir),
        workspace_dir=str(workspace),
        gold_feedback_reward=reward_flag,
        gold_feedback_task_row=dict(row) if row else None,
        gold_feedback_sab_loader="LOADER",
        gold_feedback_timeout_s=timeout_s,
    )
    hybrid = _FakeHybrid(workflow_dir, workspace, short=short, fallback=fallback)
    grader = grader or _FakeGrader(_evaluated())
    ev = GoldFeedbackEvaluator(config, hybrid=hybrid, grade_fn=grader)
    return ev, hybrid, grader, uuid, run, workspace


def _tree(root: Path) -> dict:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# --------------------------------------------------------------- config / CLI


def test_config_default_json_and_runtime_fields():
    c = Config()
    assert c.verifier_kind == "hybrid"
    assert c.gold_feedback_reward is False
    assert c.gold_feedback_task_row is None and c.gold_feedback_sab_loader is None
    c.from_json({"verifier_kind": "gold", "gold_feedback_reward": True})
    assert c.verifier_kind == "gold" and c.gold_feedback_reward is True
    dumped = c.jsonify()
    assert dumped["gold_feedback_reward"] is True
    # runtime-only benchmark context is never serialized
    assert "gold_feedback_task_row" not in dumped
    assert "gold_feedback_sab_loader" not in dumped


def test_cli_accepts_gold_and_reward_flag():
    import main as main_mod

    parser = argparse.ArgumentParser()
    main_mod.add_config_arguments(parser, Config())
    args = parser.parse_args(["--verifier_kind", "gold", "--gold_feedback_reward"])
    args.__dict__.setdefault("max_evolve_iterations", None)
    c = Config()
    main_mod.apply_config_overrides(args, c)
    assert c.verifier_kind == "gold" and c.gold_feedback_reward is True
    args2 = parser.parse_args(["--no-gold_feedback_reward"])
    args2.__dict__.setdefault("max_evolve_iterations", None)
    main_mod.apply_config_overrides(args2, c)
    assert c.gold_feedback_reward is False
    # flag absent -> config value untouched
    c.gold_feedback_reward = True
    args3 = parser.parse_args([])
    args3.__dict__.setdefault("max_evolve_iterations", None)
    main_mod.apply_config_overrides(args3, c)
    assert c.gold_feedback_reward is True
    assert main_mod.warn_gold_feedback_mode(
        types.SimpleNamespace(science_agent_bench=False), c
    )
    assert not main_mod.warn_gold_feedback_mode(
        types.SimpleNamespace(science_agent_bench=True), Config()
    )


# --------------------------------------------------------------- facade


def _facade_config(tmp_path, kind):
    return types.SimpleNamespace(
        memory_dir=str(tmp_path / "m"),
        workflow_dir=str(tmp_path / "w"),
        workspace_dir=str(tmp_path / "ws"),
        model_pricing={},
        reasoning_effort="low",
        judge_model="x/y",
        max_tokens=8,
        verifier_kind=kind,
        openrouter_provider_for=lambda m: None,
        openrouter_quantizations_for=lambda m: None,
    )


class _Stub:
    def __init__(self, *a, **k):
        self.workspace_dir = k.get("workspace_dir")


@pytest.mark.parametrize(
    "kind, expected", [("gold", GoldFeedbackEvaluator), ("GOLD", GoldFeedbackEvaluator)]
)
def test_facade_selects_gold(monkeypatch, tmp_path, kind, expected):
    monkeypatch.setattr(evaluator_mod, "GenericEvaluator", _Stub)
    monkeypatch.setattr(evaluator_mod, "ScenarioEvaluator", _Stub)
    monkeypatch.setattr(hybrid_pkg, "HybridVerifierEvaluator", _Stub)
    facade = evaluator_mod.WorkflowEvaluator(_facade_config(tmp_path, kind))
    assert facade.verifier_kind == "gold"
    assert isinstance(facade.verifier_evaluator, expected)
    assert isinstance(facade.verifier_evaluator.hybrid, _Stub)


def test_facade_unknown_kind_still_falls_back_to_hybrid(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluator_mod, "GenericEvaluator", _Stub)
    monkeypatch.setattr(evaluator_mod, "ScenarioEvaluator", _Stub)
    monkeypatch.setattr(
        evaluator_mod, "HybridVerifierEvaluator", lambda c, workspace_dir=None: "HYBRID"
    )
    facade = evaluator_mod.WorkflowEvaluator(_facade_config(tmp_path, "nonsense"))
    assert facade.verifier_evaluator == "HYBRID"


# --------------------------------------------------------------- format


def test_exec_failure_puts_decisive_error_first_and_fits():
    text = build_gold_gradient(
        _evaluated(
            ver=False,
            ver_msg=TRACEBACK,
            sr=False,
            sr_msg="VER Failed, therefore SR is false",
        )
    )
    assert len(text) <= MAX_GRADIENT_CHARS
    lines = text.splitlines()
    assert lines[0].startswith("## Execution outcome") and "FAILED" in lines[0]
    assert lines[1].startswith("Decisive error: ValueError: Found input variables")
    assert text.index("## Execution outcome") < text.index("## Grader verdict")
    assert "Not graded" in text
    # the constant SR placeholder carries no information and is not quoted
    assert "VER Failed, therefore SR is false" not in text
    view = _mutator_view(text)
    assert "Decisive error: ValueError" in view and "## Grader verdict" in view


def test_figure_judge_lists_all_scores_before_bodies():
    text = build_gold_gradient(_evaluated(sr=False, sr_msg=FIGURE_MSG))
    assert len(text) <= MAX_GRADIENT_CHARS
    assert fmt.classify(_evaluated(sr_msg=FIGURE_MSG)) == fmt.FIGURE_JUDGE
    for i, score in ((1, "50"), (2, "40"), (3, "55")):
        assert f"grader critique {i}: [FINAL SCORE]: {score}" in text
    assert text.index("grader critique 3: [FINAL SCORE]") < text.index(
        "### grader critique 1"
    )
    assert "FAIL" in text.splitlines()[1]


def test_programmatic_message_is_verbatim():
    msg = "{'data_correctness': True, 'func_correctness': False}"
    text = build_gold_gradient(_evaluated(sr=False, sr_msg=msg))
    assert msg in text
    assert "no detail beyond pass/fail" not in text


@pytest.mark.parametrize(
    "msg",
    ["N/A", "", "0 / 336", "No (status, message) result tuple in eval output: False"],
)
def test_uninformative_messages_get_explicit_note(msg):
    assert fmt.is_uninformative(msg)
    text = build_gold_gradient(_evaluated(sr=False, sr_msg=msg))
    assert "Note: the grader gave no detail beyond pass/fail." in text
    assert len(text) <= MAX_GRADIENT_CHARS


def test_eval_script_crash_and_huge_messages_are_compacted():
    crash = "Eval script failed with code 1: " + TRACEBACK.split(": ", 1)[1]
    text = build_gold_gradient(_evaluated(sr=False, sr_msg=crash))
    assert fmt.classify(_evaluated(sr_msg=crash)) == fmt.EVAL_SCRIPT_ERROR
    assert "Grader program error on the deliverable: ValueError" in text
    assert len(text) <= MAX_GRADIENT_CHARS
    huge = build_gold_gradient(_evaluated(sr=False, sr_msg="x" * 50000))
    assert len(huge) <= MAX_GRADIENT_CHARS
    small = build_gold_gradient(
        _evaluated(sr=True, sr_msg="overlap: 1.0"), max_chars=300
    )
    assert len(small) <= 300


def test_timeout_is_decisive():
    text = build_gold_gradient(
        _evaluated(ver=False, ver_msg="Code execution timeout after 900 seconds")
    )
    assert "Decisive error: Code execution timeout after 900 seconds" in text


# --------------------------------------------------------------- evaluate: gradient-only


def test_gradient_only_keeps_hybrid_reward_and_swaps_gradient(tmp_path):
    ev, hybrid, grader, uuid, run, _ = _setup(tmp_path, fallback="mean_claim")
    result = ev.evaluate(uuid)
    info = WorkflowInfo(uuid, run)
    gold_text = build_gold_gradient(_evaluated())
    # reward / fallback flag untouched (hybrid decides selection)
    assert info.overall_score == pytest.approx(0.42)
    assert info.overall_score_uncapped == pytest.approx(0.61)
    assert info.reward_is_fallback is True
    # gradient swapped in state_result AND sidecar; hybrid text preserved
    assert info.abstracted_textual_gradient == gold_text
    assert (run / "textual_gradient.txt").read_text() == gold_text
    assert (run / "textual_gradient_hybrid.txt").read_text() == HYBRID_GRADIENT
    block = json.loads((run / "state_result.json").read_text())["evaluation"][
        "verifier"
    ]
    assert block["abstractec_textual_gradient"] == gold_text
    assert block["hybrid_textual_gradient"] == HYBRID_GRADIENT
    assert block["oracle"] is True and block["benchmark_leak"] is True
    assert block["gold_feedback_warning"] == LEAK_WARNING
    assert block["reward_source"] == "hybrid_verifier"
    assert block["gold_feedback"]["status"] == "gold"
    assert block["gold_feedback"]["kind"] == fmt.PROG_CHECK_FAIL
    state = json.loads((run / "state_result.json").read_text())
    assert state["evaluation"]["gold_oracle"]["SR_message"] == "overlap: 0.0134"
    # loud warning in evaluation.txt, hybrid report kept below it
    report = (run / "evaluation.txt").read_text()
    assert report.startswith("!!!") and LEAK_WARNING in report
    assert "Hybrid Verifier Evaluation" in report
    assert (
        json.loads((run / "gold_feedback.json").read_text())["warning"] == LEAK_WARNING
    )
    assert result["abstracted_textual_gradient"] == gold_text
    assert grader.calls[0]["row"] == ROW and grader.calls[0]["loader"] == "LOADER"


# --------------------------------------------------------------- evaluate: full oracle


@pytest.mark.parametrize(
    "grade, expected",
    [
        (_evaluated(sr=True, sr_msg="overlap: 1.0", cbs=1.0), 1.0),
        (_evaluated(sr=False, cbs=0.6), 0.3),
        (
            _evaluated(ver=False, ver_msg="No Python file found in capsule", cbs=0.4),
            0.0,
        ),
    ],
)
def test_full_oracle_reward_comes_from_grader(tmp_path, grade, expected):
    ev, _, _, uuid, run, _ = _setup(
        tmp_path, reward_flag=True, grader=_FakeGrader(grade), fallback="mean_claim"
    )
    ev.evaluate(uuid)
    info = WorkflowInfo(uuid, run)
    assert info.overall_score == pytest.approx(expected)
    assert info.overall_score_uncapped == pytest.approx(expected)
    assert info.reward_is_fallback is False
    block = json.loads((run / "state_result.json").read_text())["evaluation"][
        "verifier"
    ]
    assert block["hybrid_overall_score"] == pytest.approx(0.42)
    assert block["hybrid_reward_fallback"] == "mean_claim"
    assert block["reward_source"] == "benchmark_grader"
    assert oracle_reward(grade) == pytest.approx(expected)


def test_full_oracle_censored_reward_is_neutral(tmp_path):
    grader = _FakeGrader(
        {
            "VER": None,
            "SR": None,
            "CBS": None,
            "cost": 0.0,
            "status": "excluded",
            "infra_error": "Eval sandbox build failed: x",
        }
    )
    ev, hyb, _, uuid, run, _ = _setup(tmp_path, reward_flag=True, grader=grader)
    hyb.reward = 0.95  # a hybrid win-rate that WOULD trigger the 0.9 early stop
    ev.evaluate(uuid)
    block = json.loads((run / "state_result.json").read_text())["evaluation"][
        "verifier"
    ]
    info = WorkflowInfo(uuid, run)
    assert info.overall_score == 0.0 and info.overall_score_uncapped == 0.0
    assert info.overall_score < Config().learned_score_threshold  # no early stop
    assert info.overall_score < Config().admit_threshold  # not admitted to QD archive
    assert info.reward_is_fallback is True
    assert block["reward_censored"] is True
    assert block["reward_source"] == "oracle_censored"
    assert block["hybrid_overall_score"] == pytest.approx(0.95)
    assert block["hybrid_overall_score_uncapped"] == pytest.approx(0.61)
    assert "reward_fallback_reason" in block["gold_feedback"]


def test_full_oracle_without_task_context_keeps_hybrid_reward(tmp_path):
    ev, _, _, uuid, run, _ = _setup(tmp_path, reward_flag=True, row=None)
    ev.evaluate(uuid)
    block = json.loads((run / "state_result.json").read_text())["evaluation"][
        "verifier"
    ]
    assert block["overall_score"] == pytest.approx(0.42)
    assert block["reward_source"] == "hybrid_verifier"
    assert block.get("reward_censored") is None


def test_no_task_context_falls_back_to_hybrid_gradient(tmp_path, caplog):
    ev, _, grader, uuid, run, _ = _setup(tmp_path, row=None)
    with caplog.at_level(logging.WARNING):
        ev.evaluate(uuid)
    assert grader.calls == []  # nothing graded
    info = WorkflowInfo(uuid, run)
    assert info.abstracted_textual_gradient == HYBRID_GRADIENT
    assert (run / "textual_gradient.txt").read_text() == HYBRID_GRADIENT
    block = json.loads((run / "state_result.json").read_text())["evaluation"][
        "verifier"
    ]
    assert block["gold_feedback"]["status"] == "no_task_context"
    assert "science_agent_bench" in block["gold_feedback"]["reason"]
    assert block["oracle"] is True
    assert "no_task_context" in caplog.text and LEAK_WARNING in caplog.text


def test_incomplete_task_row_is_no_context(tmp_path):
    ev, _, grader, uuid, run, _ = _setup(tmp_path, row={"instance_id": "7"})
    ev.evaluate(uuid)
    block = json.loads((run / "state_result.json").read_text())["evaluation"][
        "verifier"
    ]
    assert block["gold_feedback"]["status"] == "no_task_context"
    assert "gold_program_name" in block["gold_feedback"]["reason"]
    assert grader.calls == []


@pytest.mark.parametrize(
    "grader, reason_part",
    [
        (
            _FakeGrader(
                {
                    "VER": None,
                    "SR": None,
                    "CBS": None,
                    "cost": 0.0,
                    "status": "excluded",
                    "infra_error": "Figure-judged task 'x_eval.py' needs OPENAI_API_KEY",
                }
            ),
            "OPENAI_API_KEY",
        ),
        (_FakeGrader(None, exc=RuntimeError("sandbox exploded")), "sandbox exploded"),
    ],
)
def test_env_failure_is_censored_and_uses_hybrid_gradient(
    tmp_path, grader, reason_part
):
    ev, _, _, uuid, run, _ = _setup(tmp_path, grader=grader)
    ev.evaluate(uuid)
    text = WorkflowInfo(uuid, run).abstracted_textual_gradient
    assert text.startswith("Grader feedback unavailable for this generation")
    assert "censored" in text and reason_part in text
    assert text.endswith(HYBRID_GRADIENT)
    block = json.loads((run / "state_result.json").read_text())["evaluation"][
        "verifier"
    ]
    assert block["gold_feedback"]["status"] == "censored_infra"
    assert block["overall_score"] == pytest.approx(0.42)


def test_uninformative_verdict_is_flagged(tmp_path):
    ev, _, _, uuid, run, _ = _setup(
        tmp_path, grader=_FakeGrader(_evaluated(sr_msg="N/A"))
    )
    ev.evaluate(uuid)
    block = json.loads((run / "state_result.json").read_text())["evaluation"][
        "verifier"
    ]
    assert block["gold_feedback"]["uninformative"] is True
    assert "no detail beyond pass/fail" in block["abstracted_textual_gradient"]


def test_short_circuit_run_is_not_graded(tmp_path):
    ev, hybrid, grader, uuid, run, _ = _setup(tmp_path, short=True)
    ev.evaluate(uuid)
    assert grader.calls == []
    block = json.loads((run / "state_result.json").read_text())["evaluation"][
        "verifier"
    ]
    assert block["gold_feedback"]["status"] == "skipped_short_circuit"
    assert block["abstracted_textual_gradient"] == HYBRID_GRADIENT
    assert block["overall_score"] == 0.0


# --------------------------------------------------------------- private copy


def test_grades_a_private_copy_and_never_touches_the_live_workspace(tmp_path):
    ev, _, grader, uuid, _, workspace = _setup(tmp_path)
    before = _tree(workspace)
    ev.evaluate(uuid)
    graded = grader.calls[0]["dir"]
    assert graded != workspace and workspace not in graded.parents
    assert graded.parent.name.startswith("mimosa_gold_")
    # the grader saw the workspace content...
    assert "clintox_nn.py" in grader.calls[0]["files"]
    # ...its pred_results copy-back landed in the copy only
    assert _tree(workspace) == before
    assert not (workspace / "pred_results" / "regenerated.csv").exists()
    assert not graded.parent.exists()  # copy removed afterwards


def test_copy_removed_even_when_hybrid_raises(tmp_path):
    ev, hybrid, grader, uuid, _, _ = _setup(tmp_path)
    made = []
    from sources.evaluators.gold_feedback import evaluator as gold_ev

    real = gold_ev.make_private_copy

    def spy(ws):
        made.append(real(ws))
        return made[-1]

    gold_ev.make_private_copy = spy
    try:
        hybrid.evaluate = lambda u: (_ for _ in ()).throw(RuntimeError("hybrid down"))
        with pytest.raises(RuntimeError):
            ev.evaluate(uuid)
    finally:
        gold_ev.make_private_copy = real
    assert made and not made[0].parent.exists()


# --------------------------------------------------------------- csv_mode wiring


def test_set_gold_feedback_task_only_in_gold_mode():
    hybrid_cfg = types.SimpleNamespace(verifier_kind="hybrid")
    assert CsvEvaluationMode._set_gold_feedback_task(hybrid_cfg, ROW, "L") is False
    assert not hasattr(hybrid_cfg, "gold_feedback_task_row")
    gold_cfg = types.SimpleNamespace(verifier_kind="gold")
    assert CsvEvaluationMode._set_gold_feedback_task(gold_cfg, ROW, "L") is True
    assert (
        gold_cfg.gold_feedback_task_row == ROW
        and gold_cfg.gold_feedback_sab_loader == "L"
    )
    assert gold_cfg.gold_feedback_task_row is not ROW  # private copy of the row


# --------------------------------------------------------------- shared grading helper


class _RecordingCapsule:
    calls = []

    def __init__(self, capsule_path, task_data, sab_loader, api_cost):
        _RecordingCapsule.calls.append((capsule_path, task_data, sab_loader, api_cost))

    def evaluate_all(self):
        return {
            "VER": (True, "v"),
            "SR": (True, "s"),
            "CBS": 1.0,
            "cost": 5.0,
            "status": "evaluated",
            "summary": "x",
        }


def test_grade_directory_builds_capsule_evaluator_call(tmp_path):
    _RecordingCapsule.calls = []
    out = snapshot_grading.grade_directory(
        str(tmp_path),
        task_row=ROW,
        sab_loader="L",
        api_cost=0.25,
        evaluator_cls=_RecordingCapsule,
    )
    assert _RecordingCapsule.calls == [(Path(tmp_path), ROW, "L", 0.25)]
    assert out == {
        "VER": True,
        "VER_message": "v",
        "SR": True,
        "SR_message": "s",
        "CBS": 1.0,
        "cost": 0.25,
        "status": "evaluated",
    }
    assert list(out) == [
        "VER",
        "VER_message",
        "SR",
        "SR_message",
        "CBS",
        "cost",
        "status",
    ]


# Frozen from the PRE-refactor `_evaluate_snapshot_ablations` (inlined
# CapsuleEvaluator call), captured 2026-09-28 by running the original code
# on exactly this scenario. The refactored path must reproduce it byte for
# byte: entries, key order, and the evaluator's constructor arguments.
OLD_ABLATION_CAPTURE = json.loads(
    r"""{"ablations": [{"evolution_index": 0, "uuid": "20260902_102930_aaaaaaaa", "VER": true, "VER_message": "Code executed. Output: ok", "SR": false, "SR_message": "overlap: 0.5", "CBS": 0.61, "cost": 0.1, "status": "evaluated", "source": "snapshot"}, {"evolution_index": 1, "uuid": "20260902_103100_bbbbbbbb", "VER": true, "VER_message": "cap ok", "SR": true, "SR_message": "cap sr", "CBS": 1.0, "cost": 0.1, "status": "evaluated", "source": "capsule", "infra_error": "cap infra"}, {"evolution_index": 2, "uuid": "20260902_103500_cccccccc", "VER": null, "SR": null, "CBS": null, "cost": 0.77, "status": "excluded", "source": "snapshot", "infra_error": "Eval sandbox build failed: boom"}, {"evolution_index": 3, "uuid": "20260902_103600_dddddddd", "VER": null, "SR": null, "CBS": null, "cost": 0.1, "status": "excluded", "source": "snapshot", "infra_error": null}, {"evolution_index": 4, "uuid": "20260902_103700_eeeeeeee", "VER": null, "SR": null, "CBS": null, "cost": 0.1, "status": "excluded", "source": "snapshot", "infra_error": "Unexpected harness error: harness exploded"}, {"evolution_index": 5, "uuid": "20260902_103800_ffffffff", "VER": false, "VER_message": "No Python file found in capsule", "SR": false, "SR_message": "VER Failed, therefore SR is false", "CBS": 0.0, "cost": 0.0, "status": "evaluated", "source": "snapshot"}], "key_orders": [["evolution_index", "uuid", "VER", "VER_message", "SR", "SR_message", "CBS", "cost", "status", "source"], ["evolution_index", "uuid", "VER", "VER_message", "SR", "SR_message", "CBS", "cost", "status", "source", "infra_error"], ["evolution_index", "uuid", "VER", "SR", "CBS", "cost", "status", "source", "infra_error"], ["evolution_index", "uuid", "VER", "SR", "CBS", "cost", "status", "source", "infra_error"], ["evolution_index", "uuid", "VER", "SR", "CBS", "cost", "status", "source", "infra_error"], ["evolution_index", "uuid", "VER", "VER_message", "SR", "SR_message", "CBS", "cost", "status", "source"]], "calls": [{"dir": "aaaaaaaa", "task_data": {"instance_id": "7", "gold_program_name": "t.py", "output_fname": "pred_results/o.csv", "eval_script_name": "t_eval.py"}, "sab_loader": "LOADER", "api_cost": 0.1}, {"dir": "cccccccc", "task_data": {"instance_id": "7", "gold_program_name": "t.py", "output_fname": "pred_results/o.csv", "eval_script_name": "t_eval.py"}, "sab_loader": "LOADER", "api_cost": 0.1}, {"dir": "dddddddd", "task_data": {"instance_id": "7", "gold_program_name": "t.py", "output_fname": "pred_results/o.csv", "eval_script_name": "t_eval.py"}, "sab_loader": "LOADER", "api_cost": 0.1}, {"dir": "eeeeeeee", "task_data": {"instance_id": "7", "gold_program_name": "t.py", "output_fname": "pred_results/o.csv", "eval_script_name": "t_eval.py"}, "sab_loader": "LOADER", "api_cost": 0.1}, {"dir": "ffffffff", "task_data": {"instance_id": "7", "gold_program_name": "t.py", "output_fname": "pred_results/o.csv", "eval_script_name": "t_eval.py"}, "sab_loader": "LOADER", "api_cost": 0.0}]}"""
)


class _ScenarioCapsule:
    calls = []

    def __init__(self, capsule_path, task_data, sab_loader, api_cost):
        self.name = Path(capsule_path).name.split("_")[-1]
        _ScenarioCapsule.calls.append(
            {
                "dir": self.name,
                "task_data": task_data,
                "sab_loader": sab_loader,
                "api_cost": round(api_cost, 6),
            }
        )

    def evaluate_all(self):
        n = self.name
        if n == "aaaaaaaa":
            return {
                "VER": (True, "Code executed. Output: ok"),
                "SR": (False, "overlap: 0.5"),
                "CBS": 0.61,
                "cost": 9.0,
                "status": "evaluated",
                "summary": "s",
            }
        if n == "cccccccc":
            return {
                "VER": (None, "boom"),
                "SR": (None, "boom"),
                "CBS": None,
                "cost": 0.77,
                "status": "excluded",
                "infra_error": "Eval sandbox build failed: boom",
            }
        if n == "dddddddd":
            return {"status": "excluded"}
        if n == "eeeeeeee":
            raise RuntimeError("harness exploded")
        return {
            "VER": (False, "No Python file found in capsule"),
            "SR": (False, "VER Failed, therefore SR is false"),
            "CBS": 0.0,
            "cost": 0.0,
        }


def test_refactored_ablation_path_matches_pre_refactor_capture(monkeypatch):
    monkeypatch.setattr(csv_mode, "CapsuleEvaluator", _ScenarioCapsule)
    _ScenarioCapsule.calls = []
    ev = object.__new__(CsvEvaluationMode)
    ev.logger = logging.getLogger("tests.gold_feedback")
    sid = uuid_mod.uuid4().hex[:12]
    uuids = [
        "20260902_102930_aaaaaaaa",
        "20260902_103100_bbbbbbbb",
        "20260902_103500_cccccccc",
        "20260902_103600_dddddddd",
        "20260902_103700_eeeeeeee",
        "20260902_103800_ffffffff",
    ]
    try:
        for u in uuids:
            d = Path(f"/tmp/mimosa_run_{sid}_{u}")
            d.mkdir(parents=True)
            (d / "x.txt").write_text("x")
            time.sleep(0.01)  # distinct mtimes: orphan order is mtime order
        runs = [
            IndividualRun(
                goal="g",
                prompt="p",
                current_uuid=u,
                reward=r,
                iteration_count=i,
                cost=0.1 * (i + 1),
            )
            for i, (u, r) in enumerate(
                zip(uuids[:5], [0.2, 0.9, 0.5, 0.1, 0.3], strict=True)
            )
        ]
        row = {
            "instance_id": "7",
            "gold_program_name": "t.py",
            "output_fname": "pred_results/o.csv",
            "eval_script_name": "t_eval.py",
        }
        ed = {
            "VER": True,
            "VER_message": "cap ok",
            "SR": True,
            "SR_message": "cap sr",
            "CBS": 1.0,
            "status": "evaluated",
            "infra_error": "cap infra",
        }
        out = ev._evaluate_snapshot_ablations(
            sid, row=row, runs=runs, sab_loader="LOADER", execution_data=ed
        )
    finally:
        for p in Path("/tmp").glob(f"mimosa_run_{sid}_*"):
            shutil.rmtree(p, ignore_errors=True)
    for e in out:
        if isinstance(e.get("cost"), float):
            e["cost"] = round(e["cost"], 6)
    assert out == OLD_ABLATION_CAPTURE["ablations"]
    assert [list(e) for e in out] == OLD_ABLATION_CAPTURE["key_orders"]
    assert _ScenarioCapsule.calls == OLD_ABLATION_CAPTURE["calls"]


# --------------------------------------------------------------- review fixes


def test_config_bool_strings_and_timeout_knob():
    c = Config()
    assert c.gold_feedback_timeout_s == 1800
    for raw, expected in (
        ("false", False),
        ("0", False),
        ("no", False),
        ("true", True),
        ("1", True),
        (True, True),
        (0, False),
    ):
        c.gold_feedback_reward = not expected
        c.from_json({"gold_feedback_reward": raw})
        assert c.gold_feedback_reward is expected, raw
    c.from_json({"gold_feedback_timeout_s": 60})
    assert (
        c.gold_feedback_timeout_s == 60 and c.jsonify()["gold_feedback_timeout_s"] == 60
    )

    import main as main_mod

    parser = argparse.ArgumentParser()
    main_mod.add_config_arguments(parser, Config())
    args = parser.parse_args(["--gold_feedback_timeout_s", "90"])
    args.__dict__.setdefault("max_evolve_iterations", None)
    main_mod.apply_config_overrides(args, c)
    assert c.gold_feedback_timeout_s == 90


def test_grading_timeout_is_censored_and_copy_cleaned_by_worker(tmp_path):
    release = __import__("threading").Event()

    class _SlowGrader(_FakeGrader):
        def __call__(self, directory, task_row, sab_loader):
            self.calls.append({"dir": Path(directory)})
            release.wait(5)
            return _evaluated()

    grader = _SlowGrader(None)
    ev, _, _, uuid, run, _ = _setup(tmp_path, grader=grader, timeout_s=0.2)
    t0 = time.time()
    ev.evaluate(uuid)
    assert time.time() - t0 < 3
    block = json.loads((run / "state_result.json").read_text())["evaluation"][
        "verifier"
    ]
    assert block["gold_feedback"]["status"] == "censored_infra"
    assert "wall-clock cap" in block["gold_feedback"]["reason"]
    assert block["abstracted_textual_gradient"].endswith(HYBRID_GRADIENT)
    copy_root = grader.calls[0]["dir"].parent
    assert copy_root.exists()  # still owned by the abandoned worker
    release.set()
    for _ in range(50):
        if not copy_root.exists():
            break
        time.sleep(0.05)
    assert not copy_root.exists()  # the worker deleted it when it ended


def _phenotype_engine(kind):
    from sources.core.evolution_engine import EvolutionEngine

    seen = {}

    class _Judge:
        verifier_kind = kind

        def evaluate(self, **kw):
            seen["thread"] = __import__("threading").get_ident()
            return {"evaluation_type": "verifier", "uuid": kw["uuid"]}

    eng = object.__new__(EvolutionEngine)
    eng.judge = _Judge()
    return eng, seen


@pytest.mark.parametrize(
    "kind, off_loop", [("gold", True), ("hybrid", False), (None, False)]
)
def test_only_gold_evaluation_runs_off_the_event_loop(kind, off_loop):
    import asyncio
    import threading

    eng, seen = _phenotype_engine(kind)
    out = asyncio.run(eng._evaluate_workflow_phenotype("u1", "", None, []))
    assert out == "verifier"
    assert (seen["thread"] != threading.get_ident()) is off_loop


def _runs_from_disk(workflow_dir, uuids):
    from sources.core.schema import IndividualRun as Run

    runs = []
    for i, u in enumerate(uuids):
        info = WorkflowInfo(u, Path(workflow_dir) / u)
        runs.append(
            Run(
                goal="g",
                prompt="p",
                current_uuid=u,
                reward=info.overall_score,
                reward_is_fallback=info.reward_is_fallback,
                iteration_count=i,
                state_result=info.state_result,
            )
        )
    return runs


def test_full_oracle_gen0_grade_is_not_a_fallback_in_capsule_argmax(tmp_path):
    from sources.core.evolution_engine import _select_best_run

    u0, u1 = "20260928_120000_aaaaaaaa", "20260928_120100_bbbbbbbb"
    ev0, hyb, _, _, _, _ = _setup(
        tmp_path,
        reward_flag=True,
        uuid=u0,
        grader=_FakeGrader(_evaluated(sr=True, cbs=1.0)),
    )
    ev0.evaluate(u0)  # gen 0: hybrid n_pairs=0, grader SR=1 -> 1.0
    ev1, hyb1, _, _, _, _ = _setup(
        tmp_path,
        reward_flag=True,
        uuid=u1,
        grader=_FakeGrader(_evaluated(sr=False, cbs=0.6)),
    )
    hyb1.n_pairs = 1
    ev1.evaluate(u1)  # gen 1: real hybrid pairs, grader -> 0.3
    runs = _runs_from_disk(tmp_path / "workflows", [u0, u1])
    assert [r.reward for r in runs] == [1.0, 0.3]
    assert _select_best_run(runs).current_uuid == u0


def test_full_oracle_censored_gen_is_excluded_from_capsule_argmax(tmp_path):
    from sources.core.evolution_engine import _select_best_run

    u0, u1 = "20260928_120000_aaaaaaaa", "20260928_120100_bbbbbbbb"
    censored = _FakeGrader(
        {
            "VER": None,
            "SR": None,
            "CBS": None,
            "cost": 0.0,
            "status": "excluded",
            "infra_error": "sandbox build failed",
        }
    )
    ev0, hyb0, _, _, run0, _ = _setup(
        tmp_path, reward_flag=True, uuid=u0, grader=censored
    )
    hyb0.reward = 0.9  # hybrid win-rate kept on the censored gen
    hyb0.n_pairs = 3
    ev0.evaluate(u0)
    block = json.loads((run0 / "state_result.json").read_text())["evaluation"][
        "verifier"
    ]
    assert block["reward_fallback"] == "oracle_censored"
    assert block["overall_score"] == 0.0
    assert block["hybrid_overall_score"] == pytest.approx(0.9)
    ev1, hyb1, _, _, _, _ = _setup(
        tmp_path,
        reward_flag=True,
        uuid=u1,
        grader=_FakeGrader(_evaluated(sr=False, cbs=0.6)),
    )
    hyb1.n_pairs = 3
    ev1.evaluate(u1)
    runs = _runs_from_disk(tmp_path / "workflows", [u0, u1])
    assert runs[0].reward_is_fallback is True and runs[1].reward_is_fallback is False
    assert _select_best_run(runs).current_uuid == u1


def test_gradient_only_censored_gen_is_not_flagged(tmp_path):
    censored = _FakeGrader(
        {
            "VER": None,
            "SR": None,
            "CBS": None,
            "cost": 0.0,
            "status": "excluded",
            "infra_error": "x",
        }
    )
    ev, _, _, uuid, run, _ = _setup(tmp_path, grader=censored)
    ev.evaluate(uuid)
    block = json.loads((run / "state_result.json").read_text())["evaluation"][
        "verifier"
    ]
    assert block["reward_fallback"] is None  # hybrid value untouched


def test_evaluation_banner_not_stacked_on_reevaluation(tmp_path):
    ev, hybrid, _, uuid, run, _ = _setup(tmp_path)
    ev.evaluate(uuid)
    hybrid.evaluate = lambda u: {"uuid": u, "claims": []}  # no fresh hybrid report
    ev.evaluate(uuid)
    report = (run / "evaluation.txt").read_text()
    assert report.count(LEAK_WARNING) == 1
    assert report.count("Hybrid Verifier Evaluation") == 1


def test_selector_skips_oracle_generations_in_honest_runs(tmp_path, caplog):
    from sources.core.workflow_selection import WorkflowSelector

    wd = tmp_path / "wf"
    for u, oracle in (("honest_1", False), ("oracle_1", True), ("oracle_2", "flag")):
        d = wd / u
        d.mkdir(parents=True)
        (d / f"workflow_genotype_{u}.py").write_text("x = 1\n")
        ver = {"overall_score": 0.5}
        if oracle == "flag":
            ver["oracle"] = True
        (d / "state_result.json").write_text(
            json.dumps({"goal": "g", "answers": ["a"], "evaluation": {"verifier": ver}})
        )
        if oracle is True:
            (d / "gold_feedback.json").write_text("{}")
    with caplog.at_level(logging.WARNING):
        honest = WorkflowSelector(
            types.SimpleNamespace(workflow_dir=str(wd), verifier_kind="hybrid")
        )
    assert set(honest.workflows_info) == {"honest_1"}
    assert "LEAK GUARD" in caplog.text
    gold = WorkflowSelector(
        types.SimpleNamespace(workflow_dir=str(wd), verifier_kind="gold")
    )
    assert set(gold.workflows_info) == {"honest_1", "oracle_1", "oracle_2"}


def test_gold_init_warns_on_mixed_workflow_dir(tmp_path, caplog):
    honest = tmp_path / "workflows" / "20260101_000000_00000000"
    honest.mkdir(parents=True)
    (honest / "state_result.json").write_text("{}")
    with caplog.at_level(logging.WARNING):
        ev, *_ = _setup(tmp_path)
    assert "non-oracle generation" in caplog.text
    # the pre-existing honest gen + _setup's not-yet-evaluated gen
    assert ev._warn_mixed_workflow_dir() == 2
    ev.evaluate("20260928_120000_abcdef12")  # now an oracle gen
    assert ev._warn_mixed_workflow_dir() == 1


def test_capsule_results_get_leak_markers(tmp_path):
    path = tmp_path / "evaluation_results.json"
    path.write_text(json.dumps({"VER": True, "SR": False, "CBS": 0.4}))
    ev = object.__new__(CsvEvaluationMode)
    ev.logger = logging.getLogger("tests.gold_feedback")
    ev.config = types.SimpleNamespace(verifier_kind="gold", gold_feedback_reward=False)
    assert ev._gold_mode()
    ev._stamp_gold_capsule_results(path)
    data = json.loads(path.read_text())
    assert (data["VER"], data["SR"], data["CBS"]) == (True, False, 0.4)  # contract kept
    assert (
        data["oracle_feedback"] is True
        and data["benchmark_leak_warning"] == LEAK_WARNING
    )
    ev.config = types.SimpleNamespace(verifier_kind="hybrid")
    assert not ev._gold_mode()


def test_copy_handoff_ownership_is_exclusive():
    from sources.evaluators.gold_feedback.grader import CopyHandoff

    # worker finishes first (normal / race window): caller keeps the copy
    h = CopyHandoff()
    assert h.worker_done() is False  # worker must NOT delete
    assert h.caller_reclaims() is True  # caller owns it and the result
    # caller gives up first: the worker deletes when it ends
    h = CopyHandoff()
    assert h.caller_reclaims() is False  # abandoned
    assert h.worker_done() is True  # worker must delete


def test_worker_finishing_in_race_window_keeps_result(tmp_path, monkeypatch):
    """join() times out but the worker is done before the ownership check."""
    from sources.evaluators.gold_feedback import grader as grader_mod

    real_thread = grader_mod.threading.Thread

    class _LateJoinThread(real_thread):
        def join(self, timeout=None):
            super().join()  # worker finished ...
            return None  # ... but the caller behaves as if join timed out

    monkeypatch.setattr(grader_mod.threading, "Thread", _LateJoinThread)
    copy_dir = grader_mod.make_private_copy(_setup_ws(tmp_path))
    ctx = grader_mod.TaskContext(row=ROW, sab_loader="L")
    grade = grader_mod.grade_private_copy(
        copy_dir, ctx, grade_fn=_FakeGrader(_evaluated()), timeout_s=0.01
    )
    assert grade["status"] == "evaluated" and "copy_owned_by_worker" not in grade
    assert copy_dir.exists()  # caller still owns it
    grader_mod.discard_private_copy(copy_dir)
    assert not copy_dir.parent.exists()


def _setup_ws(tmp_path):
    ws = tmp_path / "ws_only"
    ws.mkdir()
    (ws / "a.py").write_text("x = 1\n")
    return ws


def test_zero_timeout_disables_cap_via_cli():
    import main as main_mod

    parser = argparse.ArgumentParser()
    main_mod.add_config_arguments(parser, Config())
    args = parser.parse_args(["--gold_feedback_timeout_s", "0"])
    args.__dict__.setdefault("max_evolve_iterations", None)
    c = Config()
    main_mod.apply_config_overrides(args, c)
    assert c.gold_feedback_timeout_s == 0


def test_zero_timeout_grades_inline_without_thread(tmp_path, monkeypatch):
    from sources.evaluators.gold_feedback import grader as grader_mod

    def _no_threads(*a, **k):
        raise AssertionError("no worker thread expected when the cap is disabled")

    monkeypatch.setattr(grader_mod.threading, "Thread", _no_threads)
    ev, _, grader, uuid, run, _ = _setup(tmp_path, timeout_s=0)
    ev.evaluate(uuid)
    block = json.loads((run / "state_result.json").read_text())["evaluation"][
        "verifier"
    ]
    assert block["gold_feedback"]["status"] == "gold"


def test_inner_hybrid_gets_isolated_registry(monkeypatch, tmp_path):
    from sources.evaluators.gold_feedback.evaluator import gold_hybrid_config

    captured = {}

    class _Hybrid:
        def __init__(self, config, workspace_dir=None):
            captured["config"] = config
            self.workspace_dir = workspace_dir

    monkeypatch.setattr(hybrid_pkg, "HybridVerifierEvaluator", _Hybrid)
    config = types.SimpleNamespace(
        workflow_dir=str(tmp_path / "wf"), workspace_dir=str(tmp_path / "ws")
    )
    GoldFeedbackEvaluator(config)
    inner = captured["config"]
    assert inner is not config
    assert inner.temp_dir == str(tmp_path / "wf" / "_verifier_tmp_gold")
    assert not hasattr(config, "temp_dir")  # caller config untouched
    # an explicit temp_dir base is suffixed, never shared
    config2 = types.SimpleNamespace(
        workflow_dir=str(tmp_path / "wf"), temp_dir="/x/reg"
    )
    assert gold_hybrid_config(config2).temp_dir == "/x/reg_gold"
    assert config2.temp_dir == "/x/reg"


def test_real_hybrid_registry_root_is_separate(tmp_path):
    """With the real HybridVerifierEvaluator: gold root != honest root."""
    cfg = types.SimpleNamespace(
        memory_dir=str(tmp_path / "m"),
        workflow_dir=str(tmp_path / "wf"),
        workspace_dir=str(tmp_path / "ws"),
        model_pricing={},
        reasoning_effort="low",
        judge_model="x/y",
        max_tokens=8,
        vision_judge_model=None,
        openrouter_provider_for=lambda m: None,
        openrouter_quantizations_for=lambda m: None,
    )
    honest = hybrid_pkg.HybridVerifierEvaluator(cfg)
    gold = GoldFeedbackEvaluator(cfg)
    assert honest._runner_temp_root == tmp_path / "wf" / "_verifier_tmp"
    assert gold.hybrid._runner_temp_root == tmp_path / "wf" / "_verifier_tmp_gold"

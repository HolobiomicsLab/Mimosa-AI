#!/usr/bin/env python3
"""
Tests for ScienceAgentBench evaluation error handling.

Covers the two behaviours hardened in benchmark_evaluation:
  1. ExecutionSandbox._parse_eval_output — robust parsing of the eval script's
     `(status, message)` result, and NO false-positive success on noisy output.
  2. CapsuleEvaluator.calculate_codebert_score — scores the executed file and
     records why CBS fell back to 0.0 instead of masking failures as a real 0.
"""

import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.benchmark_evaluation.execution_sandbox import ExecutionSandbox
from sources.benchmark_evaluation.capsule_evaluator import CapsuleEvaluator


def _bare_sandbox() -> ExecutionSandbox:
    """ExecutionSandbox without the heavy venv/env setup done in __init__."""
    return object.__new__(ExecutionSandbox)


# --- _parse_eval_output -----------------------------------------------------

def test_clean_success_tuple():
    sb = _bare_sandbox()
    assert sb._parse_eval_output("(1, 'all good')") == (True, "all good")


def test_clean_failure_tuple():
    sb = _bare_sandbox()
    success, msg = sb._parse_eval_output("(0, 'threshold not met')")
    assert success is False and msg == "threshold not met"


def test_boolean_status_is_accepted():
    sb = _bare_sandbox()
    assert sb._parse_eval_output("(True, 'ok')")[0] is True
    assert sb._parse_eval_output("(False, 'no')")[0] is False


def test_library_banner_before_result_does_not_break_parsing():
    # sklearn/tf/rdkit print banners to stdout before the result tuple.
    noisy = (
        "2024-01-01 12:00:00 tensorflow using CPU\n"
        "RDKit WARNING: something\n"
        "(1, \"{'data_correctness': True, 'func_correctness': True}\")"
    )
    sb = _bare_sandbox()
    assert sb._parse_eval_output(noisy)[0] is True


def test_last_tuple_wins():
    # If several tuples print, the final one is the eval() result.
    out = "(1, 'intermediate')\n(0, 'final verdict')"
    sb = _bare_sandbox()
    assert sb._parse_eval_output(out) == (False, "final verdict")


def test_figure_judge_message_with_commas():
    out = "(1, 'The figure matches: axes, colors, and legend are correct')"
    sb = _bare_sandbox()
    success, msg = sb._parse_eval_output(out)
    assert success is True and "legend" in msg


def test_stray_digit_is_not_a_success():
    # Regression: the old heuristic returned success whenever '1' appeared.
    sb = _bare_sandbox()
    assert sb._parse_eval_output("Traceback (most recent call last): line 1")[0] is False


def test_empty_output_is_failure():
    sb = _bare_sandbox()
    assert sb._parse_eval_output("")[0] is False
    assert sb._parse_eval_output("   \n  \n")[0] is False


# --- CapsuleEvaluator.calculate_codebert_score ------------------------------

class _FakeSandbox:
    def __init__(self, selected):
        self._selected = selected

    def select_generated_script(self, script_name=None):
        return self._selected


class _FakeLoader:
    def __init__(self, gold_path=None, raises=None):
        self._gold_path = gold_path
        self._raises = raises

    def get_gold_program_path(self, task_data):
        if self._raises is not None:
            raise self._raises
        return self._gold_path


def _bare_evaluator(gold_name, sandbox, loader):
    """CapsuleEvaluator with only the fields calculate_codebert_score reads."""
    import logging
    ev = object.__new__(CapsuleEvaluator)
    ev.gold_program_name = gold_name
    ev.sandbox = sandbox
    ev.sab_loader = loader
    ev.task_data = {}
    ev.logger = logging.getLogger("test")
    ev._cbs_error = None
    return ev


def test_cbs_records_error_when_no_gold_name():
    ev = _bare_evaluator("", _FakeSandbox(None), _FakeLoader())
    assert ev.calculate_codebert_score() == 0.0
    assert ev._cbs_error == "No gold program name for task"


def test_cbs_records_error_when_no_generated_file():
    ev = _bare_evaluator("clintox_nn.py", _FakeSandbox(None), _FakeLoader())
    assert ev.calculate_codebert_score() == 0.0
    assert "No generated Python file" in ev._cbs_error


def test_cbs_records_error_when_gold_missing(tmp_path):
    gen = tmp_path / "gen.py"
    gen.write_text("print(1)\n")
    loader = _FakeLoader(raises=FileNotFoundError("gold not found"))
    ev = _bare_evaluator("clintox_nn.py", _FakeSandbox(gen), loader)
    assert ev.calculate_codebert_score() == 0.0
    assert "Gold program unavailable" in ev._cbs_error


def test_cbs_success_path_clears_error(tmp_path, monkeypatch):
    gen = tmp_path / "gen.py"
    gen.write_text("print(1)\n")
    gold = tmp_path / "gold.py"
    gold.write_text("print(1)\n")
    import sources.benchmark_evaluation.capsule_evaluator as ce
    monkeypatch.setattr(ce, "calculate_codebert_score", lambda **_: 0.87)
    ev = _bare_evaluator("clintox_nn.py", _FakeSandbox(gen), _FakeLoader(gold_path=gold))
    assert ev.calculate_codebert_score() == 0.87
    assert ev._cbs_error is None


def test_cbs_records_error_on_computation_failure(tmp_path, monkeypatch):
    gen = tmp_path / "gen.py"
    gen.write_text("print(1)\n")
    gold = tmp_path / "gold.py"
    gold.write_text("print(1)\n")
    import sources.benchmark_evaluation.capsule_evaluator as ce

    def _boom(**_):
        raise RuntimeError("no module named transformers")

    monkeypatch.setattr(ce, "calculate_codebert_score", _boom)
    ev = _bare_evaluator("clintox_nn.py", _FakeSandbox(gen), _FakeLoader(gold_path=gold))
    assert ev.calculate_codebert_score() == 0.0
    assert "CBS computation failed" in ev._cbs_error


# --- Infra exclusion (evaluate_all status) ----------------------------------

from sources.benchmark_evaluation.execution_sandbox import EvalInfraError


class _EvalSandbox:
    """Fake sandbox: VER outcome fixed; eval script raises or returns as configured."""

    def __init__(self, ver=(True, "ran"), eval_result=None, eval_raises=None):
        self._ver = ver
        self._eval_result = eval_result
        self._eval_raises = eval_raises

    def run_generated_code(self, **_):
        return self._ver

    def run_eval_script(self, **_):
        if self._eval_raises is not None:
            raise self._eval_raises
        return self._eval_result

    def select_generated_script(self, *_):
        return None

    def cleanup(self):
        pass


class _EvalLoader:
    def get_eval_script_path(self, _task):
        return Path("e_eval.py"), Path("judge.py")

    def get_gold_program_path(self, _task):
        raise FileNotFoundError("gold missing")


def _make_evaluator(tmp_path, sandbox):
    task_data = {
        "instance_id": "1",
        "gold_program_name": "g.py",
        "output_fname": "o.csv",
        "eval_script_name": "e_eval.py",
    }
    ev = CapsuleEvaluator(capsule_path=tmp_path, task_data=task_data, sab_loader=_EvalLoader())
    ev.sandbox = sandbox  # skip real venv build (already non-None)
    return ev


def test_infra_error_from_eval_excludes_task(tmp_path):
    sandbox = _EvalSandbox(eval_raises=EvalInfraError("gold_results not found"))
    res = _make_evaluator(tmp_path, sandbox).evaluate_all()
    assert res["status"] == "excluded"
    assert res["VER"][0] is None and res["SR"][0] is None and res["CBS"] is None
    assert "gold_results" in res["infra_error"]


def test_genuine_ver_failure_still_counts(tmp_path):
    sandbox = _EvalSandbox(ver=(False, "code crashed"))
    res = _make_evaluator(tmp_path, sandbox).evaluate_all()
    assert res["status"] == "evaluated"
    assert res["VER"][0] is False and res["SR"][0] is False


def test_genuine_sr_failure_still_counts(tmp_path):
    sandbox = _EvalSandbox(ver=(True, "ran"), eval_result=(False, "0 / 3"))
    res = _make_evaluator(tmp_path, sandbox).evaluate_all()
    assert res["status"] == "evaluated"
    assert res["VER"][0] is True and res["SR"][0] is False


def test_sandbox_build_failure_excludes_task(tmp_path, monkeypatch):
    import sources.benchmark_evaluation.capsule_evaluator as ce

    class _BoomSandbox:
        def __init__(self, *a, **k):
            raise RuntimeError("no python3.12")

    monkeypatch.setattr(ce, "ExecutionSandbox", _BoomSandbox)
    task_data = {"instance_id": "1", "gold_program_name": "g.py",
                 "output_fname": "o.csv", "eval_script_name": "e_eval.py"}
    ev = CapsuleEvaluator(capsule_path=tmp_path, task_data=task_data, sab_loader=_EvalLoader())
    res = ev.evaluate_all()  # sandbox is None -> _ensure_sandbox builds -> raises
    assert res["status"] == "excluded"
    assert "sandbox build failed" in res["infra_error"].lower()


# --- Figure-judge detection (import-based, not substring) -------------------

def test_needs_judge_true_on_real_import():
    text = "from gpt4_visual_judge import encode_image, score_figure\n\ndef eval():\n    ...\n"
    assert ExecutionSandbox._eval_needs_judge(text) is True


def test_needs_judge_false_on_comment_mention():
    # A comment/docstring mentioning the judge must NOT force exclusion.
    text = "# uses score_figure from gpt4_visual_judge in other tasks\nimport pandas as pd\n"
    assert ExecutionSandbox._eval_needs_judge(text) is False


def test_needs_judge_false_on_plain_script_eval():
    text = "from sklearn.metrics import roc_auc_score\nimport pandas as pd\n"
    assert ExecutionSandbox._eval_needs_judge(text) is False


# --- get_eval_script_path: judge is optional, eval script is required --------

from sources.benchmark_evaluation.science_agent_bench import ScienceAgentBenchLoader


def _loader_with_eval_programs(tmp_path):
    loader = ScienceAgentBenchLoader(base_path=str(tmp_path))
    loader.eval_programs_path.mkdir(parents=True, exist_ok=True)
    return loader


def test_eval_script_path_judge_none_when_absent(tmp_path):
    loader = _loader_with_eval_programs(tmp_path)
    (loader.eval_programs_path / "x_eval.py").write_text("print((1,''))\n")
    eval_path, judge = loader.get_eval_script_path({"eval_script_name": "x_eval.py"})
    assert eval_path.name == "x_eval.py"
    assert judge is None  # non-figure tasks must not be blocked by a missing judge


def test_eval_script_path_returns_judge_when_present(tmp_path):
    loader = _loader_with_eval_programs(tmp_path)
    (loader.eval_programs_path / "x_eval.py").write_text("print((1,''))\n")
    (loader.eval_programs_path / "gpt4_visual_judge.py").write_text("# judge\n")
    _, judge = loader.get_eval_script_path({"eval_script_name": "x_eval.py"})
    assert judge is not None and judge.name == "gpt4_visual_judge.py"


def test_eval_script_path_raises_when_eval_missing(tmp_path):
    import pytest
    loader = _loader_with_eval_programs(tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.get_eval_script_path({"eval_script_name": "nope_eval.py"})


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))

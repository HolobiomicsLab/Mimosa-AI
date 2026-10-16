#!/usr/bin/env python3
"""Tests for the evaluation CLI sequential queue and port-range prompts."""

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sources.cli.evaluation_cli as evaluation_cli
from config import AddressMCP
from sources.cli.evaluation_cli import EvalRunSpec, EvaluationCLI


def _make_spec(run_id: int, workspace: str, port_min: int = 5000, port_max: int = 5200):
    config = SimpleNamespace(
        discovery_addresses=[
            AddressMCP(ip="0.0.0.0", port_min=port_min, port_max=port_max)
        ],
        workspace_dir=workspace,
        smolagent_model_id="test/model",
    )
    return EvalRunSpec(
        run_id=run_id, config=config, eval_mode="one_shot", csv_runs_limit=1
    )


class _ScriptedIO:
    """Feed canned answers to the module-level _ask / _ask_yn prompts."""

    def __init__(self, answers: list[str], yn_answers: list[bool]):
        self.answers = list(answers)
        self.yn_answers = list(yn_answers)

    def ask(self, prompt: str, default: str = "") -> str:
        answer = self.answers.pop(0)
        return answer if answer else default

    def ask_yn(self, prompt: str, default: bool = True) -> bool:
        return self.yn_answers.pop(0)


class _StubToolManager:
    def __init__(self, config):
        self.config = config

    async def discover_mcp_servers(self):
        return ["stub-mcp"]


def test_validate_queue_allows_overlapping_ports(tmp_path):
    cli = EvaluationCLI(SimpleNamespace())
    ws_a, ws_b = tmp_path / "ws_a", tmp_path / "ws_b"
    ws_a.mkdir()
    ws_b.mkdir()
    cli._queue = [
        _make_spec(1, str(ws_a), 5000, 5200),
        _make_spec(2, str(ws_b), 5000, 5200),
    ]
    assert cli._validate_queue() is True


def test_validate_queue_rejects_shared_workspace(tmp_path):
    cli = EvaluationCLI(SimpleNamespace())
    ws = tmp_path / "ws"
    ws.mkdir()
    cli._queue = [_make_spec(1, str(ws)), _make_spec(2, str(ws))]
    assert cli._validate_queue() is False


def test_launch_queue_is_sequential_and_survives_failure():
    cli = EvaluationCLI(SimpleNamespace())
    cli._queue = [_make_spec(1, "ws_a"), _make_spec(2, "ws_b"), _make_spec(3, "ws_c")]

    order: list[int] = []
    in_flight = {"now": 0, "max": 0}

    async def fake_run(spec):
        in_flight["now"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["now"])
        order.append(spec.run_id)
        await asyncio.sleep(0)  # yield so overlap would show if runs were concurrent
        in_flight["now"] -= 1
        if spec.run_id == 2:
            spec.status = "error"
            raise RuntimeError("boom")
        spec.status = "completed"

    cli._run_single_eval = fake_run
    asyncio.run(cli._launch_queue())

    assert order == [1, 2, 3]
    assert in_flight["max"] == 1
    assert [s.status for s in cli._queue] == ["completed", "error", "completed"]


def test_launch_queue_records_early_failures():
    """A crash before _run_single_eval records anything must still mark the spec."""
    cli = EvaluationCLI(SimpleNamespace())
    cli._queue = [_make_spec(1, "ws_a"), _make_spec(2, "ws_b")]

    async def fake_run(spec):
        if spec.run_id == 1:
            raise ImportError("csv_mode dependency missing")  # status stays "pending"
        spec.status = "completed"

    cli._run_single_eval = fake_run
    asyncio.run(cli._launch_queue())

    assert [s.status for s in cli._queue] == ["error", "completed"]


def _run_connectivity(monkeypatch, tmp_path, script: _ScriptedIO, previous_spec):
    monkeypatch.setattr(evaluation_cli, "_ask", script.ask)
    monkeypatch.setattr(evaluation_cli, "_ask_yn", script.ask_yn)
    monkeypatch.setattr(evaluation_cli, "ToolManager", _StubToolManager)

    cli = EvaluationCLI(SimpleNamespace())
    cli._queue = [previous_spec]

    ws = tmp_path / "ws_new"
    ws.mkdir()
    run_config = SimpleNamespace(
        discovery_addresses=[AddressMCP(ip="0.0.0.0", port_min=5000, port_max=5200)],
        workspace_dir=str(ws),
    )
    mcps = asyncio.run(cli._setup_connectivity(run_config, run_id=2))
    assert mcps == ["stub-mcp"]
    return run_config


def test_setup_connectivity_reuses_previous_runs_port_range(monkeypatch, tmp_path):
    ws_prev = tmp_path / "ws_prev"
    ws_prev.mkdir()
    previous = _make_spec(1, str(ws_prev), 6000, 6200)

    # yn: use same port range -> yes; ask: workspace prompt -> keep default
    script = _ScriptedIO(answers=[""], yn_answers=[True])
    run_config = _run_connectivity(monkeypatch, tmp_path, script, previous)

    addr = run_config.discovery_addresses[0]
    assert (addr.port_min, addr.port_max) == (6000, 6200)
    # Deep copy, not aliasing: mutating one run's address must not touch the other
    assert addr is not previous.config.discovery_addresses[0]
    assert script.answers == [] and script.yn_answers == []


def test_setup_connectivity_empty_ip_keeps_current_range(monkeypatch, tmp_path):
    ws_prev = tmp_path / "ws_prev"
    ws_prev.mkdir()
    previous = _make_spec(1, str(ws_prev), 6000, 6200)

    # yn: use same port range -> no; ask: empty IP escapes back to the
    # current (= previous run's) range, then workspace prompt keeps default.
    script = _ScriptedIO(answers=["", ""], yn_answers=[False])
    run_config = _run_connectivity(monkeypatch, tmp_path, script, previous)

    addr = run_config.discovery_addresses[0]
    assert (addr.port_min, addr.port_max) == (6000, 6200)
    assert script.answers == []


def test_setup_connectivity_accepts_overlapping_range_without_constraint(
    monkeypatch, tmp_path
):
    ws_prev = tmp_path / "ws_prev"
    ws_prev.mkdir()
    previous = _make_spec(1, str(ws_prev), 5000, 5200)

    # yn: use same port range -> no; ask: an invalid range (min > max) is
    # re-prompted, then a range fully overlapping run #1 is accepted as-is.
    script = _ScriptedIO(
        answers=["0.0.0.0", "9000", "8000", "0.0.0.0", "5000", "5200", ""],
        yn_answers=[False],
    )
    run_config = _run_connectivity(monkeypatch, tmp_path, script, previous)

    addr = run_config.discovery_addresses[0]
    assert (addr.ip, addr.port_min, addr.port_max) == ("0.0.0.0", 5000, 5200)
    assert script.answers == []  # every scripted answer was consumed


# ---------------------------------------------------------------------------
# Recovery options: restore-cache file picker
# ---------------------------------------------------------------------------


def _write_eval_notes(
    path, model="test/model", mode="iterative", status="completed", final=None
):
    data = {
        "smolagent_model_id": model,
        "eval_mode": mode,
        "status": status,
        "final_results": final
        if final is not None
        else {
            "ver_success": 8,
            "ver_total": 9,
            "sr_success": 3,
            "sr_total": 9,
            "avg_cbs": 0.92,
            "total_cost": 17.9,
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _make_notes_dir(tmp_path, monkeypatch, files_with_mtime):
    """Create a fake run_notes/evaluations dir and point the CLI at it."""
    notes_dir = tmp_path / "evaluations"
    notes_dir.mkdir()
    for name, mtime in files_with_mtime:
        p = _write_eval_notes(notes_dir / name)
        os.utime(p, (mtime, mtime))
    monkeypatch.setattr(evaluation_cli, "_EVAL_NOTES_DIR", notes_dir)
    return notes_dir


def test_ask_recovery_options_yes_asks_for_notes_file(monkeypatch, tmp_path):
    # newer mtime → listed first; picking "2" must select the older file
    _make_notes_dir(
        tmp_path,
        monkeypatch,
        [
            ("newer_iterative.json", 2000.0),
            ("older_one_shot.json", 1000.0),
        ],
    )
    script = _ScriptedIO(answers=["1", "2"], yn_answers=[True])
    monkeypatch.setattr(evaluation_cli, "_ask", script.ask)
    monkeypatch.setattr(evaluation_cli, "_ask_yn", script.ask_yn)
    cli = EvaluationCLI(SimpleNamespace())

    start_row, restore_cache, notes_file = cli._ask_recovery_options()

    assert start_row == 0
    assert restore_cache is True
    assert Path(notes_file).name == "older_one_shot.json"
    assert script.answers == [] and script.yn_answers == []


def test_ask_recovery_options_no_skips_file_picker(monkeypatch, tmp_path):
    _make_notes_dir(tmp_path, monkeypatch, [("a_iterative.json", 1000.0)])
    script = _ScriptedIO(answers=["5"], yn_answers=[False])
    monkeypatch.setattr(evaluation_cli, "_ask", script.ask)
    monkeypatch.setattr(evaluation_cli, "_ask_yn", script.ask_yn)
    cli = EvaluationCLI(SimpleNamespace())

    start_row, restore_cache, notes_file = cli._ask_recovery_options()

    assert start_row == 4
    assert restore_cache is False
    assert notes_file is None
    assert script.answers == []  # file picker never prompted


def test_choose_restore_notes_file_auto_keeps_detection(monkeypatch, tmp_path):
    _make_notes_dir(tmp_path, monkeypatch, [("a_iterative.json", 1000.0)])
    script = _ScriptedIO(answers=["a"], yn_answers=[])
    monkeypatch.setattr(evaluation_cli, "_ask", script.ask)
    cli = EvaluationCLI(SimpleNamespace())

    assert cli._choose_restore_notes_file() is None


def test_choose_restore_notes_file_accepts_bare_filename(monkeypatch, tmp_path):
    notes_dir = _make_notes_dir(
        tmp_path,
        monkeypatch,
        [
            ("20260917_chemistry_deepseek-v4-iterative.json", 1000.0),
        ],
    )
    script = _ScriptedIO(
        answers=["20260917_chemistry_deepseek-v4-iterative.json"], yn_answers=[]
    )
    monkeypatch.setattr(evaluation_cli, "_ask", script.ask)
    cli = EvaluationCLI(SimpleNamespace())

    chosen = cli._choose_restore_notes_file()

    assert chosen == str(notes_dir / "20260917_chemistry_deepseek-v4-iterative.json")


def test_choose_restore_notes_file_reprompts_on_bad_choice(monkeypatch, tmp_path):
    _make_notes_dir(tmp_path, monkeypatch, [("a_iterative.json", 1000.0)])
    script = _ScriptedIO(answers=["99", "missing.json", "1"], yn_answers=[])
    monkeypatch.setattr(evaluation_cli, "_ask", script.ask)
    cli = EvaluationCLI(SimpleNamespace())

    assert Path(cli._choose_restore_notes_file()).name == "a_iterative.json"
    assert script.answers == []


def test_choose_restore_notes_file_without_candidates(monkeypatch, tmp_path):
    _make_notes_dir(tmp_path, monkeypatch, [])
    script = _ScriptedIO(answers=["1"], yn_answers=[])
    monkeypatch.setattr(evaluation_cli, "_ask", script.ask)
    cli = EvaluationCLI(SimpleNamespace())

    assert cli._choose_restore_notes_file() is None
    assert script.answers == ["1"]  # never consumed: no file list to pick from


def test_summarise_notes_file_prefers_final_results(tmp_path):
    p = _write_eval_notes(
        tmp_path / "summary.json", mode="iterative", status="completed"
    )
    summary = EvaluationCLI._summarise_notes_file(p)
    assert "test/model" in summary
    assert "iterative" in summary
    assert "VER 8/9" in summary
    assert "SR 3/9" in summary


def test_summarise_notes_file_falls_back_to_total_eval(tmp_path):
    p = tmp_path / "flat.json"
    p.write_text(json.dumps({"model": "test/model", "total_eval": 4}), encoding="utf-8")
    summary = EvaluationCLI._summarise_notes_file(p)
    assert "4 evals" in summary
    assert "VER" not in summary


if __name__ == "__main__":
    test_launch_queue_is_sequential_and_survives_failure()
    print("✅ evaluation CLI sequential-queue smoke checks passed")

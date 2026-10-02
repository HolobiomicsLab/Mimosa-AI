#!/usr/bin/env python3
"""Tests for the evaluation CLI sequential queue and port-range prompts."""

import asyncio
import os
import sys
from types import SimpleNamespace

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sources.cli.evaluation_cli as evaluation_cli
from config import AddressMCP
from sources.cli.evaluation_cli import EvalRunSpec, EvaluationCLI


def _make_spec(run_id: int, workspace: str, port_min: int = 5000, port_max: int = 5200):
    config = SimpleNamespace(
        discovery_addresses=[AddressMCP(ip="0.0.0.0", port_min=port_min, port_max=port_max)],
        workspace_dir=workspace,
        smolagent_model_id="test/model",
    )
    return EvalRunSpec(run_id=run_id, config=config, eval_mode="one_shot", csv_runs_limit=1)


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


def test_setup_connectivity_accepts_overlapping_range_without_constraint(monkeypatch, tmp_path):
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


if __name__ == "__main__":
    test_launch_queue_is_sequential_and_survives_failure()
    print("✅ evaluation CLI sequential-queue smoke checks passed")

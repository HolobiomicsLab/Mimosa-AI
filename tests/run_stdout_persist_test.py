#!/usr/bin/env python3
"""``run_stdout.log`` persistence through the real ``evolve_generation``.

The directive LLM reads ``run_stdout`` as its fallback diagnosis when a
generation has no parent gradient; auditors need that exact text on disk.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution_engine_stop_test import _make_engine  # noqa: E402

from sources.core.evolution_engine import EvolutionEngine  # noqa: E402
from sources.core.schema import IndividualRun  # noqa: E402

RAW_STDOUT = "\x1b[31mError:\x1b[0m conformer_search failed\nline 2 é\n"


class _RawOrchestrator:
    """Returns a fixed raw stdout; succeeds or fails per flag."""

    def __init__(self, executed: bool):
        self.calls = 0
        self.executed = executed

    async def orchestrate_workflow(self, **kw):
        self.calls += 1
        uuid = f"20261014_000000_{1000 + self.calls:08d}"
        code = "print('w')" if self.executed else ""
        return f"{RAW_STDOUT}gen{self.calls}", uuid, code, self.executed


def _run_one_generation(tmp_path: Path, executed: bool) -> Path:
    """Drive one real generation; return the workflows dir."""
    engine, _, _ = _make_engine(tmp_path, _RawOrchestrator(executed))
    asyncio.run(
        engine.evolve_generation(
            [IndividualRun(goal="g", prompt="p", max_depth=1)],
            rewards_history=[],
            assertion_history=[],
            enable_evolution=False,
        )
    )
    return Path(engine.workflow_dir)


def test_successful_generation_writes_verbatim_stdout(tmp_path: Path):
    wf_dir = _run_one_generation(tmp_path, executed=True)
    log = wf_dir / "20261014_000000_00001001" / "run_stdout.log"
    assert log.read_text(encoding="utf-8") == f"{RAW_STDOUT}gen1"


def test_failed_execution_still_writes_error_text(tmp_path: Path):
    wf_dir = _run_one_generation(tmp_path, executed=False)
    log = wf_dir / "20261014_000000_00001001" / "run_stdout.log"
    assert log.read_text(encoding="utf-8") == f"{RAW_STDOUT}gen1"


def test_write_failure_is_logged_not_raised(tmp_path: Path, caplog):
    blocker = tmp_path / "file_not_dir"
    blocker.write_text("x")
    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.workflow_dir = str(blocker)
    engine.logger = logging.getLogger("test-run-stdout")
    with caplog.at_level(logging.WARNING, logger="test-run-stdout"):
        engine._persist_run_stdout("u1", RAW_STDOUT)
    assert "run_stdout write failed for u1" in caplog.text


def test_falsy_uuid_skips_write(tmp_path: Path):
    engine = types.SimpleNamespace(workflow_dir=str(tmp_path), logger=None)
    EvolutionEngine._persist_run_stdout(engine, "", RAW_STDOUT)
    assert list(tmp_path.iterdir()) == []

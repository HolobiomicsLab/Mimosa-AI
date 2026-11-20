#!/usr/bin/env python3
"""``run_stdout.log`` persistence through the real ``evolve_generation``.

The directive LLM reads a parent's ``run_stdout.log`` (through
``WorkflowInfo.run_stdout``) as an unverified raw log when that parent has
no verifier gradient; auditors need that exact text on disk.
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
from sources.core.orchestrator import WorkflowOrchestrator  # noqa: E402
from sources.core.schema import IndividualRun  # noqa: E402
from sources.core.workflow_factory import WorkflowFactory  # noqa: E402

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


def _run_generation(tmp_path: Path, orchestrator) -> Path:
    """Drive one real generation; return the workflows dir."""
    engine, _, _ = _make_engine(tmp_path, orchestrator)
    asyncio.run(
        engine.evolve_generation(
            [IndividualRun(goal="g", prompt="p", max_depth=1)],
            rewards_history=[],
            assertion_history=[],
            enable_evolution=False,
        )
    )
    return Path(engine.workflow_dir)


def _run_one_generation(tmp_path: Path, executed: bool) -> Path:
    """Drive one real generation with a fixed raw stdout."""
    return _run_generation(tmp_path, _RawOrchestrator(executed))


class _NoUuidFailingFactory:
    """Fails before any run folder exists, so the error has no uuid."""

    def __init__(self):
        self.calls = 0

    async def craft_workflow(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError(f"Failed to load tools code: boom {self.calls}")


def _real_orchestrator(factory) -> WorkflowOrchestrator:
    """A real orchestrator; only its config and factory are stubbed."""
    orchestrator = WorkflowOrchestrator.__new__(WorkflowOrchestrator)
    orchestrator.config = types.SimpleNamespace(
        orchestrator_choose_model=False, literrature_grounding=False
    )
    orchestrator.workflow_factory = factory
    orchestrator.workflow_runner = None
    return orchestrator


def _genotype_failing_factory(tmp_path: Path) -> WorkflowFactory:
    """A real factory whose LLM step fails after the run folder exists."""
    template = tmp_path / "template.py"
    template.write_text("# template\n", encoding="utf-8")
    config = types.SimpleNamespace(
        workflow_dir=str(tmp_path / "workflows"),
        memory_dir=str(tmp_path / "memory"),
        schema_code_path=str(template),
        smolagent_factory_code_path=str(template),
        prompt_workflow_creator="",
    )
    factory = WorkflowFactory(config)

    async def _no_tools():
        return "", ""

    def _bad_syntax(*args, **kwargs):
        raise ValueError("LLM generated invalid Python syntax: bad call")

    factory.load_tools_code = _no_tools
    factory.create_workflow_genotype_code = _bad_syntax
    return factory


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


def test_two_uuidless_generation_failures_keep_two_logs(tmp_path: Path):
    """A later failure must not overwrite an earlier failure's log."""
    orchestrator = _real_orchestrator(_NoUuidFailingFactory())
    _run_generation(tmp_path, orchestrator)
    wf_dir = _run_generation(tmp_path, orchestrator)
    logs = wf_dir.glob("generation_failed_*/run_stdout.log")
    assert sorted(log.read_text(encoding="utf-8") for log in logs) == [
        "WORKFLOW_GENERATION_ERROR: Failed to load tools code: boom 1",
        "WORKFLOW_GENERATION_ERROR: Failed to load tools code: boom 2",
    ]


def test_genotype_failure_writes_log_into_its_own_run_folder(tmp_path: Path):
    """The factory's run folder, not a shared one, keeps the error."""
    factory = _genotype_failing_factory(tmp_path)
    wf_dir = _run_generation(tmp_path, _real_orchestrator(factory))
    folders = [p for p in wf_dir.iterdir() if p.is_dir()]
    assert len(folders) == 1
    assert not folders[0].name.startswith("generation_failed")
    log = folders[0] / "run_stdout.log"
    assert log.read_text(encoding="utf-8") == (
        "WORKFLOW_GENERATION_ERROR: LLM generated invalid Python syntax: bad call"
    )

"""Tests for the ``blocked`` task status.

A task is blocked, not failed, when its workflow never ran because the
sandbox could not be provisioned. The signal is structural: the orchestrator
raises a typed error, reports it under a fixed prefix, the engine records it
on the run and stops evolving, and the planner marks the task instead of
retrying it.
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from sources.core.evolution_engine import EvolutionEngine
from sources.core.orchestrator import (
    DEPENDENCY_INSTALL_ERROR_PREFIX,
    DependencyInstallError,
    WorkflowOrchestrator,
)
from sources.core.planner import Planner
from sources.core.schema import IndividualRun, PlanStep, Task, TaskStatus
from sources.core.workflow_runner import ExecutionStatus
from sources.utils.planner_visualization import PlannerVisualizer

INSTALL_STDERR = "ERROR: No matching distribution found for somepkg==9.9.9"


class _FailingInstallRunner:
    async def install_dependencies(self, deps):
        return SimpleNamespace(status=ExecutionStatus.FAILED, stderr=INSTALL_STDERR)


def _orchestrator(runner):
    orch = object.__new__(WorkflowOrchestrator)
    orch.config = SimpleNamespace(
        runner_requirements=["somepkg==9.9.9"], literrature_grounding=False
    )
    orch.workflow_runner = runner
    orch._notify_execution_failure = lambda *args, **kwargs: None
    orch._prompt_agents_model_list = lambda prompt: prompt
    orch._generate_workflow_code = _fake_generation
    return orch


async def _fake_generation(*args, **kwargs):
    return "print('never runs')", "genotype", "uuid-1"


def test_schema_defaults():
    assert TaskStatus("blocked") is TaskStatus.BLOCKED
    assert Task(name="t", description="d").blocked_reason is None
    assert IndividualRun(goal="g", prompt="p").blocked_reason is None


def test_requirements_install_raises_typed_error():
    orch = _orchestrator(_FailingInstallRunner())
    with pytest.raises(DependencyInstallError) as excinfo:
        asyncio.run(orch.workflow_requirements_install())
    assert isinstance(excinfo.value, RuntimeError)
    assert INSTALL_STDERR in str(excinfo.value)


def test_orchestrate_reports_install_failure_under_prefix():
    orch = _orchestrator(_FailingInstallRunner())
    output, uuid, genotype, executed = asyncio.run(
        orch.orchestrate_workflow("goal", craft_instructions="")
    )
    assert executed is False
    assert uuid == "uuid-1"
    assert genotype == "genotype"
    assert output.startswith(DEPENDENCY_INSTALL_ERROR_PREFIX)
    assert INSTALL_STDERR in output


def test_orchestrate_leaves_other_sandbox_errors_unprefixed():
    orch = _orchestrator(_FailingInstallRunner())

    async def crash(*args, **kwargs):
        raise RuntimeError("segmentation fault in the workflow")

    orch._execute_in_sandbox = crash
    output, _, _, executed = asyncio.run(
        orch.orchestrate_workflow("goal", craft_instructions="")
    )
    assert executed is False
    assert not output.startswith(DEPENDENCY_INSTALL_ERROR_PREFIX)


def test_engine_reads_reason_from_prefix_only():
    prefixed = f"{DEPENDENCY_INSTALL_ERROR_PREFIX}: Dependency installation failed: x"
    assert EvolutionEngine._blocked_reason_from(prefixed, executed=False) == prefixed
    assert EvolutionEngine._blocked_reason_from(prefixed, executed=True) is None
    unprefixed = "Traceback: pip install failed"
    assert EvolutionEngine._blocked_reason_from(unprefixed, False) is None
    assert EvolutionEngine._blocked_reason_from("", False) is None


def _planner_with_runs(run_factory):
    planner = object.__new__(Planner)
    planner.tts = None
    planner.task_history = []
    planner._workspace_files_before_step = []
    planner._build_knowledge_aware_task = lambda task: task
    planner._get_workspace_files = lambda: []
    calls = {"evolve": 0}

    async def evolve_runs(task, judge, original_task=None, **kwargs):
        calls["evolve"] += 1
        return [run_factory()]

    planner.evolve_runs = evolve_runs
    return planner, calls


def _step():
    return PlanStep(
        name="fit model", goal_context="goal", task="fit a model", cost=1, score=0.0
    )


def test_blocked_run_marks_task_and_step_without_retry():
    reason = (
        f"{DEPENDENCY_INSTALL_ERROR_PREFIX}: Dependency installation failed: "
        f"{INSTALL_STDERR}"
    )
    planner, calls = _planner_with_runs(
        lambda: IndividualRun(goal="g", prompt="p", blocked_reason=reason)
    )
    step = _step()
    asyncio.run(planner.run_attempts({}, max_attempts=5, step=step, judge=False))
    assert calls["evolve"] == 1
    assert step.status is TaskStatus.BLOCKED
    assert [t.status for t in planner.task_history] == [TaskStatus.BLOCKED]
    assert planner.task_history[0].blocked_reason == reason
    assert planner._blocked_reason_for("fit model") == reason


def test_ordinary_failure_still_retries():
    planner, calls = _planner_with_runs(
        lambda: IndividualRun(goal="g", prompt="p", state_result={"success": [False]})
    )
    step = _step()
    asyncio.run(planner.run_attempts({}, max_attempts=3, step=step, judge=False))
    assert calls["evolve"] == 3
    assert step.status is not TaskStatus.BLOCKED
    assert {t.status for t in planner.task_history} == {TaskStatus.FAILED}
    assert planner._blocked_reason_for("fit model") == "no reason recorded"


def test_visualizer_has_a_colour_for_blocked():
    viz = object.__new__(PlannerVisualizer)
    assert viz._get_status_color(TaskStatus.BLOCKED) == PlannerVisualizer.COLOR_BLOCKED

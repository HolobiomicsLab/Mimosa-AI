"""Tests for Planner dependency checking.

Focus: ``_can_execute_step`` verifies that a completed dependency actually
produced the files it declared in ``expected_outputs``.
"""

import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from sources.core.planner import Planner
from sources.core.schema import PlanStep, Task, TaskStatus


def _planner_with(workspace: str, task_history: list[Task]) -> Planner:
    """Build a Planner without running its heavy __init__ (EvolutionEngine, LLM)."""
    planner = object.__new__(Planner)
    planner.workspace_path = workspace
    planner.task_history = task_history
    return planner


def _completed_dep(expected_outputs: list[str]) -> Task:
    return Task(
        name="step_a",
        description="produces outputs",
        status=TaskStatus.COMPLETED,
        expected_outputs=expected_outputs,
    )


def _dependent_step() -> PlanStep:
    return PlanStep(
        name="step_b",
        goal_context="ctx",
        task="consume step_a outputs",
        cost=1,
        score=0.0,
        depends_on=["step_a"],
    )


def test_dependency_outputs_present_allows_execution():
    """A completed dependency whose declared outputs exist unblocks the next step."""
    with tempfile.TemporaryDirectory() as workspace:
        (Path(workspace) / "result.csv").write_text("a,b\n1,2\n")
        planner = _planner_with(workspace, [_completed_dep(["result.csv"])])

        can_execute, missing = planner._can_execute_step(_dependent_step())

    assert can_execute is True
    assert missing == []


def test_dependency_outputs_missing_blocks_execution():
    """A completed dependency whose declared outputs are absent blocks the next step."""
    with tempfile.TemporaryDirectory() as workspace:
        (Path(workspace) / "unrelated.txt").write_text("noise")
        planner = _planner_with(workspace, [_completed_dep(["result.csv"])])

        can_execute, missing = planner._can_execute_step(_dependent_step())

    assert can_execute is False
    assert missing == ["step_a[missing_outputs:result.csv]"]


def test_dependency_without_declared_outputs_is_not_verified():
    """A dependency declaring no outputs is treated as satisfied once completed."""
    with tempfile.TemporaryDirectory() as workspace:
        planner = _planner_with(workspace, [_completed_dep([])])

        can_execute, missing = planner._can_execute_step(_dependent_step())

    assert can_execute is True
    assert missing == []


def test_incomplete_dependency_blocks_execution():
    """A dependency that has not completed blocks the next step by name."""
    with tempfile.TemporaryDirectory() as workspace:
        pending = _completed_dep(["result.csv"])
        pending.status = TaskStatus.PENDING
        planner = _planner_with(workspace, [pending])

        can_execute, missing = planner._can_execute_step(_dependent_step())

    assert can_execute is False
    assert missing == ["step_a"]


if __name__ == "__main__":
    test_dependency_outputs_present_allows_execution()
    test_dependency_outputs_missing_blocks_execution()
    test_dependency_without_declared_outputs_is_not_verified()
    test_incomplete_dependency_blocks_execution()
    print("planner dependency tests passed")

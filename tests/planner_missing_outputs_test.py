"""A step that did not produce its declared outputs must not bank the miss.

Issue #196. Both inner branches of the acceptance block used to assign
``TaskStatus.COMPLETED`` and ``break``, differing only in ``print_ok`` against
``print_warn``:

    outputs_produced, missing_outputs = self._verify_expected_outputs(step)
    step.status = TaskStatus.COMPLETED
    if outputs_produced:  print_ok(...);   break
    else:                 print_warn(...); break

So ``_verify_expected_outputs`` computed the right answer and nothing acted on
it. Observed on a real run: step ``reproduction_spec_analysis`` was accepted at
0.799 without writing ``workspace/analysis/iimn_reproduction_plan.md``, and the
run died at the *next* step's dependency gate with an attempt still unspent.

The rule now: while attempts remain, spend one on producing the deliverable; on
the last attempt keep ``COMPLETED`` so the dependency gate still names which
output is missing for which step, rather than replacing that precise message
with a generic step failure.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))

from sources.core.failure_fingerprint import compute_failure_fingerprint  # noqa: E402, F401
from sources.core.planner import Planner  # noqa: E402
from sources.core.schema import PlanStep, TaskStatus  # noqa: E402


def _accept_block() -> str:
    """The acceptance block of run_attempts, as source."""
    src = inspect.getsource(Planner.run_attempts)
    start = src.index("if evolve_success and attempt_score >= 0.7:")
    return src[start:start + 2200]


# ---- the defect, stated structurally --------------------------------------


def test_the_two_outcomes_no_longer_do_the_same_thing():
    """Producing the outputs and not producing them must diverge.

    Counts ``break`` as a *statement* — a first version counted the substring
    and matched the word inside the explanatory comment below it.
    """
    head = _accept_block()
    head = head[:head.index("step.missing_outputs")]
    breaks = re.findall(r"^\s*break\s*$", head, re.M)
    assert len(breaks) == 1, (
        f"only the produced-outputs path may break before the miss is handled; "
        f"found {len(breaks)}"
    )


def test_completed_is_not_assigned_before_the_outputs_are_checked():
    """The old code set COMPLETED unconditionally, above the if."""
    block = _accept_block()
    check = block.index("_verify_expected_outputs")
    first_completed = block.index("TaskStatus.COMPLETED")
    assert first_completed > check, "status must depend on the verification result"


def test_a_remaining_attempt_is_spent_rather_than_banked():
    block = _accept_block()
    assert "if attempt < max_attempts:" in block
    tail = block[block.index("if attempt < max_attempts:"):]
    assert "continue" in tail[:tail.index("step.status = TaskStatus.COMPLETED")]


def test_the_last_attempt_still_completes_so_the_gate_can_name_the_output():
    """Replacing it with a generic failure would lose the precise message."""
    block = _accept_block()
    tail = block[block.index("if attempt < max_attempts:"):]
    assert "TaskStatus.COMPLETED" in tail
    assert "TaskStatus.FAILED" not in tail


def test_the_miss_is_recorded_on_the_step():
    assert "step.missing_outputs" in _accept_block()


# ---- the field that carries it --------------------------------------------


def test_plan_step_carries_missing_outputs_and_defaults_empty():
    step = PlanStep(name="s", goal_context="g", task="t", cost=0, score=0.0)
    assert step.missing_outputs == []
    assert step.status is TaskStatus.PENDING


def test_missing_outputs_is_not_shared_between_steps():
    """A mutable default would make one step's miss appear on every other."""
    a = PlanStep(name="a", goal_context="g", task="t", cost=0, score=0.0)
    b = PlanStep(name="b", goal_context="g", task="t", cost=0, score=0.0)
    a.missing_outputs.append("result.csv")
    assert b.missing_outputs == []


def test_a_completed_step_is_still_distinguishable_from_a_delivering_one():
    """COMPLETED alone no longer implies the deliverable exists."""
    delivered = PlanStep(name="a", goal_context="g", task="t", cost=0, score=0.9,
                         expected_outputs=["result.csv"])
    delivered.status = TaskStatus.COMPLETED
    banked = PlanStep(name="b", goal_context="g", task="t", cost=0, score=0.9,
                      expected_outputs=["result.csv"])
    banked.status = TaskStatus.COMPLETED
    banked.missing_outputs = ["result.csv"]
    assert delivered.status is banked.status
    assert bool(delivered.missing_outputs) != bool(banked.missing_outputs)


# ---- the run-4 scenario ----------------------------------------------------


def test_run_4_would_have_spent_its_third_attempt():
    """0.799 on attempt 2 of 3, declared output absent: retry, do not accept."""
    block = _accept_block()
    # attempt 2 < max_attempts 3 -> the retry branch is the one that applies
    retry = block[block.index("if attempt < max_attempts:"):]
    assert re.search(r"print_warn\([^)]*retrying", retry, re.S), (
        "the operator must be told the step is being retried for missing outputs"
    )

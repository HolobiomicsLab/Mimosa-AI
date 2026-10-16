"""Tests for the planner's opt-in plan-approval gate.

``Planner._generate_plan_with_human_validation`` has always carried an
approval loop, but nothing wired it to configuration and its prompt read
stdin unconditionally. These tests pin the wiring: off by default, feedback
regenerates the plan with the feedback in the goal, and a headless run
refuses the gate up front instead of dying on ``input()``.
"""

import asyncio
import inspect
import io
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from sources.core.planner import Planner, UserInterventionRequired


def _planner_with_plan_stub(monkeypatch, decisions):
    """Build a bare planner whose planning and prompting are scripted.

    Args:
        monkeypatch: pytest fixture.
        decisions: Successive ``(approved, feedback)`` answers the fake
            prompt returns; the prompt fails if asked more often than that.

    Returns:
        The planner and a dict recording the goals ``make_plan`` received.
    """
    planner = object.__new__(Planner)
    seen = {"goals": [], "prompts": 0}
    answers = list(decisions)

    def make_plan(system_prompt, goal_prompt, max_retries=3):
        seen["goals"].append(goal_prompt)
        return object()

    def prompt(plan):
        seen["prompts"] += 1
        assert answers, "the operator was asked more often than scripted"
        return answers.pop(0)

    monkeypatch.setattr(planner, "_read_prompt", lambda: "system", raising=False)
    monkeypatch.setattr(planner, "make_plan", make_plan, raising=False)
    monkeypatch.setattr(planner, "_display_plan", lambda plan: None, raising=False)
    monkeypatch.setattr(
        planner, "_request_human_plan_validation", prompt, raising=False
    )
    return planner, seen


def test_default_path_never_prompts(monkeypatch):
    planner, seen = _planner_with_plan_stub(monkeypatch, decisions=[])
    planner._generate_plan_with_human_validation("goal")
    assert seen["goals"] == ["goal"]
    assert seen["prompts"] == 0


def test_feedback_regenerates_plan_with_feedback_in_goal(monkeypatch):
    planner, seen = _planner_with_plan_stub(
        monkeypatch, decisions=[(False, "use fewer steps"), (True, "")]
    )
    planner._generate_plan_with_human_validation("goal", human_approve=True)
    assert seen["prompts"] == 2
    assert len(seen["goals"]) == 2
    assert seen["goals"][0] == "goal"
    assert "use fewer steps" in seen["goals"][1]
    assert seen["goals"][1].startswith("goal")


def test_prompt_refuses_when_stdin_is_not_a_tty(monkeypatch):
    planner = object.__new__(Planner)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    with pytest.raises(UserInterventionRequired):
        planner._request_human_plan_validation(object())


def test_prompt_enter_approves_and_text_is_feedback(monkeypatch):
    planner = object.__new__(Planner)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(planner, "_display_plan", lambda plan: None, raising=False)
    answers = iter(["", "  add a validation step  "])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert planner._request_human_plan_validation(object()) == (True, "")
    assert planner._request_human_plan_validation(object()) == (
        False,
        "add a validation step",
    )


def test_start_planner_refuses_headless_gate_before_planning(monkeypatch):
    planner = object.__new__(Planner)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    def make_plan(*args, **kwargs):
        raise AssertionError("planning must not run when nobody can approve it")

    monkeypatch.setattr(planner, "_read_prompt", lambda: "system", raising=False)
    monkeypatch.setattr(planner, "make_plan", make_plan, raising=False)
    with pytest.raises(UserInterventionRequired):
        asyncio.run(planner.start_planner("goal", human_approve=True))


def test_start_planner_gate_is_off_by_default():
    parameter = inspect.signature(Planner.start_planner).parameters["human_approve"]
    assert parameter.default is False


def test_config_field_defaults_off_and_round_trips():
    config = Config()
    assert config.planner_human_approve is False
    config.from_json({"planner_human_approve": True})
    assert config.planner_human_approve is True
    assert config.jsonify()["planner_human_approve"] is True

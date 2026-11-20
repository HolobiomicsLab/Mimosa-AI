#!/usr/bin/env python3
"""``variation_mode = "resample"``: best-of-N control without workflow selection.

In resample mode every generation must be a fresh draw from the session
goal. No parent is selected (no disk scan, no archive draw, no crossover,
no mutation), at session start or between generations. The engine tests
drive the REAL ``evolve_generation`` recursion with stubbed collaborators.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config  # noqa: E402
from sources.core.evolution_engine import EvolutionEngine  # noqa: E402
from sources.core.planner import Planner  # noqa: E402
from sources.core.schema import IndividualRun  # noqa: E402
from sources.core.stopping import StopController  # noqa: E402

GOAL = "train a model\nREPRODUCTION NOTE"


class _RecordingOrchestrator:
    """Records the craft instructions of every orchestrated generation."""

    def __init__(self):
        self.prompts: list[str] = []

    async def orchestrate_workflow(self, **kw):
        self.prompts.append(kw["craft_instructions"])
        uuid = f"20261020_000000_{len(self.prompts):08d}"
        return "ok", uuid, "print('workflow')", True


def _no_selection(*args, **kwargs):
    raise AssertionError("resample mode must not select a workflow")


def _make_engine(tmp_path: Path, variation_mode: str):
    """Real EvolutionEngine with collaborators stubbed; parents always qualify."""
    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.workflow_dir = str(tmp_path)
    engine.config = types.SimpleNamespace(
        learned_score_threshold=0.9, variation_mode=variation_mode
    )
    engine.initial_population = 1  # evolve mode would mutate from gen 1
    engine.orchestrator = _RecordingOrchestrator()
    engine.variation = types.SimpleNamespace(
        record_offspring_gradient=lambda *a, **kw: None,
        last_variation_state={},
        seed_genome_prompt=lambda goal: f"seed::{goal}",
        mutation_prompt=lambda *a, **kw: "mutation prompt",
        crossover_prompt=lambda *a, **kw: "crossover prompt",
    )
    engine.selection = types.SimpleNamespace(
        strategy=types.SimpleNamespace(value="qd"),
        validate_survivor=lambda **kw: {
            "absolute_improvement": 0.0, "valid": True, "confidence": 1.0,
        },
    )
    engine.notifier = types.SimpleNamespace(send_message=lambda *a, **kw: None)
    engine.logger = logging.getLogger("test-variation-mode")
    engine.select_parent_workflow = _no_selection
    for name in (
        "_log_iteration_start", "_log_iteration_completion",
        "_update_visualizations", "_refresh_evolution_tree",
        "_refresh_archive_projection", "_persist_qd_archive",
    ):
        setattr(engine, name, lambda *a, **kw: None)
    engine._save_final_plots = lambda *a, **kw: ""
    engine.show_answers = lambda answers: None
    snapshots: list[dict] = []
    engine._persist_iteration_metrics = lambda uuid, snap: snapshots.append(snap)

    async def _evaluate(executed, judge, uuid, answers, rubric, history):
        return "verifier", 0.0

    engine._evaluate_and_calculate_cost = _evaluate
    return engine, snapshots


def test_resample_draws_every_generation_fresh_from_the_goal(tmp_path: Path):
    """Every generation gets the seed prompt of the gen-0 goal, no parents."""
    engine, snapshots = _make_engine(tmp_path, "resample")
    engine._evaluate_exploration_probe = _no_selection
    run0 = IndividualRun(
        goal=GOAL, prompt=f"seed::{GOAL}", max_depth=4, original_task="raw task"
    )
    controller = StopController(
        mode="none", min_iter=1, patience=3, threshold=0.9, max_iter=4,
        verifier_kind="policy_bt",
    )

    runs = asyncio.run(
        engine.evolve_generation(
            [run0], rewards_history=[], enable_evolution=True,
            stop_controller=controller,
        )
    )

    assert engine.orchestrator.prompts == [f"seed::{GOAL}"] * 4
    assert [r.evolution_kind for r in runs] == ["seed"] * 4
    assert all(r.parent_uuids == [] for r in runs)
    assert [s["variation_mode"] for s in snapshots] == ["resample"] * 4


def test_resample_session_start_ignores_the_default_template(tmp_path: Path):
    """Callers that rely on the fixed default template still get no parent."""
    engine, _ = _make_engine(tmp_path, "resample")
    start = inspect.signature(EvolutionEngine.start_workflow_evolution)
    default_template = start.parameters["template_uuid"].default

    assert engine._session_start_parent(GOAL, default_template) is None
    assert engine._session_start_parent(GOAL, None) is None


def test_resample_planner_skips_cached_workflow_reuse():
    """A near-perfect cached workflow must not replace the session."""
    planner = Planner.__new__(Planner)
    planner.config = types.SimpleNamespace(variation_mode="resample")
    planner.wf_selector = types.SimpleNamespace(select_best_workflows=_no_selection)
    fresh = [IndividualRun(goal=GOAL, prompt="p")]

    async def _start(**kw):
        return fresh

    planner.evolve = types.SimpleNamespace(start_workflow_evolution=_start)

    runs = asyncio.run(planner.evolve_runs(GOAL, judge=False, cached_wf_allow=True))

    assert runs is fresh


def test_config_variation_mode_round_trips_and_rejects_unknown_values():
    """Saved resample mode reloads; a typo fails loudly instead of evolving."""
    configured = Config()
    configured.from_json({"variation_mode": "Resample"})
    restored = Config()
    restored.from_json(configured.jsonify())

    assert restored.variation_mode == "resample"
    with pytest.raises(ValueError, match="variation_mode"):
        Config().from_json({"variation_mode": "redraw"})

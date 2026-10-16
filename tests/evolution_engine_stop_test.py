#!/usr/bin/env python3
"""StopController + engine stop wiring (stop plan 2026-10-08 §3 C5).

Pure controller tests (no engine) plus two engine-level tests that drive
the REAL ``evolve_generation`` recursion with stubbed collaborators:

- the champion-plateau rule stops a relative-reward run before the hard
  cap and persists the stop state in ``run_metrics`` (C2/C4);
- the hard generation cap fires even when the final generation errored
  (N3 — an errored gen at the cap used to recurse past ``max_depth``
  without bound).

HP 2026-10-08 replay: gens 1-3 tie honestly (identical science), gen 4
wins on a trivial freshness count ("3 vs 2"). The legacy 0.9 threshold
stopped that run at generation 4 of 16; the plateau rule keeps going.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.core.evolution_engine import EvolutionEngine  # noqa: E402
from sources.core.schema import IndividualRun  # noqa: E402
from sources.core.stopping import StopController  # noqa: E402


def _ctl(**kw) -> StopController:
    """Controller with the shipped defaults; kwargs override."""
    params = dict(
        mode="auto",
        min_iter=6,
        patience=3,
        threshold=0.9,
        max_iter=16,
        verifier_kind="policy_bt",
    )
    params.update(kw)
    return StopController(**params)


# --------------------------------------------------------------- pure ----
def test_auto_resolves_mode_from_verifier_kind():
    """auto -> plateau for relative verifiers, threshold for absolute."""
    for kind in ("hybrid", "policy_bt"):
        c = _ctl(verifier_kind=kind)
        assert c.mode == "plateau", kind
        assert c.selects_champion is True
        assert c.relative is True
    for kind in ("legacy", "gold", None, ""):
        c = _ctl(verifier_kind=kind)
        assert c.mode == "threshold", kind
        assert c.selects_champion is False
        assert c.relative is False
    # explicit modes are kept; "none" still ships the champion for
    # relative verifiers (the argmax is meaningless across generations)
    assert _ctl(mode="none").selects_champion is True
    assert _ctl(mode="threshold").selects_champion is False
    assert _ctl(mode="plateau", verifier_kind="legacy").selects_champion is False
    try:
        _ctl(mode="bogus")
        raise AssertionError("invalid mode accepted")
    except ValueError:
        pass


def test_relative_mode_ignores_clean_sweep_before_plateau():
    """HP 2026-10-08 replay: [none, tie, tie, won...] must NOT stop at
    gen 4 — a 1.0 arrival reward (clean sweep of predecessors) is not a
    convergence signal for a relative verifier."""
    c = _ctl()  # auto -> plateau for policy_bt
    c.observe("g1", 0, 0.5, None, failed=False)  # first real gen: champion
    assert c.champion == "g1" and c.since_change == 0
    c.observe("g2", 1, 0.5, {"g1": "tie"}, failed=False)
    c.observe("g3", 2, 0.5, {"g1": "tie", "g2": "tie"}, failed=False)
    assert c.since_change == 2
    # gen 4 sweeps every predecessor -> reward 1.0. Legacy rule: STOP.
    # Plateau rule: gen 4 is the new champion, keep going.
    c.observe("g4", 3, 1.0, {"g1": "won", "g2": "won", "g3": "won"},
              failed=False)
    assert c.champion == "g4" and c.since_change == 0
    stop, why = c.should_stop(3)
    assert stop is False, why
    # won x3 more: each beats the incumbent champion; still no stop
    c.observe("g5", 4, 1.0, {"g4": "won"}, failed=False)
    c.observe("g6", 5, 1.0, {"g5": "won"}, failed=False)
    c.observe("g7", 6, 1.0, {"g6": "won"}, failed=False)
    assert c.champion == "g7"
    for it in (4, 5, 6):
        assert c.should_stop(it) == (False, "")
    # three ties against the champion AFTER min_iter -> plateau
    c.observe("g8", 7, 0.5, {"g7": "tie"}, failed=False)
    c.observe("g9", 8, 0.5, {"g7": "tie"}, failed=False)
    c.observe("g10", 9, 0.5, {"g7": "tie"}, failed=False)
    stop, why = c.should_stop(9)
    assert stop is True and why.startswith("plateau: champion g7 "), why


def test_min_iterations_block_early_plateau_and_max_still_caps():
    """min_iter gates the plateau; the max cap stops regardless of mode."""
    c = _ctl(min_iter=4, patience=1)
    c.observe("g1", 0, 0.5, None, failed=False)  # champion
    c.observe("g2", 1, 0.5, {"g1": "tie"}, failed=False)
    # since_change (1) already >= patience (1) but only 2 gens done < 4
    assert c.should_stop(1) == (False, "")
    c.observe("g3", 2, 0.5, {"g1": "tie"}, failed=False)
    assert c.should_stop(2) == (False, "")
    c.observe("g4", 3, 0.5, {"g1": "tie"}, failed=False)
    stop, why = c.should_stop(3)  # 4 gens done >= min 4, since 3 >= 1
    assert stop is True and why.startswith("plateau:"), why
    # mode "none" never plateaus but the hard cap still applies
    n = _ctl(mode="none")
    for i in range(15):
        n.observe(f"g{i}", i, 0.5, {}, failed=False)
    assert n.should_stop(14) == (False, "")  # 15 of 16 done, no rule fires
    stop, why = n.should_stop(15)  # 16/16
    assert stop is True and why.startswith("max:"), why
    # min_iter is clamped to the cap so it can never block the last gen
    assert _ctl(min_iter=99).min_iter == 16


def test_champion_changes_only_on_win_against_champion():
    """tie/lost/missing vs the champion keep it; beating a NON-champion
    predecessor does not dethrone the champion."""
    c = _ctl()
    c.observe("g1", 0, 0.5, None, failed=False)
    c.observe("g2", 1, 0.5, {"g1": "tie"}, failed=False)
    assert (c.champion, c.since_change) == ("g1", 1)
    c.observe("g3", 2, 0.5, {"g1": "lost"}, failed=False)
    assert (c.champion, c.since_change) == ("g1", 2)
    c.observe("g4", 3, 0.5, {}, failed=False)  # champion not compared
    assert (c.champion, c.since_change) == ("g1", 3)
    c.observe("g5", 4, 1.0, {"g1": "won", "g2": "won"}, failed=False)
    assert (c.champion, c.since_change) == ("g5", 0)
    # g6 beats g1 (a predecessor) but only ties the champion g5
    c.observe("g6", 5, 1.0, {"g5": "tie", "g1": "won"}, failed=False)
    assert (c.champion, c.since_change) == ("g5", 1)


def test_failed_generations_advance_the_plateau():
    """Errored/crashed generations count as generations without change —
    but never become champion themselves."""
    c = _ctl(min_iter=3, patience=3)
    c.observe("g1", 0, 0.5, None, failed=False)
    c.observe("g2", 1, 0.0, None, failed=True)   # crashed orchestration
    c.observe("", 2, 0.0, None, failed=True)     # no uuid at all
    c.observe("g4", 3, 0.0, None, failed=True)   # short-circuit verdict
    assert c.champion == "g1" and c.since_change == 3
    stop, why = c.should_stop(3)
    assert stop is True and why.startswith("plateau:"), why
    # failures before ANY champion: nothing to compare against yet
    fresh = _ctl()
    fresh.observe("", 0, 0.0, None, failed=True)
    assert fresh.champion is None and fresh.since_change == 0
    assert fresh.should_stop(5) == (False, "")  # only the cap can stop


def test_threshold_mode_reproduces_todays_stop_with_min_gate():
    """threshold mode keeps the legacy rule (reward >= 0.9) but the new
    min-iteration gate blocks a lucky first-generation absolute score."""
    c = _ctl(mode="threshold", verifier_kind="gold", min_iter=6)
    c.observe("g1", 0, 0.99, {}, failed=False)  # absolute score 0.99
    assert c.should_stop(0) == (False, "")  # done=1 < min 6
    for i in range(1, 5):
        c.observe(f"g{i+1}", i, 0.5, {}, failed=False)
    assert c.should_stop(4) == (False, "")
    c.observe("g6", 5, 0.95, {}, failed=False)  # done=6 >= min, >= 0.9
    stop, why = c.should_stop(5)
    assert stop is True and why.startswith("threshold: reward 0.950"), why
    # below the threshold: run on
    c2 = _ctl(mode="threshold", verifier_kind="gold")
    c2.observe("g1", 0, 0.89, {}, failed=False)
    assert c2.should_stop(0) == (False, "")
    # min_iter=1 restores the ungated legacy behaviour (stop at gen 1)
    c3 = _ctl(mode="threshold", verifier_kind="gold", min_iter=1)
    c3.observe("g1", 0, 0.99, {}, failed=False)
    assert c3.should_stop(0)[0] is True
    # a failed generation's None-ish reward (0.0) never trips the gate
    c3.observe("g2", 1, 0.0, {}, failed=True)
    assert c3.should_stop(1) == (False, "")


def test_bt_mode_run_is_not_stuck_never_stopping():
    """Bradley-Terry rewards never reach 0.9 (a clean sweep of 8 preds
    scores ~0.785), so the legacy threshold meant 'never stop'; the
    plateau rule stops the run once the champion stops changing."""
    c = _ctl(mode="plateau", verifier_kind="policy_bt", min_iter=6,
             patience=3)
    rewards = [0.5, 0.584, 0.612, 0.645, 0.668, 0.701, 0.723, 0.741]
    for i, r in enumerate(rewards):  # every generation ties the champion
        outcomes = {} if i == 0 else {f"g{j}": "tie" for j in range(i)}
        c.observe(f"g{i}", i, r, outcomes, failed=False)
        if i < 5:
            assert c.should_stop(i) == (False, ""), i
    stop, why = c.should_stop(5)  # 6 done >= min 6, since 5 >= patience 3
    assert stop is True and why.startswith("plateau:"), why


def test_short_reason_tags_long_uuids():
    c = _ctl()
    long_id = "20261008_065714_15f9e48aaaa"
    c.observe(long_id, 0, 0.5, None, failed=False)
    c.observe("g2", 1, 0.5, {long_id: "tie"}, failed=False)
    c.observe("g3", 2, 0.5, {long_id: "tie"}, failed=False)
    c.observe("g4", 3, 0.5, {long_id: "tie"}, failed=False)
    stop, why = c.should_stop(5)
    assert stop is True and long_id[-8:] in why and long_id not in why


# ----------------------------------------------------- engine wiring ----
class _StubOrchestrator:
    """Counts orchestration calls; returns success or failure runs."""

    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail

    async def orchestrate_workflow(self, **kw):
        self.calls += 1
        uuid = f"20261008_000000_{1000 + self.calls:08d}"
        if self.fail:
            return "workflow failed", uuid, "", False
        return "ok", uuid, "print('workflow')", True


def _make_engine(tmp_path: Path, orchestrator: _StubOrchestrator):
    """A real EvolutionEngine with only its collaborators stubbed.

    ``evolve_generation`` itself, the metrics snapshot builder and the
    stop wiring run unmodified; each evaluation writes a state_result
    whose verifier block carries ``pair_outcomes`` tying every
    predecessor (the honest-tie HP shape).
    """
    engine = EvolutionEngine.__new__(EvolutionEngine)
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    engine.workflow_dir = str(wf_dir)
    engine.config = types.SimpleNamespace(learned_score_threshold=0.9)
    engine.initial_population = 2
    engine.orchestrator = orchestrator
    engine.variation = types.SimpleNamespace(
        record_offspring_gradient=lambda *a, **kw: None,
        last_variation_state={},
        seed_genome_prompt=lambda goal: "seed prompt",
        mutation_prompt=lambda *a, **kw: "mutation prompt",
        crossover_prompt=lambda *a, **kw: "crossover prompt",
    )
    engine.selection = types.SimpleNamespace(
        strategy=types.SimpleNamespace(value="qd"),
        validate_survivor=lambda **kw: {
            "absolute_improvement": 0.0,
            "valid": True,
            "confidence": 1.0,
            "admit_rejected": False,
            "behaviour_descriptor": [],
            "qd_score": 0.5,
            "novelty_score": 0.5,
        },
    )
    engine.notifier = types.SimpleNamespace(send_message=lambda *a, **kw: None)
    engine.logger = logging.getLogger("test-stop")
    engine.select_parent_workflow = lambda goal, template_uuid=None: ([], False)
    engine._log_iteration_start = lambda *a, **kw: None
    engine._log_iteration_completion = lambda *a, **kw: None
    engine._update_visualizations = lambda *a, **kw: None
    engine._refresh_evolution_tree = lambda *a, **kw: None
    engine._refresh_archive_projection = lambda *a, **kw: None
    engine._save_final_plots = lambda *a, **kw: ""
    engine.extract_agents_behavior = lambda state_result: ""
    engine.show_answers = lambda answers: None

    snapshots: list[dict] = []
    engine._persist_iteration_metrics = (
        lambda uuid, snap: snapshots.append(snap)
    )
    engine._persist_qd_archive = lambda entry: None

    history: list[str] = []

    async def _evaluate(executed, judge, uuid, answers, rubric, assertion_hist):
        d = wf_dir / uuid
        d.mkdir(parents=True, exist_ok=True)
        (d / "state_result.json").write_text(
            json.dumps(
                {
                    "goal": "g",
                    "answers": ["ok"],
                    "evaluation": {
                        "verifier": {
                            "overall_score": 0.5,
                            "pair_outcomes": {u: "tie" for u in history},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        history.append(uuid)
        return "verifier", 0.0

    engine._evaluate_and_calculate_cost = _evaluate
    return engine, snapshots, history


def test_engine_stops_on_plateau_and_persists_stop_state(tmp_path: Path):
    """C2/C4: the wired controller stops the run on the plateau rule
    (before the hard cap) and the stop state lands in run_metrics."""
    orch = _StubOrchestrator()
    engine, snapshots, history = _make_engine(tmp_path, orch)
    controller = _ctl(min_iter=3, patience=2, max_iter=6)

    runs = asyncio.run(
        engine.evolve_generation(
            [IndividualRun(goal="g", prompt="p", max_depth=6)],
            rewards_history=[],
            assertion_history=[],
            enable_evolution=True,
            stop_controller=controller,
        )
    )
    # gen1 champion -> gen2 tie (since 1) -> gen3 tie (since 2) = plateau
    # (3 generations, NOT the 6-generation hard cap)
    assert orch.calls == 3, orch.calls
    assert len(runs) == 3
    assert controller.champion == history[0]
    assert controller.since_change == 2
    last = snapshots[-1]
    assert last["stop_reason"].startswith("plateau:"), last["stop_reason"]
    assert last["champion_uuid"] == history[0]
    assert last["since_change"] == 2
    # earlier iterations recorded no stop
    assert snapshots[0]["stop_reason"] is None
    assert snapshots[0]["champion_uuid"] == history[0]


def test_engine_hard_caps_at_max_depth_even_on_error(tmp_path: Path):
    """N3: an errored generation AT the cap must not recurse past
    max_depth (the old `and not on_error` guard let repeated failures
    extend the run without bound)."""
    orch = _StubOrchestrator(fail=True)
    engine, snapshots, _ = _make_engine(tmp_path, orch)
    controller = _ctl(mode="none", min_iter=1, patience=1, max_iter=2)

    runs = asyncio.run(
        engine.evolve_generation(
            [IndividualRun(goal="g", prompt="p", max_depth=2)],
            rewards_history=[],
            assertion_history=[],
            enable_evolution=True,
            stop_controller=controller,
        )
    )
    assert orch.calls == 2  # hard stop: gen 1 and gen 2 only
    assert len(runs) == 2
    # failed generations never become champion; the metrics record the cap
    assert controller.champion is None
    assert snapshots[-1]["stop_reason"].startswith("max:")
    assert snapshots[-1]["champion_uuid"] is None


if __name__ == "__main__":
    import inspect
    import tempfile

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            print(f"✅ {name}")
    print("all stop tests passed")

"""Current tournament ratings must drive QD, parent choice, and capsule export."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from main import add_config_arguments, apply_config_overrides
from sources.core.evolution_engine import (
    EvolutionEngine,
    _refresh_policy_bt_rewards,
    _select_capsule_run,
    _stop_pair_outcomes,
)
from sources.core.schema import IndividualRun
from sources.core.selection import PopulationMember, SelectionPressure
from sources.core.stopping import StopController
from sources.core.workflow_selection import WorkflowSelector


def _run(uuid: str, reward: float, iteration: int = 0) -> IndividualRun:
    """Build a completed policy run for integration checks.

    Args:
        uuid (str): Workflow identity in the test tournament.
        reward (float): Historical score at arrival.
        iteration (int): Generation index.

    Returns:
        IndividualRun: A scored run with an available genotype descriptor.
    """
    run = IndividualRun(
        goal="compare measured results", prompt="improve the result",
        current_uuid=uuid, reward=reward, iteration_count=iteration, code="pass",
        state_result={"evaluation": {"verifier": {
            "verifier_kind": "policy_bt", "reward": reward,
            "overall_score": reward, "arrival_reward": reward,
            "n_pairs": iteration, "reward_fallback": False,
        }}},
    )
    run.behaviour_descriptor = [1.0, 0.0]
    return run


def _publish(run: IndividualRun, rewards: dict[str, float]) -> None:
    """Attach the evaluator's latest tournament result to a completed run.

    Args:
        run (IndividualRun): Latest generation in the test session.
        rewards (dict[str, float]): Newly fitted ratings by workflow identity.

    Returns:
        None. The run receives the evaluator's result fields.
    """
    run.state_result["evaluation"]["verifier"].update(
        tournament_rewards=rewards, tournament_version=3,
    )


def test_refreshed_ratings_change_survivor_eviction():
    """A once-strong parent must not evict a now-better child on its old score."""
    selection = SelectionPressure({}, population_size=1)
    incumbent = _run("incumbent", 0.9)
    selection.validate_survivor([incumbent], [incumbent])
    child = _run("child", 0.6, 1)
    _publish(child, {"incumbent": 0.1, "child": 0.6})

    _refresh_policy_bt_rewards([incumbent, child], selection)
    verdict = selection.validate_survivor([incumbent], [child])

    assert verdict["absolute_improvement"] == pytest.approx(0.5)
    assert [member.uuid for member in selection.archive] == ["child"]
    assert incumbent.reward_uncapped == 0.1
    previous = incumbent.state_result["evaluation"]["verifier"]
    assert previous["overall_score"] == 0.1
    assert previous["arrival_reward"] == 0.9
    assert previous["tournament_version"] == 3


def test_single_member_refresh_keeps_novelty_and_length_penalty():
    """Refreshing quality must not skip a singleton or discard its QD terms."""
    selection = SelectionPressure({}, length_penalty_baseline_chars=100)
    selection._archive = [PopulationMember(
        iteration=0, uuid="parent", reward=0.2, cost=0.0,
        genotype_chars=150, novelty_score=0.5,
    )]

    selection.refresh_rewards({"parent": 0.8})

    assert selection.archive[0].qd_score == pytest.approx(0.7)


def test_capsule_uses_current_ratings_not_arrival_or_champion():
    """The global BT leader can differ from the latest head-to-head champion."""
    first, champion, latest = _run("first", 0.1), _run("champion", 0.9, 1), _run("latest", 0.7, 2)
    runs = [first, champion, latest]
    _publish(latest, {"first": 0.8, "champion": 0.6, "latest": 0.4})
    controller = StopController(verifier_kind="policy_bt")
    controller.observe("first", 0, 0.1, {}, False)
    controller.observe("champion", 1, 0.9, {"first": "won"}, False)
    _refresh_policy_bt_rewards(runs, SelectionPressure({}))

    selected, _ = _select_capsule_run(runs, controller, "policy_bt")

    assert controller.champion == "champion"
    assert selected is first
    assert first.state_result["evaluation"]["verifier"]["arrival_reward"] == 0.1


def test_capsule_tie_break_reads_current_outcomes_not_arrival_ones():
    """Review finding: in-memory pair outcomes were frozen at arrival while
    ratings were refreshed. A later re-comparison turned gen 2's arrival win
    over gen 0 into a tie; the tie must then ship the earlier run."""
    gen0, gen2, latest = _run("gen0", 0.5), _run("gen2", 0.6, 2), _run("latest", 0.3, 3)
    gen2.state_result["evaluation"]["verifier"]["pair_outcomes"] = {"gen0": "win"}
    _publish(latest, {"gen0": 0.6, "gen2": 0.6, "latest": 0.3})
    latest.state_result["evaluation"]["verifier"]["tournament_outcomes"] = {
        "gen0": {"gen2": "tie", "latest": "win"}, "gen2": {"gen0": "tie", "latest": "win"},
        "latest": {"gen0": "loss", "gen2": "loss"}}
    _refresh_policy_bt_rewards([gen0, gen2, latest], SelectionPressure({}))

    selected, _ = _select_capsule_run([gen0, gen2, latest], StopController(verifier_kind="policy_bt"), "policy_bt")

    assert gen2.state_result["evaluation"]["verifier"]["pair_outcomes"] == {"gen0": "tie", "latest": "win"}
    assert selected is gen0


@pytest.mark.parametrize("kind, expected", [("hybrid", "champion"), ("legacy", "highest")])
def test_older_channels_keep_capsule_selection(kind, expected):
    """Only policy BT replaces the existing capsule choice rule."""
    runs = [IndividualRun(goal="g", prompt="p", current_uuid=uuid, reward=reward)
            for uuid, reward in [("champion", 0.5), ("highest", 0.9)]]
    controller = StopController(verifier_kind=kind)
    controller.observe("champion", 0, 0.5, {}, False)

    selected, _ = _select_capsule_run(runs, controller, kind)

    assert selected.current_uuid == expected


def test_policy_outcomes_advance_existing_plateau_controller():
    """A new contract 'win' must reset plateau instead of counting as a tie."""
    controller = StopController(verifier_kind="policy_bt", min_iter=1, patience=1)
    controller.observe("previous", 0, 0.5, {}, False)
    verifier = {"verifier_kind": "policy_bt", "pair_outcomes": {"previous": "win"}}

    controller.observe("new", 1, 0.7, _stop_pair_outcomes(verifier), False)

    assert controller.champion == "new"
    assert controller.should_stop(1)[0] is False
    hybrid = {"pair_outcomes": {"previous": "won"}}
    assert _stop_pair_outcomes(hybrid) == {"previous": "won"}


def _save_workflow(root: Path, uuid: str, reward: float) -> None:
    """Write the persisted workflow consumed by real parent discovery.

    Args:
        root (Path): Temporary workflow directory.
        uuid (str): Workflow directory name.
        reward (float): Current persisted rating.

    Returns:
        None. Files are created only below the temporary directory.
    """
    folder = root / uuid
    folder.mkdir(exist_ok=True)
    (folder / f"workflow_genotype_{uuid}.py").write_text("pass\n")
    (folder / "state_result.json").write_text(json.dumps({
        "goal": "compare measured results", "evaluation": {"verifier": {
            "verifier_kind": "policy_bt", "overall_score": reward,
        }},
    }))


def test_parent_selection_reloads_persisted_tournament_ratings(tmp_path, monkeypatch):
    """Cached WorkflowInfo objects must not keep an earlier tournament leader."""
    _save_workflow(tmp_path, "old-leader", 0.9)
    _save_workflow(tmp_path, "new-leader", 0.2)
    config = SimpleNamespace(workflow_dir=str(tmp_path), verifier_kind="policy_bt")
    selector = WorkflowSelector(config)
    assert selector.workflows_info["old-leader"].overall_score == 0.9
    assert selector.workflows_info["new-leader"].overall_score == 0.2
    _save_workflow(tmp_path, "old-leader", 0.1)
    _save_workflow(tmp_path, "new-leader", 0.8)
    monkeypatch.setattr(selector, "cosine_similarity", lambda left, right: 1.0)
    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.config, engine.workflow_dir = config, str(tmp_path)
    engine.workflow_selector = selector
    engine.selection = SelectionPressure({}, strategy="greedy")

    selected, crossover = engine.select_parent_workflow(
        "compare measured results", n_parents=1, crossover_rate=0.0,
    )

    assert crossover is False
    assert [workflow.uuid for workflow in selected] == ["new-leader"]
    assert selected[0].overall_score == 0.8


def test_disabled_reexecution_survives_json_roundtrip():
    """A saved execution opt-out must not become an enabled run after reload."""
    configured = Config()
    configured.from_json({"policy_bt_reexecute": "false", "policy_bt_visual_channel": False})
    restored = Config()
    restored.from_json(configured.jsonify())

    assert restored.policy_bt_reexecute is False
    assert restored.policy_bt_visual_channel is False


def test_config_rejects_retired_policy_reward():
    """An old win-rate request must not silently run a different reward model."""
    with pytest.raises(ValueError, match="bradley_terry"):
        Config().from_json({"policy_bt_reward": "win_rate"})


def test_cli_can_disable_and_reenable_reexecution():
    """Both BooleanOptionalAction paths must override the loaded config."""
    config = Config()
    parser = argparse.ArgumentParser()
    add_config_arguments(parser, config)
    for flag, expected in [("--no-policy_bt_reexecute", False), ("--policy_bt_reexecute", True)]:
        args = parser.parse_args([flag, "--policy_bt_repair_rounds", "0"])
        args.max_evolve_iterations = None
        apply_config_overrides(args, config)
        assert config.policy_bt_reexecute is expected
        assert config.policy_bt_repair_rounds == 0


def test_switching_old_hybrid_config_requires_supported_policy_reward():
    """An unused old policy knob stays harmless until that channel is selected."""
    config = Config()
    config.from_json({"verifier_kind": "hybrid", "policy_bt_reward": "win_rate"})
    parser = argparse.ArgumentParser()
    add_config_arguments(parser, config)
    args = parser.parse_args(["--verifier_kind", "policy_bt"])
    args.max_evolve_iterations = None

    with pytest.raises(ValueError, match="bradley_terry"):
        apply_config_overrides(args, config)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

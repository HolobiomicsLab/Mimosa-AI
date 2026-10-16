#!/usr/bin/env python3
"""Capsule selection must ignore first-generation fallback rewards (E37 R1).

bulk_modulus counterfactual: the first generation's reward is the mean
claim score (a fallback — no rivals to compare against yet), which lived
on a different scale from every later generation's pairwise win-rate and
outranked them all in the final argmax (0.89 fallback vs a max 0.70
pairwise reward). gen 0 shipped; gen 11 (SR=true) did not.
"""

import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.core.evolution_engine import _select_best_run  # noqa: E402
from sources.core.schema import IndividualRun  # noqa: E402
from sources.core.workflow_info import WorkflowInfo  # noqa: E402


def _run(uuid, reward, iteration, fallback=False):
    return IndividualRun(
        goal="g",
        prompt="p",
        reward=reward,
        iteration_count=iteration,
        current_uuid=uuid,
        reward_is_fallback=fallback,
    )


def test_fallback_reward_excluded_from_argmax():
    """bulk_modulus: the gen-0 fallback 0.89 must not beat gen 11's 0.70."""
    runs = [
        _run("gen0", 0.89, 0, fallback=True),
        _run("gen11", 0.70, 11),
    ]
    assert _select_best_run(runs).current_uuid == "gen11"
    # without the flag the same rewards pick gen0 — the bug being fixed
    runs[0].reward_is_fallback = False
    assert _select_best_run(runs).current_uuid == "gen0"


def test_highest_real_reward_wins_among_non_fallbacks():
    runs = [
        _run("g0", 0.89, 0, fallback=True),
        _run("g5", 0.70, 5),
        _run("g11", 0.42, 11),
    ]
    assert _select_best_run(runs).current_uuid == "g5"


def test_all_fallbacks_pick_the_best_fallback():
    """No real pairwise run exists -> the best fallback ships (argmax over
    the fallbacks), not the blindly-last run; a single candidate is itself."""
    first = _run("g0", 0.9, 0, fallback=True)
    last = _run("g1", 0.4, 1, fallback=True)
    assert _select_best_run([first, last]) is first
    assert _select_best_run([last]) is last


def test_runs_without_workspace_return_none():
    assert _select_best_run([IndividualRun(goal="g", prompt="p")]) is None
    assert _select_best_run([]) is None


def _wf_dir(tmp_path: Path, evaluation: dict) -> Path:
    d = tmp_path / "wf"
    d.mkdir(exist_ok=True)
    (d / "state_result.json").write_text(
        json.dumps({"goal": "g", "evaluation": evaluation}), encoding="utf-8"
    )
    return d


def test_workflow_info_flags_mean_claim_fallback(tmp_path: Path):
    wf = WorkflowInfo(
        "u1", _wf_dir(tmp_path, {"verifier": {"overall_score": 0.89,
                                              "reward_fallback": "mean_claim"}})
    )
    assert wf.reward_is_fallback is True
    assert wf.overall_score == 0.89  # same evaluation block backs both reads


def test_workflow_info_real_and_short_circuit_rewards_are_not_fallback(tmp_path: Path):
    pairwise = WorkflowInfo(
        "u2", _wf_dir(tmp_path, {"verifier": {"overall_score": 0.7,
                                              "reward_fallback": None}})
    )
    assert pairwise.reward_is_fallback is False
    failed = WorkflowInfo(
        "u3", _wf_dir(tmp_path, {"verifier": {"overall_score": 0.0,
                                              "reward_fallback": "short_circuit"}})
    )
    # a short-circuit 0.0 is a measured failure, not a scale-mismatched fallback
    assert failed.reward_is_fallback is False


def test_workflow_info_without_verifier_block_is_not_fallback(tmp_path: Path):
    legacy = WorkflowInfo(
        "u4", _wf_dir(tmp_path, {"generic": {"overall_score": 0.5}})
    )
    assert legacy.reward_is_fallback is False
    bare = WorkflowInfo("u5", _wf_dir(tmp_path, {}))
    assert bare.reward_is_fallback is False


def _verifier_eval(overall, fallback=None, n_pairs=None, n_wins=None):
    """The hybrid-verifier evaluation block as persisted to state_result."""
    block = {"overall_score": overall}
    if fallback is not None:
        block["reward_fallback"] = fallback
    if n_pairs is not None:
        block["n_pairs"] = n_pairs
    if n_wins is not None:
        block["n_wins"] = n_wins
    return {"verifier": block}


def _pains_brenk_runs(gen13_flag_on_run):
    """The pains_brenk run_1 shape: 13 crashed gens, a mean-claim fallback
    gen 13 (0.745), a 0.0 gen 14, and the real pairwise winner gen 15 (0.25)."""
    runs = []
    for i in range(13):  # gens 0-12: generation failed, uuid still assigned
        runs.append(IndividualRun(
            goal="g", prompt="p", reward=0.0, iteration_count=i,
            current_uuid=f"crashed{i}", state_result={},
        ))
    runs.append(IndividualRun(
        goal="g", prompt="p", reward=0.745, iteration_count=13,
        current_uuid="gen13",
        reward_is_fallback=gen13_flag_on_run,
        state_result={"evaluation": _verifier_eval(0.745, "mean_claim", 0, 0)},
    ))
    runs.append(IndividualRun(
        goal="g", prompt="p", reward=0.0, iteration_count=14,
        current_uuid="gen14",
        state_result={"evaluation": _verifier_eval(0.0, None, 1, 0)},
    ))
    runs.append(IndividualRun(
        goal="g", prompt="p", reward=0.25, iteration_count=15,
        current_uuid="gen15",
        state_result={"evaluation": _verifier_eval(0.25, None, 2, 1)},
    ))
    return runs


def test_pains_brenk_scenario_ships_real_pairwise_generation():
    """N1 repro: 12 crashed gens pad the candidate list, gen 13's 0.745
    reward is a mean_claim fallback on a different scale — the capsule must
    ship gen 15 (highest REAL pairwise reward, 0.25), with the engine-side
    flag wired (True) or not (recovered from state_result)."""
    assert _select_best_run(_pains_brenk_runs(True)).current_uuid == "gen15"
    assert _select_best_run(_pains_brenk_runs(False)).current_uuid == "gen15"


def test_select_best_run_recovers_lost_flag_from_state_result(caplog):
    """The persisted reward_fallback record is ground truth: a run whose
    engine-side flag was lost is still excluded and a warning is logged."""
    import logging as _logging

    with caplog.at_level(_logging.WARNING, logger="sources.core.evolution_engine"):
        best = _select_best_run(_pains_brenk_runs(False))
    assert best.current_uuid == "gen15"
    assert "reward_fallback=mean_claim" in caplog.text
    assert "gen13" in caplog.text


def test_select_best_run_treats_zero_pair_high_reward_as_fallback(caplog):
    """Legacy artifact without the reward_fallback key: a positive reward
    with n_pairs=0/n_wins=0 while siblings carry real pairwise records is a
    suspected fallback — excluded from the argmax with a warning."""
    import logging as _logging

    legacy = IndividualRun(
        goal="g", prompt="p", reward=0.9, iteration_count=0, current_uuid="old",
        state_result={"evaluation": _verifier_eval(0.9, None, 0, 0)},
    )
    real = IndividualRun(
        goal="g", prompt="p", reward=0.25, iteration_count=1, current_uuid="new",
        state_result={"evaluation": _verifier_eval(0.25, None, 1, 1)},
    )
    with caplog.at_level(_logging.WARNING, logger="sources.core.evolution_engine"):
        assert _select_best_run([legacy, real]).current_uuid == "new"
    assert "potential fallback" in caplog.text
    # a zero-reward no-pairs run cannot pollute the argmax and is left alone
    zero = IndividualRun(
        goal="g", prompt="p", reward=0.0, iteration_count=2, current_uuid="zero",
        state_result={"evaluation": _verifier_eval(0.0, None, 0, 0)},
    )
    assert _select_best_run([zero, real]).current_uuid == "new"


def test_reward_and_flag_read_from_same_evaluation_block(tmp_path: Path):
    """A state_result written by two evaluators: the score (0.42, generic)
    and the fallback flag must come from the SAME (generic) block — never
    the verifier block's mean_claim record."""
    mixed = WorkflowInfo("u6", _wf_dir(tmp_path, {
        "generic": {"overall_score": 0.42},
        "verifier": {"overall_score": 0.745, "reward_fallback": "mean_claim"},
    }))
    assert mixed.overall_score == 0.42
    assert mixed.reward_is_fallback is False
    # verifier-first state_result: score and flag both from verifier
    verifier = WorkflowInfo("u7", _wf_dir(tmp_path,
        _verifier_eval(0.745, "mean_claim")))
    assert verifier.overall_score == 0.745
    assert verifier.reward_is_fallback is True


def test_n1_residual_prefers_measured_fallback_over_crashed_pool():
    """protein_protein/vasp_chgcar: crashed gens (reward 0, no fallback
    flag — the verifier never ran) must not crowd out the only successful
    generation just because its reward is a mean-claim fallback."""
    runs = [
        _run("crash0", 0.0, 1),
        _run("crash1", 0.0, 2),
        _run("fb", 0.6, 2, fallback=True),
    ]
    assert _select_best_run(runs).current_uuid == "fb"


def test_n1_residual_all_crashed_no_fallback_stays_crashed():
    """Nothing measured anything: keep current behavior (last crashed gen)."""
    runs = [_run("crash0", 0.0, 1), _run("crash1", None, 2)]
    assert _select_best_run(runs).current_uuid == "crash1"


def test_n1_residual_mixed_pool_prefers_real_pairwise():
    """A real pairwise win still beats a higher fallback reward (E37 R1)."""
    runs = [_run("fb", 0.8, 1, fallback=True), _run("real", 0.3, 2)]
    assert _select_best_run(runs).current_uuid == "real"


def test_n1_residual_fallback_reward_zero_not_preferred():
    """A fallback that scored 0 adds nothing over crashed gens: keep the
    last crashed gen."""
    runs = [
        _run("crash0", 0.0, 1),
        _run("crash1", 0.0, 2),
        _run("fb0", 0.0, 2, fallback=True),
    ]
    assert _select_best_run(runs).current_uuid == "crash1"


def _run_verifier(uuid, reward, iteration, verifier, fallback=False):
    run = _run(uuid, reward, iteration, fallback=fallback)
    run.state_result = {"evaluation": {"verifier": verifier}}
    return run


def test_unmeasured_prior_excluded_from_argmax_even_when_higher():
    """N9 (E41 phonon gen5): a scorer-blind generation's reward is a neutral
    prior, not a measurement — it must not outrank a measured sibling no
    matter its value."""
    unmeasured = _run_verifier(
        "gen5", 0.9, 5, {"reward_fallback": "unmeasured_prior", "n_scored": 0}
    )
    measured = _run("gen8", 0.5714, 8)
    assert _select_best_run([unmeasured, measured]).current_uuid == "gen8"


def test_unmeasured_prior_rescues_all_zero_crashed_pool():
    """N1-residual + N9: an unmeasurable-but-artifact-bearing generation
    ships over crashed gens whose workspaces are empty."""
    crashed0 = _run("crash0", 0.0, 1)
    crashed1 = _run("crash1", 0.0, 2)
    unmeasured = _run_verifier(
        "gen2", 0.5, 3, {"reward_fallback": "unmeasured_prior", "n_scored": 0}
    )
    assert _select_best_run([crashed0, crashed1, unmeasured]).current_uuid == "gen2"


def test_workflow_info_flags_unmeasured_prior(tmp_path: Path):
    wf = WorkflowInfo(
        "u1",
        _wf_dir(
            tmp_path,
            {"verifier": {"overall_score": 0.5, "reward_fallback": "unmeasured_prior"}},
        ),
    )
    assert wf.reward_is_fallback is True
    assert wf.overall_score == 0.5  # same evaluation block backs both reads

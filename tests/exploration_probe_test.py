#!/usr/bin/env python3
"""Frontier probe (review 2026-10-09 §4.3) — trigger, injection, persistence.

The probe replaces the imaginary target gradient: exploration moved from
the verifier into the variation engine as an honest, labelled block that
is triggered by search STATE (champion plateau / all-tie objective
metrics), injected ALONGSIDE the directive — never instead of the
diagnosis — and persisted for audits. These tests defend review tests
12-18:

12. plateau trigger fires at ``since_change >= patience - 1``
13. a probe is guaranteed before the plateau stop
14. the diagnosis is never replaced (prompt keeps directive + diagnoses)
15. the block is labelled ("NOT a measured loss")
16. numeric targets, when explicitly enabled, are sign/bounds/integer safe
17. the trigger sequence is deterministic for a fixed seed
18. ``run_metrics.json`` persists ``variation_state.exploration``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sources.core.variation_engine as variation_engine_module  # noqa: E402
from sources.core.evolution_engine import (  # noqa: E402
    EvolutionEngine,
    evaluate_exploration_probe,
)
from sources.core.schema import IndividualRun  # noqa: E402
from sources.core.stopping import StopController  # noqa: E402
from sources.core.variation_engine import (  # noqa: E402
    VariationEngine,
    exploration_state,
    frontier_target,
    render_exploration_block,
)


def _ctl(**kw) -> StopController:
    """Plateau controller with shipped defaults; kwargs override."""
    params = dict(
        mode="plateau", min_iter=1, patience=3, threshold=0.9,
        max_iter=16, verifier_kind="policy_bt",
    )
    params.update(kw)
    return StopController(**params)


def _run_with_summary(value=-2, pairs=None, status="resolved", gen=0):
    """One completed IndividualRun carrying a verifier metric_summary."""
    return IndividualRun(
        goal="g", prompt="p", iteration_count=gen,
        state_result={
            "evaluation": {
                "verifier": {
                    "pair_outcomes": pairs if pairs is not None else {"prev": "tie"},
                    "metric_summary": {
                        "min_energy": {"status": status, "value": value},
                    },
                }
            }
        },
    )


_PROBE = {
    "triggered": True,
    "reason": "plateau: champion 3a1b2128 unchanged for 2 generations",
    "champion": "20261009_111111_3a1b2128",
    "since_change": 2,
    "n_pairs": 7,
    "generations_compared": 2,
    "metrics": [{"id": "min_energy", "value": -2}],
}


# --------------------------------------------------------------- pure ----
def test_probe_plateau_trigger_fires_at_patience_minus_one():
    """Test 12: since_change >= patience-1 triggers; since_change == 0
    (fresh champion, improving streak) never does with base_rate 0."""
    c = _ctl(patience=3)
    fired = []
    for i in range(4):
        c.observe(f"g{i}", i, 0.5, {}, failed=False)
        probe = evaluate_exploration_probe(c, [])
        fired.append(probe["triggered"])
        if probe["triggered"]:
            assert probe["reason"].startswith(
                f"plateau: champion g0 unchanged for {c.since_change} generations"
            )
    # champion g0 (since 0), then 1, 2, 3 non-improving generations
    assert fired == [False, False, True, True], fired
    # explicit knob overrides the patience-1 default
    c2 = _ctl(patience=3)
    for i in range(4):
        c2.observe(f"g{i}", i, 0.5, {}, failed=False)
    assert not evaluate_exploration_probe(c2, [], explore_after=5)["triggered"]
    assert evaluate_exploration_probe(c2, [], explore_after=2)["triggered"]


def test_probe_not_triggered_while_improving():
    """Defect C9: a run that keeps producing new champions (state healthy,
    base rate 0) never triggers — the old imaginary gradient fired at 25%."""
    c = _ctl()
    for i in range(5):
        c.observe(f"g{i}", i, 0.5, {f"g{i-1}": "won"} if i else {}, failed=False)
        assert c.since_change == 0
        assert not evaluate_exploration_probe(c, [])["triggered"]


def test_probe_all_tie_trigger_reads_metric_summary():
    """The OR-branch: identical resolved metrics across the tie window
    trigger even while the champion is fresh (since_change == 0)."""
    c = _ctl()
    c.observe("g0", 0, 0.5, {}, failed=False)
    runs = [_run_with_summary(-2, gen=0), _run_with_summary(-2, gen=1)]
    probe = evaluate_exploration_probe(c, runs)
    assert probe["triggered"]
    assert probe["reason"].startswith("all-tie: last 2 generations")
    assert "min_energy=-2" in probe["reason"]
    assert probe["metrics"] == [{"id": "min_energy", "value": -2}]
    assert probe["n_pairs"] == 1 and probe["generations_compared"] == 2


def test_probe_all_tie_requires_identical_resolved_values():
    """The tie claim is never guessed: differing values, unresolved
    statuses, empty pair sets or a too-short window all refuse."""
    c = _ctl()
    c.observe("g0", 0, 0.5, {}, failed=False)
    # values drifted between the two generations
    drifted = [_run_with_summary(-2, gen=0), _run_with_summary(-1.5, gen=1)]
    assert not evaluate_exploration_probe(c, drifted)["triggered"]
    # latest generation could not resolve the metric
    unresolved = [
        _run_with_summary(-2, gen=0),
        _run_with_summary(-2, gen=1, status="ambiguous"),
    ]
    assert not evaluate_exploration_probe(c, unresolved)["triggered"]
    # no pair was compared at all (generation-1 shape, defect C9)
    no_pairs = [
        _run_with_summary(-2, pairs={}, gen=0),
        _run_with_summary(-2, pairs={}, gen=1),
    ]
    assert not evaluate_exploration_probe(c, no_pairs)["triggered"]
    # only one generation of history
    assert not evaluate_exploration_probe(c, [_run_with_summary(-2)])["triggered"]


def test_probe_base_rate_deterministic_same_seed_same_sequence():
    """Test 17: identical seed + history -> identical trigger sequence."""
    def sequence(seed):
        rng = random.Random(seed)
        out = []
        for _ in range(6):
            c = _ctl()
            c.observe("g0", 0, 0.5, {}, failed=False)
            out.append(evaluate_exploration_probe(c, [], base_rate=0.7, rng=rng))
        return out

    first, second = sequence(42), sequence(42)
    assert [p["triggered"] for p in first] == [p["triggered"] for p in second]
    assert [p["reason"] for p in first] == [p["reason"] for p in second]
    assert any(p["triggered"] for p in first)  # p=0.7 fires within 6 draws
    assert all(
        p["reason"].startswith("base-rate:") for p in first if p["triggered"]
    )
    # base rate 0 never fires on a healthy state
    healthy = _ctl()
    healthy.observe("g0", 0, 0.5, {}, failed=False)
    assert not evaluate_exploration_probe(
        healthy, [], base_rate=0.0, rng=random.Random(0)
    )["triggered"]


# ------------------------------------------------------------ render ----
def test_render_block_is_labelled_and_honest():
    """Test 15: the block carries the label and only measured facts."""
    block = render_exploration_block(_PROBE)
    assert "EXPLORATION PROBE — search policy, NOT a measured loss." in block
    assert block.startswith("<exploration>\n") and block.endswith("</exploration>")
    assert "champion 3a1b2128 unchanged for 2 generations" in block
    assert "20261009_111111_3a1b2128" not in block  # short tag only
    assert "min_energy = -2 everywhere" in block
    assert "7 pairs compared" in block
    assert "The verifier cannot tell whether -2 is optimal." in block
    assert "Do not change reported numbers without a method that produces them." in block


def test_render_block_absent_when_not_triggered():
    assert render_exploration_block(None) == ""
    assert render_exploration_block(
        {"triggered": False, "reason": "", "metrics": []}
    ) == ""
    assert exploration_state(None) == {
        "triggered": False, "reason": "", "metrics": [],
    }
    assert exploration_state(_PROBE)["triggered"] is True


def test_render_block_text_only_by_default():
    """No invented numbers without explicit metric specs (review §4.3
    recommendation): the default block contains no arithmetic target."""
    assert "target" not in render_exploration_block(_PROBE).lower()


def test_frontier_target_sign_bounds_and_integer_safe():
    """Test 16: sign-aware, bounds-clipped, integer-rounded targets."""
    # lower-is-better negative metric moves STRICTLY down (defect C1)
    assert frontier_target(-2, "lower") < -2
    # bounded metric can never exceed the bound (defect C3)
    auc = frontier_target(0.95, "higher", bounds=(0.0, 1.0))
    assert 0.95 < auc <= 1.0
    # integer metric gets an integer target strictly beyond the value
    n = frontier_target(2, "higher", integer=True)
    assert isinstance(n, int) and n > 2
    assert isinstance(frontier_target(-2, "lower", integer=True), int)
    assert frontier_target(-2, "lower", integer=True) < -2
    # zero-value metric still gets a non-zero target (defect C2)
    assert frontier_target(0.0, "higher") == 0.1
    # at the bound no strictly better target exists -> None (no fabrication)
    assert frontier_target(1.0, "higher", bounds=(0.0, 1.0)) is None
    # unknown direction -> None
    assert frontier_target(1.0, "equal") is None


def test_render_target_only_for_bounded_objective_specs():
    """role=invariant metrics are never pushed (defect C5); objective
    bounded metrics get a clipped arithmetic target line."""
    specs = {
        "auc": {"role": "objective", "direction": "higher", "bounds": [0.0, 1.0]},
        "n_optimal": {"role": "invariant"},
    }
    probe = dict(_PROBE, metrics=[
        {"id": "auc", "value": 0.95}, {"id": "n_optimal", "value": 47},
    ])
    block = render_exploration_block(probe, metric_specs=specs)
    assert "auc = 1.0" in block
    assert "n_optimal" not in block.split("Optional arithmetic target")[-1]


# ---------------------------------------------------------- injection ----
class _FakeConfig:
    workflow_llm_model = "openai/gpt-4o-mini"
    reasoning_effort = "low"
    max_tokens = 8192

    def openrouter_provider_for(self, model):
        return None

    def openrouter_quantizations_for(self, model):
        return None


class _FakeProvider:
    """Offline stand-in for LLMProvider: captures prompts, returns text."""

    last = None

    def __init__(self, system_msg=None, config=None, **kwargs):
        self.system_msg = system_msg or ""
        self.config = config
        self.prompt = None
        _FakeProvider.last = self

    def __call__(self, prompt):
        self.prompt = prompt
        return (
            "Directive: make the solver prompt more domain-specific. "
            "Kind: prompt tweak. Magnitude: small tweak."
        )


class _FakeWorkflowInfo:
    def __init__(self, score=0.7, gradient="VERIFIER DIAGNOSIS: all pairs tied."):
        self.overall_score = score
        self.abstracted_textual_gradient = gradient
        self.state_result = {
            "step_name": ["solver", "verifier"],
            "answers": ["did x", "did y"],
        }


def test_mutation_prompt_injects_probe_after_directive(monkeypatch):
    """Test 14: the block sits AFTER </directive>, BEFORE the mutation
    instructions; the diagnosis still reaches the directive LLM verbatim
    (alongside, never instead)."""
    monkeypatch.setattr(variation_engine_module, "LLMProvider", _FakeProvider)
    ve = VariationEngine(_FakeConfig())
    prompt = ve.mutation_prompt(
        goal="solve the task", wf_info=_FakeWorkflowInfo(), genotype="# parent",
        run_stderr="", iteration_count=2, max_iterations=8,
        exploration_probe=_PROBE,
    )
    i_directive_end = prompt.index("</directive>")
    i_exploration = prompt.index("<exploration>")
    i_instructions = prompt.index("## MUTATION INSTRUCTIONS:")
    assert i_directive_end < i_exploration < i_instructions
    assert "NOT a measured loss" in prompt
    # the directive LLM saw the diagnosis AND the labelled block together
    provider_prompt = _FakeProvider.last.prompt
    assert "VERIFIER DIAGNOSIS: all pairs tied." in provider_prompt
    assert "<exploration>" in provider_prompt
    assert provider_prompt.index("</diagnosis>") < provider_prompt.index("<exploration>")
    # the system prompt classifies the block as search policy, not fact
    assert (
        "An <exploration> block, when present, is a search-policy request, "
        "not a measured fact" in _FakeProvider.last.system_msg
    )
    assert ve.last_variation_state["exploration"]["triggered"] is True


def test_mutation_prompt_without_probe_untouched(monkeypatch):
    monkeypatch.setattr(variation_engine_module, "LLMProvider", _FakeProvider)
    ve = VariationEngine(_FakeConfig())
    prompt = ve.mutation_prompt(
        goal="g", wf_info=_FakeWorkflowInfo(), genotype="code",
        run_stderr="", iteration_count=1, max_iterations=8,
    )
    assert "<exploration>" not in prompt
    assert "<exploration>" not in _FakeProvider.last.prompt
    assert ve.last_variation_state["exploration"] == {
        "triggered": False, "reason": "", "metrics": [],
    }


def test_crossover_prompt_carries_probe_beside_parent_diagnoses(monkeypatch):
    """Test 14 (crossover): the block augments the parent diagnoses."""
    monkeypatch.setattr(variation_engine_module, "LLMProvider", _FakeProvider)
    ve = VariationEngine(_FakeConfig())
    prompt = ve.crossover_prompt(
        goal="g",
        wf_infos=[_FakeWorkflowInfo(score=0.7), _FakeWorkflowInfo(score=0.4)],
        genotypes=["code1", "code2"], run_stderrs=["", ""],
        iteration_count=3, max_iterations=8,
        exploration_probe=_PROBE,
    )
    assert "<error_diagnosis>" in prompt  # parent diagnoses preserved
    assert prompt.index("<exploration>") < prompt.index("## YOUR TASK:")
    assert prompt.index("</error_diagnosis>") < prompt.index("<exploration>")
    assert ve.last_variation_state["exploration"]["triggered"] is True


# ------------------------------------------------------ engine wiring ----
class _StubOrchestrator:
    def __init__(self):
        self.calls = 0

    async def orchestrate_workflow(self, **kw):
        self.calls += 1
        uuid = f"20261009_000000_{1000 + self.calls:08d}"
        return "ok", uuid, "print('workflow')", True


def _make_engine(tmp_path: Path):
    """Real EvolutionEngine with collaborators stubbed; the verifier block
    written per generation carries the honest HP tie shape (metric_summary
    with min_energy = -2 resolved, every pair tied)."""
    engine = EvolutionEngine.__new__(EvolutionEngine)
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    engine.workflow_dir = str(wf_dir)
    engine.config = types.SimpleNamespace(learned_score_threshold=0.9)
    engine.initial_population = 2
    engine.orchestrator = _StubOrchestrator()
    engine.variation = types.SimpleNamespace(
        record_offspring_gradient=lambda *a, **kw: None,
        last_variation_state={},
        seed_genome_prompt=lambda goal: "seed prompt",
        mutation_prompt=lambda *a, exploration_probe=None, **kw: (
            "mutation prompt\n" + render_exploration_block(exploration_probe)
        ),
        crossover_prompt=lambda *a, exploration_probe=None, **kw: (
            "crossover prompt\n" + render_exploration_block(exploration_probe)
        ),
    )
    engine.selection = types.SimpleNamespace(
        strategy=types.SimpleNamespace(value="qd"),
        validate_survivor=lambda **kw: {
            "absolute_improvement": 0.0, "valid": True, "confidence": 1.0,
            "admit_rejected": False, "behaviour_descriptor": [],
            "qd_score": 0.5, "novelty_score": 0.5,
        },
    )
    engine.notifier = types.SimpleNamespace(send_message=lambda *a, **kw: None)
    engine.logger = logging.getLogger("test-explore")
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
    engine._persist_iteration_metrics = lambda uuid, snap: snapshots.append(snap)
    engine._persist_qd_archive = lambda entry: None

    history: list[str] = []

    async def _evaluate(executed, judge, uuid, answers, rubric, assertion_hist):
        d = wf_dir / uuid
        d.mkdir(parents=True, exist_ok=True)
        (d / "state_result.json").write_text(
            json.dumps({
                "goal": "g",
                "answers": ["ok"],
                "evaluation": {
                    "verifier": {
                        "overall_score": 0.5,
                        "pair_outcomes": {u: "tie" for u in history},
                        "metric_summary": {
                            "min_energy": {"status": "resolved", "value": -2},
                        },
                    }
                },
            }),
            encoding="utf-8",
        )
        history.append(uuid)
        return "verifier", 0.0

    engine._evaluate_and_calculate_cost = _evaluate
    return engine, snapshots, history


def _run_engine(tmp_path: Path, **ctl_kw):
    engine, snapshots, _ = _make_engine(tmp_path)
    controller = _ctl(**ctl_kw)
    runs = asyncio.run(
        engine.evolve_generation(
            [IndividualRun(goal="g", prompt="p", max_depth=8)],
            rewards_history=[], assertion_history=[],
            enable_evolution=True, stop_controller=controller,
        )
    )
    return engine, controller, runs, snapshots


def test_probe_guaranteed_before_plateau_stop(tmp_path: Path):
    """Test 13: tie, tie, tie with patience 3 — at least one prompt
    contains <exploration> BEFORE should_stop ends the run."""
    engine, controller, runs, snapshots = _run_engine(
        tmp_path, patience=3, min_iter=1, max_iter=8
    )
    assert snapshots[-1]["stop_reason"].startswith("plateau:"), snapshots[-1]["stop_reason"]
    assert controller.since_change == 3
    # Prompt slots: runs[k].prompt holds the prompt that created gen k+1
    # (the engine overwrites runs[-1].prompt after execution). runs[0]
    # keeps the initial/seed prompt; the all-tie probe fired at the end of
    # gen 1 and the plateau probe at the end of gen 2, so the prompts for
    # gens 2, 3 and 4 all carry the block — the last one just before the
    # plateau stop consumed it.
    assert "<exploration>" not in runs[0].prompt
    assert "<exploration>" in runs[1].prompt  # all-tie probe (end of gen 1)
    assert "<exploration>" in runs[3].prompt  # plateau probe (end of gen 2)
    assert "NOT a measured loss" in runs[3].prompt


def test_probe_persisted_in_run_metrics_and_variation_log(tmp_path: Path):
    """Test 18: variation_state.exploration lands in run_metrics snapshots
    and in variation_log.jsonl, reflecting the trigger."""
    engine, controller, runs, snapshots = _run_engine(
        tmp_path, patience=3, min_iter=1, max_iter=8
    )
    by_iter = {s["iteration"]: s for s in snapshots}
    # gen 0: fresh champion, single generation -> no trigger
    assert by_iter[0]["variation_state"]["exploration"]["triggered"] is False
    # gen 1 end: two generations with identical resolved metrics -> all-tie
    tie = by_iter[1]["variation_state"]["exploration"]
    assert tie["triggered"] is True
    assert tie["reason"].startswith("all-tie:")
    assert tie["metrics"] == [{"id": "min_energy", "value": -2}]
    # gen 2 end: plateau trigger (since_change == patience - 1 == 2)
    plateau = by_iter[2]["variation_state"]["exploration"]
    assert plateau["triggered"] is True
    assert plateau["reason"].startswith("plateau:")
    # variation_log rows persist the same summary through the engine write
    log_path = Path(engine.workflow_dir) / "variation_log.jsonl"
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    assert rows, "variation_log.jsonl never written"
    row2 = next(r for r in rows if r["for_iteration"] == 2)
    # row N carries the trigger that shaped gen N's prompt (evaluated at
    # the end of gen N-1): all-tie for gen 2, plateau for gen 3.
    assert row2["exploration"]["triggered"] is True
    assert row2["exploration"]["reason"].startswith("all-tie:")
    row3 = next(r for r in rows if r["for_iteration"] == 3)
    assert row3["exploration"]["reason"].startswith("plateau:")


def test_probe_engine_sequence_deterministic(tmp_path: Path):
    """Test 17 (engine level): identical runs replay the identical trigger
    sequence (state-based triggers, no ambient randomness)."""
    reasons_a = [
        s["variation_state"]["exploration"]["reason"]
        for s in _run_engine(tmp_path / "a", patience=3, min_iter=1, max_iter=8)[3]
    ]
    reasons_b = [
        s["variation_state"]["exploration"]["reason"]
        for s in _run_engine(tmp_path / "b", patience=3, min_iter=1, max_iter=8)[3]
    ]
    assert reasons_a == reasons_b and reasons_a[-1].startswith("plateau:")


# ------------------------------------------------------------- config ----
def test_explore_knobs_roundtrip_and_validate():
    """Knob plumbing follows the repo pattern: default, JSON round-trip,
    range validation, CLI override via apply_config_overrides."""
    from config import Config

    c = Config()
    assert c.explore_after_plateau is None
    assert c.explore_base_rate == 0.0
    c.from_json({"explore_after_plateau": 4, "explore_base_rate": 0.15})
    assert c.explore_after_plateau == 4 and c.explore_base_rate == 0.15
    data = c.jsonify()
    assert data["explore_after_plateau"] == 4 and data["explore_base_rate"] == 0.15

    with pytest.raises(ValueError):
        Config().from_json({"explore_base_rate": 1.5})
    with pytest.raises(ValueError):
        Config().from_json({"explore_after_plateau": -1})

    import main as main_mod

    # A parsed-args namespace like the main parser produces (unset flags
    # default to None), exercising the CLI-override branch for both knobs.
    args = argparse.Namespace(
        **{name: None for name in (
            "workflow_dir", "schema_code_path", "smolagent_factory_code_path",
            "prompt_workflow_creator", "runner_default_python_version",
            "runner_default_timeout", "runner_default_max_memory_mb",
            "runner_default_max_cpu_percent", "runner_temp_dir",
            "pushover_token", "pushover_user", "max_evolve_iterations",
        )},
        explore_after_plateau=5,
        explore_base_rate=0.2,
    )
    c2 = Config()
    main_mod.apply_config_overrides(args, c2)
    assert c2.explore_after_plateau == 5 and c2.explore_base_rate == 0.2
    # unset flags keep the config defaults (None -> patience - 1 at runtime)
    args.explore_after_plateau = None
    args.explore_base_rate = None
    c3 = Config()
    main_mod.apply_config_overrides(args, c3)
    assert c3.explore_after_plateau is None and c3.explore_base_rate == 0.0


if __name__ == "__main__":
    import inspect
    import tempfile

    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            params = inspect.signature(fn).parameters
            if "tmp_path" in params:
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            elif "monkeypatch" in params:
                with pytest.MonkeyPatch.context() as mp:
                    fn(mp)
            else:
                fn()
            print(f"✅ {name}")
    print("all exploration-probe tests passed")

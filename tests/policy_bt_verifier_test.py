"""Observable contracts for the artifact-based Policy-BT replacement."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sources.evaluators.policy_bt_verifier.evaluator import PolicyBTVerifierEvaluator
from sources.evaluators.policy_bt_verifier.registry import TaskRegistry, atomic_json
from sources.evaluators.policy_bt_verifier.results import fold_gradient
from sources.evaluators.policy_bt_verifier.tournament import fit_ratings
from sources.evaluators.policy_bt_verifier.visual import VisualChannel


def generations(*identifiers):
    return [{"uuid": key, "execution": {"eligible": True}} for key in identifiers]


def test_bt_transitivity_ties_and_permutation():
    players = generations("a", "b", "c")
    pairs = [{"a": "a", "b": "b", "winner": "a"},
             {"a": "a", "b": "c", "winner": "a"},
             {"a": "b", "b": "c", "winner": "a"}]
    ratings = fit_ratings(players, pairs)
    assert 0 < ratings["c"] < ratings["b"] < ratings["a"] < 1
    assert ratings["b"] == pytest.approx(.5)
    assert fit_ratings(list(reversed(players)), pairs) == pytest.approx(ratings)
    ties = [{**pair, "winner": "tie"} for pair in pairs]
    assert fit_ratings(players, ties) == pytest.approx(dict.fromkeys("abc", .5))


def test_bt_rejects_duplicate_edges_and_forces_ineligible_zero():
    players = generations("a", "b")
    pair = {"a": "a", "b": "b", "winner": "a"}
    with pytest.raises(ValueError, match="Repeated"):
        fit_ratings(players, [pair, {"a": "b", "b": "a", "winner": "b"}])
    players[0]["execution"]["eligible"] = False
    assert fit_ratings(players, [pair])["a"] == 0


def test_registry_rollback_restart_and_session_isolation(tmp_path):
    registry = TaskRegistry(tmp_path, ("task", "session"))
    with registry.transaction() as state:
        state["version"] = 1
    with pytest.raises(RuntimeError), registry.transaction() as state:
        state["version"] = 2
        raise RuntimeError("interrupted")
    assert TaskRegistry(tmp_path, ("task", "session")).load()["version"] == 1
    assert TaskRegistry(tmp_path, ("task", "another")).load()["version"] == 0


def _evaluator(tmp_path, monkeypatch):
    from sources.evaluators.policy_bt_verifier import evaluator as module

    config = SimpleNamespace(memory_dir=str(tmp_path / "memory"), workflow_dir=str(tmp_path / "workflows"),
                             model_pricing={}, reasoning_effort="low", judge_model="openai/test",
                             workspace_dir=str(tmp_path / "workspace"), policy_bt_reexecute=False,
                             policy_bt_repair_rounds=1, policy_bt_visual_channel=False,
                             openrouter_provider_for=lambda model: None,
                             openrouter_quantizations_for=lambda model: None)
    Path(config.workspace_dir).mkdir()
    monkeypatch.setattr(module, "reproduce", lambda *args: {"eligible": True, "reason": "Reproducibility entrypoint present."})
    evaluator = PolicyBTVerifierEvaluator(config)
    evaluator.policy = NumericPolicy()
    return evaluator


class NumericPolicy:
    """Deterministic comparison fixture; tests exercise evaluator lifecycle."""

    def prepare(self, goal, records, cached):
        return {"version": "stable", "criteria": "larger value", "visuals": {}}

    def compare(self, bundle, root):
        left = float((root / "a" / "measurement").read_text())
        right = float((root / "b" / "measurement").read_text())
        return {"winner": "tie" if left == right else "a" if left > right else "b",
                "stage": 1, "feedback_a": f"Measured value {left}",
                "feedback_b": f"Measured value {right}", "evidence": [], "channel": "code"}


def _evaluate(evaluator, uuid, value):
    atomic_json(evaluator.workflow_dir / uuid / "state_result.json", {"goal": "Maximize measured value", "answers": ["preserve"]})
    (evaluator.workspace_dir / "measurement").write_text(str(value))
    return evaluator.evaluate(uuid)


def test_evaluation_refreshes_earlier_ratings_and_regradients_parents(tmp_path, monkeypatch):
    """A parent serves the fold of ALL its current pairs, not its arrival text."""
    evaluator = _evaluator(tmp_path, monkeypatch)
    first = _evaluate(evaluator, "first", 1)
    folder = evaluator.workflow_dir / "first"
    arrival = (folder / "textual_gradient.txt").read_text()
    second = _evaluate(evaluator, "second", 2)
    third = _evaluate(evaluator, "third", 3)
    assert first["reward"] == .5
    assert third["tournament_rewards"]["first"] < second["tournament_rewards"]["first"] < .5
    assert third["pair_outcomes"] == {"first": "win", "second": "win"}
    persisted = json.loads((folder / "state_result.json").read_text())
    verifier = persisted["evaluation"]["verifier"]
    assert persisted["answers"] == ["preserve"]
    assert verifier["reward"] == third["tournament_rewards"]["first"]
    assert verifier["arrival_reward"] == .5
    assert verifier["abstracted_textual_gradient"] == "Measured value 1.0"  # its two losses
    assert verifier["arrival_textual_gradient"] == arrival
    assert (folder / "textual_gradient.txt").read_text() == "Measured value 1.0"
    history = [json.loads(line) for line in (folder / "textual_gradient_history.jsonl").read_text().splitlines()]
    assert [entry["gradient"] for entry in history] == [arrival, "Measured value 1.0"]
    assert evaluator.evaluate("third")["tournament_version"] == third["tournament_version"]


def test_instrument_failure_does_not_commit_a_scientific_loss(tmp_path, monkeypatch):
    evaluator = _evaluator(tmp_path, monkeypatch)
    _evaluate(evaluator, "first", 1)
    monkeypatch.setattr(evaluator.policy, "compare", lambda *args: (_ for _ in ()).throw(RuntimeError("adapter broken")))
    result = _evaluate(evaluator, "second", 2)
    assert result["skipped_reason"] == "verifier_instrument_error"
    assert result["reward"] is None and "adapter broken" in result["error"]
    registry = TaskRegistry(evaluator.workflow_dir / ".policy_bt", ("Maximize measured value", "standalone"))
    assert [item["uuid"] for item in registry.load()["generations"]] == ["first"]
    failed = json.loads((evaluator.workflow_dir / "second" / "state_result.json").read_text())
    assert failed["evaluation"]["verifier"]["skipped_reason"] == "verifier_instrument_error"
    assert "tournament_rewards" not in failed["evaluation"]["verifier"]


def test_verifier_llm_calls_are_counted_as_run_cost(tmp_path, monkeypatch):
    evaluator = _evaluator(tmp_path, monkeypatch)
    from sources.evaluators.policy_bt_verifier import evaluator as module

    class Recorder:
        def __init__(self, agent_name, memory_path, **_):
            self.path = Path(memory_path) / f"{agent_name}.json"

        def __call__(self, prompt, use_cache):
            self.path.write_text("{}")
            return "ok"

    monkeypatch.setattr(module, "LLMProvider", Recorder)
    evaluator._uuid = "costed"
    evaluator._ask("first")
    evaluator._ask("second")
    names = sorted(path.name for path in (evaluator.memory_dir / "costed").iterdir())
    assert len(names) == 2 and all(name.startswith("verifier_") for name in names)


def test_missing_shell_script_scores_zero_without_model_call(tmp_path, monkeypatch):
    evaluator = _evaluator(tmp_path, monkeypatch)
    from sources.evaluators.policy_bt_verifier import evaluator as module
    from sources.evaluators.policy_bt_verifier.reexec import reproduce

    monkeypatch.setattr(module, "reproduce", reproduce)
    monkeypatch.setattr(evaluator, "_ask", lambda prompt: pytest.fail("Missing script must not call LLM"))
    result = _evaluate(evaluator, "missing-script", 10)
    assert result["reward"] == 0
    assert ".sh" in result["abstracted_textual_gradient"]
    assert not result["execution"]["eligible"]


def test_gradient_prioritizes_earliest_loss_without_rating_leak():
    player = {"uuid": "current", "execution": {"eligible": True}}
    pairs = [{"a": "current", "b": "one", "winner": "a", "stage": 1, "feedback_a": "Preserve output", "feedback_b": "Other"},
             {"a": "current", "b": "two", "winner": "b", "stage": 3, "feedback_a": "Improve error", "feedback_b": "Other"},
             {"a": "three", "b": "current", "winner": "a", "stage": 2, "feedback_a": "Other", "feedback_b": "Fix invalid output"}]
    assert fold_gradient(player, pairs).splitlines() == ["Fix invalid output", "Improve error", "Preserve output"]


def test_tied_gradient_states_absolute_deficiencies_not_a_match():
    """Live case: all-tie parents fed "Your run matches the other run on all
    reported criteria" to the next mutation, which says nothing to fix."""
    criteria = [{"id": "conformance", "description": "exact header", "direction": "satisfy"},
                {"id": "auc", "description": "test AUC", "direction": "higher"},
                {"id": "hidden", "description": "masked metric", "direction": "higher"}]
    tie = {"a": "x", "b": "y", "winner": "tie", "stage": 0,
           "feedback_a": "Your run matches the other run on all reported criteria.", "feedback_b": "same",
           "normalized_a": {"values": {"conformance": False, "auc": None, "hidden": None},
                            "evidence": [{"criterion": "conformance", "detail": "header is FDA_APPROVED_prob"}]},
           "normalized_b": {"values": {"conformance": False, "auc": 0.8, "hidden": None}, "evidence": []},
           "visual_deficiencies": {"a": ["no cumulative-integral curve"], "b": []}}
    folded = fold_gradient({"uuid": "x", "execution": {"eligible": True}}, [tie], criteria).splitlines()
    assert folded == ["Your run does not satisfy 'conformance': header is FDA_APPROVED_prob",
                      "'auc' (test AUC) could not be measured from your outputs, while another run's outputs provide it.",
                      "Figure review by the vision model (observations, not measurements):",
                      "- no cumulative-integral curve"]
    met = {"values": {"conformance": True, "auc": 0.8, "hidden": None}, "evidence": []}
    clean = {**tie, "normalized_a": met, "normalized_b": met, "visual_deficiencies": {}}
    assert fold_gradient({"uuid": "x", "execution": {"eligible": True}}, [clean], criteria).startswith("Every comparison")


def test_gradient_does_not_restate_a_gate_already_lost_decisively():
    criteria = [{"id": "conformance", "stage": 2, "description": "exact header", "direction": "satisfy"}]
    own = {"values": {"conformance": False}, "evidence": [{"criterion": "conformance", "detail": "header is X_prob"}]}
    loss = {"a": "x", "b": "y", "winner": "b", "stage": 2, "feedback_a": "Rename X_prob to X.", "feedback_b": "ok",
            "normalized_a": own, "normalized_b": {"values": {"conformance": True}, "evidence": []}}
    folded = fold_gradient({"uuid": "x", "execution": {"eligible": True}}, [loss], criteria)
    assert folded == "Rename X_prob to X."


def _plot_pair(tmp_path):
    from PIL import Image

    for side in ("a", "b"):
        (tmp_path / side).mkdir()
        Image.new("RGB", (8, 8), "white").save(tmp_path / side / "plot.png")
    return {"goal": "Make a figure", "criteria": [], "paths": {"a": ["plot.png"], "b": ["plot.png"]}}


def test_visual_order_disagreement_keeps_tie_with_each_sides_deficiencies(tmp_path):
    response = json.dumps({"winner": "a", "stage": 2, "feedback_a": "Visible A", "feedback_b": "Visible B",
                           "deficiencies_a": ["no legend"], "deficiencies_b": ["wrong units"],
                           "evidence": [{"side": "a", "path": "plot.png", "observation": "visible"}]})
    verdict = VisualChannel(lambda content: response).compare(_plot_pair(tmp_path), tmp_path)
    assert verdict["winner"] == "tie"
    assert verdict["visual_deficiencies"] == {"a": ["no legend"], "b": ["wrong units"]}  # forward order
    assert verdict["visual_prompt_version"] == "visual-tiebreak-2"


def test_malformed_deficiencies_never_cost_an_agreed_visual_winner(tmp_path):
    response = json.dumps({"winner": "tie", "stage": 0, "feedback_a": "A", "feedback_b": "B",
                           "deficiencies_a": "none", "deficiencies_b": [{"x": 1}, "axis unlabeled"], "evidence": []})
    verdict = VisualChannel(lambda content: response).compare(_plot_pair(tmp_path), tmp_path)
    assert verdict["visual_deficiencies"] == {"a": [], "b": ["axis unlabeled"]}


def test_visual_verdict_keeps_code_adapter_values_for_the_fold():
    from sources.evaluators.policy_bt_verifier.evaluator import _with_visual

    code = {"winner": "tie", "stage": 0, "feedback_a": "m", "feedback_b": "m", "normalized_a": {"values": {"k": 1}}}
    decisive = {"winner": "b", "stage": 3, "feedback_a": "A lacks axis", "feedback_b": "ok", "visual_deficiencies": {"a": ["axis"], "b": []}}
    assert _with_visual(code, decisive)["normalized_a"] == {"values": {"k": 1}}
    assert _with_visual(code, decisive)["winner"] == "b"
    tied = _with_visual(code, {**decisive, "winner": "tie"})
    assert tied["winner"] == "tie" and tied["feedback_a"] == "m" and tied["visual_deficiencies"]["a"] == ["axis"]
    assert _with_visual(code, None) is code


def test_recon_feeds_malformed_answer_back_instead_of_aborting(tmp_path):
    """Live regression: one truncated recon JSON answer unscored a generation."""
    from sources.evaluators.policy_bt_verifier.recon import reconnaissance

    (tmp_path / "result.txt").write_text("energy -9")
    plan = {"action": "complete", "task_representation": "minimize energy",
            "criteria": [{"id": "energy", "stage": 1, "description": "energy", "direction": "lower"}],
            "representation": {"energy": "int"}, "visuals": {},
            "plans": {"g1": {"adapter": None, "instructions": "read result.txt"}}}
    answers, errors = iter(['{"action": "complete", "broken"', json.dumps(plan)]), []

    def ask(prompt):
        payload = json.loads(prompt.split("\nEVIDENCE_JSON\n", 1)[1].split("\nEND_EVIDENCE_JSON")[0])
        errors.append(payload.get("previous_answer_error"))
        return next(answers)

    result = reconnaissance("goal", [{"uuid": "g1", "snapshot": str(tmp_path)}], {"ask": ask})
    assert [criterion["id"] for criterion in result["criteria"]] == ["energy"]
    assert errors[0] is None and "JSONDecodeError" in errors[1]


def test_missing_goal_is_unscored_not_a_session_crash(tmp_path, monkeypatch):
    """Review regression: a torn state_result used to raise out of evaluate()."""
    evaluator = _evaluator(tmp_path, monkeypatch)
    (evaluator.workflow_dir / "torn").mkdir(parents=True)
    (evaluator.workflow_dir / "torn" / "state_result.json").write_text('{"goal": "cut')
    result = evaluator.evaluate("torn")
    assert result["skipped_reason"] == "verifier_instrument_error"
    assert result["reward"] is None


def test_torn_historical_state_does_not_block_later_generations(tmp_path, monkeypatch):
    """Review regression: one corrupt older file failed every later evaluation."""
    evaluator = _evaluator(tmp_path, monkeypatch)
    _evaluate(evaluator, "first", 1)
    (evaluator.workflow_dir / "first" / "state_result.json").write_text('{"goal": "cut')
    result = _evaluate(evaluator, "second", 2)
    assert result["pair_outcomes"] == {"first": "win"}
    assert (evaluator.workflow_dir / "first" / "state_result.json").read_text() == '{"goal": "cut'


def test_stale_prompt_version_rebuilds_instead_of_failing(tmp_path):
    """Review regression: a persisted bundle from an older prompt bricked the task."""
    from sources.evaluators.policy_bt_verifier.policy import PolicyEngine

    calls = []
    engine = PolicyEngine({"policy_bt_repair_rounds": 0}, lambda prompt: calls.append(prompt) or "not json")
    stale = {"goal": "g", "prompt_version": "policy-bt-separated-0"}
    with pytest.raises(Exception, match="budget|JSON|Expecting"):
        engine.prepare("g", [{"uuid": "u", "snapshot": str(tmp_path)}], stale)
    assert calls, "stale bundle must trigger a fresh reconnaissance, not an immediate error"


def test_reporter_renders_plain_text_without_ansi_when_colors_off():
    """The operator console must stay readable in captured output."""
    from sources.evaluators.policy_bt_verifier.reporting import Reporter

    buffer = io.StringIO()
    report = Reporter("gen_12345678", color=False, stream=buffer)
    report.bundle({"criteria": [{"id": "min_energy", "stage": 1, "direction": "lower"}],
                   "core_source": "def compare(a, b):\n    return {'winner': 'tie', 'stage': 0}\n"},
                  [])
    report.pair("gen_aaaa", "gen_bbbb", {"winner": "a", "stage": 1, "channel": "code"},
                {"common": ["results.txt"], "only_a": [], "only_b": [],
                 "counts": {"common": 1, "only_a": 0, "only_b": 0}})
    report.standings({"gen_aaaa": 0.6, "gen_bbbb": 0.4},
                     [{"a": "gen_aaaa", "b": "gen_bbbb", "winner": "a"}],
                     "gen_aaaa", {"gen_aaaa": 0.5})
    report.gradient("Your run reports energy -9; keep folding tighter.")
    report.close()
    text = buffer.getvalue()
    assert "\033[" not in text
    for marker in ("min_energy", "lower-is-better", "results.txt", "A WINS",
                   "BRADLEY-TERRY", "0.600", "1W 0L 0T", "Δ+0.100", "GRADIENT",
                   "energy -9"):
        assert marker in text, marker


def test_reporter_survives_minimal_fixture_bundles():
    """Reporting must never break an evaluation on a partial bundle."""
    from sources.evaluators.policy_bt_verifier.reporting import Reporter

    buffer = io.StringIO()
    report = Reporter("gen_x", color=False, stream=buffer)
    report.bundle({"version": "stable", "criteria": "larger value"}, ["missing"])
    report.execution({})
    report.close()
    assert "TASK POLICY" in buffer.getvalue()


def test_pair_file_diff_splits_sides_exactly(tmp_path):
    """The compared-files listing must show what each side actually had."""
    from sources.evaluators.policy_bt_verifier.reporting import pair_file_diff

    for side, names in (("a", ["run.sh", "solver.py", "results.txt"]),
                        ("b", ["run.sh", "solver.py", "extra.csv"])):
        (tmp_path / side).mkdir()
        for name in names:
            (tmp_path / side / name).write_text("x")
    diff = pair_file_diff(tmp_path)
    assert diff["common"] == ["run.sh", "solver.py"]
    assert diff["only_a"] == ["results.txt"]
    assert diff["only_b"] == ["extra.csv"]
    assert diff["counts"] == {"common": 2, "only_a": 1, "only_b": 1}

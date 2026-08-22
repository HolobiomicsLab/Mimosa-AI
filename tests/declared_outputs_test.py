"""The plan's declared outputs must reach the verifier.

Issue #196: step `reproduction_spec_analysis` declared
`workspace/analysis/iimn_reproduction_plan.md`, wrote
`iimn_reproduction_guide.md` at the workspace root instead, was scored 0.799 by
the verifier, and killed the run at the next step's dependency gate. None of the
26 claims in its frozen rubric mentioned the declared path — claims are
extracted partly from execution text and workspace listing, so the rubric
inherited the agent's choice of deliverable.

The carrier is keyed on the step task text, which the planner already passes as
`original_task` and the verifier already keys its rubric cache on. These tests
pin that coupling: if the two ever key differently, the declaration silently
stops arriving, and a silent stop is exactly the failure mode this is meant to
end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))

from sources.core.failure_fingerprint import compute_failure_fingerprint  # noqa: E402, F401
from sources.core import declared_outputs  # noqa: E402
from sources.evaluators.verifier import VerifierEvaluator  # noqa: E402


_TASK = ("Broader context:Reproduce IIMN on MSV000080492.\n---\n"
         "Your task:Analyse reproduction_iimn.md and extract the parameters.")
_OUTPUTS = ["workspace/analysis/iimn_reproduction_plan.md"]


# ---- the coupling that makes the whole scheme work -------------------------


def test_planner_key_and_verifier_key_are_the_same_function():
    """If these drift, the declaration is written where nothing reads it."""
    assert declared_outputs.task_key(_TASK) == VerifierEvaluator._task_cache_key(_TASK)


def test_key_is_stable_and_discriminating():
    assert declared_outputs.task_key(_TASK) == declared_outputs.task_key(_TASK)
    assert declared_outputs.task_key(_TASK) != declared_outputs.task_key(_TASK + " ")
    assert len(declared_outputs.task_key(_TASK)) == 16


# ---- round trip ------------------------------------------------------------


def test_record_then_load_round_trips(tmp_path: Path):
    assert declared_outputs.record(tmp_path, _TASK, _OUTPUTS) is not None
    assert declared_outputs.load(tmp_path, _TASK) == _OUTPUTS


def test_load_returns_empty_when_nothing_recorded(tmp_path: Path):
    assert declared_outputs.load(tmp_path, _TASK) == []


def test_record_is_a_no_op_without_outputs(tmp_path: Path):
    assert declared_outputs.record(tmp_path, _TASK, []) is None
    assert declared_outputs.load(tmp_path, _TASK) == []


def test_record_overwrites_because_the_plan_is_authoritative(tmp_path: Path):
    """Unlike the rubric cache, which freezes the first extraction."""
    declared_outputs.record(tmp_path, _TASK, ["old.md"])
    declared_outputs.record(tmp_path, _TASK, ["new.md"])
    assert declared_outputs.load(tmp_path, _TASK) == ["new.md"]


def test_a_corrupt_record_degrades_to_empty_not_an_exception(tmp_path: Path):
    declared_outputs.path_for(tmp_path, _TASK).write_text("{not json")
    assert declared_outputs.load(tmp_path, _TASK) == []


def test_a_record_without_the_expected_key_degrades_to_empty(tmp_path: Path):
    declared_outputs.path_for(tmp_path, _TASK).write_text(json.dumps({"other": 1}))
    assert declared_outputs.load(tmp_path, _TASK) == []


def test_a_different_step_does_not_see_this_steps_outputs(tmp_path: Path):
    declared_outputs.record(tmp_path, _TASK, _OUTPUTS)
    assert declared_outputs.load(tmp_path, "Your task:something else entirely") == []


# ---- the claims themselves -------------------------------------------------


def test_the_declared_path_becomes_a_maximum_importance_claim():
    claims = declared_outputs.as_claims(_OUTPUTS)
    assert len(claims) == 1
    claim = claims[0]
    assert claim["importance"] == 10
    assert "iimn_reproduction_plan.md" in claim["description"]
    assert claim["likely_relevant_files"] == _OUTPUTS
    assert claim["source"] == "plan_expected_outputs"


def test_claim_ids_are_slugged_and_distinct():
    claims = declared_outputs.as_claims(
        ["workspace/analysis/plan.md", "results/summary.csv"])
    ids = [c["id"] for c in claims]
    assert ids == ["declared_output_plan_md", "declared_output_summary_csv"]
    assert len(set(ids)) == 2


def test_a_directory_output_still_yields_a_claim():
    assert declared_outputs.as_claims(["workspace/analysis/"])[0]["id"] == \
        "declared_output_analysis"


# ---- injection into the verifier's claim list ------------------------------


class _WfInfo:
    def __init__(self, original_task: str, goal: str = "") -> None:
        self.original_task = original_task
        self.goal = goal or original_task


def _evaluator(tmp_path: Path) -> VerifierEvaluator:
    import logging
    v = VerifierEvaluator.__new__(VerifierEvaluator)
    v._runner_temp_root = tmp_path
    v.logger = logging.getLogger("test-declared-outputs")
    return v


_EXTRACTED = [{"id": "guide_completeness_and_accuracy", "importance": 10},
              {"id": "source_file_analyzed", "importance": 10}]


def test_declared_claims_are_prepended_so_max_claims_cannot_drop_them(tmp_path: Path):
    declared_outputs.record(tmp_path, _TASK, _OUTPUTS)
    v = _evaluator(tmp_path)
    out = v._prepend_declared_output_claims(list(_EXTRACTED), _WfInfo(_TASK))
    assert out[0]["id"] == "declared_output_iimn_reproduction_plan_md"
    assert [c["id"] for c in out[1:]] == [c["id"] for c in _EXTRACTED]


def test_no_declaration_leaves_the_rubric_exactly_as_it_was(tmp_path: Path):
    v = _evaluator(tmp_path)
    out = v._prepend_declared_output_claims(list(_EXTRACTED), _WfInfo(_TASK))
    assert out == _EXTRACTED


def test_an_already_present_id_is_not_duplicated(tmp_path: Path):
    declared_outputs.record(tmp_path, _TASK, _OUTPUTS)
    seeded = [{"id": "declared_output_iimn_reproduction_plan_md", "importance": 3}]
    v = _evaluator(tmp_path)
    out = v._prepend_declared_output_claims(list(seeded), _WfInfo(_TASK))
    assert len(out) == 1

def test_injection_never_raises_into_the_evaluate_path(tmp_path: Path):
    """A broken wf_info must cost the extra claims, not the whole evaluation."""
    declared_outputs.record(tmp_path, _TASK, _OUTPUTS)
    v = _evaluator(tmp_path)

    class _Exploding:
        @property
        def original_task(self):
            raise RuntimeError("disk gone")

    assert v._prepend_declared_output_claims(list(_EXTRACTED), _Exploding()) == _EXTRACTED


def test_the_issue_196_scenario_end_to_end(tmp_path: Path):
    """Planner declares a path; the rubric that scored 0.799 never mentioned it."""
    declared_outputs.record(tmp_path, _TASK, _OUTPUTS)
    v = _evaluator(tmp_path)
    out = v._prepend_declared_output_claims(list(_EXTRACTED), _WfInfo(_TASK))
    mentions = [c for c in out if "iimn_reproduction_plan" in json.dumps(c)]
    assert mentions, "the declared output must now be checkable"
    assert mentions[0]["importance"] == 10

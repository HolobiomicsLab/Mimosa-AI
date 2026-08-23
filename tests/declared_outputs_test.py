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


# ---- a declared directory is not a declared file --------------------------


def test_a_directory_output_asks_for_a_directory_not_a_file():
    """Observed live: a plan declared `workspace/p_iimn_task_001/data/`.

    The claim said "A file exists at that path" — which no correct run can
    satisfy for a directory, so a maximum-importance claim would have failed a
    step that did exactly what the plan asked. ``_verify_expected_outputs``
    already distinguishes the two; the claim has to as well.
    """
    claim = declared_outputs.as_claims(["workspace/p_iimn_task_001/data/"])[0]
    assert "A directory exists at that path" in claim["description"]
    assert "A file exists at that path" not in claim["description"]


def test_a_file_output_still_asks_for_a_file():
    claim = declared_outputs.as_claims(
        ["workspace/p_iimn_task_001/data/dataset_inventory.csv"])[0]
    assert "A file exists at that path" in claim["description"]
    assert "A directory exists" not in claim["description"]


def test_a_backslash_directory_is_recognised_too():
    claim = declared_outputs.as_claims(["workspace\\analysis\\"])[0]
    assert "A directory exists at that path" in claim["description"]


def test_the_live_step_2_declaration_yields_one_of_each():
    """The exact pair run 6 recorded for step `data_acquisition`."""
    claims = declared_outputs.as_claims([
        "workspace/p_iimn_task_001/data/",
        "workspace/p_iimn_task_001/data/dataset_inventory.csv",
    ])
    assert [c["id"] for c in claims] == [
        "declared_output_data", "declared_output_dataset_inventory_csv"]
    kinds = ["directory" if "A directory exists" in c["description"] else "file"
             for c in claims]
    assert kinds == ["directory", "file"]


# ---- "non-empty" is satisfied by a stub -----------------------------------


def test_the_claim_asks_for_content_not_merely_bytes():
    """Observed live, and it went the wrong way.

    Step `feature_detection_and_processing` declared five outputs. With no raw
    data reachable, the run wrote each file as a labelled placeholder — line 2
    of feature_table_qtof.csv reads "# PLACEHOLDER: ... NO REAL DATA
    AVAILABLE" — and all five claims PASSED at importance 10, one of them on
    the detail line "File exists and is non-empty (1462 bytes)".

    A claim a placeholder satisfies pushes an agent to create the file without
    pushing it to fill the file, which is the opposite of the intent.
    """
    for output in ("results/table.csv", "results/dir/"):
        desc = declared_outputs.as_claims([output])[0]["description"]
        assert "is non-empty" not in desc, "byte count is not evidence of content"
    # The data-file wording names the failure mode outright; the report wording
    # asks for substance instead, because a report may legitimately be *about*
    # missing data (see test_a_report_that_records_missing_inputs...).
    assert "placeholder text standing in for data" in \
        declared_outputs.as_claims(["results/table.csv"])[0]["description"]
    assert "not an empty stub" in \
        declared_outputs.as_claims(["results/dir/"])[0]["description"]


def test_both_kinds_carry_the_substance_requirement():
    claims = declared_outputs.as_claims(["a/table.csv", "a/dir/"])
    assert "holds actual records" in claims[0]["description"]
    assert "substantive content produced by this step" in claims[1]["description"]
    assert "A directory exists" in claims[1]["description"]
    assert "A file exists" in claims[0]["description"]


# ---- a file that IS a stub vs a file that REPORTS on one ------------------


def test_a_report_that_records_missing_inputs_still_satisfies_its_claim():
    """The first wording failed the best artefact in the workspace.

    `processing_log.md` — 292 lines, batch parameters, a recovery path — was
    failed by an A/B probe because it honestly recorded that its *sibling*
    outputs were placeholders. The disqualifier caught a file reporting on
    stubs rather than a file that is one, which penalises exactly the honesty
    the claim is meant to protect.
    """
    desc = declared_outputs.as_claims(["a/processing_log.md"])[0]["description"]
    assert "documents what was attempted" in desc
    assert "honestly records missing inputs" in desc
    assert "holds actual records" not in desc


def test_a_data_file_must_hold_records():
    for output in ("a/feature_table.csv", "a/spectra.mgf", "a/x.tsv", "a/y.json"):
        desc = declared_outputs.as_claims([output])[0]["description"]
        assert "holds actual records" in desc, output
        assert "placeholder text standing in for data" in desc, output


def test_the_two_wordings_are_mutually_exclusive():
    data = declared_outputs.as_claims(["a/t.csv"])[0]["description"]
    prose = declared_outputs.as_claims(["a/t.md"])[0]["description"]
    assert data != prose
    assert "substantive content" in prose and "substantive content" not in data


def test_an_unknown_suffix_is_treated_as_a_report():
    """Only the listed data suffixes get the record requirement."""
    desc = declared_outputs.as_claims(["a/notes.rst"])[0]["description"]
    assert "substantive content" in desc

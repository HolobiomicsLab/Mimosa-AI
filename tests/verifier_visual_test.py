"""Visual-verifier (Source G) fixes: goal plumbing, image selection, aggregation.

Change-set B (branch mimosa_v2_verifier_tweaks), see
run_notes/a_publish_ready/changes/2026-09-10_visual_verifier_findings.md.

Covers, with mocked judge/vision calls only (no network):
- the visual prompt shows the REAL task goal under TASK GOAL and the agents'
  self-reported output under an explicit UNTRUSTED label;
- image candidate collection excludes code files, prioritizes result-like
  paths, and walks the FULL workspace tree (no first-directory break);
- the visual branch sends ALL candidate images, not only the first;
- visual-branch errors count as score 0 in the aggregate and raise
  visual_evidence_missing, while non-visual errors stay excluded;
- the file-selector empty/error fallback never returns code files.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))

# Prime the import chain (verifier <-> cli circular import), same as the
# other verifier tests in this suite.
from sources.core.failure_fingerprint import compute_failure_fingerprint  # noqa: E402, F401
from sources.evaluators.verifier import VerifierEvaluator  # noqa: E402


# ---------- helpers ----------------------------------------------------------


def _make_evaluator(tmp_path: Path) -> VerifierEvaluator:
    """Build a VerifierEvaluator with paths rooted at tmp_path; LLMs mocked."""
    v = VerifierEvaluator.__new__(VerifierEvaluator)
    v.workspace_dir = tmp_path / "workspace"
    v.workspace_dir.mkdir(parents=True, exist_ok=True)
    v._runner_temp_root = tmp_path / "verifier_tmp"
    v._runner_temp_root.mkdir(parents=True, exist_ok=True)
    v.workflow_dir = tmp_path / "workflows"
    v.workflow_dir.mkdir(parents=True, exist_ok=True)
    v.logger = logging.getLogger("test-verifier-visual")
    v._DEFAULT_CLAIM_IMPORTANCE = 5
    v._HARD_FAIL_IMPORTANCE = 10
    v.hard_fail_cap = 0.3
    v._workspace_files = set()
    return v


def _visual_claim() -> dict[str, object]:
    return {
        "id": "fig_exists",
        "source": "source_g",
        "importance": 10,
        "description": "The UMAP figure exists and is scientifically plausible",
        "likely_relevant_files": [],
    }


def _write(path: Path, data: bytes = b"\x89PNG\r\n\x1a\nfakepng") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


GOAL_TEXT = (
    "Build a QSAR model on the lymph dataset and save the UMAP figure to "
    "report/umap_panel.png."
)
EXEC_TEXT = (
    "GOAL:\nThe workflow's goal was to achieve the following scientific/research "
    "objective:\n" + GOAL_TEXT + "\n\nFINAL ANSWER FROM AGENT(S) EXECUTION:\n"
    "The final answer produced by the agent(s) at the end of the workflow "
    "execution was:\n{\"status\": \"success\", \"figure\": \"looks great\"}"
)


# ---------- (a) prompt: real goal + untrusted agent output ------------------


def test_visual_prompt_shows_real_goal_and_untrusted_agent_text(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    prompt = v._build_visual_check_prompt(
        _visual_claim(), EXEC_TEXT, "umap_panel.png\t42B", "", GOAL_TEXT
    )
    goal_idx = prompt.find("TASK GOAL")
    agent_idx = prompt.find("AGENT-REPORTED OUTPUT (untrusted")
    assert goal_idx != -1, "real goal must be labeled TASK GOAL"
    assert agent_idx != -1, "agent output must carry the untrusted label"
    assert GOAL_TEXT in prompt, "the literal goal text must appear in the prompt"
    assert "FINAL ANSWER FROM AGENT(S) EXECUTION" in prompt
    # The goal text must not be presented as untrusted agent output: the
    # TASK GOAL block has to come first and carry the real goal.
    assert goal_idx < agent_idx
    assert "SCIENTIFIC CONTEXT (workflow goal)" not in prompt


def test_visual_prompt_labels_each_attached_image(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    prompt = v._build_visual_check_prompt(
        _visual_claim(), EXEC_TEXT, "listing", "", GOAL_TEXT,
        image_labels=["pred_results/fig1.png (12 bytes)", "data/panel.png (9 bytes)"],
    )
    assert "IMAGES ATTACHED" in prompt
    assert "pred_results/fig1.png (12 bytes)" in prompt
    assert "data/panel.png (9 bytes)" in prompt


# ---------- (b) image candidate collection ----------------------------------


def test_image_candidates_exclude_code_and_prefer_result_dirs(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    ws = v.workspace_dir
    # "data" sorts before "pred_results" in os.walk order, so the OLD
    # first-directory-break would have stopped at data/ and missed the
    # produced deliverable entirely.
    _write(ws / "data" / "input_structure.png")
    _write(ws / "pred_results" / "spatial_pred.png")
    _write(ws / "scripts" / "plot.py", b"import matplotlib")
    claim = {**_visual_claim(),
             "likely_relevant_files": ["scripts/plot.py", "pred_results/spatial_pred.png"]}

    candidates = v._collect_visual_image_candidates(claim, GOAL_TEXT)

    assert "scripts/plot.py" not in candidates, "code files must never be candidates"
    assert "data/input_structure.png" in candidates, "full tree must be walked"
    assert "pred_results/spatial_pred.png" in candidates
    assert candidates.index("pred_results/spatial_pred.png") < candidates.index(
        "data/input_structure.png"
    ), "result-like paths must be prioritized over input/dataset paths"


def test_image_candidates_rank_declared_output_filename_highest(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    ws = v.workspace_dir
    _write(ws / "report" / "umap_panel.png")          # neutral dir, declared in goal
    _write(ws / "pred_results" / "other.png")          # output dir, not declared
    _write(ws / "data" / "raw_input.png")
    claim = _visual_claim()

    candidates = v._collect_visual_image_candidates(claim, GOAL_TEXT)

    assert candidates[0] == "report/umap_panel.png", (
        "the deliverable filename named in the goal must rank first"
    )
    assert candidates.index("pred_results/other.png") < candidates.index(
        "data/raw_input.png"
    )


def test_image_candidates_walk_full_tree_no_first_directory_break(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    ws = v.workspace_dir
    # a-image/ sorts before z-image/: a first-directory break would stop at
    # a-image and never see z-image.
    _write(ws / "a_image" / "first.png")
    _write(ws / "z_image" / "second.png")

    candidates = v._collect_visual_image_candidates(_visual_claim(), "")

    assert candidates == ["a_image/first.png", "z_image/second.png"]


def test_visual_branch_sends_all_candidate_images_not_just_the_first(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    ws = v.workspace_dir
    _write(ws / "pred_results" / "fig1.png")
    _write(ws / "pred_results" / "fig2.png")
    _write(ws / "data" / "dataset_panel.png")

    captured: dict[str, object] = {}

    def fake_vision_judge(uuid, agent_name, prompt, images):
        captured["images"] = list(images)
        captured["prompt"] = prompt
        return {"verdict": "pass", "rationale": "figure looks plausible"}, None

    v._call_vision_judge = fake_vision_judge  # type: ignore[method-assign]
    scored = v._run_visual_branch(
        "uuid-1", _visual_claim(), {}, EXEC_TEXT, "listing", "", GOAL_TEXT
    )

    assert scored["verifier_kind"] == "visual"
    assert scored["status"] == "pass"
    assert scored["score"] == 1.0
    assert len(captured["images"]) == 3, "ALL candidate images must be sent"
    assert all(str(u).startswith("data:image/") for u in captured["images"])
    assert "TASK GOAL" in captured["prompt"]
    assert "AGENT-REPORTED OUTPUT (untrusted" in captured["prompt"]


def test_visual_branch_missing_images_is_error(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    scored = v._run_visual_branch(
        "uuid-2", _visual_claim(), {}, EXEC_TEXT, "listing", "", GOAL_TEXT
    )
    assert scored["status"] == "error"
    assert scored["verifier_kind"] == "visual"
    assert scored["score"] == 0.0


# ---------- (c) aggregation: visual errors count, others do not --------------


def _scored(cid: str, importance: int, score: float, status: str, kind: str):
    return {
        "claim": {"id": cid, "importance": importance, "source": "source_b"},
        "score": score,
        "verifier_kind": kind,
        "status": status,
        "details": "",
        "rationale": "",
    }


def test_visual_error_claim_counts_as_zero_in_aggregate(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    per_claim = [
        _scored("csv_ok", 10, 1.0, "pass", "executable"),
        _scored("fig_visual", 10, 0.0, "error", "visual"),
    ]
    result = v._aggregate(per_claim)

    # Without the fix this run would score a perfect 1.0 with zero visual
    # evidence; the visual error must sit in numerator AND denominator.
    assert result["visual_evidence_missing"] is True
    assert result["n_visual_error"] == 1
    assert result["overall_score"] == pytest.approx(0.5)
    assert result["n_scored"] == 2


def test_non_visual_error_claims_stay_excluded(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    per_claim = [
        _scored("csv_ok", 10, 1.0, "pass", "executable"),
        _scored("exec_boom", 10, 0.0, "error", "executable"),
    ]
    result = v._aggregate(per_claim)

    assert result["visual_evidence_missing"] is False
    assert result["n_visual_error"] == 0
    assert result["overall_score"] == pytest.approx(1.0), (
        "non-visual errors keep the current error-exclusion behavior"
    )
    assert result["n_scored"] == 1


def test_all_visual_errors_is_zero_not_all_verifiers_errored(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    per_claim = [
        _scored("fig_visual", 10, 0.0, "error", "visual"),
        _scored("fig_visual_2", 8, 0.0, "error", "visual"),
    ]
    result = v._aggregate(per_claim)

    assert result["overall_score"] == 0.0
    assert result["visual_evidence_missing"] is True
    assert "skipped_reason" not in result


def test_visual_exception_fallback_keeps_visual_kind(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    claim = {**_visual_claim(), "id": "fig_boom"}
    spec = {"executable": False, "reason": "visual claim: judged by the vision model"}
    generated = {"fig_boom": (claim, spec)}

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash in the visual branch")

    v._verify_claim = boom  # type: ignore[method-assign]
    result = v._verify_one_claim_safe(
        "uuid-3", claim, generated, "exec", "listing", "", "goal"
    )

    # The exception fallback must not mislabel the visual claim as "soft",
    # otherwise the aggregator would silently drop the error again.
    assert result["verifier_kind"] == "visual"
    assert result["status"] == "error"
    assert result["score"] == 0.0


# ---------- (d) selector fallback never returns code files -------------------


def _selector_workspace(v: VerifierEvaluator) -> None:
    ws = v.workspace_dir
    _write(ws / "src" / "main.py", b"import pandas\nprint(1)\n")
    _write(ws / "workflow.py", b"import pandas\nprint(1)\n")
    _write(ws / "analysis.R", b"library(ggplot2)\n")
    _write(ws / "pred_results" / "out.csv", b"a,b\n1,2\n")
    _write(ws / "report.txt", b"final report\n")
    _write(ws / "pred_results" / "panel.png")
    v._workspace_files = {
        "src/main.py", "workflow.py", "analysis.R",
        "pred_results/out.csv", "report.txt", "pred_results/panel.png",
    }


def test_selector_error_fallback_never_returns_code_files(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    _selector_workspace(v)
    v._call_judge_for_json = (  # type: ignore[method-assign]
        lambda uuid, agent_name, prompt: (None, "simulated judge outage")
    )

    files = v._llm_select_files("uuid-4", _visual_claim(), EXEC_TEXT, goal=GOAL_TEXT)

    assert files, "fallback must still point at non-code artefacts"
    assert not any(f.endswith((".py", ".R")) for f in files)
    assert "src/main.py" not in files and "workflow.py" not in files
    assert "analysis.R" not in files
    assert "pred_results/out.csv" in files


def test_selector_empty_selection_fallback_never_returns_code_files(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    _selector_workspace(v)
    v._call_judge_for_json = (  # type: ignore[method-assign]
        lambda uuid, agent_name, prompt: ({"files": []}, None)
    )

    files = v._llm_select_files("uuid-5", _visual_claim(), EXEC_TEXT, goal=GOAL_TEXT)

    assert files
    assert not any(f.endswith((".py", ".R")) for f in files)


# ---------- selector prompt carries the goal + output priority ---------------


def test_select_files_prompt_carries_goal_and_output_priority(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    prompt = v._build_select_files_prompt(
        _visual_claim(), EXEC_TEXT, "pred_results/out.csv\t12B", 6, GOAL_TEXT
    )
    assert "TASK GOAL" in prompt
    assert GOAL_TEXT in prompt
    assert "SELECTION PRIORITY" in prompt
    assert "PRODUCED OUTPUTS" in prompt
    assert "INPUT/DATASET-looking files" in prompt
    # the goal must be shown before the untrusted narration
    assert prompt.find("TASK GOAL") < prompt.find("AGENT NARRATION")

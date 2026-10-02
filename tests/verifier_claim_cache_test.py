"""Per-task rubric cache: seed once, reuse forever round-trip.

Replaces the old per-(task, source) claim-text cache. The rubric cache stores
the FINAL ranked claim list (post-dedup, post-importance) so verifier scores
stay comparable across iterations of the same task — without burning fresh
extraction / dedup / importance LLM calls every run.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))

# Prime the import chain by loading sources.core.failure_fingerprint first;
# this is the import that tests/failure_fingerprint_test.py uses and that
# implicitly resolves the verifier <-> cli circular import for every
# alphabetically-later test in the suite.
from sources.core.failure_fingerprint import compute_failure_fingerprint  # noqa: E402, F401
from sources.evaluators.verifier import VerifierEvaluator  # noqa: E402
from sources.evaluators.verifier_claim_sources import (  # noqa: E402
    ClaimContext,
    SOURCES,
)


# ---------- helpers ----------------------------------------------------------


def _make_evaluator(tmp_path: Path) -> VerifierEvaluator:
    """Build a VerifierEvaluator with paths rooted at tmp_path; LLM judge mocked."""
    config = SimpleNamespace(
        workspace_dir=str(tmp_path / "workspace"),
        temp_dir=str(tmp_path / "verifier_tmp"),
        workflow_dir=str(tmp_path / "workflows"),
        memory_dir=str(tmp_path / "memory"),
        judge_model="mock",
    )
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    (tmp_path / "verifier_tmp").mkdir(parents=True, exist_ok=True)
    (tmp_path / "workflows").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    v = VerifierEvaluator.__new__(VerifierEvaluator)
    v.workspace_dir = Path(config.workspace_dir)
    v._runner_temp_root = Path(config.temp_dir)
    v.workflow_dir = Path(config.workflow_dir)
    v.logger = logging.getLogger("test-verifier")
    v._DEFAULT_CLAIM_IMPORTANCE = 5
    v._workspace_files = set()
    return v


def _ranked_claims() -> list[dict[str, object]]:
    """Sample post-dedup, post-importance claim list as the rubric cache stores it."""
    return [
        {
            "id": "csv_exists",
            "description": "The output csv is non-empty and parseable",
            "likely_relevant_files": ["pred_results/out.csv"],
            "importance": 9,
            "importance_rationale": "literal deliverable; named in goal",
            "source": "source_b",
        },
        {
            "id": "auc_above_77",
            "description": "ROC-AUC over FDA + CT_TOX is at least 0.77",
            "likely_relevant_files": ["pred_results/out.csv", "gold.csv"],
            "importance": 10,
            "importance_rationale": "headline metric",
            "source": "source_a",
        },
    ]


# ---------- task_key -------------------------------------------------------


def test_task_cache_key_stable_per_goal():
    """Same goal → same key. Different goals → different keys."""
    k1 = VerifierEvaluator._task_cache_key("goal A")
    k2 = VerifierEvaluator._task_cache_key("goal A")
    k3 = VerifierEvaluator._task_cache_key("goal B")
    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 16
    assert all(c in "0123456789abcdef" for c in k1)


# ---------- load → None when no cache -------------------------------------


def test_load_returns_none_when_no_cache(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    assert v._load_cached_rubric("any_task_key") is None


# ---------- persist seeds the cache; never overwrites ---------------------


def test_persist_seeds_then_load_returns_ranked_list(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    task_key = v._task_cache_key("the task goal")
    v._persist_rubric(task_key, _ranked_claims())

    cache_path = v._rubric_cache_path(task_key)
    assert cache_path.exists()
    data = json.loads(cache_path.read_text())
    assert data["task_key"] == task_key
    assert {c["id"] for c in data["claims"]} == {"csv_exists", "auc_above_77"}

    loaded = v._load_cached_rubric(task_key)
    assert loaded is not None
    assert {c["id"] for c in loaded} == {"csv_exists", "auc_above_77"}
    # importance + rationale are preserved verbatim — that's the whole point
    auc = next(c for c in loaded if c["id"] == "auc_above_77")
    assert auc["importance"] == 10
    assert auc["importance_rationale"] == "headline metric"
    assert auc["source"] == "source_a"


def test_persist_never_overwrites_existing_cache(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    task_key = v._task_cache_key("g")
    v._persist_rubric(task_key, [{"id": "first", "description": "x", "importance": 5}])
    v._persist_rubric(task_key, [{"id": "second", "description": "y", "importance": 5}])
    loaded = v._load_cached_rubric(task_key)
    assert loaded is not None
    assert {c["id"] for c in loaded} == {"first"}, "first call must win"


def test_persist_skips_empty_claim_list(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    task_key = v._task_cache_key("g")
    v._persist_rubric(task_key, [])
    assert not v._rubric_cache_path(task_key).exists()
    assert v._load_cached_rubric(task_key) is None


def test_cache_is_per_task(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    task_a = v._task_cache_key("goal A")
    task_b = v._task_cache_key("goal B")
    v._persist_rubric(task_a, [{"id": "ax", "description": "a", "importance": 5}])
    v._persist_rubric(task_b, [{"id": "bx", "description": "b", "importance": 5}])
    a_loaded = v._load_cached_rubric(task_a)
    b_loaded = v._load_cached_rubric(task_b)
    assert a_loaded is not None and b_loaded is not None
    assert {c["id"] for c in a_loaded} == {"ax"}
    assert {c["id"] for c in b_loaded} == {"bx"}


# ---------- workspace re-validation -------------------------------------


def test_adapt_rubric_filters_paths_to_current_workspace(tmp_path: Path):
    """Cached `likely_relevant_files` are re-validated; gone files get dropped."""
    v = _make_evaluator(tmp_path)
    v._workspace_files = {"pred_results/out.csv"}  # gold.csv no longer exists
    adapted = v._adapt_rubric_to_workspace(_ranked_claims())
    assert len(adapted) == 2
    csv_claim = next(c for c in adapted if c["id"] == "csv_exists")
    auc_claim = next(c for c in adapted if c["id"] == "auc_above_77")
    # surviving path kept, missing path dropped
    assert csv_claim["likely_relevant_files"] == ["pred_results/out.csv"]
    assert "gold.csv" not in auc_claim["likely_relevant_files"]
    # rubric fields (importance, rationale, description, id) preserved verbatim
    assert auc_claim["importance"] == 10
    assert auc_claim["importance_rationale"] == "headline metric"
    assert auc_claim["description"] == "ROC-AUC over FDA + CT_TOX is at least 0.77"


# ---------- prompts no longer carry prior-claims plumbing -----------------


def test_source_prompts_do_not_reference_prior_claims_anymore():
    """The prior-claims instruction block was deleted with the per-source cache."""
    ctx = ClaimContext(
        goal="g", workspace_listing="file.py\t100", target_min=2, target_max=5
    )
    for source in SOURCES:
        prompt = source.build(ctx)
        assert "PRIOR CLAIMS" not in prompt, f"source {source.label} still mentions prior claims"
        assert "REPRODUCE THE SAME LIST" not in prompt, (
            f"source {source.label} still has the reproduce-list directive"
        )


# ---------- malformed cache files fail safe -------------------------------


def test_load_returns_none_on_corrupt_json(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    task_key = v._task_cache_key("g")
    path = v._rubric_cache_path(task_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json {")
    assert v._load_cached_rubric(task_key) is None


def test_load_returns_none_when_claims_list_empty(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    task_key = v._task_cache_key("g")
    path = v._rubric_cache_path(task_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task_key": task_key, "claims": []}))
    assert v._load_cached_rubric(task_key) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""Per-(task, source) claim-list cache: seed → load → render round-trip.

Replaces the deleted ``verifier_anchor_helpers_test.py``. Validates the new
seed-anchoring-free continuity layer: a per-task-per-source cache of CLAIM
TEXT (not verifier scripts), seeded once and consulted on every subsequent
extraction.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))

# Prime the import chain by loading sources.core.failure_fingerprint first;
# this is the import that tests/failure_fingerprint_test.py uses and that
# implicitly resolves the verifier <-> cli circular import for every
# alphabetically-later test in the suite.
from sources.core.failure_fingerprint import compute_failure_fingerprint  # noqa: E402, F401
from sources.evaluators import verifier as verifier_mod  # noqa: E402
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
    return v


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


# ---------- load → empty when no cache -------------------------------------


def test_load_returns_empty_string_when_no_cache(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    text = v._load_prior_claims_text("any_task_key", "b")
    assert text == ""


# ---------- persist seeds the cache; never overwrites ---------------------


def test_persist_seeds_then_load_returns_rendered_text(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    task_key = v._task_cache_key("the task goal")
    claims = [
        {
            "id": "csv_exists",
            "description": "The output csv is non-empty and parseable",
            "likely_relevant_files": ["pred_results/out.csv"],
        },
        {
            "id": "auc_above_77",
            "description": "ROC-AUC over FDA + CT_TOX is at least 0.77",
            "likely_relevant_files": ["pred_results/out.csv", "gold.csv"],
        },
    ]
    v._persist_claims_for_source(task_key, "b", claims)
    cache_path = v._claim_cache_path(task_key, "b")
    assert cache_path.exists()
    data = json.loads(cache_path.read_text())
    assert data["task_key"] == task_key
    assert data["source"] == "b"
    assert {c["id"] for c in data["claims"]} == {"csv_exists", "auc_above_77"}

    rendered = v._load_prior_claims_text(task_key, "b")
    assert "[csv_exists]" in rendered
    assert "ROC-AUC over FDA + CT_TOX is at least 0.77" in rendered
    assert "pred_results/out.csv" in rendered


def test_persist_never_overwrites_existing_cache(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    task_key = v._task_cache_key("g")
    v._persist_claims_for_source(task_key, "a", [{"id": "first", "description": "x"}])
    v._persist_claims_for_source(task_key, "a", [{"id": "second", "description": "y"}])
    data = json.loads(v._claim_cache_path(task_key, "a").read_text())
    assert {c["id"] for c in data["claims"]} == {"first"}, "first call must win"


def test_persist_skips_empty_claim_list(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    task_key = v._task_cache_key("g")
    v._persist_claims_for_source(task_key, "c", [])
    assert not v._claim_cache_path(task_key, "c").exists()


def test_cache_is_per_source_per_task(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    task_a = v._task_cache_key("goal A")
    task_b = v._task_cache_key("goal B")
    v._persist_claims_for_source(task_a, "b", [{"id": "ab", "description": "task A source b"}])
    v._persist_claims_for_source(task_a, "c", [{"id": "ac", "description": "task A source c"}])
    v._persist_claims_for_source(task_b, "b", [{"id": "bb", "description": "task B source b"}])
    assert "[ab]" in v._load_prior_claims_text(task_a, "b")
    assert "[ac]" in v._load_prior_claims_text(task_a, "c")
    assert "[bb]" in v._load_prior_claims_text(task_b, "b")
    # cross-pollution check
    assert "[ab]" not in v._load_prior_claims_text(task_b, "b")
    assert "[bb]" not in v._load_prior_claims_text(task_a, "b")


# ---------- prompt rendering with / without prior claims ------------------


def test_source_prompt_omits_prior_block_when_cache_empty():
    ctx = ClaimContext(
        goal="g", workspace_listing="file.py\t100", target_min=2, target_max=5
    )
    for source in SOURCES:
        prompt = source.build(ctx)
        assert "PRIOR CLAIMS — REPRODUCE" not in prompt, f"source {source.label} leaked prior block"


def test_source_prompt_includes_prior_block_when_cache_seeded():
    prior_text = "- [csv_exists] The output csv is non-empty"
    ctx = ClaimContext(
        goal="g", workspace_listing="x", target_min=2, target_max=5, prior_claims=prior_text
    )
    for source in SOURCES:
        prompt = source.build(ctx)
        assert "PRIOR CLAIMS — REPRODUCE" in prompt, f"source {source.label} missing prior block"
        assert "[csv_exists]" in prompt, f"source {source.label} did not interpolate prior text"
        assert "KEEP the same `id` verbatim" in prompt, f"source {source.label} missing reuse-id rule"


# ---------- malformed cache files fail safe -------------------------------


def test_load_returns_empty_on_corrupt_json(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    task_key = v._task_cache_key("g")
    path = v._claim_cache_path(task_key, "d")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json {")
    assert v._load_prior_claims_text(task_key, "d") == ""


def test_load_returns_empty_when_claims_list_missing(tmp_path: Path):
    v = _make_evaluator(tmp_path)
    task_key = v._task_cache_key("g")
    path = v._claim_cache_path(task_key, "e")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task_key": task_key, "source": "e", "claims": []}))
    assert v._load_prior_claims_text(task_key, "e") == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""Tests for the lineage-rubric helpers on ``VerifierEvaluator``.

These cover ``_persist_claims`` → ``_load_anchored_claims`` round-trip
(incl. legacy ``criticality`` → ``importance`` migration), ``_spec_from_anchor``
(executable / soft / missing-script fallback) and the ``_claims_from_anchor``
projection. We avoid ``BaseEvaluator.__init__`` so tests run without an
OpenRouter pricing call.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))

from sources.evaluators.verifier import VerifierEvaluator  # noqa: E402


class _StubVerifier(VerifierEvaluator):
    """Minimal subclass that skips ``BaseEvaluator.__init__``.

    Only attributes the rubric helpers reach for are set. The helpers under
    test depend on nothing else; if they ever do, the test will fail loudly
    instead of silently using production defaults.
    """

    def __init__(self, tmp_root: Path) -> None:  # noqa: D401
        self._runner_temp_root = Path(tmp_root)
        self.logger = logging.getLogger("test_verifier_helpers")


def _sample_claims() -> list[dict]:
    return [
        {
            "id": "ecfp_featurization",
            "description": "Workflow featurises with ECFP.",
            "importance": 9,
            "importance_rationale": "domain-required featurization",
            "source": "source_a",
            "likely_relevant_files": ["clintox_nn.py"],
        },
        {
            "id": "deps_manifest",
            "description": "Workspace ships a requirements manifest.",
            "importance": 3,
            "importance_rationale": "nice-to-have for reproducibility",
            "source": "source_b",
            "likely_relevant_files": [],
        },
    ]


def _sample_per_claim() -> list[dict]:
    return [
        {
            "claim": {"id": "ecfp_featurization"},
            "spec": {"executable": True, "code": "print('ok')"},
        },
        {
            "claim": {"id": "deps_manifest"},
            "spec": {"executable": False, "reason": "needs human judgement"},
        },
    ]


def test_persist_and_load_round_trip(tmp_path: Path) -> None:
    """Persisted rubric loads back with importance, source, and exec flag intact."""
    v = _StubVerifier(tmp_path)
    v._persist_claims("u1", _sample_claims(), _sample_per_claim())

    cache = tmp_path / "u1" / "claims.json"
    assert cache.exists(), "claims.json should be written next to verify_*.py"

    loaded = v._load_anchored_claims("u1")
    assert loaded is not None and len(loaded) == 2

    by_id = {c["id"]: c for c in loaded}
    assert by_id["ecfp_featurization"]["importance"] == 9
    assert by_id["ecfp_featurization"]["importance_rationale"] == "domain-required featurization"
    assert by_id["ecfp_featurization"]["executable"] is True
    assert by_id["deps_manifest"]["importance"] == 3
    assert by_id["deps_manifest"]["executable"] is False
    assert by_id["deps_manifest"]["reason"] == "needs human judgement"


def test_load_upgrades_legacy_criticality_to_importance(tmp_path: Path) -> None:
    """Pre-migration caches (``criticality`` only) are upgraded on read.

    Without the legacy mapping, existing rubric anchors written before the
    criticality → importance switch would score under uniform default
    importance — destroying QD comparability across the lineage. The on-read
    upgrade keeps old anchors scoreable: hard → 8, soft → 3.
    """
    v = _StubVerifier(tmp_path)
    folder = tmp_path / "u_legacy"
    folder.mkdir(parents=True)
    payload = {
        "anchor_uuid": "u_legacy",
        "claims": [
            {"id": "k_hard", "description": "x", "criticality": "hard"},
            {"id": "k_soft", "description": "y", "criticality": "soft"},
        ],
    }
    (folder / "claims.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = v._load_anchored_claims("u_legacy")
    assert loaded is not None
    by_id = {c["id"]: c for c in loaded}
    assert by_id["k_hard"]["importance"] == 8
    assert by_id["k_soft"]["importance"] == 3


def test_load_returns_none_when_cache_missing(tmp_path: Path) -> None:
    """Missing cache file → None, so caller falls back to LLM path."""
    v = _StubVerifier(tmp_path)
    assert v._load_anchored_claims("never_evaluated") is None


def test_load_returns_none_when_cache_malformed(tmp_path: Path) -> None:
    """Truncated / non-JSON cache must not raise — return None instead."""
    v = _StubVerifier(tmp_path)
    cache = tmp_path / "u_broken" / "claims.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("not json {{", encoding="utf-8")
    assert v._load_anchored_claims("u_broken") is None


def test_load_drops_claims_with_missing_or_empty_id(tmp_path: Path) -> None:
    """Entries without a non-empty string id are dropped with a warning.

    Regression: an earlier filter (``c.get("id")``) silently dropped both
    missing-id and empty-string-id entries, so the denominator and per-claim
    results diverged between the anchor and its descendants — defeating the
    point of stable scoring.
    """
    v = _StubVerifier(tmp_path)
    folder = tmp_path / "u_dirty"
    folder.mkdir(parents=True)
    payload = {
        "anchor_uuid": "u_dirty",
        "claims": [
            {"id": "real", "description": "x", "importance": 8},
            {"id": "", "description": "empty"},  # dropped
            {"description": "no id at all"},  # dropped
            {"id": 42, "description": "non-string id"},  # dropped
        ],
    }
    (folder / "claims.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = v._load_anchored_claims("u_dirty")
    assert loaded is not None
    assert [c["id"] for c in loaded] == ["real"]


def test_load_returns_none_on_empty_claim_list(tmp_path: Path) -> None:
    """An empty rubric is treated as no cache so the LLM path runs."""
    v = _StubVerifier(tmp_path)
    cache = tmp_path / "u_empty" / "claims.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"anchor_uuid": "u_empty", "claims": []}), encoding="utf-8")
    assert v._load_anchored_claims("u_empty") is None


def test_spec_from_anchor_executable_reads_script(tmp_path: Path) -> None:
    """An executable claim re-hydrates its ``code`` from the anchor folder."""
    v = _StubVerifier(tmp_path)
    folder = tmp_path / "anchor"
    folder.mkdir(parents=True)
    (folder / "verify_my_claim.py").write_text("print('hi')\n", encoding="utf-8")

    spec = v._spec_from_anchor("anchor", "my_claim", executable=True, reason="")
    assert spec == {"executable": True, "code": "print('hi')\n"}


def test_spec_from_anchor_missing_script_falls_back_to_soft(tmp_path: Path) -> None:
    """Missing executable script must NOT crash — degrade to a soft spec.

    Surviving in this case keeps the score denominator stable: the claim
    still appears in the aggregate, just as a soft check instead of skipped.
    """
    v = _StubVerifier(tmp_path)
    (tmp_path / "anchor").mkdir(parents=True)  # folder exists, script does not

    spec = v._spec_from_anchor("anchor", "ghost", executable=True, reason="")
    assert spec["executable"] is False
    assert "unreadable" in spec["reason"]


def test_spec_from_anchor_soft_uses_reason(tmp_path: Path) -> None:
    """Non-executable claims surface their cached rationale verbatim."""
    v = _StubVerifier(tmp_path)
    spec = v._spec_from_anchor("anchor", "soft", executable=False, reason="human-judged")
    assert spec == {"executable": False, "reason": "human-judged"}


def test_claims_from_anchor_drops_persistence_fields(tmp_path: Path) -> None:
    """``_claims_from_anchor`` projects back into the shape ``_verify_claim`` wants.

    Specifically, the persisted ``executable`` and ``reason`` keys must NOT
    leak into the per-claim dict — they live on the spec, not the claim.
    Importance + rationale ARE copied through so the per-claim view drives
    weighting just like a freshly-extracted claim.
    """
    v = _StubVerifier(tmp_path)
    anchored = [{
        "id": "c1",
        "description": "x",
        "importance": 9,
        "importance_rationale": "load-bearing",
        "source": "source_a",
        "likely_relevant_files": ["a.py"],
        "executable": True,
        "reason": "",
    }]
    projected = v._claims_from_anchor(anchored)
    assert projected == [{
        "id": "c1",
        "description": "x",
        "importance": 9,
        "importance_rationale": "load-bearing",
        "source": "source_a",
        "likely_relevant_files": ["a.py"],
    }]


class _RegenStubVerifier(_StubVerifier):
    """Stub that also stubs the LLM call used by anchor regeneration.

    ``llm_responses`` is consumed in order, one tuple per ``_call_judge_for_json``
    invocation. Each tuple is ``(parsed_json_or_None, err_or_None)`` mirroring
    the production helper's return shape.
    """

    def __init__(
        self,
        tmp_root: Path,
        workspace_files: list[str] | None = None,
        llm_responses: list[tuple] | None = None,
    ) -> None:
        super().__init__(tmp_root)
        self._workspace_files = set(workspace_files or [])
        self._llm_responses = list(llm_responses or [])
        self.gen_parallelism = 1  # serial so response order is deterministic

    def _call_judge_for_json(self, uuid: str, agent_name: str, prompt: str):  # type: ignore[override]
        if not self._llm_responses:
            return None, "no stub response left"
        return self._llm_responses.pop(0)


def _stale_record(cid: str, files: list[str]) -> dict:
    """One anchored record carrying the persistence fields the helpers expect."""
    return {
        "id": cid,
        "description": f"claim about {cid}",
        "importance": 7,
        "importance_rationale": "load-bearing",
        "source": "source_anchor",
        "likely_relevant_files": files,
        "executable": True,
        "reason": "",
    }


def test_partition_by_file_presence_splits_fresh_vs_stale(tmp_path: Path) -> None:
    """Records whose files are all present go fresh; any missing path is stale."""
    v = _RegenStubVerifier(tmp_path, workspace_files=["keep.csv", "still_here.py"])
    records = [
        _stale_record("fresh_one", ["keep.csv"]),
        _stale_record("stale_one", ["gone.csv"]),
        _stale_record("partial_stale", ["keep.csv", "missing.json"]),
        _stale_record("no_files", []),  # empty → nothing to invalidate
    ]
    fresh, stale = v._partition_anchored_by_file_presence(records)
    assert [r["id"] for r in fresh] == ["fresh_one", "no_files"]
    assert [r["id"] for r in stale] == ["stale_one", "partial_stale"]


def test_persist_uses_post_selection_likely_relevant_files(tmp_path: Path) -> None:
    """``_persist_claims`` must store the file list the script actually opens,
    not the extraction's (the two diverge once ``_llm_select_files`` runs).
    Without this, descendants see metadata that doesn't track the script's
    real targets and the freshness check leaks stale anchors through.
    """
    v = _RegenStubVerifier(tmp_path)
    # Original extraction picked "output.csv"; selection (post-_llm_select_files)
    # ended up targeting "workflow.py" — that's what the per_claim entry carries.
    original_claims = [{
        "id": "runtime",
        "description": "runtime under 100ms",
        "importance": 8,
        "importance_rationale": "deliverable target",
        "source": "source_a",
        "likely_relevant_files": ["output.csv"],
    }]
    per_claim = [{
        "claim": {**original_claims[0], "likely_relevant_files": ["workflow.py"]},
        "spec": {"executable": True, "code": "open('workflow.py')"},
    }]
    v._persist_claims("u1", original_claims, per_claim)
    loaded = v._load_anchored_claims("u1")
    assert loaded is not None
    assert loaded[0]["likely_relevant_files"] == ["workflow.py"]


def test_regenerate_one_stale_claim_returns_updated_record(tmp_path: Path) -> None:
    """A successful judge call yields a record with new files + soft-spec markers."""
    new_files = ["fresh_output.csv"]
    v = _RegenStubVerifier(
        tmp_path,
        workspace_files=new_files,
        llm_responses=[(
            {"id": "c1", "description": "adapted", "likely_relevant_files": new_files},
            None,
        )],
    )
    rec = _stale_record("c1", ["gone.csv"])
    regenerated = v._regenerate_one_stale_claim("u1", rec, "narration", "listing")
    assert regenerated is not None
    assert regenerated["id"] == "c1"
    assert regenerated["importance"] == 7  # carried over from the original
    assert regenerated["importance_rationale"] == "load-bearing"
    assert regenerated["description"] == "adapted"
    assert regenerated["likely_relevant_files"] == new_files
    # Cached script's hard-coded paths are wrong → force soft fallback.
    assert regenerated["executable"] is False
    assert "stale anchor" in regenerated["reason"]


def test_regenerate_one_stale_claim_drops_confabulated_paths(tmp_path: Path) -> None:
    """Paths the LLM invents that aren't in the workspace are filtered out."""
    v = _RegenStubVerifier(
        tmp_path,
        workspace_files=["real.csv"],
        llm_responses=[(
            {"id": "c2", "description": "x", "likely_relevant_files": ["real.csv", "made_up.csv"]},
            None,
        )],
    )
    regenerated = v._regenerate_one_stale_claim(
        "u1", _stale_record("c2", ["gone.csv"]), "narration", "listing"
    )
    assert regenerated is not None
    assert regenerated["likely_relevant_files"] == ["real.csv"]


def test_regenerate_one_stale_claim_drops_on_judge_error(tmp_path: Path) -> None:
    """A failed judge call (or non-dict JSON) drops the claim rather than keeping stale refs."""
    v = _RegenStubVerifier(
        tmp_path,
        workspace_files=["real.csv"],
        llm_responses=[(None, "judge timed out")],
    )
    assert v._regenerate_one_stale_claim(
        "u1", _stale_record("c3", ["gone.csv"]), "narration", "listing"
    ) is None


def test_resolve_anchored_claims_returns_empty_when_no_anchor(tmp_path: Path) -> None:
    """No anchor uuid → empty list + None so caller falls through to extraction."""
    v = _RegenStubVerifier(tmp_path)
    claims, spec_reuse = v._resolve_anchored_claims("u1", None, "exec", "listing")
    assert claims == [] and spec_reuse is None


def test_resolve_anchored_claims_returns_empty_when_cache_missing(tmp_path: Path) -> None:
    """Anchor specified but no cache file → fall through, warning logged."""
    v = _RegenStubVerifier(tmp_path)
    claims, spec_reuse = v._resolve_anchored_claims("u1", "ghost_anchor", "exec", "listing")
    assert claims == [] and spec_reuse is None


def test_resolve_anchored_claims_falls_through_when_all_dropped(tmp_path: Path) -> None:
    """Anchor exists but every stale claim's regen fails → caller re-extracts."""
    # Persist a rubric whose files don't exist in the current workspace.
    cache_dir = tmp_path / "old_anchor"
    cache_dir.mkdir()
    (cache_dir / "claims.json").write_text(
        json.dumps({"anchor_uuid": "old_anchor", "claims": [
            _stale_record("only_one", ["gone.csv"]),
        ]}),
        encoding="utf-8",
    )
    v = _RegenStubVerifier(
        tmp_path,
        workspace_files=["something_else.csv"],
        llm_responses=[(None, "judge dead")],  # regen drops this single claim
    )
    claims, spec_reuse = v._resolve_anchored_claims("u1", "old_anchor", "exec", "listing")
    assert claims == [] and spec_reuse is None


def test_resolve_anchored_claims_keeps_fresh_and_regenerated(tmp_path: Path) -> None:
    """End-to-end: fresh records reuse cached scripts; stale ones get regenerated."""
    cache_dir = tmp_path / "anc"
    cache_dir.mkdir()
    (cache_dir / "claims.json").write_text(
        json.dumps({"anchor_uuid": "anc", "claims": [
            _stale_record("fresh_claim", ["keep.csv"]),
            _stale_record("stale_claim", ["gone.csv"]),
        ]}),
        encoding="utf-8",
    )
    v = _RegenStubVerifier(
        tmp_path,
        workspace_files=["keep.csv", "new.csv"],
        llm_responses=[(
            {"id": "stale_claim", "description": "now points at new.csv",
             "likely_relevant_files": ["new.csv"]},
            None,
        )],
    )
    claims, spec_reuse = v._resolve_anchored_claims("u1", "anc", "exec", "listing")
    assert {c["id"] for c in claims} == {"fresh_claim", "stale_claim"}
    # spec_reuse_records holds only the fresh subset — the regenerated claim's
    # cached verify_stale_claim.py would reference gone.csv and must be redone.
    assert spec_reuse is not None
    assert [r["id"] for r in spec_reuse] == ["fresh_claim"]


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_persist_and_load_round_trip(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_partition_by_file_presence_splits_fresh_vs_stale(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_regenerate_one_stale_claim_returns_updated_record(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_resolve_anchored_claims_keeps_fresh_and_regenerated(Path(d))
    print("verifier_anchor_helpers_test: smoke ok")

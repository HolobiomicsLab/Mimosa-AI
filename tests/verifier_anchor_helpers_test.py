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


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_persist_and_load_round_trip(Path(d))
    print("verifier_anchor_helpers_test: smoke ok")

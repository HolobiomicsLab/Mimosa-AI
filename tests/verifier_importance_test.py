"""Tests for the importance-weighted scoring path on ``VerifierEvaluator``.

These cover the post-criticality refactor:
- ``_claim_weight`` reads importance and clamps to [1, 10].
- ``_aggregate`` weights claims by importance so a high-importance flip moves
  the score strictly more than a low-importance one (the whole reason for the
  refactor — the old 3-vs-1 step function couldn't express that).
- The thoroughness bonus only fires on high-importance passes.
- The hard-fail cap triggers on refuted ``importance >= 8`` claims.
- ``_build_report(min_importance=...)`` produces the filtered view used as the
  gradient builder's input — low-importance claims are suppressed.
- ``_declare_claim_importance`` returns the input claims with uniform default
  importance when the rater LLM is unavailable, so a transient outage does not
  crash evaluation.

We avoid ``BaseEvaluator.__init__`` so the suite runs without an OpenRouter
pricing call (same pattern as ``verifier_anchor_helpers_test.py``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))

from sources.core.evaluators import verifier as verifier_mod  # noqa: E402
from sources.core.evaluators.verifier import VerifierEvaluator  # noqa: E402


class _StubVerifier(VerifierEvaluator):
    """Minimal subclass that skips ``BaseEvaluator.__init__``.

    Only the attributes the scoring helpers read are set; we pull the
    production tunables directly off the module so the test reflects the
    live values instead of duplicating them.
    """

    def __init__(self) -> None:  # noqa: D401
        self.logger = logging.getLogger("test_verifier_importance")
        self.hard_fail_cap = verifier_mod._HARD_FAIL_CAP
        self.info_bonus_alpha = verifier_mod._INFO_BONUS_ALPHA
        self.info_bonus_beta = verifier_mod._INFO_BONUS_BETA


def _claim(cid: str, importance: int, status: str, rationale: str = "") -> dict:
    """Build a scored-claim entry in the shape ``_aggregate`` consumes."""
    return {
        "claim": {
            "id": cid,
            "description": cid,
            "importance": importance,
            "importance_rationale": rationale,
            "likely_relevant_files": [],
        },
        "status": status,
        "score": 1.0 if status == "pass" else 0.0,
        "verifier_kind": "executable",
        "details": "",
    }


def test_claim_weight_returns_importance_clamped() -> None:
    """Weight equals importance, clamped to [1, 10], default 5 on missing/bad."""
    v = _StubVerifier()
    assert v._claim_weight({"claim": {"importance": 9}}) == 9.0
    assert v._claim_weight({"claim": {"importance": 0}}) == 1.0  # clamped up
    assert v._claim_weight({"claim": {"importance": 99}}) == 10.0  # clamped down
    assert v._claim_weight({"claim": {}}) == 5.0  # default
    assert v._claim_weight({"claim": {"importance": "nine"}}) == 5.0  # bad type


def test_high_importance_flip_moves_score_more_than_low_importance_flip() -> None:
    """The whole point of the refactor: gradient strength tracks importance.

    Under the old hard/soft tier this ratio was capped at 3 (weight 3 vs 1).
    Under the new scale a 10-vs-2 flip should move the score by 5×.
    """
    v = _StubVerifier()

    # Baseline: one importance-10 PASS + one importance-2 PASS. Both flips
    # compared against this so any bonus/cap effect cancels out.
    base = v._aggregate([
        _claim("deliverable", importance=10, status="pass"),
        _claim("nitpick", importance=2, status="pass"),
    ])

    # Flip the importance-10 claim to fail.
    flip_high = v._aggregate([
        _claim("deliverable", importance=10, status="fail"),
        _claim("nitpick", importance=2, status="pass"),
    ])

    # Flip the importance-2 claim to fail.
    flip_low = v._aggregate([
        _claim("deliverable", importance=10, status="pass"),
        _claim("nitpick", importance=2, status="fail"),
    ])

    delta_high = base["base_mean"] - flip_high["base_mean"]
    delta_low = base["base_mean"] - flip_low["base_mean"]

    # Ratio is 10/2 = 5 by construction of the weighted mean. _aggregate
    # rounds base_mean to 4 dp before returning, so a 1e-2 tolerance is
    # the right structural-equality bar (the old hard/soft tier topped out
    # at a ratio of 3).
    assert delta_high > delta_low
    assert abs(delta_high / delta_low - 5.0) < 1e-2


def test_hard_fail_cap_triggers_only_on_high_importance_refutation() -> None:
    """Refuting an importance≥8 claim flips ``hard_fail_capped``; lower does not."""
    v = _StubVerifier()

    capped = v._aggregate([
        _claim("must_have", importance=8, status="fail"),
        _claim("ok", importance=5, status="pass"),
    ])
    assert capped["hard_fail_capped"] is True

    not_capped = v._aggregate([
        _claim("nice_to_have", importance=7, status="fail"),
        _claim("ok", importance=5, status="pass"),
    ])
    assert not_capped["hard_fail_capped"] is False


def test_hard_fail_cap_does_not_fire_on_errored_claims() -> None:
    """An ``error`` verdict is measurement failure, not refutation — no cap."""
    v = _StubVerifier()
    scores = v._aggregate([
        _claim("deliverable", importance=10, status="error"),
        _claim("ok", importance=6, status="pass"),
    ])
    assert scores["hard_fail_capped"] is False


def test_information_bonus_counts_only_high_importance_passes() -> None:
    """A workspace of importance-3 passes earns no bonus; importance-9 does."""
    v = _StubVerifier()

    low_only = v._aggregate([
        _claim("a", importance=3, status="pass"),
        _claim("b", importance=3, status="pass"),
        _claim("c", importance=3, status="pass"),
    ])
    high_only = v._aggregate([
        _claim("a", importance=9, status="pass"),
    ])

    assert low_only["n_high_importance_pass"] == 0
    assert low_only["information_bonus"] == 0.0
    assert high_only["n_high_importance_pass"] == 1
    assert high_only["information_bonus"] > 0.0


def test_aggregate_emits_new_telemetry_keys() -> None:
    """The score dict carries the new importance-based fields, not the old ones."""
    v = _StubVerifier()
    scores = v._aggregate([_claim("x", importance=6, status="pass")])
    assert "n_high_importance_pass" in scores
    assert "high_importance_pass_mass" in scores
    assert "n_hard_pass" not in scores  # removed by the refactor


def test_build_report_filters_below_min_importance() -> None:
    """The gradient view drops low-importance lines; the unfiltered view keeps them."""
    v = _StubVerifier()
    per_claim = [
        _claim("deliverable", importance=10, status="fail", rationale="produces the asked CSV"),
        _claim("noise", importance=2, status="fail", rationale="cosmetic only"),
    ]
    scores = v._aggregate(per_claim)

    full = v._build_report(per_claim, scores, min_importance=0)
    gradient = v._build_report(per_claim, scores, min_importance=6)

    assert "[deliverable]" in full and "[noise]" in full
    assert "[deliverable]" in gradient
    assert "[noise]" not in gradient, "importance=2 must be suppressed in the gradient view"
    assert "produces the asked CSV" in gradient, "rationale propagates into the report"


def test_declare_claim_importance_falls_back_when_rater_unavailable() -> None:
    """A judge-call failure → uniform default importance, never a crash."""
    v = _StubVerifier()

    # Force ``_call_judge_for_json`` to simulate a provider outage.
    v._call_judge_for_json = lambda *_a, **_kw: (None, "simulated judge outage")  # type: ignore[method-assign]

    claims_in = [
        {"id": "a", "description": "x", "source": "source_a", "likely_relevant_files": []},
        {"id": "b", "description": "y", "source": "source_b", "likely_relevant_files": []},
    ]
    out = v._declare_claim_importance("u_fallback", "task goal", claims_in)

    assert len(out) == 2
    for c in out:
        assert c["importance"] == VerifierEvaluator._DEFAULT_CLAIM_IMPORTANCE
        assert c["importance_rationale"] == ""


def test_declare_claim_importance_drops_duplicates_and_clamps_scale() -> None:
    """Rater output is respected: drop ids removed, importance clamped to [1,10]."""
    v = _StubVerifier()

    def _fake_judge(_uuid, _agent, _prompt):
        return ({
            "drop_ids": ["dup"],
            "importance": [
                {"id": "keep", "importance": 11, "rationale": "deliverable"},  # clamp to 10
                {"id": "other", "importance": -2, "rationale": "noise"},  # clamp to 1
            ],
        }, None)

    v._call_judge_for_json = _fake_judge  # type: ignore[method-assign]

    claims_in = [
        {"id": "keep", "description": "x", "source": "source_a", "likely_relevant_files": []},
        {"id": "dup", "description": "x", "source": "source_b", "likely_relevant_files": []},
        {"id": "other", "description": "y", "source": "source_a", "likely_relevant_files": []},
    ]
    out = v._declare_claim_importance("u_rater", "task", claims_in)

    out_by_id = {c["id"]: c for c in out}
    assert set(out_by_id) == {"keep", "other"}, "duplicate id must be dropped"
    assert out_by_id["keep"]["importance"] == 10
    assert out_by_id["other"]["importance"] == 1


def test_extract_drop_ids_ignores_malformed_input() -> None:
    """A non-list (or list of non-strings) yields an empty drop set, not a crash."""
    assert VerifierEvaluator._extract_drop_ids({"drop_ids": "oops"}) == set()
    assert VerifierEvaluator._extract_drop_ids({"drop_ids": [1, 2, "real"]}) == {"real"}
    assert VerifierEvaluator._extract_drop_ids({}) == set()


if __name__ == "__main__":
    test_claim_weight_returns_importance_clamped()
    test_high_importance_flip_moves_score_more_than_low_importance_flip()
    test_hard_fail_cap_triggers_only_on_high_importance_refutation()
    test_hard_fail_cap_does_not_fire_on_errored_claims()
    test_information_bonus_counts_only_high_importance_passes()
    test_aggregate_emits_new_telemetry_keys()
    test_build_report_filters_below_min_importance()
    test_declare_claim_importance_falls_back_when_rater_unavailable()
    test_declare_claim_importance_drops_duplicates_and_clamps_scale()
    test_extract_drop_ids_ignores_malformed_input()
    print("verifier_importance_test: smoke ok")

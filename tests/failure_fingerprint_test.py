"""Tests for the failure-fingerprint behaviour descriptor.

The centering rule is the quality firewall: an all-pass run and an all-fail
run must both yield the zero profile. Anything else lets overall quality
leak into novelty and collapses QD into greedy search.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from sources.core.failure_fingerprint import (
    DESCRIPTOR_DIM,
    SOURCES,
    compute_failure_fingerprint,
    failure_fingerprint_from_state_result,
    failure_fingerprint_from_workflow_dir,
    neutral_fingerprint,
)


def _claim(letter: str, status: str) -> dict:
    return {"claim": {"source": f"source_{letter}"}, "status": status}


def test_dimension_is_six():
    assert DESCRIPTOR_DIM == 6
    assert SOURCES == ("a", "b", "c", "d", "e", "f")


def test_all_pass_yields_zero_profile():
    """Quality firewall: an all-pass run must be neutral in profile space."""
    per_claim = [_claim(s, "pass") for s in SOURCES]
    out = compute_failure_fingerprint(per_claim)
    assert out["vector"] == [0.0] * DESCRIPTOR_DIM
    assert out["presence_mask"] == [1.0] * DESCRIPTOR_DIM
    assert out["pass_rates"] == [1.0] * DESCRIPTOR_DIM


def test_all_fail_yields_zero_profile():
    """Quality firewall: an all-fail run must also be neutral in profile space."""
    per_claim = [_claim(s, "fail") for s in SOURCES]
    out = compute_failure_fingerprint(per_claim)
    assert out["vector"] == [0.0] * DESCRIPTOR_DIM
    assert out["presence_mask"] == [1.0] * DESCRIPTOR_DIM
    assert out["pass_rates"] == [0.0] * DESCRIPTOR_DIM


def test_distinct_profiles_yield_nonzero_distance():
    """Two runs at the same overall quality but different shapes must differ."""
    a_passes_b_fails = compute_failure_fingerprint([
        _claim("a", "pass"),
        _claim("b", "fail"),
    ])["vector"]
    a_fails_b_passes = compute_failure_fingerprint([
        _claim("a", "fail"),
        _claim("b", "pass"),
    ])["vector"]
    assert a_passes_b_fails != a_fails_b_passes
    # Same overall quality (50% pass), so identical means; opposite shape.
    assert a_passes_b_fails[0] == -a_fails_b_passes[0]
    assert a_passes_b_fails[1] == -a_fails_b_passes[1]


def test_absent_source_uses_neutral_and_mask():
    """A source with zero claims must not influence the centered mean."""
    out = compute_failure_fingerprint([
        _claim("a", "pass"),
        _claim("b", "pass"),
        _claim("c", "fail"),
    ])
    assert out["presence_mask"] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    # Mean over present sources only: (1+1+0)/3 = 0.667.
    assert out["pass_rates"][:3] == [1.0, 1.0, 0.0]
    assert out["pass_rates"][3:] == [0.5, 0.5, 0.5]
    # Centered: present entries minus 0.667; absent entries forced to zero.
    assert all(abs(out["vector"][i]) > 1e-6 for i in range(3))
    assert out["vector"][3:] == [0.0, 0.0, 0.0]


def test_non_pass_statuses_count_as_failures():
    """``fail``, ``error``, and ``unsure`` all reduce the per-source pass rate."""
    out = compute_failure_fingerprint([
        _claim("a", "pass"),
        _claim("a", "fail"),
        _claim("a", "error"),
        _claim("a", "unsure"),
    ])
    assert abs(out["pass_rates"][0] - 0.25) < 1e-9


def test_source_label_normalisation():
    """Accepts ``"a"``, ``"A"``, ``"source_a"``, ``"SOURCE_A"`` interchangeably."""
    variants = [
        {"claim": {"source": "a"}, "status": "pass"},
        {"claim": {"source": "B"}, "status": "fail"},
        {"claim": {"source": "source_c"}, "status": "pass"},
        {"claim": {"source": "SOURCE_D"}, "status": "fail"},
    ]
    out = compute_failure_fingerprint(variants)
    assert out["presence_mask"] == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_unknown_source_is_ignored():
    """Claims tagged with an unknown source must not poison any bucket."""
    out = compute_failure_fingerprint([
        _claim("a", "pass"),
        {"claim": {"source": "source_z"}, "status": "fail"},
        {"claim": {"source": None}, "status": "pass"},
    ])
    assert out["presence_mask"] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_empty_per_claim_yields_neutral_vector():
    """No claims at all → all-zero vector, all-zero mask, neutral 0.5 rates."""
    out = compute_failure_fingerprint([])
    assert out["vector"] == [0.0] * DESCRIPTOR_DIM
    assert out["presence_mask"] == [0.0] * DESCRIPTOR_DIM
    assert out["pass_rates"] == [0.5] * DESCRIPTOR_DIM


def test_state_result_projection():
    """``failure_fingerprint_from_state_result`` returns the centered vector."""
    fp = compute_failure_fingerprint([_claim("a", "pass"), _claim("b", "fail")])
    state = {"evaluation": {"verifier": {"failure_fingerprint": fp}}}
    assert failure_fingerprint_from_state_result(state) == fp["vector"]


def test_state_result_projection_missing_returns_none():
    assert failure_fingerprint_from_state_result(None) is None
    assert failure_fingerprint_from_state_result({}) is None
    assert failure_fingerprint_from_state_result({"evaluation": {}}) is None
    assert failure_fingerprint_from_state_result(
        {"evaluation": {"verifier": {"failure_fingerprint": {"vector": [0.0, 0.0]}}}}
    ) is None


def test_workflow_dir_round_trip(tmp_path_factory=None):
    """Round-trip via on-disk ``state_result.json``."""
    import json
    import tempfile

    fp = compute_failure_fingerprint([_claim("a", "pass"), _claim("b", "fail")])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        uuid = "uuid123"
        (root / uuid).mkdir()
        (root / uuid / "state_result.json").write_text(
            json.dumps({"evaluation": {"verifier": {"failure_fingerprint": fp}}}),
            encoding="utf-8",
        )
        assert failure_fingerprint_from_workflow_dir(root, uuid) == fp["vector"]
        assert failure_fingerprint_from_workflow_dir(root, "missing") is None


def test_neutral_fingerprint_helper():
    assert neutral_fingerprint() == [0.0] * DESCRIPTOR_DIM


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")
    print("All failure_fingerprint_test passed.")

"""Tests for SelectionPressure archive admission."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.append(str(Path(__file__).parent.parent))

from sources.core.failure_fingerprint import DESCRIPTOR_DIM
from sources.core.selection import PopulationMember, SelectionPressure


def _fp_state(vector: list[float]) -> dict:
    """Build a state_result stub exposing a centered failure fingerprint."""
    assert len(vector) == DESCRIPTOR_DIM
    return {
        "evaluation": {
            "verifier": {
                "failure_fingerprint": {
                    "vector": vector,
                    "presence_mask": [1.0] * DESCRIPTOR_DIM,
                    "pass_rates": [0.5] * DESCRIPTOR_DIM,
                }
            }
        }
    }


def _run(
    vector: list[float] | None = None,
    reward: float = 0.9,
    uuid: str = "u",
) -> SimpleNamespace:
    return SimpleNamespace(
        reward=reward,
        reward_uncapped=reward,
        current_uuid=uuid,
        iteration_count=1,
        cost=0.0,
        state_result=_fp_state(vector if vector is not None else [0.0] * DESCRIPTOR_DIM),
    )


def test_first_run_admitted():
    sp = SelectionPressure(strategy="qd", population_size=50)
    sp._validate_open_ended(
        [_run(reward=0.0)],
        [_run(reward=0.97, uuid="seed", vector=[0.3, -0.2, 0.0, 0.0, -0.1, 0.0])],
        threshold=0.01,
    )
    assert [m.uuid for m in sp._archive] == ["seed"]


def test_distinct_failure_profile_sibling_is_admitted():
    """Regression case under the new fingerprint descriptor.

    Mirror of the ClinTox c83cfabb scenario, restated in fingerprint space:
    a lower-reward sibling whose failure profile fails on different sources
    must land — quality-only selection would Pareto-reject it.
    """
    sp = SelectionPressure(strategy="qd", population_size=50, novelty_k_neighbours=25)
    seed = PopulationMember(
        iteration=1, reward=0.97, cost=0.0, uuid="seed", reward_uncapped=1.05,
        behaviour_descriptor=[0.3, -0.2, 0.0, 0.0, -0.1, 0.0],
    )
    sp._archive = [seed]

    distinct = _run(
        reward=0.91,
        uuid="distinct",
        vector=[-0.2, 0.1, 0.0, -0.4, 0.5, 0.0],
    )

    sp._validate_open_ended([seed], [distinct], threshold=0.01)
    assert {m.uuid for m in sp._archive} == {"seed", "distinct"}


def test_invalid_candidate_rejected():
    sp = SelectionPressure(
        strategy="qd", population_size=50, min_improvement_threshold=0.01, admit_threshold=0.3,
    )
    sp._validate_open_ended([_run(reward=0.0)], [_run(reward=0.97, uuid="seed")], threshold=0.01)
    assert len(sp._archive) == 1
    # Regression: lower reward and qd_score under admit_threshold → rejected.
    flat = _run(reward=0.05, uuid="flat")
    sp._validate_open_ended([_run(reward=0.97)], [flat], threshold=0.01)
    assert "flat" not in {m.uuid for m in sp._archive}


def test_capacity_eviction_keeps_highest_qd():
    sp = SelectionPressure(strategy="qd", population_size=2, novelty_weight=0.4)
    for i, score in enumerate([0.9, 0.8, 0.95]):
        sp._validate_open_ended(
            [_run(reward=0.0)], [_run(reward=score, uuid=f"u{i}")], threshold=0.01,
        )
    assert len(sp._archive) == 2
    qd_scores = sorted(m.qd_score for m in sp._archive)
    assert qd_scores == sorted(qd_scores, reverse=False)
    assert "u1" not in {m.uuid for m in sp._archive}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")
    print("All selection_test passed.")

"""Tests for SelectionPressure archive admission.

Descriptors are passed in pre-computed (``run.behaviour_descriptor``) so
the tests stay deterministic without instantiating a sentence-transformer
backend. The selection module honours that injection path for testing.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))

# Import selection (and its genotype-embedding/code-features deps) directly:
# sources.core.__init__ pulls in litellm/sentence_transformers, which aren't
# needed here and break offline test runs.
def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


_load("sources.core.genotype_embedding", _REPO_ROOT / "sources" / "core" / "genotype_embedding.py")
_load("sources.core.code_features", _REPO_ROOT / "sources" / "core" / "code_features.py")
_selection = _load("sources.core.selection", _REPO_ROOT / "sources" / "core" / "selection.py")

PopulationMember = _selection.PopulationMember
SelectionPressure = _selection.SelectionPressure
_cosine_distance = _selection._cosine_distance
_length_penalty = _selection._length_penalty

# Distinct unit vectors used as stub embeddings — sized to look like an
# embedding (>1 dim, deterministic) without loading a real model.
_DIM: int = 6


def _unit(values: list[float]) -> list[float]:
    """Return a unit-norm copy of ``values`` for stub descriptors."""
    norm = sum(v * v for v in values) ** 0.5
    if norm == 0.0:
        return values
    return [v / norm for v in values]


def _run(
    descriptor: list[float] | None = None,
    reward: float = 0.9,
    uuid: str = "u",
    code_len: int = 100,
) -> SimpleNamespace:
    """Build a minimal IndividualRun-like stub with a pre-baked descriptor."""
    return SimpleNamespace(
        reward=reward,
        reward_uncapped=reward,
        current_uuid=uuid,
        iteration_count=1,
        cost=0.0,
        code="x" * code_len,
        behaviour_descriptor=_unit(descriptor) if descriptor else [0.0] * _DIM,
    )


def test_first_run_admitted():
    sp = SelectionPressure(config={}, strategy="qd", population_size=50)
    sp._validate_open_ended(
        [_run(reward=0.0)],
        [_run(reward=0.97, uuid="seed", descriptor=[0.3, -0.2, 0.0, 0.0, -0.1, 0.0])],
        threshold=0.01,
    )
    assert [m.uuid for m in sp._archive] == ["seed"]


def test_distinct_genotype_embedding_sibling_is_admitted():
    """A lower-reward sibling whose code embedding is far from the seed lands.

    Quality-only selection would Pareto-reject it; the cosine-distance
    novelty signal compensates.
    """
    sp = SelectionPressure(config={}, strategy="qd", population_size=50, novelty_k_neighbours=25)
    seed = PopulationMember(
        iteration=1, reward=0.97, cost=0.0, uuid="seed", reward_uncapped=1.05,
        behaviour_descriptor=_unit([0.3, -0.2, 0.0, 0.0, -0.1, 0.0]),
        genotype_chars=100,
    )
    sp._archive = [seed]

    distinct = _run(
        reward=0.91,
        uuid="distinct",
        descriptor=[-0.2, 0.1, 0.0, -0.4, 0.5, 0.0],
    )

    sp._validate_open_ended([seed], [distinct], threshold=0.01)
    assert {m.uuid for m in sp._archive} == {"seed", "distinct"}


def test_invalid_candidate_rejected():
    sp = SelectionPressure(
        config={}, strategy="qd", population_size=50, min_improvement_threshold=0.01, admit_threshold=0.3,
    )
    sp._validate_open_ended([_run(reward=0.0)], [_run(reward=0.97, uuid="seed")], threshold=0.01)
    assert len(sp._archive) == 1
    # Regression: lower reward and qd_score under admit_threshold → rejected.
    flat = _run(reward=0.05, uuid="flat")
    sp._validate_open_ended([_run(reward=0.97)], [flat], threshold=0.01)
    assert "flat" not in {m.uuid for m in sp._archive}


def test_capacity_eviction_keeps_highest_qd():
    sp = SelectionPressure(config={}, strategy="qd", population_size=2, novelty_weight=0.4)
    for i, score in enumerate([0.9, 0.8, 0.95]):
        sp._validate_open_ended(
            [_run(reward=0.0)], [_run(reward=score, uuid=f"u{i}")], threshold=0.01,
        )
    assert len(sp._archive) == 2
    qd_scores = sorted(m.qd_score for m in sp._archive)
    assert qd_scores == sorted(qd_scores, reverse=False)
    assert "u1" not in {m.uuid for m in sp._archive}


def test_length_penalty_reduces_qd_for_bloated_genotype():
    """A genotype far above baseline pays a penalty in qd_score."""
    sp = SelectionPressure(
        config={}, strategy="qd", population_size=50, novelty_weight=0.4,
        length_penalty_baseline_chars=1000, length_penalty_lambda=0.1,
    )
    descriptor = [0.7, 0.7, 0.0, 0.0, 0.0, 0.0]
    short = _run(reward=0.9, uuid="short", descriptor=descriptor, code_len=500)
    long = _run(reward=0.9, uuid="long", descriptor=descriptor, code_len=10000)
    short_result = sp._validate_open_ended([short], [short], threshold=0.01)
    sp_long = SelectionPressure(
        config={}, strategy="qd", population_size=50, novelty_weight=0.4,
        length_penalty_baseline_chars=1000, length_penalty_lambda=0.1,
    )
    long_result = sp_long._validate_open_ended([long], [long], threshold=0.01)
    assert short_result["length_penalty"] == 0.0
    assert long_result["length_penalty"] == 1.0
    assert long_result["qd_score"] < short_result["qd_score"]


def test_previous_n_mode_uses_recent_window():
    """``previous_n`` populates the sliding buffer instead of touching the archive."""
    sp = SelectionPressure(
        config={}, strategy="qd", population_size=50, novelty_weight=0.4,
        novelty_comparison="previous_n", previous_n=3,
    )
    a = _run(reward=0.5, uuid="a", descriptor=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    b = _run(reward=0.5, uuid="b", descriptor=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    c = _run(reward=0.5, uuid="c", descriptor=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    d = _run(reward=0.5, uuid="d", descriptor=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    for r in (a, b, c, d):
        sp._validate_open_ended([r], [r], threshold=0.01)
    # Buffer caps at previous_n=3; the oldest descriptor falls out.
    assert len(sp._previous_descriptors) == 3


def test_cosine_distance_helpers():
    """Sanity checks on the new vector helpers."""
    assert _cosine_distance([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert _cosine_distance([1.0, 0.0], [0.0, 1.0]) == 1.0
    assert abs(_cosine_distance([1.0, 0.0], [-1.0, 0.0]) - 2.0) < 1e-9
    assert _cosine_distance([], [1.0]) == 1.0
    assert _length_penalty(500, 1000) == 0.0
    assert _length_penalty(1000, 1000) == 0.0
    assert _length_penalty(2000, 1000) == 1.0
    assert _length_penalty(1500, 1000) == 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ✓ {name}")
    print("All selection_test passed.")

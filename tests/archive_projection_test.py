"""Tests for the QD-archive PCA projection renderer.

Descriptors are stubbed (no embedding backend); the module is loaded
directly so ``sources.core.__init__`` (litellm, sentence-transformers)
never runs.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


_ap = _load(
    "archive_projection",
    _REPO_ROOT / "sources" / "utils" / "archive_projection.py",
)


def _member(uuid: str, descriptor: list[float], reward: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(uuid=uuid, behaviour_descriptor=descriptor, reward=reward)


def _cloud(n: int = 6, dim: int = 5) -> list[SimpleNamespace]:
    """Deterministic spread-out members: seeded gaussian, distinct rows."""
    rng = np.random.default_rng(42)
    return [
        _member(f"uuid-{i:04d}", rng.normal(size=dim).tolist(), reward=i / n)
        for i in range(n)
    ]


def test_renders_png(tmp_path):
    out = _ap.render_archive_pca(_cloud(), tmp_path)
    assert out == tmp_path / "archive_pca.png"
    assert out.exists() and out.stat().st_size > 1_000


def test_output_path_override_and_highlight(tmp_path):
    target = tmp_path / "nested" / "custom.png"
    out = _ap.render_archive_pca(
        _cloud(), tmp_path, output_path=target, highlight_uuid="uuid-0002",
    )
    assert out == target and target.exists()


def test_unknown_highlight_uuid_is_ignored(tmp_path):
    out = _ap.render_archive_pca(_cloud(), tmp_path, highlight_uuid="absent")
    assert out is not None and out.exists()


def test_too_few_members_returns_none(tmp_path):
    members = _cloud()[: _ap.MIN_MEMBERS - 1]
    assert _ap.render_archive_pca(members, tmp_path) is None
    assert not (tmp_path / "archive_pca.png").exists()


def test_empty_descriptors_do_not_count(tmp_path):
    members = _cloud(_ap.MIN_MEMBERS - 1) + [
        _member("no-desc-1", []),
        _member("no-desc-2", []),
    ]
    assert _ap.render_archive_pca(members, tmp_path) is None


def test_mixed_dims_keeps_majority(tmp_path):
    members = _cloud(5, dim=4) + [_member("odd-dim", [1.0, 2.0])]
    matrix, kept = _ap._descriptor_matrix(members)
    assert matrix.shape == (5, 4)
    assert all(m.uuid != "odd-dim" for m in kept)


def test_pca_is_deterministic_and_bounded():
    vectors = np.asarray([m.behaviour_descriptor for m in _cloud(8, dim=6)])
    coords_a, explained_a = _ap._pca_2d(vectors)
    coords_b, explained_b = _ap._pca_2d(vectors)
    assert np.allclose(coords_a, coords_b)
    assert np.allclose(explained_a, explained_b)
    assert coords_a.shape == (8, 2)
    assert 0.0 < explained_a.sum() <= 1.0 + 1e-9


def test_pca_zero_variance_is_all_zeros():
    vectors = np.ones((4, 3))
    coords, explained = _ap._pca_2d(vectors)
    assert np.allclose(coords, 0.0) and np.allclose(explained, 0.0)

"""Tests for ``lineage.find_oldest_rubric_anchor``.

These tests build synthetic ``lineage_<uuid>.json`` files and verifier-cache
folders on disk and confirm the walker picks the topmost ancestor with a
cache, gracefully handles broken chains, and respects the ``max_depth``
cycle guard.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.append(str(_REPO_ROOT))

# Import lineage directly; sources.core.__init__ pulls in heavy deps
# (sentence_transformers, etc.) that aren't needed here.
_spec = importlib.util.spec_from_file_location(
    "lineage_mod", _REPO_ROOT / "sources" / "core" / "lineage.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
find_oldest_rubric_anchor = _mod.find_oldest_rubric_anchor


def _write_lineage(workflow_dir: Path, uuid: str, parents: list[str]) -> None:
    """Create ``workflow_dir/<uuid>/lineage_<uuid>.json`` with given parents."""
    folder = workflow_dir / uuid
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"lineage_{uuid}.json").write_text(
        json.dumps({
            "uuid": uuid,
            "parents": parents,
            "evolution_kind": "mutation",
            "iteration": 0,
            "created_at": "2026-06-02T12:00:00",
            "goal_snippet": "",
        }),
        encoding="utf-8",
    )


def _write_cache(tmp_root: Path, uuid: str) -> None:
    """Create an empty ``claims.json`` cache marker for *uuid*."""
    folder = tmp_root / uuid
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "claims.json").write_text("{}", encoding="utf-8")


def test_no_parents_returns_none(tmp_path: Path) -> None:
    """A root run with no parents has nothing to anchor on."""
    wf = tmp_path / "wf"
    tmp = tmp_path / "tmp"
    _write_lineage(wf, "root", parents=[])
    assert find_oldest_rubric_anchor(wf, tmp, "root") is None


def test_parent_with_cache_returned(tmp_path: Path) -> None:
    """Single-step ancestor with a cache is returned."""
    wf = tmp_path / "wf"
    tmp = tmp_path / "tmp"
    _write_lineage(wf, "parent", parents=[])
    _write_lineage(wf, "child", parents=["parent"])
    _write_cache(tmp, "parent")
    assert find_oldest_rubric_anchor(wf, tmp, "child") == "parent"


def test_topmost_ancestor_with_cache_wins(tmp_path: Path) -> None:
    """When two ancestors have caches the EARLIEST (topmost) wins.

    This is what makes scoring lineage-wide stable instead of drifting by
    one generation each step.
    """
    wf = tmp_path / "wf"
    tmp = tmp_path / "tmp"
    _write_lineage(wf, "g1", parents=[])
    _write_lineage(wf, "g2", parents=["g1"])
    _write_lineage(wf, "g3", parents=["g2"])
    _write_lineage(wf, "g4", parents=["g3"])
    _write_cache(tmp, "g1")
    _write_cache(tmp, "g3")
    assert find_oldest_rubric_anchor(wf, tmp, "g4") == "g1"


def test_no_ancestor_has_cache_returns_none(tmp_path: Path) -> None:
    """When no ancestor has a cache, the LLM path is signalled (None)."""
    wf = tmp_path / "wf"
    tmp = tmp_path / "tmp"
    _write_lineage(wf, "g1", parents=[])
    _write_lineage(wf, "g2", parents=["g1"])
    assert find_oldest_rubric_anchor(wf, tmp, "g2") is None


def test_self_cache_ignored(tmp_path: Path) -> None:
    """A cache on the uuid itself is NOT used — only ancestors anchor."""
    wf = tmp_path / "wf"
    tmp = tmp_path / "tmp"
    _write_lineage(wf, "g1", parents=[])
    _write_lineage(wf, "g2", parents=["g1"])
    _write_cache(tmp, "g2")
    assert find_oldest_rubric_anchor(wf, tmp, "g2") is None


def test_missing_parent_lineage_breaks_chain(tmp_path: Path) -> None:
    """If a parent's lineage file is missing the walk stops there.

    This matches the production case where an early ancestor's folder was
    discarded — the walker must not crash and must return whatever cached
    ancestor it managed to reach.
    """
    wf = tmp_path / "wf"
    tmp = tmp_path / "tmp"
    _write_lineage(wf, "g3", parents=["g2"])  # g2 lineage missing
    _write_cache(tmp, "g2")
    # g2's lineage is missing → walker reaches g2 once via g3's parents,
    # records its cache, then load_lineage(g2) returns None → break.
    assert find_oldest_rubric_anchor(wf, tmp, "g3") == "g2"


def test_cycle_does_not_hang(tmp_path: Path) -> None:
    """Pathological self-cycle terminates without exhausting max_depth."""
    wf = tmp_path / "wf"
    tmp = tmp_path / "tmp"
    _write_lineage(wf, "a", parents=["a"])
    # No cache anywhere — must return None promptly.
    assert find_oldest_rubric_anchor(wf, tmp, "a", max_depth=4) is None


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_topmost_ancestor_with_cache_wins(Path(d))
    print("lineage_anchor_test: smoke ok")

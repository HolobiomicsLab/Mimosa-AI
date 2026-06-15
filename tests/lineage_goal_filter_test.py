"""Tests for ``lineage.scan_all`` goal filtering.

Builds workflow folders carrying ``goal_<uuid>.txt`` + ``lineage_<uuid>.json``
sidecars and confirms ``scan_all(goal=...)`` returns only the workflows whose
goal text matches — the mechanism that keeps each run's evolution tree from
mixing with workflows produced for other goals.
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
scan_all = _mod.scan_all


def _write_workflow(workflow_dir: Path, uuid: str, goal: str | None) -> None:
    """Create ``<uuid>/`` with a lineage sidecar and optional ``goal_<uuid>.txt``."""
    folder = workflow_dir / uuid
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"lineage_{uuid}.json").write_text(
        json.dumps({
            "uuid": uuid,
            "parents": [],
            "evolution_kind": "seed",
            "iteration": 0,
            "created_at": "2026-06-13T12:00:00",
            "goal_snippet": "",
        }),
        encoding="utf-8",
    )
    if goal is not None:
        (folder / f"goal_{uuid}.txt").write_text(goal, encoding="utf-8")


def test_no_goal_returns_all(tmp_path: Path) -> None:
    """Without a goal filter every workflow folder is returned (backward compat)."""
    _write_workflow(tmp_path, "a", goal="Goal A")
    _write_workflow(tmp_path, "b", goal="Goal B")
    assert set(scan_all(tmp_path)) == {"a", "b"}


def test_goal_filter_selects_matching_only(tmp_path: Path) -> None:
    """Only workflows whose goal_<uuid>.txt matches the goal are returned."""
    _write_workflow(tmp_path, "a1", goal="Goal A")
    _write_workflow(tmp_path, "a2", goal="Goal A")
    _write_workflow(tmp_path, "b1", goal="Goal B")
    assert set(scan_all(tmp_path, goal="Goal A")) == {"a1", "a2"}
    assert set(scan_all(tmp_path, goal="Goal B")) == {"b1"}


def test_goal_filter_is_whitespace_tolerant(tmp_path: Path) -> None:
    """Trailing/leading whitespace in either side does not break the match."""
    _write_workflow(tmp_path, "a", goal="Goal A\n")
    assert set(scan_all(tmp_path, goal="  Goal A  ")) == {"a"}


def test_missing_goal_file_excluded_when_filtering(tmp_path: Path) -> None:
    """A folder without goal_<uuid>.txt is excluded once a goal filter is set."""
    _write_workflow(tmp_path, "has_goal", goal="Goal A")
    _write_workflow(tmp_path, "no_goal", goal=None)
    assert set(scan_all(tmp_path, goal="Goal A")) == {"has_goal"}
    # ...but still appears with no filter.
    assert set(scan_all(tmp_path)) == {"has_goal", "no_goal"}


def test_nonexistent_dir_returns_empty(tmp_path: Path) -> None:
    """Scanning a missing directory yields an empty mapping regardless of goal."""
    assert scan_all(tmp_path / "missing", goal="Goal A") == {}


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_goal_filter_selects_matching_only(Path(d))
    print("lineage_goal_filter_test: smoke ok")

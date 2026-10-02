"""Reconstruct evolution families and their metric series from disk.

A run's lineage is not a single file: each evolution iteration is its own
``<uuid>/`` dir carrying a ``lineage_<uuid>.json`` with ``parents``. The family
(tree) a run belongs to is the connected component of the parent graph over all
run dirs — more robust than matching goal strings, and it handles crossovers
(a node with two parents) correctly.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from . import store
from .settings import get_settings


def _index() -> dict[str, dict[str, Any]]:
    """uuid -> lineage record for every run dir (synthesises seeds for legacy)."""
    idx: dict[str, dict[str, Any]] = {}
    for run_id in store.list_run_ids():
        rec = store.read_lineage(run_id) or {}
        idx[run_id] = {
            "uuid": run_id,
            "parents": [p for p in rec.get("parents", []) if isinstance(p, str)],
            "iteration": rec.get("iteration"),
            "evolution_kind": rec.get("evolution_kind", "seed"),
            "created_at": rec.get("created_at") or store.parse_created_at(run_id),
            "goal_snippet": rec.get("goal_snippet"),
        }
    return idx


def _component(root: str, idx: dict[str, dict[str, Any]]) -> set[str]:
    """Undirected connected component containing *root* over parent edges."""
    children: dict[str, list[str]] = {u: [] for u in idx}
    for uuid, rec in idx.items():
        for parent in rec["parents"]:
            children.setdefault(parent, []).append(uuid)
    seen: set[str] = set()
    queue: deque[str] = deque([root])
    while queue:
        node = queue.popleft()
        if node in seen or node not in idx:
            continue
        seen.add(node)
        queue.extend(idx[node]["parents"])
        queue.extend(children.get(node, []))
    return seen


def tree(run_id: str) -> dict[str, Any] | None:
    """Nodes + directed parent→child edges for *run_id*'s evolution family."""
    idx = _index()
    if run_id not in idx:
        return None
    members = _component(run_id, idx)
    nodes = []
    for uuid in sorted(members):
        rec = idx[uuid]
        nodes.append(
            {
                "id": uuid,
                "iteration": rec["iteration"],
                "evolution_kind": rec["evolution_kind"],
                "created_at": rec["created_at"],
                "score": store.overall_score(uuid),
                "status": store.run_status(uuid),
                "is_focus": uuid == run_id,
            }
        )
    edges = [
        {"source": parent, "target": uuid, "kind": idx[uuid]["evolution_kind"]}
        for uuid in members
        for parent in idx[uuid]["parents"]
        if parent in members
    ]
    return {"focus": run_id, "nodes": nodes, "edges": edges}


def family_size(run_id: str) -> int:
    """Number of runs in *run_id*'s evolution family (1 = a lone seed)."""
    idx = _index()
    if run_id not in idx:
        return 0
    return len(_component(run_id, idx))


def series(run_id: str) -> dict[str, Any]:
    """Reward/cost/novelty series across the family, ordered by iteration.

    Returns one point per family member that has ``run_metrics.json``. Points
    are sorted by iteration (falling back to created_at) so the frontend can
    draw a monotonic reward/cost curve without the end-only PNG.
    """
    idx = _index()
    members = _component(run_id, idx) if run_id in idx else {run_id}
    points = []
    for uuid in members:
        metrics = store.read_run_metrics(uuid)
        if not metrics:
            continue
        points.append(
            {
                "uuid": uuid,
                "iteration": metrics.get("iteration"),
                "evolution_kind": metrics.get("evolution_kind"),
                "overall_score": metrics.get("overall_score"),
                "overall_score_uncapped": metrics.get("overall_score_uncapped"),
                "qd_score": metrics.get("qd_score"),
                "novelty_score": metrics.get("novelty_score"),
                "cumulative_cost_usd": metrics.get("cumulative_cost_usd"),
                "iteration_cost_usd": metrics.get("iteration_cost_usd"),
                "wall_time_s": metrics.get("iteration_wall_time_s"),
                "on_error": metrics.get("on_error"),
            }
        )
    points.sort(key=lambda p: (p["iteration"] if p["iteration"] is not None else 1e9))
    return {"focus": run_id, "points": points}


def qd_archive(limit: int | None = None) -> list[dict[str, Any]]:
    """Parse the append-only shared QD archive (drops the 384-float vector)."""
    path: Path = get_settings().workflow_dir / "qd_archive.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entry.pop("qd_descriptor", None)
                rows.append(entry)
    except OSError:
        return []
    if limit is not None:
        rows = rows[-limit:]
    return rows

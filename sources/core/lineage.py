"""
Lineage tracking for evolved workflows.

Each workflow folder gets a ``lineage_<uuid>.json`` sidecar recording which
parent(s) produced it and via which variation operator. The visualizer in
:mod:`sources.utils.evolution_tree` reads these files to draw the evolution
tree across an entire run / a project's workflow directory.

Format::

    {
      "uuid":           "20260512_162504_70ccefbf",
      "parents":        ["20260512_161200_aabbcc11"],
      "evolution_kind": "seed" | "mutation" | "crossover",
      "iteration":      0,
      "created_at":     "2026-05-25T15:59:00",
      "goal_snippet":   "Train a multitask model..."
    }
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

LINEAGE_FILE_TEMPLATE = "lineage_{uuid}.json"
VALID_KINDS = {"seed", "mutation", "crossover"}


def record_lineage(
    workflow_dir: str | Path,
    uuid: str,
    parents: list[str] | None,
    kind: str,
    iteration: int = 0,
    goal: str | None = None,
) -> Path | None:
    """Persist a lineage record for a newly created workflow.

    Args:
        workflow_dir: Project's workflow directory (parent of ``<uuid>/``).
        uuid: Child workflow UUID.
        parents: List of parent UUIDs (empty/None for seed).
        kind: One of ``seed``, ``mutation``, ``crossover``.
        iteration: Evolution-loop iteration that produced this workflow.
        goal: Original task / goal text (truncated for storage).

    Returns:
        Path to the written file, or None if the target folder doesn't exist.
    """
    if not uuid:
        return None
    if kind not in VALID_KINDS:
        logger.warning(f"lineage: unknown evolution_kind={kind!r}, coercing to 'seed'")
        kind = "seed"

    folder = Path(workflow_dir) / uuid
    if not folder.is_dir():
        logger.debug(f"lineage: folder {folder} missing — skip recording")
        return None

    record = {
        "uuid": uuid,
        "parents": [p for p in (parents or []) if p],
        "evolution_kind": kind,
        "iteration": int(iteration),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "goal_snippet": (goal or "").strip()[:240],
    }

    target = folder / LINEAGE_FILE_TEMPLATE.format(uuid=uuid)
    try:
        with open(target, "w") as f:
            json.dump(record, f, indent=2)
        logger.info(f"lineage: recorded {kind} {uuid} ← {record['parents']}")
        return target
    except OSError as e:
        logger.error(f"lineage: failed to write {target}: {e}")
        return None


def load_lineage(workflow_dir: str | Path, uuid: str) -> dict | None:
    """Read a lineage record. Returns None if missing or unreadable."""
    target = Path(workflow_dir) / uuid / LINEAGE_FILE_TEMPLATE.format(uuid=uuid)
    if not target.exists():
        return None
    try:
        with open(target) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"lineage: could not read {target}: {e}")
        return None


def scan_all(workflow_dir: str | Path) -> dict[str, dict]:
    """Enumerate every workflow folder and return ``{uuid: lineage_or_synthetic}``.

    Workflows without a ``lineage_*.json`` are synthesised as orphan seeds so
    the tree visualizer can still place them. This makes the tooling
    backward-compatible with workflows generated before lineage tracking
    landed.
    """
    root = Path(workflow_dir)
    if not root.is_dir():
        return {}

    records: dict[str, dict] = {}
    for entry in sorted(os.listdir(root)):
        folder = root / entry
        if not folder.is_dir() or entry.startswith("_") or entry.startswith("."):
            continue
        # Skip caches like _task_checklists
        rec = load_lineage(root, entry)
        if rec is None:
            # Synthesize a minimal record so legacy workflows still appear.
            if not (folder / "state_result.json").exists() and not (
                folder / f"workflow_genotype_{entry}.py"
            ).exists():
                continue
            rec = {
                "uuid": entry,
                "parents": [],
                "evolution_kind": "seed",
                "iteration": 0,
                "created_at": "",
                "goal_snippet": "",
                "_synthetic": True,
            }
        records[entry] = rec
    return records

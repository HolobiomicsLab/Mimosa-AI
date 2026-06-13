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
    """Read a lineage record from disk.

    Args:
        workflow_dir: Project's workflow directory (parent of ``<uuid>/``).
        uuid: Workflow UUID whose lineage record should be read.

    Returns:
        The parsed lineage dictionary, or ``None`` when the file is missing,
        unreadable, or not a JSON object.
    """
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


def find_oldest_rubric_anchor(
    workflow_dir: str | Path,
    verifier_tmp_dir: str | Path,
    uuid: str,
    anchor_filename: str = "claims.json",
    max_depth: int = 64,
) -> str | None:
    """Walk parents and return the earliest ancestor whose verifier cache exists.

    Used to anchor every descendant in an evolved lineage on the SAME rubric
    set so verifier scores stay comparable across generations. The walk
    follows ``parents[0]`` upward and only records ancestors whose
    ``<verifier_tmp_dir>/<id>/<anchor_filename>`` is present on disk.

    Args:
        workflow_dir: Project's workflow directory (parent of ``<uuid>/``).
        verifier_tmp_dir: Root of the verifier scratch tree (typically
            ``<workflow_dir>/_verifier_tmp``).
        uuid: Workflow whose ancestors are walked. The uuid itself is NOT
            considered — only ancestors.
        anchor_filename: Cache filename to check inside each ancestor's folder.
        max_depth: Safety bound on chain depth (cycle / runaway guard).

    Returns:
        UUID of the earliest ancestor with a cache, or ``None`` when no such
        ancestor exists or the chain breaks (missing lineage, cycle).
    """
    tmp_root = Path(verifier_tmp_dir)
    candidate: str | None = None
    seen: set[str] = set()
    current = uuid
    for _ in range(max_depth):
        if current in seen:
            break
        seen.add(current)
        rec = load_lineage(workflow_dir, current)
        if rec is None:
            break
        parents = rec.get("parents") or []
        if not parents:
            break
        parent = parents[0]
        if (tmp_root / parent / anchor_filename).exists():
            candidate = parent
        current = parent
    return candidate


def _read_workflow_goal(folder: Path, uuid: str) -> str | None:
    """Return the trimmed contents of ``goal_<uuid>.txt``, or None if absent/unreadable."""
    target = folder / f"goal_{uuid}.txt"
    try:
        return target.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def scan_all(
    workflow_dir: str | Path, goal: str | None = None
) -> dict[str, dict]:
    """Enumerate every workflow folder and return ``{uuid: lineage_or_synthetic}``.

    Workflows without a ``lineage_*.json`` are synthesised as orphan seeds so
    the tree visualizer can still place them. This makes the tooling
    backward-compatible with workflows generated before lineage tracking
    landed.

    Args:
        workflow_dir: Project's workflow directory to scan.
        goal: When provided, only workflows whose ``goal_<uuid>.txt`` matches
            this text (after trimming) are returned. Folders missing the goal
            file are excluded. This keeps a single run's tree from mixing with
            other goals' workflows. ``None`` scans everything.

    Returns:
        Dictionary mapping each workflow UUID to its lineage record. Records
        synthesised for legacy workflows include ``"_synthetic": True``.
        Returns an empty dict when ``workflow_dir`` does not exist.
    """
    root = Path(workflow_dir)
    if not root.is_dir():
        return {}

    goal_filter = goal.strip() if goal is not None else None
    records: dict[str, dict] = {}
    for entry in sorted(os.listdir(root)):
        folder = root / entry
        if not folder.is_dir() or entry.startswith("_") or entry.startswith("."):
            continue
        if goal_filter is not None and _read_workflow_goal(folder, entry) != goal_filter:
            continue
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

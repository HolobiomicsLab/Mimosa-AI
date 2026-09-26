"""Workspace inventory with cheap content signals (E19 Phase-A input).

Ports the measured conventions of ``experiments_verifiers/harness/e19_hybrid_bt.py``
(``union_inventory`` / ``deliverable_previews``) and the content signals of
``e14_bt_enabler.file_signal``: every workspace file is listed with its size
plus, when cheap to compute, a CSV shape (rows x cols) or PNG dimensions.
The union inventory accumulates across generations of the same task in the
per-task registry so claim extraction sees which paths are stable and which
are candidate-specific.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Budgets ported from E19 (UNION_LINE_CAP / PREVIEW_TOTAL_CAP).
UNION_LINE_CAP = 300
PREVIEW_TOTAL_CAP = 14 * 1024
PREVIEW_MAX_FILES = 3
_PREVIEW_HEAD_LINES = 11
_TEXT_PREVIEW_BYTES = 6 * 1024
_TEXT_PREVIEW_TAIL = 1024
_CSV_SNIFF_BYTES = 64 * 1024
_CSV_MAX_BYTES = 8 * 1024 * 1024
_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache"}
_SCRIPT_EXTS = {".py", ".R", ".r", ".sh", ".jl"}


def _read_head(path: Path, cap: int) -> str:
    """First *cap* bytes decoded as UTF-8 (lossy), ``""`` on error."""
    try:
        with open(path, "rb") as fh:
            return fh.read(cap).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _count_newlines(path: Path, cap_bytes: int = 64 * 1024 * 1024) -> int:
    """Cheap row-count proxy for text files (capped read)."""
    n = 0
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(4 * 1024 * 1024)
                if not chunk:
                    break
                n += chunk.count(b"\n")
                if fh.tell() > cap_bytes:
                    break
    except OSError:
        return -1
    return n


def csv_shape(path: Path, size: int) -> tuple[int, int] | None:
    """(n_rows, n_cols) for CSV/TSV files up to 8 MB via header sniffing."""
    if size > _CSV_MAX_BYTES:
        return None
    head = _read_head(path, _CSV_SNIFF_BYTES).splitlines()
    if not head or not head[0]:
        return None
    delim = max([",", ";", "\t"], key=lambda d: head[0].count(d))
    n_cols = head[0].count(delim) + 1
    return max(0, _count_newlines(path)), n_cols


def png_dims(path: Path) -> tuple[int, int] | None:
    """PNG dimensions from the IHDR header, without loading the image."""
    try:
        with open(path, "rb") as fh:
            header = fh.read(26)
        if len(header) >= 24 and header[12:16] == b"IHDR":
            w, h = struct.unpack(">II", header[16:24])
            return int(w), int(h)
    except (OSError, struct.error):
        pass
    return None


def file_signal(path: Path, size: int, ext: str) -> str:
    """Cheap content signal for one inventory line ("" when none)."""
    if ext == ".png":
        dims = png_dims(path)
        return f"{dims[0]}x{dims[1]} px" if dims else ""
    if ext in {".csv", ".tsv"}:
        shape = csv_shape(path, size)
        return f"{shape[0]} rows x {shape[1]} cols" if shape else ""
    return ""


def scan_workspace(
    workspace: Path, max_entries: int = 400
) -> dict[str, dict[str, Any]]:
    """Recursive inventory of *workspace*: rel path -> size/kind/signal.

    Args:
        workspace: Workspace root directory (missing directories yield {}).
        max_entries: Hard cap on scanned files; the largest signal-bearing
            files are kept when the cap truncates.

    Returns:
        ``{rel_path: {"bytes": int, "kind": str, "signal": str}}`` where
        *kind* is one of ``csv``/``tsv``/``png``/``script``/``text``/``other``.
    """
    out: dict[str, dict[str, Any]] = {}
    if not workspace.exists() or not workspace.is_dir():
        return out
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            rel = str(Path(root, name).relative_to(workspace))
            fp = workspace / rel
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in {".csv", ".tsv"}:
                kind = ext.lstrip(".")
            elif ext in {".png"}:
                kind = "png"
            elif ext in _SCRIPT_EXTS:
                kind = "script"
            elif ext in {".txt", ".log", ".json", ".md", ".tsv"} or ext == "":
                kind = "text"
            else:
                kind = "other"
            out[rel] = {
                "bytes": size,
                "kind": kind,
                "signal": file_signal(fp, size, ext)
                if kind in {"csv", "tsv", "png"}
                else "",
            }
            if len(out) >= max_entries:
                return out
    return out


def workspace_has_artifacts(inventory: dict[str, dict[str, Any]]) -> bool:
    """True when the inventory lists at least one non-script artifact."""
    return any(v.get("kind") != "script" for v in inventory.values())

# E26/E35 figure-task detection: a task's deliverables are figures when at
# least this share of its deliverable files are images.
FIGURE_IMAGE_SHARE = 0.5


def is_figure_task(inventory: dict[str, dict[str, Any]]) -> bool:
    """True when >=50% of the deliverable files are images (E26 detection).

    Production adaptation of E26's image-bearing-generation rule: the pool
    is the union inventory's deliverable files (everything but scripts),
    narrowed to results directories (``pred_results`` / ``results``) when
    the task has any — E26 scored the ``pred_results`` figures. A task
    whose figures are the majority deliverable gets the visual rung
    ordered FIRST (E26: visual-early 0.833 vs 0.646 as 4th rung).
    """
    deliverables = {
        p: info for p, info in inventory.items() if info.get("kind") != "script"
    }
    if not deliverables:
        return False
    in_results = {
        p: info
        for p, info in deliverables.items()
        if "pred_results" in p or "results" in Path(p).parent.as_posix().lower()
    }
    pool = in_results or deliverables
    images = sum(1 for info in pool.values() if info.get("kind") == "png")
    return images / len(pool) >= FIGURE_IMAGE_SHARE


def _deliverable_rank(rel: str, info: dict[str, Any]) -> int:
    """E19 deliverable-preference order (0 = most deliverable-ish)."""
    ext = os.path.splitext(rel)[1].lower()
    in_pr = "pred_results" in rel or "results" in rel.lower()
    if in_pr and ext in {".csv", ".tsv"}:
        return 0
    if in_pr and ext == ".json":
        return 1
    if in_pr and ext in {".txt", ".log"}:
        return 2
    if "/" not in rel and info.get("kind") == "script":
        return 3
    return 9


def deliverable_previews(
    inventory: dict[str, dict[str, Any]],
    workspace: Path,
    max_files: int = PREVIEW_MAX_FILES,
    budget: int = PREVIEW_TOTAL_CAP,
) -> str:
    """Render up to *max_files* deliverable previews under *budget* bytes.

    Ports E19's ``deliverable_previews``: CSV/TSV get the header plus first
    ten rows and a total row count; JSON gets the first 2 KiB; text/log
    files get a head+tail slice; everything else is skipped.
    """
    cands = sorted(
        (
            (rel, info)
            for rel, info in inventory.items()
            if _deliverable_rank(rel, info) < 9
        ),
        key=lambda kv: (_deliverable_rank(kv[0], kv[1]), -kv[1].get("bytes", 0)),
    )[:max_files]
    blocks: list[str] = []
    remaining = budget
    for rel, info in cands:
        fp = workspace / rel
        ext = os.path.splitext(rel)[1].lower()
        if ext in {".csv", ".tsv"}:
            head = _read_head(fp, _CSV_SNIFF_BYTES).splitlines()
            body = "\n".join(head[:_PREVIEW_HEAD_LINES])
            body += f"\n[{_count_newlines(fp)} rows total]"
        elif ext == ".json":
            body = _read_head(fp, 2048) + f"\n[{info.get('bytes', 0)} B total]"
        else:
            size = info.get("bytes", 0)
            want = _TEXT_PREVIEW_BYTES
            if size > want + _TEXT_PREVIEW_TAIL:
                try:
                    with open(fp, "rb") as fh:
                        body = fh.read(want).decode("utf-8", errors="replace")
                        body += "\n...[middle truncated]...\n"
                        fh.seek(-_TEXT_PREVIEW_TAIL, os.SEEK_END)
                        body += fh.read().decode("utf-8", errors="replace")
                except OSError:
                    continue
            else:
                body = _read_head(fp, want)
        if len(body) > remaining:
            body = body[:remaining]
        remaining -= len(body)
        sig = f"; {info['signal']}" if info.get("signal") else ""
        blocks.append(f"--- preview: {rel} ({info.get('bytes', 0)} B{sig}) ---\n{body}")
        if remaining <= 0:
            break
    return "\n".join(blocks)


def render_union_inventory(
    union: dict[str, dict[str, Any]],
    n_workspaces: int,
    current: Iterable[str] = (),
) -> str:
    """Render the union inventory for extraction/scorer prompts.

    Args:
        union: Merged per-task inventory from the registry
            (``{rel: {"count": int, "signal": str}}``).
        n_workspaces: Number of workspaces seen for this task.
        current: Paths present in the workspace being scored now; those
            get an explicit marker so the model can tell the live layout
            from historical paths.

    Returns:
        Newline-joined listing sorted by presence count (desc), capped at
        ``UNION_LINE_CAP`` lines.
    """
    n = max(1, n_workspaces)
    cur = set(current)
    lines = []
    for rel, info in sorted(
        union.items(), key=lambda kv: (-kv[1].get("count", 0), kv[0])
    ):
        sig = f"; {info['signal']}" if info.get("signal") else ""
        star = " (present in the workspace being scored)" if rel in cur else ""
        lines.append(
            f"- {rel}{sig} — present in {info.get('count', 0)}/{n} workspaces{star}"
        )
    if len(lines) > UNION_LINE_CAP:
        lines = lines[:UNION_LINE_CAP] + [
            f"[{len(lines) - UNION_LINE_CAP} more paths truncated]"
        ]
    return "\n".join(lines) if lines else "(workspace empty)"

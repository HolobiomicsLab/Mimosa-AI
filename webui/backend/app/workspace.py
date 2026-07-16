"""Browse the toolomics workspace and per-run snapshots for the preview pane.

The live workspace is wiped/restored between runs, so the durable artifacts are
the per-run snapshots under ``/tmp/mimosa_run_<session>_<uuid>/``. Files are
ranked by a priority heuristic (figures first, then fresh/large data) so the
frontend can auto-open the most interesting one.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .settings import get_settings

_EXT_KIND = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".svg": "image", ".webp": "image",
    ".csv": "data", ".tsv": "data", ".parquet": "data",
    ".md": "report", ".txt": "report", ".rst": "report",
    ".py": "code", ".sh": "code", ".r": "code", ".ipynb": "code",
    ".json": "json", ".yaml": "json", ".yml": "json",
    ".pkl": "binary", ".pt": "binary", ".pth": "binary", ".npy": "binary",
}
_SKIP = {".DS_Store", "Thumbs.db", ".gitkeep"}
_KIND_WEIGHT = {"image": 100, "report": 60, "data": 50, "code": 40, "json": 30}


def _kind(path: Path) -> str:
    return _EXT_KIND.get(path.suffix.lower(), "other")


def _priority(kind: str, size: int, mtime: float, newest: float) -> float:
    """Higher = more interesting. Figures win; ties broken by freshness/size."""
    score = float(_KIND_WEIGHT.get(kind, 10))
    if newest > 0:
        score += 25 * (mtime / newest)  # recency, normalised to the freshest file
    if size > 50_000:
        score += min(15, size / 200_000)  # bigger artifacts skew relevant
    return round(score, 2)


def _walk(root: Path, max_files: int = 500) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    files: list[dict[str, Any]] = []
    newest = 0.0
    for path in root.rglob("*"):
        if not path.is_file() or path.name in _SKIP or path.name.startswith("."):
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        newest = max(newest, st.st_mtime)
        files.append(
            {
                "path": str(path.relative_to(root)),
                "size": st.st_size,
                "mtime": st.st_mtime,
                "kind": _kind(path),
            }
        )
        if len(files) >= max_files:
            break
    for f in files:
        f["priority"] = _priority(f["kind"], f["size"], f["mtime"], newest)
    files.sort(key=lambda f: f["priority"], reverse=True)
    return files


def _snapshot_dirs() -> dict[str, Path]:
    """Map a run uuid -> its most recent /tmp snapshot dir."""
    settings = get_settings()
    base = Path(settings.snapshot_glob).parent
    pattern = Path(settings.snapshot_glob).name
    out: dict[str, Path] = {}
    if not base.is_dir():
        return out
    for path in base.glob(pattern):
        if not path.is_dir():
            continue
        m = re.search(r"(\d{8}_\d{6}_[0-9a-f]{8})", path.name)
        if m:
            out[m.group(1)] = path
    return out


def resolve_root(scope: str) -> Path | None:
    """scope='live' -> workspace dir; scope=<uuid> -> that run's snapshot."""
    if scope == "live":
        return get_settings().workspace_dir
    return _snapshot_dirs().get(scope)


def list_scopes() -> dict[str, Any]:
    live = get_settings().workspace_dir
    snapshots = sorted(_snapshot_dirs().keys(), reverse=True)
    return {
        "live": {"available": live.is_dir(), "path": str(live)},
        "snapshots": snapshots,
    }


def list_files(scope: str) -> dict[str, Any] | None:
    root = resolve_root(scope)
    if root is None:
        return None
    files = _walk(root)
    return {
        "scope": scope,
        "root": str(root),
        "files": files,
        "auto_preview": files[0]["path"] if files else None,
    }


def resolve_file(scope: str, rel_path: str) -> tuple[Path, str] | None:
    """Safely resolve a file inside a scope root (blocks path traversal)."""
    root = resolve_root(scope)
    if root is None:
        return None
    try:
        target = (root / rel_path).resolve()
        target.relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    if not target.is_file():
        return None
    return target, _kind(target)

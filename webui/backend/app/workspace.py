"""Browse the toolomics workspace and per-run snapshots for the preview pane.

The live workspace is wiped/restored between runs, so the durable artifacts are
the per-run snapshots under ``/tmp/mimosa_run_<session>_<uuid>/``. Files are
ranked by a priority heuristic (figures first, then fresh/large data) so the
frontend can auto-open the most interesting one.

The live workspace is also the input channel for launches: files uploaded from
the New-run page land here, and Mimosa snapshots them as the run's "initial
user-provided state" at launch (see sources/utils/workspace_management.py).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, BinaryIO

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


# ── Uploads into the live workspace (run inputs) ──

_UPLOAD_MAX_BYTES = 500 * 1024 * 1024
_UPLOAD_CHUNK = 1024 * 1024


def safe_upload_name(filename: str | None) -> str | None:
    """Reduce a client-sent filename to a safe basename; None if unusable.

    Rejects empty names, dotfiles, control characters, and names past the
    255-byte filesystem component limit.
    """
    name = Path(filename or "").name.strip()
    if not name or name.startswith(".") or name in _SKIP:
        return None
    if len(name.encode("utf-8", "replace")) > 255 or any(ord(c) < 32 for c in name):
        return None
    return name


def unique_destination(root: Path, name: str) -> Path:
    """``data.csv`` -> ``data (1).csv`` … until the name is free under root.

    A symlink — even a dangling one — counts as occupied, so an upload can
    never write *through* a link left in the workspace by a previous run.
    """
    stem, suffix = Path(name).stem, Path(name).suffix
    dest = root / name
    counter = 1
    while dest.exists() or dest.is_symlink():
        dest = root / f"{stem} ({counter}){suffix}"
        counter += 1
    return dest


def _reserve_destination(root: Path, name: str) -> Path:
    """Exclusively create a free destination (no overwrite, no symlink follow)."""
    while True:
        dest = unique_destination(root, name)
        try:
            dest.open("xb").close()
            return dest
        except FileExistsError:
            continue  # lost a race for this name; try the next suffix


def save_upload(name: str, stream: BinaryIO) -> dict[str, Any]:
    """Stream one uploaded file into the live workspace.

    ``name`` must come from safe_upload_name(). The destination is created
    exclusively (collisions get a `` (n)`` suffix, symlinks are never
    followed), and content streams into a dot-prefixed temp file renamed
    into place — so a launch snapshot racing the upload can't capture a
    truncated input under its real name. Returns ``{name, size, kind}``;
    raises ValueError past _UPLOAD_MAX_BYTES (nothing left on disk).
    """
    root = get_settings().workspace_dir
    root.mkdir(parents=True, exist_ok=True)
    dest = _reserve_destination(root, name)
    part = root / f".uploading-{dest.name}"
    size = 0
    try:
        part.unlink(missing_ok=True)  # stale temp from a crashed upload
        with part.open("xb") as out:
            while chunk := stream.read(_UPLOAD_CHUNK):
                size += len(chunk)
                if size > _UPLOAD_MAX_BYTES:
                    limit_mb = _UPLOAD_MAX_BYTES // (1024 * 1024)
                    raise ValueError(f"'{name}' exceeds the {limit_mb} MB upload limit")
                out.write(chunk)
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        dest.unlink(missing_ok=True)
        raise
    return {"name": dest.name, "size": size, "kind": _kind(dest)}


def save_uploads(named: list[tuple[str, BinaryIO]]) -> list[dict[str, Any]]:
    """Save an upload batch all-or-nothing.

    On any failure the files this batch already stored are removed again, so
    an error response always means "nothing from this request persisted".
    """
    root = get_settings().workspace_dir
    stored: list[Path] = []
    entries: list[dict[str, Any]] = []
    try:
        for name, stream in named:
            entry = save_upload(name, stream)
            stored.append(root / entry["name"])
            entries.append(entry)
        return entries
    except BaseException:
        for path in stored:
            path.unlink(missing_ok=True)
        raise


def delete_live_file(rel_path: str) -> bool:
    """Delete one file inside the live workspace; True when removed."""
    resolved = resolve_file("live", rel_path)
    if resolved is None:
        return False
    try:
        resolved[0].unlink()
    except FileNotFoundError:  # raced a concurrent delete — same outcome
        return False
    return True

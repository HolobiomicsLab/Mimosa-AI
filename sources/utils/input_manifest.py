"""Transfer-time manifest of task input files, with pristine table copies.

Context: a workflow can rewrite a task input before the verifier takes its
snapshot (HIV run_oct_18: the test label column was overwritten with the true
labels). The snapshot then no longer shows what the task gave.
Caller: csv_mode records each benchmark dataset transfer; the Policy-BT
evaluator compares every new generation snapshot with that record.
Action: hash every transferred file, copy the table files aside, and later
list the snapshot files whose bytes differ from the transfer.
Limit: one manifest per workspace. Each transfer replaces the previous one,
because csv_mode cleans the workspace before every task.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

POLICY_DIR = ".policy_bt"
INPUTS_DIR = "inputs"
MANIFEST_FILE = "manifest.json"
DATA_DIR = "data"
# Delimited text tables; BBBC002 keeps its counts in .txt files.
TABLE_SUFFIXES = (".csv", ".tsv", ".txt")
# Larger tables are neither copied aside nor scanned.
MAX_TABLE_BYTES = 64 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024
WORKSPACE_KEY_CHARS = 16
STATUS_VERIFIED = "verified"
STATUS_TAMPERED = "input_tampered"
STATUS_UNVERIFIED = "unverified"


def inputs_root(workflow_dir: str | Path, workspace: str | Path) -> Path:
    """Return the folder that holds the transfer record of one workspace.

    csv_mode uses it to write the record and the evaluator to read it. Both
    get the paths from the same Config, so the record is found again.

    Args:
        workflow_dir (str | Path): Campaign workflow folder (``config.workflow_dir``).
        workspace (str | Path): Workspace that receives the dataset
            (``config.workspace_dir``).

    Returns:
        Path: ``<workflow_dir>/.policy_bt/inputs/<workspace digest>``.
    """
    resolved = str(Path(workspace).resolve()).encode("utf-8")
    key = hashlib.sha256(resolved).hexdigest()[:WORKSPACE_KEY_CHARS]
    return Path(workflow_dir) / POLICY_DIR / INPUTS_DIR / key


def record_transfer(root: str | Path, workspace: str | Path, folder: str) -> Path:
    """Replace the transfer record of a workspace with one new dataset folder.

    csv_mode calls it right after it copies a dataset folder into the
    workspace and before any workflow runs. It hashes every file of that
    folder and copies each table file of acceptable size aside.

    Args:
        root (str | Path): Record folder from ``inputs_root``.
        workspace (str | Path): Workspace that received the transfer.
        folder (str): Name of the transferred folder inside the workspace.

    Returns:
        Path: The written manifest file.
    """
    root, source = Path(root), Path(workspace) / folder
    shutil.rmtree(root, ignore_errors=True)
    files = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        key = f"{folder}/{path.relative_to(source).as_posix()}"
        files[key] = _sha256(path)
        if is_table(path):
            _copy(path, root / DATA_DIR / key)
    manifest = root / MANIFEST_FILE
    staged = manifest.with_suffix(".partial")
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text(
        json.dumps({"folder": folder, "files": files}, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    staged.replace(manifest)
    return manifest


def verify_inputs(root: str | Path, snapshot: str | Path) -> dict:
    """Compare a generation snapshot with the transfer record of its workspace.

    The evaluator calls it on each new snapshot, before re-execution can
    change any file. A record whose folder is not in the snapshot belongs to
    another task, so the result is unverified. The check only records a
    fact: an unreadable record also gives unverified, with the error.

    Args:
        root (str | Path): Record folder from ``inputs_root``.
        snapshot (str | Path): Verifier-owned copy of the workspace.

    Returns:
        dict: ``status`` (verified, input_tampered or unverified),
        ``tampered`` (sorted snapshot paths that differ or are missing),
        ``data`` (folder of pristine table copies, or None) and, when the
        record could not be read, ``error``.
    """
    try:
        return _compare(Path(root), Path(snapshot))
    except (OSError, ValueError, KeyError) as error:
        return {
            "status": STATUS_UNVERIFIED,
            "tampered": [],
            "data": None,
            "error": f"{type(error).__name__}: {error}",
        }


def is_table(path: Path) -> bool:
    """Tell whether a path is a regular table file of acceptable size.

    ``record_transfer`` copies such files aside, and the verifier's column
    scan (``placeholders``) reads only such files.

    Args:
        path (Path): Candidate path.

    Returns:
        bool: True for a table suffix, a regular file (not a link) and at
        most ``MAX_TABLE_BYTES``.
    """
    return (
        path.suffix.lower() in TABLE_SUFFIXES
        and not path.is_symlink()
        and path.is_file()
        and path.stat().st_size <= MAX_TABLE_BYTES
    )


def _compare(root: Path, snapshot: Path) -> dict:
    """Hash the manifest files in a snapshot and list those that differ.

    Called by ``verify_inputs``, which turns read errors into unverified.

    Args:
        root (Path): Record folder from ``inputs_root``.
        snapshot (Path): Verifier-owned copy of the workspace.

    Returns:
        dict: ``status``, ``tampered`` and ``data`` as in ``verify_inputs``.
    """
    manifest = _load_manifest(root)
    if manifest is None or not (snapshot / manifest["folder"]).is_dir():
        return {"status": STATUS_UNVERIFIED, "tampered": [], "data": None}
    files = sorted(manifest["files"].items())
    tampered = [
        key for key, digest in files if _digest_or_none(snapshot / key) != digest
    ]
    data = root / DATA_DIR
    return {
        "status": STATUS_TAMPERED if tampered else STATUS_VERIFIED,
        "tampered": tampered,
        "data": str(data) if data.is_dir() else None,
    }


def _copy(path: Path, target: Path) -> None:
    """Copy one file and create its parent folders.

    Called by ``record_transfer`` for each transferred table.

    Args:
        path (Path): Transferred table file.
        target (Path): Location of the pristine copy.

    Returns:
        None.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def _load_manifest(root: Path) -> dict | None:
    """Read the manifest of one workspace record.

    Called by ``_compare``; a corrupt file raises ValueError there.

    Args:
        root (Path): Record folder from ``inputs_root``.

    Returns:
        dict | None: The manifest, or None when no transfer was recorded.
    """
    path = root / MANIFEST_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _digest_or_none(path: Path) -> str | None:
    """Hash one snapshot file, or return None when it is missing or a link.

    Called by ``_compare`` for each manifest entry.

    Args:
        path (Path): Snapshot path of a manifest entry.

    Returns:
        str | None: SHA-256 hex digest, or None.
    """
    if path.is_symlink() or not path.is_file():
        return None
    return _sha256(path)


def _sha256(path: Path) -> str:
    """Hash one file in fixed-size chunks.

    Called at transfer time and at snapshot time with the same rule.

    Args:
        path (Path): Regular file.

    Returns:
        str: SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        workspace, record = Path(folder) / "workspace", Path(folder) / "record"
        (workspace / "hiv").mkdir(parents=True)
        (workspace / "hiv" / "HIV_test.csv").write_text(
            "smiles,HIV_active\nC,1\nO,1\n", encoding="utf-8"
        )
        record_transfer(record, workspace, "hiv")
        print(verify_inputs(record, workspace)["status"])
        (workspace / "hiv" / "HIV_test.csv").write_text(
            "smiles,HIV_active\nC,0\nO,1\n", encoding="utf-8"
        )
        print(verify_inputs(record, workspace))

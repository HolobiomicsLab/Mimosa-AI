"""Copy verifier evidence and describe every entry without task-specific filtering.

Symbolic links are copied as links, but inventory never follows them. A snapshot
is an evidence copy, not a security boundary. Callers must not execute untrusted
programs on the host merely because the programs are in a snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

HASH_CHUNK_BYTES = 1024 * 1024
PREVIEW_SAMPLE_BYTES = 64 * 1024
PREVIEW_CHARACTERS = 12 * 1024


def decode_object(response: str) -> dict:
    """Decode one LLM JSON answer, tolerating a Markdown fence on any line count.

    Used by reconnaissance, policy generation, contract assessment, and vision.

    Args: response (str): Text returned by the injected LLM callback.
    Returns: dict parsed from the answer; raises ValueError for non-object JSON.
    """
    text = response.strip()
    if text.startswith("```"):
        text = text[3:].removeprefix("json").removeprefix("JSON")
    result = json.loads(text.removesuffix("```").strip())
    if not isinstance(result, dict):
        raise ValueError("LLM response must be a JSON object")
    return result


def _compress_lines(text: str) -> tuple[str, int]:
    """Reduce repeated adjacent lines for inventory readers.

    Args: text (str): Decoded file sample from the byte reader.
    Returns: tuple[str, int]: Preview text and number of repeated lines removed.
    """
    output, previous, count, removed = [], None, 0, 0
    for line in text.splitlines():
        if line == previous:
            count += 1
            continue
        if count > 1:
            output.append(f"[previous line repeated {count - 1} more times]")
            removed += count - 1
        output.append(line)
        previous, count = line, 1
    if count > 1:
        output.append(f"[previous line repeated {count - 1} more times]")
        removed += count - 1
    return "\n".join(output), removed


def _read_file(path: Path) -> tuple[str, bytes, int]:
    """Hash a regular file without following a replaced symbolic link.

    Args: path (Path): File found by the inventory walker.
    Returns: tuple[str, bytes, int]: SHA-256, bounded sample, and bytes read.
    """
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise OSError("Entry is no longer a regular file")
        digest, sample, size = hashlib.sha256(), bytearray(), 0
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
            sample.extend(chunk[: max(0, PREVIEW_SAMPLE_BYTES - len(sample))])
    return digest.hexdigest(), bytes(sample), size


def _preview(sample: bytes, size: int) -> dict:
    """Describe sample limits instead of hiding unsupported file contents.

    Args: sample (bytes), size (int): Initial file bytes and total byte count.
    Returns: dict: Text or hexadecimal preview and explicit coverage metadata.
    """
    try:
        text = sample.decode("utf-8")
        binary = "\x00" in text
    except UnicodeDecodeError as error:
        binary = error.end < len(sample) or size <= len(sample)
        text = sample.decode("utf-8", errors="replace")
    compact, repeated = _compress_lines(text) if not binary else (sample.hex(), 0)
    return {
        "preview": compact[:PREVIEW_CHARACTERS],
        "preview_encoding": "hex" if binary else "utf-8",
        "preview_bytes": len(sample),
        "preview_truncated": size > len(sample) or len(compact) > PREVIEW_CHARACTERS,
        "binary": binary,
        "repeated_lines_removed": repeated,
    }


def _entry(path: Path, relative: str) -> dict:
    """Describe one filesystem object for policy reconnaissance.

    Args: path (Path), relative (str): Entry and its path below the evidence root.
    Returns: dict: File measurement or a visible link, special-file, or error gap.
    """
    item = {"path": relative, "kind": "error", "size": None, "sha256": None,
            "preview": "", "preview_truncated": False, "binary": None, "error": None}
    try:
        metadata = path.lstat()
        item["size"] = metadata.st_size
        if stat.S_ISLNK(metadata.st_mode):
            item.update(kind="symlink", target=os.readlink(path), gap="not_followed")
        elif stat.S_ISREG(metadata.st_mode):
            item["kind"] = "file"
            digest, sample, size = _read_file(path)
            item.update(sha256=digest, bytes_read=size, **_preview(sample, size))
            if size != metadata.st_size:
                item["error"] = "File changed size during inventory"
        else:
            item.update(kind="special", mode=stat.S_IFMT(metadata.st_mode),
                        gap="unsupported_filesystem_object")
    except OSError as error:
        item["error"] = f"{type(error).__name__}: {error}"
    return item


def _walk(root: Path, directory: Path) -> Iterator[dict]:
    """Walk all directories without following symbolic links.

    Called recursively by inventory, starting at the evidence root. Every
    non-directory entry becomes an _entry record; an unreadable directory
    becomes a visible error record instead of a silent gap.

    Args:
        root (Path): The evidence root, used to compute relative paths.
        directory (Path): The directory currently being walked.

    Returns:
        Iterator[dict]: One inventory record per entry, in sorted order.
    """
    try:
        entries = sorted(directory.iterdir(), key=lambda path: path.name)
    except OSError as error:
        yield {"path": directory.relative_to(root).as_posix(), "kind": "error",
               "size": None, "sha256": None, "error": str(error),
               "preview": "", "preview_truncated": False, "binary": None}
        return
    for path in entries:
        if not path.is_symlink() and path.is_dir():
            yield from _walk(root, path)
        else:
            yield _entry(path, path.relative_to(root).as_posix())


def inventory(root: Path) -> list[dict]:
    """Give the policy writer a complete file inventory with bounded previews.

    Args: root (Path): Existing generation workspace or owned snapshot directory.
    Returns: list[dict]: Sorted relative paths, streaming hashes, previews, and gaps.
    """
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    return list(_walk(root, root))


def snapshot(source: Path, destination: Path) -> Path:
    """Copy evidence to a new location without changing the source.

    Args: source (Path), destination (Path): Existing root and absent copy location.
    Returns: Path: Absolute snapshot path; overlapping roots are rejected.
    """
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    resolved_destination = destination.resolve()
    if source.is_relative_to(resolved_destination) or resolved_destination.is_relative_to(source):
        raise ValueError("Snapshot roots must not overlap")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "evidence"
        shutil.copytree(source, staged, symlinks=True, ignore=_special_entries)
        staged.rename(destination)
    return destination.resolve()

def _special_entries(directory: str, names: list[str]) -> set[str]:
    """Skip FIFOs, sockets and devices, which copytree cannot copy.

    The inventory of the source already lists them as visible gaps.

    Args: directory (str): Directory being copied; names (list[str]): Its entries.
    Returns: set[str]: Entry names that are neither regular files, dirs, nor links.
    """
    skipped = set()
    for name in names:
        mode = os.lstat(os.path.join(directory, name)).st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
            skipped.add(name)
    return skipped



@contextmanager
def pair_workspace(left: Path, right: Path) -> Iterator[Path]:
    """Give one comparison its own disposable copies of both generations.

    Called by the evaluator for every real pair and by PolicyEngine._prepare_smoke
    for self-pairs. Copies land in one temporary directory as ``a`` and ``b``,
    and the whole tree is removed on exit, so comparisons never touch or see
    each other's files.

    Args:
        left (Path): First generation's snapshot directory.
        right (Path): Second generation's snapshot directory.

    Yields:
        Path: Temporary parent directory containing the ``a`` and ``b`` copies.
    """
    with tempfile.TemporaryDirectory(prefix="policy-bt-pair-") as temporary:
        root = Path(temporary)
        snapshot(left, root / "a")
        snapshot(right, root / "b")
        yield root


def _main() -> None:
    """Run a direct inventory command for local evidence inspection.

    Args: None. The command line supplies an existing root directory.
    Returns: None; JSON inventory is written to standard output.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(inventory(arguments.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()

"""Copy verifier evidence and describe every entry without task-specific filtering.

Symbolic links are copied as links, but inventory never follows them. A snapshot
is an evidence copy, not a security boundary. Callers must not execute untrusted
programs on the host merely because the programs are in a snapshot. A virtual
environment (a directory holding ``pyvenv.cfg``) is copied as its ``pyvenv.cfg``
only: it stays visible, but its installed packages are not task evidence.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

HASH_CHUNK_BYTES = 1024 * 1024
PREVIEW_SAMPLE_BYTES = 64 * 1024
PREVIEW_CHARACTERS = 12 * 1024
VENV_MARKER = "pyvenv.cfg"
# One explicit inspection page: bytes of a file, entries of a directory.
PAGE_BYTES = 8192
DIRECTORY_PAGE_ENTRIES = 200
# Directories named in a bounded listing's summary of omitted rows.
MAX_UNLISTED_DIRECTORIES = 200
# Bounded inspect loop of the contract and provenance stages.
INSPECT_ROUNDS = 6
INSPECT_REQUESTS = 8
INSPECT_PAGES = 24
INSPECTION_RULES = f"""To read a file or list a directory ("." is the root) before you
decide, return {{"action":"inspect","requests":[{{"path":"relative/path","offset":0,
"length":{PAGE_BYTES}}}]}} with 1 to {INSPECT_REQUESTS} requests. A file page holds up
to {PAGE_BYTES} bytes from offset; a directory page lists entry names from entry index
offset. The pages come back in EVIDENCE.inspections, which keeps the newest
{INSPECT_PAGES} pages ("inspection_note" says when older ones were dropped).
EVIDENCE.rounds_left counts the inspection rounds left; at 0 you must give the final
answer. EVIDENCE.previous_answer_error explains why your last answer was refused.
You have no tools: every answer, inspect or final, is one plain-text JSON object.
"""


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


def file_digests(root: Path) -> dict[str, str]:
    """Map every regular file below a snapshot root to its SHA-256 digest.

    The evaluator compares the pre-run baseline with each generation
    snapshot. Both are made by ``snapshot``, so the same entries are left out.

    Args: root (Path): Baseline or generation snapshot directory.
    Returns: dict[str, str]: Relative POSIX path to hex digest; links and
        special files are left out.
    """
    return {item["path"]: item["sha256"] for item in inventory(root) if item["kind"] == "file"}


def bounded_listing(entries: list[dict], limit: int) -> tuple[list[dict], dict[str, int]]:
    """Choose the inventory rows one prompt lists, and count the rest.

    Reconnaissance and the reproduction contract call it, because a full
    inventory can exceed the model input (live case 2026-10-21: 1,384 files
    gave a 6.3 M-character contract prompt). Shallow paths come first, so
    top-level outputs and scripts stay listed; omitted rows are counted per
    parent directory, and a directory page shows their names.

    Args:
        entries (list[dict]): Inventory rows from ``inventory``.
        limit (int): Maximum number of rows to list.

    Returns:
        tuple[list[dict], dict[str, int]]: The listed rows, and the number of
        omitted rows per parent directory (the largest
        ``MAX_UNLISTED_DIRECTORIES`` directories).
    """
    ordered = sorted(entries, key=lambda item: (item["path"].count("/"), item["path"]))
    omitted = Counter(Path(item["path"]).parent.as_posix() for item in ordered[limit:])
    return ordered[:limit], dict(omitted.most_common(MAX_UNLISTED_DIRECTORIES))


def safe_path(root: Path, relative: str) -> Path:
    """Keep artifact reads inside a real directory without symlink traversal.

    Args: root (Path): owned artifact root; relative (str): untrusted file name.
    Returns: Path confined below root; raises on unsafe paths.
    """
    if not isinstance(relative, str) or not relative:
        raise ValueError("Artifact path must be a nonempty relative string")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Absolute and parent paths are forbidden")
    candidate = root.resolve()
    for part in path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("Symlink artifact access is forbidden")
    candidate.resolve().relative_to(root.resolve())
    return candidate


def inspect_page(root: Path, request: dict) -> dict:
    """Read one explicit page of a file or a directory for an LLM stage.

    Reconnaissance and the reproduction contract serve the model's page
    requests with it. A file page holds bytes from ``offset``; a directory
    page holds up to ``DIRECTORY_PAGE_ENTRIES`` entries from entry index
    ``offset``, so the model can find files a bounded listing omitted.

    Args: root (Path): owned root; request (dict): untrusted ``path``,
        ``offset`` and ``length`` (bytes, files only).
    Returns: dict: the request with the page content, its encoding, the next
        offset and the remaining byte or entry count; raises ValueError on an
        invalid request.
    """
    offset, length = request.get("offset", 0), request.get("length", PAGE_BYTES)
    if type(offset) is not int or type(length) is not int:
        raise ValueError("Inspection offset and length must be integers")
    if offset < 0 or not 1 <= length <= PAGE_BYTES:
        raise ValueError(f"Inspection length must be 1..{PAGE_BYTES}")
    path = safe_path(root, request["path"])
    if path.is_dir():
        return _directory_page(path, request, offset)
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read(length)
    try:
        content, encoding = data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        content, encoding = base64.b64encode(data).decode("ascii"), "base64"
    return {**request, "content": content, "encoding": encoding,
            "next_offset": offset + len(data),
            "remaining_bytes": max(0, path.stat().st_size - offset - len(data))}


def _directory_page(path: Path, request: dict, offset: int) -> dict:
    """List one page of a directory's entries without following links.

    Called by ``inspect_page`` when the requested path is a directory.

    Args: path (Path): Confined directory; request (dict): The page request;
        offset (int): Index of the first entry to list.
    Returns: dict: The request with ``entries`` (name, kind, size), the next
        offset and the number of remaining entries.
    """
    names = sorted(os.listdir(path))
    entries = []
    for name in names[offset:offset + DIRECTORY_PAGE_ENTRIES]:
        metadata = (path / name).lstat()
        kind = "dir" if stat.S_ISDIR(metadata.st_mode) else "symlink" if stat.S_ISLNK(metadata.st_mode) else "file"
        entries.append({"name": name, "kind": kind, "size": metadata.st_size})
    return {**request, "entries": entries, "next_offset": offset + len(entries),
            "remaining_entries": max(0, len(names) - offset - len(entries))}


def read_page(root: Path, request: object) -> dict:
    """Serve one model page request, turning a bad request into a visible error.

    Reconnaissance and the reproduction contract call it for every request,
    so one bad path never ends the stage; the model sees the error instead.

    Args: root (Path): Owned root; request (object): Untrusted request,
        normally a dict with ``path``, ``offset`` and ``length``.
    Returns: dict: The page from ``inspect_page`` or the request with ``error``.
    """
    if not isinstance(request, dict):
        return {"request": request, "error": "Inspection request must be an object"}
    try:
        return inspect_page(root, request)
    except (OSError, ValueError, KeyError) as error:
        return {**request, "error": str(error)}


def answer_after_inspection(root: Path, instructions: str, evidence: dict,
                            ask: Callable[[str], str]) -> dict:
    """Let the model read pages step by step, then return its final answer.

    The reproduction contract and the provenance check call it with bounded
    first evidence. Each inspect answer is served by ``_serve_pages``; any
    other readable answer is final. An unreadable answer (live case
    2026-10-21: tool-call markup instead of JSON) is shown to the model in
    ``previous_answer_error`` and costs one round.

    Args: root (Path): Owned snapshot; instructions (str): Stage prompt that
        includes ``INSPECTION_RULES``; evidence (dict): JSON-ready evidence,
        changed in place; ask (Callable[[str], str]): LLM callback.
    Returns: dict: The final, still unvalidated answer.
    Raises: ValueError when the rounds end without a readable final answer.
    """
    evidence.setdefault("inspections", [])
    problem = ""
    for rounds_left in range(INSPECT_ROUNDS, -1, -1):
        evidence["rounds_left"] = rounds_left
        answer, problem = _checked_answer(instructions, evidence, ask)
        if not problem and answer.get("action") != "inspect":
            return answer
        if not problem:
            _serve_pages(root, answer["requests"], evidence)
    problem = problem or "it kept inspecting after its last round"
    raise ValueError(f"no readable final answer after {INSPECT_ROUNDS + 1} rounds: {problem}")


def _checked_answer(instructions: str, evidence: dict, ask: Callable[[str], str]) -> tuple[dict, str]:
    """Ask once and check the answer's shape, recording feedback on failure.

    Called by ``answer_after_inspection`` once per round. A refused answer
    is described in ``evidence["previous_answer_error"]`` for the next round.

    Args: instructions (str): Stage prompt; evidence (dict): Evidence,
        changed in place; ask (Callable[[str], str]): LLM callback.
    Returns: tuple[dict, str]: The decoded answer and "" when it is usable,
        or {} and the reason it was refused.
    """
    try:
        answer = decode_object(ask(instructions + "\nEVIDENCE:\n" + json.dumps(evidence, ensure_ascii=False)))
        requests = answer.get("requests")
        if answer.get("action") == "inspect" and not (isinstance(requests, list) and 1 <= len(requests) <= INSPECT_REQUESTS):
            raise ValueError(f"an inspect answer needs 1 to {INSPECT_REQUESTS} requests")
    except ValueError as error:
        evidence["previous_answer_error"] = f"{error}. Answer again with one plain-text JSON object."
        return {}, str(error)
    evidence.pop("previous_answer_error", None)
    return answer, ""


def _serve_pages(root: Path, requests: list, evidence: dict) -> None:
    """Read the requested pages into the bounded inspection transcript.

    Called by ``answer_after_inspection`` for an inspect answer. Only the
    newest ``INSPECT_PAGES`` pages are kept, with a note when older ones drop.

    Args: root (Path): Owned snapshot; requests (list): Checked page
        requests; evidence (dict): Evidence, changed in place.
    Returns: None.
    """
    pages = evidence["inspections"] + [read_page(root, request) for request in requests]
    if len(pages) > INSPECT_PAGES:
        evidence["inspection_note"] = "Oldest inspected pages were dropped; re-request a page if still needed."
    evidence["inspections"] = pages[-INSPECT_PAGES:]


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
        shutil.copytree(source, staged, symlinks=True, ignore=_ignored_entries)
        staged.rename(destination)
    return destination.resolve()

def _ignored_entries(directory: str, names: list[str]) -> set[str]:
    """Skip what copytree cannot copy and the contents of virtual environments.

    ``snapshot`` passes it to ``shutil.copytree``. FIFOs, sockets and devices
    cannot be copied; the inventory of the source lists them as visible gaps.
    A workspace venv (live case 2026-10-15: 48,573 files, 2.2 GB) made the
    reconnaissance prompt exceed the provider's 8 MB limit, so only its
    ``pyvenv.cfg`` marker is kept.

    Args: directory (str): Directory being copied; names (list[str]): Its entries.
    Returns: set[str]: Entry names to leave out of the copy.
    """
    if VENV_MARKER in names:
        return {name for name in names if name != VENV_MARKER}
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

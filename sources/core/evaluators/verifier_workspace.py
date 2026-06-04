"""
Workspace-listing, file-preview and literature-grounding helpers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sources.cli.pretty_print import (
    GREEN,
    YELLOW,
    print_box,
)

from sources.core.evaluators.grounding import get_perspicacite_grounding

# ----- File preview budgets ---------------------------------------------------
_PREVIEW_HEAD_BYTES = 8 * 1024
_PREVIEW_TAIL_BYTES = 2 * 1024
_PREVIEW_PER_CLAIM_CAP = 24 * 1024
_BINARY_SNIFF_BYTES = 4096

# ----- Workspace preview filter -----------------------------------------------
# Suffixes never previewed as text. Everything else is fed through
# ``_preview_file`` which falls back to a magic-byte hex dump for binaries it
# sniffs at read time.
_PREVIEW_DENY_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp", ".ico",
    ".pdf",
    ".pkl", ".pickle", ".npy", ".npz", ".joblib", ".h5", ".hdf5",
    ".bin", ".so", ".o", ".a", ".dll", ".dylib", ".exe",
    ".pyc", ".pyo",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".db", ".sqlite", ".sqlite3",
    ".mp3", ".mp4", ".wav", ".ogg", ".webm",
)


class _VerifierWorkspaceMixin:
    """Workspace listing, file preview and grounding methods.

    Expects the concrete class to provide:
        - ``workspace_dir`` (Path) — root of the agents' workspace.
        - ``preview_head_bytes`` / ``preview_tail_bytes`` /
          ``preview_per_claim_cap`` (int) — per-instance preview budgets.
        - ``use_grounding`` (bool) — Perspicacite opt-out flag.
        - ``logger``.
        - ``_preview_cache`` (dict[str, str]) — cache for ``_preview_file``.
        - ``_grounding_cache`` (dict[str, str]) — cache for ``_get_grounding``.

    Owns / initialises:
        - ``_workspace_files`` (set[str]) — reset on every ``_list_workspace``
          call; read by the claim parser and the per-claim file selector.
    """

    # ------------------------------------------------------------------
    # Workspace listing
    # ------------------------------------------------------------------

    def _list_workspace(self, max_entries: int = 200) -> str:
        """Workspace listing for the prompt; also populates ``_workspace_files``.

        Args:
            max_entries: Soft cap on the number of files included in the listing.

        Returns:
            Newline-joined ``<rel_path>\\t<size>B`` lines, or a placeholder
            string when the workspace is missing or empty.
        """
        ws = self.workspace_dir
        self._workspace_files = set()
        if not ws.exists():
            return "(workspace directory does not exist)"
        entries: list[str] = []
        truncated = False
        for root, dirs, files in os.walk(ws):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "node_modules"}]
            for f in files:
                rel = str(Path(root, f).relative_to(ws))
                self._workspace_files.add(rel)
                if truncated:
                    continue
                try:
                    size = (ws / rel).stat().st_size
                except OSError:
                    size = -1
                entries.append(f"{rel}\t{size}B")
                if len(entries) >= max_entries:
                    entries.append(f"... (truncated at {max_entries} entries)")
                    truncated = True
        return "\n".join(entries) if entries else "(empty workspace)"

    # ------------------------------------------------------------------
    # Literature grounding (Perspicacite)
    # ------------------------------------------------------------------

    _GROUNDING_DISABLED = "(literature grounding disabled for this run)"
    _GROUNDING_FAILED_MARKER = "Perspicacite query failed"

    def _get_grounding(self, uuid: str, execution_text: str, goal: str) -> str:
        """One Perspicacite round-trip per uuid; cached + opt-out.

        Args:
            uuid: Workflow identifier used as the cache key.
            execution_text: Agent narration / produced output text (unused but
                kept for callers that may key on it).
            goal: Workflow goal text submitted to the grounding service.

        Returns:
            Grounding text, the failure marker, or the disabled sentinel.
        """
        if not self.use_grounding:
            return self._GROUNDING_DISABLED
        if uuid in self._grounding_cache:
            return self._grounding_cache[uuid]
        try:
            grounding = get_perspicacite_grounding(goal)
        except Exception as e:
            self.logger.warning(f"Perspicacite grounding raised for {uuid}: {e}")
            grounding = f"{self._GROUNDING_FAILED_MARKER}: {e}"
        self._grounding_cache[uuid] = grounding
        is_usable = (
            grounding
            and self._GROUNDING_FAILED_MARKER not in grounding
            and grounding != self._GROUNDING_DISABLED
        )
        print_box(
            grounding,
            title=f"Perspicacite grounding · {uuid}",
            color=GREEN if is_usable else YELLOW,
            truncate=2000,
        )
        return grounding

    # ------------------------------------------------------------------
    # File preview helpers
    # ------------------------------------------------------------------

    def _validate_workspace_paths(
        self,
        raw,
        allowed: set[str] | None = None,
        max_count: int | None = None,
        label: str = "",
    ) -> list[str]:
        """Normalise, dedupe and validate a list of workspace-relative paths.

        Strips leading ``./``, drops empties, duplicates and non-strings. When
        ``allowed`` is given, paths not in that set are logged and dropped
        (hallucination guard). When ``max_count`` is given, truncates to that
        cap. Returns paths in input order.

        Lives on the workspace mixin because the validation is against
        ``self._workspace_files`` — both the claim extractor (validating LLM
        paths) and the per-claim file selector (validating verifier-input
        paths) call this; centralising here avoids the claims mixin owning a
        helper whose primary state lives elsewhere.

        Args:
            raw: Candidate iterable of path strings from a model response.
            allowed: Optional whitelist of workspace-relative paths.
            max_count: Optional cap on the number of returned paths.
            label: Identifier for the calling claim, used in debug logs.

        Returns:
            Ordered list of cleaned, deduplicated, validated relative paths.
        """
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for rf in raw:
            if not isinstance(rf, str):
                continue
            rp = rf.strip().lstrip("./")
            if not rp or rp in seen:
                continue
            if allowed is not None and rp not in allowed:
                self.logger.debug(f"Dropping confabulated path '{rp}' for {label}")
                continue
            out.append(rp)
            seen.add(rp)
            if max_count is not None and len(out) >= max_count:
                break
        return out

    def _eligible_workspace_files(self) -> list[str]:
        """Workspace files plausibly readable as text artefacts (sorted).

        Filters out compiled artefacts and obvious binaries by suffix; the
        deeper magic-byte check inside ``_preview_file`` still catches
        anything that slips through.

        Returns:
            Sorted list of workspace-relative paths eligible for text preview.
        """
        out: list[str] = []
        for f in self._workspace_files:
            lower = f.lower()
            if lower.endswith(_PREVIEW_DENY_SUFFIXES):
                continue
            if any(p in lower for p in ("/__pycache__/", "/.git/", "/.venv/")):
                continue
            out.append(f)
        return sorted(out)

    @staticmethod
    def _looks_binary(sample: bytes) -> bool:
        """Heuristic: NUL bytes or > 30% non-printables → treat as binary.

        Args:
            sample: Leading byte sample read from a file.

        Returns:
            True when the sample looks binary; False for plausibly textual data.
        """
        if not sample:
            return False
        if b"\x00" in sample:
            return True
        # Allow common whitespace + printable ASCII + UTF-8 high bytes.
        printable = sum(
            1
            for b in sample
            if b in (9, 10, 13) or 32 <= b < 127 or b >= 0x80
        )
        return printable / len(sample) < 0.7

    def _preview_file(self, rel_path: str) -> str:
        """Cached LLM-friendly preview: text head+tail, binary magic bytes, fenced.

        Args:
            rel_path: Workspace-relative path to the file to preview.

        Returns:
            Fenced preview string suitable for inclusion in a judge prompt.
        """
        if rel_path in self._preview_cache:
            return self._preview_cache[rel_path]

        # Reject anything escaping the workspace.
        ws = self.workspace_dir.resolve()
        candidate = (self.workspace_dir / rel_path).resolve()
        try:
            candidate.relative_to(ws)
        except ValueError:
            rendered = f"=== {rel_path} ===\n(refusing to preview file outside workspace)\n"
            self._preview_cache[rel_path] = rendered
            return rendered

        if not candidate.exists():
            rendered = f"=== {rel_path} ===\n(file not found in workspace)\n"
            self._preview_cache[rel_path] = rendered
            return rendered
        if candidate.is_dir():
            rendered = f"=== {rel_path} ===\n(path is a directory, not a file)\n"
            self._preview_cache[rel_path] = rendered
            return rendered

        try:
            size = candidate.stat().st_size
        except OSError as e:
            rendered = f"=== {rel_path} ===\n(stat failed: {e})\n"
            self._preview_cache[rel_path] = rendered
            return rendered

        head_budget = max(self.preview_head_bytes, _BINARY_SNIFF_BYTES)
        try:
            with open(candidate, "rb") as fh:
                head_bytes = fh.read(head_budget)
                if size > head_budget + self.preview_tail_bytes:
                    fh.seek(max(0, size - self.preview_tail_bytes))
                    tail_bytes = fh.read(self.preview_tail_bytes)
                else:
                    tail_bytes = b""
        except OSError as e:
            rendered = f"=== {rel_path} ===\n(read failed: {e})\n"
            self._preview_cache[rel_path] = rendered
            return rendered

        sample = head_bytes[:_BINARY_SNIFF_BYTES]
        if self._looks_binary(sample):
            magic = head_bytes[:16].hex(" ")
            rendered = (
                f"=== {rel_path} ({size} bytes, binary) ===\n"
                f"first 16 bytes (hex): {magic}\n"
            )
            self._preview_cache[rel_path] = rendered
            return rendered

        try:
            head_text = head_bytes[: self.preview_head_bytes].decode("utf-8", errors="replace")
            tail_text = tail_bytes.decode("utf-8", errors="replace") if tail_bytes else ""
        except Exception as e:
            rendered = f"=== {rel_path} ===\n(decode failed: {e})\n"
            self._preview_cache[rel_path] = rendered
            return rendered

        if tail_text:
            elided = size - self.preview_head_bytes - self.preview_tail_bytes
            body = (
                f"{head_text.rstrip()}\n"
                f"... ({elided} bytes elided) ...\n"
                f"{tail_text.lstrip()}"
            )
        else:
            body = head_text

        rendered = f"=== {rel_path} ({size} bytes, text) ===\n{body}\n"
        self._preview_cache[rel_path] = rendered
        return rendered

    def _render_relevant_previews(self, rel_paths: list[str]) -> str:
        """Concatenate file previews under ``preview_per_claim_cap``.

        Args:
            rel_paths: Workspace-relative paths to preview, in priority order.

        Returns:
            Concatenated previews; exhausted/over-budget entries are noted inline.
        """
        if not rel_paths:
            return "(no relevant files declared for this claim)"
        chunks: list[str] = []
        used = 0
        for rp in rel_paths:
            preview = self._preview_file(rp)
            remaining = self.preview_per_claim_cap - used
            if remaining <= 0:
                chunks.append(f"=== {rp} ===\n(preview budget exhausted; not shown)\n")
                continue
            if len(preview) > remaining:
                preview = preview[:remaining] + "\n... (preview truncated by per-claim budget)\n"
            chunks.append(preview)
            used += len(preview)
        return "\n".join(chunks)


if __name__ == "__main__":
    # Smoke check: the mixin must import + define the expected method set.
    expected = {
        "_list_workspace",
        "_get_grounding",
        "_validate_workspace_paths",
        "_eligible_workspace_files",
        "_looks_binary",
        "_preview_file",
        "_render_relevant_previews",
    }
    actual = {n for n in dir(_VerifierWorkspaceMixin) if not n.startswith("__")}
    missing = expected - actual
    assert not missing, f"workspace mixin missing methods: {missing}"
    print("verifier_workspace: smoke ok")

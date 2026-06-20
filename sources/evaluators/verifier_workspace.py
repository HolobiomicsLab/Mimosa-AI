"""
Workspace-listing, file-preview and literature-grounding helpers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# ----- File preview budgets ---------------------------------------------------
_BINARY_SNIFF_BYTES = 16384
_MAX_WORKSPACE_LISTING_ENTRIES = 200

class _VerifierWorkspaceMixin:
    """Workspace listing, file preview and grounding methods."""

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
        Strips leading ./, drops empties, duplicates and non-strings
        When ``max_count`` is given, truncates to that cap. Returns paths in input order.

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
        """Sort Workspace files by readability and path depth.

        Filters out compiled artefacts and obvious binaries by suffix; the
        deeper magic-byte check inside ``_preview_file`` still catches
        anything that slips through.

        Returns:
            Sorted list of workspace-relative paths eligible for text preview.
        """
        out: list[str] = []
        for f in self._workspace_files:
            lower = f.lower()
            if any(p in lower for p in ("/__pycache__/", "/.git/", "/.venv/")):
                continue
            out.append(f)
        eligibilityness = lambda rp: self._eligibility_score(rp)
        return sorted(out, key=eligibilityness, reverse=True)[:_MAX_WORKSPACE_LISTING_ENTRIES]

    def _eligibility_score(self, rel_path: str) -> float:
        return (
            self._non_binaryness(self.workspace_dir / rel_path)
            - 0.05 * rel_path.count("/")
        )

    @staticmethod
    def _non_binaryness(path: Path) -> float:
        """Return 0-1 score of how textual the sample is.

        Args:
            path: Path to the file to sample.
        Returns:
            float in [0.0, 1.0] representing the fraction of bytes that are
        """
        try:
            with open(path, "rb") as fh:
                sample = fh.read(_BINARY_SNIFF_BYTES)
        except OSError:
            return 0.0
        if not sample or b"\x00" in sample:
            return 0.0
        printable = sum(
            1 for b in sample
            if b in (9, 10, 13) or 32 <= b < 127 or b >= 0x80
        )
        return printable / len(sample)

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
    # Quick sanity check of the workspace listing and preview methods.
    from sources.evaluators.verifier import VerifierEvaluator
    from config import Config

    ws = Path(".").resolve()
    conf = Config()
    v = VerifierEvaluator(conf, workspace_dir=ws, use_grounding=False)
    ws_listing = v._list_workspace(max_entries=100)
    eligible = v._eligible_workspace_files()
    print(f"Eligible workspace files ({len(eligible)}):")
    print(eligible[:24])
    print(eligible[-24:])

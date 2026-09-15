"""Per-claim verifier generation, sandboxed execution and scoring."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypeVar

if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    )

from sources.cli.pretty_print import CYAN, DIM, GREEN, RED, YELLOW, print_box
from sources.core.llm_provider import LLMProvider
from sources.core.workflow_runner import (
    ExecutionResult,
    ExecutionStatus,
    RuntimeConfig,
    WorkflowRunner,
)

# ----- Verifier helper packages ----------------------------------------------
# Installed once per process so verifier scripts can rely on them being
# importable. Minimal on purpose: numerical + tabular + classical stats + ML.
_VERIFIER_BASE_PACKAGES: tuple[str, ...] = (
    "numpy", "pandas", "scipy", "scikit-learn", "pint",
    "pydantic", "pandera", "jsonschema", "sympy", "openpyxl",
    "Pillow",
)
# Python module names corresponding to ``_VERIFIER_BASE_PACKAGES`` for the
# post-install smoke check (scikit-learn → sklearn).
_VERIFIER_BASE_IMPORTS: tuple[str, ...] = (
    "numpy", "pandas", "scipy", "sklearn", "PIL",
)

_VERIFIER_PACKAGES_INSTALLED = False
_VERIFIER_INSTALL_LOCK = threading.Lock()

# ----- Tunables --------------------------------------------------------------
# Caps applied during the bounded-retry policy so a pathological claim cannot
# snowball into an open-ended install + regen loop.
_RECOVERY_MAX_ATTEMPTS = 10
_RECOVERY_MAX_INSTALL_PACKAGES = 6
_RECOVERY_INSTALL_TIMEOUT_SECONDS = 300
_RECOVERY_STDERR_FEEDBACK_LINES = 30
# How many past attempts the regeneration prompt renders in full; older ones
# are summarised as a one-line count to keep the prompt bounded.
_RECOVERY_HISTORY_RENDER = 3
# Substrings that identify a missing-dependency stderr; anything else is
# treated as a generated-code bug and routed through script regeneration.
_RECOVERY_IMPORT_MARKERS: tuple[str, ...] = (
    "ModuleNotFoundError",
    "ImportError: No module named",
    "ImportError: cannot import name",
)
# Pass-despite-error guard: markers that make a script-printed pass/fail
# verdict untrustworthy when they appear in the verifier's OWN stderr (an
# uncaught exception leaked past the printed JSON verdict).
_RECOVERY_EXCEPTION_STDERR_MARKERS: tuple[str, ...] = (
    "Traceback (most recent call last)",
    "ModuleNotFoundError",
    "ImportError: No module named",
    "ImportError: cannot import name",
    "FileNotFoundError",
    "PermissionError",
    "NameError:",
    "SyntaxError:",
    "IndentationError",
)
# Load/import failure markers that invalidate a printed "pass" when found in
# the check's own ``details`` — the salvage pattern where a script catches an
# artefact-load failure, inspects metadata instead, and still prints pass.
_RECOVERY_SALVAGE_DETAIL_MARKERS: tuple[str, ...] = (
    "ModuleNotFoundError",
    "ImportError",
    "No module named",
    "UnpicklingError",
    "cannot unpickle",
)
_BASE_INSTALL_TIMEOUT_SECONDS = 600
_RUNNER_CLEANUP_TIMEOUT = 15
_RUNNER_SMOKE_TIMEOUT = 25.0
_RUNNER_EXTRA_TIMEOUT = 10
_PIP_INSTALL_FLAGS: tuple[str, ...] = (
    "--quiet", "--disable-pip-version-check", "--break-system-packages",
)
_PRINT_TRUNCATE_BYTES = 256
_STDERR_TAIL_BYTES = 400

# ----- Visual (Source G) image selection --------------------------------------
# Image extensions accepted by the visual branch. NOTE: PDF/SVG are sent as
# raw bytes (no rasterization on this path); see the visual-branch docstring.
_VISUAL_IMAGE_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".svg"}
)
# Cap on how many candidate images are sent to the vision model in one call.
_MAX_VISUAL_IMAGES = 6
# Suffixes (and directory names) that mark a file as code/source rather than
# data or output. Visual candidates and selection fallbacks never include these.
_CODE_FILE_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py", ".pyw", ".ipynb", ".r", ".jl", ".sh", ".bash", ".zsh", ".pl",
        ".rb", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cc", ".cpp",
        ".h", ".hpp", ".cs", ".go", ".rs", ".sql", ".lua", ".ps1", ".bat",
        ".cmd", ".do", ".sas", ".m", ".php", ".swift", ".scala",
    }
)
_CODE_PATH_PARTS: frozenset[str] = frozenset(
    {"src", "source", "sources", "scripts", "script", "code", "notebooks"}
)
# Path segments that mark a file as a produced RESULT/OUTPUT rather than an
# input or dataset file; used to prioritise visual candidates.
_OUTPUT_PATH_PARTS: frozenset[str] = frozenset(
    {
        "pred_results", "pred_result", "out", "output", "outputs",
        "result", "results", "figure", "figures", "figs", "fig",
        "plot", "plots", "charts", "chart",
    }
)
_INPUT_PATH_PARTS: frozenset[str] = frozenset(
    {"data", "dataset", "datasets", "raw", "input", "inputs", "assets"}
)

VERIFIER_PROMPT_RULES = """
RULES:

- Print EXACTLY ONE JSON line to stdout, structured as:
  {{"claim_id": "<id>", "status": "pass" | "fail" | "error",
    "actual": <observed value or null>, "details": "<short string>"}}
- Read files with relative paths
  (cwd is the workspace).
- Recompute or directly check; do not trust the agent's reported numbers.
- For property checks (symmetry, range, no duplicates, ...), assert the
  property and emit "pass"/"fail" accordingly.
- Guard against vacuous comparisons. When a property reduces to a
  comparison of order statistics across two groups (e.g. "all of A >
  all of B" becoming ``min(A) > max(B)``), first check that BOTH
  operand groups are drawn from the real data distribution. No placeholder/sentinel values (commonly -999, -9999, 9999, NaN, None)
- If the previews are empty or do not show enough of the file to be sure of
  the format, prefer permissive parsing (try several reasonable splits, skip
  unparseable lines) over a strict format that may misjudge the file.
- Evidence comes from RESULT artefacts (outputs, tables, figures, reports,
  manifests, logs) and execution-observable facts — NOT from the workflow's
  source code. Do NOT read or parse workflow scripts (`.py`, `.R`,
  notebooks): no `ast`, no regex over script text, no import matching.
- When the claim is about WHAT the code did (an import, a call, a
  hyperparameter value), check the observable consequence.
- Regex  ``re`` can ONLY be used for unstructured text (logs, READMEs,
  manifests) — never for `.py`/`.R` source files.
- If a file the script needs to open to evaluate the claim is MISSING from the
  workspace (such as logs) emit ``status="fail"`` along with ``details='explain what's missing'``
- NO SALVAGE ON LOAD FAILURE. If the claim's target artefact cannot be
  loaded or parsed with the AVAILABLE libraries (missing package, unreadable
  pickle, wrong format), the claim is NOT verified: emit ``status="error"`` or
  let the exception propagate. NEVER reinterpret metadata as a substitute for
  actually loading the content — no pickle GLOBAL-opcode inspection, no file
  headers or magic bytes, no filename/naming evidence — and NEVER print
  ``status="pass"`` after catching such a load failure.
- When the goal text names an output path with a column or key schema (dataset preview, EXPECTED OUTPUT: block, or explicit 'columns exactly equal to …'), use the goal's schema as the source of truth for column/key literals.
- The file preview shows what the workflow actually produced — which may be wrong. If the preview's schema differs from the goal's, the check must use the goal's schema; the workflow's deviation is exactly what fails the claim
  (eg: Never check AF_TOX_prob when the goal show that AF_TOX is used for columns format)
- Never search the workflow source for call names, attributes, or literals.
  If the script needs to find a dropped column or a hyperparameter name,
  inspect the produced artefact (read the CSV, compare against the goal's
  ground-truth schema) instead of parsing the workflow source
- INPUT-DATA PROPERTY GUARD. If the claim asserts a property solely of
  the PROVIDED INPUT files (class balance of a given test set, overlap
  inside the provided split, dataset size, raw-input schema quirks) and
  no produced artefact is on the claim's file list, do NOT check it:
  return {{"executable": false}} with reason "input-data property — not
  output integrity". A workflow cannot fix its own inputs; scoring such
  a claim pins the reward forever. Exception: alignment claims that
  COMPARE a produced artefact against an input schema (identifier
  columns, label columns, row correspondence) are legitimate and must
  be checked.

ERROR HANDLING:

- catch and raise ONLY for SCRIPT failures — the check could not be
  performed, file unparseable (re-raise as RuntimeError(...) from e), import/env broken, unexpected error.
  These trigger recovery and regenerate the script.
- catch and emit status="fail" for PROPERTY violations — the artefact was present and
  readable, but the claim is not satisfied (wrong value, duplicate found,
  range exceeded, expected content missing, regex absent in a log).
  Do NOT raise these. Do NOT use status="error" for them.
- Mnemonic: "I could not check" → error/raise; "I checked, the answer is no" → fail.

If the claim cannot be checked deterministically with code (e.g. it concerns
the rigor of a proof, the appropriateness of a binning choice, the
defensibility of a conclusion), set "executable": false and explain briefly.

Don't forget to include the library you need such as json, numpy, etc..
You can ONLY use library from the standard library and the available imports.
"""

RECOVERY_PROMPT_RULES = """
- Diagnose the failure from the traceback above and emit a corrected script.
- Keep the output contract: print EXACTLY ONE JSON line to stdout shaped
  {{"claim_id": "<id>", "status": "pass"|"fail", "actual": <value or null>, "details": "<short string>"}}.
- Read files with relative paths (cwd is the workspace).
- If the previous failure was an ImportError, rewrite without that package using the available imports and the standard library.
- Do NOT repeat any pattern already tried in the FAILURE HISTORY below; each
  attempt lists what failed and what was done about it.
- NO SALVAGE ON LOAD FAILURE: if the artefact cannot be loaded or parsed with
  the AVAILABLE libraries, the claim is NOT verified — emit ``status="error"``
  or let the exception propagate. Never inspect pickle GLOBAL opcodes, file
  headers, or filenames as a substitute for loading the content, and never
  print ``status="pass"`` after catching a load failure.
- If the failure come from the absence of a provenance sources files (such as no logs indicated but only source that contain the information), then add a fail route ``status="fail"`` upon this file absence, with details field of the return json indicating what's missing.
"""


def _render_recovery_history(history: list[dict[str, Any]] | None) -> str:
    """Render accumulated recovery attempts for the regeneration prompt.

    Shows the last ``_RECOVERY_HISTORY_RENDER`` attempts in full (kind,
    action, short stderr tail) and summarises older ones in one line, so a
    long loop cannot blow up the prompt while the LLM still sees what was
    already tried and must not repeat.
    """
    if not history:
        return ""
    rendered = history[-_RECOVERY_HISTORY_RENDER:]
    older = len(history) - len(rendered)
    lines = []
    if older > 0:
        lines.append(f"(... {older} earlier attempt(s) already tried and failed)")
    for entry in rendered:
        tail = " | ".join((entry.get("stderr_tail") or "").splitlines()[-3:])[:300]
        lines.append(
            f"- attempt {entry.get('attempt')}: failed as "
            f"{entry.get('kind')}; action: {entry.get('action') or '(pending)'}"
            + (f"; error tail: {tail}" if tail else "")
        )
    return "FAILURE HISTORY (do not repeat these patterns):\n" + "\n".join(lines) + "\n"


T = TypeVar("T")


def _detect_language(workspace_listing: str) -> str:
    """Heuristically classify a workspace's scripting language from a listing.

    Parses each ``name<TAB>size`` row of *workspace_listing* and counts ``.py``
    versus ``.R`` extensions. Returns one of ``"python"``, ``"r"``, ``"mixed"``,
    or ``"unknown"`` so the verifier-gen prompt can warn the LLM not to assume
    Python AST when the workflow scripts are in R.
    """
    py_count = 0
    r_count = 0
    for raw in (workspace_listing or "").splitlines():
        path = raw.split("\t", 1)[0].strip().lower()
        if path.endswith(".py"):
            py_count += 1
        elif path.endswith(".r"):
            r_count += 1
    if py_count == 0 and r_count == 0:
        return "unknown"
    if py_count > 0 and r_count == 0:
        return "python"
    if r_count > 0 and py_count == 0:
        return "r"
    return "mixed"


def _run_coro_sync(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],
    thread_timeout: float | None = None,
) -> T:
    """Run an async coroutine from sync code, even if a loop already runs."""
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False
    if not in_loop:
        return asyncio.run(coro_factory())
    return _run_coro_in_worker(coro_factory, thread_timeout)




def _run_coro_in_worker(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],
    thread_timeout: float | None,
) -> T:
    """Spawn a daemon thread to run *coro_factory*; raise on timeout/error."""
    holder: dict[str, Any] = {}

    def _target() -> None:
        try:
            holder["result"] = asyncio.run(coro_factory())
        except BaseException as exc:
            holder["error"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=thread_timeout)
    if t.is_alive():
        raise TimeoutError(f"Worker thread did not finish within {thread_timeout}s")
    if "error" in holder:
        raise holder["error"]
    return holder["result"]


class _VerifierPerClaimMixin:
    """Generate, execute and score one claim's verifier program."""

    # ------------------------------------------------------------------
    # Stage 2 + 3 + 4 — generate, run and score one verifier
    # ------------------------------------------------------------------

    def _verify_claim(
        self,
        uuid: str,
        claim: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
        grounding: str = "",
        preloaded_spec: dict[str, Any] | None = None,
        goal: str = "",
    ) -> dict[str, Any]:
        """Generate, execute and score a single claim; return the result dict."""
        t_start = time.time()
        claim = self._resolve_relevant_files(
            uuid, claim, execution_text, workspace_listing, preloaded_spec, goal
        )
        self._print_claim_header(claim)
        spec = preloaded_spec or self._visual_claim_spec(claim) or self._generate_verifier(
            uuid, claim, execution_text, workspace_listing
        )
        spec, scored = self._run_and_score(
            uuid, claim, spec, execution_text, workspace_listing, grounding, goal
        )
        scored["claim"] = claim
        scored["spec"] = spec
        scored["elapsed_s"] = round(time.time() - t_start, 3)
        self._print_claim_verdict(claim, scored)
        return scored

    def _resolve_relevant_files(
        self,
        uuid: str,
        claim: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
        preloaded_spec: dict[str, Any] | None,
        goal: str = "",
    ) -> dict[str, Any]:
        """Populate ``likely_relevant_files`` via judge call when not anchored."""
        if preloaded_spec is not None:
            return claim
        rel_files = self._llm_select_files(
            uuid, claim, execution_text, goal=goal
        )
        return {**claim, "likely_relevant_files": rel_files}

    @staticmethod
    def _is_visual_claim(claim: dict[str, Any]) -> bool:
        """True when the claim belongs to a visual (Source G) source."""
        return (claim.get("source") or "").endswith("_g")

    @staticmethod
    def _visual_claim_spec(claim: dict[str, Any]) -> dict[str, Any] | None:
        """Non-executable stand-in spec for visual claims, else ``None``.

        Visual claims are judged by the vision model; generating a verifier
        script for them is a wasted LLM call whose output is discarded. When
        no spec has been preloaded, short-circuit with this marker instead.
        """
        if not (claim.get("source") or "").endswith("_g"):
            return None
        return {
            "executable": False,
            "reason": "visual claim: judged by the vision model; no verifier script",
        }

    @staticmethod
    def _print_claim_header(claim: dict[str, Any]) -> None:
        """Render the per-claim opening box."""
        files = claim.get("likely_relevant_files") or "(none)"
        body = (
            f"id:          {claim.get('id')}\n"
            f"importance:  {claim.get('importance')}\n"
            f"description: {claim.get('description')}\n"
            f"files:       {files}"
        )
        print_box(body, title=f"Verifying claim {claim.get('id')}", color=CYAN)

    @staticmethod
    def _print_claim_verdict(claim: dict[str, Any], scored: dict[str, Any]) -> None:
        """Render the final per-claim verdict box."""
        body = (
            f"id:     {claim.get('id')}\n"
            f"kind:   {scored.get('verifier_kind')}\n"
            f"status: {scored.get('status')}\n"
            f"score:  {scored.get('score')}\n"
            f"elapsed:{scored['elapsed_s']}s"
        )
        color = GREEN if scored.get("score", 0) >= 0.5 else RED
        print_box(body, title=f"Claim verdict · {claim.get('id')}", color=color)

    def _run_and_score(
        self,
        uuid: str,
        claim: dict[str, Any],
        spec: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
        grounding: str,
        goal: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Dispatch to executable or soft branch; return ``(final_spec, scored)``."""
        # Source G claims → visual branch (vision LLM inspects the figure)
        if (claim.get("source") or "").endswith("_g"):
            return spec, self._run_visual_branch(
                uuid, claim, spec, execution_text, workspace_listing, grounding, goal
            )
        if spec.get("executable") and spec.get("code"):
            spec, exec_result = self._run_verifier_with_recovery(
                uuid, claim, spec
            )
            self._print_executable_summary(claim, exec_result)
            return spec, self._score_executable(claim, spec, exec_result)
        return spec, self._run_soft_branch(
            uuid, claim, spec, execution_text, workspace_listing, grounding
        )

    @staticmethod
    def _print_executable_summary(
        claim: dict[str, Any], exec_result: dict[str, Any]
    ) -> None:
        """Render run-status / stdout / stderr boxes for an executable run."""
        cid = claim.get("id")
        run_status = exec_result.get("status", "?")
        color = GREEN if run_status == "pass" else (YELLOW if run_status == "fail" else RED)
        summary = (
            f"status:      {run_status}\n"
            f"exit_status: {exec_result.get('exit_status', '?')}\n"
            f"details:     {exec_result.get('details') or ''}"
        )
        print_box(summary, title=f"Verifier run · {cid}", color=color)
        stdout = (exec_result.get("raw_stdout") or "").strip()
        stderr = (exec_result.get("raw_stderr") or "").strip()
        if stdout:
            print_box(stdout, title=f"stdout · {cid}", color=DIM, truncate=_PRINT_TRUNCATE_BYTES)
        if stderr:
            print_box(stderr, title=f"stderr · {cid}", color=RED, truncate=_PRINT_TRUNCATE_BYTES)

    def _run_soft_branch(
        self,
        uuid: str,
        claim: dict[str, Any],
        spec: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
        grounding: str,
    ) -> dict[str, Any]:
        """Run the non-executable scoring branch and render its boxes."""
        reason = spec.get("reason", "")
        cid = claim.get("id")
        print_box(
            f"Marked non-executable.\nreason: {reason or '(none provided)'}",
            title=f"Verifier spec · {cid}", color=YELLOW,
        )
        scored = self._score_soft(
            uuid, claim, execution_text, workspace_listing, reason, grounding
        )
        verdict_color = GREEN if scored.get("score", 0) >= 0.5 else RED
        print_box(
            f"verdict:   {scored.get('status')}\nrationale: {scored.get('rationale', '')}",
            title=f"Soft check · {cid}", color=verdict_color,
        )
        return scored

    # ------------------------------------------------------------------
    # Visual branch — vision-LLM inspection of figures (Source G)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_likely_code_file(rel_path: str) -> bool:
        """True when *rel_path* looks like a script/source file, not data.

        Matches code suffixes (``.py``, ``.R``, ``.jl``, ``.sh``, ``.ipynb``,
        ...) and code-looking ancestor directories (``src/``, ``scripts/``,
        ...). Visual image candidates and selector fallbacks never include
        files flagged by this predicate.
        """
        lowered = str(rel_path).lower().replace("\\", "/")
        if Path(lowered).suffix in _CODE_FILE_SUFFIXES:
            return True
        parts = [p for p in lowered.split("/") if p]
        return any(p in _CODE_PATH_PARTS for p in parts[:-1])

    @staticmethod
    def _declared_output_image_names(goal: str) -> set[str]:
        """Image-like filenames literally named in the goal text.

        Goals routinely carry the declared deliverable path verbatim (e.g.
        ``"save it to pred_results/spatial_pred.png"``); matching basenames
        get the top priority tier when collecting visual candidates.
        """
        if not goal:
            return set()
        pattern = re.compile(
            r"[\w./\-]+\.(?:png|jpe?g|gif|webp|pdf|svg)\b", re.IGNORECASE
        )
        return {
            Path(m.group(0).replace("\\", "/")).name.lower()
            for m in pattern.finditer(goal)
        }

    def _collect_visual_image_candidates(
        self, claim: dict[str, Any], goal: str = ""
    ) -> list[str]:
        """Collect ALL non-code image candidates, result-like paths first.

        The workspace scan walks the FULL tree (the old first-directory
        ``break`` made whichever directory happened to be visited first —
        often the input dataset — win over the produced deliverable).
        """
        relevant = claim.get("likely_relevant_files") or []
        claimed = [
            str(p) for p in relevant
            if Path(str(p)).suffix.lower() in _VISUAL_IMAGE_EXTS
        ]
        claimed_set = set(claimed)
        seen = set(claimed)
        ws = Path(self.workspace_dir)
        if ws.exists():
            for root, dirs, files in os.walk(str(ws)):
                dirs[:] = [
                    d for d in dirs
                    if d not in {".git", "__pycache__", ".venv", "node_modules"}
                ]
                for fname in files:
                    if Path(fname).suffix.lower() not in _VISUAL_IMAGE_EXTS:
                        continue
                    rel = str(Path(root, fname).relative_to(ws))
                    if rel not in seen:
                        seen.add(rel)
                        claimed.append(rel)

        declared = self._declared_output_image_names(goal)

        def tier(rel: str) -> int:
            parts = [p.lower() for p in rel.replace("\\", "/").split("/") if p]
            if Path(rel).name.lower() in declared:
                return 3
            if any(p in _OUTPUT_PATH_PARTS for p in parts[:-1]):
                return 2
            if rel in claimed_set:
                return 1
            if any(p in _INPUT_PATH_PARTS for p in parts[:-1]):
                return -1
            return 0

        candidates = [
            c for c in claimed
            if not self._is_likely_code_file(c)
            and ".." not in c.split("/")
            and (ws / c).is_file()
        ]
        # Result-like deliverables first; stable alphabetical order per tier.
        candidates.sort(key=lambda rel: (-tier(rel), rel))
        return candidates

    def _run_visual_branch(
        self,
        uuid: str,
        claim: dict[str, Any],
        spec: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
        grounding: str,
        goal: str = "",
    ) -> dict[str, Any]:
        """Run visual verification: send images + claim to a vision-capable LLM.

        Args:
            uuid: Workflow identifier.
            claim: The claim dict with ``description``, ``likely_relevant_files``, etc.
            spec: The verifier spec dict (unused for visual branch, kept for
                interface consistency).
            execution_text: Agent narration / produced output.
            workspace_listing: Rendered listing of workspace files.
            grounding: Optional literature grounding block.
            goal: The workflow's real task goal (user specification).

        Returns:
            Scored result dict with ``score``, ``verifier_kind`` = ``"visual"``,
            ``status``, ``details``, and ``rationale``.
        """
        import base64
        import mimetypes

        cid = claim.get("id", "unknown")

        # ---- locate image files (claim-named + full-tree scan, no early break)
        image_paths = self._collect_visual_image_candidates(claim, goal)
        image_paths = image_paths[:_MAX_VISUAL_IMAGES]

        if not image_paths:
            print_box(
                f"No image files found in workspace for claim {cid}.",
                title=f"Visual check · {cid}", color=YELLOW,
            )
            return {
                "score": 0.0, "verifier_kind": "visual", "status": "error",
                "details": "No image files found in workspace to visually inspect",
                "rationale": "",
            }

        # ---- encode every candidate image, skipping unreadable ones ----
        ws = Path(self.workspace_dir)
        data_uris: list[str] = []
        image_labels: list[str] = []
        for rel in image_paths:
            image_path = ws / rel
            try:
                with open(image_path, "rb") as fh:
                    image_bytes = fh.read()
            except OSError as exc:
                self.logger.warning(
                    f"skipping unreadable image {rel} for claim {cid}: {exc}"
                )
                continue
            mime, _ = mimetypes.guess_type(str(image_path))
            if not mime or not mime.startswith("image/"):
                mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
            data_uris.append(
                f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
            )
            image_labels.append(f"{rel} ({len(image_bytes)} bytes)")

        if not data_uris:
            return {
                "score": 0.0, "verifier_kind": "visual", "status": "error",
                "details": f"Cannot read any candidate image file: {', '.join(image_paths[:3])}",
                "rationale": "",
            }

        # ---- build multimodal prompt ----
        prompt = self._build_visual_check_prompt(
            claim, execution_text, workspace_listing, grounding, goal,
            image_labels=image_labels,
        )
        print_box(
            f"claim: {claim.get('description', '')[:200]}\n"
            f"images ({len(data_uris)}):\n" + "\n".join(image_labels),
            title=f"Visual check · {cid}", color=CYAN,
        )

        # ---- call vision model (ALL candidate images, not just the first) ----
        verdict_data, err = self._call_vision_judge(
            uuid, f"verifier_visual_{cid}", prompt, images=data_uris
        )
        if err is not None or not isinstance(verdict_data, dict):
            print_box(
                f"Vision judge call failed: {err or 'non-dict response'}",
                title=f"Visual error · {cid}", color=RED,
            )
            return {
                "score": 0.0, "verifier_kind": "visual", "status": "error",
                "details": f"vision model call failed: {err or 'verdict JSON not an object'}",
                "rationale": "",
            }

        verdict = verdict_data.get("verdict", "unsure")
        rationale = str(verdict_data.get("rationale", ""))
        score = self._SOFT_VERDICT_SCORE.get(verdict, 0.5)
        color = GREEN if score >= 0.5 else RED
        print_box(
            f"verdict:   {verdict}\nrationale: {rationale}",
            title=f"Visual verdict · {cid}", color=color,
        )
        return {
            "score": score,
            "verifier_kind": "visual",
            "status": verdict,
            "details": rationale,
            "rationale": rationale,
        }

    def _build_visual_check_prompt(
        self,
        claim: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
        grounding: str,
        goal: str = "",
        image_labels: list[str] | None = None,
    ) -> str:
        """Prompt for the vision model to judge scientific figure(s).

        The vision model receives this text prompt alongside every candidate
        image as a data URI. The prompt asks for a targeted scientific
        judgment, not a general description. The REAL task goal (user
        specification) is shown under TASK GOAL; the agents' self-reported
        output is shown separately and explicitly marked UNTRUSTED.
        """
        grounding_block = (
            grounding.strip()[:1500] if grounding else "(no literature grounding)"
        )
        goal_block = (goal or "").strip()[:2500] or "(no goal text available)"
        agent_block = (execution_text or "").strip()[-2500:] or "(no agent output)"
        images_block = (
            "\n".join(f"- {label}" for label in (image_labels or []))
            or "(image attached)"
        )
        return f"""You are a scientific figure reviewer with domain expertise across chemistry, biology, and physics. Examine the attached image(s) carefully.

CLAIM TO VERIFY:
{claim.get('description', '')}

TASK GOAL (the user's specification for the workflow — authoritative):
{goal_block}

AGENT-REPORTED OUTPUT (untrusted — agents may claim success falsely; context only, never evidence):
{agent_block}

IMAGES ATTACHED (in send order):
{images_block}

LITERATURE GROUNDING:
{grounding_block}

YOUR TASK:
Judge whether the CLAIM is visibly TRUE or FALSE based solely on what you SEE in the attached image(s). Focus on scientific correctness and physical plausibility — not aesthetics.

GUIDANCE:
- "pass" = the figure(s) VISIBLY satisfy the claim. The structure/pattern/property the claim describes is clearly present and scientifically plausible.
- "fail" = the figure(s) VISIBLY contradict the claim. Something is wrong that a domain expert would immediately notice (impossible bond geometry, overlapping atoms, broken topology, physically nonsensical values or scale).
- "unsure" = the image resolution is too low, the relevant detail is ambiguous, or the figure type is unrecognizable. Default to "unsure" rather than guessing.

Return STRICT JSON only, no markdown, no prose outside the JSON:
{{"verdict": "pass"|"fail"|"unsure", "rationale": "<one-sentence explanation of what you saw that supports your verdict>"}}
"""

    def _call_vision_judge(
        self,
        uuid: str,
        agent_name: str,
        prompt: str,
        images: list[str],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Call a vision-capable LLM with an image-attached prompt.

        Uses ``self._vision_llm_config`` (configured in ``VerifierEvaluator``).
        Falls back gracefully when no vision model is configured.

        Args:
            uuid: Workflow identifier.
            agent_name: Short slug for the judge call (used in memory persistence).
            prompt: Text prompt to send alongside the image.
            images: List of base64 data URIs (e.g. ``"data:image/png;base64,..."``).

        Returns:
            Tuple ``(parsed_json_dict, error_string)``. One is always ``None``.
        """
        if not hasattr(self, "_vision_llm_config") or self._vision_llm_config is None:
            return None, "No vision model configured (set config.vision_judge_model)"

        # Lazy import: ``base`` pulls in the evaluator package at module load
        # time, so a top-level import here would create a circular import.
        from sources.evaluators.base import extract_json_payload

        try:
            # Build a multimodal message: text prompt + image(s)
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for img in images:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img, "detail": "high"},
                })

            # Create a fresh LLM provider for this call; it persists the
            # call in memory under ``<memory_dir>/<uuid>/<agent_name>.json``.
            memory_path = Path(self.memory_dir) / uuid
            memory_path.mkdir(parents=True, exist_ok=True)
            provider = LLMProvider(
                agent_name=agent_name,
                memory_path=str(memory_path),
                config=self._vision_llm_config,
            )
            raw = provider(content)

            parsed = extract_json_payload(raw or "")
            if not parsed:
                return None, f"vision judge returned non-JSON: {raw[:300]}"
            return json.loads(parsed), None

        except Exception as exc:
            self.logger.warning(f"Vision judge call failed for {uuid}/{agent_name}: {exc}")
            return None, str(exc)

    # ------------------------------------------------------------------
    # File selection (which workspace files the verifier should open)
    # ------------------------------------------------------------------

    def _llm_select_files(
        self,
        uuid: str,
        claim: dict[str, Any],
        execution_text: str,
        goal: str = "",
        max_files: int = 3,
    ) -> list[str]:
        """Pick workspace files most likely to hold this claim's artefact.

        The empty-selection / judge-error fallback never returns code files
        (``.py``, ``.R``, ``.jl``, ``.sh``, ``.ipynb``, ... or anything under
        ``src/``-like directories): a degenerate "everything" fallback made
        of workflow scripts points verifiers at source instead of artefacts.
        """
        eligible = self._eligible_workspace_files()
        if not eligible:
            return []
        # Fallback pool for empty/error selections: eligible MINUS code files.
        non_code = [f for f in eligible if not self._is_likely_code_file(f)]
        eligible_str = "\n".join(
            f"{f}\t{(self.workspace_dir / f).stat().st_size}B" for f in eligible
        )
        prompt = self._build_select_files_prompt(
            claim, execution_text, eligible_str, max_files, goal
        )
        cid = str(claim.get("id", "unknown"))
        parsed, err = self._call_judge_for_json(
            uuid, f"verifier_select_files_{cid}", prompt
        )
        if err or not isinstance(parsed, dict):
            self.logger.debug(f"file selection failed for {cid}: {err or 'non-dict JSON'}")
            return non_code
        selected = self._validate_workspace_paths(
            parsed.get("files"), allowed=set(eligible),
            max_count=max_files, label=cid,
        )
        return selected or non_code

    @staticmethod
    def _build_select_files_prompt(
        claim: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
        max_files: int,
        goal: str = "",
    ) -> str:
        """Build the per-claim file-selection prompt sent to the judge."""
        goal_block = (goal or "").strip()[:3000] or "(none provided)"
        return f"""You are picking which workspace files a deterministic verifier should open to check ONE atomic claim about a multi-agent workflow.

TASK GOAL (the user's specification — names the deliverables and their exact paths):
{goal_block}

WORKSPACE FILES (name<TAB>size, relative to workspace root, cwd at runtime):
{workspace_listing}

AGENT NARRATION (what the agents reported doing — untrusted; typically names the files they wrote, but they may be wrong):
{execution_text}

CLAIM TO CHECK:
- id:                       {claim.get('id')}
- description:              {claim.get('description')}
- expected_artifact_kind:   {claim.get('expected_artifact_kind') or '(unspecified)'}
- acceptable_variation:     {claim.get('acceptable_variation') or '(none)'}

Pick up to {max_files} paths from the WORKSPACE FILES listing whose contents are most likely to let a deterministic script verify this claim. Prefer RESULT artefacts — outputs, tables, figures, reports, manifests, logs — and files the agents explicitly mention writing for this artefact. Do NOT select workflow source files (`.py`, `.R`, `.jl`, `.sh`, notebooks): the verifier checks produced results, not script text. List nothing the workspace doesn't contain — never invent. If no workspace file plausibly holds the artefact, return an empty list.
Do not include any tests or debugging files that are unlikely to be part of the final artefact (e.g. "debug.log", "debug_2.py", "tmp_results.jsonl"). Focus on files that are central to the workflow's deliverable.

SELECTION PRIORITY:
- First prefer files that look like PRODUCED OUTPUTS of the workflow: paths under pred_results/, out/, output/, outputs/, results/, figures/, figs/, plots/, or matching a deliverable filename literally named in the TASK GOAL above.
- Only pick INPUT/DATASET-looking files (paths under data/, dataset/, datasets/, raw/, input/, inputs/) when the claim is explicitly about the input data itself.
- Select any logs files that are likely relevant (eg: train.log when the claim is about loss)
- The TASK GOAL is authoritative about what the deliverable is; the AGENT NARRATION is not.

Return STRICT JSON only:
  {{"files": ["<relative/path>", ...]}}
"""

    # ------------------------------------------------------------------
    # Verifier generation — ask the judge for a tiny Python script
    # ------------------------------------------------------------------

    def _generate_verifier(
        self,
        uuid: str,
        claim: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
        goal: str = "",
    ) -> dict[str, Any]:
        """Ask the judge for a verifier script (or a non-executable rationale)."""
        prompt = self._build_verifier_prompt(claim, workspace_listing, goal)
        return self._call_and_parse_verifier(uuid, claim, prompt, attempt=1)

    def _build_verifier_prompt(
        self, claim: dict[str, Any], workspace_listing: str, goal: str = ""
    ) -> str:
        """Build the verifier-generation prompt for one claim."""
        previews = self._render_relevant_previews(claim.get("likely_relevant_files", []))
        packages = ", ".join(_VERIFIER_BASE_PACKAGES)
        language = _detect_language(workspace_listing)
        return f"""
You are writing a tiny verifier program for ONE atomic claim from a multi-agent
workflow. The verifier will run inside the same workspace the agents used.

GOAL:
{goal or '(none provided)'}

The goal contains the user's specification: deliverables, paths, column names,
methodology requirements. Treat any literal identifiers in the goal as the GROUND-TRUTH SCHEMA.

WORKSPACE FILES (relative to workspace root, cwd at runtime):
{workspace_listing}

WORKFLOW LANGUAGE (heuristic from file extensions): {language}
- Regardless of language, verify against on-disk RESULT artefacts and the
  goal's ground-truth schema. Do NOT read or parse workflow source files
  (`.py`, `.R`, notebooks) as evidence — not with `ast`, not with regex.
- If no artefact or environment fact can decide the claim, return
  ``executable=false``.
- NEVER substitute a different data object for a claim's target. If the
  specific object a claim refers to (e.g. the training set actually consumed
  by the model, an intermediate table, a fitted model) is not present in the
  workspace as an artifact, treat the claim as NOT executable
  (``executable=false``) — do not approximate with a raw input file or any
  other stand-in.

RELEVANT FILE PREVIEWS (head + tail of files the claim depends on; truncated):
{previews}
FILE PREVIEWS have no authority over data schema to check when the goal specify expected input/output columns name.

CLAIM TO VERIFY:
- id: {claim['id']}
- importance: {claim.get('importance', self._DEFAULT_CLAIM_IMPORTANCE)} (1-10; 10 = literal deliverable)
- description: {claim['description']}
- likely_relevant_files: {claim.get('likely_relevant_files', [])}

RULES FOR YOUR SCRIPT:
{VERIFIER_PROMPT_RULES}

AVAILABLE IMPORTS: {packages}
Do not use any other imports, as the verifier will fail to run.

Return STRICT JSON only, in one of these two shapes:
  {{"executable": true,  "code": "<full python script as one string>"}}
  {{"executable": false, "reason": "<one sentence>"}}
"""

    def _call_and_parse_verifier(
        self,
        uuid: str,
        claim: dict[str, Any],
        prompt: str,
        attempt: int,
    ) -> dict[str, Any]:
        """Call the judge for a verifier spec; soft-fail to ``executable: False``."""
        suffix = "" if attempt == 1 else "_retry"
        agent_name = f"verifier_gen_{claim['id']}{suffix}"
        spec, err = self._call_judge_for_json(uuid, agent_name, prompt)
        if err is not None:
            return {"executable": False, "reason": err}
        if not isinstance(spec, dict):
            return {"executable": False, "reason": "verifier JSON not an object"}
        return spec

    # ------------------------------------------------------------------
    # Parallel pre-generation across claims (file selection + verifier spec)
    # ------------------------------------------------------------------

    def _select_files_and_generate_spec(
        self,
        uuid: str,
        claim: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
        goal: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pick relevant files then generate one claim's verifier spec.

        Visual (Source G) claims skip verifier-spec generation entirely: the
        generated script was always discarded by the visual branch, so the
        LLM call was pure waste. They still run file selection (capped at
        ``_MAX_VISUAL_IMAGES`` files) so the branch gets named candidates.
        """
        visual = self._is_visual_claim(claim)
        rel_files = self._llm_select_files(
            uuid, claim, execution_text, goal=goal,
            max_files=_MAX_VISUAL_IMAGES if visual else 3,
        )
        updated = {**claim, "likely_relevant_files": rel_files}
        if visual:
            spec = self._visual_claim_spec(claim) or {}
        else:
            spec = self._generate_verifier(
                uuid, updated, execution_text, workspace_listing, goal=goal
            )
        return updated, spec

    def _generate_specs_parallel(
        self,
        goal: str,
        uuid: str,
        claims: list[dict[str, Any]],
        execution_text: str,
        workspace_listing: str,
        max_workers: int,
    ) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        """Fan out file selection + verifier generation across claims via threads."""
        if not claims:
            return {}
        results: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        workers = max(1, min(len(claims), max_workers))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    self._select_files_and_generate_spec,
                    uuid, c, execution_text, workspace_listing, goal
                ): c["id"]
                for c in claims
            }
            for f in as_completed(futures):
                cid = futures[f]
                results[cid] = self._collect_spec_future(f, cid, claims)
        return results

    def _collect_spec_future(
        self,
        future: Future,
        cid: str,
        claims: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resolve one parallel-gen future; on raise return a non-executable fallback."""
        try:
            return future.result()
        except Exception as e:
            self.logger.warning(
                f"parallel spec generation failed for {cid}: {type(e).__name__}: {e}"
            )
            base = next(c for c in claims if c["id"] == cid)
            return (
                {**base, "likely_relevant_files": []},
                {"executable": False, "reason": f"spec generation raised: {e}"},
            )

    # ------------------------------------------------------------------
    # Verifier helper-package install (idempotent, lock-guarded)
    # ------------------------------------------------------------------

    def _ensure_verifier_packages(self) -> None:
        """Make verifier helper packages importable under ``sys.executable``."""
        global _VERIFIER_PACKAGES_INSTALLED
        if _VERIFIER_PACKAGES_INSTALLED:
            return
        with _VERIFIER_INSTALL_LOCK:
            if _VERIFIER_PACKAGES_INSTALLED:
                return
            _VERIFIER_PACKAGES_INSTALLED = self._install_verifier_helpers()

    def _install_verifier_helpers(self) -> bool:
        """Install base helper packages if missing; True iff importable after."""
        smoke_cmd = [sys.executable, "-c", "import " + ", ".join(_VERIFIER_BASE_IMPORTS)]
        if self._smoke_check(smoke_cmd):
            self.logger.info(
                f"Verifier helper packages already importable under {sys.executable}"
            )
            return True
        if not self._pip_install_base():
            return False
        if self._smoke_check(smoke_cmd):
            self.logger.info(
                f"Verifier helper packages ready: {list(_VERIFIER_BASE_PACKAGES)} "
                f"(via {sys.executable})"
            )
            return True
        self.logger.warning(
            "Verifier helper packages installed but not importable under "
            "sys.executable; scripts will see ImportError."
        )
        return False

    def _pip_install_base(self) -> bool:
        """Run ``pip install`` for the verifier base packages; True on rc==0."""
        cmd = [
            sys.executable, "-m", "pip", "install",
            *_PIP_INSTALL_FLAGS, *_VERIFIER_BASE_PACKAGES,
        ]
        r = self._run_pip(cmd, _BASE_INSTALL_TIMEOUT_SECONDS, list(_VERIFIER_BASE_PACKAGES))
        if r is None:
            return False
        if r.returncode != 0:
            tail = r.stderr.decode(errors="replace")[-_STDERR_TAIL_BYTES:]
            self.logger.warning(
                f"Verifier helper install failed (rc={r.returncode}); "
                f"scripts must restrict themselves to stdlib. stderr tail: {tail}"
            )
            return False
        return True

    @staticmethod
    def _smoke_check(cmd: list[str], timeout: float = _RUNNER_SMOKE_TIMEOUT) -> bool:
        """Return True iff *cmd* exits 0 within *timeout*."""
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except Exception:
            return False
        return r.returncode == 0

    # ------------------------------------------------------------------
    # Verifier execution (sandboxed inside the agents' workspace)
    # ------------------------------------------------------------------

    def _run_verifier(self, uuid: str, claim_id: str, code: str) -> dict[str, Any]:
        """Execute one verifier script and return the parsed/normalised result."""
        runner = self._build_runner(uuid)
        execution_id = f"verify_{claim_id}"
        thread_timeout = self.verifier_timeout + _RUNNER_EXTRA_TIMEOUT
        try:
            result = _run_coro_sync(
                lambda: runner.execute(code, execution_id=execution_id),
                thread_timeout=thread_timeout,
            )
        except TimeoutError as e:
            return self._error_result("timeout", f"verifier execution did not return in time: {e}")
        except Exception as e:
            return self._error_result("error", f"verifier execution raised: {type(e).__name__}: {e}")
        finally:
            self._safe_cleanup(runner)
        return self._finalize_run_result(result, claim_id)

    def _build_runner(self, uuid: str) -> WorkflowRunner:
        """Construct a sandboxed ``WorkflowRunner`` scoped to ``uuid``."""
        scratch = self._runner_temp_root / uuid
        scratch.mkdir(parents=True, exist_ok=True)
        cfg = RuntimeConfig(
            python_executable=sys.executable,
            timeout=self.verifier_timeout,
            temp_dir=scratch,
            requirements_file=None,
            use_pty=False,
        )
        return WorkflowRunner(cfg, execution_dir=str(self.workspace_dir))

    def _safe_cleanup(self, runner: WorkflowRunner) -> None:
        """Best-effort cleanup of *runner*; never raise."""
        try:
            _run_coro_sync(runner.cleanup, thread_timeout=_RUNNER_CLEANUP_TIMEOUT)
        except Exception as e:
            self.logger.debug(f"verifier runner cleanup failed: {e}")

    @staticmethod
    def _error_result(exit_tag: str, details: str) -> dict[str, Any]:
        """Shape an error exec_result with empty stdout/stderr."""
        return {
            "status": "error",
            "actual": None,
            "details": details,
            "raw_stdout": "",
            "raw_stderr": "",
            "exit_status": exit_tag,
        }

    def _finalize_run_result(
        self, result: ExecutionResult, claim_id: str
    ) -> dict[str, Any]:
        """Merge runner result fields onto the parsed stdout dict."""
        parsed = self._parse_verifier_stdout(result.stdout, claim_id)
        status = result.status
        parsed.update({
            "raw_stdout": result.stdout,
            "raw_stderr": result.stderr,
            "exit_status": status.value if isinstance(status, ExecutionStatus) else str(status),
        })
        if status == ExecutionStatus.TIMEOUT:
            parsed["status"] = "error"
            parsed["details"] = (parsed.get("details") or "") + (
                f" (script timed out after {self.verifier_timeout}s)"
            )
        elif status == ExecutionStatus.FAILED and parsed.get("status") not in ("pass", "fail"):
            parsed["status"] = "error"
            parsed["details"] = (parsed.get("details") or "") + (
                f" (script exit code {result.return_code})"
            )
        return parsed

    # ------------------------------------------------------------------
    # Bounded retry / recovery for verifier-side failures
    # ------------------------------------------------------------------

    def _run_verifier_with_recovery(
        self,
        uuid: str,
        claim: dict[str, Any],
        spec: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the verifier inside a bounded retry loop with error feedback.

        Up to ``_RECOVERY_MAX_ATTEMPTS`` total executions are allowed per
        claim. Every iteration classifies the failure (``_classify_exec_failure``)
        and applies a corrective action that FEEDS BACK what went wrong:

        * ``import`` — missing modules are parsed from the traceback, vetted
          once each through the LLM package check, and sandbox-installed. The
          vetting result is cached per module, failed installs are remembered,
          and modules that cannot be made importable are named in the next
          regeneration prompt as NOT available (``do NOT import``).
        * ``code_bug`` — the script is regenerated with the accumulated
          failure history (last tracebacks + actions already tried), an
          explicit instruction to avoid repeating the failing pattern, and
          the effective import allow-list (base + successfully installed).
        * untrustworthy printed verdicts — a pass/fail whose own stderr or
          details carry hard exception markers is demoted by
          ``_demote_pass_despite_error`` and rerouted through this loop.

        The loop stops early on the first trustworthy pass/fail. When the
        attempt budget is exhausted the last error is returned as-is
        (tagged ``recovery_exhausted``); the aggregator then scores it 0.0
        in the importance-weighted mean (see ``VerifierEvaluator._aggregate``)
        instead of excluding it — a workflow cannot raise its score by
        keeping its artefacts unparseable or slow.

        Returns:
            ``(final_spec, exec_result)`` where *exec_result* carries the
            telemetry keys ``recovery_attempts`` and ``recovery_actions``.
        """
        cid = claim["id"]
        actions: list[str] = []
        history: list[dict[str, Any]] = []
        vetted_modules: dict[str, list[str]] = {}
        unavailable_modules: set[str] = set()
        failed_installs: set[str] = set()
        installed_packages: set[str] = set()

        exec_result = self._run_verifier(uuid, cid, spec["code"])
        attempts = 1
        while True:
            guarded = self._demote_pass_despite_error(exec_result)
            if guarded is not exec_result:
                actions.append(
                    f"attempt {attempts}: demoted untrustworthy printed status "
                    f"(exception markers present); rerouting through recovery"
                )
                exec_result = guarded
            kind = self._classify_exec_failure(exec_result)
            if kind == "ok":
                break
            if attempts >= _RECOVERY_MAX_ATTEMPTS:
                actions.append(
                    f"gave up: recovery loop exhausted after {attempts} attempts"
                )
                exec_result = self._attach_giveup_reason(
                    exec_result,
                    f"recovery loop exhausted after {attempts} attempts",
                )
                break
            self.logger.info(
                f"[recovery {cid}] attempt {attempts}/{_RECOVERY_MAX_ATTEMPTS} "
                f"failed as {kind}; applying corrective action"
            )
            history.append(
                {
                    "attempt": attempts,
                    "kind": kind,
                    "stderr_tail": self._recovery_stderr_tail(exec_result),
                    "action": "",
                }
            )
            if kind == "import" and self._install_missing_packages(
                uuid, claim, exec_result,
                vetted_modules, failed_installs, installed_packages,
                unavailable_modules, actions, attempts,
            ):
                history[-1]["action"] = "installed missing packages; re-ran script"
                exec_result = self._run_verifier(uuid, cid, spec["code"])
                attempts += 1
                continue
            new_spec = self._regenerate_verifier_with_feedback(
                uuid, claim, spec, exec_result,
                history=history,
                unavailable_modules=sorted(unavailable_modules),
                extra_available=sorted(installed_packages),
            )
            if not (new_spec.get("executable") and new_spec.get("code")):
                reason = str(
                    new_spec.get("reason") or "regenerated spec missing code"
                )
                self.logger.info(
                    f"[recovery {cid}] regeneration produced no executable "
                    f"code: {reason}"
                )
                actions.append(
                    f"gave up: regeneration produced no executable code ({reason})"
                )
                exec_result = self._attach_giveup_reason(exec_result, reason)
                break
            history[-1]["action"] = "regenerated script with failure feedback"
            actions.append(f"attempt {attempts}: regenerated script")
            spec = new_spec
            exec_result = self._run_verifier(uuid, cid, spec["code"])
            attempts += 1

        exec_result["recovery_attempts"] = attempts
        exec_result["recovery_actions"] = actions
        exec_result["recovery_exhausted"] = exec_result.get("status") == "error"
        return spec, exec_result

    def _install_missing_packages(
        self,
        uuid: str,
        claim: dict[str, Any],
        exec_result: dict[str, Any],
        vetted_modules: dict[str, list[str]],
        failed_installs: set[str],
        installed_packages: set[str],
        unavailable_modules: set[str],
        actions: list[str],
        attempt: int,
    ) -> bool:
        """Vet and install the packages for *exec_result*'s missing modules.

        The LLM package-need call runs at most once per missing module
        (results cached in *vetted_modules*); packages whose install already
        failed are never retried inside this loop. Modules that cannot be
        made importable are added to *unavailable_modules* so the next
        regeneration prompt names them as forbidden imports.

        Returns:
            True when packages were installed and the caller should re-run
            the SAME script; False when the caller should regenerate instead.
        """
        cid = claim["id"]
        modules = self._missing_modules(exec_result)
        if not modules:
            return False
        stderr = exec_result.get("raw_stderr", "") or ""
        packages: list[str] = []
        for module in modules:
            if module in vetted_modules:
                packages.extend(vetted_modules[module])
                continue
            needed = self._llm_packages_needed_for_claim(uuid, claim, stderr)
            vetted_modules[module] = needed
            packages.extend(needed)
        installable = [
            p for p in packages
            if p.lower() not in failed_installs
            and p.lower() not in {i.lower() for i in installed_packages}
        ]
        if not installable:
            unavailable_modules.update(modules)
            self.logger.info(
                f"[recovery {cid}] no installable packages for {modules} "
                f"(vetted={packages or 'empty'}); regenerating without them"
            )
            return False
        if self._sandbox_install_packages(installable):
            installed_packages.update(installable)
            actions.append(
                f"attempt {attempt}: installed {installable}; re-running script"
            )
            return True
        failed_installs.update(p.lower() for p in installable)
        unavailable_modules.update(modules)
        actions.append(
            f"attempt {attempt}: install failed for {installable}; "
            f"marked unavailable"
        )
        return False

    @staticmethod
    def _missing_modules(exec_result: dict[str, Any]) -> list[str]:
        """Extract top-level module names from "No module named X" markers.

        Scans the run's stderr and details (the guard may have moved a
        caught exception into details) and returns unique top-level module
        names in first-seen order.
        """
        blob = "\n".join([
            exec_result.get("raw_stderr") or "",
            exec_result.get("details") or "",
        ])
        modules: list[str] = []
        for match in re.finditer(r"No module named '([^']+)'", blob):
            top = match.group(1).split(".")[0]
            if top and top not in modules:
                modules.append(top)
        return modules

    @staticmethod
    def _recovery_stderr_tail(exec_result: dict[str, Any]) -> str:
        """Return the last stderr lines for the failure-history feedback."""
        stderr = (exec_result.get("raw_stderr") or "").strip()
        if stderr:
            return "\n".join(stderr.splitlines()[-3:])
        return (exec_result.get("details") or "").strip()[:200]

    @classmethod
    def _demote_pass_despite_error(cls, exec_result: dict[str, Any]) -> dict[str, Any]:
        """Distrust a printed pass/fail that ships with hard exception markers.

        Two conservative channels, both modelled on the corpus:

        * the verifier's OWN ``raw_stderr`` contains an exception marker on a
          line the check did not quote in its ``details`` (quoted lines are
          the check legitimately reporting a target script's failure, e.g.
          "the workflow script exits non-zero");
        * the printed status is ``"pass"`` while ``details`` mention a
          load/import failure (the rdkit salvage pattern: unpickle failed,
          pickle GLOBAL opcodes inspected instead, pass printed anyway).

        A demoted result keeps its stdout/details for diagnosis but is
        forced to ``status="error"`` so the recovery loop retries it.

        Returns:
            The original dict when trustworthy, else a demoted copy.
        """
        if exec_result.get("status") not in ("pass", "fail"):
            return exec_result
        stderr = exec_result.get("raw_stderr") or ""
        details = exec_result.get("details") or ""
        details_norm = " ".join(details.split())
        channels: list[str] = []
        for line in stderr.splitlines():
            stripped = " ".join(line.split())
            if not stripped or stripped in details_norm:
                continue  # line the check itself quoted — not our crash
            if any(marker in stripped for marker in _RECOVERY_EXCEPTION_STDERR_MARKERS):
                channels.append("stderr")
                break
        if exec_result.get("status") == "pass" and any(
            marker in details for marker in _RECOVERY_SALVAGE_DETAIL_MARKERS
        ):
            channels.append("details")
        if not channels:
            return exec_result
        note = (
            f"(verifier guard: printed status '{exec_result.get('status')}' not "
            f"trusted — exception markers in {'+'.join(channels)}; "
            f"claim treated as unchecked)"
        )
        return {
            **exec_result,
            "status": "error",
            "details": f"{details} {note}".strip(),
        }

    @staticmethod
    def _attach_giveup_reason(
        exec_result: dict[str, Any], reason: str
    ) -> dict[str, Any]:
        """Return *exec_result* with status forced to error and *reason* appended."""
        details = (exec_result.get("details") or "").strip()
        suffix = f"(regeneration gave up: {reason})"
        return {
            **exec_result,
            "status": "error",
            "details": f"{details} {suffix}".strip(),
        }

    @classmethod
    def _classify_exec_failure(cls, exec_result: dict[str, Any]) -> str:
        """Bucket a run outcome as ``ok``, ``import``, or ``code_bug``."""
        if exec_result.get("status") != "error":
            return "ok"
        blob = (exec_result.get("raw_stderr") or "") + "\n" + (exec_result.get("details") or "")
        if any(marker in blob for marker in _RECOVERY_IMPORT_MARKERS):
            return "import"
        return "code_bug"

    def _llm_packages_needed_for_claim(
        self,
        uuid: str,
        claim: dict[str, Any],
        stderr: str,
    ) -> list[str]:
        """Ask the judge which missing pip packages are genuinely required."""
        prompt = self._build_package_check_prompt(claim, stderr)
        parsed, err = self._call_judge_for_json(
            uuid, f"verifier_pkg_check_{claim['id']}", prompt
        )
        if err is not None or not isinstance(parsed, dict):
            self.logger.debug(
                f"package-need check failed for {claim['id']}: "
                f"{err or 'non-dict JSON'}"
            )
            return []
        raw = parsed.get("packages")
        return self._clean_package_list(raw) if isinstance(raw, list) else []

    @staticmethod
    def _build_package_check_prompt(claim: dict[str, Any], stderr: str) -> str:
        """Build the package-need check prompt sent to the judge."""
        lines = (stderr or "").splitlines()[-_RECOVERY_STDERR_FEEDBACK_LINES:]
        tail = "\n".join(lines) or "(no traceback available)"
        base_pkgs = ", ".join(_VERIFIER_BASE_PACKAGES)
        return f"""
A verifier program for ONE atomic claim crashed because it tried to import an unavailable Python package.

CLAIM:
- id: {claim['id']}
- description: {claim.get('description', '')}

TRACEBACK (last {_RECOVERY_STDERR_FEEDBACK_LINES} lines):
{tail}

ALREADY AVAILABLE (do NOT list these): {base_pkgs}, plus the Python standard library.

QUESTION: are the missing packages STRICTLY required to verify this claim, or could the verifier be rewritten in pure Python using only the available imports?

Return STRICT JSON only:
  {{"packages": ["<pip name>", ...]}}

Rules:
- Empty list ({{"packages": []}}) iff the claim can be verified without third-party packages.
- Use pip-install names (``scikit-learn``, not ``sklearn``; ``Pillow``, not ``PIL``).
- Never list a package that is already available.
- At most {_RECOVERY_MAX_INSTALL_PACKAGES} entries.
"""

    @staticmethod
    def _clean_package_list(raw: list[Any]) -> list[str]:
        """Dedupe and cap the LLM-returned package list; drop already-available ones."""
        already = {p.lower() for p in _VERIFIER_BASE_PACKAGES}
        seen: set[str] = set()
        cleaned: list[str] = []
        for entry in raw:
            if not isinstance(entry, str):
                continue
            name = entry.strip()
            if not name or name.lower() in already or name.lower() in seen:
                continue
            seen.add(name.lower())
            cleaned.append(name)
            if len(cleaned) >= _RECOVERY_MAX_INSTALL_PACKAGES:
                break
        return cleaned

    def _sandbox_install_packages(self, packages: list[str]) -> bool:
        """Pip-install *packages* under ``sys.executable``; True iff rc==0."""
        if not packages:
            return False
        cmd = [sys.executable, "-m", "pip", "install", *_PIP_INSTALL_FLAGS, *packages]
        r = self._run_pip(cmd, _RECOVERY_INSTALL_TIMEOUT_SECONDS, packages)
        if r is None:
            return False
        if r.returncode != 0:
            tail = r.stderr.decode(errors="replace")[-_STDERR_TAIL_BYTES:]
            self.logger.warning(
                f"sandbox install failed (rc={r.returncode}) for {packages}; "
                f"stderr tail: {tail}"
            )
            return False
        self.logger.info(f"sandbox-installed verifier packages: {packages}")
        return True

    def _run_pip(
        self,
        cmd: list[str],
        timeout: int,
        packages: list[str],
    ) -> subprocess.CompletedProcess | None:
        """Wrap ``subprocess.run`` for pip; log and return ``None`` on failure."""
        try:
            return subprocess.run(cmd, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            self.logger.warning(f"sandbox install timed out after {timeout}s for {packages}")
        except FileNotFoundError as e:
            self.logger.warning(f"pip not found for verifier helper install: {e}")
        except Exception as e:
            self.logger.warning(f"sandbox install raised: {type(e).__name__}: {e}")
        return None

    def _regenerate_verifier_with_feedback(
        self,
        uuid: str,
        claim: dict[str, Any],
        prev_spec: dict[str, Any],
        exec_result: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
        unavailable_modules: list[str] | tuple[str, ...] = (),
        extra_available: list[str] | tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Ask the judge to fix the previous script given the traceback.

        Args:
            uuid: Workflow identifier.
            claim: The claim dict.
            prev_spec: The spec whose script just failed.
            exec_result: The failed run's result dict.
            history: Accumulated recovery attempts (kind, stderr tail,
                action) so the LLM avoids repeating failed patterns.
            unavailable_modules: Modules proven NOT installable in this
                environment; the prompt forbids importing them.
            extra_available: Packages successfully installed during earlier
                loop iterations, added to the allowed import list.

        Returns:
            New verifier spec dict (``executable`` + ``code``) or a
            non-executable fallback with a ``reason``.
        """
        prompt = self._build_regen_prompt(
            claim, prev_spec, exec_result,
            history=history,
            unavailable_modules=unavailable_modules,
            extra_available=extra_available,
        )
        return self._call_and_parse_verifier(uuid, claim, prompt, attempt=2)

    def _build_regen_prompt(
        self,
        claim: dict[str, Any],
        prev_spec: dict[str, Any],
        exec_result: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
        unavailable_modules: list[str] | tuple[str, ...] = (),
        extra_available: list[str] | tuple[str, ...] = (),
    ) -> str:
        """Build the verifier-regeneration prompt fed with the failure history.

        Beyond the previous traceback, the prompt carries the accumulated
        recovery history, the modules that are NOT available (and must not
        be imported), and the effective import allow-list (base packages
        plus anything installed during the loop).
        """
        prev_code = prev_spec.get("code", "")
        stderr_lines = (exec_result.get("raw_stderr") or "").splitlines()[
            -_RECOVERY_STDERR_FEEDBACK_LINES:
        ]
        stderr_tail = "\n".join(stderr_lines) or "(no stderr captured)"
        details = exec_result.get("details", "") or ""
        previews = self._render_relevant_previews(claim.get("likely_relevant_files", []))
        available = list(_VERIFIER_BASE_PACKAGES) + list(extra_available)
        packages = ", ".join(available)
        history_block = _render_recovery_history(history)
        unavailable_block = (
            f"""
UNAVAILABLE PACKAGES: {', '.join(unavailable_modules)} — these are NOT
available in the verifier environment and could not be installed. Do NOT
import them. Verify the claim another way with the preinstalled stack above,
or emit status="fail"/"error" if the claim cannot be checked without them.
Never inspect metadata (pickle GLOBAL opcodes, file headers, filenames) as a
substitute for loading the artefact with the available libraries.
"""
            if unavailable_modules
            else ""
        )
        return f"""
Your previous verifier script for ONE atomic claim crashed at runtime. Fix it and resubmit the FULL corrected script.

CLAIM:
- id: {claim['id']}
- importance: {claim.get('importance', self._DEFAULT_CLAIM_IMPORTANCE)} (1-10; 10 = literal deliverable)
- description: {claim['description']}
- likely_relevant_files: {claim.get('likely_relevant_files', [])}

RELEVANT FILE PREVIEWS:
{previews}

PREVIOUS SCRIPT (do not repeat its mistake):
{prev_code}

RUNTIME ERROR DETAILS: {details}

STDERR (last {_RECOVERY_STDERR_FEEDBACK_LINES} lines):
{stderr_tail}
{history_block}{unavailable_block}
AVAILABLE IMPORTS:
{packages}.
Do NOT introduce any other third-party imports.

INSTRUCTIONS:
{VERIFIER_PROMPT_RULES}
{RECOVERY_PROMPT_RULES}

Return STRICT JSON only:
  {{"executable": true, "code": "<full corrected python script as one string>"}}
"""

    # ------------------------------------------------------------------
    # Stdout parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_verifier_stdout(stdout: str, claim_id: str) -> dict[str, Any]:
        """Pull the last JSON line matching ``claim_id`` from script stdout."""
        if not stdout:
            return {"status": "error", "actual": None, "details": "no stdout from verifier"}
        for line in reversed(stdout.splitlines()):
            obj = _VerifierPerClaimMixin._try_json(line.strip())
            if isinstance(obj, dict) and obj.get("claim_id") == claim_id:
                return _VerifierPerClaimMixin._normalise_status(obj)
        return {"status": "error", "actual": None, "details": "no matching JSON line in verifier stdout"}

    @staticmethod
    def _try_json(line: str) -> Any:
        """Return ``json.loads(line)`` or ``None`` when not a JSON object."""
        if not line.startswith("{"):
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalise_status(obj: dict[str, Any]) -> dict[str, Any]:
        """Reduce a verifier-line dict to the canonical (status, actual, details) shape."""
        status = obj.get("status")
        if status not in ("pass", "fail", "error"):
            return {
                "status": "error",
                "actual": obj.get("actual"),
                "details": f"unrecognised status '{status}' from verifier",
            }
        return {
            "status": status,
            "actual": obj.get("actual"),
            "details": str(obj.get("details", "")),
        }

    # ------------------------------------------------------------------
    # Per-claim scoring
    # ------------------------------------------------------------------

    def _score_executable(
        self,
        claim: dict[str, Any],
        spec: dict[str, Any],
        exec_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert an executable-verifier run into a scored result dict."""
        status = exec_result.get("status")
        return {
            "score": 1.0 if status == "pass" else 0.0,
            "verifier_kind": "executable",
            "status": status,
            "actual": exec_result.get("actual"),
            "details": exec_result.get("details", ""),
            "raw_stdout": exec_result.get("raw_stdout", ""),
            "raw_stderr": exec_result.get("raw_stderr", ""),
            "exit_status": exec_result.get("exit_status", ""),
            "recovery_attempts": exec_result.get("recovery_attempts", 1),
            "recovery_actions": exec_result.get("recovery_actions", []),
        }

    _SOFT_VERDICT_SCORE = {"pass": 1.0, "unsure": 0.5, "fail": 0.0}

    def _score_soft(
        self,
        uuid: str,
        claim: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
        reason: str,
        grounding: str = "",
    ) -> dict[str, Any]:
        """Narrow LLM verdict for one non-executable claim, anchored on grounding."""
        prompt = self._build_soft_check_prompt(
            claim, execution_text, workspace_listing, reason, grounding
        )
        data, err = self._call_judge_for_json(
            uuid, f"verifier_soft_{claim['id']}", prompt
        )
        if err is not None or not isinstance(data, dict):
            return {
                "score": 0.0, "verifier_kind": "soft", "status": "error",
                "details": f"soft check failed: {err or 'verdict JSON not an object'}",
                "rationale": "",
            }
        verdict = data.get("verdict", "unsure")
        rationale = str(data.get("rationale", ""))
        return {
            "score": self._SOFT_VERDICT_SCORE.get(verdict, 0.5),
            "verifier_kind": "soft",
            "status": verdict,
            "details": rationale,
            "rationale": rationale,
        }

    def _build_soft_check_prompt(
        self,
        claim: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
        reason: str,
        grounding: str,
    ) -> str:
        """Build the soft-check (non-executable verdict) prompt."""
        grounding_block = grounding.strip() if grounding else "(no literature grounding available)"
        return f"""
You are checking ONE claim from a multi-agent workflow. The claim is not
executable in code; please judge it against the concrete context below.

WORKSPACE FILES:
{workspace_listing}

LITERATURE GROUNDING (peer-reviewed evidence relevant to this task):
{grounding_block}

WORKFLOW OUTPUT (context only):
{execution_text}

CLAIM:
- id: {claim['id']}
- importance: {claim.get('importance', self._DEFAULT_CLAIM_IMPORTANCE)} (1-10; 10 = literal deliverable)
- description: {claim['description']}
- likely_relevant_files: {claim.get('likely_relevant_files', [])}

REASON IT WAS MARKED NON-EXECUTABLE:
{reason or '(none)'}

Answer ONLY this question: given the workspace, output, and literature
grounding above, does the claim hold? Use one of three verdicts:
- "pass"   : the claim is well supported by the visible context AND
             consistent with the literature grounding (when applicable).
- "unsure" : context is insufficient to decide either way, or the
             literature gives no clear guidance.
- "fail"   : the claim is contradicted by the workspace context OR by the
             literature grounding.

If the literature grounding is missing or marked as unavailable, fall back
to judging against the workspace context alone — do not penalise the claim
for the absence of grounding.

Return STRICT JSON: {{"verdict": "pass" | "unsure" | "fail", "rationale": "<one sentence>"}}
"""
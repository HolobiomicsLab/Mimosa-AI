"""Per-claim verifier generation, sandboxed execution and scoring."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    )

from sources.cli.pretty_print import CYAN, DIM, GREEN, RED, YELLOW, print_box
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
_RECOVERY_MAX_INSTALL_PACKAGES = 6
_RECOVERY_INSTALL_TIMEOUT_SECONDS = 300
_RECOVERY_STDERR_FEEDBACK_LINES = 30
# Substrings that identify a missing-dependency stderr; anything else is
# treated as a generated-code bug and routed through script regeneration.
_RECOVERY_IMPORT_MARKERS: tuple[str, ...] = (
    "ModuleNotFoundError",
    "ImportError: No module named",
    "ImportError: cannot import name",
)
_BASE_INSTALL_TIMEOUT_SECONDS = 600
_RUNNER_CLEANUP_TIMEOUT = 15
_RUNNER_SMOKE_TIMEOUT = 15.0
_RUNNER_EXTRA_TIMEOUT = 10
_PIP_INSTALL_FLAGS: tuple[str, ...] = (
    "--quiet", "--disable-pip-version-check", "--break-system-packages",
)
_PRINT_TRUNCATE_BYTES = 256
_STDERR_TAIL_BYTES = 400

VERIFIER_PROMPT_RULES = """
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
  operand groups are drawn from the real data distribution: reject
  groups where values are sentinel/masked placeholders (commonly
  -999, -9999, 9999, NaN, None, or values equal to a constant across
  the entire group when the data is supposed to be continuous). If
  sentinel/masked values are present in either operand, emit
  ``status="fail"`` with a details string naming the sentinel — the
  check is unsound, not satisfied.
- Catch your own exceptions and emit status="error" with the error message in
  details — never let the script raise.
- Parse the relevant files according to the format visible in the PREVIEWS
  above. Do not invent a different format. If the previews include a header
  line (e.g. "Minimum Energy: -6") your parser must skip it gracefully.
- If the previews are empty or do not show enough of the file to be sure of
  the format, prefer permissive parsing (try several reasonable splits, skip
  unparseable lines) over a strict format that may misjudge the file.
- When the claim is about CODE STRUCTURE in a workflow script (imports,
  function calls, class instantiations, assignments), parse the script
  with the stdlib ``ast`` module instead of regex or substring search.
  Variable names are not load-bearing — never hard-code identifiers like
  ``rf_full``, ``final_model``, ``train_df``. Match on the call target
  (``ast.Call.func``: e.g. node is a ``Name`` with id
  ``"RandomForestRegressor"`` or an ``Attribute`` ending in ``.fit``),
  on the imported symbol (``ast.ImportFrom.module`` / ``.names[*].name``),
  or on the attribute path. Walk with ``ast.walk(tree)``. Regex on
  source code is brittle to whitespace, quote style, line breaks, and
  renames; reserve ``re`` for unstructured text (logs, READMEs).
- For claims that an output FILE or PATH exists (e.g. "the predictions
  CSV is at ``<exact path>``", "the deliverable file ``X`` exists"),
  the primary check is ``pathlib.Path(target).exists()`` evaluated in
  the workspace cwd. If the file is there and parseable, that alone is
  sufficient to emit ``status="pass"``. Do NOT additionally require the
  source script to contain the literal path string — quote style,
  ``os.path.join`` splits, and variable substitution will hide it.
  Inspect source code only when the file is ABSENT and you need to
  attribute the failure.
- Some claims are conditional ("if X happens, Y must hold" / "no
  fallback to Z used instead of W"). Detect the antecedent first. If
  it is FALSE — the guarded path is not present in the workspace —
  emit ``status="pass"`` with ``details="vacuously satisfied:
  <antecedent> not present"``. Do not search unrelated regions of the
  script for the consequent's keywords.
- On a "Used fallback claim", the score is inverted: 0 if the claim passes, 1 if it fails. This is to incentivize the verified program to not use fallback

What should not be done:
- Do not Embeds the workflow output, the agent's final answer, or any large
  fragment thereof as a string literal and then parses that literal. This
  is a tautology: comparing the answer to itself proves nothing.
- Do not Hard-codes the expected value (e.g. ``status == "SUCCESS"`` against an
  inlined JSON blob) instead of recomputing it from workspace files.
- Do not Returns "pass" without ever opening a file or running a real computation
  derived from on-disk state.
- Do not Declares ``likely_relevant_files`` but performs no file I/O.
- Do not Check against hard-coded value (string or numerical) that was not explicitly given in the claim description as the target to check against.

What a legitimate verifier does:
- Opens the file(s) in ``likely_relevant_files`` from the workspace cwd.
- Re-derives the value the claim asserts (recompute the energy, recount the
  contacts, re-walk the chain, re-read the metric).
- Compares the recomputed value to the small target taken from the claim
  description (e.g. "-6", "20 residues") — targets are short numeric/string
  constants, not embedded answer payloads.

If the claim cannot be checked deterministically with code (e.g. it concerns
the rigor of a proof, the appropriateness of a binning choice, the
defensibility of a conclusion), set "executable": false and explain briefly.

Don't forget to include the library you need such as json, numpy, etc..
You can use library from the standard library and the available imports.
"""

RECOVERY_PROMPT_RULES = """
- Diagnose the failure from the traceback above and emit a corrected script.
- Keep the output contract: print EXACTLY ONE JSON line to stdout shaped
  {{"claim_id": "<id>", "status": "pass"|"fail"|"error", "actual": <value or null>, "details": "<short string>"}}.
- Catch your own exceptions inside the script and emit status="error" — never let the script raise.
- Read files with relative paths (cwd is the workspace).
- Guard against vacuous comparisons. When a property reduces to a
  comparison of order statistics across two groups (e.g. "all of A >
  all of B" becoming ``min(A) > max(B)``), first check that BOTH
  operand groups are drawn from the real data distribution: reject
  groups where values are sentinel/masked placeholders (commonly
  -999, -9999, 9999, NaN, None, or values equal to a constant across
  the entire group when the data is supposed to be continuous). If
  sentinel/masked values are present in either operand, emit
  ``status="fail"`` with a details string naming the sentinel — the
  check is unsound, not satisfied.
- When the claim is about CODE STRUCTURE in a workflow script (imports,
  function calls, class instantiations, assignments), parse the script
  with the stdlib ``ast`` module instead of regex or substring search.
  Variable names are not load-bearing — never hard-code identifiers like
  ``rf_full``, ``final_model``, ``train_df``. Match on the call target
  (``ast.Call.func``: e.g. node is a ``Name`` with id
  ``"RandomForestRegressor"`` or an ``Attribute`` ending in ``.fit``),
  on the imported symbol (``ast.ImportFrom.module`` / ``.names[*].name``),
  or on the attribute path. Walk with ``ast.walk(tree)``. Regex on
  source code is brittle to whitespace, quote style, line breaks, and
  renames; reserve ``re`` for unstructured text (logs, READMEs).
- For claims that an output FILE or PATH exists (e.g. "the predictions
  CSV is at ``<exact path>``", "the deliverable file ``X`` exists"),
  the primary check is ``pathlib.Path(target).exists()`` evaluated in
  the workspace cwd. If the file is there and parseable, that alone is
  sufficient to emit ``status="pass"``. Do NOT additionally require the
  source script to contain the literal path string — quote style,
  ``os.path.join`` splits, and variable substitution will hide it.
  Inspect source code only when the file is ABSENT and you need to
  attribute the failure.
- Some claims are conditional ("if X happens, Y must hold" / "no
  fallback to Z used instead of W"). Detect the antecedent first. If
  it is FALSE — the guarded path is not present in the workspace —
  emit ``status="pass"`` with ``details="vacuously satisfied:
  <antecedent> not present"``. Do not search unrelated regions of the
  script for the consequent's keywords.
- If the previous failure was an ImportError, rewrite without that package using the available imports and the standard library.
"""


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
    ) -> dict[str, Any]:
        """Generate, execute and score a single claim; return the result dict."""
        t_start = time.time()
        claim = self._resolve_relevant_files(
            uuid, claim, execution_text, workspace_listing, preloaded_spec
        )
        self._print_claim_header(claim)
        spec = preloaded_spec or self._generate_verifier(
            uuid, claim, execution_text, workspace_listing
        )
        spec, scored = self._run_and_score(
            uuid, claim, spec, execution_text, workspace_listing, grounding
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
    ) -> dict[str, Any]:
        """Populate ``likely_relevant_files`` via judge call when not anchored."""
        if preloaded_spec is not None:
            return claim
        rel_files = self._llm_select_files(
            uuid, claim, execution_text
        )
        return {**claim, "likely_relevant_files": rel_files}

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
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Dispatch to executable or soft branch; return ``(final_spec, scored)``."""
        if spec.get("executable") and spec.get("code"):
            spec, exec_result = self._run_verifier_with_recovery(
                uuid, claim, spec, execution_text, workspace_listing
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
    # File selection (which workspace files the verifier should open)
    # ------------------------------------------------------------------

    def _llm_select_files(
        self,
        uuid: str,
        claim: dict[str, Any],
        execution_text: str,
        max_files: int = 3,
    ) -> list[str]:
        """Pick workspace files most likely to hold this claim's artefact."""
        eligible = self._eligible_workspace_files()
        if not eligible:
            return []
        eligible_str = "\n".join(
            f"{f}\t{(self.workspace_dir / f).stat().st_size}B" for f in eligible
        )
        prompt = self._build_select_files_prompt(
            claim, execution_text, eligible_str, max_files
        )
        cid = str(claim.get("id", "unknown"))
        parsed, err = self._call_judge_for_json(
            uuid, f"verifier_select_files_{cid}", prompt
        )
        if err or not isinstance(parsed, dict):
            self.logger.debug(f"file selection failed for {cid}: {err or 'non-dict JSON'}")
            return eligible
        selected = self._validate_workspace_paths(
            parsed.get("files"), allowed=set(eligible),
            max_count=max_files, label=cid,
        )
        return selected or eligible

    @staticmethod
    def _build_select_files_prompt(
        claim: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
        max_files: int,
    ) -> str:
        """Build the per-claim file-selection prompt sent to the judge."""
        return f"""You are picking which workspace files a deterministic verifier should open to check ONE atomic claim about a multi-agent workflow.

WORKSPACE FILES (name<TAB>size, relative to workspace root, cwd at runtime):
{workspace_listing}

AGENT NARRATION (what the agents reported doing — typically names the files they wrote):
{execution_text}

CLAIM TO CHECK:
- id:                       {claim.get('id')}
- description:              {claim.get('description')}
- expected_artifact_kind:   {claim.get('expected_artifact_kind') or '(unspecified)'}
- acceptable_variation:     {claim.get('acceptable_variation') or '(none)'}

Pick up to {max_files} paths from the WORKSPACE FILES listing whose contents are most likely to let a deterministic script verify this claim. Prefer files the agents explicitly mention writing for this artefact. List nothing the workspace doesn't contain — never invent. If no workspace file plausibly holds the artefact, return an empty list.
Do not include any tests or debugging files that are unlikely to be part of the final artefact (e.g. "debug.log", "debug_2.py", "tmp_results.jsonl"). Focus on files that are central to the workflow's deliverable.

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
    ) -> dict[str, Any]:
        """Ask the judge for a verifier script (or a non-executable rationale)."""
        prompt = self._build_verifier_prompt(claim, workspace_listing)
        return self._call_and_parse_verifier(uuid, claim, prompt, attempt=1)

    def _build_verifier_prompt(
        self, claim: dict[str, Any], workspace_listing: str
    ) -> str:
        """Build the verifier-generation prompt for one claim."""
        previews = self._render_relevant_previews(claim.get("likely_relevant_files", []))
        packages = ", ".join(_VERIFIER_BASE_PACKAGES)
        language = _detect_language(workspace_listing)
        return f"""
You are writing a tiny verifier program for ONE atomic claim from a multi-agent
workflow. The verifier will run inside the same workspace the agents used.

WORKSPACE FILES (relative to workspace root, cwd at runtime):
{workspace_listing}

WORKFLOW LANGUAGE (heuristic from file extensions): {language}
- When ``python``: parse workflow scripts with the stdlib ``ast`` module.
- When ``r``: the workflow scripts are R; do NOT parse them with ``ast``.
  Verify against on-disk artefacts; if a deterministic check on artefacts is
  not possible for a code-structure claim, return ``executable=false``.
- When ``mixed`` or ``unknown``: return ``executable=false``; the verifier cannot assume parsing strategy.

RELEVANT FILE PREVIEWS (head + tail of files the claim depends on; truncated):
{previews}

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
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pick relevant files then generate one claim's verifier spec."""
        rel_files = self._llm_select_files(
            uuid, claim, execution_text
        )
        updated = {**claim, "likely_relevant_files": rel_files}
        spec = self._generate_verifier(uuid, updated, execution_text, workspace_listing)
        return updated, spec

    def _generate_specs_parallel(
        self,
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
                    uuid, c, execution_text, workspace_listing,
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
        spec: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the verifier with one corrective retry on verifier-side failures.

        Import errors trigger an LLM package check + sandbox install. Code bugs
        (or empty install lists) trigger a single regeneration with traceback
        feedback. Failures still present after the retry are returned as-is;
        the aggregator excludes them from the importance-weighted mean.
        """
        cid = claim["id"]
        exec_result = self._run_verifier(uuid, cid, spec["code"])
        kind = self._classify_exec_failure(exec_result)
        if kind == "ok":
            return spec, exec_result
        self.logger.info(
            f"[recovery {cid}] initial run errored as {kind}; "
            f"attempting one corrective action"
        )
        if kind == "import":
            recovered = self._try_install_and_rerun(uuid, claim, spec, exec_result)
            if recovered is not None:
                return spec, recovered
        return self._regenerate_and_rerun(
            uuid, claim, spec, exec_result, execution_text, workspace_listing
        )

    def _try_install_and_rerun(
        self,
        uuid: str,
        claim: dict[str, Any],
        spec: dict[str, Any],
        exec_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Install LLM-vetted packages and re-run; ``None`` falls through to regen."""
        cid = claim["id"]
        stderr = exec_result.get("raw_stderr", "") or ""
        packages = self._llm_packages_needed_for_claim(uuid, claim, stderr)
        if not packages or not self._sandbox_install_packages(packages):
            self.logger.info(
                f"[recovery {cid}] no installable packages "
                f"(LLM returned {packages or 'empty'}); regenerating instead"
            )
            return None
        self.logger.info(f"[recovery {cid}] installed {packages}; re-running script")
        return self._run_verifier(uuid, cid, spec["code"])

    def _regenerate_and_rerun(
        self,
        uuid: str,
        claim: dict[str, Any],
        spec: dict[str, Any],
        exec_result: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Regenerate the script once with traceback feedback and re-run it."""
        cid = claim["id"]
        new_spec = self._regenerate_verifier_with_feedback(
            uuid, claim, spec, exec_result, execution_text, workspace_listing
        )
        if not (new_spec.get("executable") and new_spec.get("code")):
            reason = str(new_spec.get("reason") or "regenerated spec missing code")
            self.logger.info(
                f"[recovery {cid}] regeneration produced no executable code: {reason}"
            )
            return spec, self._attach_giveup_reason(exec_result, reason)
        retry = self._run_verifier(uuid, cid, new_spec["code"])
        return new_spec, retry

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
        execution_text: str,
        workspace_listing: str,
    ) -> dict[str, Any]:
        """Ask the judge to fix the previous script given the traceback."""
        prompt = self._build_regen_prompt(claim, prev_spec, exec_result)
        return self._call_and_parse_verifier(uuid, claim, prompt, attempt=2)

    def _build_regen_prompt(
        self,
        claim: dict[str, Any],
        prev_spec: dict[str, Any],
        exec_result: dict[str, Any],
    ) -> str:
        """Build the verifier-regeneration prompt fed with the previous traceback."""
        prev_code = prev_spec.get("code", "")
        stderr_lines = (exec_result.get("raw_stderr") or "").splitlines()[
            -_RECOVERY_STDERR_FEEDBACK_LINES:
        ]
        stderr_tail = "\n".join(stderr_lines) or "(no stderr captured)"
        details = exec_result.get("details", "") or ""
        previews = self._render_relevant_previews(claim.get("likely_relevant_files", []))
        packages = ", ".join(_VERIFIER_BASE_PACKAGES)
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


AVAILABLE IMPORTS:
{packages}.
Do NOT introduce any other third-party imports.

INSTRUCTIONS:
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
"""
Per-claim verifier generation, sandboxed execution and scoring.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sources.cli.pretty_print import (
    CYAN,
    DIM,
    GREEN,
    RED,
    YELLOW,
    print_box,
)
from sources.core.workflow_runner import (
    ExecutionStatus,
    RuntimeConfig,
    WorkflowRunner,
)


# ----- Verifier helper packages ----------------------------------------------
# Installed once per process so verifier scripts can rely on them being
# importable. Kept deliberately minimal: numerical + tabular + classical stats
# + standard ML primitives. Anything heavier should be inferred from the
# workflow's own declared dependencies, not bolted onto the verifier.
_VERIFIER_BASE_PACKAGES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "pint",
    "pydantic",
    "pandera",
    "jsonschema",
    "sympy"
)

# Python module names corresponding to ``_VERIFIER_BASE_PACKAGES``
# (scikit-learn → sklearn). Used by the post-install smoke check.
_VERIFIER_BASE_IMPORTS: tuple[str, ...] = ("numpy", "pandas", "scipy", "sklearn")

_VERIFIER_PACKAGES_INSTALLED = False
_VERIFIER_INSTALL_LOCK = threading.Lock()


T = TypeVar("T")


def _run_coro_sync(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],
    thread_timeout: float | None = None,
) -> T:
    """Run an async coroutine from sync code, even if a loop is already running.

    The coroutine is built lazily so it can never be orphaned on a failed run.

    Args:
        coro_factory: Zero-arg callable that constructs the coroutine to await.
        thread_timeout: When already inside a running loop, time budget in
            seconds for the worker thread to finish before raising.

    Returns:
        The value returned by the coroutine.

    Raises:
        TimeoutError: When the worker thread exceeds ``thread_timeout``.
        BaseException: Re-raises any exception raised by the coroutine.
    """
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if not in_loop:
        return asyncio.run(coro_factory())

    holder: dict[str, Any] = {}

    def _target() -> None:
        """Run the coroutine inside a worker thread and stash result/error."""
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
        """Generate, execute (if executable) and score a single claim.

        Args:
            uuid: Workflow identifier (used for judge calls and logs).
            claim: Normalised claim dict with id, importance, description.
            execution_text: Agent narration / produced output text.
            workspace_listing: Rendered listing of workspace files.
            grounding: Optional peer-reviewed literature grounding block.
            preloaded_spec: Pre-built verifier spec from a lineage anchor.
                When provided, both the file-selection LLM call and the
                verifier-generation LLM call are skipped.

        Returns:
            Scored claim dict including verifier spec, status, score, and the
            ``elapsed_s`` wall-clock spent in this call (used by the
            orchestrator to render a per-claim timing table).
        """
        t_start = time.time()
        if preloaded_spec is None:
            rel_files = self._llm_select_files(
                uuid, claim, execution_text, workspace_listing
            )
            claim = {**claim, "likely_relevant_files": rel_files}
        else:
            rel_files = list(claim.get("likely_relevant_files") or [])
        claim_text = (
            f"id:          {claim.get('id')}\n"
            f"importance:  {claim.get('importance')}\n"
            f"description: {claim.get('description')}\n"
            f"files:       {rel_files if rel_files else '(none)'}"
        )
        print_box(claim_text, title=f"Verifying claim {claim.get('id')}", color=CYAN)

        spec = (
            preloaded_spec
            if preloaded_spec is not None
            else self._generate_verifier(uuid, claim, execution_text, workspace_listing)
        )

        if spec.get("executable") and spec.get("code"):
            code = spec["code"]
            exec_result = self._run_verifier(uuid, claim["id"], code)

            exit_status = exec_result.get("exit_status", "?")
            run_status = exec_result.get("status", "?")
            details = exec_result.get("details", "") or ""
            stdout = exec_result.get("raw_stdout", "") or ""
            stderr = exec_result.get("raw_stderr", "") or ""

            run_color = GREEN if run_status == "pass" else (YELLOW if run_status == "fail" else RED)
            summary = (
                f"status:      {run_status}\n"
                f"exit_status: {exit_status}\n"
                f"details:     {details}"
            )
            print_box(summary, title=f"Verifier run · {claim.get('id')}", color=run_color)

            if stdout.strip():
                print_box(stdout, title=f"stdout · {claim.get('id')}", color=DIM, truncate=256)
            if stderr.strip():
                print_box(stderr, title=f"stderr · {claim.get('id')}", color=RED, truncate=256)

            scored = self._score_executable(claim, spec, exec_result)
        else:
            reason = spec.get("reason", "")
            print_box(
                f"Marked non-executable.\nreason: {reason or '(none provided)'}",
                title=f"Verifier spec · {claim.get('id')}",
                color=YELLOW,
            )
            scored = self._score_soft(uuid, claim, execution_text, workspace_listing, reason, grounding)
            print_box(
                f"verdict:   {scored.get('status')}\nrationale: {scored.get('rationale', '')}",
                title=f"Soft check · {claim.get('id')}",
                color=GREEN if scored.get("score", 0) >= 0.5 else RED,
            )

        scored["claim"] = claim
        scored["spec"] = spec
        scored["elapsed_s"] = round(time.time() - t_start, 3)

        final = (
            f"id:     {claim.get('id')}\n"
            f"kind:   {scored.get('verifier_kind')}\n"
            f"status: {scored.get('status')}\n"
            f"score:  {scored.get('score')}\n"
            f"elapsed:{scored['elapsed_s']}s"
        )
        print_box(
            final,
            title=f"Claim verdict · {claim.get('id')}",
            color=GREEN if scored.get("score", 0) >= 0.5 else RED,
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
        workspace_listing: str,
        max_files: int = 3,
    ) -> list[str]:
        """Pick workspace files most likely to hold this claim's artefact.

        Per-claim judge call that fuses the agent narration (names the files
        the agents wrote), the workspace listing (ground truth of what's on
        disk), and the claim description. Returns only paths present in the
        workspace — hallucinated entries are dropped. On parse or provider
        failure falls back to all eligible workspace files so the downstream
        verifier-gen step is never blind.

        Args:
            uuid: Workflow identifier (used for judge calls).
            claim: Normalised claim dict.
            execution_text: Agent narration / produced output text.
            workspace_listing: Rendered listing of workspace files.
            max_files: Maximum number of files to return.

        Returns:
            List of workspace-relative paths the verifier should open.
        """
        eligible = self._eligible_workspace_files()
        if not eligible:
            return []
        prompt = self._build_select_files_prompt(
            claim, execution_text, workspace_listing, max_files
        )
        agent_name = f"verifier_select_files_{claim.get('id', 'unknown')}"
        parsed, err = self._call_judge_for_json(uuid, agent_name, prompt)
        if err or not isinstance(parsed, dict):
            self.logger.debug(
                f"file selection failed for {claim.get('id')}: "
                f"{err or 'non-dict JSON'}"
            )
            return eligible
        selected = self._validate_workspace_paths(
            parsed.get("files"),
            allowed=set(eligible),
            max_count=max_files,
            label=str(claim.get("id", "unknown")),
        )
        return selected or eligible

    @staticmethod
    def _build_select_files_prompt(
        claim: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
        max_files: int,
    ) -> str:
        """Build the per-claim file-selection prompt sent to the judge.

        Args:
            claim: Normalised claim dict.
            execution_text: Agent narration / produced output text.
            workspace_listing: Rendered listing of workspace files.
            max_files: Maximum number of files to request.

        Returns:
            Fully formatted prompt string for the judge.
        """
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
        """Ask the judge for a verifier script (or a non-executable rationale).

        Args:
            uuid: Workflow identifier (used for judge calls).
            claim: Normalised claim dict.
            execution_text: Agent narration / produced output text.
            workspace_listing: Rendered listing of workspace files.

        Returns:
            Spec dict with either ``executable=True`` and ``code``, or
            ``executable=False`` and a ``reason``.
        """
        relevant_previews = self._render_relevant_previews(
            claim.get("likely_relevant_files", [])
        )
        packages = ', '.join(_VERIFIER_BASE_PACKAGES)
        prompt = f"""
You are writing a tiny verifier program for ONE atomic claim from a multi-agent
workflow. The verifier will run inside the same workspace the agents used.

WORKSPACE FILES (relative to workspace root, cwd at runtime):
{workspace_listing}

RELEVANT FILE PREVIEWS (head + tail of files the claim depends on; truncated):
{relevant_previews}

CLAIM TO VERIFY:
- id: {claim['id']}
- importance: {claim.get('importance', self._DEFAULT_CLAIM_IMPORTANCE)} (1-10; 10 = literal deliverable)
- description: {claim['description']}
- likely_relevant_files: {claim.get('likely_relevant_files', [])}

RULES FOR YOUR SCRIPT:
- Print EXACTLY ONE JSON line to stdout, structured as:
  {{"claim_id": "{claim['id']}", "status": "pass" | "fail" | "error",
    "actual": <observed value or null>, "details": "<short string>"}}
- Read files with relative paths
  (cwd is the workspace).
- Recompute or directly check; do not trust the agent's reported numbers.
- For property checks (symmetry, range, no duplicates, ...), assert the
  property and emit "pass"/"fail" accordingly.
- Catch your own exceptions and emit status="error" with the error message in
  details — never let the script raise.
- Parse the relevant files according to the format visible in the PREVIEWS
  above. Do not invent a different format. If the previews include a header
  line (e.g. "Minimum Energy: -6") your parser must skip it gracefully.
- If the previews are empty or do not show enough of the file to be sure of
  the format, prefer permissive parsing (try several reasonable splits, skip
  unparseable lines) over a strict format that may misjudge the file.
- When verifying usage of a specific library method is found in a script, also ensure no cheating attempt was done, such as a try-catch branching that lead to the wrong method being used on exception.
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
AVAILABLE IMPORTS: {packages}

Return STRICT JSON only, in one of these two shapes:
  {{"executable": true,  "code": "<full python script as one string>"}}
  {{"executable": false, "reason": "<one sentence>"}}
"""
        return self._call_and_parse_verifier(uuid, claim, prompt, attempt=1)

    def _call_and_parse_verifier(
        self,
        uuid: str,
        claim: dict[str, Any],
        prompt: str,
        attempt: int,
    ) -> dict[str, Any]:
        """Call the judge for a verifier spec; soft-fail to ``executable: False``.

        Args:
            uuid: Workflow identifier (used for judge calls).
            claim: Normalised claim dict.
            prompt: Pre-built verifier-generation prompt.
            attempt: 1 for the first try, >1 for retries (suffixes the agent name).

        Returns:
            Spec dict; on errors returns ``{"executable": False, "reason": ...}``.
        """
        agent_name = (
            f"verifier_gen_{claim['id']}"
            if attempt == 1
            else f"verifier_gen_{claim['id']}_retry"
        )
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
        """One claim's pre-flight: pick relevant files, then generate the spec.

        These two judge calls are sequential within a single claim (the spec
        prompt needs the file previews chosen here), so they're bundled
        together as one worker unit for parallel fan-out across claims.

        Args:
            uuid: Workflow identifier (used for judge calls).
            claim: Normalised claim dict.
            execution_text: Agent narration / produced output text.
            workspace_listing: Rendered listing of workspace files.

        Returns:
            ``(updated_claim, spec)`` — the claim with ``likely_relevant_files``
            populated, and the verifier spec from ``_generate_verifier``.
        """
        rel_files = self._llm_select_files(
            uuid, claim, execution_text, workspace_listing
        )
        updated = {**claim, "likely_relevant_files": rel_files}
        spec = self._generate_verifier(
            uuid, updated, execution_text, workspace_listing
        )
        return updated, spec

    def _generate_specs_parallel(
        self,
        uuid: str,
        claims: list[dict[str, Any]],
        execution_text: str,
        workspace_listing: str,
        max_workers: int,
    ) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        """Fan out file selection + verifier generation across claims via threads.

        The LLM calls under the hood are sync HTTP; threading is enough to
        overlap their network latency. Each claim writes to its own memory
        file (agent name is keyed by ``claim['id']``), so no cache collisions.

        Args:
            uuid: Workflow identifier (used for judge calls).
            claims: Claims that need a freshly generated spec (anchored ones
                should already be filtered out by the caller).
            execution_text: Agent narration / produced output text.
            workspace_listing: Rendered listing of workspace files.
            max_workers: Upper bound on concurrent LLM calls.

        Returns:
            Dict keyed by claim id mapping to ``(updated_claim, spec)``.
        """
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
                try:
                    results[cid] = f.result()
                except Exception as e:
                    # Mirror the soft-fail contract of _call_and_parse_verifier:
                    # never let one bad claim crash the whole batch.
                    self.logger.warning(
                        f"parallel spec generation failed for {cid}: "
                        f"{type(e).__name__}: {e}"
                    )
                    results[cid] = (
                        {**next(c for c in claims if c["id"] == cid),
                         "likely_relevant_files": []},
                        {"executable": False, "reason": f"spec generation raised: {e}"},
                    )
        return results

    # ------------------------------------------------------------------
    # Verifier helper-package install (idempotent, lock-guarded)
    # ------------------------------------------------------------------

    def _ensure_verifier_packages(self) -> None:
        """Make verifier helper packages importable under ``sys.executable``.

        The verifier ``WorkflowRunner`` is forced to use ``sys.executable`` so
        installs and runs target the same interpreter (the one running
        Mimosa, typically a uv-managed venv). Idempotent across evaluator
        instances via a module-level flag + lock. Logged-and-skipped on
        failure — verifier scripts must then restrict themselves to stdlib.
        """
        global _VERIFIER_PACKAGES_INSTALLED
        if _VERIFIER_PACKAGES_INSTALLED:
            return
        with _VERIFIER_INSTALL_LOCK:
            if _VERIFIER_PACKAGES_INSTALLED:
                return

            smoke_cmd = [
                sys.executable, "-c",
                "import " + ", ".join(_VERIFIER_BASE_IMPORTS),
            ]

            # Early-exit if the helper packages are already importable —
            # common when Mimosa runs in a venv that already has them.
            if self._smoke_check(smoke_cmd):
                _VERIFIER_PACKAGES_INSTALLED = True
                self.logger.info(
                    f"Verifier helper packages already importable under "
                    f"{sys.executable}"
                )
                return

            # ``--break-system-packages`` is the documented escape from PEP 668
            # on system Pythons; inside a venv it is silently ignored. Modern
            # uv-managed envs have a recent pip that supports the flag.
            pip_cmd = [
                sys.executable, "-m", "pip", "install", "--quiet",
                "--disable-pip-version-check", "--break-system-packages",
                *_VERIFIER_BASE_PACKAGES,
            ]
            try:
                r = subprocess.run(pip_cmd, capture_output=True, timeout=600)
            except subprocess.TimeoutExpired:
                self.logger.warning(
                    "Verifier helper package install timed out after 600s"
                )
                return
            except FileNotFoundError as e:
                self.logger.warning(
                    f"pip not found for verifier helper install: {e}"
                )
                return
            except Exception as e:
                self.logger.warning(
                    f"Verifier helper install raised: {type(e).__name__}: {e}"
                )
                return

            if r.returncode != 0:
                stderr_tail = r.stderr.decode(errors="replace")[-400:]
                self.logger.warning(
                    f"Verifier helper install failed (rc={r.returncode}); "
                    f"scripts must restrict themselves to stdlib. "
                    f"stderr tail: {stderr_tail}"
                )
                return

            # Pip can exit 0 yet land packages where the runtime Python can't
            # see them — verify by actually importing.
            if self._smoke_check(smoke_cmd):
                _VERIFIER_PACKAGES_INSTALLED = True
                self.logger.info(
                    f"Verifier helper packages ready: "
                    f"{list(_VERIFIER_BASE_PACKAGES)} (via {sys.executable})"
                )
            else:
                self.logger.warning(
                    "Verifier helper packages installed but not importable "
                    "under sys.executable; scripts will see ImportError."
                )

    @staticmethod
    def _smoke_check(cmd: list[str], timeout: float = 15.0) -> bool:
        """Return True iff *cmd* exits 0 within *timeout*.

        Args:
            cmd: Command and arguments to invoke via ``subprocess.run``.
            timeout: Wall-clock seconds before treating the run as a failure.

        Returns:
            True when the command exits with status 0; False otherwise.
        """
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except Exception:
            return False
        return r.returncode == 0

    # ------------------------------------------------------------------
    # Verifier execution (sandboxed inside the agents' workspace)
    # ------------------------------------------------------------------

    def _run_verifier(self, uuid: str, claim_id: str, code: str) -> dict[str, Any]:
        """Execute a single verifier script in the agents' workspace.

        Args:
            uuid: Workflow identifier (used to name the scratch directory).
            claim_id: Identifier of the claim being verified.
            code: Python source code of the verifier script.

        Returns:
            Dict with ``status``, ``actual``, ``details`` plus raw stdout/stderr
            and the underlying execution ``exit_status``.
        """
        scratch = self._runner_temp_root / uuid
        scratch.mkdir(parents=True, exist_ok=True)
        # Run verifiers under the exact interpreter running Mimosa: that is
        # where the verifier helper packages were installed, and it avoids
        # depending on a system pythonX.Y being on PATH. Passing this via the
        # config (rather than overriding runner._python_cmd after construction)
        # ensures WorkflowRunner's construction-time availability check uses
        # this interpreter too, instead of failing when no matching
        # python_version is found on PATH.
        runner_config = RuntimeConfig(
            python_executable=sys.executable,
            timeout=self.verifier_timeout,
            temp_dir=scratch,
            requirements_file=None,
            use_pty=False,
        )
        runner = WorkflowRunner(runner_config, execution_dir=str(self.workspace_dir))
        execution_id = f"verify_{claim_id}"
        thread_timeout = self.verifier_timeout + 10
        result = None
        try:
            result = _run_coro_sync(
                lambda: runner.execute(code, execution_id=execution_id),
                thread_timeout=thread_timeout,
            )
        except TimeoutError as e:
            return {
                "status": "error",
                "actual": None,
                "details": f"verifier execution did not return in time: {e}",
                "raw_stdout": "",
                "raw_stderr": "",
                "exit_status": "timeout",
            }
        except Exception as e:
            return {
                "status": "error",
                "actual": None,
                "details": f"verifier execution raised: {type(e).__name__}: {e}",
                "raw_stdout": "",
                "raw_stderr": "",
                "exit_status": "error",
            }
        finally:
            try:
                _run_coro_sync(runner.cleanup, thread_timeout=15)
            except Exception as e:
                self.logger.debug(f"verifier runner cleanup failed: {e}")

        parsed = self._parse_verifier_stdout(result.stdout, claim_id)
        parsed.update({
            "raw_stdout": result.stdout,
            "raw_stderr": result.stderr,
            "exit_status": result.status.value if isinstance(result.status, ExecutionStatus) else str(result.status),
        })
        if result.status == ExecutionStatus.TIMEOUT:
            parsed["status"] = "error"
            parsed["details"] = (parsed.get("details") or "") + f" (script timed out after {self.verifier_timeout}s)"
        elif result.status == ExecutionStatus.FAILED and parsed.get("status") not in ("pass", "fail"):
            parsed["status"] = "error"
            parsed["details"] = (parsed.get("details") or "") + f" (script exit code {result.return_code})"
        return parsed

    @staticmethod
    def _parse_verifier_stdout(stdout: str, claim_id: str) -> dict[str, Any]:
        """Pull the last JSON line matching ``claim_id`` from the script stdout.

        Args:
            stdout: Raw captured stdout from the verifier script.
            claim_id: Identifier the JSON line must reference.

        Returns:
            Dict with ``status``, ``actual`` and ``details``; status is set to
            ``"error"`` when no matching line is found.
        """
        if not stdout:
            return {"status": "error", "actual": None, "details": "no stdout from verifier"}
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("claim_id") == claim_id:
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
        return {"status": "error", "actual": None, "details": "no matching JSON line in verifier stdout"}

    # ------------------------------------------------------------------
    # Per-claim scoring
    # ------------------------------------------------------------------

    def _score_executable(
        self,
        claim: dict[str, Any],
        spec: dict[str, Any],
        exec_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert an executable-verifier run into a scored result dict.

        Args:
            claim: Normalised claim dict (unused but kept for signature parity).
            spec: Verifier spec returned by the judge (unused here).
            exec_result: Dict produced by ``_run_verifier``.

        Returns:
            Scored claim dict with ``score`` 1.0 for ``pass`` and 0.0 otherwise.
        """
        status = exec_result.get("status")
        score = 1.0 if status == "pass" else 0.0
        return {
            "score": score,
            "verifier_kind": "executable",
            "status": status,
            "actual": exec_result.get("actual"),
            "details": exec_result.get("details", ""),
            "raw_stdout": exec_result.get("raw_stdout", ""),
            "raw_stderr": exec_result.get("raw_stderr", ""),
            "exit_status": exec_result.get("exit_status", ""),
        }

    def _score_soft(
        self,
        uuid: str,
        claim: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
        reason: str,
        grounding: str = "",
    ) -> dict[str, Any]:
        """Narrow LLM verdict for one non-executable claim, anchored on grounding.

        Args:
            uuid: Workflow identifier (used for judge calls).
            claim: Normalised claim dict.
            execution_text: Agent narration / produced output text.
            workspace_listing: Rendered listing of workspace files.
            reason: Justification string for why the claim is non-executable.
            grounding: Optional peer-reviewed literature grounding block.

        Returns:
            Scored claim dict with ``score`` in {0.0, 0.5, 1.0} and a
            ``rationale`` from the judge.
        """
        relevant_previews = self._render_relevant_previews(
            claim.get("likely_relevant_files", [])
        )
        grounding_block = grounding.strip() if grounding else "(no literature grounding available)"
        prompt = f"""
You are checking ONE claim from a multi-agent workflow. The claim is not
executable in code; please judge it against the concrete context below.

WORKSPACE FILES:
{workspace_listing}

RELEVANT FILE PREVIEWS:
{relevant_previews}

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
        data, err = self._call_judge_for_json(
            uuid, f"verifier_soft_{claim['id']}", prompt
        )
        if err is not None or not isinstance(data, dict):
            # No usable verdict — treat as non-signal so it neither rewards nor
            # punishes the workflow. Aggregator excludes errors from the mean.
            return {
                "score": 0.0,
                "verifier_kind": "soft",
                "status": "error",
                "details": f"soft check failed: {err or 'verdict JSON not an object'}",
                "rationale": "",
            }
        verdict = data.get("verdict", "unsure")
        score = {"pass": 1.0, "unsure": 0.5, "fail": 0.0}.get(verdict, 0.5)
        return {
            "score": score,
            "verifier_kind": "soft",
            "status": verdict,
            "details": str(data.get("rationale", "")),
            "rationale": str(data.get("rationale", "")),
        }


if __name__ == "__main__":
    expected = {
        "_verify_claim",
        "_llm_select_files",
        "_build_select_files_prompt",
        "_generate_verifier",
        "_call_and_parse_verifier",
        "_select_files_and_generate_spec",
        "_generate_specs_parallel",
        "_ensure_verifier_packages",
        "_smoke_check",
        "_run_verifier",
        "_parse_verifier_stdout",
        "_score_executable",
        "_score_soft",
    }
    actual = {n for n in dir(_VerifierPerClaimMixin) if not n.startswith("__")}
    missing = expected - actual
    assert not missing, f"per-claim mixin missing methods: {missing}"
    # _run_coro_sync is module-level — sanity-check it's reachable too.
    assert callable(_run_coro_sync), "_run_coro_sync missing at module level"
    print("verifier_per_claim: smoke ok")

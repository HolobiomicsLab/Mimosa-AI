"""
Verifier-based workflow evaluator.

Replaces the "judge a whole trace with one LLM and one number" pattern with a
deterministic verification pipeline:

    1. Extract atomic claims from the workflow output (LLM, structured output).
    2. For each claim, generate a tiny verifier program (LLM) — or mark the
       claim as non-executable.
    3. Execute every verifier in a sandbox sharing the agents' workspace,
       using `WorkflowRunner` with PTY/colour disabled.
    4. Score each verifier:
         - executable claim → pass=1.0, fail/error=0.0
         - non-executable  → narrow LLM check (single claim + workspace
                              context) returning 0.0 / 0.5 / 1.0
    5. Aggregate. Failing a claim flagged as `criticality: hard` caps the
       overall score so a broken artefact cannot be rescued by good prose.

The unit of evaluation is a specific claim, not a whole trace. The LLM is only
ever asked narrow questions against concrete context.
"""

import asyncio
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Coroutine, TypeVar

from sources.core.llm_provider import LLMProvider
from sources.core.workflow_runner import (
    ExecutionStatus,
    RuntimeConfig,
    WorkflowRunner,
)

from .base import (
    BaseEvaluator,
    EvaluatorError,
    LLMEvaluationError,
    ScoreExtractionError,
    WorkflowDataError,
)

from sources.cli.pretty_print import (
    print_box, print_info,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)

# ----- Default per-script execution limits ------------------------------------
_VERIFIER_TIMEOUT_SECONDS = 60
_VERIFIER_MAX_CLAIMS = 12
_HARD_FAIL_CAP = 0.5


T = TypeVar("T")


def _run_coro_sync(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],
    thread_timeout: float | None = None,
) -> T:
    """Run an async coroutine to completion from a sync caller.

    Crucially, the coroutine is *not* constructed unless we are sure we can
    await it. Constructing a coroutine and then handing it to a failing
    ``asyncio.run`` orphans it and produces the dreaded
    ``coroutine '...' was never awaited`` RuntimeWarning at GC time.

    Args:
        coro_factory: zero-arg callable that returns a fresh coroutine.
        thread_timeout: optional join timeout for the worker thread. ``None``
            means wait forever (caller is responsible for inner timeouts).

    Returns:
        Whatever the coroutine returns.

    Raises:
        TimeoutError: if the worker thread did not complete in time.
        Any exception raised by the coroutine itself.
    """
    try:
        asyncio.get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if not in_loop:
        # Safe to build + await here; one shot, no orphan possible.
        return asyncio.run(coro_factory())

    # Nested case: a daemon worker with its own loop owns the coroutine
    # end-to-end, so it cannot be orphaned.
    holder: dict[str, Any] = {}

    def _target() -> None:
        try:
            holder["result"] = asyncio.run(coro_factory())
        except BaseException as exc:  # noqa: BLE001 — re-raised below
            holder["error"] = exc

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=thread_timeout)
    if t.is_alive():
        raise TimeoutError(
            f"Worker thread did not finish within {thread_timeout}s"
        )
    if "error" in holder:
        raise holder["error"]
    return holder["result"]


def _extract_json_payload(text: str) -> str:
    """Return the first balanced JSON object/array literal found in *text*.

    Tolerant of fenced code blocks (```json ... ```), surrounding prose, and
    trailing commentary the LLM sometimes appends after the JSON.
    """
    if not text:
        return ""
    # Strip markdown fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    # Find the first { or [ and walk to its matching close.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
    return ""


class VerifierEvaluator(BaseEvaluator):
    """Per-claim verifier-based evaluator.

    The judge LLM never assigns a vibes-based 0–1 score across the whole
    workflow. It only:
      - extracts discrete claims,
      - writes a verifier program for each,
      - or, for soft claims, answers a narrow yes/maybe/no question against
        concrete workspace context.
    """

    def __init__(
        self,
        config,
        workspace_dir: str | Path | None = None,
        verifier_timeout: int = _VERIFIER_TIMEOUT_SECONDS,
        max_claims: int = _VERIFIER_MAX_CLAIMS,
        hard_fail_cap: float = _HARD_FAIL_CAP,
    ):
        """Initialize the VerifierEvaluator.

        Args:
            config: Standard evaluator config (memory_dir, workflow_dir,
                judge_model, ...). `config.workspace_dir` is used as the
                default sandbox cwd if `workspace_dir` is not given.
            workspace_dir: Directory the verifier scripts run in. Should match
                the directory the agents wrote their artefacts to.
            verifier_timeout: Per-script timeout (seconds).
            max_claims: Hard cap on number of claims considered, to bound LLM
                and sandbox cost on long traces.
            hard_fail_cap: Upper bound on `overall_score` when any claim
                marked `criticality: hard` fails.
        """
        super().__init__(config)
        self.workspace_dir = Path(
            workspace_dir
            if workspace_dir is not None
            else getattr(config, "workspace_dir", ".")
        )
        self.verifier_timeout = verifier_timeout
        self.max_claims = max_claims
        self.hard_fail_cap = hard_fail_cap

        # Reuse the temp_dir setting from the global runner config when
        # available; otherwise fall back to a per-uuid scratch under workflow_dir.
        self._runner_temp_root = Path(
            getattr(config, "temp_dir", None) or self.workflow_dir / "_verifier_tmp"
        )
        self.logger.info(
            f"VerifierEvaluator initialized (workspace={self.workspace_dir}, "
            f"timeout={verifier_timeout}s, max_claims={max_claims})"
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(self, uuid: str) -> dict[str, Any]:
        """Run the verifier pipeline for a workflow run.

        Args:
            uuid: UUID of the workflow run.

        Returns:
            A dict with per-claim results and aggregated scores. Also
            persisted via `_save_results(scores, uuid, 'verifier')` and
            written to `<workflow_dir>/<uuid>/verifier_evaluation.txt`.
        """
        if not uuid or not isinstance(uuid, str):
            raise EvaluatorError("Invalid uuid: must be a non-empty string")

        execution_text, success = self.workflow_execution_text(uuid)
        if not execution_text:
            raise WorkflowDataError(f"Cannot generate execution text for workflow {uuid}")

        workspace_listing = self._list_workspace()
        claims = self._extract_claims(uuid, execution_text, workspace_listing, success)
        if not claims:
            self.logger.warning(f"No claims extracted for {uuid}; verifier returns 0.0")
            scores = {"overall_score": 0.0, "n_claims": 0, "n_pass": 0, "n_fail": 0}
            self._save_results(scores, uuid, "verifier")
            return {"uuid": uuid, "claims": [], **scores}

        per_claim: list[dict[str, Any]] = []
        for claim in claims[: self.max_claims]:
            result = self._verify_claim(uuid, claim, execution_text, workspace_listing)
            per_claim.append(result)

        scores = self._aggregate(per_claim)
        self._write_report(uuid, claims, per_claim, scores)
        try:
            self._save_results(scores, uuid, "verifier")
        except Exception as e:
            self.logger.error(f"Failed to persist verifier scores for {uuid}: {e}")

        return {"uuid": uuid, "claims": per_claim, **scores}

    # ------------------------------------------------------------------
    # Stage 1 — claim extraction
    # ------------------------------------------------------------------

    def _extract_claims(
        self,
        uuid: str,
        execution_text: str,
        workspace_listing: str,
        success: bool,
    ) -> list[dict[str, Any]]:
        """Ask the LLM to break the workflow output into atomic, typed claims."""
        if not success:
            # Failed runs have no artefacts to verify; record one hard claim.
            return [{
                "id": "c0_execution_succeeded",
                "description": "The workflow executed to completion and produced a non-empty answer.",
                "criticality": "hard",
            }]

        prompt = f"""
You will receive the final state of a multi-agent workflow and a listing of
files present in the agents' workspace.

WORKFLOW OUTPUT:
{execution_text}

WORKSPACE FILES (relative to workspace root):
{workspace_listing}

TASK:
Extract a list of ATOMIC CLAIMS the workflow makes. A good claim is:
- a single, checkable statement (a number, a file existence, a dataset shape,
  a structural property, a comparison against a constraint, a derivation step),
- specific enough that a small Python script could potentially verify it
  against the workspace,
- not a meta-comment about the workflow ("the analysis was thorough").

For each claim, also estimate `criticality`:
- "hard": load-bearing for the answer (final metrics, headline files,
  required computations, claimed satisfaction of the user goal).
- "soft": supporting context (intermediate sanity remarks, choices that are
  defensible but not strictly required).

Return STRICT JSON only, no prose, in this exact form:
{{
  "claims": [
    {{
      "id": "c1_short_slug",
      "description": "<concise restatement of the claim>",
      "criticality": "hard" | "soft"
    }},
    ...
  ]
}}

Aim for at most {self.max_claims} claims, prioritising the most load-bearing
ones first. Do not invent claims that the workflow did not make.
"""
        try:
            output = self._call_judge(uuid, "verifier_extract_claims", prompt)
        except Exception as e:
            raise LLMEvaluationError(f"Claim extraction failed for {uuid}: {e}") from e

        payload = _extract_json_payload(output)
        if not payload:
            raise LLMEvaluationError(f"Claim extractor returned no JSON for {uuid}")
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            raise LLMEvaluationError(f"Claim extractor returned invalid JSON for {uuid}: {e}") from e

        claims = data.get("claims", []) if isinstance(data, dict) else data
        if not isinstance(claims, list):
            raise LLMEvaluationError(f"Claim extractor JSON has no 'claims' list for {uuid}")

        cleaned: list[dict[str, Any]] = []
        for idx, c in enumerate(claims):
            if not isinstance(c, dict) or "description" not in c:
                continue
            cleaned.append({
                "id": str(c.get("id") or f"c{idx}"),
                "description": str(c["description"]).strip(),
                "criticality": "hard" if c.get("criticality") == "hard" else "soft",
            })
        return cleaned

    # ------------------------------------------------------------------
    # Stage 2 + 3 + 4 — generate, run and score one verifier
    # ------------------------------------------------------------------

    def _verify_claim(
        self,
        uuid: str,
        claim: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
    ) -> dict[str, Any]:
        """Generate, execute (if executable) and score a single claim.

        Emits diagnostic ``print_box`` panels around each stage so the user
        can see, at a glance:
          - which claim is being checked,
          - the verifier code (or reason for being marked non-executable),
          - the script's stdout/stderr/exit status,
          - the final verdict.
        """
        # 1. Header — the claim being verified
        claim_text = (
            f"id:          {claim.get('id')}\n"
            f"criticality: {claim.get('criticality')}\n"
            f"description: {claim.get('description')}"
        )
        print_box(claim_text, title=f"Verifying claim {claim.get('id')}", color=CYAN)

        # 2. Generate the verifier spec (executable code OR soft reason)
        spec = self._generate_verifier(uuid, claim, execution_text, workspace_listing)

        if spec.get("executable") and spec.get("code"):
            code = spec["code"]
            print_box(code, title=f"Verifier preview · {claim.get('id')}", color=YELLOW, truncate=512)

            # 3. Run it and surface what actually happened
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
                print_box(stdout, title=f"stdout · {claim.get('id')}", color=DIM, truncate=4000)
            if stderr.strip():
                print_box(stderr, title=f"stderr · {claim.get('id')}", color=RED, truncate=4000)

            scored = self._score_executable(claim, spec, exec_result)
        else:
            reason = spec.get("reason", "")
            print_box(
                f"Marked non-executable.\nreason: {reason or '(none provided)'}",
                title=f"Verifier spec · {claim.get('id')}",
                color=YELLOW,
            )
            scored = self._score_soft(uuid, claim, execution_text, workspace_listing, reason)
            print_box(
                f"verdict:   {scored.get('status')}\nrationale: {scored.get('rationale', '')}",
                title=f"Soft check · {claim.get('id')}",
                color=GREEN if scored.get("score", 0) >= 0.5 else RED,
            )

        scored["claim"] = claim
        scored["spec"] = spec

        # 4. Final verdict panel — green on pass-ish, red otherwise.
        final = (
            f"id:     {claim.get('id')}\n"
            f"kind:   {scored.get('verifier_kind')}\n"
            f"status: {scored.get('status')}\n"
            f"score:  {scored.get('score')}"
        )
        print_box(
            final,
            title=f"Claim verdict · {claim.get('id')}",
            color=GREEN if scored.get("score", 0) >= 0.5 else RED,
        )
        return scored

    def _generate_verifier(
        self,
        uuid: str,
        claim: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
    ) -> dict[str, Any]:
        prompt = f"""
You are writing a tiny verifier program for ONE atomic claim from a multi-agent
workflow. The verifier will run inside the same workspace the agents used.

WORKSPACE FILES (relative to workspace root, cwd at runtime):
{workspace_listing}

WORKFLOW OUTPUT (for context only — do not re-evaluate the whole thing):
{execution_text}

CLAIM TO VERIFY:
- id: {claim['id']}
- criticality: {claim['criticality']}
- description: {claim['description']}

RULES FOR YOUR SCRIPT:
- Print EXACTLY ONE JSON line to stdout, structured as:
  {{"claim_id": "{claim['id']}", "status": "pass" | "fail" | "error",
    "actual": <observed value or null>, "details": "<short string>"}}
- Use only the standard library plus numpy/pandas if needed. Read files with
  relative paths (cwd is the workspace).
- Recompute or directly check; do not trust the agent's reported numbers.
- For property checks (symmetry, range, no duplicates, ...), assert the
  property and emit "pass"/"fail" accordingly.
- Catch your own exceptions and emit status="error" with the error message in
  details — never let the script raise.
- Never make assumptions about file content.

If the claim cannot be checked deterministically with code (e.g. it concerns
the rigor of a proof, the appropriateness of a binning choice, the
defensibility of a conclusion), set "executable": false and explain briefly.

Return STRICT JSON only, in one of these two shapes:
  {{"executable": true,  "code": "<full python script as one string>"}}
  {{"executable": false, "reason": "<one sentence>"}}
"""
        try:
            raw = self._call_judge(uuid, f"verifier_gen_{claim['id']}", prompt)
        except Exception as e:
            return {"executable": False, "reason": f"verifier generation failed: {e}"}

        payload = _extract_json_payload(raw)
        if not payload:
            return {"executable": False, "reason": "verifier generator returned no JSON"}
        try:
            spec = json.loads(payload)
        except json.JSONDecodeError as e:
            return {"executable": False, "reason": f"verifier JSON invalid: {e}"}
        if not isinstance(spec, dict):
            return {"executable": False, "reason": "verifier JSON not an object"}
        return spec

    def _run_verifier(self, uuid: str, claim_id: str, code: str) -> dict[str, Any]:
        """Execute a single verifier script in the agents' workspace.

        Uses ``_run_coro_sync`` so the WorkflowRunner coroutines are always
        awaited — both ``execute`` and ``cleanup`` — whether we are called
        from a plain sync context or from inside a running asyncio loop.
        """
        scratch = self._runner_temp_root / uuid
        scratch.mkdir(parents=True, exist_ok=True)
        # Disable PTY + don't auto-install requirements — verifier scripts are
        # short, non-interactive checks.
        runner_config = RuntimeConfig(
            timeout=self.verifier_timeout,
            temp_dir=scratch,
            requirements_file=None,
            use_pty=False,
        )
        runner = WorkflowRunner(runner_config, execution_dir=str(self.workspace_dir))
        execution_id = f"verify_{claim_id}"

        # Give the worker thread a bit of slack beyond the per-script timeout
        # so the runner itself can return a TIMEOUT result rather than us
        # killing the thread blindly.
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
        except Exception as e:  # surface the underlying failure
            return {
                "status": "error",
                "actual": None,
                "details": f"verifier execution raised: {type(e).__name__}: {e}",
                "raw_stdout": "",
                "raw_stderr": "",
                "exit_status": "error",
            }
        finally:
            # Cleanup MUST also be awaited via the helper; calling
            # asyncio.run() directly from inside a running loop would leak
            # the cleanup coroutine.
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
        """Pull the last JSON line matching `claim_id` out of the script stdout."""
        if not stdout:
            return {"status": "error", "actual": None, "details": "no stdout from verifier"}
        # Walk lines from the end so trailing print-statements take precedence.
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

    def _score_executable(
        self,
        claim: dict[str, Any],
        spec: dict[str, Any],
        exec_result: dict[str, Any],
    ) -> dict[str, Any]:
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
    ) -> dict[str, Any]:
        """Narrow LLM check for non-executable claims.
        The LLM is asked one targeted question (does this single claim hold
        given this concrete context?), not an aggregate vibes score.
        """

        prompt = f"""
You are checking ONE claim from a multi-agent workflow. The claim is not
executable in code; please judge it against the concrete context below.

WORKSPACE FILES:
{workspace_listing}

WORKFLOW OUTPUT (context only):
{execution_text}

CLAIM:
- id: {claim['id']}
- criticality: {claim['criticality']}
- description: {claim['description']}

REASON IT WAS MARKED NON-EXECUTABLE:
{reason or '(none)'}

Answer ONLY this question: given the workspace and output above, does the
claim hold? Use one of three verdicts:
- "pass"   : the claim is well supported by the visible context.
- "unsure" : context is insufficient to decide either way.
- "fail"   : the claim is contradicted or clearly unsupported.

Return STRICT JSON: {{"verdict": "pass" | "unsure" | "fail", "rationale": "<one sentence>"}}
"""
        try:
            raw = self._call_judge(uuid, f"verifier_soft_{claim['id']}", prompt)
            payload = _extract_json_payload(raw)
            data = json.loads(payload) if payload else {}
        except Exception as e:
            return {
                "score": 0.0,
                "verifier_kind": "soft",
                "status": "error",
                "details": f"soft check failed: {e}",
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

    # ------------------------------------------------------------------
    # Stage 5 — aggregation
    # ------------------------------------------------------------------

    def _aggregate(self, per_claim: list[dict[str, Any]]) -> dict[str, Any]:
        if not per_claim:
            return {"overall_score": 0.0, "n_claims": 0, "n_pass": 0, "n_fail": 0}
        scores = [c["score"] for c in per_claim]
        overall = sum(scores) / len(scores)

        hard_fail = any(
            c["claim"].get("criticality") == "hard" and c["score"] < 1.0 and c["status"] != "unsure"
            for c in per_claim
        )
        capped = min(overall, self.hard_fail_cap) if hard_fail else overall

        n_pass = sum(1 for c in per_claim if c["status"] == "pass")
        n_fail = sum(1 for c in per_claim if c["status"] == "fail")
        n_error = sum(1 for c in per_claim if c["status"] == "error")
        n_unsure = sum(1 for c in per_claim if c["status"] == "unsure")
        return {
            "overall_score": round(capped, 4),
            "overall_score_uncapped": round(overall, 4),
            "hard_fail_capped": hard_fail,
            "n_claims": len(per_claim),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_error": n_error,
            "n_unsure": n_unsure,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _list_workspace(self, max_entries: int = 200) -> str:
        """Return a short, deterministic listing of the workspace."""
        ws = self.workspace_dir
        if not ws.exists():
            return "(workspace directory does not exist)"
        entries = []
        for root, dirs, files in os.walk(ws):
            # Skip noisy directories.
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "node_modules"}]
            for f in files:
                rel = Path(root, f).relative_to(ws)
                try:
                    size = (ws / rel).stat().st_size
                except OSError:
                    size = -1
                entries.append(f"{rel}\t{size}B")
                if len(entries) >= max_entries:
                    entries.append(f"... (truncated at {max_entries} entries)")
                    return "\n".join(entries)
        return "\n".join(entries) if entries else "(empty workspace)"

    def _call_judge(self, uuid: str, agent_name: str, prompt: str) -> str:
        memory_path = Path(self.memory_dir) / uuid
        memory_path.mkdir(parents=True, exist_ok=True)
        provider = LLMProvider(
            agent_name=agent_name,
            memory_path=memory_path,
            system_msg=self._get_judge_system_prompt(),
            config=self.llm_config,
        )
        return provider(prompt)

    def _write_report(
        self,
        uuid: str,
        claims: list[dict[str, Any]],
        per_claim: list[dict[str, Any]],
        scores: dict[str, Any],
    ) -> None:
        path = self.workflow_dir / uuid / "evaluation.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("Verifier Evaluation\n")
                f.write("=" * 60 + "\n")
                f.write(f"Claims: {scores['n_claims']}  pass={scores['n_pass']}  "
                        f"fail={scores['n_fail']}  error={scores.get('n_error', 0)}  "
                        f"unsure={scores.get('n_unsure', 0)}\n")
                f.write(f"Overall: {scores['overall_score']:.3f}"
                        f" (uncapped {scores.get('overall_score_uncapped', 0.0):.3f}, "
                        f"hard_fail_capped={scores.get('hard_fail_capped', False)})\n\n")
                for c in per_claim:
                    cl = c["claim"]
                    f.write(f"[{cl['id']}] ({cl['criticality']}) {cl['description']}\n")
                    f.write(f"  kind={c['verifier_kind']} status={c['status']} score={c['score']}\n")
                    if c.get("details"):
                        f.write(f"  details: {c['details']}\n")
                    if c.get("verifier_kind") == "executable" and c.get("raw_stderr"):
                        stderr_snippet = c["raw_stderr"].strip().splitlines()[-5:]
                        if stderr_snippet:
                            f.write("  stderr (tail):\n")
                            for line in stderr_snippet:
                                f.write(f"    {line}\n")
                    f.write("\n")
            self.logger.info(f"Verifier report written to {path}")
        except OSError as e:
            self.logger.error(f"Could not write verifier report for {uuid}: {e}")

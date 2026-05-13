"""Per-claim verifier-based workflow evaluator.

Pipeline: extract atomic claims → generate a verifier script per claim →
execute in the agents' workspace → score per claim → aggregate.
"""

import ast
import asyncio
import json
import math
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
from .grounding import get_perspicacite_grounding

from sources.cli.pretty_print import (
    print_box, print_info,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)

# ----- Execution limits -------------------------------------------------------
_VERIFIER_TIMEOUT_SECONDS = 60
_VERIFIER_MAX_CLAIMS = 24
_VERIFIER_MIN_CLAIMS = 3
_HARD_FAIL_CAP = 0.5

# ----- Information bonus (rewards thoroughness; saturates) --------------------
# bonus(n) = alpha * (1 - exp(-n_hard_pass / beta)); see _aggregate.
_INFO_BONUS_ALPHA = 0.15
_INFO_BONUS_BETA = 4.0

# ----- File preview budgets ---------------------------------------------------
_PREVIEW_HEAD_BYTES = 8 * 1024
_PREVIEW_TAIL_BYTES = 2 * 1024
_PREVIEW_PER_CLAIM_CAP = 24 * 1024
_BINARY_SNIFF_BYTES = 4096

# ----- Anti-cheat thresholds --------------------------------------------------
_CHEAT_MIN_LITERAL_LEN = 80
_CHEAT_OVERLAP_WINDOW = 60
_IO_MARKERS = (
    "open(",
    "Path(",
    ".read_text(",
    ".read_bytes(",
    "json.load",
    "csv.reader",
    "csv.DictReader",
    "pd.read_",
    "pandas.read_",
    "np.load",
    "np.loadtxt",
    "np.genfromtxt",
    "numpy.load",
    "numpy.loadtxt",
    "numpy.genfromtxt",
    "subprocess.",
    "os.path.exists",
    "os.path.isfile",
    "os.stat",
    "Path.exists",
    "Path.is_file",
    "glob.glob",
)


def _verifier_appears_to_cheat(
    code: str,
    execution_text: str,
    requires_io: bool,
) -> tuple[bool, str]:
    """Reject verifiers that inline the agent's answer or skip workspace I/O."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, ""

    exec_norm = " ".join(execution_text.split())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        lit = node.value
        if len(lit) < _CHEAT_MIN_LITERAL_LEN:
            continue
        lit_norm = " ".join(lit.split())
        step = max(1, _CHEAT_OVERLAP_WINDOW // 3)
        for i in range(0, max(1, len(lit_norm) - _CHEAT_OVERLAP_WINDOW + 1), step):
            window = lit_norm[i : i + _CHEAT_OVERLAP_WINDOW]
            if len(window) < _CHEAT_OVERLAP_WINDOW:
                break
            if window in exec_norm:
                return (
                    True,
                    f"verifier embeds {len(lit)} chars of the workflow output "
                    f"as a string literal and parses that instead of reading "
                    f"the workspace",
                )

    if requires_io and not any(marker in code for marker in _IO_MARKERS):
        return (
            True,
            "claim declares likely_relevant_files but the verifier performs no "
            "file I/O — it cannot be grounded in workspace state",
        )
    return False, ""


T = TypeVar("T")


def _run_coro_sync(
    coro_factory: Callable[[], Coroutine[Any, Any, T]],
    thread_timeout: float | None = None,
) -> T:
    """Run an async coroutine from sync code, even if a loop is already running.

    The coroutine is built lazily so it can never be orphaned on a failed run.
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


def _extract_json_payload(text: str) -> str:
    """First balanced JSON object/array in *text*, tolerant of fences and prose."""
    if not text:
        return ""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
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
    """Per-claim verifier-based evaluator."""

    def __init__(
        self,
        config,
        workspace_dir: str | Path | None = None,
        verifier_timeout: int = _VERIFIER_TIMEOUT_SECONDS,
        max_claims: int = _VERIFIER_MAX_CLAIMS,
        min_claims: int = _VERIFIER_MIN_CLAIMS,
        hard_fail_cap: float = _HARD_FAIL_CAP,
        preview_head_bytes: int = _PREVIEW_HEAD_BYTES,
        preview_tail_bytes: int = _PREVIEW_TAIL_BYTES,
        preview_per_claim_cap: int = _PREVIEW_PER_CLAIM_CAP,
        use_grounding: bool = True,
        info_bonus_alpha: float = _INFO_BONUS_ALPHA,
        info_bonus_beta: float = _INFO_BONUS_BETA,
    ):
        """Initialise; workspace_dir defaults to ``config.workspace_dir``."""
        super().__init__(config)
        self.workspace_dir = Path(
            workspace_dir
            if workspace_dir is not None
            else getattr(config, "workspace_dir", ".")
        )
        self.verifier_timeout = verifier_timeout
        self.max_claims = max_claims
        self.min_claims = max(0, min_claims)
        self.hard_fail_cap = hard_fail_cap
        self.preview_head_bytes = preview_head_bytes
        self.preview_tail_bytes = preview_tail_bytes
        self.preview_per_claim_cap = preview_per_claim_cap
        self.use_grounding = use_grounding
        self.info_bonus_alpha = max(0.0, info_bonus_alpha)
        self.info_bonus_beta = max(1e-6, info_bonus_beta)
        self._preview_cache: dict[str, str] = {}
        self._workspace_files: set[str] = set()
        self._grounding_cache: dict[str, str] = {}
        self._runner_temp_root = Path(
            getattr(config, "temp_dir", None) or self.workflow_dir / "_verifier_tmp"
        )
        self.logger.info(
            f"VerifierEvaluator initialized (workspace={self.workspace_dir}, "
            f"timeout={verifier_timeout}s, claims={self.min_claims}–{max_claims}, "
            f"use_grounding={use_grounding}, "
            f"info_bonus(α={self.info_bonus_alpha}, β={self.info_bonus_beta}))"
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(self, uuid: str) -> dict[str, Any]:
        """Run the verifier pipeline; persists scores under ``evaluation.verifier``."""
        if not uuid or not isinstance(uuid, str):
            raise EvaluatorError("Invalid uuid: must be a non-empty string")

        execution_text, success = self.workflow_execution_text(uuid)
        if not execution_text:
            raise WorkflowDataError(f"Cannot generate execution text for workflow {uuid}")

        # Short-circuit when the workflow produced no artefacts: avoids scoring
        # against stale state from a prior generation in a shared workspace.
        wf_info = self._load_workflow_data(uuid)
        if not wf_info.state_result and not wf_info.code:
            return self._short_circuit_failed_run(uuid)

        workspace_listing = self._list_workspace()
        grounding = self._get_grounding(uuid, execution_text) if success else self._GROUNDING_DISABLED
        claims = self._extract_claims(uuid, execution_text, workspace_listing, success, grounding)
        if not claims:
            self.logger.warning(f"No claims extracted for {uuid}; verifier returns 0.0")
            scores = {"overall_score": 0.0, "n_claims": 0, "n_pass": 0, "n_fail": 0}
            self._save_results(scores, uuid, "verifier")
            return {"uuid": uuid, "claims": [], **scores}

        per_claim: list[dict[str, Any]] = []
        for claim in claims[: self.max_claims]:
            result = self._verify_claim(uuid, claim, execution_text, workspace_listing, grounding)
            per_claim.append(result)

        scores = self._aggregate(per_claim)
        self._write_report(uuid, claims, per_claim, scores)
        try:
            self._save_results(scores, uuid, "verifier")
        except Exception as e:
            self.logger.error(f"Failed to persist verifier scores for {uuid}: {e}")

        return {"uuid": uuid, "claims": per_claim, **scores}

    def _short_circuit_failed_run(self, uuid: str) -> dict[str, Any]:
        """Return 0.0 without running scripts when the workflow produced nothing."""
        print_box(
            f"workflow {uuid} produced no code and no state_result; verifier "
            f"returns 0.0 without running scripts.",
            title="Verifier short-circuit — generation failed",
            color=RED,
        )
        scores = {
            "overall_score": 0.0,
            "overall_score_uncapped": 0.0,
            "hard_fail_capped": True,
            "n_claims": 0,
            "n_pass": 0,
            "n_fail": 0,
            "n_error": 0,
            "n_unsure": 0,
            "skipped_reason": "workflow_generation_or_execution_failed",
        }
        try:
            self._write_report(uuid, [], [], scores)
        except Exception as e:
            self.logger.error(f"Failed to write short-circuit report for {uuid}: {e}")
        try:
            self._save_results(scores, uuid, "verifier")
        except Exception as e:
            self.logger.error(f"Failed to persist short-circuit scores for {uuid}: {e}")
        return {"uuid": uuid, "claims": [], **scores}

    # ------------------------------------------------------------------
    # Stage 1 — claim extraction
    # ------------------------------------------------------------------

    def _extract_claims(
        self,
        uuid: str,
        execution_text: str,
        workspace_listing: str,
        success: bool,
        grounding: str = "",
    ) -> list[dict[str, Any]]:
        """Ask the LLM to break the workflow output into atomic, typed claims."""
        if not success:
            return [{
                "id": "c0_execution_succeeded",
                "description": "The workflow executed to completion and produced a non-empty answer.",
                "criticality": "hard",
                "likely_relevant_files": [],
            }]

        grounding_block = grounding.strip() if grounding else "(no literature grounding available)"
        prompt = f"""
You will receive the final state of a multi-agent workflow, a listing of
files present in the agents' workspace, and peer-reviewed scientific
literature grounding for the task at hand.

WORKFLOW OUTPUT:
{execution_text}

WORKSPACE FILES (relative to workspace root):
{workspace_listing}

LITERATURE GROUNDING (peer-reviewed context — what the literature says about
this kind of task; use it to know what is scientifically load-bearing):
{grounding_block}

TASK:
Extract a list of ATOMIC CLAIMS the workflow makes. A good claim is:
- a single, checkable statement (a number, a file existence, a dataset shape,
  a structural property, a comparison against a constraint, a derivation step),
- ideally specific enough that a small Python script could potentially verify it
  against the workspace, or that can be verified against scientific literature if it concerns a methodological choice or a conclusion's defensibility,
- not a meta-comment about the workflow ("the analysis was thorough").
- not a vague statement about the workflow's overall performance ("the workflow successfully identified the key drivers of the system")
- not claims about what was created but rather about what was achieved.
- not claims about reported status or that are obviously just restatements of the workflow output ("the answer is X").


For each claim, also estimate `criticality`:
- "hard": load-bearing for the answer (final metrics, headline files,
  required computations, claimed satisfaction of the user goal). Use the
  literature grounding to recognise which steps are scientifically
  load-bearing for this task — those are "hard" by default.
- "soft": supporting context (intermediate sanity remarks, choices that are
  defensible but not strictly required, decisions the literature treats as
  trivial or auxiliary).

Use the literature grounding to **prioritise** claims:
- Prefer claims that map onto the methodology the literature considers
  standard for this task (e.g. expected metrics, required preprocessing,
  established constraints).
- Avoid extracting claims about steps the literature considers irrelevant
  or trivial. Don't pad the claim list with cosmetic statements.
- If the workflow skipped a step the literature considers required, you
  may add a claim asserting the workflow performed that step (it will
  likely fail verification, which is the correct signal).

For each claim, also list `likely_relevant_files`: relative paths whose
contents the verifier would need to read in order to check the claim.
- ONLY use paths that appear verbatim in the WORKSPACE FILES listing above.
  Do not invent or guess paths the workflow's answer mentions but that are
  not in the listing.
- Use `[]` if the claim is purely about the workflow's output text and has
  no on-disk artefact to consult.
- Multi-file claims (e.g. "model in weights.pt produces predictions.csv")
  may list several files — keep them in dependency order.

Return STRICT JSON only, no prose, in this exact form:
{{
  "claims": [
    {{
      "id": "c1_short_slug",
      "description": "<concise restatement of the claim>",
      "criticality": "hard" | "soft",
      "likely_relevant_files": ["<relative/path>", ...]
    }},
    ...
  ]
}}

Aim for {self.min_claims}–{self.max_claims} claims (target at least {self.min_claims}),
prioritising the most load-bearing first. Do not invent claims that the workflow
did not make.
"""
        # Two outer attempts: first call, then one retry if fewer than min_claims
        # were returned. Each attempt also has the JSON-parse retry inside
        # _call_judge_for_json, so worst case is 4 LLM calls.
        attempts_for_count = 2
        extra_feedback = ""
        cleaned: list[dict[str, Any]] = []
        for count_attempt in range(attempts_for_count):
            full_prompt = prompt + extra_feedback
            data, err = self._call_judge_for_json(
                uuid, "verifier_extract_claims", full_prompt
            )
            if err is not None:
                raise LLMEvaluationError(f"Claim extraction failed for {uuid}: {err}")

            cleaned = self._parse_and_filter_claims(uuid, data)
            if len(cleaned) >= self.min_claims:
                return cleaned
            if count_attempt == attempts_for_count - 1:
                self.logger.warning(
                    f"Claim extractor returned only {len(cleaned)} claims after "
                    f"retry (min_claims={self.min_claims}); proceeding with "
                    f"{len(cleaned)}"
                )
                return cleaned
            self.logger.warning(
                f"Claim extractor returned only {len(cleaned)} claims "
                f"(min_claims={self.min_claims}); retrying once with feedback"
            )
            extra_feedback = (
                f"\n\nYOUR PREVIOUS RESPONSE RETURNED ONLY {len(cleaned)} CLAIMS. "
                f"The verifier requires at least {self.min_claims} atomic claims. "
                f"The workflow output likely contains more verifiable claims — "
                f"re-read it and extract additional load-bearing claims (specific "
                f"numbers, file properties, structural assertions, methodological "
                f"steps). Aim for {self.min_claims}–{self.max_claims} claims, "
                f"prioritising the most load-bearing first."
            )
        return cleaned

    def _parse_and_filter_claims(
        self,
        uuid: str,
        data: Any,
    ) -> list[dict[str, Any]]:
        """Validate the LLM JSON, normalise each claim, drop confabulated paths."""
        claims = data.get("claims", []) if isinstance(data, dict) else data
        if not isinstance(claims, list):
            raise LLMEvaluationError(f"Claim extractor JSON has no 'claims' list for {uuid}")

        cleaned: list[dict[str, Any]] = []
        for idx, c in enumerate(claims):
            if not isinstance(c, dict) or "description" not in c:
                continue
            raw_files = c.get("likely_relevant_files", [])
            if not isinstance(raw_files, list):
                raw_files = []
            relevant: list[str] = []
            seen: set[str] = set()
            for rf in raw_files:
                if not isinstance(rf, str):
                    continue
                rp = rf.strip().lstrip("./")
                if not rp or rp in seen:
                    continue
                if self._workspace_files and rp not in self._workspace_files:
                    self.logger.debug(
                        f"Dropping confabulated file '{rp}' for {c.get('id')}"
                    )
                    continue
                relevant.append(rp)
                seen.add(rp)

            cleaned.append({
                "id": str(c.get("id") or f"c{idx}"),
                "description": str(c["description"]).strip(),
                "criticality": "hard" if c.get("criticality") == "hard" else "soft",
                "likely_relevant_files": relevant,
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
        grounding: str = "",
    ) -> dict[str, Any]:
        """Generate, execute (if executable) and score a single claim."""
        rel_files = claim.get("likely_relevant_files", [])
        claim_text = (
            f"id:          {claim.get('id')}\n"
            f"criticality: {claim.get('criticality')}\n"
            f"description: {claim.get('description')}\n"
            f"files:       {rel_files if rel_files else '(none)'}"
        )
        print_box(claim_text, title=f"Verifying claim {claim.get('id')}", color=CYAN)

        spec = self._generate_verifier(uuid, claim, execution_text, workspace_listing)

        if spec.get("executable") and spec.get("code"):
            code = spec["code"]
            print_box(code, title=f"Verifier preview · {claim.get('id')}", color=YELLOW, truncate=512)
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
            scored = self._score_soft(uuid, claim, execution_text, workspace_listing, reason, grounding)
            print_box(
                f"verdict:   {scored.get('status')}\nrationale: {scored.get('rationale', '')}",
                title=f"Soft check · {claim.get('id')}",
                color=GREEN if scored.get("score", 0) >= 0.5 else RED,
            )

        scored["claim"] = claim
        scored["spec"] = spec

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
        relevant_previews = self._render_relevant_previews(
            claim.get("likely_relevant_files", [])
        )
        prompt = f"""
You are writing a tiny verifier program for ONE atomic claim from a multi-agent
workflow. The verifier will run inside the same workspace the agents used.

WORKSPACE FILES (relative to workspace root, cwd at runtime):
{workspace_listing}

RELEVANT FILE PREVIEWS (head + tail of files the claim depends on; truncated):
{relevant_previews}

WORKFLOW OUTPUT (for context only — do not re-evaluate the whole thing):
{execution_text}

CLAIM TO VERIFY:
- id: {claim['id']}
- criticality: {claim['criticality']}
- description: {claim['description']}
- likely_relevant_files: {claim.get('likely_relevant_files', [])}

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
- Parse the relevant files according to the format visible in the PREVIEWS
  above. Do not invent a different format. If the previews include a header
  line (e.g. "Minimum Energy: -6") your parser must skip it gracefully.
- If the previews are empty or do not show enough of the file to be sure of
  the format, prefer permissive parsing (try several reasonable splits, skip
  unparseable lines) over a strict format that may misjudge the file.

ANTI-PATTERNS — your verifier will be REJECTED if it does any of these:
- Embeds the workflow output, the agent's final answer, or any large
  fragment thereof as a string literal and then parses that literal. This
  is a tautology: comparing the answer to itself proves nothing.
- Hard-codes the expected value (e.g. ``status == "SUCCESS"`` against an
  inlined JSON blob) instead of recomputing it from workspace files.
- Returns "pass" without ever opening a file or running a real computation
  derived from on-disk state.
- Declares ``likely_relevant_files`` but performs no file I/O.

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

Return STRICT JSON only, in one of these two shapes:
  {{"executable": true,  "code": "<full python script as one string>"}}
  {{"executable": false, "reason": "<one sentence>"}}
"""
        spec = self._call_and_parse_verifier(uuid, claim, prompt, attempt=1)
        if not (spec.get("executable") and spec.get("code")):
            return spec

        requires_io = bool(claim.get("likely_relevant_files"))
        cheating, cheat_reason = _verifier_appears_to_cheat(
            spec["code"], execution_text, requires_io=requires_io
        )
        if not cheating:
            return spec

        self.logger.warning(
            f"[verifier {claim['id']}] cheat detected, retrying once: {cheat_reason}"
        )
        retry_prompt = prompt + f"""

YOUR PREVIOUS ATTEMPT WAS REJECTED for the following reason:
  {cheat_reason}

Rewrite the verifier so that:
  - it does NOT contain any large string literal copied from the workflow
    output or the agent's answer,
  - it OPENS and reads the file(s) listed in likely_relevant_files,
  - it RECOMPUTES the value the claim asserts from the file contents,
  - it compares the recomputed value to the small, scalar target taken from
    the claim description (not to a pasted answer payload).

Return STRICT JSON in the same format as before.
"""
        retry_spec = self._call_and_parse_verifier(uuid, claim, retry_prompt, attempt=2)
        if not (retry_spec.get("executable") and retry_spec.get("code")):
            return {
                "executable": False,
                "reason": f"rejected as tautology (retry failed): {cheat_reason}",
            }
        cheating2, cheat_reason2 = _verifier_appears_to_cheat(
            retry_spec["code"], execution_text, requires_io=requires_io
        )
        if cheating2:
            self.logger.warning(
                f"[verifier {claim['id']}] retry also cheats ({cheat_reason2}); "
                f"falling back to soft check"
            )
            return {
                "executable": False,
                "reason": (
                    f"rejected as tautology after one retry; original: "
                    f"{cheat_reason}; retry: {cheat_reason2}"
                ),
            }
        return retry_spec

    def _call_and_parse_verifier(
        self,
        uuid: str,
        claim: dict[str, Any],
        prompt: str,
        attempt: int,
    ) -> dict[str, Any]:
        """Call the judge for a verifier spec; soft-fail to ``executable: False``."""
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

    def _run_verifier(self, uuid: str, claim_id: str, code: str) -> dict[str, Any]:
        """Execute a single verifier script in the agents' workspace."""
        scratch = self._runner_temp_root / uuid
        scratch.mkdir(parents=True, exist_ok=True)
        runner_config = RuntimeConfig(
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
        """Pull the last JSON line matching ``claim_id`` from the script stdout."""
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
        grounding: str = "",
    ) -> dict[str, Any]:
        """Narrow LLM verdict for one non-executable claim, anchored on grounding."""
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
- criticality: {claim['criticality']}
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

    # ------------------------------------------------------------------
    # Stage 5 — aggregation
    # ------------------------------------------------------------------

    def _information_bonus(self, n_hard_pass: int) -> float:
        """Saturating reward for thoroughness; gameable spam yields no extra credit.

        bonus(n) = α · (1 − exp(−n / β)). Bounded above by α, monotonic in n,
        and conditional on the *passing hard* claim count so trivial or failed
        claims contribute nothing.
        """
        if n_hard_pass <= 0 or self.info_bonus_alpha <= 0.0:
            return 0.0
        return self.info_bonus_alpha * (
            1.0 - math.exp(-n_hard_pass / self.info_bonus_beta)
        )

    def _aggregate(self, per_claim: list[dict[str, Any]]) -> dict[str, Any]:
        """Mean over scored claims + information bonus; hard-fail caps the result."""
        if not per_claim:
            return {
                "overall_score": 0.0,
                "overall_score_uncapped": 0.0,
                "base_mean": 0.0,
                "information_bonus": 0.0,
                "n_hard_pass": 0,
                "hard_fail_capped": False,
                "n_claims": 0,
                "n_pass": 0,
                "n_fail": 0,
                "n_error": 0,
                "n_unsure": 0,
                "n_scored": 0,
            }

        scored = [c for c in per_claim if c.get("status") != "error"]
        n_pass = sum(1 for c in per_claim if c["status"] == "pass")
        n_fail = sum(1 for c in per_claim if c["status"] == "fail")
        n_error = sum(1 for c in per_claim if c["status"] == "error")
        n_unsure = sum(1 for c in per_claim if c["status"] == "unsure")
        n_hard_pass = sum(
            1 for c in scored
            if c["claim"].get("criticality") == "hard" and c["status"] == "pass"
        )

        if not scored:
            return {
                "overall_score": 0.0,
                "overall_score_uncapped": 0.0,
                "base_mean": 0.0,
                "information_bonus": 0.0,
                "n_hard_pass": 0,
                "hard_fail_capped": False,
                "n_claims": len(per_claim),
                "n_pass": n_pass,
                "n_fail": n_fail,
                "n_error": n_error,
                "n_unsure": n_unsure,
                "n_scored": 0,
                "skipped_reason": "all_verifiers_errored",
            }

        base_mean = sum(c["score"] for c in scored) / len(scored)
        bonus = self._information_bonus(n_hard_pass)
        # Pre-cap: clamp to [0, 1] before applying the hard-fail cap so the
        # bonus can never push past 1.0 nor rescue a broken run.
        pre_cap = max(0.0, min(1.0, base_mean + bonus))
        # Hard-fail cap fires only on a real refutation, not on errors or unsure.
        hard_fail = any(
            c["claim"].get("criticality") == "hard"
            and c["status"] == "fail"
            for c in scored
        )
        overall = min(pre_cap, self.hard_fail_cap) if hard_fail else pre_cap

        return {
            "overall_score": round(overall, 4),
            "overall_score_uncapped": round(pre_cap, 4),
            "base_mean": round(base_mean, 4),
            "information_bonus": round(bonus, 4),
            "n_hard_pass": n_hard_pass,
            "hard_fail_capped": hard_fail,
            "n_claims": len(per_claim),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_error": n_error,
            "n_unsure": n_unsure,
            "n_scored": len(scored),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _list_workspace(self, max_entries: int = 200) -> str:
        """Workspace listing for the prompt; also populates ``_workspace_files``."""
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

    def _get_grounding(self, uuid: str, execution_text: str) -> str:
        """One Perspicacite round-trip per uuid; cached + opt-out."""
        if not self.use_grounding:
            return self._GROUNDING_DISABLED
        if uuid in self._grounding_cache:
            return self._grounding_cache[uuid]
        try:
            grounding = get_perspicacite_grounding(execution_text)
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

    @staticmethod
    def _looks_binary(sample: bytes) -> bool:
        """Heuristic: NUL bytes or > 30% non-printables → treat as binary."""
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
        """Cached LLM-friendly preview: text head+tail, binary magic bytes, fenced."""
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
        """Concatenate file previews under ``preview_per_claim_cap``."""
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

    def _call_judge(self, uuid: str, agent_name: str, prompt: str) -> str:
        """One judge round-trip; raises whatever the LLM provider raises."""
        memory_path = Path(self.memory_dir) / uuid
        memory_path.mkdir(parents=True, exist_ok=True)
        provider = LLMProvider(
            agent_name=agent_name,
            memory_path=memory_path,
            system_msg=self._get_judge_system_prompt(),
            config=self.llm_config,
        )
        return provider(prompt)

    _JSON_RETRY_FEEDBACK = (
        "Your previous response could not be parsed as JSON. Reply with ONLY "
        "the JSON object, no prose, no markdown fences, no commentary."
    )

    def _call_judge_for_json(
        self,
        uuid: str,
        agent_name: str,
        prompt: str,
    ) -> tuple[Any, str | None]:
        """Call the judge expecting JSON; one retry on parse failure.

        Returns ``(parsed, None)`` on success or ``(None, error_str)`` on
        terminal failure. Both call exceptions and JSON-parse errors are
        absorbed so callers never have to wrap in try/except.
        """
        last_err: str | None = None
        cur_prompt = prompt
        for attempt in (1, 2):
            agent = agent_name if attempt == 1 else f"{agent_name}_retry"
            try:
                raw = self._call_judge(uuid, agent, cur_prompt)
            except Exception as e:
                last_err = f"judge call failed: {type(e).__name__}: {e}"
                self.logger.warning(f"[{agent}] {last_err}")
                # Retrying when the call itself raised is unlikely to help; bail.
                return None, last_err

            payload = _extract_json_payload(raw or "")
            if payload:
                try:
                    return json.loads(payload), None
                except json.JSONDecodeError as e:
                    last_err = f"invalid JSON: {e}"
            else:
                last_err = "no JSON object found in response"

            if attempt == 1:
                self.logger.warning(
                    f"[{agent}] JSON parse failed ({last_err}); retrying once"
                )
                cur_prompt = (
                    f"{prompt}\n\nPREVIOUS ATTEMPT FAILED: {last_err}\n"
                    f"{self._JSON_RETRY_FEEDBACK}\n"
                )
        return None, last_err

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
                        f"unsure={scores.get('n_unsure', 0)}  "
                        f"scored={scores.get('n_scored', 0)}\n")
                f.write(f"Overall: {scores['overall_score']:.3f}"
                        f" (uncapped {scores.get('overall_score_uncapped', 0.0):.3f}, "
                        f"hard_fail_capped={scores.get('hard_fail_capped', False)})\n")
                f.write(f"  base_mean={scores.get('base_mean', 0.0):.3f}  "
                        f"information_bonus={scores.get('information_bonus', 0.0):.3f}  "
                        f"n_hard_pass={scores.get('n_hard_pass', 0)}\n")
                f.write("Note: errored verifiers are excluded from the mean and "
                        "do not trip the hard-fail cap.\n")
                f.write("Information bonus rewards thoroughness (passing hard "
                        "claims) with a saturating curve; gameable spam yields "
                        "no extra credit.\n\n")
                for c in per_claim:
                    cl = c["claim"]
                    f.write(f"[{cl['id']}] ({cl['criticality']}) {cl['description']}\n")
                    rel = cl.get("likely_relevant_files", [])
                    if rel:
                        f.write(f"  relevant_files: {rel}\n")
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

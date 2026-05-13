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

import ast
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
from .grounding import get_perspicacite_grounding

from sources.cli.pretty_print import (
    print_box, print_info,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)

# ----- Default per-script execution limits ------------------------------------
_VERIFIER_TIMEOUT_SECONDS = 60
_VERIFIER_MAX_CLAIMS = 24
_HARD_FAIL_CAP = 0.5

# ----- File preview budgets ---------------------------------------------------
# Per-file budget when including text content in the verifier-gen prompt. Files
# bigger than _PREVIEW_HEAD_BYTES + _PREVIEW_TAIL_BYTES are sent as
# "<head> ... (N bytes elided) ... <tail>" so the model still sees both ends
# (which is where format clues — headers, footers — usually live).
_PREVIEW_HEAD_BYTES = 8 * 1024
_PREVIEW_TAIL_BYTES = 2 * 1024
# Hard cap on combined preview size shipped per claim, so a multi-file claim
# can't blow up the prompt.
_PREVIEW_PER_CLAIM_CAP = 24 * 1024
# Bytes sniffed when deciding text vs binary.
_BINARY_SNIFF_BYTES = 4096

# ----- Anti-cheat thresholds --------------------------------------------------
# Minimum length of a string literal in the verifier code that we will bother
# checking against the workflow output. Anything shorter is too small to
# encode the agent's answer payload meaningfully.
_CHEAT_MIN_LITERAL_LEN = 80
# Length of the sliding window used to detect verbatim overlap between a
# verifier's string literals and the workflow output. 60 chars is large
# enough that legitimate constants ("HPHPPHHPHPPHPHHPPHPH", short JSON keys)
# don't trigger, but small enough to catch any real attempt to inline the
# agent's answer payload.
_CHEAT_OVERLAP_WINDOW = 60
# Tokens whose presence in the verifier code indicates real workspace I/O.
# Used to enforce "if you declared relevant files, you must read them."
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
    """Static scan of generated verifier code for tautology smells.

    Two distinct red flags, each independently sufficient:

    1. Large string literals lifted from the workflow output. If the verifier
       pastes a >=``_CHEAT_MIN_LITERAL_LEN``-char chunk of the agent's
       reported answer into a string and parses that, it is checking the
       answer against itself. We detect this with a sliding window of width
       ``_CHEAT_OVERLAP_WINDOW`` over each string literal and look for any
       window verbatim-present in the (whitespace-normalised) execution_text.

    2. Missing I/O. If the claim declared at least one relevant file but the
       verifier code contains no recognisable read primitive
       (``open(``, ``pd.read_``, ``json.load``, ...), it cannot possibly be
       grounded in workspace state.

    Returns ``(is_cheating, reason)``. ``reason`` is short, human-readable,
    and safe to feed back to the LLM as retry instructions.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Let the runner surface the syntax error itself; not our concern.
        return False, ""

    # Whitespace-normalise the workflow output once.
    exec_norm = " ".join(execution_text.split())

    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        lit = node.value
        if len(lit) < _CHEAT_MIN_LITERAL_LEN:
            continue
        lit_norm = " ".join(lit.split())
        # Sliding window with a step that covers all positions cheaply.
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
        preview_head_bytes: int = _PREVIEW_HEAD_BYTES,
        preview_tail_bytes: int = _PREVIEW_TAIL_BYTES,
        preview_per_claim_cap: int = _PREVIEW_PER_CLAIM_CAP,
        use_grounding: bool = True,
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
        self.preview_head_bytes = preview_head_bytes
        self.preview_tail_bytes = preview_tail_bytes
        self.preview_per_claim_cap = preview_per_claim_cap
        self.use_grounding = use_grounding
        # Per-instance cache of rendered file previews, keyed by relative path
        # string. Populated lazily on first read; survives for the lifetime of
        # the evaluator so multiple claims referencing the same file don't
        # re-read or re-render its bytes.
        self._preview_cache: dict[str, str] = {}
        # Per-instance cache of the workspace listing (rebuilt per evaluate()).
        self._workspace_files: set[str] = set()
        # Per-uuid Perspicacite grounding cache. One round-trip per workflow,
        # reused across Stage 1 (claim selection) and the soft check (Stage 4).
        self._grounding_cache: dict[str, str] = {}

        # Reuse the temp_dir setting from the global runner config when
        # available; otherwise fall back to a per-uuid scratch under workflow_dir.
        self._runner_temp_root = Path(
            getattr(config, "temp_dir", None) or self.workflow_dir / "_verifier_tmp"
        )
        self.logger.info(
            f"VerifierEvaluator initialized (workspace={self.workspace_dir}, "
            f"timeout={verifier_timeout}s, max_claims={max_claims}, "
            f"use_grounding={use_grounding})"
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

        # Short-circuit: workflow generation/execution totally failed.
        # Use the *artefact-existence* signal here, NOT the brittle
        # `success` flag from workflow_execution_text — that flag does
        # `not "[]" in json.dumps(answers)` and falsely flips False for
        # any successful run whose answer JSON happens to contain an
        # empty list (e.g. `{"warnings": [], "verdict": "PASS"}`).
        # The workspace_dir is shared across evolution generations, so
        # running verifier scripts on a run that produced no code AND no
        # state_result would silently score against whatever the previous
        # generation left behind. Return 0.0 immediately while still
        # emitting the report + state files so downstream readers see a
        # real-but-zero entry.
        wf_info = self._load_workflow_data(uuid)
        if not wf_info.state_result and not wf_info.code:
            return self._short_circuit_failed_run(uuid)

        workspace_listing = self._list_workspace()
        # Single Perspicacite round-trip per workflow; reused below.
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
        """Return a 0.0 verifier score without running any scripts.
        Used when the workflow produced no code AND no state_result.
        """
        print_box(
            f"workflow {uuid} produced no code and no state_result; "
            f"verifier returns 0.0 without running scripts. "
            f"(Avoids scoring against stale workspace from prior generations.)",
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
            # Failed runs have no artefacts to verify; record one hard claim.
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
- specific enough that a small Python script could potentially verify it
  against the workspace,
- not a meta-comment about the workflow ("the analysis was thorough").

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
            # Filter likely_relevant_files against the actual workspace listing.
            # This drops paths the agent's answer mentions confidently but that
            # don't exist on disk (typos, wrong directory, hallucinated names).
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
                # Accept only paths that the workspace walker actually saw.
                # `_workspace_files` is populated by `_list_workspace`.
                if self._workspace_files and rp not in self._workspace_files:
                    self.logger.debug(
                        f"Dropping confabulated relevant file '{rp}' for claim "
                        f"{c.get('id')}: not in workspace listing"
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
        """Generate, execute (if executable) and score a single claim.

        Emits diagnostic ``print_box`` panels around each stage so the user
        can see, at a glance:
          - which claim is being checked,
          - the verifier code (or reason for being marked non-executable),
          - the script's stdout/stderr/exit status,
          - the final verdict.
        """
        # 1. Header — the claim being verified
        rel_files = claim.get("likely_relevant_files", [])
        claim_text = (
            f"id:          {claim.get('id')}\n"
            f"criticality: {claim.get('criticality')}\n"
            f"description: {claim.get('description')}\n"
            f"files:       {rel_files if rel_files else '(none)'}"
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
            scored = self._score_soft(uuid, claim, execution_text, workspace_listing, reason, grounding)
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

        # Static cheat-check. If the LLM hard-coded the answer or skipped I/O,
        # demand a retry with explicit feedback. Cheats are common enough that
        # one retry is worth the LLM call; persistent cheats fall back to the
        # soft check (which is now grounded by file previews).
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
            # Retry produced no executable code — keep the rejection reason.
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
        """Single round-trip: call the judge, parse its JSON spec, return it.

        Errors are returned as ``{"executable": False, "reason": ...}`` rather
        than raised, so the caller can decide whether to retry or demote.
        """
        agent_name = (
            f"verifier_gen_{claim['id']}"
            if attempt == 1
            else f"verifier_gen_{claim['id']}_retry"
        )
        try:
            raw = self._call_judge(uuid, agent_name, prompt)
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
        grounding: str = "",
    ) -> dict[str, Any]:
        """Narrow LLM check for non-executable claims.
        The LLM is asked one targeted question (does this single claim hold
        given this concrete context?), not an aggregate vibes score.

        When ``grounding`` is available, the verdict is anchored against
        peer-reviewed literature in addition to the workspace context — the
        soft check is where literature has the most leverage, since these
        claims often concern methodological appropriateness rather than
        recomputable values.
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
        """Aggregate per-claim results into a final score.

        Key rule: ``status == "error"`` is treated as **non-signal**, not as a
        refutation. Verifier-script crashes (e.g. the LLM guessed the wrong
        file format and the parser bailed) tell us nothing about whether the
        underlying claim is true. Errors are therefore:
          - excluded from the mean,
          - excluded from the hard-fail-cap trigger,
          - reported separately so the operator can spot a flaky verifier
            without it dragging the workflow's score down.

        If every claim errored we fall back to ``overall_score = 0.0`` since
        we have no signal at all to report.
        """
        if not per_claim:
            return {
                "overall_score": 0.0,
                "overall_score_uncapped": 0.0,
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

        if not scored:
            # No usable signal. Don't pretend to score.
            return {
                "overall_score": 0.0,
                "overall_score_uncapped": 0.0,
                "hard_fail_capped": False,
                "n_claims": len(per_claim),
                "n_pass": n_pass,
                "n_fail": n_fail,
                "n_error": n_error,
                "n_unsure": n_unsure,
                "n_scored": 0,
                "skipped_reason": "all_verifiers_errored",
            }

        overall = sum(c["score"] for c in scored) / len(scored)
        # Hard-fail cap fires only on a real refutation of a hard claim, not
        # on a verifier-script crash and not on an unsure soft check.
        hard_fail = any(
            c["claim"].get("criticality") == "hard"
            and c["status"] == "fail"
            for c in scored
        )
        capped = min(overall, self.hard_fail_cap) if hard_fail else overall

        return {
            "overall_score": round(capped, 4),
            "overall_score_uncapped": round(overall, 4),
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
        """Return a short, deterministic listing of the workspace.

        Side effect: populates ``self._workspace_files`` with every relative
        path encountered (not just the first ``max_entries``), so the claim
        extractor's ``likely_relevant_files`` field can be intersected against
        a complete set later — even for workspaces that exceed the listing
        cap shown to the LLM.
        """
        ws = self.workspace_dir
        self._workspace_files = set()
        if not ws.exists():
            return "(workspace directory does not exist)"
        entries: list[str] = []
        truncated = False
        for root, dirs, files in os.walk(ws):
            # Skip noisy directories.
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
        """Return the (cached) literature grounding for *uuid*.

        Strategy:
          - Short-circuit when ``use_grounding`` is False (cheap dev runs).
          - One Perspicacite round-trip per uuid; reused for Stage 1 (claim
            selection) and the soft check.
          - Failures are absorbed by ``get_perspicacite_grounding`` itself
            (returns a fallback string), so the caller never has to handle
            an exception. We cache the fallback too — retrying within the
            same evaluator instance is unlikely to help.
        """
        if not self.use_grounding:
            return self._GROUNDING_DISABLED
        if uuid in self._grounding_cache:
            return self._grounding_cache[uuid]
        try:
            # Free function (no `self`); see sources/core/evaluators/grounding.py
            grounding = get_perspicacite_grounding(execution_text)
        except Exception as e:
            # Defensive: get_perspicacite_grounding already catches its own
            # errors, but we belt-and-suspenders so a Perspicacite outage
            # never sinks the whole verifier evaluation.
            self.logger.warning(f"Perspicacite grounding raised for {uuid}: {e}")
            grounding = f"{self._GROUNDING_FAILED_MARKER}: {e}"
        self._grounding_cache[uuid] = grounding
        # Surface grounding to the operator — green when it looks usable, red
        # when it's a failure marker. Truncated so it doesn't dominate logs.
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
        """Render an LLM-friendly preview of a workspace file.

        Strategy:
          - Cache results keyed on `rel_path`; identical lookups are free.
          - If the path is missing or not a file, say so explicitly.
          - Binary files: header line with size + first 16 bytes hex.
          - Text files ≤ head+tail budget: full content.
          - Text files larger than the budget: head + truncation marker + tail
            so format clues at both ends are visible.

        The returned string is wrapped in `=== <path> (...) ===` fences so the
        verifier-generation prompt can show several previews unambiguously.
        """
        if rel_path in self._preview_cache:
            return self._preview_cache[rel_path]

        # Resolve safely; reject anything escaping the workspace.
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

        # Read just enough to classify and (if text) to fill the head budget.
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
        """Render a multi-file preview block, capped at preview_per_claim_cap."""
        if not rel_paths:
            return "(no relevant files declared for this claim)"
        chunks: list[str] = []
        used = 0
        for rp in rel_paths:
            preview = self._preview_file(rp)
            # If adding this preview would blow the per-claim cap, truncate it
            # to whatever budget remains. If no budget remains, stop.
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
                        f"unsure={scores.get('n_unsure', 0)}  "
                        f"scored={scores.get('n_scored', 0)}\n")
                f.write(f"Overall: {scores['overall_score']:.3f}"
                        f" (uncapped {scores.get('overall_score_uncapped', 0.0):.3f}, "
                        f"hard_fail_capped={scores.get('hard_fail_capped', False)})\n")
                f.write("Note: errored verifiers are excluded from the mean and "
                        "do not trip the hard-fail cap.\n\n")
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

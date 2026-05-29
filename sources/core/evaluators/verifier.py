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
import subprocess
import sys
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
    print_box, print_info, print_ok, print_step, print_warn,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)

# ----- Execution limits -------------------------------------------------------
_VERIFIER_TIMEOUT_SECONDS = 180
_VERIFIER_MAX_CLAIMS = 35
_VERIFIER_MIN_CLAIMS = 15
_HARD_FAIL_CAP = 0.99 # temporary to disable so signal stay smooth

# ----- Information bonus (rewards thoroughness; saturates) --------------------
# bonus(n) = alpha * (1 - exp(-n_hard_pass / beta)); see _aggregate.
_INFO_BONUS_ALPHA = 0.05
_INFO_BONUS_BETA = 8.0

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

# ----- Empty-run marker -------------------------------------------------------
# Emitted by base.workflow_execution_text when no state_result and no code exist.
# We use this as the precise empty-run signal, NOT the brittle `success` flag
# (which trips False on any successful run whose answers JSON contains "[]").
_EMPTY_RUN_MARKER = "workflow execution fully failed"

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
)
_VERIFIER_PACKAGES_INSTALLED = False
_VERIFIER_INSTALL_LOCK = threading.Lock()


# ----- Claim extraction: rules shared across all three source prompts ---------
_CLAIM_RULES_BLOCK = """For each claim, also estimate `criticality`:
- "hard": load-bearing for the answer (final metrics, headline files,
  required computations, claimed satisfaction of the user goal, required
  methodology steps that are the only path to success according to the literature).
- "soft": supporting context (intermediate sanity remarks, choices that are
  defensible but not strictly required, methods decisions the literature cites).

For each claim, also list `likely_relevant_files`: relative paths whose
contents the verifier would need to read in order to check the claim.
- ONLY use paths that appear verbatim in the WORKSPACE FILES listing above.
  Do not invent or guess paths the workflow's answer mentions but that are
  not in the listing.
- Use `[]` if the claim is purely about the workflow's output text and has
  no on-disk artefact to consult.

POLARITY (mandatory). Every claim is a POSITIVE SUCCESS ASSERTION about what
the workflow ACHIEVED scientifically. A claim is well-formed only if
"verified TRUE" is equivalent to "the workflow succeeded at this aspect".
Never extract a claim that a FAILURE MODE would satisfy. Extract the
success condition the workflow failed: a workflow that produced no usable
answer should FAIL the claim "produced <the deliverable, meeting <the
bar>>", not pass the claim "the final answer is empty".

DISCRIMINATION TEST. Before including any "hard" claim, ask: "If the workflow
had done nothing scientifically meaningful — only moved files, saved a
checkpoint, but never produced a correct and complete answer — would this
claim still verify TRUE?" If YES: the claim is worthless. Reframe it into the
functional success condition it is a proxy for, or drop it.

ARTIFACT CLAIMS — STRICT. Bare file-existence or file-size claims are NOT
scientific achievements and are NEVER "hard". Extract an artifact claim only
chained to a functional property that makes it load-bearing — not
"predictions.csv exists" but "predictions.csv contains a valid probability in
[0,1] for every row of the test set". Maximum 2 soft artifact claims total.

Return STRICT JSON only, no prose, in this exact form:
{
  "claims": [
    {
      "id": "<short_slug>",
      "description": "<concise restatement of the claim>",
      "criticality": "hard" | "soft",
      "likely_relevant_files": ["<relative/path>", ...]
    },
    ...
  ]
}
"""


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
        use_cheat_detector: bool = True,
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
        self.use_cheat_detector = use_cheat_detector
        self._preview_cache: dict[str, str] = {}
        self._workspace_files: set[str] = set()
        self._grounding_cache: dict[str, str] = {}
        self._runner_temp_root = Path(
            getattr(config, "temp_dir", None) or self.workflow_dir / "_verifier_tmp"
        )
        self._task_spec: str = ""
        self.logger.info(
            f"VerifierEvaluator initialized (workspace={self.workspace_dir}, "
            f"timeout={verifier_timeout}s, claims={self.min_claims}–{max_claims}, "
            f"use_grounding={use_grounding}, "
            f"info_bonus(α={self.info_bonus_alpha}, β={self.info_bonus_beta}))"
        )
        self._prompt_gradient_history: list[str] = []


    def _build_abstracted_prompt_gradient(self, uuid, report: str) -> str:
        """Residual signal passed to provide directional signal to orchestrator - avoid Goodhart's cheating"""
        history = "\n".join(self._prompt_gradient_history[-5:])  # include recent prompt_gradient history for context, up to 5 past runs
        prompt = f"""
        You must summarise the judge's detailed report into a concise prompt_gradient of the agents's behavior and failure modes, in plain language that a human user can understand.
        The prompt_gradient should be actionable and focused on the most critical issues affecting the workflow's performance, especially those that caused hard claim failures or a cheat penalty.
        The prompt_gradient should not leak what the verifier score against (e.g. "the workflow failed to read the file 'data.csv'"), but should still convey the core issues in a way to give an overall sense of what went wrong.
        The prompt_gradient could mention anything forbidden that contributed to failure such as fallback, short, hacks, or cheating.
        The prompt_gradient does not suggest solutions.
        Here is the verifier's detailed report for workflow {uuid}:
        {report}
        Here are the past diagnoses for recent workflows, which may provide additional context on common failure:
        {history}
        Make a short code name for the prompt_gradient followed by a one sentence prompt_gradient of the workflow's behavior and failure modes, focused on the most critical issues, without mentioning specific claim verdicts or scores.
        Format: "<prompt_gradient_CODE>:<one-sentence prompt_gradient>"
        If possible, reuses prompt_gradient codes from past runs when the failure modes are similar, to help track recurring issues.
        Example:
        FALLBACK_ECFP_CLASSIFIER:The workflow produced a correctly shaped prediction table, but it appears to use a fallback rather than a trained ECFP-based classifier.
        """
        diag = self._call_judge(
            uuid,
            "verifier_abstract_prompt_gradient",
            prompt,
        )
        self._prompt_gradient_history.append(diag)
        return diag.strip() or "UNDIAGNOSED:No prompt_gradient could be extracted from the verifier report."

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
        if not wf_info.state_result or not wf_info.code:
            return self._short_circuit_failed_run(uuid)

        workspace_listing = self._list_workspace()
        # Use explicit empty-run marker rather than the brittle `success` flag
        # — `success` is `not "[]" in json.dumps(answers)`, which mis-fires
        # whenever an answer payload contains a (possibly nested) empty list.
        is_truly_empty = (
            not execution_text or _EMPTY_RUN_MARKER in execution_text
        )
        grounding = (
            self._get_grounding(uuid, execution_text, wf_info.goal)
            if not is_truly_empty
            else self._GROUNDING_DISABLED
        )
        claims = self._extract_claims(
            uuid, wf_info.goal, execution_text, workspace_listing, is_truly_empty, grounding
        )
        if not claims:
            self.logger.warning(f"No claims extracted for {uuid}; verifier returns 0.0")
            scores = {"overall_score": 0.0, "n_claims": 0, "n_pass": 0, "n_fail": 0}
            self._save_results(scores, uuid, "verifier")
            return {"uuid": uuid, "claims": [], **scores}

        self._ensure_verifier_packages()

        per_claim: list[dict[str, Any]] = []
        for claim in claims[: self.max_claims]:
            result = self._verify_claim(
                uuid, claim, execution_text, workspace_listing, grounding
            )
            per_claim.append(result)

        scores = self._aggregate(per_claim)

        # Layer 3: independent cheat audit over the agents' produced script.
        cheat = None # NOTE: cheat_detector was crap. Will need to be rethink.

        self._write_report(uuid, claims, per_claim, scores, cheat=cheat)

        report = self._build_report(per_claim, scores, cheat)
        prompt_gradient = self._build_abstracted_prompt_gradient(uuid, report)
        scores["abstracted_prompt_gradient"] = prompt_gradient
        self._persist_prompt_gradient(uuid, prompt_gradient)

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
            "abstracted_prompt_gradient": "workflow code failed to generate or execute; ensure code is properly formatted and that the workflow runs without crashing",
            "cheat_penalty": 0.0,
        }
        try:
            self._write_report(uuid, [], [], scores, cheat=None)
        except Exception as e:
            self.logger.error(f"Failed to write short-circuit report for {uuid}: {e}")
        self._persist_prompt_gradient(uuid, "")
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
        goal: str,
        execution_text: str,
        workspace_listing: str,
        is_truly_empty: bool,
        grounding: str = "",
    ) -> list[dict[str, Any]]:
        """Extract atomic claims by polling three independent source prompts.
        Source A asks an LLM for what the LITERATURE demands of a correct
        solution; Source B asks what the USER explicitly required in the goal
        text; Source C asks what the AGENTS reported doing in their narration.
        """
        if is_truly_empty:
            return [{
                "id": "c0_execution_succeeded",
                "description": "The workflow executed to completion and produced a non-empty answer.",
                "criticality": "hard",
                "likely_relevant_files": [],
            }]
        per_source_min, per_source_max = self._per_source_targets(n_sources=6)
        sources = (
            ("a", self._build_source_a_prompt(goal, grounding, workspace_listing, per_source_min, per_source_max)),
            ("b", self._build_source_b_prompt(goal, workspace_listing, per_source_min, per_source_max)),
            ("c", self._build_source_c_prompt(goal, execution_text, workspace_listing, per_source_min, per_source_max)),
            ("d", self._build_source_d_prompt(goal, workspace_listing, per_source_min, per_source_max)),
            ("e", self._build_source_e_prompt(goal, workspace_listing, per_source_min, per_source_max)),
            ("f", self._build_source_f_prompt(goal, execution_text, workspace_listing, per_source_min, per_source_max)),
        )

        merged: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for label, prompt in sources:
            data, err = self._call_judge_for_json(
                uuid, f"verifier_extract_claims_{label}", prompt
            )
            if err is not None:
                self.logger.warning(
                    f"Claim extraction source {label} failed for {uuid}: {err}"
                )
                continue
            for claim in self._parse_and_filter_claims(uuid, data):
                print_info(f"Extracted claim {claim['id']} from source {label} for {uuid}")
                claim_id = claim["id"]
                if claim_id in seen_ids:
                    claim_id = f"{claim_id}_{label}"
                claim["id"] = claim_id
                claim["source"] = f"source_{label}"
                seen_ids.add(claim_id)
                merged.append(claim)

        print_ok(f"Extracted claims for workflow {uuid} from {len(sources)} sources...")
        if len(merged) < self.min_claims:
            print_warn("Claim extraction yielded fewer than the minimum required claims ")
            self.logger.warning(
                f"Claim extraction yielded only {len(merged)} claims "
                f"(min_claims={self.min_claims}); proceeding with what we got"
            )
        return merged

    def _per_source_targets(self, n_sources: int = 3) -> tuple[int, int]:
        """Per-source min/max claim targets derived from the global bounds.

        Scaled by the number of extraction sources so the union still respects
        ``self.max_claims`` without starving later sources.
        """
        n_sources = max(1, n_sources)
        per_min = max(2, self.min_claims // n_sources)
        per_max = max(per_min, max(2, self.max_claims // n_sources))
        return per_min, per_max

    def _build_source_a_prompt(
        self,
        goal: str,
        grounding: str,
        workspace_listing: str,
        target_min: int,
        target_max: int,
    ) -> str:
        """Source A — what the LITERATURE demands of a correct solution."""
        grounding_block = grounding.strip() if grounding else "(no literature grounding available)"
        return f"""You are extracting SOURCE A claims for a verification rubric: requirements the peer-reviewed literature places on any correct solution to this task, independent of what the agents actually did.

WORKFLOW GOAL:
{goal}

LITERATURE GROUNDING:
{grounding_block}

WORKSPACE FILES (relative to workspace root):
{workspace_listing}

TASK:
Extract Source-A claims. These are the things the LITERATURE demands of a
correct solution, regardless of whether the agents performed them:
- Required methodology steps (e.g. "data was normalised before PCA",
  "cross-validation was performed with k≥5", "the energy minimisation
  converged to a stationary point").
- Required outputs / quality bars the field treats as load-bearing
  (e.g. "the predicted structure has RMSD ≤ X to the reference",
  "the regression model reports an R² on a held-out test set").
- Required constraints / sanity properties standard in the field
  (e.g. "probabilities sum to 1", "the contact matrix is symmetric",
  "the conformation is a valid self-avoiding walk").

MANDATORY GOAL CLAIM. The first claim MUST assert that the workflow
produced the specific scientific deliverable the task requested AND that
it meets the literature-standard success criterion. If the task names a
quantitative bar (accuracy ≥ x, energy ≤ y, AUC ≥ z, p < α), this claim
must encode that bar — not merely "a result exists". Phrase it so a
workflow that skipped, faked, or left the deliverable empty FAILS it.
Mark it "hard". Source-A claims about literature-required steps are
"hard" by default.

{_CLAIM_RULES_BLOCK}

Aim for {target_min}–{target_max} Source-A claims.
"""

    def _build_source_b_prompt(
        self,
        goal: str,
        workspace_listing: str,
        target_min: int,
        target_max: int,
    ) -> str:
        """Source B — what the USER explicitly required in the goal text."""
        return f"""You are extracting SOURCE B claims for a verification rubric: requirements the user explicitly stated in the workflow goal, independent of what the literature would have demanded and independent of what the agents actually did.

WORKFLOW GOAL:
{goal}

WORKSPACE FILES (relative to workspace root):
{workspace_listing}

TASK:
Read ONLY the goal text above. Extract claims that capture instructions
and deliverables the user spelled out. A Source-B claim FAILS if the
agents skipped, weakened, or substituted what the user asked for — even
if the literature would have accepted the substitution.

Look for, in the goal:
- Explicit deliverables ("produce a CSV with columns A,B,C", "save the
  trained model to disk", "render a phylogenetic tree as SVG").
- Explicit method / tool choices ("use random forest with 100 trees",
  "run BLAST against the nr database", "fit with sklearn's PCA").
- Explicit numeric or qualitative success bars ("accuracy ≥ 90%",
  "energy ≤ −20 kJ/mol", "p < 0.05", "all residues classified").
- Explicit comparisons or controls ("compare against a random baseline",
  "include a negative control", "report both train and test metrics").
- Explicit scope constraints ("over the 2020–2024 window", "for the
  test split only", "use the 20-mer sequence HPHPPHHPHPPHPHHPPHPH").
- Explicit output format constraints ("as JSON", "one row per sample",
  "rounded to 3 decimal places").
- Input dataset's exact column names, order, and data types in your output. Ensure agents don't add suffixes (e.g., _prob, _score) or rename columns unless the task explicitly specifies a different output schema. Any deviation from the source format is an error.
- Any hint, advice, recommandations, treat them as explicit user requirements that must be followed



If the goal is short and contains few explicit requirements, return a
short list — DO NOT pad with claims the user did not write. It is fine
to return fewer than {target_min} claims when the goal is terse; do not
invent constraints.

{_CLAIM_RULES_BLOCK}

Aim for up to {target_max} Source-B claims, but only as many as the goal
text actually warrants.
"""

    def _build_source_c_prompt(
        self,
        goal: str,
        execution_text: str,
        workspace_listing: str,
        target_min: int,
        target_max: int,
    ) -> str:
        """Source C — what the AGENTS reported doing in their narration."""
        return f"""You are extracting SOURCE C claims for a verification rubric: concrete computations and artefacts the agents reported producing, so the verifier can check the agents did not lie or hallucinate.

WORKFLOW GOAL:
{goal}

WORKFLOW OUTPUT (agents narration — the workflow's self-report):
{execution_text}

WORKSPACE FILES (relative to workspace root):
{workspace_listing}

TASK:
Extract Source-C claims: things the agents CLAIM to have done, that we can verify against the
on-disk artefacts or using calculations. A Source-C claim FAILS if what the agents reported
cannot be reproduced from the files they wrote. Methodological claims are NOT source C - do not include them.

Look for, in the narration and workspace:
- Concrete computations the agents reported (specific numbers, metrics,
  intermediate values, decisions made) that can be verified with small calculations in python.
- Workspace changes the agents claim to have produced (files written,
  formats used, structural properties of outputs).
Do not:
- Do not extract aspirational claims about what the agents "tried" to do or "considered" doing.
- Do not extract claims about fallback or any alternative paths that is trying to "hack" the solutions.
- Do not extract claims that would require parsing a python program to verify.

Ignore aspirational language and meta-narration ("we tried", "we
considered"); extract only verifiable assertions about state on disk
or computed results.

{_CLAIM_RULES_BLOCK}

Aim for {target_min}–{target_max} Source-C claims.
"""

    def _build_source_d_prompt(
        self,
        goal: str,
        workspace_listing: str,
        target_min: int,
        target_max: int,
    ) -> str:
        """Source D — mathematical sanity properties of the produced artefacts."""
        return f"""You are extracting SOURCE D claims for a verification rubric: closed-form mathematical sanity properties any correct solution to this task must satisfy, derivable from the TYPE of objects the task produces — independent of the literature, the user wording, and what the agents reported.

WORKFLOW GOAL:
{goal}

WORKSPACE FILES (relative to workspace root):
{workspace_listing}

TASK:
Extract Source-D claims. These are mathematical invariants and structural
properties that follow from the type of object produced and that a small
numerical check can confirm directly against the on-disk artefact. A
Source-D claim FAILS if the artefact violates a property any correct
solution would have respected.

Look for properties such as:
- Probability constraints (values in [0,1]; rows of a probability matrix
  sum to 1; class probabilities non-negative).
- Matrix / tensor properties (symmetry of distance or covariance matrices;
  positive semi-definiteness of covariance; zero diagonal of distance
  matrices; triangle inequality; correct shapes / dimensions).
- Numerical sanity (no NaN, no infinity, no negative variances, no
  negative counts, no out-of-domain values for log/sqrt).
- Conservation, monotonicity, dimensional consistency (an energy below a
  physical upper bound; cumulative distributions monotonic; unit
  consistency between inputs and outputs).
- Structural validity (a self-avoiding walk has no repeated coordinates;
  a tree on n nodes has n-1 edges; an alignment has matching sequence
  lengths; a graph's adjacency matrix matches its edge list).
- Cardinality / shape consistency (output row count matches input row
  count on a per-row task; predictions equal the test set size; feature
  counts agree across train and test).

Do NOT extract:
- Methodology choices — those are Source A.
- Literal user-required values — those are Source B.
- Things the agents merely claim — those are Source C.
- Bare existence of files — forbidden by the artifact-claim rule below.

Prefer claims that can be checked with a tiny numpy / pandas script
reading the relevant artefact. Most Source-D claims are "hard" by default:
violating a mathematical invariant means the result is not just
suboptimal, it is incorrect.

{_CLAIM_RULES_BLOCK}

Aim for {target_min}–{target_max} Source-D claims, but only ones grounded
in the actual artefacts visible in the workspace listing. Do not invent
properties for objects the task does not produce.
"""

    def _build_source_e_prompt(
        self,
        goal: str,
        workspace_listing: str,
        target_min: int,
        target_max: int,
    ) -> str:
        """Source E — non-negotiable computational reproducibility / CS practice."""
        return f"""You are extracting SOURCE E claims for a verification rubric: NON-NEGOTIABLE computational reproducibility requirements an independent computer scientist would demand to re-run this work on a fresh machine — independent of the science, the user wording, and the agents' narration.

WORKFLOW GOAL:
{goal}

WORKSPACE FILES (relative to workspace root):
{workspace_listing}

TASK:
Extract Source-E claims that capture HARD computational-reproducibility
requirements. The scope is intentionally narrow: only things without
which a second party CANNOT re-run this work on a fresh machine. The
bar is "can it be re-run", NOT "is it nicely engineered".

ALLOWED claim shapes (each MUST chain a file / structural property to a
functional reproducibility consequence — never bare existence):
- The workspace declares its dependencies in a standard manifest
  (`requirements.txt`, `pyproject.toml`, or `environment.yml`) AND the
  declared packages cover the third-party imports actually used by the
  produced code — i.e. the manifest is non-empty and is not missing a
  library that the workspace's `.py` files import.
- The produced code contains no hard-coded absolute filesystem paths
  outside the workspace (no `/home/...`, no `/Users/...`, no `C:\\...`)
  that would break on another machine.
- If the produced code uses stochastic operations (random sampling,
  shuffling, model training, weight init, train/test split), a random
  seed is fixed in code (`numpy.random.seed`, `random.seed`,
  `torch.manual_seed`, `random_state=...`) so the run is reproducible.
- A clearly identifiable runnable entrypoint exists (a single top-level
  `.py` such as `main.py`, `run.py`, `pipeline.py`, or unambiguous from
  the layout) so a re-runner knows what to launch.
- The workspace is not pathologically cluttered with junk (no thousands
  of unrelated files; no obvious accumulation of failed intermediate
  dumps that would confuse a re-runner).
- Outputs are written to relative paths inside the workspace, not to
  system or user-home locations.

EXPLICITLY FORBIDDEN — DO NOT extract claims about any of these:
- README files, documentation, markdown, or doc presence of any kind.
- Docstrings, comments, or in-code documentation.
- Tests, test coverage, or test presence.
- Code style (PEP8, line length, naming conventions, formatting).
- Type hints / type annotations.
- Logging structure, log file presence, or log verbosity.
- Module organisation, package layout, "clean architecture".
This source verifies non-negotiable computer-science PRACTICE — not
engineering aesthetics. If a property is merely "nice to have", drop it.

Each Source-E claim MUST chain a file / structural property to a
functional reproducibility consequence — never bare existence. Example
WELL-FORMED claim: "the workspace declares its dependencies in a standard
manifest covering the packages actually imported by the produced code".
Example MALFORMED claim: "a requirements.txt file exists in the workspace".

Use "hard" criticality ONLY for the deps-manifest, absolute-paths, and
seed-on-stochastic claims — those genuinely block re-execution. Use
"soft" for the entrypoint, clutter, and output-location claims.

{_CLAIM_RULES_BLOCK}

Aim for up to {target_max} Source-E claims, but only as many as the
workspace actually warrants — fewer is fine. Do not pad.
"""

    def _build_source_f_prompt(
        self,
        goal: str,
        execution_text: str,
        workspace_listing: str,
        target_min: int,
        target_max: int,
    ) -> str:
        """Source F — statistical fingerprint / non-triviality of the result."""
        return f"""You are extracting SOURCE F claims for a verification rubric: statistical-fingerprint and non-triviality checks that distinguish a REAL scientific result from a vacuous, degenerate, or leakage-inflated one — independent of the literature, the user wording, and the agents' narration.

WORKFLOW GOAL:
{goal}

WORKFLOW OUTPUT (agents narration — names the headline metrics they report):
{execution_text}

WORKSPACE FILES (relative to workspace root):
{workspace_listing}

TASK:
Extract Source-F claims. These check that the produced result is
NON-TRIVIAL and STATISTICALLY REAL — i.e. that it could not have been
achieved by a degenerate, leaking, or hard-coded "solution". A Source-F
claim FAILS if the on-disk artefact bears the fingerprint of a vacuous
success.

Look for properties such as:
- The headline metric beats a trivial baseline by a non-trivial margin
  (random / majority-class / mean predictor / shuffled-label baseline);
  on a balanced binary task, accuracy is above 0.55; on a regression
  task, the model beats the mean predictor in R² or RMSE.
- The prediction distribution is not degenerate: not constant, not all
  one class, not a single value repeated, not uniformly 0.5, with non-zero
  variance across rows in continuous outputs.
- No data-leakage signatures: train and test sets are disjoint (no
  overlapping IDs or rows); the test set is not a subset of training data;
  perfect or near-perfect scores on a known-hard task are flagged as
  suspect unless the artefact explicitly justifies them.
- No suspicious hard-coded or fallback patterns in outputs (predictions
  all identical, all integers when probabilities were expected, exact
  reproduction of an input column as the "prediction").
- Sample sizes are adequate for the test (n above a sensible floor for
  the statistic being claimed; enough samples per class for stratified
  metrics).
- Where probabilities are produced, they show inter-class separation
  rather than collapsing to a single point.

Do NOT extract:
- Methodology requirements — Source A.
- Literal user-required values — Source B.
- Things the agents merely report — Source C.
- Mathematical invariants like "probabilities in [0,1]" — Source D.
- Reproducibility / CS-hygiene properties — Source E.

Source-F claims are typically "hard" when they target the headline
result: a result statistically indistinguishable from a baseline is not
a scientific success. Skip baseline claims for tasks with no obvious
null to compare against — do not invent one.

{_CLAIM_RULES_BLOCK}

Aim for {target_min}–{target_max} Source-F claims, only as many as the
on-disk artefacts can actually support.
"""

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
            raw_files = c.get("likely_relevant_files") or []
            relevant = self._validate_workspace_paths(
                raw_files,
                allowed=self._workspace_files or None,
                label=str(c.get("id") or f"c{idx}"),
            )
            cleaned.append({
                "id": str(c.get("id") or f"c{idx}"),
                "description": str(c["description"]).strip(),
                "criticality": "hard" if c.get("criticality") == "hard" else "soft",
                "likely_relevant_files": relevant,
            })
        return cleaned

    def _validate_workspace_paths(
        self,
        raw: Any,
        allowed: set[str] | None = None,
        max_count: int | None = None,
        label: str = "",
    ) -> list[str]:
        """Normalise, dedupe and validate a list of workspace-relative paths.

        Strips leading ``./``, drops empties, duplicates and non-strings. When
        ``allowed`` is given, paths not in that set are logged and dropped
        (hallucination guard). When ``max_count`` is given, truncates to that
        cap. Returns paths in input order.
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
        rel_files = self._llm_select_files(
            uuid, claim, execution_text, workspace_listing
        )
        claim = {**claim, "likely_relevant_files": rel_files}
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
            #print_box(code, title=f"Verifier preview · {claim.get('id')}", color=YELLOW, truncate=256)
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

Return STRICT JSON only:
  {{"files": ["<relative/path>", ...]}}
"""

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
- Use only the standard library plus the verifier helper packages
  (numpy, pandas, scipy, scikit-learn). Read files with relative paths
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

Return STRICT JSON only, in one of these two shapes:
  {{"executable": true,  "code": "<full python script as one string>"}}
  {{"executable": false, "reason": "<one sentence>"}}
"""
        spec = self._call_and_parse_verifier(uuid, claim, prompt, attempt=1)
        if not (spec.get("executable") and spec.get("code")):
            return spec
        return spec

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

    # Python module names corresponding to ``_VERIFIER_BASE_PACKAGES``
    # (scikit-learn → sklearn). Used by the post-install smoke check.
    _VERIFIER_BASE_IMPORTS: tuple[str, ...] = ("numpy", "pandas", "scipy", "sklearn")

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
                "import " + ", ".join(self._VERIFIER_BASE_IMPORTS),
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
        """Return True iff *cmd* exits 0 within *timeout*."""
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except Exception:
            return False
        return r.returncode == 0

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
        # Use the Python that's running Mimosa, not the system python3.12 the
        # runner's resolver picks: that interpreter is where the verifier
        # helper packages were installed.
        runner._python_cmd = [sys.executable]
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

    def _get_grounding(self, uuid: str, execution_text: str, goal: str) -> str:
        """One Perspicacite round-trip per uuid; cached + opt-out."""
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

    def _eligible_workspace_files(self) -> list[str]:
        """Workspace files plausibly readable as text artefacts (sorted).

        Filters out compiled artefacts and obvious binaries by suffix; the
        deeper magic-byte check inside ``_preview_file`` still catches
        anything that slips through.
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

    @staticmethod
    def _apply_cheat_penalty(
        scores: dict[str, Any], cheat
    ) -> dict[str, Any]:
        """Subtract cheat penalty from the capped overall score; floor at 0.0."""
        penalty = float(cheat.penalty) if cheat is not None else 0.0
        capped = float(scores.get("overall_score", 0.0))
        final = max(0.0, capped - penalty)
        scores["overall_score_before_cheat"] = round(capped, 4)
        scores["cheat_penalty"] = round(penalty, 4)
        scores["overall_score"] = round(final, 4)
        if cheat is not None:
            scores["cheat_detector"] = cheat.to_dict()
        return scores

    # ------------------------------------------------------------------
    # Layer 1 — abstracted prompt_gradient for the mutator
    # ------------------------------------------------------------------


    @staticmethod
    def _fallback_prompt_gradient(
        scores: dict[str, Any], cheat
    ) -> str:
        """Deterministic fallback when the abstractor LLM is unavailable."""
        n_pass = scores.get("n_pass", 0)
        n_fail = scores.get("n_fail", 0)
        n_claims = scores.get("n_claims", 0)
        overall = scores.get("overall_score", 0.0)
        bits = [
            f"Run scored {overall:.2f}; "
            f"{n_pass}/{n_claims} checks passed, {n_fail} refuted."
        ]
        if scores.get("hard_fail_capped"):
            bits.append(
                "A load-bearing requirement was not met — the next iteration "
                "must change approach rather than refine details."
            )
        if cheat is not None and cheat.behavioral:
            bits.append(
                "An independent audit flagged shortcuts in the produced code: "
                + "; ".join(cheat.behavioral[:3])
                + "."
            )
        return " ".join(bits)

    def _persist_prompt_gradient(self, uuid: str, prompt_gradient: str) -> None:
        """Write the prompt_gradient to ``prompt_gradient.txt`` alongside the report."""
        if not prompt_gradient:
            return
        path = self.workflow_dir / uuid / "prompt_gradient.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(prompt_gradient, encoding="utf-8")
        except OSError as e:
            self.logger.warning(f"could not write prompt_gradient.txt for {uuid}: {e}")

    def _build_report(
        self,
        per_claim: list[dict[str, Any]],
        scores: dict[str, Any],
        cheat
    ) -> str:
        lines: list[str] = []
        w = lines.append

        w("Verifier Evaluation")
        w("=" * 60)
        w(
            f"Claims: {scores['n_claims']}  pass={scores['n_pass']}  "
            f"fail={scores['n_fail']}  error={scores.get('n_error', 0)}  "
            f"unsure={scores.get('n_unsure', 0)}  "
            f"scored={scores.get('n_scored', 0)}"
        )
        w(
            f"Overall: {scores['overall_score']:.3f}"
            f" (pre-cheat {scores.get('overall_score_before_cheat', scores['overall_score']):.3f}, "
            f"uncapped {scores.get('overall_score_uncapped', 0.0):.3f}, "
            f"hard_fail_capped={scores.get('hard_fail_capped', False)})"
        )
        w(
            f"  base_mean={scores.get('base_mean', 0.0):.3f}  "
            f"information_bonus={scores.get('information_bonus', 0.0):.3f}  "
            f"n_hard_pass={scores.get('n_hard_pass', 0)}  "
            f"cheat_penalty={scores.get('cheat_penalty', 0.0):.3f}"
        )

        for c in per_claim:
            cl = c["claim"]
            w(f"[{cl['id']}] ({cl['criticality']}) {cl['description']}")
            rel = cl.get("likely_relevant_files", [])
            if rel:
                w(f"  relevant_files: {rel}")
            w(f"  kind={c['verifier_kind']} status={c['status']} score={c['score']}")
            if c.get("details"):
                w(f"  details: {c['details']}")
            if c.get("verifier_kind") == "executable" and c.get("raw_stderr"):
                stderr_snippet = c["raw_stderr"].strip().splitlines()[-5:]
                if stderr_snippet:
                    w("  stderr (tail):")
                    for line in stderr_snippet:
                        w(f"    {line}")
            w("")

        if cheat is not None:
            w("-" * 60)
            w("Independent cheat audit")
            w(f"  penalty: {cheat.penalty:.3f}")
            if cheat.error:
                w(f"  error:   {cheat.error}")
            if cheat.behavioral:
                w("  behavioral findings (also fed to mutator):")
                for b in cheat.behavioral:
                    w(f"    - {b}")
            if cheat.mechanism:
                w("  mechanism findings (audit-only — NOT fed to mutator):")
                for m in cheat.mechanism:
                    w(f"    - {m}")
            w("")

        return "\n".join(lines) + "\n"


    def _write_report(
        self,
        uuid: str,
        claims: list[dict[str, Any]],
        per_claim: list[dict[str, Any]],
        scores: dict[str, Any],
        cheat
    ) -> None:
        path = self.workflow_dir / uuid / "evaluation.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        report = self._build_report(per_claim, scores, cheat)
        try:
            path.write_text(report, encoding="utf-8")
            self.logger.info(f"Verifier report written to {path}")
        except OSError as e:
            self.logger.error(f"Could not write verifier report for {uuid}: {e}")

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # noqa: E402
    from config import Config
    config = Config()
    verifier = VerifierEvaluator(config, config.workspace_dir)
    verifier.evaluate("20260528_090358_ff330fde")
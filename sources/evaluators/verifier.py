"""Per-claim verifier-based workflow evaluator (orchestrator).

This module owns the public ``VerifierEvaluator`` class and the pipeline that
turns a workflow run into a numeric score plus a textual gradient. The
mechanical steps — extracting claims, generating and running per-claim
scripts, listing the workspace, rendering file previews — live in sibling
modules and are mixed in:

* .verifier_claims — claim extraction (six sources) + importance rating.
* .verifier_per_claim — verifier-script generation, sandbox execution,pass/fail/error scoring.
* .verifier_workspace — workspace listing, file previews, literature grounding cache.
"""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sources.cli.pretty_print import (
    CYAN,
    RED,
    GREEN,
    print_box,
    print_ok,
)

from sources.core import declared_outputs
from sources.core.failure_fingerprint import (
    DESCRIPTOR_DIM as _FP_DIM,
    compute_failure_fingerprint,
)
from sources.core.llm_provider import LLMConfig

from .base import (
    BaseEvaluator,
    EvaluatorError,
    WorkflowDataError,
)
from .verifier_claims import _VerifierClaimExtractionMixin
from .verifier_per_claim import _VerifierPerClaimMixin
from .verifier_workspace import _VerifierWorkspaceMixin
from sources.evaluators.grounding import get_perspicacite_grounding

# ----- Execution limits -------------------------------------------------------
_VERIFIER_TIMEOUT_SECONDS = 180
_VERIFIER_MAX_CLAIMS = 100
_VERIFIER_MIN_CLAIMS = 10
_HARD_FAIL_CAP = 0.7
_VERIFIER_GEN_PARALLELISM = 16
# Per-claim verifier execution fan-out. Capped low because higher concurrency doesn't alway help
_VERIFIER_EXEC_PARALLELISM = 4

# ----- Empty-run marker -------------------------------------------------------
_EMPTY_RUN_MARKER = "workflow execution fully failed"

class VerifierEvaluator(
    BaseEvaluator,
    _VerifierClaimExtractionMixin,
    _VerifierPerClaimMixin,
    _VerifierWorkspaceMixin,
):
    """Per-claim verifier-based evaluator."""

    _DEFAULT_CLAIM_IMPORTANCE = 5
    _GRADIENT_MIN_IMPORTANCE = 3
    _HARD_FAIL_IMPORTANCE = 10

    def __init__(
        self,
        config: "Config",
        workspace_dir: str | Path | None = None,
        verifier_timeout: int = _VERIFIER_TIMEOUT_SECONDS,
        max_claims: int = _VERIFIER_MAX_CLAIMS,
        min_claims: int = _VERIFIER_MIN_CLAIMS,
        hard_fail_cap: float = _HARD_FAIL_CAP,
        preview_head_bytes: int = 16 * 1024,
        preview_tail_bytes: int = 8 * 1024,
        preview_per_claim_cap: int = 32 * 1024,
        use_grounding: bool = True,
        gen_parallelism: int = _VERIFIER_GEN_PARALLELISM,
        exec_parallelism: int = _VERIFIER_EXEC_PARALLELISM,
    ) -> None:
        """Initialise the evaluator; ``workspace_dir`` defaults to ``config.workspace_dir``.

        Args:
            config: Mimosa Config object; provides paths, logger, judge config.
            workspace_dir: Override for the agents' workspace root.
            verifier_timeout: Per-script execution timeout, in seconds.
            max_claims: Hard upper bound on the number of claims verified.
            min_claims: Soft lower bound (logs a warning when extraction yields fewer).
            hard_fail_cap: Maximum overall score allowed when a hard claim fails.
            preview_head_bytes: Bytes of file head to render in previews.
            preview_tail_bytes: Bytes of file tail to render in previews.
            preview_per_claim_cap: Total preview budget allowed per claim.
            use_grounding: When True, fetch peer-reviewed literature grounding.
            gen_parallelism: Max concurrent LLM calls for verifier generation.
            exec_parallelism: Max concurrent claim verifications (executable
                sandbox runs and soft-LLM checks share this pool).
        """
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
        # KB the verifier grounds from (may be the full ground-truth KB; this is
        # the verifier side, distinct from the agent's fair-tier KB).
        self.verifier_kb_name = getattr(config, "perspicacite_verifier_kb_name", None)
        self.gen_parallelism = max(1, int(gen_parallelism))
        self.exec_parallelism = max(1, int(exec_parallelism))
        self._workspace_files: set[str] = set()
        self._grounding_cache: dict[str, str] = {}
        self._runner_temp_root = Path(
            getattr(config, "temp_dir", None) or self.workflow_dir / "_verifier_tmp"
        )
        self._task_spec: str = ""

        # ---- vision model for Source G ----
        vision_model = getattr(config, "vision_judge_model", None)
        if vision_model:
            provider, model = (
                vision_model.split("/", 1)
                if "/" in vision_model
                else ("openai", vision_model)
            )
            self._vision_llm_config = LLMConfig().from_dict({
                "model": model,
                "provider": provider,
                "temperature": 0.1,
                "reasoning_effort": config.reasoning_effort,
                "max_tokens": 1024,
                "openrouter_provider": (
                    config.openrouter_provider_for(vision_model)
                    if hasattr(config, "openrouter_provider_for")
                    else None
                ),
                "openrouter_quantizations": (
                    config.openrouter_quantizations_for(vision_model)
                    if hasattr(config, "openrouter_quantizations_for")
                    else None
                ),
            })
            self.logger.info(
                f"Vision judge configured: {vision_model}"
            )
        else:
            self._vision_llm_config = None
            self.logger.info(
                "No vision_judge_model configured; Source G will skip visual checks"
            )
        self.logger.info(
            f"VerifierEvaluator initialized (workspace={self.workspace_dir}, "
            f"timeout={verifier_timeout}s, claims={self.min_claims}–{max_claims}, "
            f"use_grounding={use_grounding})"
        )
        self._textual_gradient_history: list[str] = []

    # ------------------------------------------------------------------
    # textual gradient — only signal returned to the mutator
    # ------------------------------------------------------------------

    def _build_abstractec_textual_gradient(self, uuid: str, report: str, execution_text: str) -> str:
        """Residual signal passed to the orchestrator to avoid Goodhart's cheating.

        Args:
            uuid: Workflow identifier being summarised.
            report: Full verifier report text to abstract.
            execution_text: Full workflow execution text (agent narration and
                produced output) for context on what the workflow did.

        Returns:
            Short code-tagged single-sentence diagnosis usable by the mutator.
        """
        prompt = f"""
        You convert the verifiers's report into per claim short, directional diagnosis.

        GROUND TRUTH AND TRUST ORDER (critical):
        - The verifier's deterministic checks are ground truth.
        - The agent execution narration below is UNTRUSTED. Agents may declare success while producing degenerate output.
        - Agent execution can however be used to explain why and what failure happened, but never as evidence that the task succeeded.
            Example: Verifier report contact-count ceiling below; agent report contact-count exceeded ceiling because it...
            Example: Invalid conformation; relevant: agent reported he is "Off-by-one in coord map"
        - Do not mention an agent reported failure unless it is confirmed by the verifier.
        - Sort diagnosis by importance: a failure in a high-importance claim is more actionable than a failure in a low-importance claim.

        Here is the agents execution text (agent narration and produced output):
        {execution_text}

        Here is the deterministic verifier's detailed report for workflow {uuid}:
        {report}

        Focus on the report on the most important claims and the most actionable diagnosis.
        Be concise and specific, avoid vague language. Order claims by importance.

        OUTPUT:
        Format: "<diagnosis_CODE>:\n<- <short diagnosis of most important failed claim>

        Example:
        FALLBACK_ECFP_CLASSIFIER:
        -Use of fallback rather than a trained ECFP classifier
        """
        diag = self._call_judge(
            uuid,
            "verifier_abstract_textual_gradient",
            prompt,
        )
        self._textual_gradient_history.append(diag)
        return diag.strip() or "UNDIAGNOSED:No diagnosis could be extracted from the verifier report."

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
            grounding = get_perspicacite_grounding(goal, kb_name=self.verifier_kb_name)
        except Exception as e:
            self.logger.warning(f"Perspicacite grounding raised for {uuid}: {e}")
            grounding = f"{self._GROUNDING_FAILED_MARKER}: {e}"
        self._grounding_cache[uuid] = grounding
        print_box(
            grounding,
            title=f"Perspicacite grounding · {uuid}", color=GREEN, truncate=2048,
        )
        return grounding

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(self, uuid: str) -> dict[str, Any]:
        """Run the verifier pipeline; persists scores under ``evaluation.verifier``.

        Claim continuity across iterations is achieved by a per-task rubric
        cache (``_load_cached_rubric`` / ``_persist_rubric``): the first run
        freezes the full ranked claim list (ids, descriptions, importance),
        every subsequent run reuses it verbatim and skips extraction, dedup,
        and importance rating. Per-claim verifier scripts are still generated
        fresh against the current workspace.

        Args:
            uuid: Workflow identifier to evaluate.

        Returns:
            Dict with ``uuid``, the per-claim results, and aggregate scores.

        Raises:
            EvaluatorError: When ``uuid`` is not a non-empty string.
            WorkflowDataError: When no execution text can be derived.
        """
        if not uuid or not isinstance(uuid, str):
            raise EvaluatorError("Invalid uuid: must be a non-empty string")

        t_total = time.time()
        phase_timings: list[tuple[str, float]] = []

        t = time.time()
        execution_text, success = self.workflow_execution_text(uuid)
        if not execution_text:
            raise WorkflowDataError(f"Cannot generate execution text for workflow {uuid}")

        # Short-circuit when the workflow produced no artefacts: avoids scoring
        # against stale state from a prior generation in a shared workspace.
        wf_info = self._load_workflow_data(uuid)
        if not wf_info.state_result or not wf_info.code:
            return self._short_circuit_failed_run(uuid)

        workspace_listing = self._list_workspace()
        phase_timings.append(("setup (exec text + workspace listing)", time.time() - t))
        print_ok(f"[verifier {uuid}] setup done in {phase_timings[-1][1]:.1f}s")

        t = time.time()
        grounding = self._get_grounding(uuid, execution_text, wf_info.goal)
        phase_timings.append(("grounding fetch", time.time() - t))
        print_ok(f"[verifier {uuid}] grounding fetch done in {phase_timings[-1][1]:.1f}s")

        t = time.time()
        is_truly_empty = not execution_text or _EMPTY_RUN_MARKER in execution_text
        claims = self._extract_claims(
            uuid, wf_info.goal, execution_text, workspace_listing, is_truly_empty, grounding,
            cache_key_text=wf_info.original_task or wf_info.goal,
        )
        claims = self._prepend_declared_output_claims(claims, wf_info)
        phase_timings.append(("claim extraction + importance", time.time() - t))
        print_ok(
            f"[verifier {uuid}] claim extraction + importance done in "
            f"{phase_timings[-1][1]:.1f}s ({len(claims)} claims)"
        )
        if not claims:
            self.logger.warning(f"No claims extracted for {uuid}; verifier returns 0.0")
            scores = {"overall_score": 0.0, "n_claims": 0, "n_pass": 0, "n_fail": 0}
            self._save_results(scores, uuid, "verifier")
            return {"uuid": uuid, "claims": [], **scores}

        self._ensure_verifier_packages()
        claims_to_verify = claims[: self.max_claims]

        st = time.time()
        generated = self._generate_specs_parallel(
            wf_info.goal,
            uuid,
            claims_to_verify,
            execution_text,
            workspace_listing,
            self.gen_parallelism,
        )
        gen_dt = time.time() - st
        phase_timings.append(("spec generation (parallel)", gen_dt))
        print_box(
            f"Verifier spec generation for {len(claims_to_verify)} claims took "
            f"{gen_dt:.1f}s (parallelism={self.gen_parallelism})",
            title="Verifier generation timing",
        )

        t_loop = time.time()
        per_claim = self._verify_claims_parallel(
            uuid,
            claims_to_verify,
            generated,
            execution_text,
            workspace_listing,
            grounding,
        )
        loop_dt = time.time() - t_loop
        phase_timings.append(
            (f"per-claim verification (parallel x{self.exec_parallelism})", loop_dt)
        )
        print_ok(
            f"[verifier {uuid}] per-claim verification done in {loop_dt:.1f}s "
            f"({len(per_claim)} claims, parallelism={self.exec_parallelism})"
        )
        self._print_per_claim_timings(per_claim)

        scores = self._aggregate(per_claim)
        scores["failure_fingerprint"] = compute_failure_fingerprint(per_claim)

        self._write_report(uuid, claims, per_claim, scores)

        gradient_report = self._build_report(
            per_claim,
            scores,
            min_importance=self._GRADIENT_MIN_IMPORTANCE,
        )
        t = time.time()
        textual_gradient = self._build_abstractec_textual_gradient(
            uuid, gradient_report, execution_text
        )
        phase_timings.append(("textual gradient builder", time.time() - t))
        print_ok(
            f"[verifier {uuid}] textual gradient done in "
            f"{phase_timings[-1][1]:.1f}s"
        )
        scores["abstractec_textual_gradient"] = textual_gradient
        self._persist_textual_gradient(uuid, textual_gradient)

        try:
            self._save_results(scores, uuid, "verifier")
        except Exception as e:
            self.logger.error(f"Failed to persist verifier scores for {uuid}: {e}")

        phase_timings.append(("TOTAL evaluate()", time.time() - t_total))
        self._print_phase_summary(uuid, phase_timings)

        return {"uuid": uuid, "claims": per_claim, **scores}

    def _verify_claims_parallel(
        self,
        uuid: str,
        claims_to_verify: list[dict[str, Any]],
        generated: dict[str, tuple[dict[str, Any], dict[str, Any]]],
        execution_text: str,
        workspace_listing: str,
        grounding: str,
    ) -> list[dict[str, Any]]:
        """Fan claim verification out across ``self.exec_parallelism`` threads.

        Args:
            uuid: Workflow identifier (used for judge calls + sandbox dirs).
            claims_to_verify: Claims in the desired output order.
            generated: Specs produced by ``_generate_specs_parallel``.
            execution_text: Agent narration / produced output text.
            workspace_listing: Rendered listing of workspace files.
            grounding: Optional peer-reviewed literature grounding block.

        Returns:
            Scored per-claim list in the same order as ``claims_to_verify``.
        """
        if not claims_to_verify:
            return []
        results: list[dict[str, Any] | None] = [None] * len(claims_to_verify)
        workers = max(1, min(len(claims_to_verify), self.exec_parallelism))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    self._verify_one_claim_safe,
                    uuid, claim, generated,
                    execution_text, workspace_listing, grounding,
                ): idx
                for idx, claim in enumerate(claims_to_verify)
            }
            for f in as_completed(futures):
                idx = futures[f]
                results[idx] = f.result()
        return [r for r in results if r is not None]

    def _verify_one_claim_safe(
        self,
        uuid: str,
        claim: dict[str, Any],
        generated: dict[str, tuple[dict[str, Any], dict[str, Any]]],
        execution_text: str,
        workspace_listing: str,
        grounding: str,
    ) -> dict[str, Any]:
        """Resolve the right spec for one claim and run ``_verify_claim``.

        Soft-fails to an ``error`` result so one bad claim cannot crash the
        thread-pool batch.
        """
        cid = claim["id"]
        target_claim, preloaded_spec = generated[cid]
        try:
            return self._verify_claim(
                uuid,
                target_claim,
                execution_text,
                workspace_listing,
                grounding,
                preloaded_spec=preloaded_spec,
            )
        except Exception as e:
            self.logger.warning(
                f"parallel claim verification failed for {cid}: "
                f"{type(e).__name__}: {e}"
            )
            return {
                "claim": target_claim,
                "spec": preloaded_spec,
                "score": 0.0,
                "verifier_kind": "executable" if preloaded_spec.get("executable") else "soft",
                "status": "error",
                "details": f"verify_claim raised: {type(e).__name__}: {e}",
                "elapsed_s": 0.0,
            }

    @staticmethod
    def _print_per_claim_timings(per_claim: list[dict[str, Any]]) -> None:
        """Render a sorted table of per-claim elapsed times.

        Slow claims rise to the top so the cost concentration is obvious at a
        glance — typically a handful of executable scripts dominate the loop.
        """
        if not per_claim:
            return
        rows = sorted(
            per_claim,
            key=lambda c: float(c.get("elapsed_s") or 0.0),
            reverse=True,
        )
        total = sum(float(c.get("elapsed_s") or 0.0) for c in rows)
        header = f"{'claim_id':<38} {'kind':<11} {'status':<7} {'elapsed_s':>10}"
        body = "\n".join(
            f"{str((c.get('claim') or {}).get('id') or '?')[:38]:<38} "
            f"{str(c.get('verifier_kind') or '?'):<11} "
            f"{str(c.get('status') or '?'):<7} "
            f"{float(c.get('elapsed_s') or 0.0):>10.2f}"
            for c in rows
        )
        print_box(
            f"{header}\n{body}\n"
            f"{'-' * 68}\n"
            f"sum of per-claim elapsed: {total:.1f}s over {len(rows)} claims",
            title="Per-claim verification timings (slowest first)",
            color=CYAN,
        )

    @staticmethod
    def _print_phase_summary(uuid: str, phases: list[tuple[str, float]]) -> None:
        """Render the end-of-evaluate phase breakdown."""
        if not phases:
            return
        body = "\n".join(f"{name:<46} {dt:>8.2f}s" for name, dt in phases)
        print_box(
            body,
            title=f"Verifier phase summary · {uuid}",
            color=CYAN,
        )

    def _short_circuit_failed_run(self, uuid: str) -> dict[str, Any]:
        """Return 0.0 without running scripts when the workflow produced nothing.

        Args:
            uuid: Workflow identifier of the failed run.

        Returns:
            Score dict with ``overall_score=0.0`` and a ``skipped_reason`` marker.
        """
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
            "abstractec_textual_gradient": "workflow code failed to generate or execute; ensure code is properly formatted and that the workflow runs without crashing",
            "failure_fingerprint": {
                "vector": [0.0] * _FP_DIM,
                "presence_mask": [0.0] * _FP_DIM,
                "pass_rates": [0.5] * _FP_DIM,
            },
        }
        try:
            self._write_report(uuid, [], [], scores)
        except Exception as e:
            self.logger.error(f"Failed to write short-circuit report for {uuid}: {e}")
        self._persist_textual_gradient(uuid, "")
        try:
            self._save_results(scores, uuid, "verifier")
        except Exception as e:
            self.logger.error(f"Failed to persist short-circuit scores for {uuid}: {e}")
        return {"uuid": uuid, "claims": [], **scores}

    # ------------------------------------------------------------------
    # Per-task rubric cache — freezes the final (post-dedup, post-importance)
    # claim list so verifier scores are comparable across iterations of the
    # SAME task. On cache hit, claim extraction, dedup, and importance rating
    # are all skipped; only `likely_relevant_files` is re-validated against
    # the current workspace.
    # ------------------------------------------------------------------

    _RUBRIC_CACHE_FILENAME_FMT = "rubric_cache_{task_key}.json"

    @property
    def verifier_temp_root(self) -> Path:
        """Public alias for the verifier scratch root (``_verifier_tmp/``)."""
        return self._runner_temp_root

    @staticmethod
    def _task_cache_key(goal: str) -> str:
        """Stable 16-hex-char key derived from the task goal text.

        Same goal → same key, across runs and machines. Hashing the goal (not
        the workflow uuid) is what makes the cache shared between iterations
        of the SAME task and distinct between DIFFERENT tasks.
        """
        return declared_outputs.task_key(goal)

    def _rubric_cache_path(self, task_key: str) -> Path:
        """On-disk path of the frozen rubric for one task."""
        return self._runner_temp_root / self._RUBRIC_CACHE_FILENAME_FMT.format(
            task_key=task_key
        )

    def _prepend_declared_output_claims(
        self,
        claims: list[dict[str, Any]],
        wf_info: Any,
    ) -> list[dict[str, Any]]:
        """Put the plan's declared outputs at the head of the rubric.

        The plan writes these before the step runs, so unlike every extracted
        claim they cannot have been shaped by what the agent chose to do. They
        are prepended rather than appended so ``claims[:max_claims]`` cannot
        drop them, and they are added *after* ``_extract_claims`` has persisted
        the rubric so they never enter the frozen cache — the cache freezes what
        a run produced, these belong to the plan and must follow it.

        Absent declaration, unreadable file, or a claim id already present: the
        list is returned unchanged.
        """
        try:
            task_text = getattr(wf_info, "original_task", "") or getattr(wf_info, "goal", "")
            outputs = declared_outputs.load(self._runner_temp_root, task_text)
            if not outputs:
                return claims
            existing = {c.get("id") for c in claims}
            extra = [c for c in declared_outputs.as_claims(outputs)
                     if c["id"] not in existing]
            if not extra:
                return claims
            self.logger.info(
                "Declared outputs from the plan added as %d mandatory claim(s): %s",
                len(extra), ", ".join(o for o in outputs)
            )
            return extra + claims
        except Exception:
            self.logger.exception("Could not apply declared-output claims")
            return claims

    def _load_cached_rubric(self, task_key: str) -> list[dict[str, Any]] | None:
        """Return the cached rubric claim list, or None when no cache exists.

        ``None`` means "fresh extraction required". A returned list is the
        verbatim frozen rubric — ids, descriptions, importance, and
        rationales are reused as-is; only file paths get re-validated by
        ``_adapt_rubric_to_workspace`` before use.
        """
        path = self._rubric_cache_path(task_key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self.logger.warning(f"Could not read rubric cache {path}: {e}")
            return None
        claims = data.get("claims") if isinstance(data, dict) else None
        if not isinstance(claims, list) or not claims:
            return None
        return claims

    def _persist_rubric(
        self,
        task_key: str,
        claims: list[dict[str, Any]],
    ) -> None:
        """Seed the rubric from the FIRST successful extraction; no-op afterwards.

        Never overwrites an existing cache: the first workflow that produces a
        non-empty ranked claim list for this task defines the rubric for every
        subsequent workflow on the same task. Wiping the cache file (or
        ``cleanup.sh``) reseeds.
        """
        if not claims:
            return
        path = self._rubric_cache_path(task_key)
        if path.exists():
            return
        payload = {"task_key": task_key, "claims": claims}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.logger.info(
                f"Rubric cache seeded: {len(claims)} claims at {path}"
            )
        except OSError as e:
            self.logger.warning(f"Could not write rubric cache {path}: {e}")

    def _adapt_rubric_to_workspace(
        self,
        claims: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Re-validate `likely_relevant_files` against the current workspace.

        Importance, id, description, and rationale are preserved verbatim —
        those are the frozen rubric. Only the per-claim file hints adapt to
        the workspace at hand so per-claim verifier scripts get pointed at
        files that actually exist in this iteration.
        """
        out: list[dict[str, Any]] = []
        for c in claims:
            files = self._validate_workspace_paths(
                c.get("likely_relevant_files") or [],
                allowed=self._workspace_files or None,
                label=str(c.get("id", "?")),
            )
            out.append({**c, "likely_relevant_files": files})
        return out

    # ------------------------------------------------------------------
    # Stage 5 — importance-weighted aggregation
    # ------------------------------------------------------------------

    def _claim_weight(self, c: dict[str, Any]) -> float:
        """Aggregator weight = self-reported importance (1–10).

        Clamps to ``[1, 10]`` so a stray rater overshoot can't dominate, and
        falls back to ``_DEFAULT_CLAIM_IMPORTANCE`` when the field is missing
        (e.g. an old anchored rubric before back-compat mapping ran).

        Args:
            c: Scored claim dict carrying ``claim["importance"]``.

        Returns:
            Float weight in ``[1.0, 10.0]``.
        """
        raw = c["claim"].get("importance", self._DEFAULT_CLAIM_IMPORTANCE)
        try:
            return max(1.0, min(10.0, float(raw)))
        except (TypeError, ValueError):
            return float(self._DEFAULT_CLAIM_IMPORTANCE)

    def _aggregate(self, per_claim: list[dict[str, Any]]) -> dict[str, Any]:
        """Importance-weighted mean; capped on top-tier fail.

        Each claim is weighted by its rater-assigned importance (1-10) rather
        than the old hard=3/soft=1 step function. This gives the optimizer a
        smooth gradient: flipping an importance-10 deliverable claim moves the
        score ~5× more than flipping a low-importance hygiene claim.

        Args:
            per_claim: List of per-claim scored dicts from ``_verify_claim``.

        Returns:
            Aggregate score dict with overall/base/cap and per-status counts.
        """
        if not per_claim:
            return self._empty_aggregate_result(n_claims=0)

        scored = [c for c in per_claim if c.get("status") != "error"]
        n_pass = sum(1 for c in per_claim if c["status"] == "pass")
        n_fail = sum(1 for c in per_claim if c["status"] == "fail")
        n_error = sum(1 for c in per_claim if c["status"] == "error")
        n_unsure = sum(1 for c in per_claim if c["status"] == "unsure")

        if not scored:
            return self._empty_aggregate_result(
                n_claims=len(per_claim),
                n_pass=n_pass,
                n_fail=n_fail,
                n_error=n_error,
                n_unsure=n_unsure,
                skipped_reason="all_verifiers_errored",
            )

        total_w = sum(self._claim_weight(c) for c in scored)
        base_mean = sum(c["score"] * self._claim_weight(c) for c in scored) / total_w
        # Pre-cap: clamp to [0, 1] before applying the hard-fail cap
        pre_cap = max(0.0, min(1.0, base_mean))
        # Hard-fail cap fires only on a real refutation of a top-importance claim
        hard_fail = any(
            self._claim_weight(c) >= self._HARD_FAIL_IMPORTANCE
            and c["status"] == "fail"
            for c in scored
        )
        overall = min(pre_cap, self.hard_fail_cap) if hard_fail else pre_cap

        return {
            "overall_score": round(overall, 4),
            "overall_score_uncapped": round(pre_cap, 4),
            "base_mean": round(base_mean, 4),
            "hard_fail_capped": hard_fail,
            "n_claims": len(per_claim),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_error": n_error,
            "n_unsure": n_unsure,
            "n_scored": len(scored),
        }

    @staticmethod
    def _empty_aggregate_result(
        n_claims: int = 0,
        n_pass: int = 0,
        n_fail: int = 0,
        n_error: int = 0,
        n_unsure: int = 0,
        skipped_reason: str | None = None,
    ) -> dict[str, Any]:
        """Zero-score aggregate dict in the standard ``_aggregate`` shape.

        Factored out so the two early-return branches in ``_aggregate``
        (no per-claim entries at all, vs. all-errored runs) share one source
        of truth for the output schema.
        """
        result: dict[str, Any] = {
            "overall_score": 0.0,
            "overall_score_uncapped": 0.0,
            "base_mean": 0.0,
            "hard_fail_capped": False,
            "n_claims": n_claims,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_error": n_error,
            "n_unsure": n_unsure,
            "n_scored": 0,
        }
        if skipped_reason is not None:
            result["skipped_reason"] = skipped_reason
        return result

    def _persist_textual_gradient(self, uuid: str, textual_gradient: str) -> None:
        """Write the textual_gradient to ``textual_gradient.txt`` alongside the report.

        Args:
            uuid: Workflow identifier; selects the on-disk output folder.
            textual_gradient: Text to persist; empty strings are skipped.
        """
        if not textual_gradient:
            return
        path = self.workflow_dir / uuid / "textual_gradient.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(textual_gradient, encoding="utf-8")
        except OSError as e:
            self.logger.warning(f"could not write textual_gradient.txt for {uuid}: {e}")

    # ------------------------------------------------------------------
    # Report rendering + persistence
    # ------------------------------------------------------------------

    _REPORT_SEPARATOR_BAR = "=" * 60
    _REPORT_STDERR_TAIL_LINES = 5

    def _build_report(
        self,
        per_claim: list[dict[str, Any]],
        scores: dict[str, Any],
        min_importance: int = 0,
    ) -> str:
        """Render a plain-text report from per-claim results and aggregate scores."""
        lines: list[str] = list(self._format_report_header(scores, min_importance))
        for c in per_claim:
            if c["status"] == "error":
                continue # this avoid execution error of verifier passed to textual gradient
            lines.extend(self._format_claim_entry(c, min_importance))
        return "\n".join(lines) + "\n"

    def _format_report_header(
        self, scores: dict[str, Any], min_importance: int
    ) -> list[str]:
        """Render the report header (title, counts, score breakdown, filter note)."""
        lines = [
            "Verifier Evaluation",
            self._REPORT_SEPARATOR_BAR,
            (
                f"Claims: {scores['n_claims']}  pass={scores['n_pass']}  "
                f"fail={scores['n_fail']}  error={scores.get('n_error', 0)}  "
                f"unsure={scores.get('n_unsure', 0)}  "
                f"scored={scores.get('n_scored', 0)}"
            ),
            (
                f"Overall: {scores['overall_score']:.3f}, "
                f"uncapped {scores.get('overall_score_uncapped', 0.0):.3f}, "
                f"hard_fail_capped={scores.get('hard_fail_capped', False)})"
            ),
            f"  base_mean={scores.get('base_mean', 0.0):.3f}",
        ]
        if min_importance > 0:
            lines.append(f"(filtered view: importance ≥ {min_importance})")
        return lines

    def _format_claim_entry(
        self, c: dict[str, Any], min_importance: int
    ) -> list[str]:
        """Render one claim's block; empty list when filtered out."""
        cl = c["claim"]
        imp = int(cl.get("importance", self._DEFAULT_CLAIM_IMPORTANCE))
        if imp < min_importance:
            return []
        lines = [self._format_claim_header(cl, imp)]
        rel = cl.get("likely_relevant_files", [])
        if rel:
            lines.append(f"  relevant_files: {rel}")
        lines.append(f"  kind={c['verifier_kind']} status={c['status']} score={c['score']}")
        if c.get("details"):
            lines.append(f"  details: {c['details']}")
        lines.extend(self._format_stderr_tail(c))
        lines.append("")
        return lines

    @staticmethod
    def _format_claim_header(cl: dict[str, Any], imp: int) -> str:
        """Format one claim's heading line (optionally suffixed with rationale)."""
        rationale = str(cl.get("importance_rationale") or "").strip()
        if rationale:
            return f"[{cl['id']}] (importance={imp}; {rationale}) {cl['description']}"
        return f"[{cl['id']}] (importance={imp}) {cl['description']}"

    @classmethod
    def _format_stderr_tail(cls, c: dict[str, Any]) -> list[str]:
        """Render the tail of stderr for executable verifiers; empty otherwise."""
        if c.get("verifier_kind") != "executable" or not c.get("raw_stderr"):
            return []
        tail = c["raw_stderr"].strip().splitlines()[-cls._REPORT_STDERR_TAIL_LINES:]
        if not tail:
            return []
        return ["  stderr (tail):", *(f"    {line}" for line in tail)]


    def _write_report(
        self,
        uuid: str,
        claims: list[dict[str, Any]],
        per_claim: list[dict[str, Any]],
        scores: dict[str, Any]
    ) -> None:
        """Persist the built report under ``<workflow_dir>/<uuid>/evaluation.txt``.

        Args:
            uuid: Workflow identifier; selects the on-disk output folder.
            claims: Raw extracted claim list (kept for signature parity).
            per_claim: List of per-claim scored dicts to render.
            scores: Aggregated score dict from ``_aggregate``.
        """
        path = self.workflow_dir / uuid / "evaluation.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        report = self._build_report(per_claim, scores)
        try:
            path.write_text(report, encoding="utf-8")
            self.logger.info(f"Verifier report written to {path}")
        except OSError as e:
            self.logger.error(f"Could not write verifier report for {uuid}: {e}")


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))  # noqa: E402
    from config import Config
    config = Config()
    verifier = VerifierEvaluator(config, config.workspace_dir)
    verifier.evaluate("20260812_054206_697f0f45")

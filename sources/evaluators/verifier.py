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

from sources.core.failure_fingerprint import (
    DESCRIPTOR_DIM as _FP_DIM,
    compute_failure_fingerprint,
)

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
_HARD_FAIL_CAP = 0.99  # disabled so signal stay smooth
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
    _LEGACY_CRITICALITY_TO_IMPORTANCE = {"hard": 8, "soft": 3}

    def __init__(
        self,
        config: "Config",
        workspace_dir: str | Path | None = None,
        verifier_timeout: int = _VERIFIER_TIMEOUT_SECONDS,
        max_claims: int = _VERIFIER_MAX_CLAIMS,
        min_claims: int = _VERIFIER_MIN_CLAIMS,
        hard_fail_cap: float = _HARD_FAIL_CAP,
        preview_head_bytes: int = 8 * 1024,
        preview_tail_bytes: int = 2 * 1024,
        preview_per_claim_cap: int = 24 * 1024,
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
        self.gen_parallelism = max(1, int(gen_parallelism))
        self.exec_parallelism = max(1, int(exec_parallelism))
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
            grounding = get_perspicacite_grounding(goal)
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

    def evaluate(
        self,
        uuid: str,
        rubric_anchor_uuid: str | None = None,
    ) -> dict[str, Any]:
        """Run the verifier pipeline; persists scores under ``evaluation.verifier``.

        Args:
            uuid: Workflow identifier to evaluate.
            rubric_anchor_uuid: Optional ancestor whose cached rubric to reuse.

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
        claims, spec_reuse_records = self._resolve_anchored_claims(
            uuid, rubric_anchor_uuid, execution_text, workspace_listing
        )
        if not claims:
            is_truly_empty = not execution_text or _EMPTY_RUN_MARKER in execution_text
            claims = self._extract_claims(
                uuid, wf_info.goal, execution_text, workspace_listing, is_truly_empty, grounding
            )
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
        anchor_preloaded, needs_generation = self._partition_specs_by_anchor(
            claims_to_verify, spec_reuse_records, rubric_anchor_uuid
        )

        st = time.time()
        generated = self._generate_specs_parallel(
            uuid,
            needs_generation,
            execution_text,
            workspace_listing,
            self.gen_parallelism,
        )
        gen_dt = time.time() - st
        phase_timings.append(("spec generation (parallel)", gen_dt))
        print_box(
            f"Verifier spec generation for {len(needs_generation)} claims took "
            f"{gen_dt:.1f}s (parallelism={self.gen_parallelism})",
            title="Verifier generation timing",
        )

        t_loop = time.time()
        per_claim = self._verify_claims_parallel(
            uuid,
            claims_to_verify,
            anchor_preloaded,
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
        self._persist_claims(uuid, claims, per_claim)

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
        anchor_preloaded: dict[str, dict[str, Any]],
        generated: dict[str, tuple[dict[str, Any], dict[str, Any]]],
        execution_text: str,
        workspace_listing: str,
        grounding: str,
    ) -> list[dict[str, Any]]:
        """Fan claim verification out across ``self.exec_parallelism`` threads.

        Args:
            uuid: Workflow identifier (used for judge calls + sandbox dirs).
            claims_to_verify: Claims in the desired output order.
            anchor_preloaded: Specs reconstructed from a lineage anchor.
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
                    uuid, claim, anchor_preloaded, generated,
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
        anchor_preloaded: dict[str, dict[str, Any]],
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
        if cid in anchor_preloaded:
            target_claim = claim
            preloaded_spec = anchor_preloaded[cid]
        else:
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
    # Lineage rubric reuse — read claims/scripts from an ancestor's cache
    # ------------------------------------------------------------------

    _CLAIMS_CACHE_FILENAME = "claims.json"

    @property
    def verifier_temp_root(self) -> Path:
        """Public alias for the verifier scratch root (``_verifier_tmp/``).

        Exposed so callers (lineage walkers, anchor resolvers) read from the
        same authoritative path the evaluator itself uses, even when
        ``config.temp_dir`` is overridden.
        """
        return self._runner_temp_root

    def _anchor_dir(self, uuid: str) -> Path:
        """Return the on-disk verifier cache folder for ``uuid``.

        Args:
            uuid: Workflow identifier whose cache directory is needed.

        Returns:
            Path to ``_verifier_tmp/<uuid>/`` (existence not guaranteed).
        """
        return self._runner_temp_root / uuid

    def _claims_cache_path(self, uuid: str) -> Path:
        """Return the rubric-cache JSON path inside ``uuid``'s anchor dir."""
        return self._anchor_dir(uuid) / self._CLAIMS_CACHE_FILENAME

    @staticmethod
    def _rubric_record(claim: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
        """Build one persisted rubric entry from a claim + its verifier spec.

        ``importance`` (int 1-10) replaces the old ``criticality`` tier. The
        rationale is persisted alongside so anchored descendants render the
        same gradient view as the anchor.
        """
        return {
            "id": claim.get("id"),
            "description": claim.get("description", ""),
            "importance": int(
                claim.get("importance", VerifierEvaluator._DEFAULT_CLAIM_IMPORTANCE)
            ),
            "importance_rationale": str(claim.get("importance_rationale") or ""),
            "source": claim.get("source", ""),
            "likely_relevant_files": list(claim.get("likely_relevant_files") or []),
            "executable": bool(spec.get("executable")),
            "reason": str(spec.get("reason") or ""),
        }

    def _persist_claims(
        self,
        uuid: str,
        claims: list[dict[str, Any]],
        per_claim: list[dict[str, Any]],
    ) -> None:
        """Persist the rubric so descendants can reuse it for stable scoring.

        Writes ``_verifier_tmp/<uuid>/claims.json`` next to the
        ``verify_<id>.py`` scripts that were already saved as a side effect of
        execution. Best-effort: write failures are logged and swallowed.

        Args:
            uuid: Workflow identifier whose anchor folder receives the JSON.
            claims: Raw claim list as returned by ``_extract_claims``.
            per_claim: Scored claim list; supplies the per-claim verifier spec.
        """
        # Only `likely_relevant_files` is sourced from per_claim — `_llm_select_files`
        # can shift the list away from the extraction's, and the cached script
        # opens the post-selection set. Description/importance/rationale stay on
        # the original claim so descendants still see the rater's metadata even
        # when a per_claim stub omits it (tests do this).
        per_claim_by_id = {
            (c.get("claim") or {}).get("id"): c
            for c in per_claim
        }
        rubric: list[dict[str, Any]] = []
        for c in claims:
            entry = per_claim_by_id.get(c.get("id"))
            if entry is None:
                rubric.append(self._rubric_record(c, {}))
                continue
            updated_files = (entry.get("claim") or {}).get("likely_relevant_files")
            persist_claim = (
                {**c, "likely_relevant_files": updated_files}
                if updated_files is not None
                else c
            )
            rubric.append(self._rubric_record(persist_claim, entry.get("spec") or {}))
        payload = {"anchor_uuid": uuid, "claims": rubric}
        path = self._claims_cache_path(uuid)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.logger.info(f"Verifier rubric cache written to {path}")
        except OSError as e:
            self.logger.warning(f"Could not write claims cache for {uuid}: {e}")

    def _load_anchored_claims(self, anchor_uuid: str) -> list[dict[str, Any]] | None:
        """Load a cached rubric from ``anchor_uuid``'s verifier folder.

        Anchors written before the criticality→importance migration are
        upgraded on read so old lineages keep scoring without re-running
        extraction: ``criticality=hard`` → ``importance=8``, ``soft`` → ``3``.

        Args:
            anchor_uuid: Ancestor workflow whose rubric should be reused.

        Returns:
            The cached rubric entries, or ``None`` when the cache file is
            missing, unreadable, or doesn't contain a non-empty claim list.
        """
        path = self._claims_cache_path(anchor_uuid)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            self.logger.warning(f"Could not read anchored rubric {path}: {e}")
            return None
        claims = data.get("claims") if isinstance(data, dict) else None
        if not isinstance(claims, list) or not claims:
            return None
        kept = [
            self._upgrade_legacy_anchor(c)
            for c in claims
            if isinstance(c, dict) and isinstance(c.get("id"), str) and c["id"]
        ]
        if len(kept) != len(claims):
            self.logger.warning(
                f"Dropped {len(claims) - len(kept)} anchored claim(s) "
                f"from {path} with missing or non-string id"
            )
        return kept or None

    @classmethod
    def _upgrade_legacy_anchor(cls, rec: dict[str, Any]) -> dict[str, Any]:
        """Map old ``criticality`` field to ``importance`` when absent.

        New schemas pass through untouched; old schemas get a synthesised
        importance derived from the prior hard/soft tier so descendants score
        without re-running extraction.
        """
        if "importance" in rec:
            return rec
        legacy = rec.get("criticality")
        if isinstance(legacy, str):
            rec = {
                **rec,
                "importance": cls._LEGACY_CRITICALITY_TO_IMPORTANCE.get(
                    legacy, cls._DEFAULT_CLAIM_IMPORTANCE
                ),
            }
        return rec

    def _partition_specs_by_anchor(
        self,
        claims_to_verify: list[dict[str, Any]],
        anchored_records: list[dict[str, Any]] | None,
        rubric_anchor_uuid: str | None,
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        """Split claims into anchor-preloaded specs vs. those needing generation.

        Anchor-cache hits are O(disk read); every other claim costs two judge
        round-trips (file selection + spec gen).
        """
        anchored_specs = {
            c["id"]: (bool(c.get("executable")), str(c.get("reason") or ""))
            for c in (anchored_records or [])
        }
        anchor_preloaded: dict[str, dict[str, Any]] = {}
        needs_generation: list[dict[str, Any]] = []
        for claim in claims_to_verify:
            if anchored_specs and rubric_anchor_uuid and claim["id"] in anchored_specs:
                executable, reason = anchored_specs[claim["id"]]
                anchor_preloaded[claim["id"]] = self._spec_from_anchor(
                    rubric_anchor_uuid, claim["id"], executable, reason
                )
            else:
                needs_generation.append(claim)
        return anchor_preloaded, needs_generation

    def _spec_from_anchor(
        self,
        anchor_uuid: str,
        claim_id: str,
        executable: bool,
        reason: str,
    ) -> dict[str, Any]:
        """Reconstruct a verifier spec from an ancestor's on-disk cache.

        Args:
            anchor_uuid: Ancestor whose cached script is read.
            claim_id: Claim identifier; names the ``verify_<id>.py`` file.
            executable: Whether the cached claim was marked executable.
            reason: Cached non-executable rationale (ignored when executable).

        Returns:
            Spec dict in the same shape as ``_generate_verifier`` returns.
            Falls back to a non-executable spec when an executable script is
            missing on disk, so downstream scoring still runs deterministically.
        """
        if executable:
            script = self._anchor_dir(anchor_uuid) / f"verify_{claim_id}.py"
            try:
                code = script.read_text(encoding="utf-8")
            except OSError as e:
                self.logger.warning(
                    f"Anchored verifier missing for {claim_id} at {script}: {e}"
                )
                return {"executable": False, "reason": f"anchored script unreadable: {e}"}
            return {"executable": True, "code": code}
        return {"executable": False, "reason": reason or ""}

    def _claims_from_anchor(self, anchored: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Project cached rubric entries into the shape ``_verify_claim`` expects.

        Drops the persisted ``executable``/``reason`` fields — they're consumed
        later by ``_spec_from_anchor`` — and keeps everything ``_aggregate``
        and report-writing read off the claim dict, including ``importance``
        and its rationale.

        Args:
            anchored: Records as returned by ``_load_anchored_claims`` (already
                upgraded from any legacy ``criticality`` field).

        Returns:
            Claim dicts mirroring ``_extract_claims`` output.
        """
        return [
            {
                "id": c.get("id"),
                "description": c.get("description", ""),
                "importance": int(
                    c.get("importance", self._DEFAULT_CLAIM_IMPORTANCE)
                ),
                "importance_rationale": str(c.get("importance_rationale") or ""),
                "source": c.get("source", "anchor"),
                "likely_relevant_files": list(c.get("likely_relevant_files") or []),
            }
            for c in anchored
        ]

    # ------------------------------------------------------------------
    # Anchor freshness — drop or adapt claims whose cached files are gone
    # ------------------------------------------------------------------

    _STALE_ANCHOR_REASON = "regenerated from stale anchor; original files absent"

    def _resolve_anchored_claims(
        self,
        uuid: str,
        rubric_anchor_uuid: str | None,
        execution_text: str,
        workspace_listing: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Load the ancestor's rubric and refresh it against the current workspace.

        Returns ``([], None)`` to signal the caller should fall back to LLM
        claim extraction: either no anchor was specified, the cache was
        unreadable, or every cached claim was dropped because its files are
        missing and regeneration failed. The second return value is the
        subset of records whose cached ``verify_<id>.py`` is still safe to
        reuse — anything regenerated has new file targets that the cached
        script does not know about.

        Args:
            uuid: Workflow identifier (used for the regen judge calls).
            rubric_anchor_uuid: Ancestor whose rubric should be reused.
            execution_text: Agent narration; helps the regen LLM name new files.
            workspace_listing: Current workspace listing (one entry per line).

        Returns:
            ``(claims, spec_reuse_records)`` — projected claim list and the
            records whose cached script may be reused as-is.
        """
        if not rubric_anchor_uuid:
            return [], None
        anchored_records = self._load_anchored_claims(rubric_anchor_uuid)
        if not anchored_records:
            self.logger.warning(
                f"rubric_anchor_uuid={rubric_anchor_uuid} has no readable "
                f"cache; falling back to LLM claim extraction"
            )
            return [], None
        fresh, stale = self._partition_anchored_by_file_presence(anchored_records)
        regenerated = self._regenerate_stale_anchor_claims(
            uuid, stale, execution_text, workspace_listing
        )
        merged = fresh + regenerated
        if not merged:
            self.logger.warning(
                f"rubric_anchor_uuid={rubric_anchor_uuid}: all "
                f"{len(anchored_records)} cached claims dropped (files missing, "
                f"regen failed); falling back to LLM claim extraction"
            )
            return [], None
        dropped = len(stale) - len(regenerated)
        self.logger.info(
            f"Reusing rubric anchor {rubric_anchor_uuid} for {uuid}: "
            f"{len(fresh)} fresh / {len(regenerated)} regenerated / "
            f"{dropped} dropped (total {len(merged)} cached claims)"
        )
        return self._claims_from_anchor(merged), fresh

    def _partition_anchored_by_file_presence(
        self,
        anchored_records: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split anchored claims by whether their cached files still exist.

        A record is fresh when every entry in its cached
        ``likely_relevant_files`` is present in the current workspace, or when
        the list is empty (nothing to invalidate). Stale records reference at
        least one file the descendant workflow no longer produced; reusing
        their cached verifier script would hit a missing path. Accuracy here
        relies on ``_persist_claims`` storing the POST-selection file list
        (the paths the cached script actually opens) — the
        ``_llm_select_files`` step is the source of truth on what the script
        targets.

        Args:
            anchored_records: Rubric entries as returned by ``_load_anchored_claims``.

        Returns:
            ``(fresh, stale)`` partition, input order preserved in both lists.
        """
        fresh: list[dict[str, Any]] = []
        stale: list[dict[str, Any]] = []
        workspace_files = self._workspace_files
        for rec in anchored_records:
            rel_files = rec.get("likely_relevant_files") or []
            if not rel_files or all(rp in workspace_files for rp in rel_files):
                fresh.append(rec)
            else:
                stale.append(rec)
        return fresh, stale

    def _regenerate_stale_anchor_claims(
        self,
        uuid: str,
        stale_records: list[dict[str, Any]],
        execution_text: str,
        workspace_listing: str,
    ) -> list[dict[str, Any]]:
        """Adapt stale anchored claims to the current workspace.

        Each stale record is regenerated by a judge call seeded with the
        original claim. Importance and rationale carry over verbatim — only
        ``description`` and ``likely_relevant_files`` are refreshed so the
        downstream spec generator can target files that actually exist.
        Calls fan out across ``self.gen_parallelism`` threads.

        Args:
            uuid: Workflow identifier (used for the judge call).
            stale_records: Anchored records whose cached files no longer exist.
            execution_text: Agent narration; helps the LLM name new artefacts.
            workspace_listing: Rendered listing of files in the current workspace.

        Returns:
            Regenerated rubric records in the cached-record shape. Records
            whose regeneration failed are dropped — a stale reference is
            worse than no claim.
        """
        if not stale_records:
            return []
        workers = max(1, min(len(stale_records), self.gen_parallelism))
        regenerated: list[dict[str, Any] | None] = [None] * len(stale_records)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    self._regenerate_one_stale_claim,
                    uuid, rec, execution_text, workspace_listing,
                ): idx
                for idx, rec in enumerate(stale_records)
            }
            for f in as_completed(futures):
                idx = futures[f]
                regenerated[idx] = f.result()
        return [r for r in regenerated if r is not None]

    def _regenerate_one_stale_claim(
        self,
        uuid: str,
        rec: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
    ) -> dict[str, Any] | None:
        """Single-claim regeneration; ``None`` signals the claim should be dropped."""
        cid = str(rec.get("id") or "")
        prompt = self._build_anchor_regen_prompt(rec, execution_text, workspace_listing)
        data, err = self._call_judge_for_json(
            uuid, f"verifier_regen_anchored_{cid}", prompt
        )
        if err is not None or not isinstance(data, dict):
            self.logger.warning(
                f"anchor regeneration failed for {cid}: "
                f"{err or 'non-dict JSON'}; dropping claim"
            )
            return None
        description = str(data.get("description") or rec.get("description") or "").strip()
        new_files = self._validate_workspace_paths(
            data.get("likely_relevant_files") or [],
            allowed=self._workspace_files or None,
            label=cid,
        )
        # Drop the cached executable script: its paths point at files that
        # are gone. The spec generator will write a fresh script targeting
        # the new likely_relevant_files (or fall back to a soft check when
        # the list is empty).
        return {
            **rec,
            "description": description,
            "likely_relevant_files": new_files,
            "executable": False,
            "reason": self._STALE_ANCHOR_REASON,
        }

    def _build_anchor_regen_prompt(
        self,
        rec: dict[str, Any],
        execution_text: str,
        workspace_listing: str,
    ) -> str:
        """Build the prompt asking the judge to adapt one stale anchored claim."""
        missing = [
            rp for rp in (rec.get("likely_relevant_files") or [])
            if rp not in self._workspace_files
        ]
        importance = rec.get("importance", self._DEFAULT_CLAIM_IMPORTANCE)
        rationale = str(rec.get("importance_rationale") or "")
        return f"""You are adapting a verification claim from an ancestor workflow whose file layout has changed.
The original claim referenced files that NO LONGER EXIST in the current workspace. Keep the
checked property identical (same idea, same importance); only the file references should move.

ORIGINAL CLAIM (from the ancestor's cached rubric):
- id:                    {rec.get('id')}
- importance:            {importance} (1-10; 10 = literal deliverable)
- importance_rationale:  {rationale}
- description:           {rec.get('description', '')}
- previously referenced: {rec.get('likely_relevant_files') or []}
- missing in current:    {missing}

CURRENT WORKSPACE FILES (name<TAB>size, relative to workspace root):
{workspace_listing}

AGENT NARRATION (what the current run reported producing — may name the new files):
{execution_text}

TASK:
1. Keep the same checked property — do not rewrite the claim into a different check.
2. Reuse the original id verbatim (rubric lineage depends on it).
3. Update the description ONLY if the new file naming/layout makes the old wording wrong.
4. Pick up to 3 paths from the CURRENT WORKSPACE FILES listing whose contents let a
   deterministic verifier check this claim today. Never invent paths.
5. If no workspace file plausibly holds the artefact this claim was about, return an
   empty list — the verifier will fall back to a soft check rather than run a stale script.

Return STRICT JSON only:
  {{"id": "<verbatim original id>", "description": "<adapted description>", "likely_relevant_files": ["<rel/path>", ...]}}
"""

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
    verifier.evaluate("20260619_104434_4261183e")

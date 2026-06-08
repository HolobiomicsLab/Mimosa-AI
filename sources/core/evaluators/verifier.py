"""Per-claim verifier-based workflow evaluator (orchestrator).

This module owns the public ``VerifierEvaluator`` class and the pipeline that
turns a workflow run into a numeric score plus a textual gradient. The
mechanical steps — extracting claims, generating and running per-claim
scripts, listing the workspace, rendering file previews — live in sibling
modules and are mixed in:

* :mod:`.verifier_claims` — claim extraction (six sources) + importance rating.
* :mod:`.verifier_per_claim` — verifier-script generation, sandbox execution,
  pass/fail/error scoring.
* :mod:`.verifier_workspace` — workspace listing, file previews, literature
  grounding cache.

What stays here is what makes the verifier ``the verifier`` rather than a
generic LLM judge: the orchestration in :meth:`evaluate`, the
importance-weighted aggregation, the on-disk evaluation report, the prompt
gradient handed back to the mutator, and the lineage rubric reuse that lets
descendants score against an ancestor's claim set for stable QD ranking.
"""

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from sources.cli.pretty_print import (
    CYAN,
    RED,
    print_box,
    print_ok,
)

from .base import (
    BaseEvaluator,
    EvaluatorError,
    WorkflowDataError,
)
from .verifier_claims import _VerifierClaimExtractionMixin
from .verifier_per_claim import _VerifierPerClaimMixin
from .verifier_workspace import _VerifierWorkspaceMixin

# ----- Execution limits -------------------------------------------------------
_VERIFIER_TIMEOUT_SECONDS = 180
_VERIFIER_MAX_CLAIMS = 90
_VERIFIER_MIN_CLAIMS = 30
_HARD_FAIL_CAP = 0.99  # disabled so signal stay smooth
_VERIFIER_GEN_PARALLELISM = 16
# Per-claim verifier execution fan-out. Capped low because executable verifier
# scripts can be CPU-bound (numpy/pandas on full artefacts); higher concurrency
# starves rather than helps. Threading is enough because the slow path is
# subprocess I/O via WorkflowRunner.
_VERIFIER_EXEC_PARALLELISM = 4

# bonus(m) = alpha * (1 - exp(-importance_pass_mass / beta)); see _aggregate.
_INFO_BONUS_ALPHA = 0.05
_INFO_BONUS_BETA = 8.0

# ----- Empty-run marker -------------------------------------------------------
_EMPTY_RUN_MARKER = "workflow execution fully failed"

class VerifierEvaluator(
    BaseEvaluator,
    _VerifierClaimExtractionMixin,
    _VerifierPerClaimMixin,
    _VerifierWorkspaceMixin,
):
    """Per-claim verifier-based evaluator.

    The class itself owns the orchestration pipeline; the mechanical steps
    are inherited from the three sibling mixins. ``BaseEvaluator`` supplies
    the LLM judge call helpers (``_call_judge``, ``_call_judge_for_json``,
    ``_get_judge_system_prompt``) that every stage uses.
    """

    _DEFAULT_CLAIM_IMPORTANCE = 5
    _GRADIENT_MIN_IMPORTANCE = 3
    _INFO_BONUS_MIN_IMPORTANCE = 6
    _HARD_FAIL_IMPORTANCE = 8
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
        info_bonus_alpha: float = _INFO_BONUS_ALPHA,
        info_bonus_beta: float = _INFO_BONUS_BETA,
        use_cheat_detector: bool = True,
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
            info_bonus_alpha: Asymptotic ceiling of the information bonus.
            info_bonus_beta: Saturation rate of the information bonus.
            use_cheat_detector: Reserved; cheat detector currently disabled.
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
        self.info_bonus_alpha = max(0.0, info_bonus_alpha)
        self.info_bonus_beta = max(1e-6, info_bonus_beta)
        self.use_cheat_detector = use_cheat_detector
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
            f"use_grounding={use_grounding}, "
            f"info_bonus(α={self.info_bonus_alpha}, β={self.info_bonus_beta}))"
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
        history = "\n".join(self._textual_gradient_history[-5:])
        prompt = f"""
        You convert the judge's report into per claim short, directional diagnosis that steers the
        next mutation.

        GROUND TRUTH AND TRUST ORDER (critical):
        - The verifier's deterministic checks are ground truth.
        - The agent execution narration below is UNTRUSTED. Agents may declare success
          while producing degenerate output. Use agent execution only to explain why and what failure happened, never as
          evidence that the task succeeded.
          Do not mention a failure if it was clearly corrected by agent downstream and verifier report confirm the correction
            Example: error regarding module X reported fixed and verification confirm proper behavior regarding module X).
        - Claim programs that crashed (tracebacks, NameError, serialization errors) are verifier
          measurement failures. Do not report them.

        Here is the agents execution text (agent narration and produced output):
        {execution_text}
        Here is the deterministic verifier's detailed report for workflow {uuid}:
        {report}
        OUTPUT:
        Format: "<diagnosis_CODE>:\n<- <short diagnosis error/success claim 1>\n<- <short diagnosis error/success claim 2>\n... (up to 25 lines of diagnosis)"
        Warning: Do not surface failures that would require modifying provided inputs (e.g. files under `data/`). Surface the next most critical fixable issue instead.
        Example:
        FALLBACK_ECFP_CLASSIFIER:\n-Use of fallback rather than a trained ECFP classifier-\n- Error with numpy: ...\nNo requirements.txt found....
        """
        diag = self._call_judge(
            uuid,
            "verifier_abstract_textual_gradient",
            prompt,
        )
        self._textual_gradient_history.append(diag)
        return diag.strip() or "UNDIAGNOSED:No diagnosis could be extracted from the verifier report."

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        uuid: str,
        rubric_anchor_uuid: str | None = None,
    ) -> dict[str, Any]:
        """Run the verifier pipeline; persists scores under ``evaluation.verifier``.

        When ``rubric_anchor_uuid`` names an ancestor whose verifier cache
        exists (a ``claims.json`` written by a previous evaluation), the claim
        list and executable verifier scripts are reused verbatim from that
        ancestor. This skips the LLM claim-extraction and verifier-generation
        stages, giving identical rubrics across an evolved lineage so scores
        are directly comparable. The anchor's scripts are executed against the
        *current* uuid's workspace, not the anchor's.

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

        # Use explicit empty-run marker rather than the brittle ``success`` flag
        # — ``success`` is ``not "[]" in json.dumps(answers)``, which mis-fires
        # whenever an answer payload contains a (possibly nested) empty list.
        is_truly_empty = (
            not execution_text or _EMPTY_RUN_MARKER in execution_text
        )
        t = time.time()
        grounding = (
            self._get_grounding(uuid, execution_text, wf_info.goal)
            if not is_truly_empty
            else self._GROUNDING_DISABLED
        )
        phase_timings.append(("grounding fetch", time.time() - t))
        print_ok(f"[verifier {uuid}] grounding fetch done in {phase_timings[-1][1]:.1f}s")

        t = time.time()
        anchored_records = (
            self._load_anchored_claims(rubric_anchor_uuid)
            if rubric_anchor_uuid
            else None
        )
        if anchored_records:
            claims = self._claims_from_anchor(anchored_records)
            self.logger.info(
                f"Reusing rubric anchor {rubric_anchor_uuid} for {uuid}: "
                f"{len(claims)} cached claims"
            )
        else:
            if rubric_anchor_uuid:
                self.logger.warning(
                    f"rubric_anchor_uuid={rubric_anchor_uuid} has no readable "
                    f"cache; falling back to LLM claim extraction"
                )
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

        t = time.time()
        self._ensure_verifier_packages()
        phase_timings.append(("ensure verifier packages", time.time() - t))
        print_ok(
            f"[verifier {uuid}] ensure_verifier_packages done in "
            f"{phase_timings[-1][1]:.1f}s"
        )

        anchored_specs = {
            c["id"]: (bool(c.get("executable")), str(c.get("reason") or ""))
            for c in (anchored_records or [])
        }
        claims_to_verify = claims[: self.max_claims]

        # Pre-resolve specs: anchor-cache hits are O(disk read) each, but every
        # other claim costs two judge round-trips (file selection + spec gen).
        # Fan those out via threads so the per-claim loop only pays for the
        # sandbox execution, not for serial LLM latency.
        anchor_preloaded: dict[str, dict[str, Any]] = {}
        needs_generation: list[dict[str, Any]] = []
        for claim in claims_to_verify:
            if anchored_records and rubric_anchor_uuid and claim["id"] in anchored_specs:
                executable, reason = anchored_specs[claim["id"]]
                anchor_preloaded[claim["id"]] = self._spec_from_anchor(
                    rubric_anchor_uuid, claim["id"], executable, reason
                )
            else:
                needs_generation.append(claim)

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

        # Layer 3: independent cheat audit over the agents' produced script.
        cheat = None  # NOTE: cheat_detector was crap. Will need to be rethink.

        self._write_report(uuid, claims, per_claim, scores, cheat=cheat)
        self._persist_claims(uuid, claims, per_claim)

        # The gradient builder only sees the high-importance slice of the
        # report so the mutator is not nudged by low-importance noise. The
        # on-disk report keeps the full view for auditing.
        gradient_report = self._build_report(
            per_claim,
            scores,
            cheat,
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

        Claim ordering is preserved so the downstream report, aggregator, and
        rubric cache see the same order they would in the old sequential loop.
        A per-claim exception is converted into an ``error`` result rather than
        crashing the whole batch — mirrors the soft-fail contract of
        ``_generate_specs_parallel``.

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
            "cheat_penalty": 0.0,
        }
        try:
            self._write_report(uuid, [], [], scores, cheat=None)
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
        spec_by_id = {
            (c.get("claim") or {}).get("id"): (c.get("spec") or {})
            for c in per_claim
        }
        rubric = [
            self._rubric_record(c, spec_by_id.get(c.get("id")) or {})
            for c in claims
        ]
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
    # Stage 5 — importance-weighted aggregation
    # ------------------------------------------------------------------

    def _information_bonus(self, importance_pass_mass: float) -> float:
        """Saturating reward for thoroughness; gameable spam yields no extra credit.

        bonus(m) = α · (1 − exp(−m / β)). Bounded above by α, monotonic in m,
        and conditional on the *passing high-importance mass* so trivial or
        failed claims contribute nothing. ``m`` is the sum of importance
        (capped at 10 per claim) for passes with importance ≥ 6, divided by 10
        — units are roughly "equivalent number of importance-10 passes".

        Args:
            importance_pass_mass: Importance-weighted pass mass (see above).

        Returns:
            Non-negative bonus value, asymptotically bounded by ``info_bonus_alpha``.
        """
        if importance_pass_mass <= 0.0 or self.info_bonus_alpha <= 0.0:
            return 0.0
        return self.info_bonus_alpha * (
            1.0 - math.exp(-importance_pass_mass / self.info_bonus_beta)
        )

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
        """Importance-weighted mean + thoroughness bonus; capped on top-tier fail.

        Each claim is weighted by its rater-assigned importance (1-10) rather
        than the old hard=3/soft=1 step function. This gives the optimizer a
        smooth gradient: flipping an importance-10 deliverable claim moves the
        score ~5× more than flipping a low-importance hygiene claim.

        The thoroughness bonus saturates on importance-weighted mass of
        high-importance passes (importance ≥ ``_INFO_BONUS_MIN_IMPORTANCE``).
        The hard-fail cap fires when any claim with importance
        ≥ ``_HARD_FAIL_IMPORTANCE`` is refuted (not errored, not unsure).

        Args:
            per_claim: List of per-claim scored dicts from ``_verify_claim``.

        Returns:
            Aggregate score dict with overall/base/bonus/cap and per-status counts.
        """
        if not per_claim:
            return self._empty_aggregate_result(n_claims=0)

        scored = [c for c in per_claim if c.get("status") != "error"]
        n_pass = sum(1 for c in per_claim if c["status"] == "pass")
        n_fail = sum(1 for c in per_claim if c["status"] == "fail")
        n_error = sum(1 for c in per_claim if c["status"] == "error")
        n_unsure = sum(1 for c in per_claim if c["status"] == "unsure")
        # High-importance passes drive the thoroughness bonus.
        high_imp_passes = [
            c for c in scored
            if c["status"] == "pass"
            and self._claim_weight(c) >= self._INFO_BONUS_MIN_IMPORTANCE
        ]
        n_high_importance_pass = len(high_imp_passes)
        high_importance_pass_mass = (
            sum(self._claim_weight(c) for c in high_imp_passes) / 10.0
        )

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
        bonus = self._information_bonus(high_importance_pass_mass)
        # Pre-cap: clamp to [0, 1] before applying the hard-fail cap so the
        # bonus can never push past 1.0 nor rescue a broken run.
        pre_cap = max(0.0, min(1.0, base_mean + bonus))
        # Hard-fail cap fires only on a real refutation of a top-importance
        # claim — not on errors or unsure verdicts.
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
            "information_bonus": round(bonus, 4),
            "n_high_importance_pass": n_high_importance_pass,
            "high_importance_pass_mass": round(high_importance_pass_mass, 4),
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
            "information_bonus": 0.0,
            "n_high_importance_pass": 0,
            "high_importance_pass_mass": 0.0,
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

    # ------------------------------------------------------------------
    # Cheat penalty + fallback textual gradient
    #
    # These two methods are reserved for the cheat-detector rewrite (see the
    # ``cheat = None`` in ``evaluate`` and the ``_CHEAT_*`` constants at the
    # top of the file). They are kept here, not called, so the rewrite can
    # re-wire them without re-deriving their contracts.
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_cheat_penalty(
        scores: dict[str, Any], cheat: Any
    ) -> dict[str, Any]:
        """Subtract cheat penalty from the capped overall score; floor at 0.0.

        Args:
            scores: Mutable score dict from ``_aggregate`` to be amended.
            cheat: Cheat-detector report exposing ``penalty`` and ``to_dict()``,
                or ``None`` to apply a zero penalty.

        Returns:
            The same ``scores`` dict, updated with cheat-related fields.
        """
        penalty = float(cheat.penalty) if cheat is not None else 0.0
        capped = float(scores.get("overall_score", 0.0))
        final = max(0.0, capped - penalty)
        scores["overall_score_before_cheat"] = round(capped, 4)
        scores["cheat_penalty"] = round(penalty, 4)
        scores["overall_score"] = round(final, 4)
        if cheat is not None:
            scores["cheat_detector"] = cheat.to_dict()
        return scores

    @staticmethod
    def _fallback_textual_gradient(
        scores: dict[str, Any], cheat: Any
    ) -> str:
        """Deterministic fallback when the abstractor LLM is unavailable.

        Args:
            scores: Aggregated score dict from ``_aggregate``.
            cheat: Cheat-detector report with optional ``behavioral`` findings.

        Returns:
            Single-sentence human-readable summary of the run's outcome.
        """
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

    def _build_report(
        self,
        per_claim: list[dict[str, Any]],
        scores: dict[str, Any],
        cheat: Any,
        min_importance: int = 0,
    ) -> str:
        """Render a plain-text report from per-claim results and aggregate scores.

        Args:
            per_claim: List of per-claim scored dicts.
            scores: Aggregated score dict from ``_aggregate``.
            cheat: Cheat-detector report, or ``None``.
            min_importance: When > 0, only claims with importance ≥ this value
                are rendered. The aggregate header still reflects the full run.
                Used to build a noise-suppressed view for the prompt-gradient
                builder, while the on-disk ``evaluation.txt`` keeps the full
                report (``min_importance=0``).

        Returns:
            Multi-line report string terminated with a newline.
        """
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
            f"n_high_importance_pass={scores.get('n_high_importance_pass', 0)}  "
            f"high_importance_pass_mass={scores.get('high_importance_pass_mass', 0.0):.3f}  "
            f"cheat_penalty={scores.get('cheat_penalty', 0.0):.3f}"
        )
        if min_importance > 0:
            w(f"(filtered view: importance ≥ {min_importance})")

        for c in per_claim:
            cl = c["claim"]
            imp = int(cl.get("importance", self._DEFAULT_CLAIM_IMPORTANCE))
            if imp < min_importance:
                continue
            rationale = str(cl.get("importance_rationale") or "").strip()
            header = (
                f"[{cl['id']}] (importance={imp}; {rationale}) {cl['description']}"
                if rationale
                else f"[{cl['id']}] (importance={imp}) {cl['description']}"
            )
            w(header)
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
        cheat: Any,
    ) -> None:
        """Persist the built report under ``<workflow_dir>/<uuid>/evaluation.txt``.

        Args:
            uuid: Workflow identifier; selects the on-disk output folder.
            claims: Raw extracted claim list (kept for signature parity).
            per_claim: List of per-claim scored dicts to render.
            scores: Aggregated score dict from ``_aggregate``.
            cheat: Cheat-detector report, or ``None``.
        """
        path = self.workflow_dir / uuid / "evaluation.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        report = self._build_report(per_claim, scores, cheat)
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
    verifier.evaluate("20260528_090358_ff330fde")

"""``HybridVerifierEvaluator`` — the E35 PRIME (v3) verification channel.

Subclasses the repo's ``BaseEvaluator`` (judge LLM machinery, workflow
loading, ``_save_results`` persistence under ``evaluation.verifier``)
and orchestrates the verifier-v3 pipeline per generation:

1. short-circuit parity for empty runs (synthetic
   ``execution_produced_artifacts`` claim, reward 0.0, non-empty
   gradient);
2. workspace inventory + per-task registry load/merge;
3. evidence layers, in ladder order: the E26 visual rung for figure
   tasks (detection >=50% image deliverables; goal-anchored visual
   claims frozen before scoring, kimi-k3 at temperature 0) → claims
   extraction (E30 content lever + E29 T1 firewall) → scorer
   build/repair → scorer execution; see ``layers``;
4. variance filter over the registry's score history; pairwise reward
   against every previous generation of the task under the task's
   ladder order (``FIGURE_STAGES`` for figure tasks — the visual rung
   decides FIRST; ``STAGES`` otherwise); layer gates (the E24
   execution gate caps crash/no-entry/timeout at 0.0, divergent at
   0.5);
5. V5 decisive-first gradient + ``evaluation.txt`` +
   ``state_result.json`` persistence.

LLM access reuses the judge ``LLMConfig`` machinery from ``BaseEvaluator``
(judge model, cached provider calls) with temperature pinned to 0.0 — the
E19 measured convention; visual scoring runs on the dedicated
``vision_judge_model`` (kimi-k3, temperature 0) through the same cached
provider with base64 image parts. The E19/E19b harnesses additionally
disable the DeepSeek reasoner via a provider-specific ``extra_body``
flag; the repo ``LLMProvider`` does not expose that knob, so this module
uses the same safe fallback the legacy verifier runs in production (no
reasoning parameter for non-GPT reasoning models, provider-level retry
on empty responses) and its own JSON/code repair loops.

Scorer execution uses the existing pinned-subprocess pattern:
``RuntimeConfig(python_executable=sys.executable, use_pty=False)`` +
``WorkflowRunner`` with the workspace as execution dir. No sandbox or
timeout behavior is weakened.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import sys
import threading
import time
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config import Config

from sources.cli.pretty_print import CYAN, RED, print_box, print_ok
from sources.core.failure_fingerprint import DESCRIPTOR_DIM as _FP_DIM
from sources.core.llm_provider import LLMConfig, LLMProvider
from sources.core.workflow_runner import RuntimeConfig, WorkflowRunner
from sources.evaluators.base import (
    BaseEvaluator,
    EvaluatorError,
)

from . import claims as claims_mod
from . import gradient as gradient_mod
from . import inventory as inv_mod
from .aggregation import (
    bradley_terry_reward,
    claim_stats,
    mean_claim_score,
    win_rate_reward,
)
from .layers import (
    ClaimsEvidenceLayer,
    EvidenceLayer,
    ExecutionGateLayer,
    LayerContext,
    LayerScore,
    VisualEvidenceLayer,
)
from .registry import TaskRegistry
from .scorers import ExecOutcome

_HYBRID_SYSTEM_PROMPT = (
    "You are a precise engineering assistant that follows output-format "
    "contracts exactly. When asked for strict JSON, reply with only the "
    "JSON object. When asked for a fenced Python block, reply with only "
    "that block."
)

_RUNNER_CLEANUP_TIMEOUT = 15
_RUNNER_EXTRA_TIMEOUT = 10

_SHORT_CIRCUIT_GRADIENT = (
    "workflow code failed to generate or execute; ensure code is properly "
    "formatted and that the workflow runs without crashing"
)


def _run_coro_sync(
    coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    thread_timeout: float | None = None,
) -> Any:
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
    coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    thread_timeout: float | None,
) -> Any:
    """Spawn a daemon thread to run *coro_factory*; raise on timeout/error."""
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
        raise TimeoutError(f"Worker thread did not finish within {thread_timeout}s")
    if "error" in holder:
        raise holder["error"]
    return holder["result"]


class HybridVerifierEvaluator(BaseEvaluator):
    """E35 PRIME (v3) hybrid verifier: claims + deterministic policy
    scorers + an early visual rung + an execution gate + a pairwise
    temporal-ladder reward.

    The pairwise policy (``temporal`` elimination by default) produces
    win/loss/tie outcomes against every previous generation of the task;
    the reward aggregates them as a raw win-rate (default — measured best
    on the frozen arena, E19c 2026-09-25) or as a Bradley–Terry
    logistic-MLE strength (``hybrid_verifier_reward="bradley_terry"``),
    then the E24 execution gate caps it by re-execution status.
    """

    def __init__(
        self,
        config: Config,
        workspace_dir: str | Path | None = None,
        num_claims: int = 10,
        refinement_rounds: int = 2,
        scorer_timeout_s: int = 60,
        pairwise_mode: str = "temporal",
        reward_mode: str = "win_rate",
        gen_parallelism: int = 8,
        digest_max_files: int = 8,
        extra_layers: list[EvidenceLayer] | None = None,
        visual_rung: bool = True,
        execution_gate: bool = True,
    ) -> None:
        """Initialise the hybrid verifier; see the package docstring for the design.

        Args:
            config: Mimosa Config object (paths, judge model, pricing).
            workspace_dir: Override for the agents' workspace root; defaults
                to ``config.workspace_dir``.
            reward_mode: ``win_rate`` (default — raw majority score vs
                previous generations; measured best on the frozen arena,
                E19c 2026-09-25) or ``bradley_terry`` (logistic-MLE
                strengths over all pairwise outcomes among the task's
                generations, reward = sigmoid(beta_now − mean beta)).
                ``mean_diff``, ``sign_sum``, ``escalation``.
            gen_parallelism: Max concurrent scorer build/execute workers.
            digest_max_files: Cap on files sampled by the format digest.
            extra_layers: Additional evidence layers chained after the
                default claims layer (E24-style extension point).
            visual_rung: E26/E35 visual rung for figure tasks (goal-
                anchored visual claims scored by the vision model, ladder
                stage BEFORE ``script``); config knob
                ``hybrid_verifier_visual_rung`` (default True).
            execution_gate: E24/E35 execution gate — crash/no-entry/
                timeout re-execution caps the reward at 0.0, divergent at
                0.5; config knob ``hybrid_verifier_execution_gate``
                (default True).
        """
        super().__init__(config)
        self.workspace_dir = Path(
            workspace_dir
            if workspace_dir is not None
            else getattr(config, "workspace_dir", ".")
        )
        self.num_claims = max(
            3, int(getattr(config, "hybrid_verifier_num_claims", num_claims))
        )
        self.refinement_rounds = max(
            0,
            int(
                getattr(config, "hybrid_verifier_refinement_rounds", refinement_rounds)
            ),
        )
        self.scorer_timeout_s = max(
            5,
            int(getattr(config, "hybrid_verifier_scorer_timeout_s", scorer_timeout_s)),
        )
        self.pairwise_mode = str(
            getattr(config, "hybrid_verifier_pairwise_mode", pairwise_mode)
        )
        self.reward_mode = str(
            getattr(config, "hybrid_verifier_reward", reward_mode)
        ).lower()
        if self.reward_mode not in ("bradley_terry", "win_rate"):
            self.reward_mode = "bradley_terry"
        self.gen_parallelism = max(1, int(gen_parallelism))
        self.visual_rung = bool(
            getattr(config, "hybrid_verifier_visual_rung", visual_rung)
        )
        self.execution_gate = bool(
            getattr(config, "hybrid_verifier_execution_gate", execution_gate)
        )
        self._runner_temp_root = Path(
            getattr(config, "temp_dir", None) or self.workflow_dir / "_verifier_tmp"
        )
        # E19 measured convention: deterministic generation at temperature 0.
        self.llm_config.temperature = 0.0
        self._vision_llm_config = self._build_vision_config(config)
        self._claims_layer = ClaimsEvidenceLayer(
            llm_text=self._llm_text,
            run_script=self._execute_scorer_script,
            logger=self.logger,
            num_claims=self.num_claims,
            refinement_rounds=self.refinement_rounds,
            scorer_timeout_s=self.scorer_timeout_s,
            gen_parallelism=self.gen_parallelism,
            digest_max_files=max(
                1,
                int(
                    getattr(
                        config, "hybrid_verifier_digest_max_files", digest_max_files
                    )
                ),
            ),
        )
        layers: list[EvidenceLayer] = [self._claims_layer]
        self._visual_layer: VisualEvidenceLayer | None = None
        if self.visual_rung:
            self._visual_layer = VisualEvidenceLayer(
                vision_call=self._vision_text,
                logger=self.logger,
            )
            layers.insert(0, self._visual_layer)
        self._gate_layer: ExecutionGateLayer | None = None
        if self.execution_gate:
            self._gate_layer = ExecutionGateLayer(logger=self.logger)
            layers.append(self._gate_layer)
        self.layers: list[EvidenceLayer] = [*layers, *(extra_layers or [])]
        self.logger.info(
            f"HybridVerifierEvaluator initialized (workspace={self.workspace_dir}, "
            f"claims={self.num_claims}, refinement_rounds={self.refinement_rounds}, "
            f"scorer_timeout={self.scorer_timeout_s}s, pairwise={self.pairwise_mode}, "
            f"visual_rung={self.visual_rung}, execution_gate={self.execution_gate})"
        )

    def _build_vision_config(self, config: Config) -> Any | None:
        """Vision-model LLMConfig (kimi-k3 transport, temperature 0)."""
        vision_model = getattr(config, "vision_judge_model", None)
        if not vision_model:
            self.logger.info(
                "No vision_judge_model configured; visual rung will "
                "soft-fail its vision calls"
            )
            return None
        provider, model = (
            vision_model.split("/", 1)
            if "/" in vision_model
            else ("openai", vision_model)
        )
        return LLMConfig().from_dict(
            {
                "model": model,
                "provider": provider,
                # E26 measured convention: deterministic visual scoring.
                "temperature": 0.0,
                "reasoning_effort": config.reasoning_effort,
                "max_tokens": 2048,
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
            }
        )

    # ------------------------------------------------------------------
    # LLM + subprocess plumbing
    # ------------------------------------------------------------------

    def _llm_text(self, uuid: str, agent_name: str, prompt: str) -> str:
        """One judge round-trip through the cached repo provider."""
        memory_path = Path(self.memory_dir) / uuid
        memory_path.mkdir(parents=True, exist_ok=True)
        provider = LLMProvider(
            agent_name=agent_name,
            memory_path=memory_path,
            system_msg=_HYBRID_SYSTEM_PROMPT,
            config=self.llm_config,
        )
        return provider(prompt)

    def _vision_text(
        self, uuid: str, agent_name: str, prompt: str, images: list[Path]
    ) -> str:
        """One vision-model round-trip with image files attached (E26).

        Mirrors the legacy verifier's multimodal transport: text prompt +
        base64 data-URI image parts through ``LLMProvider`` on the
        dedicated ``vision_judge_model`` (kimi-k3, temperature 0). Raises
        when no vision model is configured — callers degrade per-call.
        """
        if self._vision_llm_config is None:
            raise RuntimeError(
                "no vision_judge_model configured; cannot run the visual rung"
            )
        content: list[Any] = [{"type": "text", "text": prompt}]
        for img in images:
            b64 = base64.b64encode(img.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                }
            )
        memory_path = Path(self.memory_dir) / uuid
        memory_path.mkdir(parents=True, exist_ok=True)
        provider = LLMProvider(
            agent_name=agent_name,
            memory_path=memory_path,
            system_msg=_HYBRID_SYSTEM_PROMPT,
            config=self._vision_llm_config,
        )
        return provider(content)

    def _build_runner(self, uuid: str, workspace: Path) -> WorkflowRunner:
        """Pinned-subprocess runner: ``sys.executable``, cwd = workspace."""
        scratch = self._runner_temp_root / uuid
        scratch.mkdir(parents=True, exist_ok=True)
        cfg = RuntimeConfig(
            python_executable=sys.executable,
            timeout=self.scorer_timeout_s,
            temp_dir=scratch,
            requirements_file=None,
            use_pty=False,
        )
        return WorkflowRunner(cfg, execution_dir=str(workspace))

    def _execute_scorer_script(
        self, workspace: Path, code: str, execution_id: str
    ) -> ExecOutcome:
        """Run one scorer script; soft-fails to a tagged ExecOutcome."""
        runner = self._build_runner(f"hybrid_{execution_id}", workspace)
        try:
            result = _run_coro_sync(
                lambda: runner.execute(code, execution_id=execution_id),
                thread_timeout=self.scorer_timeout_s + _RUNNER_EXTRA_TIMEOUT,
            )
            status = result.status
            timed_out = str(status) == "ExecutionStatus.TIMEOUT"
            return ExecOutcome(
                rc=int(result.return_code if result.return_code is not None else -1),
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                timeout=timed_out,
            )
        except TimeoutError as e:
            return ExecOutcome(rc=-1, stdout="", stderr=str(e), timeout=True)
        except Exception as e:  # noqa: BLE001 — degrade per-claim, never crash
            return ExecOutcome(rc=-1, stdout="", stderr=f"{type(e).__name__}: {e}")
        finally:
            with contextlib.suppress(Exception):  # best-effort cleanup
                _run_coro_sync(runner.cleanup, thread_timeout=_RUNNER_CLEANUP_TIMEOUT)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(self, uuid: str) -> dict[str, Any]:
        """Run the hybrid pipeline; persists scores under ``evaluation.verifier``.

        Args:
            uuid: Workflow identifier to evaluate.

        Returns:
            Dict with ``uuid``, the per-claim results and aggregate scores.

        Raises:
            EvaluatorError: When ``uuid`` is not a non-empty string.
        """
        if not uuid or not isinstance(uuid, str):
            raise EvaluatorError("Invalid uuid: must be a non-empty string")

        t_total = time.time()
        wf_info = self._load_workflow_data(uuid)

        # Short-circuit when the workflow produced nothing: avoids scoring
        # against stale state from a prior generation in a shared workspace.
        if not wf_info.state_result or not wf_info.code:
            return self._short_circuit_failed_run(uuid)

        goal = (wf_info.goal or "").strip()
        if not goal:
            return self._short_circuit_failed_run(
                uuid,
                reason="no_goal_recorded",
                gradient="no goal was recorded for this workflow; the verifier "
                "cannot build a task rubric without the task goal",
            )

        inventory = inv_mod.scan_workspace(self.workspace_dir)
        registry = TaskRegistry.load(self._runner_temp_root, goal, logger=self.logger)
        registry.merge_inventory(inventory)
        context = LayerContext(
            uuid=uuid,
            goal=goal,
            workspace=self.workspace_dir,
            registry=registry,
            generation_index=len(registry.generations),
        )

        # ---- evidence collection (all layers; failures degrade per-item) ----
        collected: list[LayerScore] = []
        for layer in self.layers:
            try:
                collected.extend(layer.collect(context))
            except Exception as e:  # noqa: BLE001 — a layer never kills the run
                self.logger.warning(
                    f"[{uuid}] evidence layer {getattr(layer, 'name', '?')} "
                    f"raised: {type(e).__name__}: {e}"
                )

        now_scores: dict[str, float | None] = {s.claim_id: s.score for s in collected}
        evidence = {s.claim_id: s.evidence for s in collected}

        # Record first (reward filled after aggregation) so per-claim stats
        # see the current generation; upsert keeps re-evaluation idempotent.
        registry.record_generation(uuid, now_scores, evidence, reward=None)

        # ---- variance filter over the full observed history ----
        def _update_claim_states() -> list[str]:
            """Mark dead/alive from observed history; return surviving ids."""
            surv: list[str] = []
            for c in registry.claims:
                cid = c["id"]
                stats = claim_stats(registry.observed_scores(cid))
                if stats["dropped"]:
                    registry.mark_dead(cid, stats["drop_reason"])
                else:
                    registry.mark_alive(cid)
                    if now_scores.get(cid) is not None:
                        surv.append(cid)
            return surv

        surviving = _update_claim_states()

        # ---- intra-generation refinement: layers may replace dead claims ----
        dead_ids = [c["id"] for c in registry.claims if c.get("state") == "dead"]
        for layer in self.layers:
            revise = getattr(layer, "revise", None)
            if revise is None or not callable(revise) or not dead_ids:
                continue
            try:
                extra = revise(context, dead_ids)
            except Exception as e:  # noqa: BLE001 — refinement never kills the run
                self.logger.warning(
                    f"[{uuid}] revise of layer {getattr(layer, 'name', '?')} "
                    f"raised: {e}"
                )
                continue
            if not extra:
                continue
            collected.extend(extra)
            for s in extra:
                now_scores[s.claim_id] = s.score
                evidence[s.claim_id] = s.evidence
            registry.record_generation(uuid, now_scores, evidence, reward=None)
            surviving = _update_claim_states()

        # ---- pairwise reward vs every previous generation ----
        previous = registry.previous_generations(uuid)
        prev_vectors = [g.get("scores") or {} for g in previous]
        # Figure tasks (E26/E35): the visual rung sits BEFORE "script" on
        # the ladder, so a visual-stage failure dominates every later one.
        figure_task = inv_mod.is_figure_task(registry.inventory)
        stage_order = claims_mod.FIGURE_STAGES if figure_task else claims_mod.STAGES
        # Surviving claims in TEMPORAL order (stage rank, temporal_index).
        surviving_claims = [
            c
            for c in claims_mod.sort_claims_temporally(registry.claims, stage_order)
            if c["id"] in set(surviving)
        ]
        reward_data = win_rate_reward(
            now_scores,
            prev_vectors,
            surviving_claims,
            mode=self.pairwise_mode,
            stages=stage_order,
        )
        mean_score = mean_claim_score(now_scores, surviving)
        reward = float(reward_data["reward"])
        bt_data: dict[str, Any] | None = None
        reward_mode = getattr(self, "reward_mode", "win_rate")
        if reward_mode == "bradley_terry" and prev_vectors:
            bt_data = bradley_terry_reward(
                now_scores,
                prev_vectors,
                surviving_claims,
                mode=self.pairwise_mode,
                stages=stage_order,
            )
            reward = float(bt_data["reward"])

        # ---- layer gates (optional caps; None = no cap) ----
        caps = []
        exec_facts: dict[str, Any] | None = None
        for layer in self.layers:
            try:
                cap = layer.gate(context, collected)
            except Exception as e:  # noqa: BLE001
                self.logger.warning(
                    f"[{uuid}] gate of layer {getattr(layer, 'name', '?')} raised: {e}"
                )
                cap = None
            if cap is not None:
                caps.append(float(cap))
            facts = getattr(layer, "last_facts", None)
            if isinstance(facts, dict):
                exec_facts = facts
        if caps:
            reward = min([reward, *caps])
        pair_records = []
        for rec, gen in zip(reward_data["pairs"], previous, strict=False):
            record: dict[str, Any] = {
                "prev_uuid": gen.get("uuid"),
                "prev_scores": gen.get("scores") or {},
                "outcome": rec["outcome"],
            }
            if "detail" in rec:
                record["detail"] = rec["detail"]
            else:
                record.update(
                    d=rec.get("d", 0), dm=rec.get("dm", 0.0), k_eff=rec.get("k_eff", 0)
                )
            pair_records.append(record)

        any_measured = any(s is not None for s in now_scores.values())
        unmeasured_prior = None
        if not any_measured and not caps:
            if inv_mod.workspace_has_artifacts(inventory):
                # N9 (E41 phonon gen5): every scorer failed on a workspace
                # that produced real artifacts — an instrumentation failure,
                # not a workflow failure. A hard 0.0 buries such generations
                # (the SR-true gen5 scored 0.0 while SR-false gen8's 0.5714
                # won the argmax and shipped) and lies to the mutator ("you
                # regressed to trash" when measurement, not quality,
                # collapsed). Degrade to the neutral midpoint and flag the
                # reward as fallback-scale: capsule selection excludes it
                # from the argmax whenever measured siblings exist, and the
                # N1-residual branch accepts it over all-zero crashed pools.
                reward = 0.5
                unmeasured_prior = "unmeasured_prior"
            else:
                # Empty workspace: nothing was produced — 0.0 is fair.
                reward = 0.0
        # (n_scored == 0 WITH an applied gate cap keeps the cap's verdict:
        # the E24 re-execution measured the crash/divergence directly.)

        dead_claims = [
            {**c, "drop_reason": c.get("drop_reason") or "non_discriminative"}
            for c in registry.claims
            if c.get("state") == "dead"
        ]
        scores = {
            "overall_score": round(max(0.0, min(1.0, reward)), 4),
            "overall_score_uncapped": round(mean_score, 4),
            "mean_claim_score": round(mean_score, 4),
            "win_rate": round(float(reward_data["reward"]), 4),
            "n_pairs": reward_data["n_pairs"],
            "n_wins": reward_data["wins"],
            "n_losses": reward_data["losses"],
            "n_ties": reward_data["ties"],
            "pairwise_mode": self.pairwise_mode,
            "reward_mode": reward_mode,
            "figure_task": figure_task,
            "stage_order": list(stage_order),
            "execution_gate": exec_facts,
            "bt_strength": (bt_data or {}).get("strength_now"),
            "bt_converged": (bt_data or {}).get("converged"),
            "reward_fallback": unmeasured_prior or reward_data["fallback"],
            "n_claims": len(registry.claims),
            "n_surviving": len(surviving),
            "n_dropped": len(dead_claims),
            "n_scored": sum(1 for s in now_scores.values() if s is not None),
            "n_scorer_failures": sum(1 for s in now_scores.values() if s is None),
            "n_replacements": int(registry.refinements.get(uuid, 0)),
            "claim_summary": [
                {
                    "id": s.claim_id,
                    "category": s.category,
                    "statement": s.statement[:512],
                    "score": s.score,
                    "surviving": s.claim_id in surviving,
                    "evidence": (s.evidence or "")[:512],
                }
                for s in collected
            ],
        }
        if not any_measured:
            scores["skipped_reason"] = "all_scorers_failed"

        # ---- artifacts ----
        textual_gradient = gradient_mod.build_gradient(
            uuid=uuid,
            goal=goal,
            now_scores=now_scores,
            evidence=evidence,
            claims=claims_mod.sort_claims_temporally(registry.claims, stage_order),
            surviving=surviving,
            pair_records=pair_records,
            reward=scores["overall_score"],
            win_rate=scores["win_rate"],
            mean_score=mean_score,
            dead_claims=dead_claims,
            exec_facts=exec_facts,
            unmeasured=unmeasured_prior is not None,
            stages=stage_order,
        )
        scores["abstracted_textual_gradient"] = textual_gradient
        # Legacy-typo key kept for one release so old readers keep working;
        # the correct key above is primary (fixes VERIFIER_ARCHITECTURE §6).
        scores["abstractec_textual_gradient"] = textual_gradient
        scores["failure_fingerprint"] = {
            "vector": [0.0] * _FP_DIM,
            "presence_mask": [0.0] * _FP_DIM,
            "pass_rates": [0.5] * _FP_DIM,
        }

        registry.record_generation(
            uuid, now_scores, evidence, reward=scores["overall_score"]
        )
        registry.save()

        report = gradient_mod.build_evaluation_report(
            uuid,
            scores,
            claims_mod.sort_claims_temporally(registry.claims, stage_order),
            now_scores,
            evidence,
            surviving,
            pair_records,
        )
        self._write_artifacts(uuid, report, textual_gradient)
        try:
            self._save_results(scores, uuid, "verifier")
        except Exception as e:  # noqa: BLE001 — persistence failure logged only
            self.logger.error(
                f"Failed to persist hybrid verifier scores for {uuid}: {e}"
            )

        print_box(
            f"reward {scores['overall_score']:.3f} | mean claim "
            f"{mean_score:.3f} | {scores['n_surviving']}/{scores['n_claims']} "
            f"claims survive | {scores['n_wins']}W/{scores['n_losses']}L/"
            f"{scores['n_ties']}T vs {scores['n_pairs']} previous | "
            f"{time.time() - t_total:.1f}s",
            title=f"Hybrid verifier · {uuid}",
            color=CYAN,
        )
        return {
            "uuid": uuid,
            "claims": [{**vars(s)} for s in collected],
            **scores,
        }

    # ------------------------------------------------------------------
    # Short-circuit (empty-run parity with the legacy verifier)
    # ------------------------------------------------------------------

    def _short_circuit_failed_run(
        self,
        uuid: str,
        reason: str = "workflow_generation_or_execution_failed",
        gradient: str = _SHORT_CIRCUIT_GRADIENT,
    ) -> dict[str, Any]:
        """Score an empty run: synthetic artifact claim, reward 0.0.

        Mirrors the legacy sentinel claim (``c0_execution_succeeded``):
        one synthetic claim ``execution_produced_artifacts`` scored 0/1
        from the workspace listing, reward 0.0 on failure, and a
        NON-EMPTY gradient so the engine never sees an empty one from
        the short-circuit path.
        """
        print_box(
            f"workflow {uuid} produced no code and no state_result "
            f"({reason}); hybrid verifier returns 0.0 without running "
            f"scorers.",
            title="Hybrid verifier short-circuit — generation failed",
            color=RED,
        )
        inventory = inv_mod.scan_workspace(self.workspace_dir)
        artifact_score = 1.0 if inv_mod.workspace_has_artifacts(inventory) else 0.0
        claim_summary = [
            {
                "id": "execution_produced_artifacts",
                "category": "completeness",
                "statement": "the workflow execution produced artifact files "
                "in the workspace",
                "score": artifact_score,
                "surviving": True,
                "evidence": (
                    f"{len(inventory)} files listed in the workspace"
                    if inventory
                    else "workspace empty"
                ),
            }
        ]
        gradient_text = (
            f"{gradient}\n"
            f"short-circuit check: execution_produced_artifacts="
            f"{artifact_score:.0f} from the workspace listing."
        )
        scores = {
            "overall_score": 0.0,
            "overall_score_uncapped": 0.0,
            "mean_claim_score": 0.0,
            "win_rate": 0.0,
            "n_pairs": 0,
            "n_wins": 0,
            "n_losses": 0,
            "n_ties": 0,
            "pairwise_mode": self.pairwise_mode,
            "reward_fallback": "short_circuit",
            "n_claims": 1,
            "n_surviving": 1,
            "n_dropped": 0,
            "n_scored": 1,
            "n_scorer_failures": 0,
            "n_replacements": 0,
            "claim_summary": claim_summary,
            "skipped_reason": reason,
            "abstracted_textual_gradient": gradient_text,
            "abstractec_textual_gradient": gradient_text,
            "failure_fingerprint": {
                "vector": [0.0] * _FP_DIM,
                "presence_mask": [0.0] * _FP_DIM,
                "pass_rates": [0.5] * _FP_DIM,
            },
        }
        report = (
            "Hybrid Verifier Evaluation (E19/E19b)\n" + "=" * 60 + "\n"
            f"Short-circuit: {reason}\n"
            "Claims: 1 (synthetic)  surviving=1\n"
            "Reward: 0.000 (execution failed)\n\n"
            "[execution_produced_artifacts] (completeness; SURVIVING) the "
            "workflow execution produced artifact files in the workspace\n"
            f"  score: {artifact_score:.3f}\n"
            f"  evidence: {claim_summary[0]['evidence']}\n"
        )
        self._write_artifacts(uuid, report, gradient_text)
        try:
            self._save_results(scores, uuid, "verifier")
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to persist short-circuit scores for {uuid}: {e}")
        return {"uuid": uuid, "claims": claim_summary, **scores}

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def _write_artifacts(self, uuid: str, report: str, gradient_text: str) -> None:
        """Write ``evaluation.txt`` + ``textual_gradient.txt`` for *uuid*."""
        out_dir = self.workflow_dir / uuid
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "evaluation.txt").write_text(report, encoding="utf-8")
            (out_dir / "textual_gradient.txt").write_text(
                gradient_text, encoding="utf-8"
            )
            print_ok(f"[hybrid {uuid}] evaluation.txt + textual_gradient.txt written")
        except OSError as e:
            self.logger.error(f"Could not write artifacts for {uuid}: {e}")

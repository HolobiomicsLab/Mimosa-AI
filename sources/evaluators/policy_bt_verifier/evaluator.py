"""``PolicyBTVerifierEvaluator`` — the E48-inspired policy/BT verifier.

Productionization of the offline E44-E48 architecture (plan §1): per task,
ONE deterministic workspace digest per generation (backed by the
verifier's OWN fresh re-execution), ONE LLM reconnaissance config, ONE
LLM-written deterministic pairwise-comparison policy (AST-screened,
smoke-tested, repair-looped with every E48 safeguard), kimi per-PAIR
visual comparison for figure-dominant tasks, and a Bradley-Terry-shaped
tournament whose DEFAULT reward is the arrival-moment raw win-rate vs
predecessors (E19c convention — history is never retroactively
rewritten). The gradient is the elimination-first fold of the
generation's pairwise feedbacks, optionally refined by the E47
evidence-checked refiner.

Channel selection follows the measured split: kimi decides
figure-dominant deliverables (>=50% image share, E26), the code policy
decides data deliverables (including image-bearing-but-programmatic
tasks). Every stage emits a ``[policy_bt][<uuid8>] STAGE`` stdout line
and a JSON record to ``verifier_policy_bt.log`` (plan §6b).

Persistence mirrors the hybrid verifier: per-task registry under the
runner temp root (goal-hash keyed) stores digests/ledgers/verdicts/
policy/recon, so predecessor comparisons never re-read ``/tmp`` or the
live workspace (plan §5b).

``evaluate`` is an orchestrator: each pipeline stage lives in its own
``_stage``-shaped helper (all <= 40 lines) that reads and writes the
shared ``_EvalState`` — see ``PolicyBTVerifierEvaluator.evaluate`` for
the stage order.
"""

from __future__ import annotations

import base64
import dataclasses
import functools
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sources.cli.pretty_print import CYAN, GREEN, RED, print_box, print_ok
from sources.core.failure_fingerprint import DESCRIPTOR_DIM as _FP_DIM
from sources.core.llm_provider import LLMConfig, LLMProvider
from sources.evaluators.base import (
    BaseEvaluator,
    EvaluatorError,
    extract_json_payload,
)

from . import digest as digest_mod
from . import gradient as gradient_mod
from . import metrics as metrics_mod
from . import policy as policy_mod
from . import recon as recon_mod
from . import verdicts as verdicts_mod
from . import visual as visual_mod
from .logging_utils import StageLogger
from .reexec import ReexecRunner, write_ledger_file
from .registry import TaskRegistry, code_verdict_key, task_key

if TYPE_CHECKING:
    from config import Config

_SYSTEM_PROMPT = (
    "You are a precise engineering assistant that follows output-format "
    "contracts exactly. When asked for strict JSON, reply with only the "
    "JSON object."
)

_SHORT_CIRCUIT_GRADIENT = (
    "workflow code failed to generate or execute; ensure code is properly "
    "formatted and that the workflow runs without crashing"
)
#: Sensible per-task cap on digests shown in the recon prompt.
_RECON_MAX_DIGESTS = 12

#: Reward cap when every decisive pair rested on unverifiable
#: self-reported metric evidence (P0.7 — HP gen 4's reward 1.0 crossed
#: the early-stop threshold on a boolean-vs-coordinate artifact).
WEAK_EVIDENCE_REWARD_CAP = 0.75
#: Deciding keys that name self-reported (workspace-claimed) metrics.
_SELF_REPORTED_KEY = re.compile(r"(log_metrics|json_metrics)", re.I)
#: Deciding keys that only measure deliverable FRESHNESS (N2, HP
#: 2026-10-08): a raw count difference of produced_this_run deliverables
#: ("3 vs 2") is not quality evidence — a workflow can win by writing
#: more files while the science is unchanged.
_FRESHNESS_KEY = re.compile(r"produced_this_run", re.I)
#: Deciding keys naming the goal contract — presence/absence of a
#: goal-named file is objective and MAY decide (N2 exception).
_CONTRACT_KEY = re.compile(r"contract_files", re.I)

#: Pair-record outcome spelling -> canonical reward vocabulary.
#: Records carry "won"/"lost"/"tie" (gradient fold + StopController
#: read those); ``win_rate_reward`` counts "win"/"loss"/"tie".
_OUTCOME_CANONICAL = {"won": "win", "lost": "loss", "tie": "tie"}


def _outcome_spellings(winner: str, uuid: str) -> tuple[str, str]:
    """(canonical, record) outcome spellings for a normalized *winner*.

    Single source of truth for the two outcome vocabularies — the main
    pair loop and the schema-refresh re-evaluation must agree on both.
    """
    if winner == "tie":
        return "tie", "tie"
    won = winner == uuid
    return ("win" if won else "loss"), ("won" if won else "lost")


def _primary_metric_keys(primary_metrics: list[Any] | None) -> list[str]:
    """Lower-cased ``key`` entries of the recon TASK CONFIG primary metrics."""
    return [
        str(m.get("key", "")).strip().lower()
        for m in primary_metrics or []
        if isinstance(m, dict) and m.get("key")
    ]


def _decisive_weak_evidence(
    verdict: dict[str, Any], primary_metrics: list[Any] | None
) -> bool:
    """True when one decisive verdict rests on weak evidence.

    Weak evidence classes:

    - **no deciding key at all** — an unverifiable verdict;
    - **pure freshness (N2)** — every deciding key only measures
      ``produced_this_run`` counts: more files is not better science,
      UNLESS the deciding key names a goal contract file
      (``contract_files``) or matches a recon primary-metric key, which
      is objective;
    - **self-reported metrics** (``log_metrics``/``json_metrics``) with
      confidence <= 0.55, or naming no recon primary-metric key.

    Structural/exec evidence (exit codes, schema, figures) is objective
    and never weak.

    Args:
        verdict: One pair verdict (``deciding``, ``confidence`` keys).
        primary_metrics: ``recon_cfg["primary_metrics"]`` entries.

    Returns:
        Whether this decisive verdict counts as weak evidence.
    """
    deciding = [str(k) for k in (verdict.get("deciding") or [])]
    if not deciding:
        return True
    if all(_FRESHNESS_KEY.search(k) for k in deciding):
        if any(_CONTRACT_KEY.search(k) for k in deciding):
            return False
        primary = _primary_metric_keys(primary_metrics)
        return not any(p in k.lower() for k in deciding for p in primary)
    if not any(_SELF_REPORTED_KEY.search(k) for k in deciding):
        return False
    try:
        conf = float(verdict.get("confidence"))
    except (TypeError, ValueError):
        conf = 0.0
    if conf <= 0.55:
        return True
    primary = _primary_metric_keys(primary_metrics)
    if not primary:
        return True  # no recon primary metrics: self-reported rungs unverifiable
    return not any(p in k.lower() for k in deciding for p in primary)


def find_run_snapshot(uuid: str) -> Path | None:
    """Session-scoped ``/tmp/mimosa_run_<session>_<uuid>`` snapshot.

    Matches by uuid SUFFIX (Mimosa uuids start with a date stamp that
    collides across campaigns); returns the most recently modified hit.
    """
    candidates = [
        p for p in Path("/tmp").glob(f"mimosa_run_*{uuid}*"[:1024]) if p.is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _sandbox_listing(ws: Path, fresh: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Complete file listing of the re-execution sandbox copy."""
    files: list[dict[str, Any]] = []
    for p in sorted(ws.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ws).as_posix()
        try:
            size = p.stat().st_size
        except OSError:
            size = None
        files.append(
            {
                "path": rel,
                "bytes": size,
                "fresh": bool((fresh or {}).get(rel)),
            }
        )
    return files


def _encode_figure(img: Path, log: StageLogger) -> dict[str, Any] | None:
    """One figure as an ``image_url`` block; None when capped/unreadable.

    The 6 MB cap filters UPLOADS only (P3.4) — a skipped oversized
    figure is still a produced deliverable.
    """
    try:
        if img.stat().st_size > visual_mod.MAX_IMAGE_BYTES:
            log.warn(
                f"figure {img.name} exceeds the 6 MB upload cap; "
                f"skipped from the vision call (still a produced "
                f"deliverable)"
            )
            return None
        b64 = base64.b64encode(img.read_bytes()).decode("ascii")
    except OSError as e:
        log.warn(f"figure {img.name} unreadable: {e}")
        return None
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{b64}",
            "detail": "high",
        },
    }


def _no_policy_verdict() -> dict[str, Any]:
    """Honest tie when no comparison policy exists for the task."""
    return {
        "winner": "tie",
        "channel": "no_policy",
        "feedback_winner": "",
        "feedback_looser": (
            "no comparison policy available for this task "
            "(generation failed policy build); honest tie"
        ),
        "_cached": False,
    }


def _pair_verdict_lines(pair_records: list[dict[str, Any]]) -> list[str]:
    """One ``- vs <uuid>:`` summary line per pair record."""
    return [
        f"- vs {p['prev_uuid']}: {p['outcome']} via {p['channel']} "
        f"(deciding={p.get('deciding')})"
        for p in pair_records
    ]


def _artifact_names(digest: dict[str, Any] | None) -> set[str]:
    """Artifact file names one side of a pair produced (from its digest)."""
    digest = digest or {}
    names = {
        str(dv.get("path")) for dv in digest.get("deliverables") or [] if dv.get("path")
    }
    names.update((digest.get("json_metrics") or {}).keys())
    names.update((digest.get("json_flags") or {}).keys())
    return names


def _name_summary(names: list[str], cap: int = 4) -> str:
    """Compact ``[n]: a, b, c +k`` rendering of a file-name list."""
    if not names:
        return "[0]: (none)"
    head = ", ".join(names[:cap])
    tail = f" +{len(names) - cap}" if len(names) > cap else ""
    return f"[{len(names)}]: {head}{tail}"


@dataclasses.dataclass
class _EvalState:
    """Mutable per-``evaluate()`` pipeline state threaded between stages.

    One instance per ``evaluate()`` call. The stage methods of
    :class:`PolicyBTVerifierEvaluator` fill these fields in pipeline
    order; anything still at its default signals a skipped stage.
    """

    uuid: str
    goal: str
    t_total: float = 0.0
    log: StageLogger | None = None
    registry: TaskRegistry | None = None
    tkey: str = ""
    sandbox_root: Path | None = None
    runner: ReexecRunner | None = None
    ledger: dict[str, Any] = dataclasses.field(default_factory=dict)
    current_ws: Path | None = None
    digest_now: dict[str, Any] = dataclasses.field(default_factory=dict)
    preds: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    excluded: list[str] = dataclasses.field(default_factory=list)
    recon_cfg: dict[str, Any] | None = None
    recon_cache: str = "hit"
    # N6/P2.3: a refreshed recon is HELD in memory and committed only
    # after the policy rewrite passes — committing first left every
    # later pair a no_policy tie (no policy under the new sha) and
    # re-billed the policy LLM every generation.
    pending_recon: tuple[dict[str, Any], str] | None = None
    recon_refreshed: bool = False
    current_recon_version: str | None = None
    primary_metrics: list[Any] = dataclasses.field(default_factory=list)
    effective_policy_sha: str = ""
    policy_src: str | None = None
    policy_meta: dict[str, Any] = dataclasses.field(default_factory=dict)
    smoke_digests: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    resmoke_failed: bool = False
    figure_task: bool = False
    visual: visual_mod.VisualChannel | None = None
    vis_cap: int = 0  # policy_bt_visual_max_pairs_per_gen; 0 = unlimited
    fresh_visual_pairs: int = 0
    pair_records: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    outcomes: list[str] = dataclasses.field(default_factory=list)
    count: dict[str, int] = dataclasses.field(
        default_factory=lambda: dict(
            n_cached=0,
            n_decisive=0,
            n_tie=0,
            n_disagree=0,
            n_visual=0,
            n_errors=0,
            n_weak=0,
            vis_fresh=0,
            vis_cached=0,
        )
    )
    reward_data: dict[str, Any] = dataclasses.field(default_factory=dict)
    reward: float = 0.5
    reward_weak_evidence: bool = False
    reward_capped: float = 0.0
    bt_data: dict[str, Any] | None = None
    raw_gradient: str = ""
    refinement: Any = None
    gradient_text: str = ""
    rstats: dict[str, Any] = dataclasses.field(default_factory=dict)
    scores: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _PairCtx:
    """Bundle of ``_pair_verdict``'s ladder arguments (plan §5 step 6).

    ``_pair_verdict``'s signature is pinned by the test suite; the
    ladder stages below it take this bundle instead.
    """

    uuid: str
    goal: str
    digest_now: dict[str, Any]
    current_ws: Path
    runner: ReexecRunner
    pred: dict[str, Any]
    policy_src: str | None
    policy_sha: str
    visual: Any
    registry: TaskRegistry
    log: StageLogger


class PolicyBTVerifierEvaluator(BaseEvaluator):
    """The E48 verifier as a selectable Mimosa verification channel.

    Selected via ``config.verifier_kind = "policy_bt"``. The knobs are
    the ``policy_bt_*`` config entries (plan §2); all defaults come from
    the E48 measurements.
    """

    def __init__(
        self,
        config: Config,
        workspace_dir: str | Path | None = None,
    ) -> None:
        """Initialise knobs, LLM configs and the verifier scratch root.

        Args:
            config: Mimosa Config object (paths, judge model, pricing).
            workspace_dir: Override for the agents' workspace root;
                defaults to ``config.workspace_dir``.
        """
        super().__init__(config)
        self.workspace_dir = Path(
            workspace_dir
            if workspace_dir is not None
            else getattr(config, "workspace_dir", ".")
        )
        self._init_knobs(config)
        # N9: registry generations carry the tag of the campaign session
        # that wrote them, so a NEW campaign's generation 1 comparing
        # against a PREVIOUS campaign's generations is a loud WARN, not
        # a silent contamination (the engine's StopController champion
        # is session-scoped by construction).
        self._session_tag = f"s{int(time.time())}-{uuid4().hex[:8]}"
        self._runner_temp_root = Path(
            getattr(config, "temp_dir", None) or self.workflow_dir / "_verifier_tmp"
        )
        # E19 measured convention: deterministic generation at temperature 0.
        self.llm_config.temperature = 0.0
        self._vision_llm_config = self._build_vision_config(config)
        self.logger.info(
            f"PolicyBTVerifierEvaluator initialized (workspace="
            f"{self.workspace_dir}, reward={self.reward_mode}, reexec_timeout="
            f"{self.reexec_timeout_s}s, visual={self.visual_channel}, "
            f"refiner={self.refiner})"
        )

    def _init_knobs(self, config: Config) -> None:
        """Load every ``policy_bt_*`` knob (E48-measured defaults)."""
        self.reward_mode = str(getattr(config, "policy_bt_reward", "win_rate")).lower()
        if self.reward_mode not in ("win_rate", "bradley_terry"):
            self.reward_mode = "win_rate"
        self.scorer_timeout_s = max(
            5, int(getattr(config, "policy_bt_scorer_timeout_s", 60))
        )
        self.visual_channel = bool(getattr(config, "policy_bt_visual_channel", True))
        self.refiner = bool(getattr(config, "policy_bt_refiner", True))
        self.repair_rounds = max(0, int(getattr(config, "policy_bt_repair_rounds", 3)))
        self.max_pairs = max(0, int(getattr(config, "policy_bt_max_pairs", 0)))
        self.reexec_timeout_s = max(
            5, int(getattr(config, "policy_bt_reexec_timeout_s", 900))
        )
        self.venv_reuse = bool(getattr(config, "policy_bt_venv_reuse", True))
        self.visual_tiebreak_code = bool(
            getattr(config, "policy_bt_visual_tiebreak_code", False)
        )
        self.visual_max_pairs_per_gen = max(
            0, int(getattr(config, "policy_bt_visual_max_pairs_per_gen", 0))
        )
        self.verbose = bool(getattr(config, "policy_bt_verbose", False))
        self._init_refresh_knobs(config)

    def _init_refresh_knobs(self, config: Config) -> None:
        """Schema-error-triggered refresh knobs (no milestone schedule).

        When the policy raises SchemaError on a pair (unknown keys or
        digest paths), the recon is refreshed with ALL accumulated
        digests and the policy rewritten; ``recon_refresh_gap``
        generations minimum between refreshes prevents thrashing.
        """
        self.recon_refresh_gap = max(
            1, int(getattr(config, "policy_bt_recon_refresh_gap", 2))
        )
        self._gens_since_refresh = 0

    # ------------------------------------------------------------------
    # LLM plumbing (judge + vision, cached provider, spend-tracked)
    # ------------------------------------------------------------------

    def _build_vision_config(self, config: Config) -> LLMConfig | None:
        """Vision-model LLMConfig (kimi-k3 transport, temperature 0)."""
        vision_model = getattr(config, "vision_judge_model", None)
        if not vision_model:
            self.logger.info(
                "No vision_judge_model configured; the kimi pair channel "
                "will soft-fail to the code policy"
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
                "temperature": 0.0,
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
                "key": "",
            }
        )

    @staticmethod
    def _config_hash(cfg: dict[str, Any]) -> str:
        """Stable hash of a recon config's content (for versioning)."""
        import hashlib as _hl

        return _hl.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def _record_spend(
        self, agent_name: str, memory_path: Path, log: StageLogger
    ) -> None:
        """Best-effort token/cost accounting from the provider's dump."""
        try:
            data = json.loads(
                (memory_path / f"{agent_name}.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return
        usage = data.get("usage") or {}
        model = str(data.get("model") or "")
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        pricing = self.model_pricing.get(model) or self.model_pricing.get(
            model.replace("openrouter/", "", 1)
        )
        cost = (
            pt / 1e6 * float(pricing["input"]) + ct / 1e6 * float(pricing["output"])
            if pricing
            else None
        )
        log.add_cost(cost)

    @staticmethod
    def _derived_llm_config(
        cfg: Any, max_tokens: int, reasoning_effort: str | None = None
    ) -> LLMConfig:
        """Temperature-0 call config derived from *cfg* with a token cap.

        ``reasoning_effort`` overrides the source config's effort (the
        code-gen judge calls pin ``"low"`` — token space beats
        reasoning depth).
        """
        return LLMConfig().from_dict(
            {
                "model": cfg.model,
                "provider": cfg.provider,
                "temperature": 0.0,
                "reasoning_effort": reasoning_effort or cfg.reasoning_effort,
                "max_tokens": max(0, int(max_tokens)),
                "openrouter_provider": cfg.openrouter_provider,
                "openrouter_quantizations": cfg.openrouter_quantizations,
                "key": cfg.key,
            }
        )

    def _llm_json(
        self,
        uuid: str,
        role: str,
        prompt: str,
        max_tokens: int,
        log: StageLogger,
    ) -> Any:
        """One judge call parsed as JSON (or the raw text on parse fail)."""
        agent = f"policy_bt_{role}"
        memory_path = Path(self.memory_dir) / uuid
        memory_path.mkdir(parents=True, exist_ok=True)
        cfg = self._derived_llm_config(
            self.llm_config, max_tokens, reasoning_effort="low"
        )
        provider = LLMProvider(
            agent_name=agent,
            memory_path=memory_path,
            system_msg=_SYSTEM_PROMPT,
            config=cfg,
        )
        raw = provider(prompt)
        self._record_spend(agent, memory_path, log)
        payload = extract_json_payload(raw or "")
        if not payload:
            return raw
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return raw

    @staticmethod
    def _vision_content(
        prompt: str,
        figs_a: list[Path],
        figs_b: list[Path],
        log: StageLogger,
    ) -> list[Any]:
        """Multimodal content: prompt + per-candidate labeled figures.

        P3.2: each side's figures are announced by a text label before
        their images ("Candidate A figures:" / "Candidate B figures:")
        so the judge can attribute what it sees.
        """
        content: list[Any] = [{"type": "text", "text": prompt}]
        for label, figs in (
            ("Candidate A figures:", figs_a),
            ("Candidate B figures:", figs_b),
        ):
            content.append({"type": "text", "text": label})
            for img in figs:
                item = _encode_figure(img, log)
                if item is not None:
                    content.append(item)
        return content

    def _vision_json(
        self,
        uuid: str,
        prompt: str,
        figs_a: list[Path],
        figs_b: list[Path],
        max_tokens: int,
        log: StageLogger,
        role: str = "",
    ) -> str:
        """One kimi vision round-trip (per-candidate labeled images).

        P3.1: *role* (attempt + max_tokens) is appended to the provider
        agent name — a retry can never be served the cached failed
        attempt.
        """
        if self._vision_llm_config is None:
            raise RuntimeError(
                "no vision_judge_model configured; cannot run the kimi pair channel"
            )
        content = self._vision_content(prompt, figs_a, figs_b, log)
        agent = "policy_bt_kimi_pair" + (f"_{role}" if role else "")
        memory_path = Path(self.memory_dir) / uuid
        memory_path.mkdir(parents=True, exist_ok=True)
        cfg = self._derived_llm_config(self._vision_llm_config, max_tokens)
        provider = LLMProvider(
            agent_name=agent,
            memory_path=memory_path,
            system_msg=_SYSTEM_PROMPT,
            config=cfg,
        )
        raw = provider(content)
        self._record_spend(agent, memory_path, log)
        return raw

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(self, uuid: str) -> dict[str, Any]:
        """Run the policy_bt pipeline; persists under ``evaluation.verifier``.

        Args:
            uuid: Workflow identifier to evaluate.

        Returns:
            Dict with the reward, pair stats and both textual-gradient
            keys (``abstracted_textual_gradient`` and the legacy-typo
            ``abstractec_textual_gradient``).

        Raises:
            EvaluatorError: When ``uuid`` is not a non-empty string.
        """
        if not uuid or not isinstance(uuid, str):
            raise EvaluatorError("Invalid uuid: must be a non-empty string")
        wf_info = self._load_workflow_data(uuid)
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
        st = _EvalState(uuid=uuid, goal=goal, t_total=time.time())
        self._open_task(st)
        self._reexecute(st)  # stage 5: fresh re-execution
        self._digest_current(st)  # digest of THIS generation
        self._resolve_predecessors(st)  # tiered artifact resolution (§5b)
        self._log_digest_summaries(st)
        self._ensure_recon(st)  # stage 3: recon TASK CONFIG
        self._resolve_metric_ledger(st)  # primary-metric ledger (§4.1)
        self._ensure_policy(st)  # stage 4: write -> screen -> smoke -> repair
        self._select_channels(st)  # E26 figure-dominance routing
        self._run_pair_loop(st)  # stage 6: verdicts vs every predecessor
        self._schema_error_refresh(st)  # post-loop schema-error recovery
        self._log_pairs_stage(st)
        self._compute_reward(st)  # stage 7: arrival-moment reward
        self._persist_generation(st)  # N7/P2.1: persist right after reward
        self._build_gradient(st)  # stage 8: fold + optional refiner
        self._build_scores(st)
        return self._finalize(st)

    # ------------------------------------------------------------------
    # Pipeline stages (in evaluate() call order)
    # ------------------------------------------------------------------

    def _open_task(self, st: _EvalState) -> None:
        """Per-evaluation stage logger, registry and sandbox root."""
        st.log = StageLogger(
            st.uuid,
            sink=self.workflow_dir / st.uuid / "verifier_policy_bt.log",
            verbose=self.verbose,
        )
        st.registry = TaskRegistry.load(self._runner_temp_root, st.goal, self.logger)
        st.tkey = task_key(st.goal)
        st.sandbox_root = self._runner_temp_root / "policy_bt" / st.tkey

    def _reexecute(self, st: _EvalState) -> None:
        """Stage 5: fresh re-execution of THIS generation."""
        st.runner = ReexecRunner(
            st.sandbox_root,
            timeout_s=self.reexec_timeout_s,
            venv_reuse=self.venv_reuse,
            log=st.log,
        )
        st.ledger = st.runner.run(self.workspace_dir, st.uuid)
        write_ledger_file(
            st.ledger, st.sandbox_root / f"policy_bt_ledger_{st.uuid}.json"
        )
        st.current_ws = st.runner.run_copy_path(st.uuid)
        self._log_reexec(st)

    def _log_reexec(self, st: _EvalState) -> None:
        """REEXEC stage log + the source-mutation guard (P2.8)."""
        ledger = st.ledger
        listing = _sandbox_listing(st.current_ws, ledger.get("fresh_artifacts"))
        if not st.current_ws.exists():  # re-execution never produced a copy
            listing = [
                {"path": "<live workspace fallback>", "bytes": None, "fresh": False}
            ]
        st.log.payload(
            "REEXEC",
            (
                f"sandbox=copy({ledger.get('n_files_copied', 0)} files) "
                f"venv={'hit' if not ledger.get('venv_created') else 'created'} "
                f"pkgs=+{ledger.get('packages_installed', 0)} "
                f"src={ledger.get('main_script_source', 'n/a')} "
                f"exit={ledger.get('exit_code')} "
                f"wall={ledger.get('wall_time_s')}s "
                f"fresh={len(ledger.get('fresh_artifacts') or {})}"
            ),
            {
                "sandbox": str(st.current_ws),
                "n_files_listed": len(listing),
                "files": listing,
                "commands": ledger.get("commands"),
                "install_source": ledger.get("install_source"),
            },
            error=ledger.get("launch_error"),
        )
        # P2.8: a mutated source workspace invalidates the re-execution
        # guard itself — WARN loudly; the crash rung treats this ledger
        # as a distinct crash class (manifest_mutated).
        if ledger.get("source_manifest_intact") is False:
            st.log.warn(
                "source workspace was MUTATED during re-execution "
                "(source_manifest_intact=False) — this generation's "
                "ledger is untrustworthy; crash rung class "
                "manifest_mutated applies"
            )

    def _digest_current(self, st: _EvalState) -> None:
        """Digest of this generation, from the re-execution copy."""
        if not st.current_ws.exists():  # re-execution never produced a copy
            st.current_ws = self.workspace_dir
        st.digest_now = digest_mod.build_digest(
            st.uuid, st.current_ws, st.ledger, st.goal
        )
        st.registry.record_generation(
            st.uuid,
            st.digest_now,
            st.ledger,
            reward=None,
            session=self._session_tag,
        )
        dsum = digest_mod.digest_summary(st.digest_now)
        st.log.stage(
            "DIGEST",
            (
                f"tables={dsum['tables']} arrays={dsum['arrays']} "
                f"figures={dsum['figures']} metrics={dsum['metrics']} "
                f"deliverables={dsum['deliverables']} "
                f"id_order={dsum['id_order']}"
            ),
        )

    def _resolve_predecessors(self, st: _EvalState) -> None:
        """Tiered artifact resolution for every predecessor (plan §5b)."""
        for g in st.registry.previous_generations(st.uuid):
            if g.get("digest"):
                st.preds.append(g)
                continue
            self._bootstrap_or_exclude(st, g)
        self._cap_predecessors(st)
        self._warn_foreign_sessions(st)

    def _bootstrap_or_exclude(self, st: _EvalState, g: dict[str, Any]) -> None:
        """Predecessor without a registered digest: snapshot or exclude."""
        snap = find_run_snapshot(g["uuid"])
        if snap is None:
            st.excluded.append(g["uuid"])
            st.log.warn(
                f"predecessor {g['uuid'][-8:]} is unscoreable: no "
                f"registered digest, no /tmp snapshot, not the live "
                f"workspace — EXCLUDED from pairwise comparison "
                f"(never fabricated)"
            )
            return
        boot_ledger = {
            "bootstrap": f"snapshot:{snap.name}",
            "exit_code": None,
        }
        boot_digest = digest_mod.build_digest(g["uuid"], snap, boot_ledger, st.goal)
        for g2 in st.registry.generations:
            if g2.get("uuid") == g["uuid"]:
                g2["digest"] = boot_digest
                g2["ledger"] = boot_ledger
        st.preds.append(
            {
                "uuid": g["uuid"],
                "arrival": g.get("arrival"),
                "session": g.get("session"),
                "digest": boot_digest,
                "ledger": boot_ledger,
            }
        )
        st.log.warn(
            f"predecessor {g['uuid'][-8:]} had no registered digest; "
            f"rebuilt from /tmp snapshot {snap.name} (no fresh "
            f"execution facts — exec block is null)"
        )

    def _cap_predecessors(self, st: _EvalState) -> None:
        """``policy_bt_max_pairs``: keep only the most recent predecessors."""
        if not (self.max_pairs and len(st.preds) > self.max_pairs):
            return
        dropped = st.preds[: len(st.preds) - self.max_pairs]
        st.preds = st.preds[len(st.preds) - self.max_pairs :]
        st.log.warn(
            f"policy_bt_max_pairs={self.max_pairs}: comparing only the "
            f"{len(st.preds)} most recent predecessors "
            f"({len(dropped)} older dropped)"
        )

    def _warn_foreign_sessions(self, st: _EvalState) -> None:
        """N9: cross-session contamination is loud, not silent.

        The registry persists per goal across campaigns; without this a
        new campaign's generation 1 competes against the previous
        campaign's generations (champion selection stays session-scoped
        in the engine either way).
        """
        foreign = [p["uuid"] for p in st.preds if p.get("session") != self._session_tag]
        if not foreign:
            return
        st.log.warn(
            f"{len(foreign)} predecessor(s) were recorded by ANOTHER "
            f"campaign session (e.g. {foreign[0][-8:]}); head-to-head "
            f"outcomes span sessions — clean the registry or run "
            f"cleanup.sh for a fresh campaign"
        )

    def _log_digest_summaries(self, st: _EvalState) -> None:
        """DIGESTS payload: per-generation summaries of all compared."""

        def _digest_entry(u: str, dg: dict[str, Any]) -> dict[str, Any]:
            s = digest_mod.digest_summary(dg)
            jm = dg.get("json_metrics") or {}
            return {
                "uuid": u,
                **s,
                "json_metric_files": {k: len(v) for k, v in jm.items()},
                "deliverable_paths": [
                    dv.get("path") for dv in dg.get("deliverables") or []
                ],
                "exec": {
                    k: (dg.get("exec") or {}).get(k)
                    for k in ("exit_code", "wall_time_s", "timed_out")
                },
            }

        compared = [_digest_entry(st.uuid, st.digest_now)] + [
            _digest_entry(p["uuid"], p["digest"]) for p in st.preds
        ]
        st.log.payload(
            "DIGESTS",
            (
                f"digest summaries logged for {len(compared)} generations "
                f"(current + {len(st.preds)} preds, "
                f"{len(st.excluded)} excluded)"
            ),
            compared,
        )

    def _ensure_recon(self, st: _EvalState) -> None:
        """Stage 3: recon TASK CONFIG (first call + held refreshes).

        The schema-error-triggered refresh happens AFTER the pair loop
        (``_schema_error_refresh``) — the policy signals SchemaError on
        unknown keys, then the recon is rebuilt.
        """
        st.recon_cfg = st.registry.get_recon(recon_mod.RECON_PROMPT_VERSION)
        if st.recon_cfg is None:
            # First generation: build from this digest alone.
            prompt = recon_mod.build_prompt(st.goal, [st.digest_now])
            parsed = self._llm_json(st.uuid, "recon", prompt, 4000, st.log)
            st.recon_cfg = recon_mod.ensure_shape(parsed)
            if st.recon_cfg:
                st.registry.set_recon(
                    st.recon_cfg,
                    recon_mod.RECON_PROMPT_VERSION,
                    recon_version=self._config_hash(st.recon_cfg),
                )
            else:
                st.log.warn("recon call unparseable; continuing with empty config")
            st.recon_cache = "miss"
        st.log.stage("RECON", f"cache={st.recon_cache} (task goal-hash {st.tkey})")
        st.log.payload(
            "RECON",
            f"task config logged (goal-hash {st.tkey}, cache={st.recon_cache})",
            st.recon_cfg or {},
        )

    def _resolve_metric_ledger(self, st: _EvalState) -> None:
        """ONE deterministic primary-metric resolution per generation.

        The ledger is stored on the digest itself (review 2026-10-09
        §4.1): the policy compares primary_values, the gradient fold
        renders METRICS from them. Ledgers carry the recon version they
        were resolved under; a refresh that adds aliases re-resolves
        every predecessor (deterministic, free, no LLM).
        """
        st.current_recon_version = (
            st.pending_recon[1] if st.pending_recon else st.registry.get_recon_version()
        )
        st.primary_metrics = (st.recon_cfg or {}).get("primary_metrics") or []
        llm_extract = self._metric_llm_extract(st)
        st.digest_now["primary_values"] = metrics_mod.resolve_primary_values(
            st.digest_now, st.primary_metrics, llm_extract=llm_extract
        )
        st.digest_now["primary_values_version"] = st.current_recon_version
        for p in st.preds:
            self._re_resolve_pred(
                p, st.primary_metrics, llm_extract, st.current_recon_version
            )
        self._log_metric_ledger(st)

    def _metric_llm_extract(self, st: _EvalState) -> Any:
        """LLM text extractor wired to ``_llm_json``.

        LLM text extraction replaces ALL regex scraping (operator
        directive 2026-10-10: extraction from raw output is an LLM
        call).
        """

        def _extract_llm_json(prompt, mt, role):
            return self._llm_json(st.uuid, role, prompt, mt, st.log)

        def llm_extract(tl, mk):
            return metrics_mod.extract_text_metrics_llm(
                tl, mk, llm_json=_extract_llm_json
            )

        return llm_extract

    def _re_resolve_pred(
        self,
        p: dict[str, Any],
        primary_metrics: list[Any],
        llm_extract: Any,
        version: str | None,
    ) -> None:
        """Predecessor ledger re-resolves under a new recon version."""
        dg = p.get("digest")
        if not isinstance(dg, dict) or not dg:
            return
        if dg.get("primary_values_version") != version:
            dg["primary_values"] = metrics_mod.resolve_primary_values(
                dg, primary_metrics, llm_extract=llm_extract
            )
            dg["primary_values_version"] = version

    def _log_metric_ledger(self, st: _EvalState) -> None:
        """METRICS stage: compact ledger summary of the current generation."""
        n_res = n_amb = n_miss = 0
        parts = []
        for _mid, _r in st.digest_now["primary_values"].items():
            if _r["status"] == "resolved":
                n_res += 1
                parts.append(f"{_mid}={_r['value']:g} [{_r['source']}]")
            elif _r["status"] == "ambiguous":
                n_amb += 1
            else:
                n_miss += 1
        st.log.stage(
            "METRICS",
            (
                f"resolved={n_res} ambiguous={n_amb} missing={n_miss} "
                f"({' '.join(parts) if parts else 'none'})"
            ),
        )
        st.log.payload(
            "METRICS",
            "primary metric ledger logged (current generation)",
            st.digest_now["primary_values"],
        )

    def _ensure_policy(self, st: _EvalState) -> None:
        """Stage 4: obtain the pairwise policy (cache -> resmoke -> write)."""
        st.effective_policy_sha = self._policy_sha(st.current_recon_version)
        policy_rec = st.registry.get_policy(st.effective_policy_sha)
        st.policy_src, st.policy_meta = None, {}
        # P0.3/P0.4: the smoke covers EVERY distinct digest shape of the
        # task (current + all predecessors, capped), and a CACHE HIT is
        # re-validated against it every generation — a gen-1 policy that
        # only ever self-compared must re-smoke when gen-2's shape
        # arrives.
        st.smoke_digests = [st.digest_now] + [
            g["digest"]
            for g in st.preds
            if isinstance(g.get("digest"), dict) and g["digest"]
        ]
        st.smoke_digests = st.smoke_digests[: policy_mod.SMOKE_MAX_DIGESTS]
        if policy_rec is not None and self._try_cached_policy(st, policy_rec):
            return
        action = "rewrite" if (st.recon_refreshed or st.resmoke_failed) else "fresh"
        self._write_policy(st, action)

    @staticmethod
    def _policy_sha(recon_version: str | None) -> str:
        """Composite cache sha: prompt template hash + recon config hash.

        A config change from a refresh naturally invalidates the policy
        and verdict caches without touching the decision ladder.
        """
        return policy_mod.POLICY_SHA + (
            "_" + recon_version[:8] if recon_version else ""
        )

    def _try_cached_policy(self, st: _EvalState, policy_rec: dict[str, Any]) -> bool:
        """Cache-hit path; False when the re-smoke fails (P0.4 rewrite)."""
        st.policy_src = policy_rec.get("source")
        st.policy_meta = policy_rec.get("meta") or {}
        ok, err = policy_mod.smoke_test(
            st.policy_src or "",
            st.smoke_digests,
            timeout=policy_mod.SMOKE_TIMEOUT_S,
            max_pairs=policy_mod.SMOKE_MAX_PAIRS_CACHE_HIT,
        )
        if ok:
            st.log.stage(
                "POLICY",
                f"cache=hit sha=..{st.effective_policy_sha[-4:]} resmoke=pass",
            )
            self._log_policy_payload(st, "cached", st.policy_meta)
            return True
        st.resmoke_failed = True
        st.policy_src = None
        st.log.warn(
            f"cached policy failed the re-smoke on a new digest "
            f"shape ({str(err)[:200]}); rewriting the policy"
        )
        return False

    def _write_policy(self, st: _EvalState, action: str) -> None:
        """Fresh write or rewrite — same V3_PROMPT ladder, new knowledge."""

        def llm_json(prompt, mt, role):
            return self._llm_json(st.uuid, role, prompt, mt, st.log)

        st.policy_src, st.policy_meta = policy_mod.ensure_policy(
            st.goal,
            st.recon_cfg or {},
            st.smoke_digests,
            llm_json,
            repair_rounds=self.repair_rounds,
            smoke_timeout_s=policy_mod.SMOKE_TIMEOUT_S,
            log=st.log,
        )
        if st.policy_src is not None:
            self._commit_new_policy(st, action)
        else:
            self._recover_rejected_refresh(st, action)

    def _commit_new_policy(self, st: _EvalState, action: str) -> None:
        """Commit the rewritten policy (and its pending recon, N6) + logs."""
        if st.pending_recon is not None:
            # N6: the rewrite passed — NOW the refreshed recon is
            # committed (the policy record below is keyed to its
            # version, so the two commit together).
            st.registry.set_recon(
                st.pending_recon[0],
                recon_mod.RECON_PROMPT_VERSION,
                recon_version=st.pending_recon[1],
            )
        st.registry.set_policy(
            {
                "sha": st.effective_policy_sha,
                "source": st.policy_src,
                "meta": {
                    **st.policy_meta,
                    "recon_version": st.current_recon_version,
                },
            }
        )
        pm = st.policy_meta
        st.log.stage(
            "POLICY",
            (
                f"cache={action} sha=..{st.effective_policy_sha[-4:]} "
                f"(attempts={pm.get('attempts')} "
                f"repairs={pm.get('repairs')} "
                f"escalations={pm.get('escalations')} "
                f"smoke={pm.get('smoke')})"
            ),
        )
        self._log_policy_payload(
            st,
            action,
            {**st.policy_meta, "recon_version": st.current_recon_version},
        )
        st.log.preview(
            f"policy source ({action} write, def compare ladder)",
            st.policy_src or "",
            max_lines=14,
            sha=st.effective_policy_sha,
        )

    def _log_policy_payload(
        self, st: _EvalState, label: str, meta: dict[str, Any]
    ) -> None:
        """POLICY payload: the policy source + meta (for the log sink)."""
        st.log.payload(
            "POLICY",
            (
                f"{label} policy source logged "
                f"({len(st.policy_src or '')} chars, "
                f"sha=..{st.effective_policy_sha[-4:]})"
            ),
            {
                "sha": st.effective_policy_sha,
                "source": st.policy_src or "",
                "meta": meta,
            },
        )

    def _recover_rejected_refresh(self, st: _EvalState, action: str) -> None:
        """N6: a rejected rewrite keeps the OLD recon + OLD policy."""
        if st.pending_recon is not None:
            # Rewrite rejected — keep the OLD recon + OLD policy (when
            # one exists) instead of stranding the task with no policy
            # under the new sha.
            st.pending_recon = None
            st.recon_refreshed = False
            st.recon_cfg = st.registry.get_recon(recon_mod.RECON_PROMPT_VERSION)
            self._restore_old_policy(st)
        st.log.warn(
            f"policy {action} FAILED after repairs "
            f"({st.policy_meta.get('last_error')}); pairs degrade to "
            f"the deterministic crash rung + ties"
        )
        st.log.stage("POLICY", f"cache={action} FAILED", smoke="fail")

    def _restore_old_policy(self, st: _EvalState) -> None:
        """Fall back to the last committed policy record, if any."""
        old_version = st.registry.get_recon_version()
        old_sha = self._policy_sha(old_version)
        old_rec = st.registry.get_policy(old_sha)
        if old_rec is None:
            st.log.stage("POLICY", "cache=refresh-rejected (no old policy either)")
            return
        st.policy_src = old_rec.get("source")
        st.policy_meta = old_rec.get("meta") or {}
        st.effective_policy_sha = old_sha
        st.log.stage(
            "POLICY",
            (
                f"cache=refresh-rejected sha=..{old_sha[-4:]} "
                f"(rewrite failed; old recon + old policy kept)"
            ),
        )

    def _select_channels(self, st: _EvalState) -> None:
        """Channel selection: figure-dominant -> kimi (E26 rule)."""
        pooled: list[dict[str, Any]] = []
        for g in st.preds:
            pooled.extend(g["digest"].get("deliverables") or [])
        pooled.extend(st.digest_now.get("deliverables") or [])
        st.figure_task = digest_mod.is_figure_dominant({"deliverables": pooled})
        st.visual = (
            self._make_visual_channel(st.uuid, st.goal, st.registry, st.log)
            if (st.figure_task and self.visual_channel)
            else None
        )
        if st.figure_task and self.visual_channel and st.visual is None:
            st.log.warn(
                "figure-dominant task but no vision_judge_model configured; "
                "the kimi pair channel is disabled — the code policy decides"
            )

    def _run_pair_loop(self, st: _EvalState) -> None:
        """Stage 6: pair verdicts vs every predecessor."""
        st.vis_cap = self.visual_max_pairs_per_gen
        for g in st.preds:
            allow_visual = st.visual is not None and (
                st.vis_cap == 0 or st.fresh_visual_pairs < st.vis_cap
            )
            verdict = self._pair_verdict(
                st.uuid,
                st.goal,
                st.digest_now,
                st.ledger,
                st.current_ws,
                st.runner,
                g,
                st.policy_src,
                st.effective_policy_sha,
                st.visual,
                allow_visual,
                st.registry,
                st.log,
            )
            self._log_artifact_diff(st, g, verdict)
            self._tally_pair(st, g["uuid"], verdict)
        self._log_visual_stage(st)

    @staticmethod
    def _log_artifact_diff(
        st: _EvalState, pred: dict[str, Any], verdict: dict[str, Any]
    ) -> None:
        """FILES line: artifact-name diff between the two pair sides.

        Fresh pairs only — a cached verdict re-logs nothing. Full name
        lists (unabridged) live in the sink record.
        """
        if verdict.get("_cached"):
            return
        other = pred["uuid"]
        mine = _artifact_names(st.digest_now)
        theirs = _artifact_names(pred.get("digest"))
        a_only, b_only = sorted(mine - theirs), sorted(theirs - mine)
        common = sorted(mine & theirs)
        a_txt = st.log.paint(_name_summary(a_only), GREEN)
        st.log.payload(
            "FILES",
            f"vs {other[-8:]} A-only{a_txt} · "
            f"B-only{_name_summary(b_only)} · common={len(common)}",
            {
                "uuid_a": st.uuid,
                "uuid_b": other,
                "a_only": a_only,
                "b_only": b_only,
                "common": common,
            },
        )

    def _tally_pair(self, st: _EvalState, other: str, verdict: dict[str, Any]) -> None:
        """Fold one pair verdict into counters, records and PAIR logs."""
        was_cached = bool(verdict.get("_cached"))
        self._pop_pair_flags(st, verdict)
        winner = verdict.get("winner")
        if winner not in ("tie", st.uuid, other):
            winner = "tie"
            verdict["winner"] = "tie"
        canonical, rec_outcome, feedback = self._pair_outcome(st, verdict, winner)
        if verdict.get("error"):
            st.count["n_errors"] += 1
        elif canonical != "tie" and _decisive_weak_evidence(
            verdict, st.primary_metrics
        ):
            st.count["n_weak"] += 1
        if verdict.get("antisymmetry") == "disagreement":
            st.count["n_disagree"] += 1
        record = {
            "prev_uuid": other,
            "outcome": rec_outcome,
            "winner": winner,
            "channel": verdict.get("channel"),
            "deciding": verdict.get("deciding"),
            "confidence": verdict.get("confidence"),
            "mirrored": verdict.get("mirrored"),
            "feedback": feedback,
            "detail": {
                "feedback_winner": verdict.get("feedback_winner") or "",
                "feedback_looser": verdict.get("feedback_looser") or "",
            },
        }
        st.pair_records.append(record)
        self._log_pair(st, other, verdict, record, canonical, was_cached)

    @staticmethod
    def _pop_pair_flags(st: _EvalState, verdict: dict[str, Any]) -> None:
        """Pop the internal tally flags off a fresh/cached verdict."""
        if verdict.pop("_fresh_visual_pair", False):
            st.fresh_visual_pairs += 1
            st.count["vis_fresh"] += 1
        if verdict.pop("_cached_visual", False):
            st.count["vis_cached"] += 1
        if verdict.pop("_visual", False):
            st.count["n_visual"] += 1
        if verdict.pop("_cached", False):
            st.count["n_cached"] += 1

    @staticmethod
    def _pair_outcome(
        st: _EvalState, verdict: dict[str, Any], winner: str
    ) -> tuple[str, str, str]:
        """Canonical outcome for one pair + the losing side's feedback.

        N1 (2026-10-08 audit): a LOSS is decisive too — the old loss
        branch set neither counter nor feedback (UnboundLocalError out
        of evaluate(), stale feedback reuse, undercounted n_decisive).
        """
        canonical, rec_outcome = _outcome_spellings(winner, st.uuid)
        st.outcomes.append(canonical)
        if canonical == "tie":
            st.count["n_tie"] += 1
        else:
            st.count["n_decisive"] += 1
        won = winner == st.uuid
        feedback = (
            verdict.get("feedback_winner") if won else verdict.get("feedback_looser")
        ) or ""
        return canonical, rec_outcome, feedback

    def _log_pair(
        self,
        st: _EvalState,
        other: str,
        verdict: dict[str, Any],
        record: dict[str, Any],
        canonical: str,
        was_cached: bool,
    ) -> None:
        """Per-pair detail + PAIR payload lines (§6b)."""
        self._log_pair_detail(st, other, verdict, record)
        self._log_pair_payload(st, other, verdict, record, canonical, was_cached)

    @staticmethod
    def _log_pair_detail(
        st: _EvalState,
        other: str,
        verdict: dict[str, Any],
        record: dict[str, Any],
    ) -> None:
        """One-line detail for the console/verbose sink."""
        winner = record["winner"]
        feedback = record["feedback"]
        st.log.detail(
            f"(gen {st.uuid[-8:]} vs {other[-8:]}) "
            f"winner={winner[-8:] if winner != 'tie' else 'tie'} "
            f"rung={verdict.get('channel')} "
            f"deciding={verdict.get('deciding')} "
            f"feedback-head={str(feedback)[:80]}"
        )

    @staticmethod
    def _log_pair_payload(
        st: _EvalState,
        other: str,
        verdict: dict[str, Any],
        record: dict[str, Any],
        canonical: str,
        was_cached: bool,
    ) -> None:
        """Full PAIR JSON payload for the log sink."""
        winner = record["winner"]
        st.log.payload(
            "PAIR",
            (
                f"vs {other} winner="
                f"{winner if winner == 'tie' else winner[-8:]} "
                f"channel={verdict.get('channel')} "
                f"deciding={verdict.get('deciding')}"
            ),
            {
                "uuid_a": st.uuid,
                "uuid_b": other,
                "winner": winner,
                "outcome": canonical,
                "channel": verdict.get("channel"),
                "deciding": verdict.get("deciding"),
                "rungs": verdict.get("rungs"),
                "confidence": verdict.get("confidence"),
                "mirrored": verdict.get("mirrored"),
                "antisymmetry": verdict.get("antisymmetry"),
                "cached": was_cached,
                "feedback_winner": (verdict.get("feedback_winner") or ""),
                "feedback_looser": (verdict.get("feedback_looser") or ""),
            },
        )

    def _log_visual_stage(self, st: _EvalState) -> None:
        """VISUAL stage line: kimi pair stats (figure-dominant tasks)."""
        if st.visual is None:
            return
        n_dis = st.count["n_disagree"]
        st.log.stage(
            "VISUAL",
            (
                f"kimi pairs={st.count['n_visual']} "
                f"fresh={st.count['vis_fresh']} "
                f"cached={st.count['vis_cached']} decisive="
                f"{sum(1 for p in st.pair_records if p['outcome'] != 'tie' and p['channel'] == 'kimi_pair')} "
                f"mirror={'ok' if n_dis == 0 else f'{n_dis} disagreements'}"
            ),
        )

    def _schema_error_refresh(self, st: _EvalState) -> None:
        """Schema-error-triggered recon refresh + policy rewrite.

        If the policy signalled SchemaError on any pair, refresh the
        recon with ALL digests (including the schema-breaking one),
        rewrite the policy, and re-evaluate the failed pairs. No
        milestone schedule — the policy itself triggers the refresh.
        """
        schema_error_uuids = self._collect_schema_errors(st)
        if schema_error_uuids and (self._gens_since_refresh >= self.recon_refresh_gap):
            st.log.warn(
                f"SchemaError from policy on {len(schema_error_uuids)} "
                f"pairs — triggering recon refresh + policy rewrite"
            )
            all_digests = [
                g2["digest"] for g2 in st.registry.generations if g2.get("digest")
            ][-_RECON_MAX_DIGESTS:]
            if all_digests:
                new_cfg = self._refresh_recon_config(st, all_digests)
                if new_cfg:
                    self._apply_schema_refresh(st, new_cfg, schema_error_uuids)
                else:
                    st.log.warn("schema-refresh recon unparseable; keeping old")
        self._gens_since_refresh += 1

    def _collect_schema_errors(self, st: _EvalState) -> set[str]:
        """Predecessor uuids whose cached verdict flagged schema_error."""
        errors = set()
        for g in st.preds:
            vkey = f"{st.effective_policy_sha}|{st.uuid}|{g['uuid']}"
            v = st.registry.get_verdict(vkey)
            if v and v.get("schema_error"):
                errors.add(g["uuid"])
        return errors

    def _refresh_recon_config(
        self, st: _EvalState, all_digests: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Rebuild the recon config from every accumulated digest."""
        prompt = recon_mod.build_prompt(st.goal, all_digests)
        parsed = self._llm_json(st.uuid, "recon_refresh", prompt, 4000, st.log)
        return recon_mod.ensure_shape(parsed)

    def _apply_schema_refresh(
        self,
        st: _EvalState,
        new_cfg: dict[str, Any],
        schema_error_uuids: set[str],
    ) -> None:
        """Commit the refreshed recon, rewrite the policy, re-run pairs."""
        new_version = self._config_hash(new_cfg)
        st.registry.set_recon(
            new_cfg, recon_mod.RECON_PROMPT_VERSION, recon_version=new_version
        )
        st.current_recon_version = new_version
        st.effective_policy_sha = self._policy_sha(new_version)
        if not self._rewrite_refreshed_policy(st, new_cfg):
            return
        self._gens_since_refresh = 0
        st.log.stage(
            "RECON",
            f"cache=schema-refreshed version {new_version[:6]} "
            f"policy rewritten ({len(schema_error_uuids)} pairs to "
            f"re-evaluate)",
        )
        st.log.preview(
            f"refreshed policy source (schema refresh, sha "
            f"..{st.effective_policy_sha[-4:]})",
            st.policy_src or "",
            max_lines=14,
        )
        self._reevaluate_schema_pairs(st, schema_error_uuids)

    def _rewrite_refreshed_policy(
        self, st: _EvalState, new_cfg: dict[str, Any]
    ) -> bool:
        """Rewrite the policy under the refreshed recon; False on failure."""

        def llm_json(prompt, mt, role):
            return self._llm_json(st.uuid, role, prompt, mt, st.log)

        smoke_digests = [st.digest_now] + [g2["digest"] for g2 in st.preds[:2]]
        new_policy_src, new_meta = policy_mod.ensure_policy(
            st.goal,
            new_cfg,
            smoke_digests,
            llm_json,
            repair_rounds=self.repair_rounds,
            smoke_timeout_s=self.scorer_timeout_s,
            log=st.log,
        )
        if not new_policy_src:
            st.log.warn("schema-refresh policy rewrite FAILED; keeping old")
            return False
        st.registry.set_policy(
            {
                "sha": st.effective_policy_sha,
                "source": new_policy_src,
                "meta": new_meta,
            }
        )
        st.policy_src = new_policy_src
        return True

    def _reevaluate_schema_pairs(
        self, st: _EvalState, schema_error_uuids: set[str]
    ) -> None:
        """Re-evaluate the schema-error pairs under the rewritten policy.

        Each re-evaluation REPLACES its pair record (outcome, winner,
        channel, deciding, feedback); the canonical outcomes and the
        outcome-derived tallies are then recomputed from the final
        records, so stage 7's reward and the weak-evidence guard see
        the re-evaluated verdicts — never the schema-error ties they
        replaced.
        """
        for rec in st.pair_records:
            other_uuid = rec.get("prev_uuid") or rec.get("other")
            if other_uuid in schema_error_uuids:
                self._reevaluate_one_pair(st, rec, other_uuid)
        st.outcomes = [
            _OUTCOME_CANONICAL.get(rec.get("outcome"), "tie") for rec in st.pair_records
        ]
        self._recount_pairs(st)

    def _reevaluate_one_pair(
        self, st: _EvalState, rec: dict[str, Any], other_uuid: str
    ) -> None:
        """Re-run ONE schema-error pair; replaces its record + feedback."""
        for g2 in st.preds:
            if g2["uuid"] != other_uuid:
                continue
            new_verdict = self._pair_verdict(
                st.uuid,
                st.goal,
                st.digest_now,
                st.ledger,
                st.current_ws,
                st.runner,
                g2,
                st.policy_src,
                st.effective_policy_sha,
                st.visual,
                True,
                st.registry,
                st.log,
            )
            if "winner" in new_verdict:
                self._replace_pair_record(st, rec, new_verdict)
                self._log_reevaluated_pair(st, other_uuid, rec, new_verdict)
            return

    @staticmethod
    def _replace_pair_record(
        st: _EvalState, rec: dict[str, Any], verdict: dict[str, Any]
    ) -> None:
        """Overwrite one pair record with the re-evaluated verdict."""
        winner = verdict.get("winner")
        if winner not in ("tie", st.uuid, rec["prev_uuid"]):
            winner = "tie"
        _, rec_outcome = _outcome_spellings(winner, st.uuid)
        won = winner == st.uuid
        rec.update(
            {
                "outcome": rec_outcome,
                "winner": winner,
                "channel": verdict.get("channel"),
                "deciding": verdict.get("deciding"),
                "confidence": verdict.get("confidence"),
                "mirrored": verdict.get("mirrored"),
                "feedback": (
                    verdict.get("feedback_winner")
                    if won
                    else verdict.get("feedback_looser")
                )
                or "",
                "detail": {
                    "feedback_winner": verdict.get("feedback_winner") or "",
                    "feedback_looser": verdict.get("feedback_looser") or "",
                },
            }
        )

    @staticmethod
    def _log_reevaluated_pair(
        st: _EvalState, other: str, rec: dict[str, Any], verdict: dict[str, Any]
    ) -> None:
        """PAIR payload for one schema-refresh re-evaluation."""
        canonical = _OUTCOME_CANONICAL.get(rec.get("outcome"), "tie")
        winner = rec["winner"]
        st.log.payload(
            "PAIR",
            (
                f"re-evaluated vs {other} under the refreshed policy "
                f"winner={winner if winner == 'tie' else winner[-8:]} "
                f"channel={verdict.get('channel')} "
                f"deciding={verdict.get('deciding')}"
            ),
            {
                "uuid_a": st.uuid,
                "uuid_b": other,
                "winner": winner,
                "outcome": canonical,
                "channel": verdict.get("channel"),
                "deciding": verdict.get("deciding"),
                "confidence": verdict.get("confidence"),
                "mirrored": verdict.get("mirrored"),
                "schema_refresh_reeval": True,
            },
        )

    @staticmethod
    def _recount_pairs(st: _EvalState) -> None:
        """Recompute outcome-derived tallies from the final pair records.

        Channel/error counters (n_errors, cached/visual stats) stay as
        the stage-6 observations — they are reporting-only; the reward
        and the weak-evidence guard read the tallies recomputed here.
        """
        st.count["n_tie"] = sum(1 for o in st.outcomes if o == "tie")
        st.count["n_decisive"] = sum(1 for o in st.outcomes if o != "tie")
        st.count["n_weak"] = sum(
            1
            for rec in st.pair_records
            if rec["outcome"] != "tie"
            and _decisive_weak_evidence(rec, st.primary_metrics)
        )

    def _log_pairs_stage(self, st: _EvalState) -> None:
        """PAIRS stage line with the pair-loop tallies."""
        antisym = (
            "ok"
            if st.count["n_disagree"] == 0
            else f"{st.count['n_disagree']}-disagreements"
        )
        st.log.stage(
            "PAIRS",
            (
                f"predecessors={len(st.preds)} "
                f"cached={st.count['n_cached']} "
                f"decisive={st.count['n_decisive']} "
                f"tie={st.count['n_tie']} "
                f"errors={st.count['n_errors']} "
                f"weak={st.count['n_weak']} "
                f"antisym={antisym}"
            ),
            excluded=len(st.excluded),
        )

    def _compute_reward(self, st: _EvalState) -> None:
        """Stage 7: arrival-moment reward (history never rewritten)."""
        st.reward_data = verdicts_mod.win_rate_reward(st.outcomes)
        st.reward = float(st.reward_data["reward"])
        st.bt_data = None
        if self.reward_mode == "bradley_terry" and st.preds:
            self._bradley_terry_reward(st)
        # P0.7 weak-evidence guard: when EVERY decisive pair rested on
        # self-reported metric evidence the policy cannot ground in the
        # recon primary metrics, the reward is capped so a spurious
        # 1.0 can never trip the learned early-stop threshold.
        st.reward_weak_evidence = (
            bool(st.pair_records)
            and st.count["n_decisive"] > 0
            and st.count["n_weak"] == st.count["n_decisive"]
        )
        if st.reward_weak_evidence:
            st.reward = min(st.reward, WEAK_EVIDENCE_REWARD_CAP)
            st.log.warn(
                f"weak-evidence guard: all {st.count['n_decisive']} "
                f"decisive pairs rest on self-reported metrics outside "
                f"the recon primary metrics; reward capped at "
                f"{WEAK_EVIDENCE_REWARD_CAP}"
            )
        rd = st.reward_data
        st.log.stage(
            "REWARD",
            (
                f"mode={self.reward_mode} value={st.reward:.3f} "
                f"({rd['wins']}W/{rd['losses']}L/{rd['ties']}T vs "
                f"predecessors) "
                f"weak={'yes' if st.reward_weak_evidence else 'no'}"
            ),
        )
        self._log_tournament(st)

    def _log_tournament(self, st: _EvalState) -> None:
        """Live Bradley-Terry standings after the reward stage.

        In ``bradley_terry`` mode this is the reward's own fit; in the
        default ``win_rate`` mode the same fit is computed for DISPLAY
        ONLY (pure numeric, no LLM) so the operator always sees the
        tournament table.
        """
        if not st.pair_records:
            return
        bt = st.bt_data
        if bt is None:
            bt = verdicts_mod.bradley_terry_reward(
                [g["uuid"] for g in st.registry.previous_generations(st.uuid)]
                + [st.uuid],
                [
                    {"a": st.uuid, "b": p["prev_uuid"], "winner": p["winner"]}
                    for p in st.pair_records
                ],
            )
        note = (
            "reward mode"
            if st.bt_data is not None
            else f"display-only; reward={self.reward_mode}"
        )
        self._print_standings(st, bt, note)

    def _print_standings(self, st: _EvalState, bt: dict[str, Any], note: str) -> None:
        """The aligned standings table rows (current gen highlighted)."""
        rows: list[list[Any]] = []
        styles: list[str | None] = []
        for uuid, rating in sorted(
            bt.get("ratings", {}).items(), key=lambda kv: -kv[1]
        ):
            mark = " ·this" if uuid == st.uuid else ""
            rows.append(
                [uuid[-8:] + mark, f"{rating:.2f}", "█" * max(1, int(rating * 4))]
            )
            styles.append(GREEN if uuid == st.uuid else None)
        st.log.table(
            f"bradley-terry standings ({note})",
            ["gen", "rating", "strength"],
            rows,
            row_styles=styles,
            pick=(bt.get("pick") or "")[-8:],
        )

    def _bradley_terry_reward(self, st: _EvalState) -> None:
        """Optional BT fit (``policy_bt_reward = bradley_terry``)."""
        prefix = [g["uuid"] for g in st.registry.previous_generations(st.uuid)] + [
            st.uuid
        ]
        st.bt_data = verdicts_mod.bradley_terry_reward(
            prefix,
            [
                {
                    "a": st.uuid,
                    "b": p["prev_uuid"],
                    "winner": p["winner"],
                }
                for p in st.pair_records
            ],
        )
        ratings = st.bt_data["ratings"]
        mean_r = sum(ratings.values()) / max(1, len(ratings))
        st.reward = ratings.get(st.uuid, mean_r)

    def _persist_generation(self, st: _EvalState) -> None:
        """N7/P2.1: generation + verdicts persist RIGHT after the reward.

        Every later stage (gradient fold, refiner, artifacts) is
        optional and must never lose them.
        """
        st.reward_capped = round(max(0.0, min(1.0, st.reward)), 4)
        st.registry.record_generation(
            st.uuid,
            st.digest_now,
            st.ledger,
            reward=st.reward_capped,
            stats={
                "wins": st.reward_data["wins"],
                "losses": st.reward_data["losses"],
                "ties": st.reward_data["ties"],
            },
            session=self._session_tag,
        )
        st.registry.save()

    def _build_gradient(self, st: _EvalState) -> None:
        """Stage 8: elimination-first fold + optional E47 refiner.

        §4.2: the fold renders the METRICS section from the LEDGER
        (deterministic, data-side), never regex-parsed from the
        LLM-written feedback.
        """
        coverage = recon_mod.coverage_footer(st.recon_cfg)
        fold_ledger = {
            "current": st.digest_now.get("primary_values") or {},
            "preds": {
                p["uuid"]: (p.get("digest") or {}).get("primary_values") or {}
                for p in st.preds
            },
            "metrics": st.primary_metrics,
        }
        st.raw_gradient, fold_meta = gradient_mod.fold_gradient(
            st.uuid, st.pair_records, coverage, ledger=fold_ledger
        )
        rd = st.reward_data
        st.log.payload(
            "GRADIENT_RAW",
            (
                f"raw gradient logged ({len(st.raw_gradient)} chars, "
                f"{rd['wins']}W/{rd['losses']}L/{rd['ties']}T)"
            ),
            {
                "gradient": st.raw_gradient,
                "coverage_footer_used": coverage,
                "coverage_footer_appended": fold_meta["footer_appended"],
                "fold_tied": fold_meta["tied"],
                "fold_not_assessed": fold_meta["not_assessed"],
            },
        )
        self._refine_gradient(st)
        self._log_gradient(st)

    def _refine_gradient(self, st: _EvalState) -> None:
        """E47 evidence-checked refiner; failure degrades to the raw fold."""

        def llm_json(prompt, mt, role):
            return self._llm_json(st.uuid, role, prompt, mt, st.log)

        st.refinement = None
        if not self.refiner:
            return
        preview = gradient_mod.build_preview(st.current_ws, st.digest_now, st.ledger)
        try:
            st.refinement = gradient_mod.refine_gradient(
                llm_json, st.uuid, st.raw_gradient, preview
            )
        except Exception as e:  # noqa: BLE001 — N7: refiner failure
            # degrades to the raw fold; it must never abort evaluate()
            # after the reward and verdicts were already computed.
            st.log.warn(
                f"refiner failed ({type(e).__name__}: {str(e)[:200]}); "
                f"using the raw gradient fold"
            )
            st.refinement = None

    def _log_gradient(self, st: _EvalState) -> None:
        """GRADIENT stage + the final refined-gradient payload (S5)."""
        st.gradient_text = gradient_mod.refined_text(st.raw_gradient, st.refinement)
        st.rstats = gradient_mod.refiner_stats(st.refinement)
        rd = st.reward_data
        rs = st.rstats
        st.log.stage(
            "GRADIENT",
            (
                f"fold: {rd['losses']}L/{rd['wins']}W/{rd['ties']}T "
                f"refiner={'on' if self.refiner else 'off'} "
                f"issues={rs['issues']} missing={rs['missing']} "
                f"corr={rs['corr']}"
            ),
        )
        st.log.payload(
            "GRADIENT_REFINED",
            (
                f"final gradient logged ({len(st.gradient_text)} chars, "
                f"refiner={'on' if self.refiner else 'off'}, "
                f"issues={rs['issues']} missing={rs['missing']} "
                f"corr={rs['corr']})"
            ),
            {
                "gradient": st.gradient_text,
                "refinement": st.refinement,
                "refiner_used": bool(st.refinement is not None),
            },
        )

    def _build_scores(self, st: _EvalState) -> None:
        """Assemble the engine-facing scores dict (§2.5 contract)."""
        st.scores = {
            **self._score_core(st),
            **self._score_policy(st),
            **self._score_tail(st),
        }

    def _score_core(self, st: _EvalState) -> dict[str, Any]:
        """Reward/pair-count scores + the pair_outcomes engine contract."""
        rd = st.reward_data
        return {
            "overall_score": st.reward_capped,
            "win_rate": round(float(rd["reward"]), 4),
            "reward_mode": self.reward_mode,
            "reward_weak_evidence": st.reward_weak_evidence,
            "n_weak_evidence_pairs": st.count["n_weak"],
            "n_pair_errors": st.count["n_errors"],
            "n_pairs": rd["n_pairs"],
            "n_wins": rd["wins"],
            "n_losses": rd["losses"],
            "n_ties": rd["ties"],
            "n_predecessors": len(st.preds),
            "n_excluded_predecessors": len(st.excluded),
            # Engine contract (stop plan 2026-10-08 §2.5): normalised
            # head-to-head outcomes of THIS generation vs each
            # predecessor; the engine's StopController reads only this.
            "pair_outcomes": {p["prev_uuid"]: p["outcome"] for p in st.pair_records},
            "figure_task": st.figure_task,
            "visual_pairs": st.count["n_visual"],
        }

    @staticmethod
    def _score_policy(st: _EvalState) -> dict[str, Any]:
        """Policy/re-exec/metric-ledger observability scores."""
        return {
            "policy_sha": st.effective_policy_sha,
            "policy_attempts": st.policy_meta.get("attempts"),
            "policy_repairs": st.policy_meta.get("repairs"),
            "policy_smoke": st.policy_meta.get("smoke"),
            # Compact metric ledger summary (review §4.3): the frontier
            # probe reads this to detect all-tie objective metrics —
            # the verifier's measured values, nothing synthetic.
            "metric_summary": {
                mid: {"status": r.get("status"), "value": r.get("value")}
                for mid, r in (st.digest_now.get("primary_values") or {}).items()
            },
            "reexec": {
                "exit_code": st.ledger.get("exit_code"),
                "wall_time_s": st.ledger.get("wall_time_s"),
                "timed_out": st.ledger.get("timed_out"),
                "source_manifest_intact": st.ledger.get("source_manifest_intact"),
                "n_fresh_artifacts": len(st.ledger.get("fresh_artifacts") or {}),
            },
        }

    @staticmethod
    def _score_tail(st: _EvalState) -> dict[str, Any]:
        """BT/refiner/spend tails + both gradient keys + fingerprint."""
        return {
            "bt_pick": (st.bt_data or {}).get("pick"),
            "bt_ratings": (st.bt_data or {}).get("ratings"),
            "refiner_issues": st.rstats["issues"],
            "refiner_missing_files": st.rstats["missing"],
            "refiner_corrections": st.rstats["corr"],
            "llm_calls": st.log.n_llm_calls,
            "llm_spend_usd": round(st.log.spend_usd, 6),
            "abstracted_textual_gradient": st.gradient_text,
            # Legacy-typo key kept for engine compatibility (the hybrid
            # writes both; the engine reads one).
            "abstractec_textual_gradient": st.gradient_text,
            "failure_fingerprint": {
                "vector": [0.0] * _FP_DIM,
                "presence_mask": [0.0] * _FP_DIM,
                "pass_rates": [0.5] * _FP_DIM,
            },
        }

    def _finalize(self, st: _EvalState) -> dict[str, Any]:
        """Write artifacts, persist scores, DONE stage + summary box."""
        self._write_artifacts(
            st.uuid,
            st.scores,
            st.pair_records,
            st.raw_gradient,
            st.gradient_text,
        )
        try:
            self._save_results(st.scores, st.uuid, "verifier")
        except Exception as e:  # noqa: BLE001 — persistence failure logged only
            self.logger.error(f"Failed to persist policy_bt scores for {st.uuid}: {e}")
        wall = time.time() - st.t_total
        rd = st.reward_data
        st.log.stage(
            "DONE",
            (
                f"reward={st.reward_capped} "
                f"total_llm={st.log.spend_line()} "
                f"wall={wall:.1f}s"
            ),
        )
        print_box(
            f"reward {st.reward_capped:.3f} | {rd['wins']}W/"
            f"{rd['losses']}L/{rd['ties']}T vs "
            f"{rd['n_pairs']} previous | llm {st.log.spend_line()} | "
            f"{wall:.1f}s",
            title=f"Policy-BT verifier · {st.uuid}",
            color=CYAN,
        )
        return {"uuid": st.uuid, "pairs": st.pair_records, **st.scores}

    # ------------------------------------------------------------------
    # Pair verdict (crash rung -> figure rung -> channel -> code policy)
    # ------------------------------------------------------------------
    def _make_visual_channel(
        self,
        uuid: str,
        goal: str,
        registry: TaskRegistry,
        log: StageLogger,
    ) -> visual_mod.VisualChannel | None:
        """Visual channel wired to the registry cache + vision call.

        Returns None (and logs why) when no vision model is configured —
        the caller then keeps the code policy as the decision channel.
        """
        if self._vision_llm_config is None:
            return None

        def cache_get(key: str) -> dict[str, Any] | None:
            # E48 self-heal scan: cached parse-fails never satisfy a hit;
            # any healed entry for the key wins.
            for entry in registry.visual_entries(key):
                if entry.get("parse_fail"):
                    continue
                return entry
            return None

        def cache_put(key: str, entry: dict[str, Any]) -> None:
            registry.add_visual_entry(key, entry)

        return visual_mod.VisualChannel(
            vision_call=lambda prompt, fa, fb, mt, role: self._vision_json(
                uuid, prompt, fa, fb, mt, log, role=role
            ),
            cache_get=cache_get,
            cache_put=cache_put,
            log=log,
        )

    def _pair_verdict(
        self,
        uuid: str,
        goal: str,
        digest_now: dict[str, Any],
        ledger: dict[str, Any],
        current_ws: Path,
        runner: ReexecRunner,
        pred: dict[str, Any],
        policy_src: str | None,
        policy_sha: str,
        visual: Any,
        allow_visual: bool,
        registry: TaskRegistry,
        log: StageLogger,
    ) -> dict[str, Any]:
        """One pair verdict with the full ladder (plan §5 step 6).

        Ladder: deterministic crash rung -> (figure tasks) figure rung ->
        kimi per-pair channel (both orders) -> code policy (both orders,
        antisymmetry asserted) -> honest tie.
        """
        other = pred["uuid"]
        # Deterministic exec-crash rung FIRST (safeguard 8), both channels.
        rung = verdicts_mod.crash_rung(uuid, ledger, other, pred.get("ledger") or {})
        if rung is not None:
            return {**rung, "_cached": False}
        ctx = _PairCtx(
            uuid=uuid,
            goal=goal,
            digest_now=digest_now,
            current_ws=current_ws,
            runner=runner,
            pred=pred,
            policy_src=policy_src,
            policy_sha=policy_sha,
            visual=visual,
            registry=registry,
            log=log,
        )
        extra: dict[str, Any] = {}
        if visual is not None:
            verdict, extra = self._visual_ladder(ctx, allow_visual)
            if verdict is not None:
                return verdict
        return self._code_policy_verdict(ctx, extra)

    def _predecessor_root(self, ctx: _PairCtx, other: str) -> Path | None:
        """Predecessor workspace: sandbox copy, else /tmp snapshot (P3.3).

        NO live-workspace fallback — the live workspace holds THIS
        generation's files; judging predecessor figures from it compares
        the present against itself.
        """
        pred_root = ctx.runner.run_copy_path(other)
        if not pred_root.exists():
            pred_root = find_run_snapshot(other)
        if pred_root is not None:
            return pred_root
        ctx.log.warn(
            f"no sandbox copy or /tmp snapshot for predecessor "
            f"{other[-8:]}; skipping the kimi channel for this "
            f"pair — the code policy decides"
        )
        return None

    def _visual_ladder(
        self, ctx: _PairCtx, allow_visual: bool
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Figure rung + kimi pair channel for one pair (P3.2-P3.5).

        Returns ``(verdict, extra)`` — a None *verdict* defers to the
        code policy; *extra* carries the fresh-visual flag a deferred
        kimi tie must still count (P3.5).
        """
        other = ctx.pred["uuid"]
        extra: dict[str, Any] = {}
        pred_root = self._predecessor_root(ctx, other)
        if pred_root is None:
            return None, extra
        key = f"vis|{visual_mod.VISUAL_PROMPT_VERSION}|{ctx.uuid}|{other}"
        cached = ctx.registry.get_verdict(key)
        figs_now = visual_mod.select_figures(ctx.digest_now, ctx.current_ws)
        figs_pred = visual_mod.select_figures(ctx.pred["digest"], pred_root)
        # P3.4: the deliverable rung decides on DIGEST figure presence;
        # the 6 MB cap only filters uploads.
        n_now = len(ctx.digest_now.get("figures") or [])
        n_pred = len(ctx.pred["digest"].get("figures") or [])
        frung = visual_mod.figure_rung(ctx.uuid, n_now, other, n_pred)
        if frung is not None:
            ctx.registry.set_verdict(key, frung)
            return {**frung, "_cached": False, "_visual": True}, extra
        if not (figs_now and figs_pred):
            # Both sides lack figures -> the code policy decides.
            return None, extra
        if cached is not None:
            hit = {**cached, "_cached": True, "_visual": True, "_cached_visual": True}
            return hit, extra
        return self._judge_visual_pair(
            ctx, other, figs_now, figs_pred, allow_visual, key
        )

    def _judge_visual_pair(
        self,
        ctx: _PairCtx,
        other: str,
        figs_now: list[Path],
        figs_pred: list[Path],
        allow_visual: bool,
        key: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Fresh kimi judgment, honoring the per-gen budget cap (P3.5)."""
        if not allow_visual:
            ctx.log.warn(
                f"visual pair budget cap "
                f"({self.visual_max_pairs_per_gen}/gen) tripped "
                f"on pair ({ctx.uuid[-8:]} vs {other[-8:]}); "
                f"falling to the code-policy path"
            )
            return None, {}
        verdict = ctx.visual.judge_pair(ctx.goal, ctx.uuid, figs_now, other, figs_pred)
        if verdict is None:
            return None, {}
        if not (self.visual_tiebreak_code and verdict.get("winner") == "tie"):
            ctx.registry.set_verdict(key, verdict)
            return (
                {
                    **verdict,
                    "_cached": False,
                    "_visual": True,
                    "_fresh_visual_pair": True,
                },
                {},
            )
        # P3.5: visual_tiebreak_code=on — a kimi tie defers to the code
        # policy (the fresh visual call still consumed budget).
        ctx.log.detail(
            f"kimi tie on pair ({ctx.uuid[-8:]} vs {other[-8:]}); "
            f"visual_tiebreak_code=on — the code policy decides"
        )
        return None, {"_fresh_visual_pair": True}

    def _code_policy_verdict(
        self, ctx: _PairCtx, extra: dict[str, Any]
    ) -> dict[str, Any]:
        """Code-policy path: both orders, antisymmetry asserted."""
        if ctx.policy_src is None:
            return _no_policy_verdict()
        other = ctx.pred["uuid"]
        key = code_verdict_key(ctx.policy_sha, ctx.uuid, other)
        cached = ctx.registry.get_verdict(key)
        if cached is not None:
            return {**cached, "_cached": True, **extra}
        # P2.10: the pair run honours policy_bt_scorer_timeout_s.
        verdict = verdicts_mod.code_pair_verdict(
            functools.partial(policy_mod.run_policy, timeout=self.scorer_timeout_s),
            ctx.policy_src,
            ctx.uuid,
            ctx.digest_now,
            other,
            ctx.pred["digest"],
            log=ctx.log,
        )
        # P0.5: an ERRORED pair verdict is NEVER cached — a policy that
        # errors once must be able to produce a decisive verdict on the
        # next evaluation instead of a permanent cached tie.
        if "error" not in verdict:
            ctx.registry.set_verdict(key, verdict)
        return {**verdict, "_cached": False, **extra}

    # ------------------------------------------------------------------
    # Short-circuit (empty-run parity with the hybrid/legacy verifiers)
    # ------------------------------------------------------------------

    def _short_circuit_failed_run(
        self,
        uuid: str,
        reason: str = "workflow_generation_or_execution_failed",
        gradient: str = _SHORT_CIRCUIT_GRADIENT,
    ) -> dict[str, Any]:
        """Score an empty run: reward 0.0 and a NON-EMPTY gradient."""
        print_box(
            f"workflow {uuid} produced no code and no state_result "
            f"({reason}); policy_bt verifier returns 0.0 without running "
            f"the pipeline.",
            title="Policy-BT verifier short-circuit — generation failed",
            color=RED,
        )
        artifact_score = 1.0 if any(self.workspace_dir.iterdir()) else 0.0
        gradient_text = (
            f"{gradient}\n"
            f"short-circuit check: execution_produced_artifacts="
            f"{artifact_score:.0f} from the workspace listing."
        )
        scores = self._short_circuit_scores(reason, gradient_text)
        self._write_artifacts(uuid, scores, [], gradient_text, gradient_text)
        try:
            self._save_results(scores, uuid, "verifier")
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to persist short-circuit scores: {e}")
        return {"uuid": uuid, "pairs": [], **scores}

    def _short_circuit_scores(self, reason: str, gradient_text: str) -> dict[str, Any]:
        """Scores for a short-circuited (failed) generation."""
        return {
            "overall_score": 0.0,
            "win_rate": 0.0,
            "reward_mode": self.reward_mode,
            "reward_weak_evidence": False,
            "n_pairs": 0,
            "n_wins": 0,
            # Engine contract §2.5: short-circuit ran no pairs.
            "pair_outcomes": {},
            "n_predecessors": 0,
            "n_excluded_predecessors": 0,
            "skipped_reason": reason,
            "abstracted_textual_gradient": gradient_text,
            "abstractec_textual_gradient": gradient_text,
            "failure_fingerprint": {
                "vector": [0.0] * _FP_DIM,
                "presence_mask": [0.0] * _FP_DIM,
                "pass_rates": [0.5] * _FP_DIM,
            },
        }

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def _write_artifacts(
        self,
        uuid: str,
        scores: dict[str, Any],
        pair_records: list[dict[str, Any]],
        raw_gradient: str,
        gradient_text: str,
    ) -> None:
        """Write ``evaluation.txt`` + ``textual_gradient.txt`` for *uuid*."""
        out_dir = self.workflow_dir / uuid
        lines = [
            "Policy-BT Verifier Evaluation (E44-E48 architecture)",
            "=" * 60,
            f"reward: {scores.get('overall_score')} (mode={scores.get('reward_mode')})",
            f"pairs: {scores.get('n_wins')}W/{scores.get('n_losses')}L/"
            f"{scores.get('n_ties')}T vs {scores.get('n_predecessors')} "
            f"predecessors ({scores.get('n_excluded_predecessors')} excluded)",
            f"re-execution: {scores.get('reexec')}",
            f"policy: sha={scores.get('policy_sha')} "
            f"attempts={scores.get('policy_attempts')} "
            f"smoke={scores.get('policy_smoke')}",
            "",
            "PAIR VERDICTS:",
        ]
        lines += _pair_verdict_lines(pair_records)
        lines += ["", "RAW GRADIENT (fold):", raw_gradient]
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "evaluation.txt").write_text("\n".join(lines), encoding="utf-8")
            (out_dir / "textual_gradient.txt").write_text(
                gradient_text, encoding="utf-8"
            )
            print_ok(
                f"[policy_bt {uuid}] evaluation.txt + textual_gradient.txt written"
            )
        except OSError as e:
            self.logger.error(f"Could not write artifacts for {uuid}: {e}")

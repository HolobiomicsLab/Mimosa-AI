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
"""

from __future__ import annotations

import base64
import functools
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sources.cli.pretty_print import CYAN, RED, print_box, print_ok
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
from . import reexec as reexec_mod
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
# self-reported metric evidence (P0.7 — HP gen 4's reward 1.0 crossed
# the early-stop threshold on a boolean-vs-coordinate artifact).
WEAK_EVIDENCE_REWARD_CAP = 0.75
#: Deciding keys that name self-reported (workspace-claimed) metrics.
_SELF_REPORTED_KEY = re.compile(r"(log_metrics|json_metrics)", re.I)
#: Deciding keys that only measure deliverable FRESHNESS (N2, HP
# 2026-10-08): a raw count difference of produced_this_run deliverables
# ("3 vs 2") is not quality evidence — a workflow can win by writing
# more files while the science is unchanged.
_FRESHNESS_KEY = re.compile(r"produced_this_run", re.I)
#: Deciding keys naming the goal contract — presence/absence of a
# goal-named file is objective and MAY decide (N2 exception).
_CONTRACT_KEY = re.compile(r"contract_files", re.I)


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

    uuid-anchored resolution (never a blind glob): only directories whose
    name ends in exactly ``_<uuid>`` with a session-style prefix match.
    """
    if not uuid:
        return None
    candidates = [p for p in Path("/tmp").glob(f"mimosa_run_*_{uuid}") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _sandbox_listing(
    ws: Path, fresh: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Complete file listing of the re-execution sandbox copy.

    Every file (relative path, bytes, fresh-this-run flag) except the
    skip-dirs the manifests already ignore (venv, caches) — so the log
    shows exactly what the digest builder saw.
    """
    files: list[dict[str, Any]] = []
    if not ws.exists():
        return files
    fresh = fresh or {}
    for p in sorted(ws.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ws).as_posix()
        if any(part in reexec_mod.MANIFEST_SKIP_DIRS for part in Path(rel).parts):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        files.append({"path": rel, "bytes": size, "fresh": rel in fresh})
    return files


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
        self.reward_mode = str(
            getattr(config, "policy_bt_reward", "win_rate")
        ).lower()
        if self.reward_mode not in ("win_rate", "bradley_terry"):
            self.reward_mode = "win_rate"
        self.scorer_timeout_s = max(
            5, int(getattr(config, "policy_bt_scorer_timeout_s", 60))
        )
        self.visual_channel = bool(
            getattr(config, "policy_bt_visual_channel", True)
        )
        self.refiner = bool(getattr(config, "policy_bt_refiner", True))
        self.repair_rounds = max(
            0, int(getattr(config, "policy_bt_repair_rounds", 3))
        )
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
        # Incremental recon refresh: at these generation counts, rebuild the
        # recon config with ALL accumulated digests (not just the first
        # generation's). If the config changed materially, the policy is
        # rewritten with the same decision ladder but updated schema/metric
        # knowledge. Old verdicts stay cached under the old policy sha.
        # Schema-error-triggered refresh: when the policy raises SchemaError
        # on a pair (unknown keys/paths), the recon is refreshed with ALL
        # digests and the policy rewritten. A minimum gap prevents thrashing.
        self.recon_refresh_gap = max(
            1, int(getattr(config, "policy_bt_recon_refresh_gap", 2))
        )
        self._gens_since_refresh = 0
        # N9: registry generations carry the tag of the campaign session
        # that wrote them, so a NEW campaign's generation 1 comparing
        # against a PREVIOUS campaign's generations is a loud WARN, not
        # a silent contamination (the engine's StopController champion
        # is session-scoped by construction).
        self._session_tag = f"s{int(time.time())}-{uuid4().hex[:8]}"
        self._runner_temp_root = Path(
            getattr(config, "temp_dir", None)
            or self.workflow_dir / "_verifier_tmp"
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
        cfg = LLMConfig().from_dict(
            {
                "model": self.llm_config.model,
                "provider": self.llm_config.provider,
                "temperature": 0.0,
                "reasoning_effort": "low",  # code-gen: token space > reasoning depth
                "max_tokens": max(0, int(max_tokens)),
                "openrouter_provider": self.llm_config.openrouter_provider,
                "openrouter_quantizations": self.llm_config.openrouter_quantizations,
                "key": self.llm_config.key,
            }
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

        P3.2: each side's figures are announced by a text label before
        their images ("Candidate A figures:" / "Candidate B figures:")
        so the judge can attribute what it sees.
        P3.1: *role* (attempt + max_tokens) is appended to the provider
        agent name — a retry can never be served the cached failed
        attempt. The 6 MB cap filters UPLOADS only (P3.4); a skipped
        oversized figure is still a produced deliverable.
        """
        if self._vision_llm_config is None:
            raise RuntimeError(
                "no vision_judge_model configured; cannot run the kimi "
                "pair channel"
            )

        content: list[Any] = [{"type": "text", "text": prompt}]
        for label, figs in (
            ("Candidate A figures:", figs_a),
            ("Candidate B figures:", figs_b),
        ):
            content.append({"type": "text", "text": label})
            for img in figs:
                try:
                    if img.stat().st_size > visual_mod.MAX_IMAGE_BYTES:
                        log.warn(
                            f"figure {img.name} exceeds the 6 MB upload cap; "
                            f"skipped from the vision call (still a produced "
                            f"deliverable)"
                        )
                        continue
                    b64 = base64.b64encode(img.read_bytes()).decode("ascii")
                except OSError as e:
                    log.warn(f"figure {img.name} unreadable: {e}")
                    continue
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    }
                )
        agent = "policy_bt_kimi_pair" + (f"_{role}" if role else "")
        memory_path = Path(self.memory_dir) / uuid
        memory_path.mkdir(parents=True, exist_ok=True)
        cfg = LLMConfig().from_dict(
            {
                "model": self._vision_llm_config.model,
                "provider": self._vision_llm_config.provider,
                "temperature": 0.0,
                "reasoning_effort": self._vision_llm_config.reasoning_effort,
                "max_tokens": max(0, int(max_tokens)),
                "openrouter_provider": (
                    self._vision_llm_config.openrouter_provider
                ),
                "openrouter_quantizations": (
                    self._vision_llm_config.openrouter_quantizations
                ),
                "key": self._vision_llm_config.key,
            }
        )
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

        t_total = time.time()
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

        log = StageLogger(
            uuid,
            sink=self.workflow_dir / uuid / "verifier_policy_bt.log",
            verbose=self.verbose,
        )
        registry = TaskRegistry.load(self._runner_temp_root, goal, self.logger)
        tkey = task_key(goal)
        sandbox_root = self._runner_temp_root / "policy_bt" / tkey

        # ---- stage 5: fresh re-execution of THIS generation ----
        runner = ReexecRunner(
            sandbox_root,
            timeout_s=self.reexec_timeout_s,
            venv_reuse=self.venv_reuse,
            log=log,
        )
        ledger = runner.run(self.workspace_dir, uuid)
        write_ledger_file(
            ledger, sandbox_root / f"policy_bt_ledger_{uuid}.json"
        )
        current_ws = runner.run_copy_path(uuid)
        listing = _sandbox_listing(current_ws, ledger.get("fresh_artifacts"))
        if not current_ws.exists():  # re-execution never produced a copy
            listing = [
                {
                    "path": "<live workspace fallback>",
                    "bytes": None,
                    "fresh": False,
                }
            ]
        log.payload(
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
                "sandbox": str(current_ws),
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
            log.warn(
                "source workspace was MUTATED during re-execution "
                "(source_manifest_intact=False) — this generation's "
                "ledger is untrustworthy; crash rung class "
                "manifest_mutated applies"
            )

        # ---- digest of this generation (from the re-execution copy) ----
        if not current_ws.exists():  # re-execution never produced a copy
            current_ws = self.workspace_dir
        digest_now = digest_mod.build_digest(uuid, current_ws, ledger, goal)
        registry.record_generation(
            uuid, digest_now, ledger, reward=None, session=self._session_tag
        )
        dsum = digest_mod.digest_summary(digest_now)
        log.stage(
            "DIGEST",
            (
                f"tables={dsum['tables']} arrays={dsum['arrays']} "
                f"figures={dsum['figures']} metrics={dsum['metrics']} "
                f"deliverables={dsum['deliverables']} "
                f"id_order={dsum['id_order']}"
            ),
        )

        # ---- predecessors: tiered artifact resolution (plan §5b) ----
        excluded: list[str] = []
        preds: list[dict[str, Any]] = []
        for g in registry.previous_generations(uuid):
            if g.get("digest"):
                preds.append(g)
                continue
            snap = find_run_snapshot(g["uuid"])
            if snap is not None:
                boot_ledger = {
                    "bootstrap": f"snapshot:{snap.name}",
                    "exit_code": None,
                }
                boot_digest = digest_mod.build_digest(
                    g["uuid"], snap, boot_ledger, goal
                )
                for g2 in registry.generations:
                    if g2.get("uuid") == g["uuid"]:
                        g2["digest"] = boot_digest
                        g2["ledger"] = boot_ledger
                preds.append(
                    {
                        "uuid": g["uuid"],
                        "arrival": g.get("arrival"),
                        "session": g.get("session"),
                        "digest": boot_digest,
                        "ledger": boot_ledger,
                    }
                )
                log.warn(
                    f"predecessor {g['uuid'][-8:]} had no registered digest; "
                    f"rebuilt from /tmp snapshot {snap.name} (no fresh "
                    f"execution facts — exec block is null)"
                )
            else:
                excluded.append(g["uuid"])
                log.warn(
                    f"predecessor {g['uuid'][-8:]} is unscoreable: no "
                    f"registered digest, no /tmp snapshot, not the live "
                    f"workspace — EXCLUDED from pairwise comparison "
                    f"(never fabricated)"
                )
        if self.max_pairs and len(preds) > self.max_pairs:
            dropped = preds[: len(preds) - self.max_pairs]
            preds = preds[len(preds) - self.max_pairs :]
            log.warn(
                f"policy_bt_max_pairs={self.max_pairs}: comparing only the "
                f"{len(preds)} most recent predecessors "
                f"({len(dropped)} older dropped)"
            )

        # N9: cross-session contamination is loud, not silent — the
        # registry persists per goal across campaigns; without this a
        # new campaign's generation 1 competes against the previous
        # campaign's generations (champion selection stays session-scoped
        # in the engine either way).
        foreign = [
            p["uuid"]
            for p in preds
            if p.get("session") != self._session_tag
        ]
        if foreign:
            log.warn(
                f"{len(foreign)} predecessor(s) were recorded by ANOTHER "
                f"campaign session (e.g. {foreign[0][-8:]}); head-to-head "
                f"outcomes span sessions — clean the registry or run "
                f"cleanup.sh for a fresh campaign"
            )

        # ---- per-generation digest summaries (everything compared) ----
        def _digest_entry(u: str, dg: dict[str, Any]) -> dict[str, Any]:
            s = digest_mod.digest_summary(dg)
            jm = dg.get("json_metrics") or {}
            return {
                "uuid": u,
                **s,
                "json_metric_files": {
                    k: len(v) for k, v in jm.items()
                },
                "deliverable_paths": [
                    dv.get("path") for dv in dg.get("deliverables") or []
                ],
                "exec": {
                    k: (dg.get("exec") or {}).get(k)
                    for k in ("exit_code", "wall_time_s", "timed_out")
                },
            }

        compared = [
            _digest_entry(uuid, digest_now),
        ] + [_digest_entry(p["uuid"], p["digest"]) for p in preds]
        log.payload(
            "DIGESTS",
            (
                f"digest summaries logged for {len(compared)} generations "
                f"(current + {len(preds)} preds, {len(excluded)} excluded)"
            ),
            compared,
        )

        # ---- stage 3: recon TASK CONFIG (first call + incremental refresh) ----
        recon_cfg = registry.get_recon(recon_mod.RECON_PROMPT_VERSION)
        recon_cache = "hit"
        recon_refreshed = False
        # N6/P2.3: a refreshed recon is HELD in memory and committed only
        # after the policy rewrite passes — committing first left every
        # later pair a no_policy tie (no policy under the new sha) and
        # re-billed the policy LLM every generation.
        pending_recon: tuple[dict[str, Any], str] | None = None
        if recon_cfg is None:
            # First generation: build from this digest alone.
            slot_digests = [digest_now]
            prompt = recon_mod.build_prompt(goal, slot_digests)
            parsed = self._llm_json(
                uuid, "recon", prompt, 4000, log
            )
            recon_cfg = recon_mod.ensure_shape(parsed)
            if recon_cfg:
                registry.set_recon(
                    recon_cfg, recon_mod.RECON_PROMPT_VERSION,
                    recon_version=self._config_hash(recon_cfg),
                )
            else:
                log.warn("recon call unparseable; continuing with empty config")
            recon_cache = "miss"
        # (Schema-error-triggered refresh happens AFTER the pair loop —
        #  the policy signals SchemaError on unknown keys, then the recon
        #  is rebuilt. No milestone schedule; see the post-loop block.)
        log.stage("RECON", f"cache={recon_cache} (task goal-hash {tkey})")

        log.payload(
            "RECON",
            f"task config logged (goal-hash {tkey}, cache={recon_cache})",
            recon_cfg or {},
        )

        # ---- metric ledger (review 2026-10-09 §4.1) ----
        # ONE deterministic resolution of the recon primary metrics per
        # generation, stored on the digest itself: the policy compares
        # primary_values, the gradient fold renders METRICS from them
        # and the scores expose a compact summary. Ledgers carry the
        # recon version they were resolved under; a refresh that adds
        # aliases re-resolves every predecessor (deterministic, free,
        # no LLM).
        current_recon_version = (
            pending_recon[1] if pending_recon else registry.get_recon_version()
        )
        primary_metrics = (recon_cfg or {}).get("primary_metrics") or []
        # LLM text extractor — replaces ALL regex scraping (operator
        # directive 2026-10-10: extraction from raw output is an LLM call).
        def _extract_llm_json(prompt, mt, role):
            return self._llm_json(uuid, role, prompt, mt, log)

        def llm_extract(tl, mk):
            return metrics_mod.extract_text_metrics_llm(
                tl, mk, llm_json=_extract_llm_json
            )

        digest_now["primary_values"] = metrics_mod.resolve_primary_values(
            digest_now, primary_metrics, llm_extract=llm_extract
        )
        digest_now["primary_values_version"] = current_recon_version
        n_res = n_amb = n_miss = 0
        for p in preds:
            dg = p.get("digest")
            if not isinstance(dg, dict) or not dg:
                continue
            if dg.get("primary_values_version") != current_recon_version:
                dg["primary_values"] = metrics_mod.resolve_primary_values(
                    dg, primary_metrics, llm_extract=llm_extract
                )
                dg["primary_values_version"] = current_recon_version
        _pv_summary_parts = []
        for _mid, _r in digest_now["primary_values"].items():
            if _r["status"] == "resolved":
                n_res += 1
                _pv_summary_parts.append(
                    f"{_mid}={_r['value']:g} [{_r['source']}]"
                )
            elif _r["status"] == "ambiguous":
                n_amb += 1
            else:
                n_miss += 1
        log.stage(
            "METRICS",
            (
                f"resolved={n_res} ambiguous={n_amb} missing={n_miss} "
                f"({' '.join(_pv_summary_parts) if _pv_summary_parts else 'none'})"
            ),
        )
        log.payload(
            "METRICS",
            "primary metric ledger logged (current generation)",
            digest_now["primary_values"],
        )

        # ---- stage 4: policy (write -> screen -> smoke -> repair) ----
        # Composite sha: prompt template hash + recon config hash, so a
        # config change from the refresh naturally invalidates the policy
        # and verdict caches without touching the decision ladder.
        # (current_recon_version comes from the metric-ledger block above.)
        effective_policy_sha = (
            policy_mod.POLICY_SHA
            + ("_" + current_recon_version[:8] if current_recon_version else "")
        )
        policy_rec = registry.get_policy(effective_policy_sha)
        policy_src: str | None = None
        policy_meta: dict[str, Any] = {}
        # P0.3/P0.4: the smoke covers EVERY distinct digest shape of the
        # task (current + all predecessors, capped), and a CACHE HIT is
        # re-validated against it every generation — a gen-1 policy that
        # only ever self-compared must re-smoke when gen-2's shape arrives.
        smoke_digests = [digest_now] + [
            g["digest"]
            for g in preds
            if isinstance(g.get("digest"), dict) and g["digest"]
        ]
        smoke_digests = smoke_digests[: policy_mod.SMOKE_MAX_DIGESTS]
        resmoke_failed = False
        if policy_rec is not None:
            policy_src = policy_rec.get("source")
            policy_meta = policy_rec.get("meta") or {}
            ok, err = policy_mod.smoke_test(
                policy_src or "",
                smoke_digests,
                timeout=policy_mod.SMOKE_TIMEOUT_S,
                max_pairs=policy_mod.SMOKE_MAX_PAIRS_CACHE_HIT,
            )
            if ok:
                log.stage(
                    "POLICY",
                    f"cache=hit sha=..{effective_policy_sha[-4:]} "
                    f"resmoke=pass",
                )
                log.payload(
                    "POLICY",
                    f"cached policy source logged "
                    f"({len(policy_src or '')} chars, "
                    f"sha=..{effective_policy_sha[-4:]})",
                    {
                        "sha": effective_policy_sha,
                        "source": policy_src or "",
                        "meta": policy_meta,
                    },
                )
            else:
                # P0.4: a cached policy that crashes on a NEW digest shape
                # is never used — fall to the rewrite path.
                resmoke_failed = True
                log.warn(
                    f"cached policy failed the re-smoke on a new digest "
                    f"shape ({str(err)[:200]}); rewriting the policy"
                )
                policy_src = None
        if policy_rec is None or resmoke_failed:
            # Fresh write (first time) or rewrite after recon refresh /
            # re-smoke failure — same V3_PROMPT ladder, updated knowledge.
            action = (
                "rewrite" if (recon_refreshed or resmoke_failed) else "fresh"
            )
            llm_json = lambda prompt, mt, role: self._llm_json(  # noqa: E731
                uuid, role, prompt, mt, log
            )
            policy_src, policy_meta = policy_mod.ensure_policy(
                goal,
                recon_cfg or {},
                smoke_digests,
                llm_json,
                repair_rounds=self.repair_rounds,
                smoke_timeout_s=policy_mod.SMOKE_TIMEOUT_S,
                log=log,
            )
            if policy_src is not None:
                if pending_recon is not None:
                    # N6: the rewrite passed — NOW the refreshed recon is
                    # committed (the policy record below is keyed to its
                    # version, so the two commit together).
                    registry.set_recon(
                        pending_recon[0],
                        recon_mod.RECON_PROMPT_VERSION,
                        recon_version=pending_recon[1],
                    )
                registry.set_policy(
                    {
                        "sha": effective_policy_sha,
                        "source": policy_src,
                        "meta": {**policy_meta, "recon_version":
                                 current_recon_version},
                    }
                )
                log.stage(
                    "POLICY",
                    (
                        f"cache={action} sha=..{effective_policy_sha[-4:]} "
                        f"(attempts={policy_meta.get('attempts')} "
                        f"repairs={policy_meta.get('repairs')} "
                        f"escalations={policy_meta.get('escalations')} "
                        f"smoke={policy_meta.get('smoke')})"
                    ),
                )
                log.payload(
                    "POLICY",
                    (
                        f"generated policy source logged ({len(policy_src)} "
                        f"chars, sha=..{effective_policy_sha[-4:]})"
                    ),
                    {
                        "sha": effective_policy_sha,
                        "source": policy_src,
                        "meta": {**policy_meta,
                                 "recon_version": current_recon_version},
                    },
                )
            else:
                if pending_recon is not None:
                    # N6: rewrite rejected — keep the OLD recon + OLD
                    # policy (when one exists) instead of stranding the
                    # task with no policy under the new sha.
                    pending_recon = None
                    recon_refreshed = False
                    recon_cfg = registry.get_recon(
                        recon_mod.RECON_PROMPT_VERSION
                    )
                    old_version = registry.get_recon_version()
                    old_sha = (
                        policy_mod.POLICY_SHA
                        + ("_" + old_version[:8] if old_version else "")
                    )
                    old_rec = registry.get_policy(old_sha)
                    if old_rec is not None:
                        policy_src = old_rec.get("source")
                        policy_meta = old_rec.get("meta") or {}
                        effective_policy_sha = old_sha
                        log.stage(
                            "POLICY",
                            (
                                f"cache=refresh-rejected sha=..{old_sha[-4:]} "
                                f"(rewrite failed; old recon + old policy "
                                f"kept)"
                            ),
                        )
                    else:
                        log.stage(
                            "POLICY",
                            "cache=refresh-rejected (no old policy either)",
                        )
                log.warn(
                    f"policy {action} FAILED after repairs "
                    f"({policy_meta.get('last_error')}); pairs degrade to "
                    f"the deterministic crash rung + ties"
                )
                log.stage("POLICY", f"cache={action} FAILED", smoke="fail")

        # ---- channel selection: figure-dominant -> kimi (E26 rule) ----
        pooled: list[dict[str, Any]] = []
        for g in preds:
            pooled.extend(g["digest"].get("deliverables") or [])
        pooled.extend(digest_now.get("deliverables") or [])
        figure_task = digest_mod.is_figure_dominant({"deliverables": pooled})

        # ---- stage 6: pair verdicts vs every predecessor ----
        visual = self._make_visual_channel(uuid, goal, registry, log) if (
            figure_task and self.visual_channel
        ) else None
        if figure_task and self.visual_channel and visual is None:
            log.warn(
                "figure-dominant task but no vision_judge_model configured; "
                "the kimi pair channel is disabled — the code policy decides"
            )
        vis_cap = self.visual_max_pairs_per_gen  # 0 = unlimited
        fresh_visual_pairs = 0
        pair_records: list[dict[str, Any]] = []
        outcomes: list[str] = []
        n_cached = n_decisive = n_tie = n_disagree = n_visual = 0
        n_errors = n_weak = 0
        vis_fresh = vis_cached = 0
        for g in preds:
            other = g["uuid"]
            allow_visual = visual is not None and (
                vis_cap == 0 or fresh_visual_pairs < vis_cap
            )
            verdict = self._pair_verdict(
                uuid,
                goal,
                digest_now,
                ledger,
                current_ws,
                runner,
                g,
                policy_src,
                effective_policy_sha,
                visual,
                allow_visual,
                registry,
                log,
            )
            was_cached = bool(verdict.get("_cached"))
            if verdict.pop("_fresh_visual_pair", False):
                fresh_visual_pairs += 1
                vis_fresh += 1
            if verdict.pop("_cached_visual", False):
                vis_cached += 1
            if verdict.pop("_visual", False):
                n_visual += 1
            if verdict.pop("_cached", False):
                n_cached += 1
            winner = verdict.get("winner")
            if winner not in ("tie", uuid, other):
                winner = "tie"
                verdict["winner"] = "tie"
            if winner == "tie":
                outcomes.append("tie")
                n_tie += 1
                feedback = verdict.get("feedback_looser") or ""
            elif winner == uuid:
                outcomes.append("win")
                n_decisive += 1
                feedback = verdict.get("feedback_winner") or ""
            else:
                outcomes.append("loss")
                # N1 (2026-10-08 audit): the loss branch used to set
                # neither counter nor feedback — a loss-FIRST pair raised
                # UnboundLocalError out of evaluate(), later losses reused
                # the previous pair's feedback, and the weak-evidence cap
                # compared n_weak (counted losses) against an
                # undercounted n_decisive.
                n_decisive += 1
                feedback = verdict.get("feedback_looser") or ""
            if verdict.get("error"):
                n_errors += 1
            elif outcomes[-1] != "tie" and _decisive_weak_evidence(
                verdict, primary_metrics
            ):
                n_weak += 1
            if verdict.get("antisymmetry") == "disagreement":
                n_disagree += 1
            pair_records.append(
                {
                    "prev_uuid": other,
                    "outcome": {"win": "won", "loss": "lost", "tie": "tie"}[
                        outcomes[-1]
                    ],
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
            )
            log.detail(
                f"(gen {uuid[-8:]} vs {other[-8:]}) winner={winner[-8:] if winner != 'tie' else 'tie'} "
                f"rung={verdict.get('channel')} deciding={verdict.get('deciding')} "
                f"feedback-head={str(feedback)[:80]}"
            )
            log.payload(
                "PAIR",
                (
                    f"vs {other} winner="
                    f"{winner if winner == 'tie' else winner[-8:]} "
                    f"channel={verdict.get('channel')} "
                    f"deciding={verdict.get('deciding')}"
                ),
                {
                    "uuid_a": uuid,
                    "uuid_b": other,
                    "winner": winner,
                    "outcome": outcomes[-1],
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
        if visual is not None:
            log.stage(
                "VISUAL",
                (
                    f"kimi pairs={n_visual} fresh={vis_fresh} "
                    f"cached={vis_cached} decisive="
                    f"{sum(1 for p in pair_records if p['outcome'] != 'tie' and p['channel'] == 'kimi_pair')} "
                    f"mirror={'ok' if n_disagree == 0 else f'{n_disagree} disagreements'}"
                ),
            )
        antisym = "ok" if n_disagree == 0 else f"{n_disagree}-disagreements"
        # ---- schema-error-triggered recon refresh (replaces milestones) ----
        # If the policy signalled SchemaError on any pair, refresh the recon
        # with ALL digests (including the schema-breaking one), rewrite the
        # policy, and re-evaluate the failed pairs.
        # Also check the raw verdicts for the schema_error field
        schema_error_uuids = set()
        for g in preds:
            vkey = f"{effective_policy_sha}|{uuid}|{g['uuid']}"
            v = registry.get_verdict(vkey)
            if v and v.get("schema_error"):
                schema_error_uuids.add(g["uuid"])
        if schema_error_uuids and self._gens_since_refresh >= self.recon_refresh_gap:
            log.warn(
                f"SchemaError from policy on {len(schema_error_uuids)} pairs — "
                f"triggering recon refresh + policy rewrite"
            )
            all_digests = [g2["digest"] for g2 in registry.generations
                           if g2.get("digest")][-_RECON_MAX_DIGESTS:]
            if all_digests:
                prompt = recon_mod.build_prompt(goal, all_digests)
                parsed = self._llm_json(
                    uuid, "recon_refresh", prompt, 4000, log
                )
                new_cfg = recon_mod.ensure_shape(parsed)
                if new_cfg:
                    new_version = self._config_hash(new_cfg)
                    registry.set_recon(
                        new_cfg, recon_mod.RECON_PROMPT_VERSION,
                        recon_version=new_version,
                    )
                    current_recon_version = new_version
                    effective_policy_sha = (
                        policy_mod.POLICY_SHA + "_" + new_version[:8]
                    )
                    # Rewrite the policy with the new config
                    smoke_digests = [digest_now] + [g2["digest"] for g2 in preds[:2]]
                    llm_json = lambda prompt2, mt, role: self._llm_json(  # noqa: E731
                        uuid, role, prompt2, mt, log
                    )
                    new_policy_src, new_meta = policy_mod.ensure_policy(
                        goal, new_cfg, smoke_digests, llm_json,
                        repair_rounds=self.repair_rounds,
                        smoke_timeout_s=self.scorer_timeout_s, log=log,
                    )
                    if new_policy_src:
                        registry.set_policy({
                            "sha": effective_policy_sha,
                            "source": new_policy_src,
                            "meta": new_meta,
                        })
                        policy_src = new_policy_src
                        self._gens_since_refresh = 0
                        log.stage(
                            "RECON",
                            f"cache=schema-refreshed version {new_version[:6]} "
                            f"policy rewritten ({len(schema_error_uuids)} pairs to re-evaluate)",
                        )
                        # Re-evaluate the schema-error pairs
                        for rec in pair_records:
                            other_uuid = rec.get("prev_uuid") or rec.get("other")
                            if other_uuid in schema_error_uuids:
                                # Invalidate the cached verdict
                                # Re-run with the new policy
                                for g2 in preds:
                                    if g2["uuid"] == other_uuid:
                                        new_verdict = self._pair_verdict(
                                            uuid, goal, digest_now, ledger,
                                            current_ws, runner, g2,
                                            policy_src, effective_policy_sha,
                                            visual, True, registry, log,
                                        )
                                        if "winner" in new_verdict:
                                            rec["outcome"] = new_verdict["winner"] if new_verdict["winner"] == uuid else (
                                                "lost" if new_verdict["winner"] != "tie" else "tie"
                                            )
                                            rec["feedback"] = new_verdict.get("feedback_looser", "")
                                        break
                        # Recompute reward data from updated pair_records
                        outcomes = [rec.get("outcome", "tie") for rec in pair_records]
                        reward_data = {
                            "wins": outcomes.count("won"),
                            "losses": outcomes.count("lost"),
                            "ties": outcomes.count("tie"),
                        }
                    else:
                        log.warn("schema-refresh policy rewrite FAILED; keeping old")
                else:
                    log.warn("schema-refresh recon unparseable; keeping old")
        self._gens_since_refresh += 1

        log.stage(
            "PAIRS",
            (
                f"predecessors={len(preds)} cached={n_cached} "
                f"decisive={n_decisive} tie={n_tie} errors={n_errors} "
                f"weak={n_weak} antisym={antisym}"
            ),
            excluded=len(excluded),
        )

        # ---- stage 7: reward (arrival-moment; never rewritten) ----
        reward_data = verdicts_mod.win_rate_reward(outcomes)
        reward = float(reward_data["reward"])
        bt_data: dict[str, Any] | None = None
        if self.reward_mode == "bradley_terry" and preds:
            prefix = [g["uuid"] for g in registry.previous_generations(uuid)] + [uuid]
            bt_data = verdicts_mod.bradley_terry_reward(
                prefix,
                [
                    {"a": uuid, "b": p["prev_uuid"], "winner": p["winner"]}
                    for p in pair_records
                ],
            )
            ratings = bt_data["ratings"]
            mean_r = sum(ratings.values()) / max(1, len(ratings))
            reward = ratings.get(uuid, mean_r)
        # P0.7 weak-evidence guard: when EVERY decisive pair rested on
        # self-reported metric evidence the policy cannot ground in the
        # recon primary metrics, the reward is capped so a spurious
        # 1.0 can never trip the learned early-stop threshold.
        reward_weak_evidence = bool(pair_records) and n_decisive > 0 and (
            n_weak == n_decisive
        )
        if reward_weak_evidence:
            reward = min(reward, WEAK_EVIDENCE_REWARD_CAP)
            log.warn(
                f"weak-evidence guard: all {n_decisive} decisive pairs rest "
                f"on self-reported metrics outside the recon primary "
                f"metrics; reward capped at {WEAK_EVIDENCE_REWARD_CAP}"
            )
        log.stage(
            "REWARD",
            (
                f"mode={self.reward_mode} value={reward:.3f} "
                f"({reward_data['wins']}W/{reward_data['losses']}L/"
                f"{reward_data['ties']}T vs predecessors) "
                f"weak={'yes' if reward_weak_evidence else 'no'}"
            ),
        )

        # ---- persist (N7/P2.1): generation + verdicts RIGHT after the
        # reward — every later stage (gradient fold, refiner, artifacts)
        # is optional and must never lose them ----
        reward_capped = round(max(0.0, min(1.0, reward)), 4)
        stats = {
            "wins": reward_data["wins"],
            "losses": reward_data["losses"],
            "ties": reward_data["ties"],
        }
        registry.record_generation(
            uuid, digest_now, ledger,
            reward=reward_capped, stats=stats, session=self._session_tag,
        )
        registry.save()


        # ---- stage 8: gradient fold + optional refiner ----
        # §4.2: the fold renders the METRICS section from the LEDGER
        # (deterministic, data-side), never regex-parsed from the
        # LLM-written feedback; it returns what it appended so the log
        # reports the real condition (S5).
        coverage = recon_mod.coverage_footer(recon_cfg)
        fold_ledger = {
            "current": digest_now.get("primary_values") or {},
            "preds": {
                p["uuid"]: (p.get("digest") or {}).get("primary_values")
                or {}
                for p in preds
            },
            "metrics": primary_metrics,
        }
        raw_gradient, fold_meta = gradient_mod.fold_gradient(
            uuid, pair_records, coverage, ledger=fold_ledger
        )
        log.payload(
            "GRADIENT_RAW",
            f"raw gradient logged ({len(raw_gradient)} chars, "
            f"{reward_data['wins']}W/{reward_data['losses']}L/"
            f"{reward_data['ties']}T)",
            {
                "gradient": raw_gradient,
                "coverage_footer_used": coverage,
                "coverage_footer_appended": fold_meta["footer_appended"],
                "fold_tied": fold_meta["tied"],
                "fold_not_assessed": fold_meta["not_assessed"],
            },
        )
        refinement = None
        if self.refiner:
            preview = gradient_mod.build_preview(current_ws, digest_now, ledger)
            llm_json = lambda prompt, mt, role: self._llm_json(  # noqa: E731
                uuid, role, prompt, mt, log
            )
            try:
                refinement = gradient_mod.refine_gradient(
                    llm_json, uuid, raw_gradient, preview
                )
            except Exception as e:  # noqa: BLE001 — N7: refiner failure
                # degrades to the raw fold; it must never abort evaluate()
                # after the reward and verdicts were already computed.
                log.warn(
                    f"refiner failed ({type(e).__name__}: {str(e)[:200]}); "
                    f"using the raw gradient fold"
                )
                refinement = None
        gradient_text = gradient_mod.refined_text(raw_gradient, refinement)
        rstats = gradient_mod.refiner_stats(refinement)
        log.stage(
            "GRADIENT",
            (
                f"fold: {reward_data['losses']}L/{reward_data['wins']}W/"
                f"{reward_data['ties']}T refiner={'on' if self.refiner else 'off'} "
                f"issues={rstats['issues']} missing={rstats['missing']} "
                f"corr={rstats['corr']}"
            ),
        )
        log.payload(
            "GRADIENT_REFINED",
            (
                f"final gradient logged ({len(gradient_text)} chars, "
                f"refiner={'on' if self.refiner else 'off'}, "
                f"issues={rstats['issues']} missing={rstats['missing']} "
                f"corr={rstats['corr']})"
            ),
            {
                "gradient": gradient_text,
                "refinement": refinement,
                "refiner_used": bool(refinement is not None),
            },
        )

        # ---- scores ----
        scores = {
            "overall_score": reward_capped,
            "win_rate": round(float(reward_data["reward"]), 4),
            "reward_mode": self.reward_mode,
            "reward_weak_evidence": reward_weak_evidence,
            "n_weak_evidence_pairs": n_weak,
            "n_pair_errors": n_errors,
            "n_pairs": reward_data["n_pairs"],
            "n_wins": reward_data["wins"],
            "n_losses": reward_data["losses"],
            "n_ties": reward_data["ties"],
            "n_predecessors": len(preds),
            "n_excluded_predecessors": len(excluded),
            # Engine contract (stop plan 2026-10-08 §2.5): normalised
            # head-to-head outcomes of THIS generation vs each
            # predecessor; the engine's StopController reads only this.
            "pair_outcomes": {
                p["prev_uuid"]: p["outcome"] for p in pair_records
            },
            "figure_task": figure_task,
            "visual_pairs": n_visual,
            "policy_sha": effective_policy_sha,
            "policy_attempts": policy_meta.get("attempts"),
            "policy_repairs": policy_meta.get("repairs"),
            "policy_smoke": policy_meta.get("smoke"),
            # Compact metric ledger summary (review §4.3): the frontier
            # probe reads this to detect all-tie objective metrics —
            # the verifier's measured values, nothing synthetic.
            "metric_summary": {
                mid: {"status": r.get("status"), "value": r.get("value")}
                for mid, r in (digest_now.get("primary_values") or {}).items()
            },
            "reexec": {
                "exit_code": ledger.get("exit_code"),
                "wall_time_s": ledger.get("wall_time_s"),
                "timed_out": ledger.get("timed_out"),
                "source_manifest_intact": ledger.get("source_manifest_intact"),
                "n_fresh_artifacts": len(ledger.get("fresh_artifacts") or {}),
            },
            "bt_pick": (bt_data or {}).get("pick"),
            "bt_ratings": (bt_data or {}).get("ratings"),
            "refiner_issues": rstats["issues"],
            "refiner_missing_files": rstats["missing"],
            "refiner_corrections": rstats["corr"],
            "llm_calls": log.n_llm_calls,
            "llm_spend_usd": round(log.spend_usd, 6),
            "abstracted_textual_gradient": gradient_text,
            # Legacy-typo key kept for engine compatibility (the hybrid
            # writes both; the engine reads one).
            "abstractec_textual_gradient": gradient_text,
            "failure_fingerprint": {
                "vector": [0.0] * _FP_DIM,
                "presence_mask": [0.0] * _FP_DIM,
                "pass_rates": [0.5] * _FP_DIM,
            },
        }
        self._write_artifacts(
            uuid, scores, pair_records, raw_gradient, gradient_text
        )
        try:
            self._save_results(scores, uuid, "verifier")
        except Exception as e:  # noqa: BLE001 — persistence failure logged only
            self.logger.error(
                f"Failed to persist policy_bt scores for {uuid}: {e}"
            )
        wall = time.time() - t_total
        log.stage(
            "DONE",
            (
                f"reward={reward_capped} total_llm={log.spend_line()} "
                f"wall={wall:.1f}s"
            ),
        )
        print_box(
            f"reward {reward_capped:.3f} | {reward_data['wins']}W/"
            f"{reward_data['losses']}L/{reward_data['ties']}T vs "
            f"{reward_data['n_pairs']} previous | llm {log.spend_line()} | "
            f"{wall:.1f}s",
            title=f"Policy-BT verifier · {uuid}",
            color=CYAN,
        )
        return {"uuid": uuid, "pairs": pair_records, **scores}

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
        p_digest = pred["digest"]
        p_ledger = pred.get("ledger") or {}

        # Deterministic exec-crash rung FIRST (safeguard 8), both channels.
        rung = verdicts_mod.crash_rung(uuid, ledger, other, p_ledger)
        if rung is not None:
            return {**rung, "_cached": False}

        extra: dict[str, Any] = {}
        if visual is not None:
            pred_root = runner.run_copy_path(other)
            if not pred_root.exists():
                pred_root = find_run_snapshot(other)
            if pred_root is None:
                # P3.3: NO live-workspace fallback — the live workspace
                # holds THIS generation's files; judging predecessor
                # figures from it compares the present against itself.
                # Skip the visual ladder for this pair entirely.
                pred_root = None
                log.warn(
                    f"no sandbox copy or /tmp snapshot for predecessor "
                    f"{other[-8:]}; skipping the kimi channel for this "
                    f"pair — the code policy decides"
                )
            if pred_root is not None:
                key = f"vis|{visual_mod.VISUAL_PROMPT_VERSION}|{uuid}|{other}"
                cached = registry.get_verdict(key)
                figs_now = visual_mod.select_figures(digest_now, current_ws)
                figs_pred = visual_mod.select_figures(p_digest, pred_root)
                # P3.4: the deliverable rung decides on DIGEST figure
                # presence; the 6 MB cap only filters uploads.
                frung = visual_mod.figure_rung(
                    uuid,
                    len(digest_now.get("figures") or []),
                    other,
                    len(p_digest.get("figures") or []),
                )
                if frung is not None:
                    registry.set_verdict(key, frung)
                    return {**frung, "_cached": False, "_visual": True}
                if figs_now and figs_pred:
                    if cached is not None:
                        return {**cached, "_cached": True, "_visual": True,
                                "_cached_visual": True}
                    if allow_visual:
                        verdict = visual.judge_pair(
                            goal, uuid, figs_now, other, figs_pred
                        )
                        if verdict is not None and not (
                            self.visual_tiebreak_code
                            and verdict.get("winner") == "tie"
                        ):
                            registry.set_verdict(key, verdict)
                            return {
                                **verdict,
                                "_cached": False,
                                "_visual": True,
                                "_fresh_visual_pair": True,
                            }
                        if verdict is not None:
                            # P3.5: visual_tiebreak_code=on — a kimi tie
                            # defers to the code policy (the fresh visual
                            # call still consumed budget).
                            extra = {"_fresh_visual_pair": True}
                            log.detail(
                                f"kimi tie on pair ({uuid[-8:]} vs "
                                f"{other[-8:]}); visual_tiebreak_code=on — "
                                f"the code policy decides"
                            )
                    else:
                        log.warn(
                            f"visual pair budget cap "
                            f"({self.visual_max_pairs_per_gen}/gen) tripped "
                            f"on pair ({uuid[-8:]} vs {other[-8:]}); "
                            f"falling to the code-policy path"
                        )
                # Both sides lack figures -> the code policy decides.

        # Code-policy path (both orders, antisymmetry asserted).
        if policy_src is None:
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
        key = code_verdict_key(policy_sha, uuid, other)
        cached = registry.get_verdict(key)
        if cached is not None:
            return {**cached, "_cached": True, **extra}
        # P2.10: the pair run honours policy_bt_scorer_timeout_s.
        verdict = verdicts_mod.code_pair_verdict(
            functools.partial(
                policy_mod.run_policy, timeout=self.scorer_timeout_s
            ),
            policy_src,
            uuid,
            digest_now,
            other,
            p_digest,
            log=log,
        )
        # P0.5: an ERRORED pair verdict is NEVER cached — a policy that
        # errors once must be able to produce a decisive verdict on the
        # next evaluation instead of a permanent cached tie.
        if "error" not in verdict:
            registry.set_verdict(key, verdict)
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
        scores = {
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
        self._write_artifacts(uuid, scores, [], gradient_text, gradient_text)
        try:
            self._save_results(scores, uuid, "verifier")
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to persist short-circuit scores: {e}")
        return {"uuid": uuid, "pairs": [], **scores}

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
            f"reward: {scores.get('overall_score')} "
            f"(mode={scores.get('reward_mode')})",
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
        for p in pair_records:
            lines.append(
                f"- vs {p['prev_uuid']}: {p['outcome']} via {p['channel']} "
                f"(deciding={p.get('deciding')})"
            )
        lines += ["", "RAW GRADIENT (fold):", raw_gradient]
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "evaluation.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )
            (out_dir / "textual_gradient.txt").write_text(
                gradient_text, encoding="utf-8"
            )
            print_ok(f"[policy_bt {uuid}] evaluation.txt + textual_gradient.txt written")
        except OSError as e:
            self.logger.error(f"Could not write artifacts for {uuid}: {e}")

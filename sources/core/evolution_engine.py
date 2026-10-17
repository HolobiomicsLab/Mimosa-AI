"""Layer 2: the evolution loop that improves multi-agent workflows.

This module owns the per-goal evolution session of Mimosa-AI. Each
generation runs a candidate workflow in the Layer 3 sandbox and scores
it with the Layer 4 verifier. It then refreshes the quality-diversity
archive and asks the variation engine for the next mutation, crossover,
or seed prompt. Module helpers also decide the frontier exploration
probe and pick the final capsule workspace. The planner, the benchmark
CSV runner, and the CLI import EvolutionEngine and call
start_workflow_evolution once per goal.
"""

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sources.benchmark_evaluation.scenario_loader import ScenarioLoader
from sources.cli.pretty_print import (
    CYAN,
    GREEN,
    YELLOW,
    print_box,
    print_err,
    print_info,
    print_iteration_header,
    print_ok,
    print_phase,
    print_section,
    print_summary,
    print_warn,
)
from sources.evaluators.evaluator import WorkflowEvaluator
from sources.utils.notify import PushNotifier
from sources.utils.pricing import PricingCalculator
from sources.utils.run_metrics import append_jsonl, write_run_metrics
from sources.utils.visualization import VisualizationUtils
from sources.utils.workspace_management import WorkspaceManager

from .lineage import record_lineage
from .orchestrator import WorkflowOrchestrator
from .schema import IndividualRun, SelectionLog
from .selection import SelectionPressure
from .stopping import StopController
from .variation_engine import VariationEngine, exploration_state
from .workflow_info import FALLBACK_REWARD_KINDS, WorkflowInfo
from .workflow_selection import WorkflowSelector

if TYPE_CHECKING:
    from config import Config


def _scalar(value: Any) -> float | None:
    """Normalize loose numeric fields for the metrics snapshots.

    Snapshot builders call this with values read from run records and
    verdict dicts, where scores may be ``None``, strings, or NaN. It
    converts the value to ``float`` and rejects missing or non-finite
    data.

    Args:
        value (Any): Raw score from a run record or a QD verdict.

    Returns:
        float | None: The finite value, or ``None`` for missing,
        non-numeric, or NaN input.
    """
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # filter NaN


def _to_jsonable(obj: Any) -> Any:
    """Make a metric value safe for a JSONL row.

    Persistence builders call this with values that may be dataclasses
    such as ``SelectionLog``. It unwraps dataclasses to plain dicts and
    passes everything else through unchanged.

    Args:
        obj (Any): Metric value from a run record; may be ``None``.

    Returns:
        Any: ``asdict(obj)`` for a dataclass, ``None`` for ``None``
        input, and the value unchanged otherwise.
    """
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    return obj

def _verifier_block(run: "IndividualRun") -> dict:
    """Read the run's persisted verifier scores.

    Reward refresh and capsule-selection helpers call this with an
    :class:`IndividualRun` completed by the Layer 4 evaluator. It
    returns the ``state_result["evaluation"]["verifier"]`` block that
    the evaluator wrote.

    Args:
        run (IndividualRun): Completed run holding the evaluation
            payload.

    Returns:
        dict: The verifier block, or an empty dict when the run has no
        state, no evaluation, or a non-dict block.
    """
    evaluation = (run.state_result or {}).get("evaluation") or {}
    block = evaluation.get("verifier")
    return block if isinstance(block, dict) else {}


def _refresh_policy_bt_rewards(
    runs: list["IndividualRun"], selection: SelectionPressure,
) -> None:
    """Give survivor selection the latest tournament ratings.

    The engine calls this after each evaluation, with the session runs
    and the archive they compete in. When the newest verifier block is a
    policy-BT result with tournament rewards, it copies each run's fresh
    Bradley-Terry rating and current pair outcomes into the run record and
    the rating into the archive; arrival scores stay in the verifier
    record. Any other verifier block is left untouched.

    Args:
        runs (list[IndividualRun]): Completed runs in arrival order.
        selection (SelectionPressure): Current session archive to update.

    Returns:
        None. Run records and matching archive members are updated.
    """
    latest = _verifier_block(runs[-1]) if runs else {}
    if latest.get("verifier_kind") != "policy_bt" or "tournament_rewards" not in latest:
        return
    rewards, outcomes = latest["tournament_rewards"], latest.get("tournament_outcomes") or {}
    for run in runs:
        if run.current_uuid not in rewards:
            continue
        run.state_result = run.state_result or {}
        block = run.state_result.setdefault("evaluation", {}).setdefault("verifier", {})
        block.setdefault("arrival_reward", run.reward)
        run.reward = run.reward_uncapped = rewards[run.current_uuid]
        run.reward_is_fallback = False
        block.update(
            verifier_kind="policy_bt", reward=run.reward, overall_score=run.reward,
            overall_score_uncapped=run.reward, reward_fallback=False,
            tournament_rewards=rewards, tournament_version=latest["tournament_version"],
            pair_outcomes=outcomes.get(run.current_uuid, block.get("pair_outcomes") or {}),
        )
    selection.refresh_rewards(rewards)


def _stop_pair_outcomes(verifier: dict) -> dict[str, str] | None:
    """Translate policy tournament outcomes for the stop controller.

    The engine calls this once per generation with the run's persisted
    verifier block, to feed ``StopController.observe``. Policy-BT labels
    are mapped to controller labels; other verifier channels pass their
    pair-outcome dict through unchanged.

    Args:
        verifier (dict): Persisted ``evaluation.verifier`` result of the
            run that was just scored.

    Returns:
        dict[str, str] | None: Predecessor UUIDs mapped to controller
        outcomes (``won``/``lost``/``tie``), or ``None`` when the block
        records no pair outcomes.
    """
    outcomes = verifier.get("pair_outcomes")
    if verifier.get("verifier_kind") != "policy_bt" or outcomes is None:
        return outcomes
    labels = {"win": "won", "loss": "lost", "tie": "tie"}
    return {uuid: labels[outcome] for uuid, outcome in outcomes.items()}


#: Frontier probe (review 2026-10-09 §4.3): how many recent generations
#: must report identical measured metrics for the all-tie trigger.
_EXPLORE_TIE_WINDOW = 2

#: Seed for the base-rate RNG. Fixed (not run-dependent) so identical
#: configs replay identically; the base rate defaults to 0 anyway.
_EXPLORE_RNG_SEED = 20261009

#: Verifier outcomes that carry no reward and must not enter the QD archive.
_UNSCORED_VERIFIER_REASONS = frozenset(
    {"workflow_generation_or_execution_failed", "verifier_instrument_error"}
)


def _objective_tie_window(
    runs: list["IndividualRun"], window: int,
) -> dict | None:
    """Detect identical measured metrics across the last ``window`` runs.

    The frontier probe calls this as its all-tie trigger (review
    2026-10-09 §4.3).

    Reads each run's ``evaluation.verifier.metric_summary`` — the compact
    ledger summary the verifier persists (id → status/value). The window
    counts only runs that carry a verifier block, so cold-start entries
    never dilute it.

    Args:
        runs (list[IndividualRun]): Completed runs, oldest first (last
            element = latest).
        window (int): Number of recent generations that must agree.

    Returns:
        ``{"n_pairs": int, "metrics": [{"id", "value"}, ...]}`` when the
        latest generation compared at least one pair AND every metric of
        its summary is resolved with the same value in every generation
        of the window; ``None`` otherwise (never guesses).
    """
    if window < 1:
        return None
    blocks: list[dict] = []
    for run in reversed(runs):
        block = _verifier_block(run)
        if block.get("metric_summary") or block.get("pair_outcomes"):
            blocks.append(block)
            if len(blocks) >= window:
                break
    if len(blocks) < window:
        return None
    latest = blocks[0]
    pairs = latest.get("pair_outcomes") or {}
    if not pairs:
        return None  # no comparison happened (defect C9: gen 1 has 0 pairs)
    summary = latest.get("metric_summary") or {}
    metrics: list[dict] = []
    for mid, entry in summary.items():
        entry = entry if isinstance(entry, dict) else {}
        if str(entry.get("status") or "") != "resolved":
            return None
        value = entry.get("value")
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        for other in blocks[1:]:
            prev = other.get("metric_summary") or {}
            prev_entry = prev.get(mid)
            prev_entry = prev_entry if isinstance(prev_entry, dict) else {}
            if (
                str(prev_entry.get("status") or "") != "resolved"
                or prev_entry.get("value") != value
            ):
                return None
        metrics.append({"id": mid, "value": value})
    if not metrics:
        return None
    return {"n_pairs": len(pairs), "metrics": metrics}


def evaluate_exploration_probe(
    controller: StopController,
    runs: list["IndividualRun"],
    *,
    explore_after: int | None = None,
    base_rate: float = 0.0,
    rng: random.Random | None = None,
    tie_window: int = _EXPLORE_TIE_WINDOW,
) -> dict:
    """Evaluate the frontier-probe trigger (review 2026-10-09 §4.3).

    The evolution engine calls this once per generation, after the stop
    observation and before the stop decision, to request wider search
    when learning stalls. The trigger is deterministic, state-based, and
    verifier-agnostic:

    - **plateau**: the champion is unchanged for ``>= explore_after``
      generations (default ``patience - 1``, guaranteeing at least one
      probe before the plateau stop);
    - **all-tie**: the last ``tie_window`` generations report identical
      resolved values for every measured metric (read from
      ``metric_summary``) — the verifier measured but cannot rank;
    - **base-rate** (optional): ``base_rate`` probability per generation,
      drawn from the engine's seeded RNG so runs replay identically.

    The probe is a search-policy request only: rewards, verdicts and the
    textual gradient are untouched, and the produced dict is what the
    variation engine renders as the labelled ``<exploration>`` block.

    Args:
        controller (StopController): Session stop controller holding the
            champion and ``since_change`` state.
        runs (list[IndividualRun]): Completed runs, oldest first, as
            collected by the engine.
        explore_after (int | None): Plateau threshold; ``None`` =
            ``patience - 1``.
        base_rate (float): Extra per-generation trigger probability in
            [0, 1], from config.
        rng (random.Random | None): Seeded RNG for the base rate;
            ``None`` skips that trigger.
        tie_window (int): Generations that must tie for the all-tie
            trigger.

    Returns:
        ``{"triggered": bool, "reason": str, "champion": str | None,
        "since_change": int, "n_pairs": int | None,
        "generations_compared": int | None, "metrics": [...]}``.
    """
    patience = max(1, int(getattr(controller, "patience", 3)))
    after = patience - 1 if explore_after is None else max(0, int(explore_after))
    base_rate = float(base_rate or 0.0)
    champion = controller.champion
    since_change = controller.since_change
    probe: dict[str, Any] = {
        "triggered": False,
        "reason": "",
        "champion": champion,
        "since_change": since_change,
        "n_pairs": None,
        "generations_compared": None,
        "metrics": [],
    }
    tie = _objective_tie_window(runs, tie_window)
    if tie is not None:
        probe["n_pairs"] = tie["n_pairs"]
        probe["generations_compared"] = tie_window
        probe["metrics"] = tie["metrics"]
    if champion is not None and since_change >= after:
        probe["triggered"] = True
        probe["reason"] = (
            f"plateau: champion {StopController._tag(champion)} "
            f"unchanged for {since_change} generations"
        )
    elif tie is not None:
        listed = ", ".join(f"{m['id']}={m['value']}" for m in tie["metrics"])
        probe["triggered"] = True
        probe["reason"] = (
            f"all-tie: last {tie_window} generations report identical "
            f"values for every measured metric ({listed})"
        )
    elif base_rate > 0.0 and rng is not None and rng.random() < base_rate:
        probe["triggered"] = True
        probe["reason"] = f"base-rate: scheduled exploration (p={base_rate:.2f})"
    return probe


def _run_reward_is_fallback(run: "IndividualRun", candidates: list["IndividualRun"]) -> bool:
    """Decide whether *run*'s reward is a mean-claim fallback (N1).

    Capsule selection calls this for every candidate run, with the full
    candidate list for cross-checks, before the final argmax. Three
    signals, in order of trust:

    Args:
        run (IndividualRun): Candidate run being judged.
        candidates (list[IndividualRun]): All candidate runs of the
            session, used for the sibling cross-check.

    Returns:
        bool: ``True`` when the reward is a fallback and must stay out
        of the capsule argmax; ``False`` for a real pairwise reward.
    """
    verifier = _verifier_block(run)
    if verifier.get("reward_fallback") in FALLBACK_REWARD_KINDS:
        if not run.reward_is_fallback:
            logging.getLogger(__name__).warning(
                "[SELECT] run %s reward_is_fallback was False but its "
                "state_result records reward_fallback=%s; using "
                "the persisted record (fallback excluded from the argmax)",
                run.current_uuid,
                verifier.get("reward_fallback"),
            )
        return True
    if run.reward_is_fallback:
        return True
    if verifier.get("reward_source") == "benchmark_grader":
        return False
    n_pairs = _scalar(verifier.get("n_pairs"))
    n_wins = _scalar(verifier.get("n_wins"))
    has_real_pairwise = any(
        (_scalar(_verifier_block(other).get("n_pairs")) or 0) > 0
        for other in candidates
        if other is not run
    )
    if (
        (run.reward or 0.0) > 0.0
        and (n_pairs or 0) == 0
        and (n_wins or 0) == 0
        and has_real_pairwise
    ):
        logging.getLogger(__name__).warning(
            "[SELECT] run %s carries reward %.3f with no pairwise "
            "comparisons recorded (n_pairs=0, n_wins=0) while sibling "
            "runs have real pairwise rewards; treating it as a potential "
            "fallback and excluding it from the argmax",
            run.current_uuid,
            run.reward or 0.0,
        )
        return True
    return False


# Ratings closer than this are the same rating.
RATING_TIE_TOLERANCE = 1e-9


def _head_to_head(run: IndividualRun, rival: IndividualRun) -> str | None:
    """Read the current direct pair outcome of one run against one rival.

    Capsule selection uses it to break a tie between top-rated runs.
    :func:`_policy_capsule_run` calls it for every two tied runs. It reads
    the run's ``pair_outcomes``, which :func:`_refresh_policy_bt_rewards`
    keeps current together with the ratings; the rival's mirrored entry is
    the fallback.

    Args:
        run (IndividualRun): Run whose outcome is requested.
        rival (IndividualRun): The other run of the pair.

    Returns:
        str | None: ``"win"``, ``"loss"`` or ``"tie"`` for *run*, or
        ``None`` when neither block records the pair.
    """
    mirror = {"win": "loss", "loss": "win", "tie": "tie"}
    own = (_verifier_block(run).get("pair_outcomes") or {}).get(rival.current_uuid)
    other = (_verifier_block(rival).get("pair_outcomes") or {}).get(run.current_uuid)
    return own or mirror.get(other)


def _policy_capsule_run(runs: list[IndividualRun]) -> IndividualRun:
    """Pick the top Bradley-Terry run, breaking exact rating ties fairly.

    Equal ratings come from equal win/tie records, not from equal quality,
    so the newest run must not win a tie (2026-10-17: drugex shipped a
    failing generation tied with a passing gen 0). :func:`_select_best_run`
    calls it with the policy-BT runs. It keeps the runs at the top rating,
    then prefers the run with the most head-to-head wins against the other
    tied runs, then the earliest run.

    Args:
        runs (list[IndividualRun]): Non-empty policy-BT runs with a uuid
            and a current rating in ``reward``.

    Returns:
        IndividualRun: The run that ships as the capsule.
    """
    top = max(r.reward or 0.0 for r in runs)
    tied = [r for r in runs if abs((r.reward or 0.0) - top) <= RATING_TIE_TOLERANCE]
    wins = {id(r): sum(_head_to_head(r, rival) == "win" for rival in tied if rival is not r) for r in tied}
    return min(tied, key=lambda r: (-wins[id(r)], r.iteration_count))


def _select_best_run(runs: list[IndividualRun]) -> IndividualRun | None:
    """Pick the run whose workspace ships as the task capsule.

    :func:`_select_capsule_run` calls this once at session end, with the
    completed session runs.

    Args:
        runs (list[IndividualRun]): Completed session runs; entries
            without a ``current_uuid`` are ignored.

    Returns:
        IndividualRun | None: The run whose workspace ships, or ``None``
        when no run carries a UUID.
    """
    candidates = [r for r in runs if r.current_uuid]
    if not candidates:
        return None
    key = lambda r: (r.reward if r.reward is not None else 0.0, r.iteration_count)  # noqa: E731
    policy_runs = [r for r in candidates if _verifier_block(r).get("verifier_kind") == "policy_bt"]
    if policy_runs:
        return _policy_capsule_run(policy_runs)
    pairwise = [r for r in candidates if not _run_reward_is_fallback(r, candidates)]

    if pairwise and all((r.reward or 0) == 0 for r in pairwise):
        measured_fallbacks = [
            r for r in candidates
            if _run_reward_is_fallback(r, candidates) and (r.reward or 0) > 0
        ]
        if measured_fallbacks:
            return max(measured_fallbacks, key=key)

    return max(pairwise or candidates, key=key)


def _select_capsule_run(
    runs: list[IndividualRun], controller: StopController, verifier_kind: str | None,
) -> tuple[IndividualRun | None, str]:
    """Choose the workspace to export after the evolution session.

    ``start_workflow_evolution`` calls this once after the loop, with the
    completed session runs and the stop controller, to restore and export
    the best workspace. Policy BT uses current fitted ratings; older
    relative verifiers keep their champion.

    Args:
        runs (list[IndividualRun]): Session runs with refreshed rewards.
        controller (StopController): Existing stopping state for this session.
        verifier_kind (str | None): Active evaluator channel.

    Returns:
        tuple[IndividualRun | None, str]: Selected run and selection reason.
    """
    best = _select_best_run(runs)
    if verifier_kind == "policy_bt" or (best and _verifier_block(best).get("verifier_kind") == "policy_bt"):
        return best, "argmax current Bradley-Terry rating"
    rule = "argmax arrival-moment reward (_select_best_run)"
    if controller.selects_champion and controller.champion:
        champion = next((r for r in runs if r.current_uuid == controller.champion), None)
        if champion is not None:
            return champion, f"final champion {controller.champion} (head-to-head)"
        print_warn(f"[STOP] champion {controller.champion} has no run in this session; falling back to {rule}")
    return best, rule


class EvolutionEngine:
    """Drive the evolution loop for one goal at a time.

    The planner, the benchmark CSV runner, and the CLI construct one
    engine per goal and call :meth:`start_workflow_evolution`. The
    engine wires the sandbox orchestrator, the verifier evaluator, the
    variation engine, and the quality-diversity selection pressure. In
    the loop, the last element of ``runs`` is always the attempt being
    executed; the list grows by one run per generation.
    """
    def __init__(
        self,
        config: "Config",
        viz_utils: VisualizationUtils | None = None,
        process_id: int | None = None,
    ) -> None:
        """Wire the evaluator, orchestrator, variation, and selection parts.

        Callers such as the planner and the benchmark runner construct the
        engine once per goal, with the loaded application config. It
        builds the collaborator objects but starts no session.

        Args:
            config (Config): Application config (config file) giving the
                workflow directory, pricing, and notification credentials.
            viz_utils (VisualizationUtils | None): Optional visualization
                helper from the caller; a fresh one is built when omitted.
            process_id (int | None): Optional caller-assigned id, used in
                logs.

        Returns:
            None. The engine only wires its collaborator objects.
        """
        self.config = config
        self.workflow_dir = config.workflow_dir
        self.model_pricing = config.model_pricing
        self.notifier = PushNotifier(config.pushover_token, config.pushover_user)
        self.viz_utils = viz_utils or VisualizationUtils()
        self.process_id = process_id
        self.pricing = PricingCalculator(config)
        self.logger = logging.getLogger(__name__)
        self.workflow_selector = WorkflowSelector(config)
        self.orchestrator = WorkflowOrchestrator(config)
        self.variation = VariationEngine(config)
        self.judge = WorkflowEvaluator(config)
        self.workspace_mgr = WorkspaceManager(self.config, self.logger)
        self.selection = SelectionPressure(
            config,
            min_improvement_threshold=getattr(config, "min_improvement_threshold", 0.01),
            strategy=getattr(config, "selection_strategy", "qd"),
            population_size=getattr(config, "population_size", 50), # max individuals to keep in the selection pool
            novelty_k_neighbours=getattr(config, "novelty_k_neighbours", 15),
            novelty_weight=getattr(config, "novelty_weight", 0.25),
            admit_threshold=getattr(config, "admit_threshold", 0.3),
            novelty_comparison=getattr(config, "novelty_comparison", "archive_knn"),
            previous_n=getattr(config, "novelty_previous_n", 5),
            length_penalty_baseline_chars=getattr(config, "length_penalty_baseline_chars", 5000),
            length_penalty_lambda=getattr(config, "length_penalty_lambda", 0.05),
        )
        self.initial_population = getattr(config, "initial_population", 2) # number of initial random workflows before enabling mutation
        self._exploration_probe: dict | None = None
        self._exploration_rng: random.Random | None = None
    
    def get_workspace_manager_session_id(self) -> str | None:
        """Report the current WorkspaceManager session id.

        The benchmark harness calls this to locate this run's
        per-iteration workspace snapshots under /tmp
        (mimosa_run_<session_id>_<run_uuid>).

        Returns:
            str | None: The active session id, or ``None`` when
            ``begin_session`` has not run yet.
        """
        return self.workspace_mgr.session_id

    def _explore_rng(self) -> random.Random:
        """Return the engine-owned, seeded RNG for the base-rate trigger.

        The exploration-probe wrapper calls this once per generation. It
        creates the generator on first use, seeded with the fixed module
        seed so identical configs replay identically.

        Returns:
            random.Random: The engine's long-lived seeded generator.
        """
        rng = getattr(self, "_exploration_rng", None)
        if rng is None:
            rng = random.Random(_EXPLORE_RNG_SEED)
            self._exploration_rng = rng
        return rng

    def _evaluate_exploration_probe(
        self, controller: StopController, runs: list[IndividualRun],
    ) -> dict:
        """Evaluate the frontier-probe trigger with the config knobs.

        ``evolve_generation`` calls this once per generation, after the
        stop observation.

        Thin wrapper over :func:`evaluate_exploration_probe`: resolves
        ``explore_after_plateau`` (None → patience − 1) and
        ``explore_base_rate`` from config and supplies the seeded RNG.

        Args:
            controller (StopController): Session stop controller.
            runs (list[IndividualRun]): Completed runs, oldest first.

        Returns:
            dict: The probe dict from :func:`evaluate_exploration_probe`
            (``triggered``, ``reason``, and the measured details).
        """
        return evaluate_exploration_probe(
            controller,
            runs,
            explore_after=getattr(self.config, "explore_after_plateau", None),
            base_rate=float(
                getattr(self.config, "explore_base_rate", 0.0) or 0.0
            ),
            rng=self._explore_rng(),
        )

    async def mockup(self, wf: WorkflowInfo | None, goal: str) -> list[IndividualRun]:
        """Use existing workflow data instead of orchestrating a fresh run.

        ``start_workflow_evolution`` calls this in mockup mode, passing a
        workflow loaded from disk, to reuse its stored answers and score.

        Args:
            wf (WorkflowInfo | None): Pre-existing workflow info with
                state, answers, and score; ``None`` is rejected.
            goal (str): Task description, used when ``wf.goal`` is
                missing.

        Returns:
            A single-element list with the mocked :class:`IndividualRun`.

        Raises:
            ValueError: When no workflow template is supplied.
        """
        if wf is None:
            raise ValueError("❌ Mockup mode requires a valid workflow template. "
                             "Please provide a template_uuid or ensure workflows exist in the workflow directory.")
        print_phase("MOCKUP MODE", color="\033[93m")
        print_info(f"Using existing workflow data from: {wf.uuid}")
        mock_run = IndividualRun(
            goal=wf.goal or goal,
            prompt=wf.code or goal,
            template_uuid=wf.uuid,
            workflow_template=wf,
            max_depth=1,
            judge=True,
            scenario_rubric=None,
            original_task=wf.original_task,
            current_uuid=wf.uuid,
            answers=wf.answers,
            state_result=wf.state_result,
            reward=wf.overall_score,
            iteration_count=1
        )
        agents_answers = self.extract_agents_behavior(wf.state_result)
        self.show_answers(agents_answers)
        _, _ = await self._evaluate_and_calculate_cost(
            True, mock_run.judge, wf.uuid, mock_run.answers, mock_run.scenario_rubric, []
        )
        print_ok(f"Mockup run completed with reward: {wf.overall_score:.1f}")
        return [mock_run]

    def load_phenotype_result(self, uuid: str) -> Any:
        """Load the result of a previously executed workflow state.

        Mockup mode and manual inspection call this with a workflow UUID
        to reload its stored evaluation payload from disk.

        Args:
            uuid (str): UUID of the workflow state to load.

        Returns:
            Parsed JSON contents of the workflow's ``state_result.json``,
            or ``None`` when the file is missing.

        Raises:
            ValueError: When the state file exists but cannot be read.
        """
        try:
            with open(f"{self.workflow_dir}/{uuid}/state_result.json") as f:
                return json.loads(f.read().strip())
        except FileNotFoundError:
            return None
        except Exception as e:
            raise ValueError(f"❌ Error reading workflow state: {str(e)}") from e

    def load_workflow_genotype_code(self, workflow_id: str) -> str:
        """Load the genotype source of a stored workflow.

        Inspection and tooling code calls this with a workflow UUID to
        read the Python genotype stored for that run.

        Args:
            workflow_id (str): UUID identifying the workflow folder.

        Returns:
            The Python source of the workflow genotype.

        Raises:
            ValueError: When the folder, the code file or its contents are
                not accessible.
        """
        workflow_path = f"{self.workflow_dir}/{workflow_id}"
        if not os.path.exists(workflow_path):
            raise ValueError(
                f"❌ Workflow for ID {workflow_id} not found in {self.workflow_dir}."
            )

        try:
            with open(f"{workflow_path}/workflow_genotype_{workflow_id}.py") as f:
                return f.read()
        except FileNotFoundError as e:
            raise ValueError(
                f"❌ Workflow code file not found for ID {workflow_id} in {workflow_path}."
            ) from e
        except Exception as e:
            raise ValueError(f"❌ Error reading workflow code: {str(e)}") from e

    def get_total_rewards(self, wf_state: Any, eval_type: str) -> float:
        """Read the total reward from a workflow state.

        Inspection and reporting code calls this with a completed
        workflow state and the evaluation type used for the run.

        Args:
            wf_state (Any): Workflow state dict; falsy values
                short-circuit to ``0.0``.
            eval_type (str): Either ``"generic"`` or ``"scenario"``;
                other values yield ``0.0``.

        Returns:
            The reward extracted from the relevant evaluation slice.
        """
        if not wf_state or not eval_type:
            return 0.0
        if eval_type == "generic":
            return wf_state["evaluation"]["generic"]["overall_score"]
        elif eval_type == "scenario":
            return wf_state["evaluation"]["scenario"]["score"]
        else:
            return 0.0

    def extract_agents_behavior(self, wf_state: Any) -> str:
        """Extract the per-agent answers from the workflow state.

        Mockup display and notifications call this with a completed
        workflow state to summarize what the agents produced.

        Args:
            wf_state (Any): Workflow state dict, possibly ``None``.

        Returns:
            A newline-joined ``agent <name>: <truncated answer>...`` block, the
            raw ``answers`` string for non-list values, or an empty string when
            no answers were captured.
        """
        if not wf_state or "answers" not in wf_state:
            return ""

        agents_answers = (
            "\n".join(f"agent {n}: {str(x)[:16384]}..." for (n, x) in zip(wf_state["step_name"], wf_state["answers"], strict=True))
            if isinstance(wf_state["answers"], list)
            else wf_state["answers"]
        )
        return agents_answers

    def show_answers(self, agents_answers: str) -> None:
        """Render the per-agent answer block in a coloured CLI box.

        The engine calls this after each run to show the agent answers
        in the console.

        Args:
            agents_answers (str): Pre-formatted string of agent answers
                from :meth:`extract_agents_behavior`.

        Returns:
            None. The block is printed to the console.
        """
        print_box(agents_answers, title="Workflow Agents Answers", color=YELLOW)

    def select_parent_workflow(
        self,
        goal: str,
        template_uuid: str | None = None,
        crossover_rate: float | None = None,
        n_parents: int | None = None,
    ) -> tuple[list[WorkflowInfo], bool]:
        """Select one or more parent workflows under evolutionary pressure.

        The engine calls this at session start and before each new
        generation, with the task goal and an optional template override.
        Under policy-BT it refreshes the selector's workflow inventory
        first. It returns no parents when the workflow directory is
        missing or empty.

        Args:
            goal (str): Task description for similarity matching.
            template_uuid (str | None): If provided, skip selection and
                load this workflow directly (single-parent mutation
                path).
            crossover_rate (float | None): Probability in [0, 1] that
                crossover is attempted when several candidate workflows
                exist; defaults come from config.
            n_parents (int | None): Number of parents to select when
                crossover fires; defaults come from config.

        Returns:
            (list[WorkflowInfo], use_crossover) — selected parent(s) and
            a flag telling the caller whether to apply crossover or mutation.
        """
        if not os.path.exists(self.workflow_dir):
            return [], False

        workflows = [
            f for f in os.listdir(self.workflow_dir)
            if os.path.isfile(os.path.join(self.workflow_dir, f, "state_result.json"))
        ]
        if not workflows:
            return [], False

        # Explicit template → single parent, mutation only
        if template_uuid is not None:
            wf = WorkflowInfo(template_uuid, Path(f"{self.workflow_dir}/{template_uuid}"))
            return [wf], False

        if crossover_rate is None:
            crossover_rate = getattr(self.config, "crossover_rate", 0.4)
        if n_parents is None:
            n_parents = getattr(self.config, "n_parents", 2)

        if getattr(self.config, "verifier_kind", None) == "policy_bt":
            self.workflow_selector.workflows_info = self.workflow_selector.discover_workflows()

        selected, use_crossover = self.workflow_selector.select_parent_workflows(
            goal=goal,
            selection_pressure=self.selection,
            n_parents=n_parents,
            crossover_rate=crossover_rate,
            threshold_similarity=getattr(self.config, "parent_threshold_similarity", 0.8),
            threshold_score=getattr(self.config, "parent_threshold_score", 0.01),
        )

        mode = "CROSSOVER" if use_crossover else "MUTATION"
        print_section("PARENTS SELECTION")
        print_info(f"Strategy: {self.selection.strategy.value} — Mode: {mode}")
        print_info(f"Selected {len(selected)} parent(s) from workflow pool")
        for i, wf in enumerate(selected):
            print_info(f"  Parent {i+1}: {wf.uuid}  (score={wf.overall_score:.2f})")

        return selected, use_crossover

    def get_genotype_instructions(
        self, goal: str, wf: WorkflowInfo | None, max_iterations: int = 10
    ) -> str:
        """Build the genotype prompt for mutation or for the first individual.

        The engine calls this for run 0 and for seed generations, passing
        the task goal and the chosen parent (``None`` for a cold start).

        Args:
            goal (str): Task description for the new workflow.
            wf (WorkflowInfo | None): Parent workflow info; when
                ``None`` a seed prompt is built.
            max_iterations (int): Total planned attempts, passed through
                to the variation engine.

        Returns:
            Either a mutation prompt (when `wf` is given) or a seed prompt.
        """
        if wf:
            return self.variation.mutation_prompt(
                goal, wf, wf.code, 0, max_iterations=max_iterations
            )
        else:
            return self.variation.seed_genome_prompt(goal)

    def _mutation_parent_or_last_child(
        self, parent_workflows: list[WorkflowInfo], child_uuid: str | None
    ) -> WorkflowInfo | None:
        """Choose the workflow that the next mutation changes.

        ``evolve_generation`` calls it on the mutation branch with the
        selected parents and the UUID of the generation that just ran.
        It returns the first selected parent. When no parent qualified,
        it returns the just-finished generation, so its code, verifier
        gradient and run log all describe the same run. A generation
        without a genotype on disk gives ``None`` (parentless mutation).

        Args:
            parent_workflows (list[WorkflowInfo]): Parents from
                :meth:`select_parent_workflow`; may be empty.
            child_uuid (str | None): UUID of the just-finished generation
                from ``orchestrate_workflow``; ``None`` or ``""`` when the
                run got no UUID.

        Returns:
            WorkflowInfo | None: The first parent, else a fresh accessor
            for the just-finished generation (read after its evaluation),
            else ``None``.
        """
        if parent_workflows:
            return parent_workflows[0]
        if not child_uuid:
            return None
        child = WorkflowInfo(child_uuid, Path(self.workflow_dir) / child_uuid)
        if not child.code:
            return None
        print_info(
            f"No parent qualified; mutating the last generation {child_uuid}."
        )
        return child

    async def start_workflow_evolution(
        self,
        goal: str,
        template_uuid: str | None = "20260512_162504_70ccefbf",
        judge: bool = True,
        scenario_rubric: str | None = None,
        enable_evolution: bool = False,
        original_task: str | None = None,
        single_agent_mode: bool = False,
        mockup_mode: bool = False,
    ) -> list[IndividualRun]:
        """Start the evolution session for one goal.

        The CLI, the planner, and the benchmark runner call this once per
        goal. It resets the archive and variation history, snapshots the
        workspace, runs generations until a stop rule fires, and then
        restores the best workspace as the task capsule.

        Args:
            goal (str): The primary goal or objective to reach (may be
                knowledge-wrapped).
            template_uuid (str | None): UUID of a workflow template to
                start from.
            judge (bool): Whether judging mode is enabled for evaluation.
            scenario_rubric (str | None): ID of the scenario for
                evaluation.
            enable_evolution (bool): Whether learning mode is on. Keeps
                improving the workflow score even after all agents report
                success.
            original_task (str | None): Original unwrapped task for
                similarity matching.
            single_agent_mode (bool): Force single-agent orchestration.
            mockup_mode (bool): If True, reuse existing workflow data
                from ``select_parent_workflow`` instead of orchestrating
                a fresh run. Useful for testing and debugging.

        Returns:
            The list of :class:`IndividualRun` produced across the evolution.
        """
        wf = None
        max_iteration = self.config.max_learning_evolve_iterations if enable_evolution else 1
        enable_evolution = enable_evolution and not single_agent_mode
        judge = judge and not single_agent_mode
        max_iteration = (
            self.config.max_learning_evolve_iterations if enable_evolution else 1
        )

        # Reset archive and variation history at session start so each task
        # starts with a clean slate (prevents observer-history contamination
        # across rows).
        self.selection._archive = []
        self.config.policy_bt_session_id = uuid4().hex
        self.variation.score_history.clear()
        self.variation.textual_gradient_history.clear()
        self.variation.agent_count_history.clear()

        parents, _ = self.select_parent_workflow(
            goal, template_uuid=template_uuid
        )
        wf = parents[0] if parents else None

        if mockup_mode:
            return await self.mockup(wf, goal)

        # ── Workspace lifecycle: snapshot → clean → restore before first run ─
        self.workspace_mgr.begin_session()

        craft_instructions = self.get_genotype_instructions(goal, wf, max_iterations=max_iteration)

        rewards_history = []
        assertion_history = []  # Track [passed, total] per iteration
        self.viz_utils.create_rewards_curve_plot(goal)

        from sources.evaluators.policy_bt_verifier.reexec import reproduction_goal_note
        goal += "\n" + reproduction_goal_note(self.config) + "\n"

        # ── Stop control (stop-condition plan 2026-10-08) ─────────────
        stop_controller = StopController(
            mode=self.config.early_stop_mode,
            min_iter=self.config.min_learning_evolve_iterations,
            patience=self.config.plateau_patience,
            threshold=self.config.learned_score_threshold,
            max_iter=max_iteration,
            verifier_kind=getattr(self.judge, "verifier_kind", None),
        )
        print_info(
            f"[STOP] mode={stop_controller.mode} "
            f"({stop_controller.resolution_note()})"
        )

        # First run lineage: template-mutation if a parent was loaded, else seed.
        run0_kind = "mutation" if wf is not None else "seed"
        run0_parents = [wf.uuid] if wf is not None else []
        run0 = IndividualRun(
            goal=goal,
            prompt=craft_instructions,
            template_uuid=template_uuid,
            workflow_template=wf,
            max_depth=max_iteration,
            judge=judge,
            scenario_rubric=scenario_rubric,
            original_task=original_task,
            parent_uuids=run0_parents,
            evolution_kind=run0_kind,
        )

        runs = await self.evolve_generation(
            [run0],
            rewards_history=rewards_history,
            assertion_history=assertion_history,
            enable_evolution=enable_evolution,
            single_agent_mode=single_agent_mode,
            workspace_mgr=self.workspace_mgr,
            stop_controller=stop_controller,
        )

        # ── Restore workspace to the best run's saved state ──────────────────
        try:
            best_run, capsule_rule = _select_capsule_run(
                runs, stop_controller, getattr(self.judge, "verifier_kind", None),
            )
            if best_run and best_run.current_uuid:
                print_info(
                    f"Best run [{capsule_rule}]: {best_run.current_uuid} "
                    f"(score={f'{best_run.reward:.3f}' if best_run.reward is not None else 'N/A'})"
                )
                self.workspace_mgr.restore_best(best_run.current_uuid)
                try:
                    self._export_astra(best_run.current_uuid, goal)
                except Exception as e:
                    print_err(f"Error in Astra Export: {str(e)}")
                    pass
            else:
                print_warn("No successful run found; workspace restored to initial state.")
                self.workspace_mgr.restore_best("")  # triggers fallback
        except Exception:
            print_err("Unknown error in workspace restauration defaulting to latest workspace state.")
            pass

        return runs

    async def evolve_generation(
        self,
        runs: list[IndividualRun],
        rewards_history: list[float] | None = None,
        assertion_history: list[list[int]] | None = None,
        enable_evolution: bool = False,
        single_agent_mode: bool = False,
        workspace_mgr: WorkspaceManager | None = None,
        stop_controller: StopController | None = None,
    ) -> list[IndividualRun]:
        """Run one generation of the evolution loop and recurse if needed.

        The engine calls this with the run list from run 0, then
        recurses once per new generation until a depth cap, a stop rule,
        or the disabled-evolution exit ends the loop. The caller passes
        mutable run and history lists that the loop updates in place.

        Args:
            runs (list[IndividualRun]): Mutable list of runs accumulated
                so far; the last element is the current attempt.
            rewards_history (list[float] | None): Per-iteration reward
                values, updated in place.
            assertion_history (list[list[int]] | None): Per-iteration
                ``[passed, total]`` pairs for scenario evaluation,
                updated in place.
            enable_evolution (bool): When True, keep evolving even past
                nominal success up to a learning threshold.
            single_agent_mode (bool): Force single-agent orchestration.
            workspace_mgr (WorkspaceManager | None): Workspace lifecycle
                manager; ``None`` skips snapshot/restore plumbing.
            stop_controller (StopController | None): Session stop
                controller (created in :meth:`start_workflow_evolution`);
                fed one observation per generation and consulted for the
                early-stop decision.

        Returns:
            The (mutated) runs list, with the final attempt populated and
            ``plot`` set on the last run.
        """
        self._log_iteration_start(runs[-1].goal, runs[-1].iteration_count, runs[-1].max_depth)

        iteration_start_time = time.time()
        on_error = False
        uuid = None
        current_iteration_cost = 0.0  # Cost for this iteration only, not cumulative
        verdict: dict | None = None  # populated only when survivor validation runs

        # ── Reset workspace to the initial state before each run ─────
        if workspace_mgr is not None:
            workspace_mgr.reset_for_run()

        # Execute workflow
        print_info(f"Run {runs[-1].iteration_count + 1} of {runs[-1].max_depth}")
        run_stdout, uuid, workflow_genotype_code, executed = await self.orchestrator.orchestrate_workflow(
            goal=runs[-1].goal,
            craft_instructions=runs[-1].prompt,
            original_task=runs[-1].original_task,
            single_agent_mode=single_agent_mode
        )
        self._persist_run_stdout(uuid, run_stdout)
        wf_info = WorkflowInfo(uuid, Path(f"{self.workflow_dir}/{uuid}"))
        self._save_evolution_prompt_artifact(uuid, runs[-1].prompt)
        # Persist lineage as soon as we have a uuid so the evolution tree can
        # include even runs that subsequently fail evaluation.
        if uuid:
            record_lineage(
                self.workflow_dir,
                uuid,
                parents=runs[-1].parent_uuids,
                kind=runs[-1].evolution_kind,
                iteration=runs[-1].iteration_count,
                goal=runs[-1].original_task or runs[-1].goal,
            )
        on_error = not executed
        if on_error:
            # Tracebacks end with the raised exception; the head is runner noise.
            print_err(f"Workflow failed:\n...{run_stdout[-1024:]}")
        # ── Snapshot workspace results produced by this run ───────────────────
        if workspace_mgr is not None and uuid:
            workspace_mgr.save_run_snapshot(uuid)

        if workflow_genotype_code:
            # Evaluate and calculate costs
            eval_type, current_iteration_cost = await self._evaluate_and_calculate_cost(
                executed, runs[-1].judge, uuid, runs[-1].answers, runs[-1].scenario_rubric, assertion_history
            )

        verifier = (wf_info.state_result or {}).get("evaluation", {}).get("verifier", {})
        is_failure = (
            on_error
            or verifier.get("skipped_reason") in _UNSCORED_VERIFIER_REASONS
        )
        if uuid:
            # Reward wiring (N1): score, uncapped score and the fallback
            # flag are read back from the run's state_result in ONE pass,
            # outside the genotype guard — a run whose orchestration
            # returned no genotype string must not keep a stale
            # reward_is_fallback=False while its evaluation on disk
            # records a mean_claim fallback (pains_brenk gen 13).
            runs[-1].reward = wf_info.overall_score
            runs[-1].reward_uncapped = wf_info.overall_score_uncapped
            runs[-1].reward_is_fallback = wf_info.reward_is_fallback
            runs[-1].code = wf_info.code
            # Best-so-far *before* this offspring contributes; used by the
            # VariationEngine observer (plateau streak / success rate) for
            # the search-state block and telemetry.
            best_before = max(rewards_history) if rewards_history else None
            child_score = (
                None if is_failure or wf_info.overall_score is None
                else float(wf_info.overall_score)
            )
            self.variation.record_offspring_gradient(
                wf_info.abstracted_textual_gradient,
                is_failure=is_failure,
                child_score=child_score,
                best_before=best_before,
            )

        runs[-1].current_uuid = uuid
        runs[-1].answers = wf_info.answers if wf_info else []
        runs[-1].state_result = wf_info.state_result if wf_info else {}
        _refresh_policy_bt_rewards(runs, self.selection)
        agents_answers = self.extract_agents_behavior(wf_info.state_result) if wf_info else ""
        self.show_answers(agents_answers)
        if rewards_history is not None and wf_info is not None:
            rewards_history.append(wf_info.overall_score)

        if stop_controller is not None:
            stop_controller.observe(
                uuid=uuid or "",
                iteration=runs[-1].iteration_count,
                reward=wf_info.overall_score,
                pair_outcomes=_stop_pair_outcomes(verifier),
                failed=is_failure or not uuid,
            )

        exploration_probe: dict | None = None
        if stop_controller is not None:
            exploration_probe = self._evaluate_exploration_probe(
                stop_controller, runs
            )
            self._exploration_probe = exploration_probe
            self.variation.last_variation_state["exploration"] = (
                exploration_state(exploration_probe)
            )
            if exploration_probe["triggered"]:
                print_info(
                    "[EXPLORE] frontier probe: "
                    f"{exploration_probe['reason']}"
                )

        # ── Survivor validation: gate + populate _archive (steady-state population)
        if uuid and not is_failure:
            baseline_runs = runs[:-1] if len(runs) > 1 else [runs[-1]]
            verdict = self.selection.validate_survivor(
                baseline_runs=baseline_runs,
                new_runs=[runs[-1]],
            )
            runs[-1].selection_log = SelectionLog(
                from_iteration=max(runs[-1].iteration_count - 1, 0),
                to_iteration=runs[-1].iteration_count,
                improvement_type=self.selection.strategy.value,
                delta_reward=verdict["absolute_improvement"],
                is_validated=verdict["valid"],
                confidence=verdict["confidence"],
                admit_rejected=verdict.get("admit_rejected", False),
            )

        # Update visualizations
        self._update_visualizations(
            rewards_history, assertion_history,
            runs[-1].goal, runs[-1].scenario_rubric, uuid
        )
        self._refresh_evolution_tree(runs[-1].goal, uuid)
        self._refresh_archive_projection(uuid)

        # Calculate cumulative cost and update runs[-1].cost for accurate tracking
        runs[-1].cost = runs[-1].cost + current_iteration_cost

        # Persist per-iteration metrics & QD verdict to disk.
        ctx = {
            "verdict": verdict,
            "iteration_cost": current_iteration_cost,
            "iteration_start_time": iteration_start_time,
            "on_error": on_error,
            "stop_controller": stop_controller,
            "exploration_probe": exploration_probe,
        }
        self._persist_iteration_metrics(
            uuid, self._build_run_metrics_snapshot(runs[-1], wf_info, ctx)
        )
        if verdict is not None:
            self._persist_qd_archive(
                self._build_qd_archive_entry(runs[-1], uuid, verdict)
            )

        # Log and notify completion (show per-iteration cost, not cumulative)
        if wf_info:
            self._log_iteration_completion(
                runs[-1].iteration_count, runs[-1].max_depth, iteration_start_time,
                wf_info.overall_score, current_iteration_cost, runs[-1].goal, uuid, wf_info.state_result, rewards_history
            )
        
        if runs[-1].iteration_count >= runs[-1].max_depth-1:
            print_info("Maximum recursive depth reached.")
            return runs

        if wf_info and enable_evolution:
            if stop_controller is not None:
                stop, why = stop_controller.should_stop(runs[-1].iteration_count)
            else:
                # Defensive: a caller that bypassed start_workflow_evolution
                # keeps the legacy inline threshold check.
                stop = (
                    wf_info.overall_score is not None
                    and wf_info.overall_score >= self.config.learned_score_threshold
                )
                why = "threshold: legacy inline check"
            if stop:
                print_ok(f"Evolution engine stopping: {why}")
                self._save_final_plots(assertion_history, rewards_history, uuid)
                self.notifier.send_message(
                    f"Done learning task: {wf_info.goal[:256]} \n"
                    f"Stop reason: {why}\n"
                    f"Final UUID: {uuid}\n"
                    f"Iterations: {runs[-1].iteration_count + 1}/{runs[-1].max_depth}\n",
                    title="Evolution done learning task.",
                    priority=0
                )
                return runs
            # Not stopping yet: fall through to keep evolving.
        elif not on_error:
            self._save_final_plots(assertion_history, rewards_history, uuid)
            print_ok("Completed workflow execution. Evolution disabled.")
            self.notifier.send_message(
                f"Task completed successfully!\n"
                f"Goal: {runs[-1].goal[:128]}...\n"
                f"Final UUID: {uuid}\n"
                f"Iterations: {runs[-1].iteration_count + 1}/{runs[-1].max_depth}\n"
                f"All workflows successful!",
                title=f"Evolution success - {uuid}",
                priority=0
            )
            return runs

        # ── Evolutionary parent selection: mutation or crossover ──────
        parent_workflows, use_crossover = self.select_parent_workflow(
            runs[-1].goal, template_uuid=None
        )

        task_goal = runs[-1].original_task or runs[-1].goal

        next_kind: str
        next_parent_uuids: list[str]
        if use_crossover and len(runs) >= self.initial_population:
            # CROSSOVER — recombine multiple parent genotypes
            print_phase("CROSSOVER VARIATION", color=CYAN)
            runs[-1].prompt = self.variation.crossover_prompt(
                goal=task_goal,
                wf_infos=parent_workflows,
                genotypes=[pw.code or "" for pw in parent_workflows],
                iteration_count=runs[-1].iteration_count,
                max_iterations=runs[-1].max_depth,
                exploration_probe=exploration_probe,
            )
            next_kind = "crossover"
            next_parent_uuids = [pw.uuid for pw in parent_workflows if pw and pw.uuid]
        elif len(runs) >= self.initial_population:
            # MUTATION — perturb the single best parent
            print_phase("MUTATION VARIATION", color=YELLOW)
            primary_parent = self._mutation_parent_or_last_child(
                parent_workflows, uuid
            )
            code = primary_parent.code if primary_parent else ""
            runs[-1].prompt = self.variation.mutation_prompt(
                task_goal, primary_parent, code,
                runs[-1].iteration_count, max_iterations=runs[-1].max_depth,
                exploration_probe=exploration_probe,
            )
            next_kind = "mutation" if primary_parent else "seed"
            next_parent_uuids = [primary_parent.uuid] if primary_parent else []
        else:
            # SEED — create initial random workflow(s) without a parent (cold start)
            print_phase("SEED POPULATION", color=GREEN)
            runs[-1].prompt = self.get_genotype_instructions(task_goal, None, max_iterations=runs[-1].max_depth)
            next_kind = "seed"
            next_parent_uuids = []

        runs.append(IndividualRun(
            goal=runs[-1].goal,
            prompt=runs[-1].prompt,
            cost=runs[-1].cost,  # Correct cumulative cost
            current_uuid=uuid,
            template_uuid=None,
            workflow_template=runs[-1].workflow_template if (wf_info and wf_info.state_result) else None,
            iteration_count=runs[-1].iteration_count + 1,
            max_depth=runs[-1].max_depth,
            judge=runs[-1].judge,
            answers=wf_info.answers if wf_info else [],
            state_result=wf_info.state_result if wf_info else {},
            scenario_rubric=runs[-1].scenario_rubric,
            original_task=runs[-1].original_task,  # PRESERVE original_task for workflow selection
            parent_uuids=next_parent_uuids,
            evolution_kind=next_kind,
        ))

        self._persist_variation_log(
            self._build_variation_log_entry(runs[-1], parent_uuid_of=uuid)
        )

        runs = await self.evolve_generation(
            runs,
            rewards_history=rewards_history,
            assertion_history=assertion_history,
            enable_evolution=enable_evolution,
            single_agent_mode=single_agent_mode,
            workspace_mgr=workspace_mgr,
            stop_controller=stop_controller,
        )

        runs[-1].plot = self._save_final_plots(assertion_history, rewards_history, uuid)
        return runs

    def _export_astra(self, best_uuid: str, goal: str) -> None:
        """Export the best workflow's trace to ASTRA after the session.

        ``start_workflow_evolution`` calls this once, with the capsule
        UUID and the goal, when ``config.export_astra`` is on (opt-in,
        off by default). Failure is non-fatal: a missing memory
        directory, an LLM hiccup, or a YAML write error must not break
        the evolution loop. See
        :class:`sources.transparency.AstraExporter` for the pipeline.

        Args:
            best_uuid (str): UUID of the winning run's workflow folder.
            goal (str): Task goal passed through to the exporter.

        Returns:
            None. Export failures are logged, never raised.
        """
        if not getattr(self.config, "export_astra", False):
            return
        try:
            from sources.transparency import AstraExporter
            AstraExporter(self.config, self.logger).export(best_uuid, goal)
        except Exception as exc:
            self.logger.warning(f"[ASTRA] export skipped: {exc}")
            print_warn(f"ASTRA export failed (non-fatal): {exc}")

    def _get_human_validation(self) -> bool:
        """Ask the operator whether to continue the workflow.

        Interactive runs call this when a retry needs human approval; it
        reads a yes/no answer from the terminal.

        Returns:
            bool: ``True`` when the user typed ``yes`` or ``y``, and
            ``False`` otherwise.
        """
        human_validation = input("Attempt to retry task? (yes/no): ").strip().lower()
        if human_validation not in ["yes", "y"]:
            return False
        return True

    def _log_iteration_start(self, goal: str, iteration_count: int, max_depth: int) -> None:
        """Log the start of one generation.

        ``evolve_generation`` calls this with the current attempt's data
        to print the iteration header and log the goal.

        Args:
            goal (str): Task description for the current iteration.
            iteration_count (int): Zero-based iteration index.
            max_depth (int): Total planned iterations.

        Returns:
            None. Output goes to the log and the console.
        """
        logger = logging.getLogger(__name__)
        print_iteration_header(iteration_count + 1, max_depth)
        print_box(goal, title="📋 CURRENT TASK", truncate=256)
        logger.info(f"[ITERATION START] {iteration_count + 1}/{max_depth} - {goal[:50]}...")

    async def _evaluate_and_calculate_cost(
        self, executed: bool, judge: bool, uuid: str,
        agent_answers: str, scenario_rubric: str, assertion_history: list
    ) -> tuple[str, float]:
        """Score the finished workflow and price the iteration.

        ``evolve_generation`` calls this after orchestration, with the
        run's judge flag and captured answers. Evaluation runs only when
        judging is on and the workflow executed; the cost is always
        computed.

        Args:
            executed (bool): Whether the workflow ran to completion.
            judge (bool): Whether judging is enabled for this run.
            uuid (str): Workflow UUID; falsy values short-circuit
                evaluation.
            agent_answers (str): Pre-formatted agent answers passed to
                the judge.
            scenario_rubric (str): Scenario rubric ID, when applicable.
            assertion_history (list): Per-iteration ``[passed, total]``
                pairs, updated in place.

        Returns:
            ``(eval_type, exec_cost)`` — the evaluator type used (or ``None``)
            and the USD cost computed for the workflow.
        """
        logger = logging.getLogger(__name__)
        eval_type = None
        exec_cost = 0.0

        if judge and uuid and executed:
            agent_answers = agent_answers if executed else "workflow failed to execute."
            eval_type = await self._evaluate_workflow_phenotype(uuid, agent_answers, scenario_rubric, assertion_history)
        # Calculate cost regardless of execution success
        cost_start = time.time()
        exec_cost = self.pricing.calculate_cost(uuid)
        cost_time = time.time() - cost_start
        logger.info(f"[WORKFLOW COST] {uuid} cost calculated in {cost_time:.3f}s")

        return eval_type, exec_cost

    async def _evaluate_workflow_phenotype(
        self, uuid: str, agent_answers: str, scenario_rubric: str, assertion_history: list
    ) -> str:
        """Evaluate the workflow through the Layer 4 judge.

        ``_evaluate_and_calculate_cost`` calls this when judging is on.
        Gold-verifier grading runs in a worker thread so concurrent CSV
        workers keep their event loop; every other verifier runs inline.
        Scenario evaluations also update the assertion history.

        Args:
            uuid (str): Workflow UUID being evaluated.
            agent_answers (str): Pre-formatted agent answers passed to
                the judge.
            scenario_rubric (str): Scenario rubric ID, when applicable.
            assertion_history (list): Per-iteration ``[passed, total]``
                pairs, updated in place for scenario evaluations.

        Returns:
            The evaluator type string reported by the judge.
        """
        logger = logging.getLogger(__name__)
        print_phase("WORKFLOW EVALUATION PHASE")
        eval_start = time.time()
        if getattr(self.judge, "verifier_kind", None) == "gold":
            eval_result = await asyncio.to_thread(
                self.judge.evaluate,
                uuid=uuid,
                agent_answers=agent_answers,
                evaluator_type="verifier",
                scenario_rubric=scenario_rubric,
            )
        else:
            eval_result = self.judge.evaluate(uuid=uuid,
                                              agent_answers=agent_answers,
                                              evaluator_type="verifier",
                                              scenario_rubric=scenario_rubric)
        eval_type = eval_result['evaluation_type']
        eval_time = time.time() - eval_start
        logger.info(f"[WORKFLOW EVALUATION] {uuid}:\n{json.dumps(eval_result, indent=2)}")
        print_ok(f"Workflow evaluation completed in {eval_time:.3f}s")
        # Track assertion progress for scenario evaluation
        if scenario_rubric and isinstance(eval_result, dict) and assertion_history is not None:
            self._update_assertion_history(eval_result, assertion_history)
        return eval_type

    def _update_assertion_history(self, eval_result: dict, assertion_history: list) -> None:
        """Record one scenario evaluation in the assertion history.

        The phenotype evaluator calls this for scenario runs, with the
        judge output, to track assertion progress across iterations.

        Args:
            eval_result (dict): Judge output containing assertion or
                point counts.
            assertion_history (list): ``[passed, total]`` pairs,
                appended in place.

        Returns:
            None. The history list is updated in place.
        """
        passed = eval_result.get('passed_assertions', eval_result.get('earned_points', 0))
        total = eval_result.get('total_assertions', eval_result.get('total_points', 100))
        assertion_history.append([passed, total])
        pct = passed / total * 100 if total > 0 else 0
        print_info(f"📊 Assertions progress: {passed}/{total} ({pct:.0f}%)")

    def _update_visualizations(
        self, rewards_history: list, assertion_history: list,
        goal: str, scenario_rubric: str, uuid: str,
    ) -> None:
        """Refresh the plots after one generation.

        ``evolve_generation`` calls this with the running histories. The
        assertion plot wins when assertion data exists; otherwise the
        rewards curve is updated.

        Args:
            rewards_history (list): Per-iteration reward values.
            assertion_history (list): Per-iteration ``[passed, total]``
                pairs.
            goal (str): Task description (unused but kept for symmetry).
            scenario_rubric (str): Scenario rubric ID for assertion plots.
            uuid (str): Workflow UUID, used to write the plot artifact.

        Returns:
            None. Plot files are refreshed on disk.
        """
        if assertion_history:
            self._update_assertion_plot(assertion_history, scenario_rubric, uuid)
        elif rewards_history:
            self._update_rewards_plot(rewards_history)

    def _update_rewards_plot(self, rewards_history: list[float]) -> None:
        """Refresh the rewards-over-iterations curve.

        ``_update_visualizations`` calls this when a session has rewards
        but no assertion data.

        Args:
            rewards_history (list[float]): Per-iteration reward values.

        Returns:
            None. The curve is refreshed in the plot buffer.
        """
        self.viz_utils.update_rewards_curve(rewards_history)

    def _update_assertion_plot(
        self, assertion_history: list,
        scenario_rubric: str, uuid: str,
    ) -> None:
        """Update the assertion progress plot.

        ``_update_visualizations`` calls this for scenario runs, with
        the running history and the rubric that defines the totals.

        Args:
            assertion_history (list): Per-iteration ``[passed, total]``
                pairs.
            scenario_rubric (str): Scenario rubric ID used to look up
                totals.
            uuid (str): Workflow UUID, used to write the plot artifact.

        Returns:
            None. Plot failures are logged, never raised.
        """
        scenario = ScenarioLoader().load_scenario(scenario_rubric)
        total_assertions = len(scenario.get("assertions", [])) if scenario else 0
        self.viz_utils.update_assertion_progress_plot(assertion_history, total_assertions)
        plot_filename = f"{self.workflow_dir}/{uuid}/assertion_progress.png"
        self.viz_utils.save_plot(plot_filename)
        print_info(f"📊 Assertion progress plot updated: {plot_filename}")

    def _log_iteration_completion(
        self, iteration_count: int, max_depth: int, iteration_start_time: float,
        wf_rewards: float, exec_cost: float, goal: str, uuid: str,
        wf_state: Any, rewards_history: list,
    ) -> None:
        """Log the end of one generation and notify the operator.

        ``evolve_generation`` calls this after the bookkeeping, with the
        finished attempt's scores and cost.

        Args:
            iteration_count (int): Zero-based iteration index just
                completed.
            max_depth (int): Total planned iterations.
            iteration_start_time (float): ``time.time()`` taken at
                iteration start.
            wf_rewards (float): Reward achieved this iteration.
            exec_cost (float): USD cost spent this iteration.
            goal (str): Task description, included in the notification
                body.
            uuid (str): Workflow UUID just completed.
            wf_state (Any): Workflow state used to extract agent answers.
            rewards_history (list): Per-iteration reward values,
                included verbatim.

        Returns:
            None. A push notification is sent after logging.
        """
        logger = logging.getLogger(__name__)
        iteration_time = time.time() - iteration_start_time
        logger.info(
            f"[ITERATION END] {iteration_count}/{max_depth} completed in {iteration_time:.3f}s - "
            f"Rewards: {wf_rewards:.1f}, Cost: {exec_cost:.3f} USD"
        )
        print_summary(
            f"ITERATION {iteration_count}/{max_depth} COMPLETE",
            [
                ("Rewards", f"{wf_rewards:.1f}"),
                ("Cost", f"${exec_cost:.6f}"),
                ("Time", f"{iteration_time:.3f}s"),
            ],
        )
        self.notifier.send_message(
            f"Iteration {iteration_count + 1} completed.\n"
            f"Goal: {goal[:128]}...\n"
            f"Cost: {exec_cost:.6f} USD.\n"
            f"Rewards history: {rewards_history}"
            f"Answers: {self.extract_agents_behavior(wf_state)}\n",
            title=f"Workflow {uuid} completed.",
        )

    def _refresh_evolution_tree(
        self, goal: str | None = None, uuid: str | None = None
    ) -> None:
        """Re-render the goal-specific evolution-tree PNG after one generation.

        ``evolve_generation`` calls this with the current goal and UUID
        after each iteration. It scans only workflows whose
        ``goal_<uuid>.txt`` matches *goal* so trees
        from different runs no longer pile into one root-level image, and writes
        the result to ``<workflow_dir>/<uuid>/evolution_tree.png``.

        Best-effort: scanning failures are logged and swallowed so an issue
        rendering the tree never aborts an evolution run. The visualizer is
        imported lazily to avoid a circular import via ``sources.core``.

        Args:
            goal (str | None): Task goal used to filter the scan;
                ``None`` renders the unfiltered tree.
            uuid (str | None): Workflow UUID whose folder receives the
                PNG; the render is skipped when falsy.

        Returns:
            None. Render failures are logged, never raised.
        """
        if not uuid:
            return
        workflow_path = Path(self.workflow_dir) / uuid
        if not workflow_path.is_dir():
            return
        try:
            from sources.utils.evolution_tree import render_evolution_tree
            output = render_evolution_tree(
                self.workflow_dir,
                output_path=workflow_path / "evolution_tree.png",
                goal=goal,
            )
            if output is not None:
                self.logger.info(f"Evolution tree refreshed: {output}")
        except Exception as e:
            self.logger.warning(f"Failed to refresh evolution tree: {e}")

    def _refresh_archive_projection(self, uuid: str | None = None) -> None:
        """Re-render the PCA projection PNG of the QD archive after one generation.

        ``evolve_generation`` calls this after each iteration. Projects
        the SelectionPressure archive's behaviour descriptors and writes
        ``<workflow_dir>/archive_pca.png``, starring *uuid* when it is present.

        Best-effort: rendering failures are logged and swallowed so an archive
        image issue never aborts an iterative refinement run. The renderer is
        imported lazily to avoid a circular import via ``sources.utils``.

        Args:
            uuid (str | None): Workflow UUID to star in the projection;
                ``None`` renders the archive without a highlight.

        Returns:
            None. Render failures are logged, never raised.
        """
        try:
            from sources.utils.archive_projection import render_archive_pca
            output = render_archive_pca(
                self.selection.archive,
                self.workflow_dir,
                highlight_uuid=uuid,
            )
            if output is not None:
                self.logger.info(f"Archive projection refreshed: {output}")
        except Exception as e:
            self.logger.warning(f"Failed to refresh archive projection: {e}")

    def _save_evolution_prompt_artifact(self, uuid: str, prompt: str) -> None:
        """Persist the variation/seed prompt into the workflow's folder.

        ``evolve_generation`` calls this right after orchestration, with
        the new UUID and the prompt that produced the run, so every
        genotype keeps its full prompt lineage on disk.

        Args:
            uuid (str): Workflow UUID whose folder receives the
                artifact; no-op on falsy values.
            prompt (str): Prompt text to persist; no-op when empty or
                when the folder does not yet exist. I/O errors are
                logged, never raised.

        Returns:
            None. The prompt is written only when the folder exists.
        """
        if not uuid or not prompt:
            return
        workflow_path = os.path.join(self.workflow_dir, uuid)
        if not os.path.isdir(workflow_path):
            return
        try:
            prompt_path = os.path.join(workflow_path, f"evolution_prompt_{uuid}.md")
            with open(prompt_path, "w") as f:
                f.write(prompt)
            self.logger.info(f"Saved evolution prompt to: {prompt_path}")
        except Exception as e:
            self.logger.error(f"Failed to save evolution prompt: {e}")

    def _build_run_metrics_snapshot(
        self, run: "IndividualRun", wf_info: "WorkflowInfo", ctx: dict,
    ) -> dict:
        """Assemble the per-iteration dict written to ``run_metrics.json``.

        ``evolve_generation`` calls this at the end of each generation,
        with the finished run and the iteration locals. Besides the stop
        state, ``variation_state.exploration`` carries
        the frontier-probe trigger evaluated at the END of this
        generation (it shaped the NEXT prompt): ``{"triggered": bool,
        "reason": str, "metrics": [...]}`` (review 2026-10-09 §4.3).

        Args:
            run: Current ``IndividualRun`` (post-evaluation, post cumulative-cost update).
            wf_info: Workflow info exposing scores.
            ctx: Iteration locals: ``verdict``, ``iteration_cost``,
                ``iteration_start_time``, ``on_error``, ``stop_controller``,
                ``exploration_probe``.

        Returns:
            dict: JSON-serializable per-iteration metrics snapshot.
        """
        verdict = ctx.get("verdict") or {}
        stop = ctx.get("stop_controller")
        stop_state: dict[str, Any] = {"champion_uuid": None, "since_change": None,
                                      "stop_reason": None}
        if stop is not None:
            stopping, reason = stop.should_stop(run.iteration_count)
            stop_state = {
                "champion_uuid": stop.champion,
                "since_change": stop.since_change,
                "stop_reason": reason if stopping else None,
            }
        return {
            "uuid": run.current_uuid,
            "iteration": run.iteration_count,
            "evolution_kind": run.evolution_kind,
            "parent_uuids": list(run.parent_uuids or []),
            "iteration_wall_time_s": round(time.time() - ctx["iteration_start_time"], 3),
            "iteration_cost_usd": float(ctx["iteration_cost"]),
            "cumulative_cost_usd": float(run.cost),
            "overall_score": _scalar(wf_info.overall_score),
            "overall_score_uncapped": _scalar(wf_info.overall_score_uncapped),
            "reward_is_fallback": bool(run.reward_is_fallback),
            "on_error": bool(ctx["on_error"]),
            **stop_state,
            "selection_log": _to_jsonable(run.selection_log),
            "qd_descriptor": list(verdict.get("behaviour_descriptor", [])),
            "qd_score": _scalar(verdict.get("qd_score")),
            "novelty_score": _scalar(verdict.get("novelty_score")),
            "variation_state": dict(self.variation.last_variation_state),
            "finished_at": datetime.utcnow().isoformat(),
        }

    def _build_qd_archive_entry(
        self, run: "IndividualRun", uuid: str, verdict: dict,
    ) -> dict:
        """Assemble one ``qd_archive.jsonl`` row from a survivor verdict.

        ``evolve_generation`` calls this after survivor validation, with
        the validated run and its verdict, to persist the QD archive
        decision.

        Args:
            run (IndividualRun): The validated run.
            uuid (str): UUID of the run's workflow folder.
            verdict (dict): Survivor verdict from
                ``SelectionPressure.validate_survivor``.

        Returns:
            dict: JSON-serializable archive row (scores, behaviour
            descriptor, novelty, and archive effect).
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "iteration": run.iteration_count,
            "uuid": uuid,
            "evolution_kind": run.evolution_kind,
            "parent_uuids": list(run.parent_uuids or []),
            "qd_descriptor": list(verdict.get("behaviour_descriptor", [])),
            "qd_score": _scalar(verdict.get("qd_score")),
            "quality_norm": _scalar(verdict.get("quality_norm")),
            "novelty_norm": _scalar(verdict.get("novelty_norm")),
            "novelty_score": _scalar(verdict.get("novelty_score")),
            "is_valid": bool(verdict.get("valid")),
            "admit_rejected": bool(verdict.get("admit_rejected")),
            "archive_size": verdict.get("archive_size"),
            "evicted_uuid": verdict.get("evicted_uuid"),
        }

    def _build_variation_log_entry(
        self, next_run: "IndividualRun", parent_uuid_of: str | None,
    ) -> dict:
        """Assemble one ``variation_log.jsonl`` row.

        ``evolve_generation`` calls this right after appending the next
        run, to snapshot the variation state that shaped its prompt.

        Args:
            next_run (IndividualRun): The freshly-appended run whose
                prompt was just produced by the variation engine.
            parent_uuid_of (str | None): UUID of the run that produced
                ``next_run``'s prompt (the predecessor in the recursion).

        Returns:
            dict: JSON-serializable log row (operator, parent lineage,
            and the variation state).
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "for_iteration": next_run.iteration_count,
            "from_uuid": parent_uuid_of,
            "evolution_kind": next_run.evolution_kind,
            "parent_uuids": list(next_run.parent_uuids or []),
            **dict(self.variation.last_variation_state),
        }

    def _persist_iteration_metrics(self, uuid: str, snapshot: dict) -> None:
        """Write ``run_metrics.json`` next to the workflow run.

        ``evolve_generation`` calls this once per generation with the
        metrics snapshot. Write failures are logged, never raised; a
        falsy UUID skips the write.

        Args:
            uuid (str): Workflow UUID identifying the destination folder.
            snapshot (dict): JSON-serializable mapping of per-iteration
                metrics.

        Returns:
            None. Write failures are logged, never raised.
        """
        if not uuid:
            return
        try:
            write_run_metrics(Path(self.workflow_dir) / uuid, snapshot)
        except Exception as e:
            self.logger.warning(f"run_metrics write failed for {uuid}: {e}")

    def _persist_run_stdout(self, uuid: str, run_stdout: str) -> None:
        """Write the raw workflow output to ``run_stdout.log``.

        The variation engine reads this file through
        ``WorkflowInfo.run_stdout`` as the unverified raw log of this
        generation when it is a parent without a verifier gradient.
        ``evolve_generation`` calls this right after ``orchestrate_workflow``
        returns, also for failed runs. This method writes the text verbatim
        (ANSI escapes kept) in UTF-8 into the workflow folder.

        Args:
            uuid (str): Workflow UUID from ``orchestrate_workflow``; a falsy
                value skips the write.
            run_stdout (str): Raw execution output, or the error text, from
                ``orchestrate_workflow``.

        Returns:
            None. Write failures are logged, never raised.
        """
        if not uuid:
            return
        folder = Path(self.workflow_dir) / uuid
        try:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "run_stdout.log").write_text(run_stdout or "", encoding="utf-8")
        except OSError as e:
            self.logger.warning(f"run_stdout write failed for {uuid}: {e}")

    def _persist_qd_archive(self, entry: dict) -> None:
        """Append one QD validation verdict to ``qd_archive.jsonl``.

        ``evolve_generation`` calls this after survivor validation, with
        a row from :meth:`_build_qd_archive_entry`.

        Args:
            entry (dict): Row from :meth:`_build_qd_archive_entry`.

        Returns:
            None. Append failures are logged, never raised.
        """
        try:
            append_jsonl(Path(self.workflow_dir) / "qd_archive.jsonl", entry)
        except Exception as e:
            self.logger.warning(f"qd_archive append failed: {e}")

    def _persist_variation_log(self, entry: dict) -> None:
        """Append one mutation-scope snapshot to ``variation_log.jsonl``.

        ``evolve_generation`` calls this when a new run joins the loop,
        with a row from :meth:`_build_variation_log_entry`.

        Args:
            entry (dict): Row from :meth:`_build_variation_log_entry`.

        Returns:
            None. Append failures are logged, never raised.
        """
        try:
            append_jsonl(Path(self.workflow_dir) / "variation_log.jsonl", entry)
        except Exception as e:
            self.logger.warning(f"variation_log append failed: {e}")

    def _save_final_plots(self, assertion_history: list, reward_history: list, uuid: str) -> str:
        """Save the session's final reward-progress plot.

        The engine calls this on stop and at the end of the recursion,
        with the running histories. When both histories are empty, no
        plot is written and the path stays empty.

        Args:
            assertion_history (list): Per-iteration ``[passed, total]``
                pairs.
            reward_history (list): Per-iteration reward values.
            uuid (str): Workflow UUID whose folder receives the plot.

        Returns:
            The path of the saved plot, or an empty string when neither
            history contains data.
        """
        plot_filename = ""
        if assertion_history or reward_history:
            plot_filename = f"{self.workflow_dir}/{uuid}/reward_progress.png"
            self.viz_utils.save_plot(plot_filename)
        print_info(f"📊 Reward progress plot saved: {plot_filename}")
        return plot_filename

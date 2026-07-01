"""
Neuroevolution-inspired, LLM driven evolution of Multi-Agents workflows.
"""

import json
import logging
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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
from sources.benchmark_evaluation.scenario_loader import ScenarioLoader
from sources.utils.notify import PushNotifier
from sources.utils.pricing import PricingCalculator
from sources.utils.run_metrics import append_jsonl, write_run_metrics
from sources.utils.visualization import VisualizationUtils
from sources.utils.workspace_management import WorkspaceManager

from .lineage import record_lineage
from .orchestrator import WorkflowOrchestrator
from .schema import IndividualRun, SelectionLog
from .selection import SelectionPressure
from .variation_engine import VariationEngine
from .workflow_info import WorkflowInfo
from .workflow_selection import WorkflowSelector


def _scalar(value: Any) -> float | None:
    """Coerce ``value`` to a finite float, or ``None`` when missing/non-numeric."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # filter NaN


def _to_jsonable(obj: Any) -> Any:
    """Convert dataclasses (e.g. ``SelectionLog``) to plain dicts; pass through otherwise."""
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    return obj


class EvolutionEngine:
    """Evolution Engine: Handle the evolution of Multi-agents workflows."""
    def __init__(
        self,
        config: "Config",
        viz_utils: VisualizationUtils | None = None,
        process_id: int | None = None,
    ) -> None:
        """Wire up evaluator, orchestrator, variation and selection components.

        Args:
            config: Application configuration providing workflow directory,
                pricing and notification credentials.
            viz_utils: Optional visualization helper; a fresh one is built when
                not supplied.
            process_id: Optional caller-assigned identifier (used for logs).
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
        self.selection = SelectionPressure(
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

    async def mockup(self, wf: WorkflowInfo | None, goal: str) -> list[IndividualRun]:
        """Use existing workflow data instead of orchestrating a fresh run.

        Args:
            wf: Pre-existing workflow info supplying state, answers and score.
            goal: Task description, used when `wf.goal` is missing.

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

        Args:
            uuid: UUID of the workflow state to load.

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
        """Load the workflow code for a given workflow ID.

        Args:
            workflow_id: UUID identifying the workflow folder.

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
        """Calculate the total rewards from the workflow state.

        Args:
            wf_state: Workflow state dict; falsy values short-circuit to 0.0.
            eval_type: Either ``"generic"`` or ``"scenario"``; other values
                yield ``0.0``.

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
        """Extract the answers from the workflow state.

        Args:
            wf_state: Workflow state dict, possibly ``None``.

        Returns:
            A newline-joined ``agent <name>: <truncated answer>...`` block, the
            raw ``answers`` string for non-list values, or an empty string when
            no answers were captured.
        """
        if not wf_state or "answers" not in wf_state:
            return ""

        agents_answers = (
            "\n".join(f"agent {n}: {str(x)[:256]}..." for (n, x) in zip(wf_state["step_name"], wf_state["answers"], strict=True))
            if isinstance(wf_state["answers"], list)
            else wf_state["answers"]
        )
        return agents_answers

    def show_answers(self, agents_answers: str) -> None:
        """Render the per-agent answer block in a coloured CLI box.

        Args:
            agents_answers: Pre-formatted string of agent answers.
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

        Args:
            goal: Task description for similarity matching.
            template_uuid: If provided, skip selection and load this workflow
                directly (single-parent mutation path).
            crossover_rate: Probability ∈ [0, 1] that crossover is attempted
                when multiple candidate workflows exist.
            n_parents: Number of parents to select when crossover fires.

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
        """Get the genotype prompt for mutation or for the very first individual.

        Args:
            goal: Task description for the new workflow.
            wf: Parent workflow info; when ``None`` a seed prompt is built.
            max_iterations: Total planned attempts, passed through to the
                variation engine.

        Returns:
            Either a mutation prompt (when `wf` is given) or a seed prompt.
        """
        if wf:
            return self.variation.mutation_prompt(
                goal, wf, wf.code, "", 0, max_iterations=max_iterations
            )
        else:
            return self.variation.seed_genome_prompt(goal)

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
        """Start the learning process for achieving a specified goal.

        Args:
            goal: The primary goal or objective to be accomplished (may be
                knowledge-wrapped).
            template_uuid: UUID of a workflow template to use.
            judge: Whether to enable judging mode for evaluation.
            scenario_rubric: ID of scenario for evaluation.
            enable_evolution: Whether in learning mode. Will keep attempting
                to improve the workflow score even if all agents report
                success state.
            original_task: Original unwrapped task for similarity matching.
            single_agent_mode: Force single-agent orchestration.
            mockup_mode: If True, use existing workflow data from
                ``select_parent_workflow`` instead of calling
                ``orchestrate_workflow``. Useful for testing and debugging.

        Returns:
            The list of :class:`IndividualRun` produced across the evolution.
        """
        wf = None
        max_iteration = self.config.max_learning_evolve_iterations if enable_evolution else 1

        # Reset archive at session start
        self.selection._archive = []

        parents, _ = self.select_parent_workflow(
            goal, template_uuid=template_uuid
        )
        wf = parents[0] if parents else None

        if mockup_mode:
            return await self.mockup(wf, goal)

        # ── Workspace lifecycle: snapshot → clean → restore before first run ─
        workspace_mgr = WorkspaceManager(self.config, self.logger)
        workspace_mgr.begin_session()

        craft_instructions = self.get_genotype_instructions(goal, wf, max_iterations=max_iteration)

        rewards_history = []
        assertion_history = []  # Track [passed, total] per iteration
        self.viz_utils.create_rewards_curve_plot(goal)

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
            workspace_mgr=workspace_mgr,
        )

        # ── Restore workspace to the best run's saved state ──────────────────
        try:
            best_run = max(
                (r for r in runs if r.current_uuid),
                key=lambda r: (r.reward if r.reward is not None else 0.0, r.iteration_count),
                default=None,
            )
            if best_run and best_run.current_uuid:
                print_info(
                    f"Best run: {best_run.current_uuid} "
                    f"(score={f'{best_run.reward:.3f}' if best_run.reward is not None else 'N/A'})"
                )
                workspace_mgr.restore_best(best_run.current_uuid)
                self._export_astra(best_run.current_uuid, goal)
            else:
                print_warn("No successful run found; workspace restored to initial state.")
                workspace_mgr.restore_best("")  # triggers fallback inside WorkspaceManager
        finally:
            workspace_mgr.cleanup()

        return runs

    async def evolve_generation(
        self,
        runs: list[IndividualRun],
        rewards_history: list[float] | None = None,
        assertion_history: list[list[int]] | None = None,
        enable_evolution: bool = False,
        single_agent_mode: bool = False,
        workspace_mgr: WorkspaceManager | None = None,
    ) -> list[IndividualRun]:
        """Run one iteration of the evolution loop and recurse if needed.

        Args:
            runs: Mutable list of runs accumulated so far; the last element is
                the current attempt being executed.
            rewards_history: Per-iteration reward values, updated in place.
            assertion_history: Per-iteration ``[passed, total]`` pairs for
                scenario evaluation, updated in place.
            enable_evolution: When True, keep evolving even past nominal
                success up to a learning threshold.
            single_agent_mode: Force single-agent orchestration.
            workspace_mgr: Workspace lifecycle manager; ``None`` skips
                snapshot/restore plumbing.

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
            print_err(f"Workflow failed:\n{run_stdout[:512]}")
        # ── Snapshot workspace results produced by this run ───────────────────
        if workspace_mgr is not None and uuid:
            workspace_mgr.save_run_snapshot(uuid)

        if workflow_genotype_code:
            # Evaluate and calculate costs
            eval_type, current_iteration_cost = await self._evaluate_and_calculate_cost(
                executed, runs[-1].judge, uuid, runs[-1].answers, runs[-1].scenario_rubric, assertion_history
            )
            runs[-1].reward = wf_info.overall_score
            runs[-1].reward_uncapped = wf_info.overall_score_uncapped
            runs[-1].code = wf_info.code

        if uuid:
            verifier = (wf_info.state_result or {}).get("evaluation", {}).get("verifier", {})
            is_failure = (
                on_error
                or verifier.get("skipped_reason") == "workflow_generation_or_execution_failed"
            )
            # Best-so-far *before* this offspring contributes; used by the
            # Rechenberg 1/5 rule in VariationEngine to decide whether
            # mutation boldness should grow.
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
        runs[-1].answers = wf_info.answers
        runs[-1].state_result = wf_info.state_result
        agents_answers = self.extract_agents_behavior(wf_info.state_result)
        self.show_answers(agents_answers)
        rewards_history.append(wf_info.overall_score)

        # ── Survivor validation: gate + populate _archive (steady-state population)
        if uuid and not on_error:
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

        # Calculate cumulative cost and update runs[-1].cost for accurate tracking
        runs[-1].cost = runs[-1].cost + current_iteration_cost

        # Persist per-iteration metrics & QD verdict to disk.
        ctx = {
            "verdict": verdict,
            "iteration_cost": current_iteration_cost,
            "iteration_start_time": iteration_start_time,
            "on_error": on_error,
        }
        self._persist_iteration_metrics(
            uuid, self._build_run_metrics_snapshot(runs[-1], wf_info, ctx)
        )
        if verdict is not None:
            self._persist_qd_archive(
                self._build_qd_archive_entry(runs[-1], uuid, verdict)
            )

        # Log and notify completion (show per-iteration cost, not cumulative)
        self._log_iteration_completion(
            runs[-1].iteration_count, runs[-1].max_depth, iteration_start_time,
            wf_info.overall_score, current_iteration_cost, runs[-1].goal, uuid, wf_info.state_result, rewards_history
        )

        # Check termination conditions
        if runs[-1].iteration_count >= runs[-1].max_depth-1 and not on_error:
            print_info("Maximum recursive depth reached.")
            return runs
        if enable_evolution:
            if wf_info.overall_score >= self.config.learned_score_threshold:
                print_ok("Evolution engine reached learning threshold.")
                self._save_final_plots(assertion_history, rewards_history, uuid)
                self.notifier.send_message(
                    f"Done learning task: {wf_info.goal[:256]} \n"
                    f"Final UUID: {uuid}\n"
                    f"Iterations: {runs[-1].iteration_count + 1}/{runs[-1].max_depth}\n",
                    title="Evolution done learning task.",
                    priority=0
                )
                return runs
            # Below threshold with --learn: fall through to keep evolving.
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
                run_stderrs=[run_stdout] * len(parent_workflows),
                iteration_count=runs[-1].iteration_count,
                max_iterations=runs[-1].max_depth,
            )
            next_kind = "crossover"
            next_parent_uuids = [pw.uuid for pw in parent_workflows if pw and pw.uuid]
        elif len(runs) >= self.initial_population:
            # MUTATION — perturb the single best parent
            print_phase("MUTATION VARIATION", color=YELLOW)
            primary_parent = parent_workflows[0] if parent_workflows else None
            code = primary_parent.code if primary_parent else ""
            runs[-1].prompt = self.variation.mutation_prompt(
                task_goal, primary_parent, code, run_stdout,
                runs[-1].iteration_count, max_iterations=runs[-1].max_depth,
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
            workflow_template=runs[-1].workflow_template if wf_info.state_result else None,
            iteration_count=runs[-1].iteration_count + 1,
            max_depth=runs[-1].max_depth,
            judge=runs[-1].judge,
            answers=wf_info.answers,
            state_result=wf_info.state_result,
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
        )

        runs[-1].plot = self._save_final_plots(assertion_history, rewards_history, uuid)
        return runs

    def _export_astra(self, best_uuid: str, goal: str) -> None:
        """Best-effort post-run ASTRA export of the best workflow's trace.

        Gated on ``config.export_astra`` (opt-in, off by default). Failure is
        non-fatal: a missing memory directory, an LLM hiccup, or a YAML write
        error must not break the evolution loop. See
        :class:`sources.transparency.AstraExporter` for the pipeline.
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
        """Get human validation for continuing the workflow.

        Returns:
            ``True`` if the user typed ``yes`` / ``y``, ``False`` otherwise.
        """
        human_validation = input("Attempt to retry task? (yes/no): ").strip().lower()
        if human_validation not in ["yes", "y"]:
            return False
        return True

    def _log_iteration_start(self, goal: str, iteration_count: int, max_depth: int) -> None:
        """Log the start of an iteration.

        Args:
            goal: Task description for the current iteration.
            iteration_count: Zero-based iteration index.
            max_depth: Total planned iterations.
        """
        logger = logging.getLogger(__name__)
        print_iteration_header(iteration_count + 1, max_depth)
        print_box(goal, title="📋 CURRENT TASK", truncate=256)
        logger.info(f"[ITERATION START] {iteration_count + 1}/{max_depth} - {goal[:50]}...")

    async def _evaluate_and_calculate_cost(
        self, executed: bool, judge: bool, uuid: str,
        agent_answers: str, scenario_rubric: str, assertion_history: list
    ) -> tuple[str, float]:
        """Evaluate workflow and calculate cost.

        Args:
            executed: Whether the workflow ran to completion.
            judge: Whether judging is enabled for this run.
            uuid: Workflow UUID; falsy values short-circuit evaluation.
            agent_answers: Pre-formatted agent answers passed to the judge.
            scenario_rubric: Scenario rubric ID, when applicable.
            assertion_history: Per-iteration ``[passed, total]`` pairs,
                updated in place.

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
        """Evaluate the workflow and update assertion history.

        Args:
            uuid: Workflow UUID being evaluated.
            agent_answers: Pre-formatted agent answers passed to the judge.
            scenario_rubric: Scenario rubric ID, when applicable.
            assertion_history: Per-iteration ``[passed, total]`` pairs,
                updated in place for scenario evaluations.

        Returns:
            The evaluator type string reported by the judge.
        """
        logger = logging.getLogger(__name__)
        print_phase("WORKFLOW EVALUATION PHASE")
        eval_start = time.time()
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
        """Update assertion history with evaluation results.

        Args:
            eval_result: Judge output containing assertion or point counts.
            assertion_history: List of ``[passed, total]`` pairs, appended in place.
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
        """Update all visualizations with current data.

        Args:
            rewards_history: Per-iteration reward values.
            assertion_history: Per-iteration ``[passed, total]`` pairs.
            goal: Task description (unused but kept for symmetry).
            scenario_rubric: Scenario rubric ID for assertion plots.
            uuid: Workflow UUID, used to write the plot artefact.
        """
        if assertion_history:
            self._update_assertion_plot(assertion_history, scenario_rubric, uuid)
        elif rewards_history:
            self._update_rewards_plot(rewards_history)

    def _update_rewards_plot(self, rewards_history: list[float]) -> None:
        """Refresh the rewards-over-iterations curve.

        Args:
            rewards_history: Per-iteration reward values.
        """
        self.viz_utils.update_rewards_curve(rewards_history)

    def _update_assertion_plot(
        self, assertion_history: list,
        scenario_rubric: str, uuid: str,
    ) -> None:
        """Update assertion progress plot.

        Args:
            assertion_history: Per-iteration ``[passed, total]`` pairs.
            scenario_rubric: Scenario rubric ID used to look up totals.
            uuid: Workflow UUID, used to write the plot artefact.
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
        """Log iteration completion and send notification.

        Args:
            iteration_count: Zero-based iteration index just completed.
            max_depth: Total planned iterations.
            iteration_start_time: ``time.time()`` taken at iteration start.
            wf_rewards: Reward achieved this iteration.
            exec_cost: USD cost spent this iteration.
            goal: Task description, included in the notification body.
            uuid: Workflow UUID just completed.
            wf_state: Workflow state used to extract agent answers.
            rewards_history: Per-iteration reward values, included verbatim.
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
        """Re-render the goal-specific evolution-tree PNG after each iteration.

        Scans only workflows whose ``goal_<uuid>.txt`` matches *goal* so trees
        from different runs no longer pile into one root-level image, and writes
        the result to ``<workflow_dir>/<uuid>/evolution_tree.png``.

        Best-effort: scanning failures are logged and swallowed so an issue
        rendering the tree never aborts an evolution run. The visualizer is
        imported lazily to avoid a circular import via ``sources.core``.
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

    def _save_evolution_prompt_artifact(self, uuid: str, prompt: str) -> None:
        """Persist the variation/seed prompt that produced this workflow into its folder.

        Args:
            uuid: Workflow UUID whose folder receives the artefact.
            prompt: Prompt text to persist. No-op on falsy inputs or when
                the folder does not yet exist; I/O errors are logged.
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

        Args:
            run: Current ``IndividualRun`` (post-evaluation, post cumulative-cost update).
            wf_info: Workflow info exposing scores.
            ctx: Iteration locals: ``verdict``, ``iteration_cost``,
                ``iteration_start_time``, ``on_error``.
        """
        verdict = ctx.get("verdict") or {}
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
            "on_error": bool(ctx["on_error"]),
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
        """Assemble one ``qd_archive.jsonl`` row from a survivor verdict."""
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

        Args:
            next_run: The freshly-appended ``IndividualRun`` whose prompt
                was just produced by the variation engine.
            parent_uuid_of: UUID of the run that produced ``next_run``'s
                prompt (the predecessor in the recursion).
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

        Args:
            uuid: Workflow UUID identifying the destination folder.
            snapshot: JSON-serialisable mapping of per-iteration metrics.
        """
        if not uuid:
            return
        try:
            write_run_metrics(Path(self.workflow_dir) / uuid, snapshot)
        except Exception as e:
            self.logger.warning(f"run_metrics write failed for {uuid}: {e}")

    def _persist_qd_archive(self, entry: dict) -> None:
        """Append one QD validation verdict to ``qd_archive.jsonl``."""
        try:
            append_jsonl(Path(self.workflow_dir) / "qd_archive.jsonl", entry)
        except Exception as e:
            self.logger.warning(f"qd_archive append failed: {e}")

    def _persist_variation_log(self, entry: dict) -> None:
        """Append one mutation-scope snapshot to ``variation_log.jsonl``."""
        try:
            append_jsonl(Path(self.workflow_dir) / "variation_log.jsonl", entry)
        except Exception as e:
            self.logger.warning(f"variation_log append failed: {e}")

    def _save_final_plots(self, assertion_history: list, reward_history: list, uuid: str) -> str:
        """Save final assertion plots.

        Args:
            assertion_history: Per-iteration ``[passed, total]`` pairs.
            reward_history: Per-iteration reward values.
            uuid: Workflow UUID whose folder receives the plot.

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

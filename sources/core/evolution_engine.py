"""
Darwinian Evolution of multi-agent workflows.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import random


from sources.utils.notify import PushNotifier
from sources.utils.pricing import PricingCalculator
from sources.utils.shared_visualization import SharedVisualizationData
from sources.utils.visualization import VisualizationUtils
from sources.evaluation.scenario_loader import ScenarioLoader

from sources.utils.workspace_management import WorkspaceManager
from .orchestrator import WorkflowOrchestrator
from .variation_engine import VariationEngine
from .workflow_info import WorkflowInfo
from .workflow_selection import WorkflowSelector
from .schema import IndividualRun, SelectionLog
from .selection import SelectionPressure
from sources.cli.pretty_print import (
    print_ok, print_warn, print_err, print_info,
    print_phase, print_section,
    print_iteration_header, print_box,
    print_summary, print_agent_answers,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)


def check_answer_success(answer: str) -> bool:
    """Check if an answer indicates success using pattern matching.
    Args:
        answer: The answer string to check
    Returns:
        bool: True if answer indicates success, False otherwise
    """
    answer_lower = str(answer).lower()
    failure_patterns = [
        r'\bfailed\b', r'\berror\b', r'\bfailure\b',
    ]
    return all(not re.search(pattern, answer_lower) for pattern in failure_patterns)

def evaluate_workflow_success(wf_info: WorkflowInfo, answers: list) -> bool:
    """
    Evaluate workflow success using multiple criteria.
    Args:
        wf_info: WorkflowInfo object containing state and evaluation results
        answers: List of answers from workflow agents
    Returns:
        bool: True if workflow is considered successful, False otherwise
    """
    if wf_info.state_result and 'evaluation' in wf_info.state_result:
        eval_data = wf_info.state_result['evaluation']

        if 'scenario' in eval_data and eval_data['scenario']:
            passed = eval_data['scenario'].get('passed_assertions', 0)
            total = eval_data['scenario'].get('total_assertions', 1)
            return (passed / total) >= 0.8
        if 'generic' in eval_data and eval_data['generic']:
            score = eval_data['generic'].get('overall_score', 0.0)
            return score >= 0.7
    if answers:
        return check_answer_success(answers[-1])
    return False

class EvolutionEngine:
    """Darwin Machine for evolution of workflow workflows."""
    def __init__(
        self,
        config,
        viz_utils: VisualizationUtils = None,
        process_id: int = None,
    ) -> None:
        from sources.evaluation.evaluator import WorkflowEvaluator
        self.config = config
        self.workflow_dir = config.workflow_dir
        self.model_pricing = config.model_pricing
        self.workflow_selector = WorkflowSelector(config)
        self.orchestrator = WorkflowOrchestrator(config)
        self.judge = WorkflowEvaluator(config)
        self.selection = SelectionPressure(min_improvement_threshold=0.05)
        self.variation = VariationEngine()
        self.notifier = PushNotifier(config.pushover_token, config.pushover_user)
        self.viz_utils = viz_utils or VisualizationUtils()
        self.process_id = process_id
        self.pricing = PricingCalculator(config)
        self.logger = logging.getLogger(__name__)

    def mockup(self, wf, goal):
        """
        Mockup mode: use existing workflow data without calling orchestrate_workflow
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
        flow_answers = self.extract_agents_behavior(wf.state_result)
        self.show_answers(flow_answers)
        print_ok(f"Mockup run completed with reward: {wf.overall_score:.1f}")
        return [mock_run]

    def load_phenotype_result(self, uuid: str) -> any:
        """Load the result of a previously executed workflow state.
        Args:
            uuid: UUID of the workflow state to load
        Returns:
            str: The output of the workflow state if found, None otherwise
        """
        try:
            with open(f"{self.workflow_dir}/{uuid}/state_result.json") as f:
                return json.loads(f.read().strip())
        except FileNotFoundError:
            print(f"Workflow state for UUID {uuid} not found in {self.workflow_dir}.")
            return None
        except Exception as e:
            raise ValueError(f"❌ Error reading workflow state: {str(e)}") from e

    def load_workflow_genotype_code(self, workflow_id: str) -> str:
        """
        Load the workflow code for a given workflow ID.
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

    def get_total_rewards(self, wf_state: any, eval_type: str) -> float:
        """Calculate the total rewards from the workflow state."""
        if not wf_state or not eval_type:
            return 0.0
        if eval_type == "generic":
            return wf_state["evaluation"]["generic"]["overall_score"]
        elif eval_type == "scenario":
            return wf_state["evaluation"]["scenario"]["score"]
        else:
            return 0.0

    def extract_agents_behavior(self, wf_state: any) -> str:
        """Extract the answers from the workflow state."""
        if not wf_state or "answers" not in wf_state:
            return ""

        flow_answers = (
            "\n".join(f"agent {n}: {str(x)[:256]}..." for (n, x) in zip(wf_state["step_name"], wf_state["answers"], strict=True))
            if isinstance(wf_state["answers"], list)
            else wf_state["answers"]
        )
        return flow_answers

    def show_answers(self, flow_answers) -> None:
        print_box(flow_answers, title="Workflow Agents Answers", color=YELLOW)

    def select_parent_workflow(
        self,
        goal: str,
        template_uuid: str = None,
        crossover_rate: float = 0.3,
        n_parents: int = 2,
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
            print(f"Workflow directory {self.workflow_dir} does not exist.")
            return [], False

        workflows = [f for f in os.listdir(self.workflow_dir)]
        if not workflows:
            print(f"No workflows found in {self.workflow_dir}.")
            return [], False

        # Explicit template → single parent, mutation only
        if template_uuid is not None:
            wf = WorkflowInfo(template_uuid, Path(f"{self.workflow_dir}/{template_uuid}"))
            return [wf], False

        # Evolutionary selection via SelectionPressure
        selected, use_crossover = self.workflow_selector.select_parent_workflows(
            goal=goal,
            selection_pressure=self.selection,
            n_parents=n_parents,
            crossover_rate=crossover_rate,
            threshold_similarity=0.9,
            threshold_score=0.1,
        )

        mode = "CROSSOVER" if use_crossover else "MUTATION"
        print_section("PARENTS SELECTION")
        print_info(f"Strategy: {self.selection.strategy.value} — Mode: {mode}")
        print_info(f"Selected {len(selected)} parent(s) from workflow pool")
        for i, wf in enumerate(selected):
            print_info(f"  Parent {i+1}: {wf.uuid}  (score={wf.overall_score:.2f})")

        return selected, use_crossover

    def get_genotype_instructions(self, goal, wf, max_iterations: int = 10) -> str:
        """
        Get the genotype prompt for either mutation workflow or generating the first individual.
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
        template_uuid: str | None = None,
        judge: bool = True,
        scenario_rubric: str = None,
        max_iteration: int = 1,
        enable_evolution: bool = False,
        original_task: str = None,
        single_agent_mode: bool = False,
        mockup_mode: bool = False
    ) -> list[IndividualRun]:
        """
        Start the learning process for achieving a specified goal.
        Args:
        - goal (str): The primary goal or objective to be accomplished (may be knowledge-wrapped).
         template_uuid (str | None, optional): UUID of a workflow template to use.
        - judge (bool, optional): Whether to enable judging mode for evaluation.
        - scenario_rubric (str, optional): ID of scenario for evaluation.
        - max_iteration (int): Maximum number of iterations.
        - enable_evolution (bool): Whether in learning mode. Will keep attempt at improving workflow score even if all agents report success state.
        - original_task (str, optional): Original unwrapped task for similarity matching.
        - mockup_mode (bool, optional): If True, use existing workflow data from select_parent_workflow
            instead of calling orchestrate_workflow. Useful for testing and debugging.
        """
        if max_iteration is None:
            max_iteration = 1
        if enable_evolution:
            max_iteration = self.config.max_learning_evolve_iterations if max_iteration <= 1 else max_iteration

        parents, _use_crossover = self.select_parent_workflow(
            goal, template_uuid=template_uuid
        )
        # For the initial run we always use the primary (best) parent
        wf = parents[0] if parents else None

        if mockup_mode:
            return self.mockup(wf, goal)

        # ── Workspace lifecycle: snapshot → clean → restore before first run ─
        workspace_mgr = WorkspaceManager(self.config, self.logger)
        workspace_mgr.begin_session()

        craft_instructions = self.get_genotype_instructions(goal, wf, max_iterations=max_iteration)

        rewards_history = []
        assertion_history = []  # Track [passed, total] per iteration
        self.viz_utils.create_rewards_curve_plot(goal)

        run0 = IndividualRun(
            goal=goal,
            prompt=craft_instructions,
            template_uuid=template_uuid,
            workflow_template=wf,
            max_depth=max_iteration,
            judge=judge,
            scenario_rubric=scenario_rubric,
            original_task=original_task
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
                key=lambda r: r.reward if r.reward is not None else 0.0,
                default=None,
            )
            if best_run and best_run.current_uuid:
                print_info(
                    f"Best run: {best_run.current_uuid} "
                    f"(score={best_run.reward:.3f if best_run.reward is not None else 'N/A'})"
                )
                workspace_mgr.restore_best(best_run.current_uuid)
            else:
                print_warn("No successful run found; workspace restored to initial state.")
                workspace_mgr.restore_best("")  # triggers fallback inside WorkspaceManager
        finally:
            workspace_mgr.cleanup()

        return runs

    async def evolve_generation(
        self,
        runs: list[IndividualRun],
        rewards_history: list[float] = None,
        assertion_history: list[list[int]] = None,
        enable_evolution: bool = False,
        single_agent_mode: bool = False,
        workspace_mgr: WorkspaceManager = None,
    ):
        """Run a evolution loop for the workflow."""
        self._log_iteration_start(runs[-1].goal, runs[-1].iteration_count, runs[-1].max_depth)

        iteration_start_time = time.time()
        on_error = False
        uuid = None
        current_iteration_cost = 0.0  # Cost for this iteration only, not cumulative

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
        if "WORKFLOW_GENERATION_ERROR" in run_stdout:
            print_err(f"Workflow generation failed:\n{run_stdout[256:]}")
            on_error = True
        # ── Snapshot workspace results produced by this run ───────────────────
        if workspace_mgr is not None and uuid:
            workspace_mgr.save_run_snapshot(uuid)

        if workflow_genotype_code:
            # Evaluate and calculate costs
            eval_type, current_iteration_cost = await self._evaluate_and_calculate_cost(
                executed, runs[-1].judge, uuid, runs[-1].answers, runs[-1].scenario_rubric, assertion_history
            )
            runs[-1].reward = wf_info.overall_score

        runs[-1].current_uuid = uuid
        runs[-1].answers = wf_info.answers
        runs[-1].state_result = wf_info.state_result
        flow_answers = self.extract_agents_behavior(wf_info.state_result)
        self.show_answers(flow_answers)
        rewards_history.append(wf_info.overall_score)

        # Update visualizations
        self._update_visualizations(
            rewards_history, assertion_history,
            runs[-1].goal, runs[-1].scenario_rubric, uuid
        )

        # Calculate cumulative cost and update runs[-1].cost for accurate tracking
        runs[-1].cost = runs[-1].cost + current_iteration_cost

        # Log and notify completion (show per-iteration cost, not cumulative)
        self._log_iteration_completion(
            runs[-1].iteration_count, runs[-1].max_depth, iteration_start_time,
            wf_info.overall_score, current_iteration_cost, runs[-1].goal, uuid, wf_info.state_result, rewards_history
        )

        all_success = evaluate_workflow_success(wf_info, runs[-1].answers)

        # Check termination conditions
        if runs[-1].iteration_count >= runs[-1].max_depth-1 and not on_error:
            print_info("Maximum recursive depth reached.")
            return runs
        if enable_evolution and wf_info.overall_score > self.config.learned_score_threshold:
            # reach learning threshold
            print_ok("Evolution engine done learning task.")
            self._save_final_plots(assertion_history, rewards_history, uuid)
            self.notifier.send_message(
                f"Done learning task: {wf_info.goal[:256]} \n"
                f"Final UUID: {uuid}\n"
                f"Iterations: {runs[-1].iteration_count + 1}/{runs[-1].max_depth}\n",
                title="Evolution done learning task.",
                priority=0
            )
            return runs
        elif not on_error:
            if not enable_evolution and all_success:
                self._save_final_plots(assertion_history, rewards_history, uuid)
                print_ok("evolution engine completed task successfully.")
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

        if use_crossover and len(parent_workflows) >= 2:
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
        else:
            # MUTATION — perturb the single best parent
            print_phase("MUTATION VARIATION", color=YELLOW)
            primary_parent = parent_workflows[0] if parent_workflows else None
            code = primary_parent.code if primary_parent else ""
            runs[-1].prompt = self.variation.mutation_prompt(
                task_goal, primary_parent, code, run_stdout,
                runs[-1].iteration_count, max_iterations=runs[-1].max_depth,
            )

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
            original_task=runs[-1].original_task  # PRESERVE original_task for workflow selection
        ))

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

    def _get_human_validation(self) -> bool:
        """Get human validation for continuing the workflow."""
        human_validation = input("Attempt to retry task? (yes/no): ").strip().lower()
        if human_validation not in ["yes", "y"]:
            print("Exiting evolution loop.")
            return False
        return True

    def _log_iteration_start(self, goal: str, iteration_count: int, max_depth: int):
        """Log the start of an iteration."""
        logger = logging.getLogger(__name__)
        print_iteration_header(iteration_count + 1, max_depth)
        print_box(goal, title="📋 CURRENT TASK", truncate=256)
        logger.info(f"[ITERATION START] {iteration_count + 1}/{max_depth} - {goal[:50]}...")

    async def _evaluate_and_calculate_cost(
        self, executed: bool, judge: bool, uuid: str,
        agent_answers: str, scenario_rubric: str, assertion_history: list
    ) -> tuple[str, float]:
        """Evaluate workflow and calculate cost."""
        logger = logging.getLogger(__name__)
        eval_type = None
        exec_cost = 0.0

        if judge and uuid:
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
        """Evaluate the workflow and update assertion history."""
        logger = logging.getLogger(__name__)
        print_phase("WORKFLOW EVALUATION PHASE")
        eval_start = time.time()
        eval_result = self.judge.evaluate(uuid=uuid, agent_answers=agent_answers, scenario_rubric=scenario_rubric)
        eval_type = 'scenario' if scenario_rubric else 'generic'
        eval_time = time.time() - eval_start
        logger.info(f"[WORKFLOW EVALUATION] {uuid}:\n{json.dumps(eval_result, indent=2)}")
        print_ok(f"Workflow evaluation completed in {eval_time:.3f}s")
        # Track assertion progress for scenario evaluation
        if scenario_rubric and isinstance(eval_result, dict) and assertion_history is not None:
            self._update_assertion_history(eval_result, assertion_history)
        return eval_type

    def _update_assertion_history(self, eval_result: dict, assertion_history: list):
        """Update assertion history with evaluation results."""
        passed = eval_result.get('passed_assertions', eval_result.get('earned_points', 0))
        total = eval_result.get('total_assertions', eval_result.get('total_points', 100))
        assertion_history.append([passed, total])
        pct = passed / total * 100 if total > 0 else 0
        print_info(f"📊 Assertions progress: {passed}/{total} ({pct:.0f}%)")

    def _update_visualizations(
        self, rewards_history: list, assertion_history: list,
        goal: str, scenario_rubric: str, uuid: str
    ):
        """Update all visualizations with current data."""
        if assertion_history:
            self._update_assertion_plot(assertion_history, scenario_rubric, uuid)
        elif rewards_history:
            self._update_rewards_plot(rewards_history)

    def _update_rewards_plot(self, rewards_history):
        self.viz_utils.update_rewards_curve(rewards_history)

    def _update_assertion_plot(
        self, assertion_history: list,
        scenario_rubric: str, uuid: str
    ):
        """Update assertion progress plot."""
        from sources.evaluation.scenario_loader import ScenarioLoader
        scenario = ScenarioLoader().load_scenario(scenario_rubric)
        total_assertions = len(scenario.get("assertions", [])) if scenario else 0
        self.viz_utils.update_assertion_progress_plot(assertion_history, total_assertions)
        plot_filename = f"{self.workflow_dir}/{uuid}/assertion_progress.png"
        self.viz_utils.save_plot(plot_filename)
        print_info(f"📊 Assertion progress plot updated: {plot_filename}")

    def _log_iteration_completion(
        self, iteration_count: int, max_depth: int, iteration_start_time: float,
        wf_rewards: float, exec_cost: float, goal: str, uuid: str,
        wf_state: any, rewards_history: list
    ):
        """Log iteration completion and send notification."""
        logger = logging.getLogger(__name__)
        iteration_time = time.time() - iteration_start_time
        logger.info(
            f"[ITERATION END] {iteration_count + 1}/{max_depth} completed in {iteration_time:.3f}s - "
            f"Rewards: {wf_rewards:.1f}, Cost: {exec_cost:.3f} USD"
        )
        print_summary(
            f"ITERATION {iteration_count + 1}/{max_depth} COMPLETE",
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

    def _save_final_plots(self, assertion_history: list, reward_history: list, uuid: str) -> str:
        """Save final assertion plots."""
        plot_filename = ""
        if assertion_history or reward_history:
            plot_filename = f"{self.workflow_dir}/{uuid}/reward_progress.png"
            self.viz_utils.save_plot(plot_filename)
        print_info(f"📊 Reward progress plot saved: {plot_filename}")
        return plot_filename

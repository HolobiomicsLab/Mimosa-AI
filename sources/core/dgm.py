"""
Darwin Godel Machine
"""

import json
import logging
import os
import time
from pathlib import Path

from sources.post_processing.evaluator import WorkflowEvaluator
from sources.utils.notify import PushNotifier
from sources.utils.pricing import PricingCalculator
from sources.utils.shared_visualization import SharedVisualizationData
from sources.utils.visualization import VisualizationUtils
from sources.utils.scenario_loader import ScenarioLoader

from .orchestrator import WorkflowOrchestrator
from .workflow_info import WorkflowInfo
from .workflow_selection import WorkflowSelector
from .schema import GodelRun


class GodelMachine:
    """Darwin Godel Machine for self-improvement workflows."""

    def __init__(
        self,
        config,
        viz_utils: VisualizationUtils = None,
        shared_viz_data: SharedVisualizationData = None,
        process_id: int = None,
    ) -> None:
        self.config = config
        self.workflow_dir = config.workflow_dir
        self.model_pricing = config.model_pricing
        self.workflow_selector = WorkflowSelector(config)
        self.orchestrator = WorkflowOrchestrator(config)
        self.judge = WorkflowEvaluator(config)
        self.notifier = PushNotifier(config.pushover_token, config.pushover_user)
        self.viz_utils = viz_utils or VisualizationUtils()
        self.process_id = process_id
        self.pricing = PricingCalculator(config)

    def load_wf_state_result(self, uuid: str) -> any:
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
        return None

    def load_workflow_code(self, workflow_id: str) -> str:
        """
        Load the workflow code for a given workflow ID.
        """
        workflow_path = f"{self.workflow_dir}/{workflow_id}"
        if not os.path.exists(workflow_path):
            raise ValueError(
                f"❌ Workflow for ID {workflow_id} not found in {self.workflow_dir}."
            )

        try:
            with open(f"{workflow_path}/workflow_code_{workflow_id}.py") as f:
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

    def get_flow_answers(self, wf_state: any) -> str:
        """Extract the answers from the workflow state."""
        if not wf_state or "answers" not in wf_state:
            return ""

        flow_answers = (
            "\n".join(f"agent {n}: {str(x)[:256]}..." for (n, x) in zip(wf_state["step_name"], wf_state["answers"], strict=True))
            if isinstance(wf_state["answers"], list)
            else wf_state["answers"]
        )
        return flow_answers

    def show_answers(self, flow_answers):
        print(f"\n\033[96m{'> WORKFLOW AGENTS ANSWERS':^60}\033[0m")
        print(f"\033[96m{'─' * 60}\033[0m")
        print(f"\033[96m{flow_answers}\033[0m")
        print(f"\033[96m{'─' * 60}\033[0m\n")

    def improvement_prompt(
        self,
        goal: str,
        wf_state: any,
        flow_code: str,
        run_stdout: str,
        iteration_count: int,
    ) -> str:
        flow_answers = ""

        if wf_state is not None:
            flow_answers = self.get_flow_answers(wf_state)
            self.show_answers(flow_answers)
        else:
            flow_answers = run_stdout.strip()

        improv_prompt = "Previous attempt failed. Learn from mistakes and improve the multi-agent workflow."
        if flow_code is not None:
            improv_prompt = "\n".join([
                "## WORKFLOW ATTEMPT ANALYSIS:",
                "Your previous attempt at generating a workflow did not succeed or was incomplete.",
                "Your goal was: ",
                goal,
                "Reflect on your previous attempt and identify what went wrong.",
                "\n",
                "## Previous workflow code:",
                "<python>",
                flow_code,
                "</python>",
                "\n",
                "## Previous execution results:",
                "This is the answer from each agent during the last execution.",
                "<results>",
                flow_answers,
                "</results>",
                "\n",
                "## FAILURE ANALYSIS:",
                "1. Analyze the previous workflow code and its execution results.",
                "2. Evaluate task completion. Did the workflow really achieve the goal?",
                "3. Think of specific improvements to the workflow code to address these failures.",
                "\n",
                "## IMPROVEMENT SUGGESTIONS:",
                "\n",
                "1. If the workflow code execution failed, analyze the error messages and fix the python code.",
                "2. If an agent failed due to limitation of its tool capabilities, consider an alternative tool that would allow to complete the task (eg: use web search tool if arxiv search doesn't seem to work).",
                "3. If an agent failed due to lack of information, consider adding an initial research step to gather more information before attempting the main task.",
                "4. If agent didn't behave as expected, consider changing the prompt or the agent role to better align with the task requirements.",
                "5. Consider adding fallback agents that can take over if a primary agent fails.",
                "6. Consider adding error handling and validation steps to ensure robustness.",
                "7. Consider breaking down complex tasks into smaller, manageable sub-tasks.",
                "8. Consider adding feedback loops where agents can review and refine each other's outputs.",
                "9. Always Consider alternative strategies. Tool seem to fail or not fit ? Then explore other tools or approaches that might be more effective."
                "\n",
                "Getting invalid syntax (<workflow>, line 284) error with no clear message ? It may be because the code and prompt exceed your token limits, make sacrifice for shorter prompt or a simpler workflow.",
                "Generate an IMPROVED version that addresses identified failure modes or with added steps for reaching the goal.",
                "The new version must be different from the previous attempt.\n"
            ])

        return "".join(
            [
                f"Attempt {iteration_count + 1} of workflow generation.\n",
                improv_prompt,
                "Target goal:\n",
                goal,
            ]
        )

    def select_workflow_template(self, goal, template_uuid: str = None) -> str:
        """Select and load a workflow template from the workflow directory or by UUID.

        Args:
            template_uuid: Optional UUID of workflow template to load
        Returns:
            str: The workflow template content if found, None otherwise
        """
        if not os.path.exists(self.workflow_dir):
            print(f"Workflow directory {self.workflow_dir} does not exist.")
            return None
        workflows = [f for f in os.listdir(self.workflow_dir)]
        if not workflows:
            print(f"No workflows found in {self.workflow_dir}.")
            return None

        # default to selecting best workflow if no template UUID provided
        if template_uuid is None:
            candidates = self.workflow_selector.select_best_workflows(
                goal=goal,
                threshold_similary=0.8,
                threshod_score=0.0,
            )
            print(f"\n\033[96m{'🎯 WORKFLOW SELECTION':^60}\033[0m")
            print(f"\033[96m{'─' * 60}\033[0m")
            print(f"\033[96mSelected {len(candidates)} candidates.\033[0m")
            print(f"\033[96mTop candidate: {candidates[0].uuid if candidates else str(None)}\033[0m")
            print(f"\033[96m{'─' * 60}\033[0m\n")
            return WorkflowInfo(candidates[0].uuid, Path(f"{self.workflow_dir}/{candidates[0].uuid}")) if candidates else None
        # load specified template UUID
        return WorkflowInfo(template_uuid, Path(f"{self.workflow_dir}/{template_uuid}"))

    async def start_dgm(
        self,
        goal: str,
        template_uuid: str | None = None,
        judge: bool = False,
        scenario_id: str = None,
        max_iteration: int = 5,
        learning_mode: bool = True
    ) -> list[GodelRun]:
        """
        Start the Dynamic Goal Management (DGM) process for achieving a specified goal.
        Args:
        - goal (str): The primary goal or objective to be accomplished.
         template_uuid (str | None, optional): UUID of a workflow template to use.
        - judge (bool, optional): Whether to enable judging mode for evaluation.
        - answer (str, optional): A predefined correct answer for evaluation system.
        """

        wf = self.select_workflow_template(
            goal, template_uuid=template_uuid
        )

        if wf:
            craft_instructions = self.improvement_prompt(
                goal, wf.state_result, wf.code, "", 0
            )
        else:
            craft_instructions = goal

        rewards_history = []
        assertion_history = []  # Track [passed, total] per iteration

        if self.process_id is None and max_iteration > 1:
            print("Setup reward visualization.")
            self.viz_utils.create_rewards_curve_plot(goal)
        elif scenario_id and judge:
            print("Setup scenario visualization.")
            scenario = ScenarioLoader().load_scenario(scenario_id)
            if scenario:
                total_assertions = len(scenario.get("assertions", []))
                self.viz_utils.create_assertion_progress_plot(
                    scenario_id, total_assertions
                )

        run0 = GodelRun(
            goal=goal,
            prompt=craft_instructions,
            template_uuid=template_uuid,
            workflow_template=wf,
            max_depth=max_iteration,
            judge=judge,
            scenario_id=scenario_id,
        )

        return await self.recursive_self_improvement(
            [run0],
            rewards_history=rewards_history,
            assertion_history=assertion_history,
            learning_mode=learning_mode
        )

    async def recursive_self_improvement(
        self,
        runs: list[GodelRun],
        rewards_history: list[float] = None,
        assertion_history: list[list[int]] = None,
        learning_mode: bool = False
    ):
        """Run a self-improvement loop for the workflow."""
        self._log_iteration_start(runs[-1].goal, runs[-1].iteration_count, runs[-1].max_depth)

        iteration_start_time = time.time()
        uuid = None

        # Execute workflow
        run_stdout, uuid, workflow_code, executed = await self.orchestrator.orchestrate_workflow(
            goal=runs[-1].goal,
            craft_instructions=runs[-1].prompt,
        )
        wf_info = WorkflowInfo(uuid, Path(f"{self.workflow_dir}/{uuid}"))

        runs[-1].current_uuid = uuid
        runs[-1].answers = wf_info.answers
        runs[-1].state_result = wf_info.state_result
        flow_answers = self.get_flow_answers(wf_info.state_result)
        self.show_answers(flow_answers)

        # Evaluate and calculate costs
        eval_type, total_cost = await self._evaluate_and_calculate_cost(
            executed, runs[-1].judge, uuid, runs[-1].answers, runs[-1].scenario_id, assertion_history
        )

        # Update tracking data
        rewards_history.append(wf_info.overall_score)
        # Update visualizations
        self._update_visualizations(
            rewards_history, assertion_history,
            runs[-1].goal, runs[-1].scenario_id, uuid
        )
        # Log and notify completion
        self._log_iteration_completion(
            runs[-1].iteration_count, runs[-1].max_depth, iteration_start_time,
            wf_info.overall_score, total_cost, runs[-1].goal, uuid, wf_info.state_result, rewards_history
        )

        if runs[-1].answers:
            all_success =  all(["success" in str(x).lower() for x in runs[-1].answers])
        else:
            all_success = False
        # Check termination conditions
        if (runs[-1].iteration_count >= runs[-1].max_depth-1 or all_success) and not learning_mode:
            self._save_final_plots(assertion_history, rewards_history, uuid)

            # Send success notification when all iterations complete
            if all_success:
                self.notifier.send_message(
                    f"DGM completed successfully!\n"
                    f"Goal: {runs[-1].goal[:128]}...\n"
                    f"Final UUID: {uuid}\n"
                    f"Iterations: {runs[-1].iteration_count + 1}/{runs[-1].max_depth}\n"
                    f"All workflows successful!",
                    title=f"DGM success - {uuid}",
                    priority=0
                )

            return runs

        # Continue recursion
        runs[-1].prompt = self.improvement_prompt(
            runs[-1].goal, wf_info.state_result, workflow_code, run_stdout, runs[-1].iteration_count
        )

        # add godel run class instance to list
        runs.append(GodelRun(
            goal=runs[-1].goal,
            prompt=runs[-1].prompt,
            cost=total_cost,
            current_uuid=uuid,
            template_uuid=None,
            workflow_template=runs[-1].workflow_template if wf_info.state_result else None,
            iteration_count=runs[-1].iteration_count + 1,
            max_depth=runs[-1].max_depth,
            judge=runs[-1].judge,
            answers=wf_info.answers,
            state_result=wf_info.state_result,
            scenario_id=runs[-1].scenario_id
        ))

        runs = await self.recursive_self_improvement(
            runs,
            rewards_history=rewards_history,
            assertion_history=assertion_history
        )

        runs[-1].plot = self._save_final_plots(assertion_history, rewards_history, uuid)
        return runs

    def _get_human_validation(self) -> bool:
        """Get human validation for continuing the workflow."""
        human_validation = input("Attempt to retry task? (yes/no): ").strip().lower()
        if human_validation not in ["yes", "y"]:
            print("Exiting self-improvement loop.\n")
            return False
        return True

    def _log_iteration_start(self, goal: str, iteration_count: int, max_depth: int):
        """Log the start of an iteration."""
        logger = logging.getLogger(__name__)

        print(f"\n\033[94m{'=' * 60}\033[0m")
        print(f"\033[94mITERATION {iteration_count + 1}/{max_depth} - Self-Improvement Loop.\n\033[0m"
              f"\033[94mDGM Will attempt to retry and improve workflow on same task.\033[0m")
        print(f"\033[94m{'=' * 60}\033[0m")
        print(f"\n\033[94m{'📋 CURRENT TASK':^60}\033[0m")
        print(f"\033[94m{'─' * 60}\033[0m")
        goal_lines = goal.split('\n')
        for line in goal_lines:
            if len(line) <= 256:
                print(f"\033[94m  {line}\033[0m")
            else:
                truncated = line[:256]
                remaining = len(line) - 256
                print(f"\033[94m  {truncated}...({remaining} remaining characters not displayed)\033[0m")
        print(f"\033[94m{'─' * 60}\033[0m\n")
        logger.info(f"[ITERATION START] {iteration_count + 1}/{max_depth} - {goal[:50]}...")

    async def _evaluate_and_calculate_cost(
        self, executed: bool, judge: bool, uuid: str,
        answer: str, scenario_id: str, assertion_history: list
    ) -> tuple[str, float]:
        """Evaluate workflow and calculate cost."""
        logger = logging.getLogger(__name__)
        eval_type = None
        total_cost = 0.0

        if executed and judge and uuid:
            eval_type = await self._evaluate_workflow(uuid, answer, scenario_id, assertion_history)

        # Calculate cost regardless of execution success
        # This includes workflow generation LLM costs even when execution fails
        cost_start = time.time()
        total_cost = self.pricing.calculate_cost(uuid)
        cost_time = time.time() - cost_start
        logger.info(f"[WORKFLOW COST] {uuid} cost calculated in {cost_time:.3f}s")

        return eval_type, total_cost

    async def _evaluate_workflow(
        self, uuid: str, answer: str, scenario_id: str, assertion_history: list
    ) -> str:
        """Evaluate the workflow and update assertion history."""
        logger = logging.getLogger(__name__)

        print(f"\n\033[94m{'⚖️  WORKFLOW EVALUATION PHASE':^80}\033[0m")
        print(f"\033[94m{'=' * 80}\033[0m")

        eval_start = time.time()
        eval_result = self.judge.evaluate(uuid=uuid, answer=answer, scenario_id=scenario_id)
        eval_type = 'scenario' if scenario_id else 'generic'
        eval_time = time.time() - eval_start

        logger.info(f"[WORKFLOW EVALUATION] {uuid} evaluated in {eval_time:.3f}s")
        print(f"\033[94m✅ Workflow evaluation completed in {eval_time:.3f}s\033[0m")

        # Track assertion progress for scenario evaluation
        if scenario_id and isinstance(eval_result, dict) and assertion_history is not None:
            self._update_assertion_history(eval_result, assertion_history)

        return eval_type

    def _update_assertion_history(self, eval_result: dict, assertion_history: list):
        """Update assertion history with evaluation results."""
        passed = eval_result.get('passed_assertions', 0)
        total = eval_result.get('total_assertions', 0)
        assertion_history.append([passed, total])
        print(f"\033[94m📊 Assertions Progress: {passed}/{total} "
              f"({passed/total*100 if total > 0 else 0:.0f}%)\033[0m")

    def _update_visualizations(
        self, rewards_history: list, assertion_history: list,
        goal: str, scenario_id: str, uuid: str
    ):
        """Update all visualizations with current data."""
        # Update assertion plot if available
        if assertion_history:
            self._update_assertion_plot(assertion_history, scenario_id, uuid)
        elif rewards_history:
            self._update_rewards_plot(rewards_history)

    def _update_rewards_plot(self, rewards_history):
        self.viz_utils.update_rewards_curve(rewards_history)

    def _update_assertion_plot(
        self, assertion_history: list,
        scenario_id: str, uuid: str
    ):
        """Update assertion progress plot."""
        from sources.utils.scenario_loader import ScenarioLoader

        scenario = ScenarioLoader().load_scenario(scenario_id)
        total_assertions = len(scenario.get("assertions", [])) if scenario else 0

        self.viz_utils.update_assertion_progress_plot(
            assertion_history, total_assertions
        )

        # Save plot after each update for real-time monitoring
        plot_filename = f"{self.workflow_dir}/{uuid}/assertion_progress.png"
        self.viz_utils.save_plot(plot_filename)
        print(f"\033[94m📊 Assertion progress plot updated: {plot_filename}\033[0m")

    def _log_iteration_completion(
        self, iteration_count: int, max_depth: int, iteration_start_time: float,
        wf_rewards: float, total_cost: float, goal: str, uuid: str,
        wf_state: any, rewards_history: list
    ):
        """Log iteration completion and send notification."""
        logger = logging.getLogger(__name__)
        iteration_time = time.time() - iteration_start_time

        logger.info(
            f"[ITERATION END] {iteration_count + 1}/{max_depth} completed in {iteration_time:.3f}s - "
            f"Rewards: {wf_rewards:.1f}, Cost: {total_cost:.3f} USD"
        )

        print(f"\n\033[94m{'-' * 60}\033[0m")
        print(f"\033[94mTotal rewards: {wf_rewards:.1f}\033[0m")
        print(f"\033[94mTotal cost: {total_cost:.6f} USD\033[0m")
        print(f"\033[94mIteration time: {iteration_time:.3f}s\033[0m")
        print(f"\033[94m{'-' * 60}\033[0m\n")

        self.notifier.send_message(
            f"Iteration {iteration_count + 1} completed.\n"
            f"Goal: {goal[:128]}...\n"
            f"Cost: {total_cost:.6f} USD.\n"
            f"Rewards history: {rewards_history}"
            f"Answers: {self.get_flow_answers(wf_state)}\n",
            title=f"Workflow {uuid} completed.",
        )

    def _save_final_plots(self, assertion_history: list, reward_history: list, uuid: str) -> str:
        """Save final assertion plots."""
        plot_filename = ""
        if assertion_history or reward_history:
            plot_filename = f"{self.workflow_dir}/{uuid}/reward_progress.png"
            self.viz_utils.save_plot(plot_filename)
        print(f"\033[94m📊 Assertion progress plot saved to: {plot_filename}\033[0m")
        return plot_filename

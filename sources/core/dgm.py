"""
Darwin Godel Machine
"""

import json
import logging
import os
import time

from sources.core.evaluator import WorkflowEvaluator
from sources.utils.notify import PushNotifier
from sources.utils.pricing import PricingCalculator
from sources.utils.shared_visualization import SharedVisualizationData
from sources.utils.visualization import VisualizationUtils

from .orchestrator import WorkflowOrchestrator
from .workflow_selection import WorkflowSelector


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
        self.shared_viz_data = shared_viz_data
        self.process_id = process_id
        self.pricing = PricingCalculator(config)

    def load_flow_state_result(self, uuid: str) -> any:
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

    def get_total_rewards(self, flow_state: any, eval_type: str) -> float:
        """Calculate the total rewards from the workflow state."""
        if not flow_state or not eval_type:
            return 0.0
        if eval_type == "generic":
            return flow_state["evaluation"]["generic"]["overall_score"]
        elif eval_type == "scenario":
            return flow_state["evaluation"]["scenario"]["score"]
        else:
            return 0.0

    def get_flow_answers(self, flow_state: any) -> str:
        """Extract the answers from the workflow state."""
        if not flow_state or "answers" not in flow_state:
            return ""

        return (
            "\n".join(str(x) for x in flow_state["answers"])
            if isinstance(flow_state["answers"], list)
            else flow_state["answers"]
        )

    def improvement_prompt(
        self,
        goal: str,
        flow_state: any,
        flow_code: str,
        run_stdout: str,
        iteration_count: int,
    ) -> str:
        flow_answers = ""

        if flow_state is not None:
            flow_answers = self.get_flow_answers(flow_state)
        else:
            flow_answers = run_stdout.strip()
        improv_prompt = "You must generate a multi-agent workflow for the goal."
        if flow_code is not None:
            improv_prompt = "\n".join(
                [
                    "Previously written workflow code:",
                    flow_code,
                    "Previous attempt resulted in agents ending with following answers:",
                    flow_answers,
                    "You must improve the workflow based on previous execution results.",
                    "Only change a prompt, add an agent, change a tool, etc.. ",
                ]
            )

        return "".join(
            [
                f"Attempt {iteration_count + 1} of workflow generation.\n",
                improv_prompt,
                "Target goal:",
                goal,
            ]
        )

    def select_workflow_template(self, goal_prompt, template_uuid: str = None) -> str:
        """Select and load a workflow template by UUID.

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
        if template_uuid is None:
            candidates = self.workflow_selector.select_best_workflows(
                goal=goal_prompt,
            )
            print(f"Selected {len(candidates)} candidates for goal '{goal_prompt}'")
            return candidates[0].code if candidates else None
        workflow_path = f"{self.workflow_dir}/{template_uuid}"
        if not os.path.exists(workflow_path):
            raise ValueError(
                f"❌ Workflow for ID {template_uuid} not found in {self.workflow_dir}."
            )

        try:
            with open(
                f"{workflow_path}/workflow_code_{template_uuid}.py",
            ) as f:
                return f.read()
        except FileNotFoundError as e:
            raise ValueError(
                f"❌ Workflow code file not found for ID {template_uuid} in {workflow_path}."
            ) from e
        except Exception as e:
            raise ValueError(f"❌ Error reading workflow template: {str(e)}") from e

    async def start_dgm(
        self,
        goal_prompt: str,
        template_uuid: str | None = None,
        judge: bool = False,
        answer: str = None,
        scenario_id: str = None,
        human_validation: bool = False,
        max_iteration: int = 5,
    ):
        """
        Start the Dynamic Goal Management (DGM) process for achieving a specified goal.
        Args:
        - goal_prompt (str): The primary goal or objective to be accomplished.
         template_uuid (str | None, optional): UUID of a workflow template to use.
        - judge (bool, optional): Whether to enable judging mode for evaluation.
        - answer (str, optional): A predefined correct answer for evaluation system.
        - human_validation (bool, optional): Whether human validation is required.
        """

        template = self.select_workflow_template(
            goal_prompt, template_uuid=template_uuid
        )

        print(f"\n{'📋 CURRENT GOAL':^60}")
        print(f"{'─' * 60}")
        print(f"  {goal_prompt}")
        print(f"{'─' * 60}\n")

        rewards_history = []
        plot_data = None

        if self.shared_viz_data and self.process_id is not None:
            plot_data = None
        else:
            plot_data = self.viz_utils.create_rewards_curve_plot(goal_prompt)

        return await self.recursive_self_improvement(
            goal_prompt,
            goal_prompt,
            template_uuid=template_uuid,
            workflow_template=template,
            max_depth=max_iteration,
            judge=judge,
            answer=answer,
            scenario_id=scenario_id,
            need_human_validation=human_validation,
            rewards_history=rewards_history,
            plot_data=plot_data,
        )

    async def recursive_self_improvement(
        self,
        goal,
        prompt: str,
        template_uuid: str | None = None,
        workflow_template: str | None = None,
        iteration_count: int = 0,
        max_depth: int = 5,
        judge: bool = False,
        need_human_validation: bool = False,
        rewards_history: list[float] = None,
        plot_data: tuple = None,
        answer: str = None,
        scenario_id: str = None,
    ):
        """Run a self-improvement loop for the workflow.

        Args:
            prompt: The goal prompt for workflow generation
            goal: The goal to achieve with the workflow
            template_uuid: Optional UUID of workflow template to use
            workflow_template: Optional workflow template code to use
            iteration_count: Current iteration count (for recursion)
            max_depth: Maximum depth of recursion
            plot_data: Tuple containing (fig, ax, line) for VisualizationUtils plotting
        Returns:
            str: Final execution status message
        """
        logger = logging.getLogger(__name__)
        iteration_start_time = time.time()
        total_cost = 0.0
        uuid = None  # Initialize uuid to avoid undefined reference

        if iteration_count > 0 and need_human_validation:
            human_validation = (
                input("Continue with next iteration? (yes/no): ").strip().lower()
            )
            if human_validation not in ["yes", "y"]:
                print("Exiting self-improvement loop.\n")
                return template_uuid

        print(f"\n\033[94m{'=' * 60}\033[0m")
        print(
            f"\033[94mITERATION {iteration_count + 1}/{max_depth} - Self-Improvement Loop\033[0m"
        )
        print(f"\033[94m{'=' * 60}\033[0m")
        print(f"\n\033[94m{'📋 CURRENT GOAL':^60}\033[0m")
        print(f"\033[94m{'─' * 60}\033[0m")
        print(f"\033[94m  {goal}\033[0m")
        print(f"\033[94m{'─' * 60}\033[0m\n")

        logger.info(
            f"[ITERATION START] {iteration_count + 1}/{max_depth} - {goal[:50]}..."
        )

        run_stdout, uuid, executed = await self.orchestrator.orchestrate_workflow(
            goal_prompt=prompt,
            workflow_template=workflow_template if iteration_count == 0 else None,
        )
        eval_type = None
        if executed:
            if judge:
                print(f"\n\033[94m{'⚖️  WORKFLOW EVALUATION PHASE':^80}\033[0m")
                print(f"\033[94m{'=' * 80}\033[0m")
                eval_start = time.time()
                eval_type = self.judge.evaluate(
                    uuid=uuid, answer=answer, scenario_id=scenario_id
                )
                eval_time = time.time() - eval_start
                logger.info(
                    f"[WORKFLOW EVALUATION] {uuid} evaluated in {eval_time:.3f}s"
                )
                print(
                    f"\033[94m✅ Workflow evaluation completed in {eval_time:.3f}s\033[0m"
                )

            cost_start = time.time()
            total_cost = self.pricing.calculate_cost(uuid)
            cost_time = time.time() - cost_start
            logger.info(f"[WORKFLOW COST] {uuid} cost calculated in {cost_time:.3f}s")

        flow_state = self.load_flow_state_result(uuid)
        flow_rewards = self.get_total_rewards(flow_state, eval_type)
        rewards_history.append(flow_rewards)

        # Update visualization - either shared (parallel mode) or individual plot
        if self.shared_viz_data and self.process_id is not None:
            iterations = list(range(1, len(rewards_history) + 1))
            self.shared_viz_data.write_curve_data(
                process_id=self.process_id,
                iterations=iterations,
                rewards=rewards_history,
                goal=goal,
                status="running",
            )
        elif plot_data:
            self.viz_utils.update_rewards_curve(plot_data, rewards_history)

        iteration_time = time.time() - iteration_start_time
        logger.info(
            f"[ITERATION END] {iteration_count + 1}/{max_depth} completed in {iteration_time:.3f}s - Rewards: {flow_rewards:.1f}, Cost: {total_cost:.3f} USD"
        )

        print(f"\n\033[94m{'-' * 60}\033[0m")
        print(f"\033[94mTotal rewards: {flow_rewards:.1f}\033[0m")
        print(f"\033[94mTotal cost: {total_cost:.3f} USD\033[0m")
        print(f"\033[94mIteration time: {iteration_time:.3f}s\033[0m")
        print(f"\033[94m{'-' * 60}\033[0m\n")
        self.notifier.send_message(
            f"Iteration {iteration_count + 1} completed.\n \
            Goal: {goal}\n \
            UUID: {uuid}\n \
            Reward : {flow_rewards:.2f}\n \
            Answers: {self.get_flow_answers(flow_state)}\n \
            Cost: {total_cost:.3f} USD.\n \
            Rewards history: {rewards_history}",
            title=f"Workflow {uuid} completed.",
        )

        if iteration_count >= max_depth:
            print(f"Maximum iterations reached ({max_depth}).")
            return uuid

        flow_code = self.select_workflow_template(
            goal_prompt=goal, template_uuid=template_uuid
        )
        prompt = self.improvement_prompt(
            goal, flow_state, flow_code, run_stdout, iteration_count
        )
        await self.recursive_self_improvement(
            goal,
            prompt,
            template_uuid=None,
            workflow_template=flow_code if flow_state else None,
            iteration_count=iteration_count + 1,
            max_depth=max_depth,
            judge=judge,
            need_human_validation=need_human_validation,
            rewards_history=rewards_history,
            plot_data=plot_data,
            scenario_id=scenario_id,
        )
        return uuid

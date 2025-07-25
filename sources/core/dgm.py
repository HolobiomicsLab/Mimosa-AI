"""
Darwin Godel Machine
"""

import os
import json

from sources.core.judge import WorkflowJudge
from sources.utils.visualization import VisualizationUtils
from sources.utils.shared_visualization import SharedVisualizationData
from sources.utils.notify import PushNotifier

from .orchestrator import WorkflowOrchestrator
from .workflow_selection import WorkflowSelector


class GodelMachine:
    """Darwin Godel Machine for self-improvement workflows."""

    def __init__(self, config, viz_utils: VisualizationUtils = None, shared_viz_data: SharedVisualizationData = None, process_id: int = None) -> None:
        self.config = config
        self.workflow_dir = config.workflow_dir
        self.model_pricing = config.model_pricing
        self.workflow_selector = WorkflowSelector(config)
        self.orchestrator = WorkflowOrchestrator(config)
        self.judge = WorkflowJudge(config)
        self.notifier = PushNotifier(config.pushover_token, config.pushover_user)
        self.viz_utils = viz_utils or VisualizationUtils()
        self.shared_viz_data = shared_viz_data
        self.process_id = process_id

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
            print(
                f"Workflow state for UUID {uuid} not found in {self.workflow_dir}."
            )
            return None
        except Exception as e:
            raise ValueError(f"❌ Error reading workflow state: {str(e)}") from e
        return None

    def load_workflow_code(self, uuid: str) -> str:
        """
        Load the workflow code for a given UUID.
        """
        try:
            with open(f"{self.workflow_dir}/{uuid}/workflow_code_{uuid}.py") as f:
                return f.read()
        except FileNotFoundError as e:
            raise ValueError(
                f"❌ Workflow code for UUID {uuid} not found in {self.workflow_dir}."
            ) from e
        except Exception as e:
            raise ValueError(f"❌ Error reading workflow code: {str(e)}") from e

    def get_total_rewards(self, flow_state: any) -> float:
        """Calculate the total rewards from the workflow state."""
        if not flow_state:
            return 0.0
        if "evaluation_scores" not in flow_state:
            return 0.0
        return flow_state["evaluation_scores"]["overall_score"]

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
            flow_answers = (
                run_stdout.strip()
            )

        return f"""
You must iteratively improve the workflow based on the previous execution results.

Previous workflow code :
{flow_code}

Previous generation attempt ({iteration_count}) result in the following output:

{flow_answers}

Given this output, you need to improve the workflow.
Do not change the whole workflow. Change a prompt, add an agent, change a tool, etc.. 
Create a new workflow that improves the previous one with the single change you choose.
Add extensive comments in the code to explain your changes.
        """

    def select_workflow_template(self, goal_prompt, template_uuid: str = None) -> str:
        """Select and load a workflow template by UUID.

        Args:
            template_uuid: Optional UUID of workflow template to load
        Returns:
            str: The workflow template content if found, None otherwise
        """
        if not os.path.exists(self.workflow_dir):
            return None
        workflows = [f for f in os.listdir(self.workflow_dir)]
        if not workflows:
            return None
        if template_uuid is None:
            candidates = self.workflow_selector.select_best_workflows(
                goal=goal_prompt,
            )
            return candidates[0].code if candidates else None
        try:
            with open(
                f"{self.workflow_dir}/{template_uuid}/workflow_code_{template_uuid}.py",
            ) as f:
                return f.read()
        except FileNotFoundError as e:
            raise ValueError(
                f"❌ Workflow for UUID {template_uuid} not in {self.workflow_dir}."
            ) from e
        except Exception as e:
            raise ValueError(f"❌ Error reading workflow template: {str(e)}") from e

    async def start_dgm(
        self,
        goal_prompt: str,
        template_uuid: str | None = None,
        judge: bool = False,
        human_validation: bool = False,
    ):
        template = self.select_workflow_template(goal_prompt,
                                                 template_uuid=template_uuid)
        
        rewards_history = []
        plot_data = None
        
        if self.shared_viz_data and self.process_id is not None:
            plot_data = None
        else:
            plot_data = self.viz_utils.create_rewards_curve_plot(goal_prompt)
        
        await self.recursive_self_improvement(
            goal_prompt,
            goal_prompt,
            template_uuid=template_uuid,
            workflow_template=template,
            max_depth=10,
            judge=judge,
            need_human_validation=human_validation,
            rewards_history=rewards_history,
            plot_data=plot_data,
        )

    async def recursive_self_improvement(
        self,
        prompt: str,
        goal: str,
        template_uuid: str | None = None,
        workflow_template: str | None = None,
        iteration_count: int = 0,
        max_depth: int = 5,
        judge: bool = False,
        need_human_validation: bool = False,
        rewards_history: list[float] = None,
        plot_data: tuple = None,
    ) -> str:
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
        flow_output = ""
        total_cost = 0.0

        if iteration_count > 0 and need_human_validation:
            human_validation = (
                input("Continue with next iteration? (yes/no): ").strip().lower()
            )
            if human_validation not in ["yes", "y"]:
                print("Exiting self-improvement loop.\n")
                return flow_output

        print(f"\n{'=' * 60}")
        print(f"ITERATION {iteration_count + 1}/{max_depth} - Self-Improvement Loop")
        print(f"{'=' * 60}")
        print(f"\n{'📋 CURRENT GOAL':^60}")
        print(f"{'─' * 60}")
        print(f"  {goal}")
        print(f"{'─' * 60}\n")

        run_stdout, uuid, executed = await self.orchestrator.orchestrate_workflow(
            goal_prompt=prompt,
            workflow_template=workflow_template if iteration_count == 0 else None
        )
        if executed:
            if judge:
                self.judge.evaluate(uuid)
            total_cost = self.judge.calculate_cost(uuid)

        flow_state = self.load_flow_state_result(uuid)
        flow_rewards = self.get_total_rewards(flow_state)
        rewards_history.append(flow_rewards)
        
        # Update visualization - either shared (parallel mode) or individual plot
        if self.shared_viz_data and self.process_id is not None:
            iterations = list(range(1, len(rewards_history) + 1))
            self.shared_viz_data.write_curve_data(
                process_id=self.process_id,
                iterations=iterations,
                rewards=rewards_history,
                goal=goal,
                status="running"
            )
        elif plot_data:
            self.viz_utils.update_rewards_curve(plot_data, rewards_history)
        
        print(f"\n{'-' * 60}")
        print(f"Total rewards: {flow_rewards:.1f}")
        print(f"Total cost: {total_cost:.3f} USD")
        print(f"{'-' * 60}\n")
        self.notifier.send_message(
            f"Iteration {iteration_count + 1} completed.\n \
            Cost: {total_cost:.3f} USD.\n \
            Score : {self.get_total_rewards(flow_state):.2f}",
            title=f"Workflow {uuid} completed.",
        )

        #flow_code = self.load_workflow_code(template_uuid if template_uuid else uuid)
        flow_code = self.select_workflow_template(goal_prompt=goal,
                                                  template_uuid=template_uuid)
        prompt = self.improvement_prompt(
            goal, flow_state, flow_code, run_stdout, iteration_count
        )
        if iteration_count >= max_depth:
            print(f"Maximum iterations reached ({max_depth}).")
            return flow_output
        await self.recursive_self_improvement(
            prompt,
            goal,
            template_uuid=None,
            workflow_template=flow_code if flow_state else None,
            iteration_count=iteration_count + 1,
            max_depth=max_depth,
            judge=judge,
            need_human_validation=need_human_validation,
            rewards_history=rewards_history,
            plot_data=plot_data,
        )
        return flow_output

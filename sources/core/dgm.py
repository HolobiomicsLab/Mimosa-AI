"""
Darwin Godel Machine
"""

import json
import os
from pathlib import Path

from sources.core.judge import WorkflowJudge

from .notify import PushNotifier
from .orchestrator import WorkflowOrchestrator


class GodelMachine:
    """Darwin Godel Machine for self-improvement workflows."""

    def __init__(self, config) -> None:
        self.config = config
        self.workflow_dir = config.workflow_dir
        self.model_pricing = config.model_pricing
        self.orchestrator = WorkflowOrchestrator(config)
        self.judge = WorkflowJudge(config)
        self.notifier = PushNotifier(config.pushover_token, config.pushover_user)

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
                f"❌ Workflow state for UUID {uuid} not found in {self.workflow_dir}."
            )
            return None
        except Exception as e:
            raise ValueError(f"❌ Error reading workflow state: {str(e)}")
        return None

    def load_workflow_code(self, uuid: str) -> str:
        """
        Load the workflow code for a given UUID.
        """
        try:
            with open(f"{self.workflow_dir}/{uuid}/workflow_code_{uuid}.py") as f:
                return f.read()
        except FileNotFoundError:
            raise ValueError(
                f"❌ Workflow code for UUID {uuid} not found in {self.workflow_dir}."
            )
        except Exception as e:
            raise ValueError(f"❌ Error reading workflow code: {str(e)}")

    def get_total_rewards(self, flow_state: any) -> float:
        """Calculate the total rewards from the workflow state."""
        return 0.0  # TODO

    def get_flow_answers(self, flow_state: any) -> str:
        """Extract the answers from the workflow state."""
        if not flow_state or "answers" not in flow_state:
            return ""
        return (
            "\n".join(flow_state["answers"])
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
        flow_rewards = 0.0
        flow_answers = ""
        if flow_state is not None:
            flow_rewards = self.get_total_rewards(flow_state)
            flow_answers = self.get_flow_answers(flow_state)
        else:
            flow_answers = (
                run_stdout.strip()
            )  # if run failed, use stdout/stderr as fallback
        print(f"\n===\nTotal rewards accumulated: {flow_rewards}")
        return f"""
You are a self-improving AI agent. Your goal is to improve the workflow code iteratively based on the results of previous iterations.

Previous workflow code you generated:
{flow_code}

Previous generation attempt ({iteration_count}) resulted in the following output:

{flow_answers}

Learn from this output and improve the workflow generation.
        """

    def select_workflow_template(self, template_uuid: str | None = None) -> str:
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
            # TODO implement a auto-selection mechanism for available workflows
            return None
        try:
            with open(
                f"{self.workflow_dir}/{template_uuid}/workflow_code_{template_uuid}.py",
            ) as f:
                return f.read()
        except FileNotFoundError:
            raise ValueError(
                f"❌ Workflow template for UUID {template_uuid} not found in {self.workflow_dir}."
            )
        except Exception as e:
            raise ValueError(f"❌ Error reading workflow template: {str(e)}")

    async def start_dgm(
        self,
        goal_prompt: str,
        template_uuid: str | None = None,
        judge: bool = True,
    ):
        template = self.select_workflow_template(template_uuid=template_uuid)
        await self.recursive_self_improvement(
            goal_prompt,
            goal_prompt,
            template_uuid=template_uuid,
            workflow_template=template,
            judge=judge,
        )

    async def recursive_self_improvement(
        self,
        prompt: str,
        goal: str,
        template_uuid: str | None = None,
        workflow_template: str | None = None,
        iteration_count: int = 0,
        max_depth: int = 5,
        judge: bool = True,
    ) -> str:
        """Run a self-improvement loop for the workflow.

        Args:
            prompt: The goal prompt for workflow generation, same as goal on first iteration
            goal: The goal to achieve with the workflow
            template_uuid: Optional UUID of workflow template to use
            workflow_template: Optional workflow template code to use
            iteration_count: Current iteration count (for recursion)
            max_depth: Maximum depth of recursion
        Returns:
            str: Final execution status message
        """
        flow_output = ""
        print(f"\n{'=' * 60}")
        print(f"ITERATION {iteration_count + 1}/5 - Self-Improvement Loop")
        print(f"{'=' * 60}")
        if iteration_count > 0:
            human_validation = (
                input("Continue with next iteration? (yes/no): ").strip().lower()
            )
            if human_validation not in ["yes", "y"]:
                print("Exiting self-improvement loop.")
                print()
                return flow_output
        print(f"\n{'📋 CURRENT GOAL':^60}")
        print(f"{'─' * 60}")
        print(f"  {goal}")
        print(f"{'─' * 60}\n")

        run_stdout, uuid , executed= await self.orchestrator.orchestrate_workflow(
            prompt, template_uuid, workflow_template
        )
        if executed:
            if judge:
                self.judge.evaluate(uuid)
            total_cost = self.judge.calculate_cost(uuid)
            print(f"Total workflow cost: {total_cost:.3f} USD")
        flow_state = self.load_flow_state_result(uuid)
        self.notifier.send_message(
            str(flow_state) if flow_state else run_stdout,
            title=f"Workflow {uuid} completed.",
        )
        flow_code = self.load_workflow_code(template_uuid if template_uuid else uuid)
        prompt = self.improvement_prompt(
            goal, flow_state, flow_code, run_stdout, iteration_count
        )
        template_uuid = None
        if iteration_count >= max_depth:
            print(
                f"Maximum iterations reached ({max_depth}). Ending self-improvement loop."
            )
            return flow_output
        await self.recursive_self_improvement(
            prompt,
            goal,
            template_uuid,
            workflow_template=flow_code if flow_state else None,
            iteration_count=iteration_count + 1,
        )
        return flow_output

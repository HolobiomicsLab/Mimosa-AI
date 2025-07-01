"""
Darwin Godel Machine
"""

import json
import os
from typing import Optional

from core.orchestrator import WorkflowOrchestrator

class GodelMachine:
    """Darwin Godel Machine for self-improvement workflows."""
    def __init__(self, config) -> None:
        self.config = config
        self.orchestrator = WorkflowOrchestrator(config)

    def load_flow_state_result(self, uuid: str) -> any:
        """Load the result of a previously executed workflow state.
        
        Args:
            uuid: UUID of the workflow state to load
        Returns:
            str: The output of the workflow state if found, None otherwise
        """
        try:
            with open(f"{self.workflow_dir}/{uuid}/state_result_{uuid}.json", 'r') as f:
                return json.loads(f.read().strip())
        except FileNotFoundError:
            raise ValueError(f"❌ Workflow state for UUID {uuid} not found in {self.workflow_dir}.")
        except Exception as e:
            raise ValueError(f"❌ Error reading workflow state: {str(e)}")
    
    def get_total_rewards(self, flow_state: any) -> float:
        """Calculate the total rewards from the workflow state."""
        if not flow_state or 'rewards' not in flow_state:
            return 0.0
        return sum(flow_state['rewards']) if isinstance(flow_state['rewards'], list) else flow_state['rewards']
    
    def get_flow_answers(self, flow_state: any) -> str:
        """Extract the answers from the workflow state."""
        if not flow_state or 'answers' not in flow_state:
            return ""
        return "\n".join(flow_state['answers']) if isinstance(flow_state['answers'], list) else flow_state['answers']

    def improvement_prompt(self, flow_state: any, iteration_count: int) -> str:
        flow_rewards = self.get_total_rewards(flow_state)
        flow_answers = self.get_flow_answers(flow_state)
        print(f"\n===\nTotal rewards accumulated: {flow_rewards}")
        return f"""
Previous generation attempt ({iteration_count}) resulted in the following output:

{flow_answers}

Learn from this output and improve the workflow generation.
        """
    async def recursive_self_improvement(self, goal_prompt: str,
                                           template_uuid: Optional[str] = None) -> str:
        """Run a self-improvement loop for the workflow.
        
        Args:
            goal_prompt: The goal description for the workflow
            template_uuid: Optional UUID of a workflow template to load
        Returns:
            str: Final execution status message
        """
        print("Starting self-improvement loop...")
        flow_output = ""

        for iteration_count in range(0, 5):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration_count + 1}/5 - Self-Improvement Loop")
            print(f"{'='*60}")
            
            human_validation = input("Continue with next iteration? (yes/no): ").strip().lower()
            if human_validation not in ["yes", "y"]:
                print("Exiting self-improvement loop.")
                break
                
            print(f"\n{'📋 CURRENT GOAL':^60}")
            print(f"{'─'*60}")
            print(f"  {goal_prompt}")
            print(f"{'─'*60}\n")
            
            _, uuid = await self.orchestrator.orchestrate_workflow(goal_prompt, template_uuid)
            flow_state = self.load_flow_state_result(uuid)
            goal_prompt += "\n" + self.improvement_prompt(flow_state, iteration_count)

        return flow_output
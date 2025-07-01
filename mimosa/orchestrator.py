
import json
import time
import sys, os
from typing import Optional

from core.workflow_factory import WorkflowFactory
from core.code_runner import WorkflowRunner, RuntimeConfig, ExecutionStatus

class WorkflowOrchestrator:
    """Main Meta-Agent workflow orchestration class.

    Attributes:
        workflow_dir (str): Directory containing workflow templates
    """
    
    def __init__(self, config) -> None:
        """Initialize the Mimosa application.
        
        Args:
            config: Configuration object containing paths and settings
        """
        self.workflow_dir = config.workflow_dir
        self.workflow_factory = WorkflowFactory(config)

        self.runner_config = RuntimeConfig(
            python_version=config.runner_default_python_version,
            timeout=config.runner_default_timeout,
            max_memory_mb=config.runner_default_max_memory_mb,
        )
        self.workflow_runner = WorkflowRunner(self.runner_config)

    def select_workflow_template(self, template_uuid: Optional[str] = None) -> str:
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
            with open(f"{self.workflow_dir}/{template_uuid}/workflow_code_{template_uuid}.py", 'r') as f:
                return f.read()
        except FileNotFoundError:
            raise ValueError(f"❌ Workflow template for UUID {template_uuid} not found in {self.workflow_dir}.")
        except Exception as e:
            raise ValueError(f"❌ Error reading workflow template: {str(e)}")
    
    async def workflow_requirements_install(self):
        deps = [
            "python-dotenv",
            "fastmcp==2.8.1",
            "requests>=2.31.0",
            "smolagents[all]",
            "langgraph>=0.4.7",
            "matplotlib>=3.9.0",
            "numpy>=2.0.0",
            "python_a2a",
            "opentelemetry-sdk",
            "opentelemetry-exporter-otlp",
            "openinference-instrumentation-smolagents"
        ]
        print("Installing dependencies...")
        dep_result = await self.workflow_runner.install_dependencies(deps)
        if dep_result.status != ExecutionStatus.COMPLETED:
            raise RuntimeError(f"Dependency installation failed: {dep_result.stderr}")
    
    async def workflow_sandbox_run(self, workflow_code: str) -> str:
        """Run the workflow code in a sandboxed environment."""
        def progress_handler(line: str):
            print(f"[LOG] {line}")

        print("Running workflow in python sandbox...")
        result = await self.workflow_runner.execute(workflow_code, progress_callback=progress_handler)
        await self.workflow_runner.cleanup()
        if result.status == ExecutionStatus.COMPLETED:
            print("Workflow execution completed successfully.")
            return result.stdout or result.stderr or "No output from workflow execution." 
        else:
            print(f"Workflow failed: {result.stderr}")
            raise Exception(f"Workflow execution failed: {result.stderr}")
    
    async def orchestrate_workflow(self, goal_prompt: str,
                                   template_uuid: Optional[str] = None,
                                  ) -> str:
        """Execute a workflow with the given goal prompt.
        
        Args:
            goal_prompt: The goal description for the workflow
            template_uuid: Optional UUID of a workflow template to load
        Returns:
            str: Execution status message
        """
        execution_output = ""

        workflow_code, uuid = self.workflow_factory.craft_workflow(
            goal_prompt,
            template_workflow=self.select_workflow_template(template_uuid=template_uuid),
            template_uuid=template_uuid,
            save_workflow=(template_uuid is None),
        )
        try:
            await self.workflow_requirements_install()
            execution_output = await self.workflow_sandbox_run(workflow_code)
        except Exception as e:
            print(f"❌ Error during execution: {e}")
            import traceback
            traceback.print_exc()
            return str(e), uuid
        finally:
            print("\nCleaning up sandbox...")
        output = execution_output.strip() if execution_output else "Workflow executed successfully with no output."
        return output, uuid
    
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
            
            _, uuid = await self.orchestrate_workflow(goal_prompt, template_uuid)
            flow_state = self.load_flow_state_result(uuid)
            goal_prompt += "\n" + self.improvement_prompt(flow_state, iteration_count)
            
        return flow_output
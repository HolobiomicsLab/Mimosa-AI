
import sys, os
from core.craft_workflow import WorkflowCrafting
from core.runner import WorkflowRunner, RuntimeConfig, ExecutionStatus
from typing import Optional

class WorkflowOrchestrator:
    """Main workflow orchestration class for Mimosa.

    Attributes:
        workflow_dir (str): Directory containing workflow templates
    """
    
    def __init__(self, workflow_dir: str = "workflows") -> None:
        """Initialize the Mimosa application.
        
        Args:
            workflow_dir: Path to directory containing workflow templates
        """
        self.workflow_dir = workflow_dir
        self.workflow_crafter = WorkflowCrafting(tools_dir="modules/tools",
                                                 workflow_dir=self.workflow_dir)
        self.runner_config = RuntimeConfig(
            python_version="3.10",
            timeout=3600,
            max_memory_mb=1024
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
            return "Workflow executed successfully"
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

        workflow_code = self.workflow_crafter.craft_workflow(
            goal_prompt,
            template_workflow=self.select_workflow_template(template_uuid=template_uuid),
            save_workflow=(template_uuid is None),
        )
        try:
            await self.workflow_requirements_install()
            await self.workflow_sandbox_run(workflow_code)
        except Exception as e:
            print(f"❌ Error during execution: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Workflow execution failed: {str(e)}")
        finally:
            print("\nCleaning up sandbox...")
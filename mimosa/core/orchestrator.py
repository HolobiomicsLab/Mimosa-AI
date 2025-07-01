
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
        """Initialize the Workflow orchestrator.
        
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
    
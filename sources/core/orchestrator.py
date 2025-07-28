"""
This class orchestrates the execution of workflows in a sandboxed environment.
"""

from .workflow_factory import WorkflowFactory
from .workflow_runner import ExecutionStatus, RuntimeConfig, WorkflowRunner


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
        self.config = config
        self.workflow_dir = config.workflow_dir
        self.workflow_factory = WorkflowFactory(config)

        self.runner_config = RuntimeConfig(
            python_version=self.config.runner_default_python_version,
            timeout=self.config.runner_default_timeout,
            max_memory_mb=self.config.runner_default_max_memory_mb,
        )
        self.workflow_runner = WorkflowRunner(self.runner_config)

    async def workflow_requirements_install(self):
        deps = self.config.runner_requirements
        print("Installing dependencies...")
        dep_result = await self.workflow_runner.install_dependencies(deps)
        if dep_result.status != ExecutionStatus.COMPLETED:
            raise RuntimeError(f"Dependency installation failed: {dep_result.stderr}")

    async def workflow_sandbox_run(self, workflow_code: str) -> str:
        """Run the workflow code in a sandboxed environment."""

        def progress_handler(line: str):
            print(f"{line}")

        print("Running workflow in python sandbox...")
        result = await self.workflow_runner.execute(
            workflow_code, progress_callback=progress_handler
        )
        if result.status == ExecutionStatus.COMPLETED:
            print("Workflow execution completed successfully.")
            return (
                result.stdout or result.stderr or "No output from workflow execution."
            )
        else:
            print(f"Workflow failed: {result.stderr}")
            raise Exception(f"Workflow execution failed: {result.stderr}")

    async def orchestrate_workflow(
        self,
        goal_prompt: str,
        workflow_template: str | None = None,
    ) -> str:
        """Execute a workflow with the given goal prompt.

        Args:
            goal_prompt: The goal description for the workflow
            workflow_template: Optional workflow template code to use
        Returns:
            str: Execution status message
        """
        execution_output = ""

        workflow_code, uuid = await self.workflow_factory.craft_workflow(
            goal_prompt,
            template_workflow=workflow_template,
            save_workflow=True,
        )
        try:
            await self.workflow_requirements_install()
            execution_output = await self.workflow_sandbox_run(workflow_code)
        except Exception as e:
            print(f"❌ Error during {uuid} workflow execution: {e}")
            import traceback

            traceback.print_exc()
            return str(e), uuid, False
        finally:
            print("\nCleaning up sandbox...")
        output = (
            execution_output.strip()
            if execution_output
            else "Workflow executed successfully with no output."
        )
        return output, uuid, True

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        try:
            await self.workflow_runner.cleanup()
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            import traceback

            traceback.print_exc()

    def __del__(self):
        """Cleanup resources on deletion - sync fallback."""
        # Note: This is a fallback - proper cleanup should use async context manager
        import asyncio
        from contextlib import suppress

        try:
            # Only attempt sync cleanup if no event loop is running
            with suppress(RuntimeError):
                # If we have a running loop, schedule cleanup as a task
                asyncio.create_task(self.workflow_runner.cleanup())
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            import traceback

            traceback.print_exc()

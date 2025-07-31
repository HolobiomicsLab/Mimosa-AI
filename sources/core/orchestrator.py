"""
This class orchestrates the execution of workflows in a sandboxed environment.
"""

import time
import logging

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
        print(f"\033[96m📦 Installing workflow dependencies: {deps}\033[0m")
        dep_result = await self.workflow_runner.install_dependencies(deps)
        if dep_result.status != ExecutionStatus.COMPLETED:
            raise RuntimeError(f"Dependency installation failed: {dep_result.stderr}")

    async def workflow_sandbox_run(self, workflow_code: str) -> str:
        """Run the workflow code in a sandboxed environment."""

        def progress_handler(line: str):
            print(f"\033[96m  🔄 {line}\033[0m")

        print("\033[96m🚀 Executing workflow in Python sandbox...\033[0m")
        result = await self.workflow_runner.execute(
            workflow_code, progress_callback=progress_handler
        )
        if result.status == ExecutionStatus.COMPLETED:
            print(f"\033[96m✅ Workflow execution completed successfully in {result.execution_time:.3f}s\033[0m")
            return (
                result.stdout or result.stderr or "No output from workflow execution."
            )
        else:
            print(f"\033[91m❌ Workflow execution failed: {result.stderr}\033[0m")
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
        logger = logging.getLogger(__name__)
        workflow_start_time = time.time()
        execution_output = ""

        logger.info(f"[WORKFLOW START] Orchestrating workflow - {goal_prompt[:50]}...")
        print(f"\n\033[96m{'🏗️  WORKFLOW GENERATION PHASE':^80}\033[0m")
        print(f"\033[96m{'=' * 80}\033[0m")

        # Workflow generation timing
        generation_start = time.time()
        workflow_code, uuid = await self.workflow_factory.craft_workflow(
            goal_prompt,
            template_workflow=workflow_template,
            save_workflow=True,
        )
        generation_time = time.time() - generation_start
        logger.info(f"[WORKFLOW GENERATION] {uuid} generated in {generation_time:.3f}s")
        print(f"\033[96m✅ Workflow {uuid} generated successfully in {generation_time:.3f}s\033[0m")

        try:
            # Dependencies installation phase
            print(f"\n\033[96m{'📦 DEPENDENCIES INSTALLATION PHASE':^80}\033[0m")
            print(f"\033[96m{'=' * 80}\033[0m")
            deps_start = time.time()
            await self.workflow_requirements_install()
            deps_time = time.time() - deps_start
            logger.info(f"[WORKFLOW DEPS] {uuid} dependencies installed in {deps_time:.3f}s")
            print(f"\033[96m✅ Dependencies installed successfully in {deps_time:.3f}s\033[0m")
            
            # Execution phase
            print(f"\n\033[96m{'🚀 WORKFLOW EXECUTION PHASE':^80}\033[0m")
            print(f"\033[96m{'=' * 80}\033[0m")
            exec_start = time.time()
            execution_output = await self.workflow_sandbox_run(workflow_code)
            exec_time = time.time() - exec_start
            logger.info(f"[WORKFLOW EXECUTION] {uuid} executed in {exec_time:.3f}s")
            print(f"\033[96m✅ Workflow executed successfully in {exec_time:.3f}s\033[0m")
            
        except Exception as e:
            workflow_time = time.time() - workflow_start_time
            logger.info(f"[WORKFLOW ERROR] {uuid} failed after {workflow_time:.3f}s - {str(e)}")
            print(f"❌ Error during {uuid} workflow execution: {e}")
            import traceback

            traceback.print_exc()
            return str(e), uuid, False
        finally:
            print("\nCleaning up sandbox...")
            
        workflow_time = time.time() - workflow_start_time
        logger.info(f"[WORKFLOW END] {uuid} completed in {workflow_time:.3f}s")
        
        print(f"\n\033[96m{'✨ WORKFLOW COMPLETION SUMMARY':^80}\033[0m")
        print(f"\033[96m{'=' * 80}\033[0m")
        print(f"\033[96m📋 Workflow UUID: {uuid}\033[0m")
        print(f"\033[96m⏱️  Total Time: {workflow_time:.3f}s\033[0m")
        print(f"\033[96m  • Generation: {generation_time:.3f}s\033[0m")
        print(f"\033[96m  • Dependencies: {deps_time:.3f}s\033[0m")
        print(f"\033[96m  • Execution: {exec_time:.3f}s\033[0m")
        print(f"\033[96m{'=' * 80}\033[0m\n")
        
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

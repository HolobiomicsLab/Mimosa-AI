"""
This class orchestrates the execution of workflows in a sandboxed environment.
"""

import logging
import time

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
        logging.getLogger(__name__)

        def progress_handler(line: str):
            print(line)

        print("\033[96m🚀 Executing workflow in Python sandbox...\033[0m")
        result = await self.workflow_runner.execute(
            workflow_code, progress_callback=progress_handler
        )
        if result.status == ExecutionStatus.COMPLETED:
            print(
                f"\033[96m✅ Workflow execution completed successfully in {result.execution_time:.3f}s\033[0m"
            )
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
    ) -> tuple[str, str, bool]:
        """Execute a workflow with the given goal prompt.

        Args:
            goal_prompt: The goal description for the workflow
            workflow_template: Optional workflow template code to use
        Returns:
            tuple[str, str, bool]: (execution_output, workflow_uuid, success_flag)
        """
        logger = logging.getLogger(__name__)

        workflow_start_time = time.time()
        execution_output = ""

        logger.info(f"[WORKFLOW START] Orchestrating workflow - {goal_prompt[:50]}...")
        print(f"\n\033[96m{'🏗️  WORKFLOW GENERATION PHASE':^80}\033[0m")
        print(f"\033[96m{'=' * 80}\033[0m")

        # Workflow generation timing
        generation_start = time.time()
        try:
            workflow_code, uuid = await self.workflow_factory.craft_workflow(
                goal_prompt,
                template_workflow=workflow_template,
                save_workflow=True,
            )
        except Exception as e:
            generation_time = time.time() - generation_start
            # Extract UUID from exception message if available
            error_msg = str(e)
            if error_msg.startswith("UUID:") and "|" in error_msg:
                uuid_part, actual_error = error_msg.split("|", 1)
                workflow_uuid = uuid_part.replace("UUID:", "")
                logger.warning(f"[WORKFLOW GENERATION ERROR] {actual_error} - letting DGM handle retry")
                return f"WORKFLOW_GENERATION_ERROR: {actual_error}", workflow_uuid, False
            else:
                logger.warning(f"[WORKFLOW GENERATION ERROR] {error_msg} - letting DGM handle retry")
                return f"WORKFLOW_GENERATION_ERROR: {error_msg}", "generation_failed", False
        
        generation_time = time.time() - generation_start
        logger.info(f"[WORKFLOW GENERATION] {uuid} generated in {generation_time:.3f}s")
        print(
            f"\033[96m✅ Workflow {uuid} generated successfully in {generation_time:.3f}s\033[0m"
        )

        try:
            # Dependencies installation phase
            print(f"\n\033[96m{'📦 DEPENDENCIES INSTALLATION PHASE':^80}\033[0m")
            print(f"\033[96m{'=' * 80}\033[0m")
            deps_start = time.time()
            await self.workflow_requirements_install()
            deps_time = time.time() - deps_start
            logger.info(
                f"[WORKFLOW DEPS] {uuid} dependencies installed in {deps_time:.3f}s"
            )
            print(
                f"\033[96m✅ Dependencies installed successfully in {deps_time:.3f}s\033[0m"
            )

            # Execution phase
            print(f"\n\033[96m{'🚀 WORKFLOW EXECUTION PHASE':^80}\033[0m")
            print(f"\033[96m{'=' * 80}\033[0m")
            exec_start = time.time()
            execution_output = await self.workflow_sandbox_run(workflow_code)
            exec_time = time.time() - exec_start
            logger.info(f"[WORKFLOW EXECUTION] {uuid} executed in {exec_time:.3f}s")
            print(
                f"\033[96m✅ Workflow executed successfully in {exec_time:.3f}s\033[0m"
            )

        except Exception as e:
            workflow_time = time.time() - workflow_start_time
            logger.info(
                f"[WORKFLOW ERROR] {uuid} failed after {workflow_time:.3f}s - {str(e)}"
            )
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
        try:
            # Check if we're during Python shutdown
            import sys

            if sys.meta_path is None:
                return

            # Import at module level to avoid shutdown issues
            import asyncio
            from contextlib import suppress

            # Only attempt cleanup if workflow_runner still exists
            if hasattr(self, "workflow_runner") and self.workflow_runner is not None:
                # Try to get current event loop and schedule cleanup
                with suppress(RuntimeError, AttributeError):
                    try:
                        loop = asyncio.get_running_loop()
                        if not loop.is_closed():
                            loop.create_task(self.workflow_runner.cleanup())
                    except RuntimeError:
                        # No running loop, try to run cleanup synchronously if possible
                        # This is a last resort and may not work for all cleanup operations
                        pass
        except Exception:
            # Silently ignore cleanup errors during shutdown
            pass

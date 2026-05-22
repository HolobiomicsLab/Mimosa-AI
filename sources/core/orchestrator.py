"""
This class orchestrates the execution of workflows in a sandboxed environment.
"""

import logging
import time

from sources.utils.notify import PushNotifier
from sources.utils.perspicacite_client import (
    format_scientific_context,
    query_perspicacite,
)
from sources.cli.pretty_print import (
    print_ok, print_err, print_info,
    print_phase, print_summary,
)
from .workflow_factory import WorkflowFactory
from .single_agent_factory import SingleAgentFactory
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
        self.single_agent_factory = SingleAgentFactory(config)
        self.workflow_factory = WorkflowFactory(config)
        self.workflow_dir = config.workflow_dir
        self.notifier = PushNotifier(config.pushover_token, config.pushover_user)

        self.runner_config = RuntimeConfig(
            python_version=self.config.runner_default_python_version,
            timeout=self.config.runner_default_timeout,
            max_memory_mb=self.config.runner_default_max_memory_mb,
        )
        self.workflow_runner = WorkflowRunner(self.runner_config, self.config.workspace_dir)

    async def workflow_requirements_install(self):
        deps = self.config.runner_requirements
        print_info(f"📦 Installing workflow dependencies: {deps}")
        dep_result = await self.workflow_runner.install_dependencies(deps)
        if dep_result.status != ExecutionStatus.COMPLETED:
            raise RuntimeError(f"Dependency installation failed: {dep_result.stderr}")

    async def workflow_sandbox_run(self, workflow_genotype_code: str) -> str:
        """Run the workflow code in a sandboxed environment."""
        logging.getLogger(__name__)

        def progress_handler(line: str):
            print(line)

        print_info("▶ Executing workflow in Python sandbox…")
        result = await self.workflow_runner.execute(
            workflow_genotype_code, progress_callback=progress_handler
        )
        if result.status == ExecutionStatus.COMPLETED:
            print_ok(f"Workflow execution completed in {result.execution_time:.3f}s")
            return (
                result.stdout or result.stderr or "No output from workflow execution."
            )
        else:
            print_err(f"Workflow execution failed: {result.stderr}")
            raise Exception(f"Workflow execution failed: {result.stderr}")
    
    def perspicacite_grounding_task(self, task):
        prompt = f"""You are a scientific literature specialist supporting a scientist on a task.
SCIENTIFIC TASK:
{task}

Query the literature to design a litterature grounded approach.

1. LITERATURE FOUNDATION
   - Canonical methods and key papers (authors, year, venue)
   - Field conventions that must be followed
   - Standard validation criteria

2. WORKFLOW DESIGN
   For each stage, specify:
   - Expert role needed (e.g., "Data Curator", "Method Specialist", "Quality Reviewer")
   - Inputs required and outputs produced
   - Completion criteria grounded in literature
   - Conditions to proceed to next stage
   - Known failure modes and recovery strategies from best practices
   - Complexity and uncertainty estimates (flag high-risk steps)

CONSTRAINTS: Cite sources for all methodological claims. Note where literature is sparse or conflicting.
        """
        try:
            response = query_perspicacite(prompt) or "No relevant scientific context."
            return response
        except Exception as e:
            return "Query failed. Unable to help with scientific litterature"

    def _ground_with_perspicacite(self, task: str, craft_instructions: str) -> str:
        """Enrich craft instructions with scientific context from Perspicacite-AI.

        Returns the original instructions if grounding is unavailable.
        """
        print_phase(
            "🔬 Querying Perspicacite-AI for scientific context... (This can take several minutes)"
        )
        scientific_context = self.perspicacite_grounding_task(task)
        if not scientific_context:
            print(
                "\033[93m⚠️  [Perspicacite] Service unavailable or returned no "
                "results – proceeding without scientific grounding.\033[0m"
            )
            return craft_instructions
        print_info(
            f"\033[94m[Perspicacite] Scientific context:\n{scientific_context[:2048]}...\033[0m"
        )
        return format_scientific_context(task, scientific_context) + craft_instructions

    async def _generate_workflow_code(
        self,
        goal: str,
        craft_instructions: str,
        single_agent_mode: bool,
        original_task: str | None,
    ) -> tuple[str, str, str]:
        """Generate workflow code via the appropriate factory.

        Returns (complete_code, workflow_genotype_code, uuid).
        Raises on generation failure; factories may encode "UUID:<uuid>|<msg>".
        """
        if single_agent_mode:
            return await self.single_agent_factory.craft_single_agent(
                goal, original_task=original_task
            )
        return await self.workflow_factory.craft_workflow(
            goal, craft_instructions, save_workflow=True, original_task=original_task
        )

    @staticmethod
    def _parse_generation_error(error: Exception) -> tuple[str, str]:
        """Extract (uuid, message) from a factory generation error.

        Returns ("generation_failed", message) when no uuid is encoded.
        """
        msg = str(error)
        if msg.startswith("UUID:") and "|" in msg:
            uuid_part, actual_error = msg.split("|", 1)
            return uuid_part.replace("UUID:", ""), actual_error
        return "generation_failed", msg

    async def _execute_in_sandbox(self, complete_code: str) -> tuple[str, float, float]:
        """Install dependencies and run the workflow in a sandbox.

        Returns (execution_output, deps_install_time, exec_time).
        Raises on dependency install or execution failure.
        """
        print_phase("DEPENDENCIES INSTALLATION PHASE")
        deps_start = time.time()
        await self.workflow_requirements_install()
        deps_time = time.time() - deps_start
        print_ok(f"Dependencies installed in {deps_time:.3f}s")

        print_phase("WORKFLOW EXECUTION PHASE")
        exec_start = time.time()
        output = await self.workflow_sandbox_run(complete_code)
        exec_time = time.time() - exec_start
        print_ok(f"Workflow executed in {exec_time:.3f}s")
        return output, deps_time, exec_time

    def _notify_execution_failure(self, uuid: str, goal: str, workflow_time: float, error: Exception) -> None:
        """Send a Pushover notification for an execution failure."""
        self.notifier.send_message(
            f"Workflow {uuid} execution failed after {workflow_time:.1f}s\n"
            f"Goal: {goal[:128]}...\n"
            f"Error: {str(error)[:256]}",
            title=f"Workflow {uuid} execution failed",
            priority=1,
        )

    async def orchestrate_workflow(
        self,
        goal: str,
        craft_instructions: str,
        original_task: str = None,
        single_agent_mode: bool = False,
        no_run: bool = False,
    ) -> tuple[str, str, str, bool]:
        """Execute a workflow end-to-end: grounding → generation → sandbox run.

        Errors are captured into the return tuple so the evolution loop can
        recover via re-attempt; `executed=False` is the structural signal.

        Args:
            goal: Workflow goal (may be knowledge-wrapped).
            craft_instructions: Recipe for the LLM workflow generator.
            original_task: Unwrapped task for similarity matching.
            single_agent_mode: Use the single-agent factory instead of multi-agent.
            no_run: Skip dependency install and sandbox execution.

        Returns:
            (execution_output, workflow_uuid, workflow_genotype_code, executed).
            `executed` is False on any generation or execution error.
        """
        logger = logging.getLogger(__name__)
        workflow_start = time.time()
        science_task = original_task or goal

        craft_instructions = self._ground_with_perspicacite(science_task, craft_instructions)

        logger.info(f"[WORKFLOW START] Orchestrating workflow - {goal[:50]}...")
        print_phase("WORKFLOW GENERATION PHASE")
        generation_start = time.time()
        try:
            complete_code, workflow_genotype_code, uuid = await self._generate_workflow_code(
                goal, craft_instructions, single_agent_mode, original_task
            )
        except Exception as e:
            failed_uuid, error_msg = self._parse_generation_error(e)
            logger.warning(f"[WORKFLOW_GENERATION_ERROR]\n{error_msg}\n")
            print_err(error_msg)
            return f"WORKFLOW_GENERATION_ERROR: {error_msg}", failed_uuid, "error", False

        generation_time = time.time() - generation_start
        logger.info(f"[WORKFLOW GENERATION] {uuid} generated in {generation_time:.3f}s")
        print_ok(f"Workflow {uuid} generated in {generation_time:.3f}s")

        if no_run:
            return "", uuid, workflow_genotype_code, True

        try:
            execution_output, deps_time, exec_time = await self._execute_in_sandbox(complete_code)
        except Exception as e:
            workflow_time = time.time() - workflow_start
            logger.info(f"[WORKFLOW ERROR] {uuid} failed after {workflow_time:.3f}s - {e}")
            print_err(f"Error during {uuid} workflow execution: {e}")
            import traceback
            traceback.print_exc()
            self._notify_execution_failure(uuid, goal, workflow_time, e)
            return str(e), uuid, workflow_genotype_code, False
        finally:
            print_info("Cleaning up sandbox…")

        workflow_time = time.time() - workflow_start
        logger.info(f"[WORKFLOW END] {uuid} completed in {workflow_time:.3f}s")
        print_summary(
            "✨ WORKFLOW COMPLETION SUMMARY",
            [
                ("UUID", uuid),
                ("Total time", f"{workflow_time:.3f}s"),
                ("Generation", f"{generation_time:.3f}s"),
                ("Dependencies", f"{deps_time:.3f}s"),
                ("Execution", f"{exec_time:.3f}s"),
            ],
        )
        output = execution_output.strip() if execution_output else "Workflow executed successfully with no output."
        return output, uuid, workflow_genotype_code, True

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        try:
            await self.workflow_runner.cleanup()
        except Exception as e:
            print_err(f"Error during cleanup: {e}")
            import traceback

            traceback.print_exc()

    def __del__(self):
        """Cleanup resources on deletion - sync fallback."""
        try:
            import sys

            if sys.meta_path is None:
                return

            import asyncio
            from contextlib import suppress

            if hasattr(self, "workflow_runner") and self.workflow_runner is not None:
                with suppress(RuntimeError, AttributeError):
                    try:
                        loop = asyncio.get_running_loop()
                        if not loop.is_closed():
                            loop.create_task(self.workflow_runner.cleanup())
                    except RuntimeError:
                        pass
        except Exception:
            pass

"""
This class orchestrates the execution of workflows in a sandboxed environment.
"""

import logging
import time
from typing import Any

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

    def __init__(self, config: "Config") -> None:
        """Initialize the Workflow orchestrator.

        Args:
            config: Configuration object containing paths and settings (workspace
                dir, workflow dir, Pushover credentials, runner defaults, etc.).
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
            temp_dir=self.config.runner_temp_dir,
        )
        self.workflow_runner = WorkflowRunner(self.runner_config, self.config.workspace_dir)

    async def workflow_requirements_install(self) -> None:
        """Install the runner's declared dependencies into the sandbox.

        Raises:
            RuntimeError: If the dependency installation does not complete
                successfully.
        """
        deps = self.config.runner_requirements
        print_info(f"📦 Installing workflow dependencies: {deps}")
        dep_result = await self.workflow_runner.install_dependencies(deps)
        if dep_result.status != ExecutionStatus.COMPLETED:
            raise RuntimeError(f"Dependency installation failed: {dep_result.stderr}")

    async def workflow_sandbox_run(self, workflow_genotype_code: str) -> str:
        """Run the workflow code in a sandboxed environment.

        Args:
            workflow_genotype_code: Python source code for the workflow to
                execute inside the sandbox.

        Returns:
            The workflow execution output (stdout, or stderr/fallback message
            when stdout is empty).

        Raises:
            Exception: If the workflow execution does not reach COMPLETED status.
        """
        logging.getLogger(__name__)

        def progress_handler(line: str) -> None:
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
            stderr = result.stderr or ""
            if "TimeoutExpired" in stderr or "timed out" in stderr.lower():
                error_type = "TIMEOUT"
            elif "SyntaxError" in stderr:
                error_type = "SYNTAX_ERROR"
            else:
                error_type = "RUNTIME_ERROR"
            print_err(f"[{error_type}] Workflow execution failed: {stderr}")
            raise Exception(f"[{error_type}] Workflow execution failed: {stderr}")

    def perspicacite_grounding_task(self, task: str) -> str:
        """Query Perspicacite-AI for a literature-grounded approach to a task.

        Args:
            task: The scientific task description to ground.

        Returns:
            The literature-grounded response text, or a fallback message when
            the service is unavailable or returns no relevant context.
        """
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
            response = query_perspicacite(
                prompt, kb_name=self.config.perspicacite_agent_kb_name
            ) or "No relevant scientific context."
            return response
        except Exception as e:
            return "Query failed. Unable to help with scientific litterature"

    def _ground_with_perspicacite(self, task: str, craft_instructions: str) -> str:
        """Enrich craft instructions with scientific context from Perspicacite-AI.

        Returns the original instructions if grounding is unavailable.

        Args:
            task: The scientific task to query for context.
            craft_instructions: Original craft instructions to be enriched.

        Returns:
            The craft instructions, optionally prepended with formatted
            scientific context retrieved from Perspicacite-AI.
        """
        # Agent-side grounding is opt-out: disable it (or point it at a
        # leak-free KB via perspicacite_agent_kb_name) for fair benchmarking,
        # since the default web-search can surface the source paper.
        if not self.config.perspicacite_agent_grounding_enabled:
            return craft_instructions
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
        return format_scientific_context(task, scientific_context) + '\n' + craft_instructions

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

        Args:
            goal: Workflow goal (may be knowledge-wrapped).
            craft_instructions: Recipe for the LLM workflow generator.
            single_agent_mode: When True, dispatch to the single-agent factory;
                otherwise use the multi-agent workflow factory.
            original_task: Unwrapped task for similarity matching, or None.

        Returns:
            A tuple ``(complete_code, workflow_genotype_code, uuid)`` produced
            by the selected factory.

        Raises:
            Exception: Any error raised by the underlying factory; the message
                may be encoded as ``"UUID:<uuid>|<msg>"``.
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

        Args:
            error: Exception raised by a workflow factory during generation.

        Returns:
            A tuple ``(uuid, message)``. When the error message uses the
            ``"UUID:<uuid>|<msg>"`` convention, the encoded UUID and message
            are returned; otherwise ``("generation_failed", str(error))``.
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

        Args:
            complete_code: Full Python source code for the workflow to execute.

        Returns:
            A tuple ``(execution_output, deps_install_time, exec_time)`` where
            times are in seconds.

        Raises:
            RuntimeError: If dependency installation fails.
            Exception: If workflow execution fails inside the sandbox.
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
        """Send a Pushover notification for an execution failure.

        Args:
            uuid: Identifier of the workflow that failed.
            goal: Workflow goal text (truncated in the notification body).
            workflow_time: Elapsed time before the failure, in seconds.
            error: Exception describing the failure (truncated in the body).
        """
        self.notifier.send_message(
            f"Workflow {uuid} execution failed after {workflow_time:.1f}s\n"
            f"Goal: {goal[:128]}...\n"
            f"Error: {str(error)[:256]}",
            title=f"Workflow {uuid} execution failed",
            priority=1,
        )
    
    def _prompt_agents_model_list(self, craft_instructions: str) -> str:
        """Prompt the orchestrator the list of allowed models choice.

        Args:
            craft_instructions: Original craft instructions to be enriched.

        Returns:
            The craft instructions, optionally prepended with model selection
            instructions.
        """
        if not self.config.orchestrator_choose_model:
            unallowed_banner = """
\nYou are not allowed to choose per agent model. The workflow will use the default model(s) specified in the configuration.\n
"""
            return craft_instructions + unallowed_banner

        model_list = self.config.smolagent_model_id if isinstance(self.config.smolagent_model_id, list) else [self.config.smolagent_model_id]
        model_list_str = '\n - '.join(model_list)
        model_list_prompt = (
            "You can choose per agent model for this workflow from the following list:\n"
            f"{model_list_str}\n"
            "Do not choose a model that is not in the list. If you do not want to choose a model, you can leave it empty and the default model will be used.\n"
            "Specify the model parameter for each agent with the value of your choice."
        )
        return model_list_prompt + '\n' + craft_instructions

    async def orchestrate_workflow(
        self,
        goal: str,
        craft_instructions: str,
        original_task: str | None = None,
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

        craft_instructions = self._prompt_agents_model_list(craft_instructions)

        if self.config.literrature_grounding == True:
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

    async def __aenter__(self) -> "WorkflowOrchestrator":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit with proper cleanup."""
        try:
            await self.workflow_runner.cleanup()
        except Exception as e:
            print_err(f"Error during cleanup: {e}")
            import traceback

            traceback.print_exc()

    def __del__(self) -> None:
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

async def test_workflow_orchestrator():
    evolution_prommt = '''
put here evolution prompt to test
'''
    goal = "..."
    from config import Config
    config = Config()
    orch = WorkflowOrchestrator(config)
    await orch._generate_workflow_code(
        goal, evolution_prommt, single_agent_mode=False, original_task=goal
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_workflow_orchestrator())
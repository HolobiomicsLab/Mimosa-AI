"""End-to-end workflow production and sandboxed execution (Layer 3).

This file owns the run pipeline that turns one task into an executed
workflow: optional Perspicacite-AI literature grounding, genotype code
generation through the workflow factories, then dependency install and
execution inside the sandboxed runner. The evolution engine (Layer 2)
calls :class:`WorkflowOrchestrator.orchestrate_workflow` for every
candidate it wants to create and score; console output and push
notifications report progress. Generation and execution failures are
returned, not raised, so the evolution loop can re-attempt.
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
    """Produce workflow code and execute it inside the sandboxed runner.

    The evolution engine constructs one orchestrator per run with the
    application config and keeps it across generations; the async
    context manager protocol guarantees runner cleanup. Invariant:
    :meth:`orchestrate_workflow` never raises on workflow failure — it
    reports errors through its return tuple instead.
    """

    def __init__(self, config: "Config") -> None:
        """Build the factories, notifier, and sandbox runner.

        The evolution engine calls it once per run. It wires the
        single-agent and workflow factories to the same config and
        creates the runner from the configured defaults; no sandbox is
        started here.

        Args:
            config: Application configuration with workspace and
                workflow directories, runner defaults, Pushover
                credentials, and model settings (config file).
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

        :meth:`_execute_in_sandbox` calls it before every execution so
        the workflow's imports resolve inside the runner. The package
        list comes from ``config.runner_requirements`` (config file).

        Raises:
            RuntimeError: If the dependency installation does not
                complete successfully.
        """
        deps = self.config.runner_requirements
        print_info(f"📦 Installing workflow dependencies: {deps}")
        dep_result = await self.workflow_runner.install_dependencies(deps)
        if dep_result.status != ExecutionStatus.COMPLETED:
            raise RuntimeError(f"Dependency installation failed: {dep_result.stderr}")

    async def workflow_sandbox_run(self, workflow_genotype_code: str) -> str:
        """Run the workflow code in the sandboxed runner.

        :meth:`_execute_in_sandbox` calls it with the complete
        generated source; output lines stream to stdout while the run
        progresses. A COMPLETED result returns the captured output;
        timeouts, syntax errors, and other failures raise so the caller
        can report them.

        Args:
            workflow_genotype_code: Python source code for the workflow
                (LLM factory output).

        Returns:
            str: The workflow output — stdout, or stderr, or a
            placeholder message when stdout is empty.

        Raises:
            Exception: When the run does not reach COMPLETED status;
            the message is prefixed with TIMEOUT, SYNTAX_ERROR, or
            RUNTIME_ERROR.
        """
        logging.getLogger(__name__)

        def progress_handler(line: str) -> None:
            """Print one sandbox output line to the console."""
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

        :meth:`_ground_with_perspicacite` calls it during grounding when
        the run enables literature grounding. The task text is embedded
        in a fixed prompt that asks for canonical methods, expert roles,
        and validation criteria; service errors degrade to a fallback
        message instead of raising.

        Args:
            task: The scientific task description to ground (current
                run's goal or original task).

        Returns:
            str: The literature-grounded response, or a fallback
            message when the service is unavailable or returns nothing.
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
            response = query_perspicacite(prompt) or "No relevant scientific context."
            return response
        except Exception as e:
            return "Query failed. Unable to help with scientific litterature"

    def _ground_with_perspicacite(self, task: str, craft_instructions: str) -> str:
        """Prepend Perspicacite-AI scientific context to craft instructions.

        :meth:`orchestrate_workflow` calls it when literature grounding
        is enabled, before code generation. When the grounding query
        fails or returns nothing, the instructions pass through
        unchanged.

        Args:
            task: The scientific task to query for context (original
                task text).
            craft_instructions: Recipe for the LLM workflow generator
                (evolution prompt output).

        Returns:
            str: The craft instructions, prepended with the formatted
            scientific context when grounding succeeded.
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
        return format_scientific_context(task, scientific_context) + '\n' + craft_instructions

    async def _generate_workflow_code(
        self,
        goal: str,
        craft_instructions: str,
        single_agent_mode: bool,
        original_task: str | None,
    ) -> tuple[str, str, str]:
        """Generate workflow code via the selected factory.

        :meth:`orchestrate_workflow` calls it once per candidate. It
        dispatches to the single-agent factory in single-agent mode and
        to the multi-agent workflow factory otherwise; both persist the
        generated workflow and return its UUID.

        Args:
            goal: Workflow goal, possibly knowledge-wrapped (evolution
                engine).
            craft_instructions: Recipe for the LLM workflow generator
                (evolution prompt, grounding, and model-list notes).
            single_agent_mode: When ``True``, dispatch to the
                single-agent factory; otherwise the multi-agent one.
            original_task: Unwrapped task for similarity matching, or
                ``None``.

        Returns:
            tuple[str, str, str]: ``(complete_code, workflow_genotype_code,
            uuid)`` produced by the selected factory.

        Raises:
            Exception: Any factory error; the message may be encoded as
            ``"UUID:<uuid>|<msg>"`` — see
            :meth:`_parse_generation_error`.
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
        """Extract ``(uuid, message)`` from a factory generation error.

        :meth:`orchestrate_workflow` calls it in the generation error
        handler so a partially created workflow keeps a usable UUID in
        logs and return values.

        Args:
            error: Exception raised by a workflow factory during
                generation.

        Returns:
            tuple[str, str]: ``(uuid, message)``. When the message uses
            the ``"UUID:<uuid>|<msg>"`` convention, the encoded UUID and
            message are returned; otherwise
            ``("generation_failed", str(error))``.
        """
        msg = str(error)
        if msg.startswith("UUID:") and "|" in msg:
            uuid_part, actual_error = msg.split("|", 1)
            return uuid_part.replace("UUID:", ""), actual_error
        return "generation_failed", msg

    async def _execute_in_sandbox(self, complete_code: str) -> tuple[str, float, float]:
        """Install dependencies and run the workflow in the sandbox.

        :meth:`orchestrate_workflow` calls it for every generated
        candidate, and both phases are timed for the completion
        summary.

        Args:
            complete_code: Full Python source code for the workflow
                (factory output).

        Returns:
            tuple[str, float, float]: ``(execution_output,
            deps_install_time, exec_time)`` with times in seconds.

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

        :meth:`orchestrate_workflow` calls it once per failed run so the
        operator learns about the failure even without watching the
        console.

        Args:
            uuid: Identifier of the workflow that failed (factory
                output).
            goal: Workflow goal text (truncated in the notification
                body).
            workflow_time: Elapsed time before the failure, in seconds.
            error: Exception describing the failure (truncated in the
                body).
        """
        self.notifier.send_message(
            f"Workflow {uuid} execution failed after {workflow_time:.1f}s\n"
            f"Goal: {goal[:128]}...\n"
            f"Error: {str(error)[:256]}",
            title=f"Workflow {uuid} execution failed",
            priority=1,
        )
    
    def _prompt_agents_model_list(self, craft_instructions: str) -> str:
        """Prepend the per-agent model-choice note to craft instructions.

        :meth:`orchestrate_workflow` calls it before generation. When
        ``config.orchestrator_choose_model`` is off, a banner appended
        after the instructions tells the generator to keep the default
        models; when on, the allowed model list is prepended.

        Args:
            craft_instructions: Recipe for the LLM workflow generator
                (evolution prompt output).

        Returns:
            str: The craft instructions with the model-choice note
            prepended or appended.
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

        The evolution engine calls it once per candidate generation.
        Errors are captured into the return tuple so the loop can
        re-attempt; ``executed=False`` is the structural failure signal,
        and a Pushover notice is sent for execution failures.

        Args:
            goal: Workflow goal, possibly knowledge-wrapped (evolution
                engine).
            craft_instructions: Recipe for the LLM workflow generator.
            original_task: Unwrapped task for similarity matching, or
                ``None``.
            single_agent_mode: Use the single-agent factory instead of
                the multi-agent one.
            no_run: Skip dependency install and sandbox execution.

        Returns:
            tuple[str, str, str, bool]: ``(execution_output,
            workflow_uuid, workflow_genotype_code, executed)``.
            ``executed`` is ``False`` on any generation or execution
            error; with ``no_run`` the output is ``""``.
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
        """Enter the async context and return the orchestrator.

        Evolution runs wrap the orchestrator in ``async with`` so
        runner cleanup is guaranteed.

        Returns:
            WorkflowOrchestrator: This orchestrator instance.
        """
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the async context and clean up the sandbox runner.

        Runs for every ``async with`` block, including error paths.
        Cleanup errors are printed, never raised.

        Args:
            exc_type: Exception class of an in-flight error, or
                ``None``.
            exc_val: Exception instance of an in-flight error, or
                ``None``.
            exc_tb: Traceback of an in-flight error, or ``None``.
        """
        try:
            await self.workflow_runner.cleanup()
        except Exception as e:
            print_err(f"Error during cleanup: {e}")
            import traceback

            traceback.print_exc()

    def __del__(self) -> None:
        """Best-effort synchronous cleanup when the orchestrator is deleted.

        The interpreter may run it during shutdown with no event loop,
        so every failure is suppressed; normal runs should rely on the
        async context manager instead.
        """
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
    """Manual smoke entry: generate one workflow with a placeholder prompt.

    Developers run this file directly to exercise the generation
    factories without a full evolution loop. It builds the default
    config and orchestrator, then calls the multi-agent generation
    path; results print to stdout.
    """
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
"""Layer 0 planner for Mimosa-AI: goal to plan to executed tasks.

This module owns :class:`Planner`, which turns a free-text goal into a
literature-grounded, validated :class:`Plan` of :class:`PlanStep` objects and
then executes each step through the Layer 2 evolution engine. It also defines
the planner error surface: ``UserInterventionRequired``,
``PlanValidationError``, and ``DependencyError``. The CLI goal mode
(``sources/cli/onboard_cli.py``), the CSV benchmark mode
(``sources/benchmark_evaluation/csv_mode.py``), the ``sources.core``
re-exports, and the planner tests import this module. It runs when an
operator starts a multi-step goal session or a benchmark evaluation, before
any workflow executes.
"""

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from sources.cli.pretty_print import (
    BOLD,
    CYAN,
    DIM,
    RESET,
    print_err,
    print_info,
    print_ok,
    print_phase,
    print_section,
    print_summary,
    print_warn,
)
from sources.extensibility.text_to_speech import create_tts_service
from sources.utils.list_files import list_files
from sources.utils.notify import PushNotifier
from sources.utils.perspicacite_client import (
    query_perspicacite,
)
from sources.utils.planner_visualization import PlannerVisualizer

from .evolution_engine import EvolutionEngine
from .llm_provider import LLMConfig, LLMProvider, extract_model_pattern
from .schema import IndividualRun, Plan, PlanStep, Task, TaskStatus
from .workflow_selection import WorkflowSelector


class UserInterventionRequired(Exception):
    """A decision needs a human, and no human is reachable.

    ``start_planner`` and ``_request_human_plan_validation`` raise it when
    plan approval is requested but stdin is not a TTY, so an unattended run
    fails with the question it could not ask rather than with
    ``EOF when reading a line``. Unattended callers catch it and turn plan
    approval off.
    """


class PlanValidationError(Exception):
    """Raised when a generated plan fails structural validation.

    ``make_plan`` and ``_parse_and_validate_plan`` raise it when the LLM plan
    has no steps, an empty goal, or a step that cannot be built. Callers
    catch it to retry plan generation.
    """
    pass


class DependencyError(Exception):
    """Raised when a task's dependencies are not satisfied.

    It documents the dependency-gate failure case for planner users and is
    re-exported by ``sources.core``. No code in this module raises it today;
    the dependency gate currently reports through ``ValueError`` and console
    messages.
    """
    pass


class Planner:
    """Layer 0 planner: decompose a goal and drive step execution.

    The CLI goal mode, the CSV benchmark mode, and tests construct one
    ``Planner`` per session right after config loading. The planner turns
    the goal into a validated :class:`Plan`, runs every step through the
    Layer 2 evolution engine, and records each outcome in ``task_history``.
    Key invariant: ``task_history`` only grows with executed steps, and
    ``current_plan`` always holds the accepted plan.
    """

    def __init__(self, config: "Config", enable_tts: bool = True) -> None:
        """Set up planner state and collaborators.

        The planner owner (CLI goal mode, CSV benchmark mode, tests) calls
        it once, right after config loading. It stores the config, builds
        the evolution engine, workflow selector, Pushover notifier, and TTS
        service, resolves the planner LLM config from the configured model
        pattern, and records the host platform for visualization choices.

        Args:
            config: Runtime configuration object (from the run config file)
                exposing workspace paths, planner LLM settings, Pushover
                credentials, reasoning effort, and max_tokens.
            enable_tts: When True (constructor default), instantiate a
                text-to-speech service used for announcing task lifecycle
                events.

        Returns:
            None.

        Raises:
            ValueError: If ``config`` is ``None``.
        """
        if config is None:
            raise ValueError("❌ Planner: Configuration cannot be None")

        self.config = config
        self.workspace_path = config.workspace_dir
        self.evolve = EvolutionEngine(config)
        self.task_history: list[Task] = []
        self.current_plan: Plan | None = None
        self.wf_selector = WorkflowSelector(self.config)
        self.notifier = PushNotifier(config.pushover_token, config.pushover_user)
        provider, model = extract_model_pattern(self.config.planner_llm_model)
        self.config_llm = LLMConfig(
            model=model,
            provider=provider,
            reasoning_effort=self.config.reasoning_effort,
            max_tokens=getattr(self.config, 'max_tokens', 8192),
            openrouter_provider=None,
        )
        self._workspace_files_before_step: set[str] = set()  # Track files before step execution
        self.visualizer: PlannerVisualizer | None = None
        self.visualizer_thread: threading.Thread | None = None
        self.use_visualization: bool = True  # Can be disabled if pygame not available
        self.is_macos: bool = sys.platform == "darwin"  # Detect macOS for threading workaround
        self.is_windows: bool = sys.platform == "win32"  # Detect Windows for path handling
        self.tts = create_tts_service() if enable_tts else None

    def perspicacite_grounding(self, goal: str) -> str:
        """Fetch literature-grounded planning guidance from Perspicacite-AI.

        ``make_scientific_grounded_prompt`` calls it before plan generation,
        passing the operator's free-text goal. It sends a fixed
        literature-review prompt to the Perspicacite service
        (``sources.utils.perspicacite_client``) and returns the service text.

        Args:
            goal: Free-text task description; comes from the operator's goal.

        Returns:
            str: The literature guidance text, or a fallback message when the
            service fails or returns nothing.
        """
        prompt = f"""You are a scientific literature specialist supporting an AI expert on a task.

TASK TO SUPPORT:
{goal}

Retrieve peer-reviewed sources and established methodologies. Synthesize findings into actionable planning guidance.

OUTPUT FORMAT:
1. RELEVANT LITERATURE
   - Key papers/theory (cite authors, year, venue)
   - Foundational methods standard in this domain

2. ESTABLISHED WORKFLOW
   - Step-by-step methodology from current literature
   - Critical decision points and typical resolutions
   - Common pitfalls and literature-backed avoidance strategies

3. PRACTICAL GUIDANCE FOR PLANNER
   - Sub-tasks to create based on canonical approaches
   - Standard validation steps
   - Resource/data requirements

CONSTRAINTS: Prioritize reproducible, well-cited methods. Flag domain conventions. Note if literature is sparse/conflicting.
        """
        try:
            response = query_perspicacite(prompt) or "No relevant scientific context."
            return response
        except Exception:
            return "Query failed. Unable to help with scientific litterature"

    def make_scientific_grounded_prompt(self, goal: str) -> str:
        """Wrap the goal with retrieved scientific context.

        ``make_plan`` calls it to build the user prompt for each planning
        round. It queries Perspicacite-AI for literature first, then embeds
        that context plus the goal in a planning prompt that demands detailed
        task descriptions with full input and output file paths.

        Args:
            goal: Original free-text goal; comes from the operator through
                ``start_planner``.

        Returns:
            str: The planning prompt with scientific context embedded.
        """
        print_phase(
            "🔬 Querying Perspicacite-AI for scientific context... (This can take several minutes)"
        )
        scientific_context = self.perspicacite_grounding(goal)
        print(f"🔍 Scientific knowledge retrieved:\n{scientific_context[:2048]}...\n---")

        return f"""
You are a top-tier scientific in research. When generating the plan, please incorporate relevant scientific principles, theories, or findings that could inform the approach to achieving the goal. This will help ensure that the plan is not only practical but also grounded in scientific understanding.
Scientific context related to the goal:
{scientific_context}
You must generate a plan for goal:\n
{goal}\n
Important: Every task description should be very detailled and specific with the full path of all input output files specified.
"""

    def make_plan(self, system_prompt: str, goal_prompt: str, max_retries: int = 3) -> Plan:
        """Generate a validated plan from the planner LLM, with retries.

        ``_generate_plan_with_human_validation`` calls it once per planning
        round. It grounds the goal with literature, calls the planner LLM
        through the cached ``LLMProvider("plan_creator")``, extracts the
        JSON code block, and validates the plan; failed attempts wait with
        exponential backoff and, after the first, retry with the error text
        embedded. Exhausted attempts send a Pushover alert before raising.

        Args:
            system_prompt: Planner system prompt; read from the configured
                prompt file by ``_read_prompt``.
            goal_prompt: Goal text, possibly already extended with the
                workspace file listing, human feedback, or error context.
            max_retries: Generation attempt budget; caller state, default 3.

        Returns:
            Plan: The validated plan.

        Raises:
            ValueError: If either prompt is empty, or all attempts fail
                (parsing, validation, or LLM transport errors).
        """
        if not system_prompt or not isinstance(system_prompt, str):
            raise ValueError("❌ Planner: system_prompt must be a non-empty string")
        if not goal_prompt or not isinstance(goal_prompt, str):
            raise ValueError("❌ Planner: goal_prompt must be a non-empty string")

        last_error = None

        base_prompt = self.make_scientific_grounded_prompt(goal_prompt)
        prompt = base_prompt
        for attempt in range(1, max_retries + 1):
            try:
                print_info(f"Plan generation attempt {attempt}/{max_retries}")

                memory_path = self.config.memory_dir
                # Retries bypass the flat cache: with an unchanged prompt a
                # cached unparseable response would be replayed verbatim.
                raw_plan = LLMProvider("plan_creator", memory_path=memory_path, system_msg=system_prompt, config=self.config_llm, use_flat_cache=True)(prompt, use_cache=(attempt == 1))

                if not raw_plan or not isinstance(raw_plan, str):
                    raise ValueError("LLM returned empty or invalid response")

                print_info(f"Received plan response ({len(raw_plan)} characters)")
                plan_dict = self._extract_json_from_code_block(raw_plan)
                if plan_dict is None:
                    raise ValueError("Failed to extract valid JSON from LLM response\n")
                plan = self._parse_and_validate_plan(plan_dict, goal_prompt)
                print_ok(f"Plan generated and validated — {len(plan.steps)} step(s)")
                return plan

            except (ValueError, PlanValidationError, json.JSONDecodeError) as e:
                last_error = e
                error_msg = str(e)
                is_truncation = False
                if "Unterminated string" in error_msg or "Unexpected end of data" in error_msg:
                    is_truncation = True
                    error_msg = f"{error_msg} (This often indicates the response was truncated due to max_tokens limit)"
                print_warn(f"Attempt {attempt} failed: {error_msg}")
                if is_truncation:
                    print_info(f"Tip: consider increasing max_tokens (current: {getattr(self.config, 'max_tokens', 'not set')})")

                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    print_info(f"Waiting {wait_time}s before retry…")
                    time.sleep(wait_time)

                    # Feed the error back into the retry prompt so the LLM can
                    # correct its output format on the next attempt.
                    prompt = self._enhance_prompt_with_error(base_prompt, error_msg)
                else:
                    print_err(f"All {max_retries} attempts failed")

            except Exception as e:
                last_error = e
                print_err(f"Unexpected error in attempt {attempt}: {str(e)}")
                if attempt >= max_retries:
                    break
                time.sleep(2 ** attempt)

        # All retries exhausted - send notification
        error_details = f"Failed after {max_retries} attempts. Last error: {str(last_error)}"
        self.notifier.send_message(
            f"Plan generation failed after {max_retries} attempts\n"
            f"Goal: {goal_prompt[:128]}...\n"
            f"Error: {str(last_error)[:256]}",
            title="Plan generation failed",
            priority=1
        )
        raise ValueError(f"❌ Planner: Failed to generate a valid plan from the LLM. {error_details}") from last_error

    def _parse_and_validate_plan(self, plan_dict: dict[str, Any], goal: str) -> Plan:
        """Convert a raw plan dictionary into a validated Plan.

        ``make_plan`` calls it right after JSON extraction; the input is LLM
        output. It checks the steps list, builds each ``PlanStep`` with
        defaults for missing fields, anchors every step to the canonical
        goal, and finally builds the validated ``Plan``.

        Args:
            plan_dict: Dictionary containing plan data, with at least a
                non-empty ``"steps"`` list and optionally a ``"goal"`` string;
                comes from the LLM response.
            goal: Fallback goal text used when ``plan_dict`` does not supply
                one, and as the canonical goal stored on the returned plan.

        Returns:
            Plan: Validated plan object.

        Raises:
            PlanValidationError: If steps are missing, not a list, empty, or
                a step or the plan cannot be constructed.
        """
        if "steps" not in plan_dict:
            raise PlanValidationError("❌ Planner: No steps found in the generated plan")

        if not isinstance(plan_dict["steps"], list):
            raise PlanValidationError("❌ Planner: Steps should be a list")

        if not plan_dict["steps"]:
            raise PlanValidationError("❌ Planner: Plan must contain at least one step")

        plan_goal = plan_dict.get("goal", "") or goal
        if not plan_goal:
            raise PlanValidationError("❌ Planner: Plan must have a goal")

        steps = []
        for i, step_dict in enumerate(plan_dict["steps"]):
            try:
                step = PlanStep(
                    name=step_dict.get("name", f"step_{i}"),
                    task=step_dict.get("task", ""),
                    goal_context=plan_goal,
                    cost=0.0,
                    score=0.0,
                    depends_on=step_dict.get("depends_on", []),
                    required_inputs=step_dict.get("required_inputs", []),
                    expected_outputs=step_dict.get("expected_outputs", []),
                    complexity=step_dict.get("complexity", "medium")
                )
                steps.append(step)
            except ValueError as e:
                raise PlanValidationError(f"❌ Planner: Invalid step {i}: {str(e)}") from e

        try:
            plan = Plan(goal=goal, steps=steps)
        except ValueError as e:
            raise PlanValidationError(f"❌ Planner: Plan validation failed: {str(e)}") from e

        return plan

    @staticmethod
    def _extract_json_from_code_block(text: str) -> dict[str, Any] | None:
        """Extract a JSON object from an LLM response.

        Tries, in order: each fenced code block (```` ```json ```` or plain
        ```` ``` ````), the whole response when it starts with ``{``/``[``,
        and decoding from each ``{`` (to tolerate surrounding prose and
        earlier broken blocks). The first candidate that parses as a JSON
        object is returned.

        Args:
            text: Raw LLM response potentially containing a JSON plan.

        Returns:
            The decoded JSON object, or ``None`` when the text contains no
            JSON-looking candidate at all.

        Raises:
            json.JSONDecodeError: If JSON-looking candidates were found but
                none of them parse (the last decode error is re-raised so
                callers can detect e.g. truncation).
        """
        candidates = []
        for match in re.finditer(r"```[ \t]*(?:json)?[ \t]*\r?\n(.*?)```", text, re.DOTALL | re.IGNORECASE):
            block = match.group(1).strip()
            if block:
                candidates.append(block)

        stripped = text.strip()
        if stripped.startswith(("{", "[")):
            candidates.append(stripped)

        last_decode_error = None
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError as e:
                last_decode_error = e
                continue
            if isinstance(parsed, dict):
                return parsed

        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                parsed, _ = decoder.raw_decode(text, match.start())
            except json.JSONDecodeError as e:
                last_decode_error = e
                continue
            if isinstance(parsed, dict):
                return parsed

        if last_decode_error is not None:
            raise last_decode_error
        return None

    @staticmethod
    def _enhance_prompt_with_error(original_prompt: str, error_msg: str) -> str:
        """Fold the previous failure into the goal for the next retry.

        ``make_plan`` calls it between retries (after the first attempt) so
        the LLM avoids repeating the error. It embeds the error text plus
        strict JSON formatting requirements above the original request.

        Args:
            original_prompt: Goal prompt of the current planning round.
            error_msg: Error text from the failed attempt; comes from parse
                or validation failures.

        Returns:
            str: The retry prompt with error context embedded.
        """
        enhancement = f"""
IMPORTANT: Previous attempt failed with error:
{error_msg}

Please ensure your response:
1. Contains ONLY valid JSON (no additional text)
2. Uses proper JSON syntax with correct quotes and commas
3. Includes all required fields: "goal" and "steps"
4. Each step has: "name", "task", "depends_on", "required_inputs", "expected_outputs", "complexity"

Original request:
{original_prompt}
"""
        return enhancement

    def _read_prompt(self) -> str:
        """Load the planner system prompt from the configured file.

        ``_generate_plan_with_human_validation`` calls it once per planning
        round. It reads ``config.prompt_planner`` from the file system.

        Returns:
            str: The prompt file content.

        Raises:
            RuntimeError: If the prompt file cannot be read.
        """
        try:
            with open(self.config.prompt_planner, encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"❌ Planner: Error reading prompt: {str(e)}") from e

    def _build_knowledge_aware_task(self, task_description: str) -> str:
        """Prefix a step task with answers learned from completed tasks.

        ``run_attempts`` calls it before every evolution attempt; the input
        is the wrapped step task text. It collects the final answers of
        COMPLETED tasks in ``task_history`` and embeds them as prior
        knowledge above the task.

        Args:
            task_description: Current step task text; comes from the plan
                step.

        Returns:
            str: The task text unchanged when no completed answers exist,
            else the text wrapped with a "learned knowledge" section.
        """
        if not self.task_history:
            return task_description

        knowledge_sections = []
        for task in self.task_history:
            if task.final_answers and task.status == TaskStatus.COMPLETED:
                tfa = [str(a) for a in task.final_answers]
                answers_text = '\n\t - '.join(tfa)
                knowledge_sections.append(f"* From task '{task.name}':\n\t - {answers_text}")

        if not knowledge_sections:
            return task_description

        return '\n'.join([
            "From previous tasks you learned:",
            *knowledge_sections,
            "",
            "Now, use this knowledge to complete the following task:",
            task_description
        ])

    def _check_stop_condition(self, plan: Plan) -> bool:
        """Report whether the plan contains an explicit ``stop`` step.

        ``start_planner`` calls it right after plan acceptance; the input is
        the generated plan. It scans every step for the sentinel name or task
        "stop" so the LLM can end a session without execution.

        Args:
            plan: The plan whose steps are scanned for a sentinel ``stop`` name
                or task.

        Returns:
            bool: True when any step's task or name (case-insensitive,
            stripped) equals ``"stop"``; False otherwise.
        """
        for step in plan.steps:
            if step.task.lower().strip() == "stop" or step.name.lower().strip() == "stop":
                return True
        return False

    def _display_plan(self, plan: Plan) -> None:
        """Print the plan to the console for the operator.

        ``_generate_plan_with_human_validation`` calls it before validation,
        so goal-mode operators see what they approve. It prints the goal
        and, per step, the name, complexity, task, dependencies, inputs, and
        expected outputs.

        Args:
            plan: The plan to display.
        """
        print_phase("📋 EXECUTION PLAN")
        print(f"  {BOLD}Goal:{RESET} {plan.goal}\n")
        for i, step in enumerate(plan.steps, 1):
            print(f"  {CYAN}{BOLD}{i}. {step.name.upper()}{RESET}  {DIM}[{step.complexity}]{RESET}")
            print(f"     {step.task}")
            if step.depends_on:
                print(f"     {DIM}Depends on: {', '.join(step.depends_on)}{RESET}")
            if step.required_inputs:
                print(f"     {DIM}Inputs: {', '.join(step.required_inputs)}{RESET}")
            if step.expected_outputs:
                print(f"     {DIM}Outputs: {', '.join(step.expected_outputs)}{RESET}")
            print()

    def _request_human_plan_validation(self, plan: Plan) -> tuple[bool, str]:
        """Ask the operator to approve the displayed plan or send feedback.

        ``_generate_plan_with_human_validation`` calls it when plan approval
        is on, after ``_display_plan`` has printed the plan. It reads one
        line from stdin: an empty line approves, typed text becomes
        regeneration feedback.

        Args:
            plan: The plan presented to the user for review (used by the
                surrounding flow that calls :meth:`_display_plan` first).

        Returns:
            tuple[bool, str]: (is_approved, feedback)
                - is_approved: True if human pressed Enter (approve), False otherwise.
                - feedback: User's correction/feedback if plan not approved.

        Raises:
            UserInterventionRequired: If stdin is not a TTY, so the prompt
                could not be answered.
        """
        if not sys.stdin.isatty():
            raise UserInterventionRequired(
                "Plan approval was requested but stdin is not a TTY, so nobody "
                "can answer the prompt. Run interactively or unset "
                "planner_human_approve."
            )
        print_section("👤 HUMAN VALIDATION REQUIRED")
        print(f"  {DIM}Please review the plan above.{RESET}")
        print(f"  {DIM}Press [ENTER] to approve  ·  Type feedback and [ENTER] to regenerate{RESET}\n")

        user_input = input(f"  {BOLD}➤  Your decision: {RESET}").strip()
        if not user_input:
            print_ok("Plan approved. Proceeding with execution…")
            return True, ""
        else:
            print_info(f"Feedback received: {user_input}")
            print_info("Regenerating plan based on your feedback…")
            return False, user_input

    def _generate_plan_with_human_validation(self, goal: str, human_approve: bool = False) -> Plan:
        """Generate a plan, looping until approval when validation is on.

        ``start_planner`` calls it before any execution. Each round reads the
        system prompt, generates a plan, displays it, and — when approval is
        on — waits for the operator; typed feedback is folded into the next
        goal.

        Args:
            goal: Free-text goal; comes from the operator via
                ``start_planner``.
            human_approve: When True, prompt the user to approve or revise the
                generated plan before accepting it; when False, the first
                successfully generated plan is returned immediately.

        Returns:
            Plan: Human-approved plan object.

        Raises:
            ValueError: If plan generation fails.
            UserInterventionRequired: If approval is requested but stdin is
                not a TTY.
        """
        system_prompt = self._read_prompt()
        plan_approved = False
        human_feedback = ""

        while not plan_approved:
            current_goal = goal
            if human_feedback:
                current_goal = f"{goal}\n\nHUMAN FEEDBACK ON PREVIOUS PLAN:\n{human_feedback}\n\nPlease address this feedback in the new plan."
            plan = self.make_plan(system_prompt, current_goal)
            if plan is None:
                raise ValueError("❌ Planner: Failed to generate a valid plan")
            self._display_plan(plan)
            if human_approve:
                plan_approved, human_feedback = self._request_human_plan_validation(plan)
            else:
                plan_approved, human_feedback = True, ""
        return plan

    def _init_visualization(self, plan: Plan) -> None:
        """Start the pygame visualization for the accepted plan.

        ``start_planner`` calls it once, after the plan is accepted; the
        input is the accepted plan. On macOS the window is driven from the
        main thread (Cocoa requirement); on Linux and Windows a daemon
        thread pumps events at 30 FPS. Any failure disables visualization,
        and the run continues without it.

        Args:
            plan: The execution plan to visualize.
        """
        if not self.use_visualization:
            return

        try:
            self.visualizer = PlannerVisualizer(plan)
            print_ok("Visualization window initialized")

            # On Linux and Windows, use a separate thread for event handling.
            # On macOS, event handling must be done from main thread (will be called periodically)
            if not self.is_macos:
                def visualization_loop():
                    """Pump visualization events from the daemon thread.

                    Only the non-macOS branch starts this thread; it drains
                    window events at 30 FPS until the visualizer stops.
                    """
                    import pygame
                    clock = pygame.time.Clock()
                    while self.visualizer and self.visualizer.is_running():
                        self.visualizer.handle_events()
                        clock.tick(30)  # 30 FPS

                self.visualizer_thread = threading.Thread(target=visualization_loop, daemon=True)
                self.visualizer_thread.start()
                platform_name = "Windows" if self.is_windows else "Linux"
                print_info(f"Visualization running in separate thread ({platform_name})")
            else:
                print_info("Visualization will update from main thread (macOS)")

        except Exception as e:
            print_warn(f"Could not initialize visualization: {str(e)}")
            print_warn("Continuing without visualization…")
            self.use_visualization = False
            self.visualizer = None

    def _update_visualization(self, total_cost: float = 0.0) -> None:
        """Refresh the visualization window with current task states.

        ``start_planner`` calls it around every step; the input is the
        running ``task_history`` and the cumulative cost. On macOS it also
        pumps pygame events from the main thread, because no event thread
        exists there. Errors are warned and swallowed so the run continues.

        Args:
            total_cost: Cumulative cost so far; accumulated by
                ``start_planner``.
        """
        if self.visualizer and self.use_visualization:
            try:
                # On mac handle events from main thread
                if self.is_macos:
                    self.visualizer.handle_events()

                self.visualizer.update_tasks(self.task_history, total_cost=total_cost)
            except Exception as e:
                print_warn(f"Error updating visualization: {str(e)}")

    def _cleanup_visualization(self) -> None:
        """Close the visualization window and stop its event thread.

        ``start_planner`` calls it on success and inside its error handler,
        so the window never outlives the run. It stops the event loop,
        closes the window, joins the thread for at most 0.5 s, and clears
        the visualizer state. Errors are warned and swallowed.

        Returns:
            None.
        """
        if self.visualizer:
            try:
                # Mark visualizer as not running to stop the thread (if using thread)
                self.visualizer.running = False
                # On mac handle any remaining events before closing
                if self.is_macos:
                    self.visualizer.handle_events()
                self.visualizer.close()
                if self.visualizer_thread and self.visualizer_thread.is_alive():
                    self.visualizer_thread.join(timeout=0.5)
                self.visualizer = None
                self.visualizer_thread = None
                print_ok("Visualization window closed")
            except Exception as e:
                print_warn(f"Error closing visualization: {str(e)}")


    def _get_workspace_files(self) -> list[str]:
        """List all workspace files recursively as relative POSIX paths.

        Snapshot capture, input/output verification, and the produced-output
        diff all call it; the input is the file system state under
        ``workspace_path``. It skips hidden files and junk directories
        (dot-folders, ``__pycache__``, ``node_modules``, ``.git``) and always
        returns forward slashes so comparisons against plan-defined paths
        work on every platform including Windows.

        Returns:
            list[str]: Relative file paths from the workspace root, using
            forward slashes as separator; empty when the workspace is
            missing or the walk fails.
        """
        files = []
        try:
            workspace_path = Path(self.workspace_path)
            if not workspace_path.exists():
                print_warn(f"Workspace path does not exist: {self.workspace_path}")
                return files

            for root, dirs, filenames in os.walk(workspace_path):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', '.git']]

                for filename in filenames:
                    if not filename.startswith('.'):
                        file_path = Path(root) / filename
                        try:
                            relative_path = file_path.relative_to(workspace_path)
                            # Always use forward slashes for cross-platform consistency
                            files.append(relative_path.as_posix())
                        except ValueError:
                            continue
        except Exception as e:
            print_warn(f"Error scanning workspace files: {str(e)}")

        return files

    def _capture_workspace_snapshot(self) -> list[str]:
        """Record the workspace file set before a step runs.

        ``start_planner`` calls it immediately before ``run_attempts`` so the
        step's produced outputs can be diffed afterwards. It stores the file
        list on ``self._workspace_files_before_step``.

        Returns:
            list[str]: The captured file list (the same list stored on
            ``self._workspace_files_before_step``).
        """
        self._workspace_files_before_step = self._get_workspace_files()
        print_info(f"Workspace snapshot: {len(self._workspace_files_before_step)} file(s)")
        return self._workspace_files_before_step

    def _verify_required_inputs(self, step: PlanStep) -> tuple[bool, list[str]]:
        """Check that a step's required inputs exist in the workspace.

        No caller in the codebase today; it is the input-side gate that
        pairs with ``_verify_expected_outputs``. It scans the current
        workspace files and compares each declared input (normalised to
        forward slashes) against them.

        Args:
            step: The plan step whose ``required_inputs`` are checked; comes
                from the accepted plan.

        Returns:
            tuple[bool, list[str]]: (all_available, missing_inputs) — True
            when every required input exists, else the list of missing input
            paths.
        """
        missing_inputs = []

        workspace_files = self._get_workspace_files()
        for required_input in step.required_inputs:
            # Normalise to forward slashes so Windows backslashes never cause
            # a false-negative when comparing against plan-defined paths.
            normalised_input = Path(required_input).as_posix()
            if normalised_input not in workspace_files:
                missing_inputs.append(required_input)
        return len(missing_inputs) == 0, missing_inputs

    def _verify_expected_outputs(self, step: PlanStep) -> tuple[bool, list[str]]:
        """Check that a step's expected outputs exist in the workspace.

        ``_can_execute_step`` calls it for finished dependencies and
        ``run_attempts`` for the just-finished step; the input is a
        ``PlanStep``. A workspace file matches an expected output when its
        path contains the expected path, or when at least 70 percent of the
        expected file-stem words (split on ``_``, ``-``, ``.``) appear in
        the file stem.

        Args:
            step: The plan step whose ``expected_outputs`` are checked.

        Returns:
            tuple[bool, list[str]]: (all_produced, missing_outputs) — True
            when every expected output matches a workspace file, else the
            list of unmatched expected paths.
        """
        missing_outputs = []
        workspace_files = self._get_workspace_files()

        for expected_output in step.expected_outputs:
            # Normalise expected path to forward slashes for cross-platform comparison
            normalised_expected = Path(expected_output).as_posix()
            # Use Path.stem to strip the extension in a platform-agnostic way
            exp_stem = Path(expected_output).stem.lower()
            exp_terms = set(re.sub(r'[_\-.]', ' ', exp_stem).split())
            found = any(
                normalised_expected in actual or
                len(exp_terms & set(re.sub(r'[_\-.]', ' ', Path(actual).stem.lower()).split())) >= len(exp_terms) * 0.7
                for actual in workspace_files
            )
            if not found:
                missing_outputs.append(expected_output)

        return len(missing_outputs) == 0, missing_outputs

    def _can_execute_step(self, step: PlanStep) -> tuple[bool, list[str]]:
        """Check that a step's dependencies completed with outputs on disk.

        ``start_planner`` calls it before each step that declares
        dependencies; the input is the next ``PlanStep``. A dependency
        passes when its ``Task`` in ``task_history`` is COMPLETED and its
        expected outputs pass ``_verify_expected_outputs``.

        Args:
            step: The plan step to check.

        Returns:
            tuple[bool, list[str]]: (can_execute, missing_dependencies) —
            True when every dependency is done, else the names of missing or
            output-incomplete dependencies.
        """
        missing_deps = []

        for dep_name in step.depends_on:
            dep_task = next((task for task in self.task_history if task.name == dep_name), None)
            if dep_task is None or dep_task.status != TaskStatus.COMPLETED:
                missing_deps.append(dep_name)
                continue
            expected_outputs = getattr(dep_task, "expected_outputs", None)
            if expected_outputs:
                outputs_ok, missing_outputs = self._verify_expected_outputs(dep_task)
                if not outputs_ok:
                    missing_deps.append(
                        f"{dep_name}[missing_outputs:{','.join(missing_outputs)}]"
                    )
        return len(missing_deps) == 0, missing_deps

    def request_user_exit(self, msg: str) -> None:
        """Ask the operator to continue or abort when the run is stuck.

        ``start_planner`` calls it when a step cannot proceed (for example on
        missing dependencies). It sends a Pushover notification, prints the
        message, and reads one y/n line from stdin.

        Args:
            msg: Message shown both in the Pushover notification body and
                printed to stdout before the prompt.

        Returns:
            None: When the operator answers y or yes.

        Raises:
            SystemExit: When the operator answers anything else; the process
                exits with status 1.
        """
        self.notifier.send_message(
            f"Mimosa is requesting exit:\n{msg}",
            title="Mimosa exit request."
        )
        print(msg)
        choice = input("\nContinue ? (y(yes)/n(no))")
        if choice.lower() == "y" or choice.lower() == "yes":
            return
        print("\n---\nExited upon user request.\n---\n")
        exit(1)

    async def evolve_runs(
        self,
        task: str,
        judge: bool,
        cached_wf_allow: bool = True,
        original_task: str | None = None,
    ) -> list[IndividualRun]:
        """Run one step's iterative learning, reusing cached workflows first.

        ``run_attempts`` calls it for every attempt; the input is the
        knowledge-wrapped task text. On the first attempt it looks for a
        near-perfect cached workflow through the ``WorkflowSelector`` and,
        when one succeeded before, returns that single run; otherwise it
        starts a fresh evolution session (Layer 2) and returns its runs.

        Args:
            task: Task description string (may be knowledge-wrapped); comes
                from ``run_attempts``.
            judge: Whether to use judge evaluation; comes from the run
                configuration.
            cached_wf_allow: Whether to allow reusing high-quality cached
                workflows discovered by the workflow selector; True on the
                first attempt only.
            original_task: Original unwrapped task for similarity matching;
                used in preference to ``task`` for cache lookup.

        Returns:
            List[IndividualRun]: List of Evolution runs (possibly a single
            cached run, the runs produced by the evolution engine, or an
            empty list when no runs were produced).

        Raises:
            ValueError: If task is invalid or Evolution execution fails.
        """
        if not task or not isinstance(task, str):
            raise ValueError("❌ Planner: Task must be a non-empty string")


        print_info(f"Starting Iterative-Learning for task: {task[:60]}…")

        try:
            # Use original_task for lookup to avoid knowledge wrapper interference
            lookup_task = original_task if original_task else task

            # Check for high-quality cached workflows
            past_wf_lookups = self.wf_selector.select_best_workflows(
                lookup_task, threshold_similarity=0.98, threshold_score=0.99
            ) if cached_wf_allow else []

            if past_wf_lookups and len(past_wf_lookups) > 0:
                best_match = past_wf_lookups[0]
                if best_match is None:
                    print_warn("Best match is None, proceeding with new Evolution run")
                #elif self._get_evolve_success(best_match):
                elif best_match.is_success:
                    print_ok(f"Using cached workflow result  UUID: {getattr(best_match, 'uuid', 'N/A')}")

                    run = IndividualRun(
                        goal=best_match.goal,
                        prompt=best_match.goal,
                        answers=best_match.answers,
                        state_result=best_match.state_result,
                        current_uuid=best_match.uuid,
                        reward=best_match.overall_score,
                        workflow_template=best_match.code
                    )
                    return [run]

            # Generate new workflows via Evolution

            if self.evolve is None:
                raise ValueError("❌ Planner: instance is None")

            runs = await self.evolve.start_workflow_evolution(
                goal=task,
                template_uuid=None,
                judge=judge,
                enable_evolution=True,
                original_task=original_task
            )

            if runs is None:
                print_warn("Runs is None, returning empty list")
                return []

            return runs

        except Exception as e:
            raise ValueError(f"❌ Planner: Evolution execution failed: {str(e)}") from e

    def _get_evolve_success(self, run: IndividualRun) -> bool:
        """Read the final success flag from a run's ``state_result``.

        ``run_attempts`` calls it to decide step completion; the input is the
        last ``IndividualRun`` of an attempt. It reads the last element of
        ``state_result["success"]``, where the verifier persists per-step
        success flags.

        Args:
            run: An individual evolution run whose ``state_result`` carries a
                ``"success"`` list of booleans.

        Returns:
            bool: The last element of the ``success`` list, or ``False`` when
            the field is missing or malformed.
        """
        run_state_result = getattr(run, 'state_result', None) or {}
        success_list = run_state_result.get('success', [False]) if isinstance(run_state_result, dict) else [False]
        return success_list[-1]

    async def run_attempts(
        self,
        attempt_counts: dict[str, int],
        max_attempts: int,
        step: PlanStep,
        judge: bool,
    ) -> PlanStep:
        """Execute one plan step with retries and record the outcome.

        ``start_planner`` calls it for every step of the accepted plan. Each
        attempt wraps the step task with prior knowledge, runs
        ``evolve_runs``, and appends a ``Task`` to ``task_history``. A run
        with success and a score of at least 0.7 is accepted after its
        expected outputs are checked on disk; low scores retry until
        ``max_attempts``. Attempt cost and the last score land on the step,
        and TTS announces progress and results when enabled.

        Args:
            attempt_counts: Per-step attempt counters shared across the plan;
                caller state from ``start_planner`` (recreated when None).
            max_attempts: Retry budget per step; comes from
                ``max_task_retry`` (clamped to at least 1 when invalid).
            step: The plan step to execute; comes from the accepted plan.
            judge: Whether to use judge evaluation; run configuration flag.

        Returns:
            PlanStep: The same step with cost, score, and status updated
            (unchanged when ``step`` is None).
        """
        if step is None:
            print_err("Step is None, cannot execute")
            return step

        if attempt_counts is None:
            attempt_counts = {}

        if max_attempts is None or max_attempts < 1:
            max_attempts = 1
            print_warn(f"Invalid max_attempts, using default: {max_attempts}")

        step_name = getattr(step, 'name', 'unknown_step')
        goal = getattr(step, 'goal_context', '')
        task = getattr(step, 'task', '')
        step_task = f"Broader context:{goal}\n---\nYour task:{task}"
        attempt = attempt_counts.get(step_name, 0)
        attempt_cost = 0
        attempt_score = 0.0
        final_answers = []
        while attempt < max_attempts:
            attempt += 1
            attempt_counts[step_name] = attempt

            print_info(f"Attempt {attempt}/{max_attempts} for task: {step_name}")
            if self.tts:
                self.tts.speak(f"now starting task {step_name}", voice_index=0)

            try:
                enhanced_task = self._build_knowledge_aware_task(step_task)
                # Pass both enhanced task and original task for proper workflow matching
                evolve_runs = await self.evolve_runs(
                    enhanced_task,
                    judge,
                    cached_wf_allow=(attempt<=1),
                    original_task=step_task  # Pass original for similarity matching
                )

                attempt_cost += sum([r.cost for r in evolve_runs])
                last_run = evolve_runs[-1]

                final_answers = []
                final_uuid = None
                workflow_uuid = None

                if last_run is not None:
                    final_answers = getattr(last_run, 'answers', []) or []
                    final_uuid = getattr(last_run, 'current_uuid', None)
                    workflow_uuid = getattr(last_run, 'workflow_template', None)

                evolve_success = self._get_evolve_success(last_run)
                attempt_score = last_run.reward
                task = Task(
                    name=step_name,
                    description=step_task,
                    evolve_runs=evolve_runs or [],
                    final_answers=final_answers,
                    final_uuid=final_uuid,
                    workflow_uuid=workflow_uuid,
                    status=TaskStatus.COMPLETED if evolve_success else TaskStatus.FAILED,
                    depends_on=getattr(step, 'depends_on', []) or [],
                    required_inputs=getattr(step, 'required_inputs', []) or [],
                    expected_outputs=getattr(step, 'expected_outputs', []) or [],
                    complexity=getattr(step, 'complexity', 'medium'),
                    produced_outputs=[f for f in self._get_workspace_files() if f not in self._workspace_files_before_step]
                )

                self.task_history.append(task)

                if evolve_success and attempt_score >= 0.7:
                    time.sleep(10) # wait for files update
                    outputs_produced, missing_outputs = self._verify_expected_outputs(step)
                    step.status = TaskStatus.COMPLETED
                    if outputs_produced:
                        print_ok(f"Task '{step_name}' completed successfully")
                        break
                    else:
                        print_warn(f"Task '{step_name}' completed but missing expected outputs: {missing_outputs}")
                        break
                else:
                    print_err(f"Task {step_name} (uuid: {final_uuid}) failed with score {attempt_score}")
                    if self.tts:
                        self.tts.speak(f"Task {step_name} failure, retrying...", voice_index=0)
                    continue

            except Exception as e:
                raise e

        step.cost = attempt_cost
        step.score = attempt_score
        if self.tts:
            answer = '. '.join([x[:128] for x in final_answers if x]) if final_answers else "No answers produced."
            tts_text = f"""
            Task completed. Score: {attempt_score}, Cost: {attempt_cost}. {answer}
            """
            self.tts.speak(tts_text, voice_index=0)
        return step

    async def start_planner(
        self,
        goal: str,
        judge: bool = True,
        max_task_retry: int = 5,
        human_approve: bool = False,
    ) -> list[Task]:
        """Plan a goal and execute every step end to end.

        The CLI goal mode and the CSV benchmark mode call it; the input is
        the operator's free-text goal. It prepends the workspace file listing
        to the goal, generates and approves a plan (optionally with a human
        in the loop), runs every step with retries through the evolution
        engine, notifies Pushover on failure and completion, and cleans up
        the visualization window on any exit path.

        Args:
            goal: Free-text operator goal.
            judge: Whether to use a judge for evaluation; run configuration
                flag, default True.
            max_task_retry: Maximum number of retries for each task; caller
                state, default 5.
            human_approve: When True, show the plan and wait for the operator
                to approve it (Enter) or send it back with feedback before
                any step runs. Interactive runs only.

        Returns:
            List[Task]: The executed task history; returned early when the
            plan carries the ``stop`` sentinel.

        Raises:
            ValueError: If goal is invalid or planning fails
            UserInterventionRequired: If approval was requested but stdin is
                not a TTY
        """
        if not goal or not isinstance(goal, str):
            raise ValueError("❌ Planner: Goal must be a non-empty string")

        if human_approve and not sys.stdin.isatty():
            # Refuse before the planning LLM call: otherwise the plan is
            # generated, printed into a log, and the run dies on input().
            raise UserInterventionRequired(
                "planner_human_approve is set but stdin is not a TTY, so the "
                "plan could not be approved. Run interactively or unset it."
            )

        goal = "\nAvailable files:\n" + list_files(self.config.workspace_dir) + "\n" + goal
        print_info(f"Starting planner with goal: {goal[:80]}…")

        try:
            # Generate plan with human validation loop
            self.current_plan = self._generate_plan_with_human_validation(
                goal, human_approve=human_approve
            )

            if self.current_plan is None:
                raise ValueError("❌ Planner: Failed to generate a valid plan")

            # Initialize visualization after plan is approved
            self._init_visualization(self.current_plan)
            # Check for stop condition
            if self._check_stop_condition(self.current_plan):
                print_info("Stop condition found in plan. Exiting.")
                return self.task_history

            # Validate plan has steps
            if not hasattr(self.current_plan, 'steps') or not self.current_plan.steps:
                raise ValueError("❌ Planner: Plan has no executable steps")

            # Execute plan steps
            attempt_counts = {}

            lst_step = None
            total_cost = 0
            for step_idx, step in enumerate(self.current_plan.steps):
                if step is None:
                    print_warn(f"Step {step_idx + 1} is None, skipping")
                    continue
                step_name = getattr(step, 'name', f'step_{step_idx}')
                print_phase(
                    f"📋 STEP {step_idx + 1}/{len(self.current_plan.steps)}  ·  {step_name}",
                )
                # Check if step can be executed (dependencies satisfied)
                if lst_step:
                    can_execute, missing_deps = self._can_execute_step(step)
                    if not can_execute:
                        self.request_user_exit(f"Cannot execute step '{step_name}' — missing dependencies: {missing_deps}")

                # Execute the step with retry logic
                step.status = TaskStatus.RUNNING
                self._update_visualization(total_cost)  # Update to show running status
                max_attempts = max_task_retry

                try:
                    self._capture_workspace_snapshot()  # Snapshot before execution for output diff
                    step = await self.run_attempts(attempt_counts, max_attempts, step, judge)
                    total_cost += step.cost
                    self._update_visualization(total_cost)  # Update after step completes
                except Exception as e:
                    step.status = TaskStatus.FAILED
                    self._update_visualization(total_cost)  # Update to show failed status
                    raise Exception(f"❌ Critical error in step execution: {str(e)}") from e
                lst_step = step

                if step.status != TaskStatus.COMPLETED:
                    step.status = TaskStatus.FAILED
                    self.notifier.send_message(
                        f"Task '{step_name}' failed after {max_attempts} attempts\n"
                        f"Goal: {goal[:128]}...\n"
                        f"Step: {step_idx + 1}/{len(self.current_plan.steps)}",
                        title=f"Task '{step_name}' failed",
                        priority=1
                    )
                    raise Exception(f"❌ Giving up on task '{step_name}' after {max_attempts} attempts")

            completed_tasks = sum(1 for t in self.task_history if t.status == TaskStatus.COMPLETED)
            print_summary(
                "🏁 PLANNER EXECUTION COMPLETE",
                [
                    ("Tasks executed", str(len(self.task_history))),
                    ("Tasks completed", str(completed_tasks)),
                    ("Total cost", f"${total_cost:.6f}"),
                ],
            )
            self.notifier.send_message(
                f"Planner completed successfully!\n"
                f"Goal: {goal[:128]}...\n"
                f"Completed: {completed_tasks}/{len(self.task_history)} tasks",
                title="Planner execution completed",
                priority=0
            )

            if self.visualizer:
                self._cleanup_visualization()
            return self.task_history

        except Exception as e:
            print_err(f"Critical error in planner execution: {str(e)}")
            self.notifier.send_message(str(e), title="error during Mimosa execution.")
            self._cleanup_visualization()
            raise ValueError(f"❌ Planner: Execution failed: {str(e)}") from e

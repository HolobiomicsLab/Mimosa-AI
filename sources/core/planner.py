import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path, PurePosixPath
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
from sources.utils.llm_json import loads_llm_json
from sources.utils.perspicacite_client import (
    query_perspicacite,
)
from sources.utils.planner_visualization import PlannerVisualizer

from . import declared_outputs
from .artifact_contracts import (
    ArtifactContract,
    ArtifactValidationError,
    ContractValidationError,
    load_artifact_contract,
)
from .evolution_engine import EvolutionEngine
from .llm_provider import LLMConfig, LLMProvider, extract_model_pattern
from .schema import IndividualRun, Plan, PlanStep, Task, TaskStatus
from .workflow_selection import WorkflowSelector


class UserInterventionRequired(Exception):
    """A decision needs a human, and no human is reachable.

    Raised instead of blocking on ``input()`` when stdin is not a TTY, so an
    unattended run fails with the question it could not ask rather than with
    ``EOF when reading a line``.
    """


class PlanValidationError(Exception):
    """Exception raised when plan validation fails."""
    pass


class DependencyError(Exception):
    """Exception raised when task dependencies are not satisfied."""
    pass


_CONTRACT_PLANNER_SYSTEM_PROMPT = """You order an already-authorized execution graph.
Return exactly one JSON object with contract_digest and steps. Each steps entry
must contain only name. Include every required step exactly once in a valid
topological order. Do not add, remove, rename, or redefine any step or artifact.
"""


class Planner:
    """
    Enhanced planner class for long-term task planning with dependency management
    and input/output verification.
    """

    def __init__(self, config: "Config", enable_tts: bool = True) -> None:
        """Initialize the planner.

        Args:
            config: Configuration object exposing workspace paths, planner
                LLM settings, Pushover credentials, and reasoning effort.
            enable_tts: When True, instantiate a text-to-speech service used
                for announcing task lifecycle events.

        Raises:
            ValueError: If ``config`` is ``None``.
        """
        if config is None:
            raise ValueError("❌ Planner: Configuration cannot be None")

        self.config = config
        self.logger = logging.getLogger(__name__)
        self.workspace_path = config.workspace_dir
        self.evolve = EvolutionEngine(config)
        self.task_history: list[Task] = []
        self.current_plan: Plan | None = None
        self._active_contract: ArtifactContract | None = None
        self._contract_tasks: dict[str, Task] = {}
        self.wf_selector = WorkflowSelector(self.config)
        self.notifier = PushNotifier(config.pushover_token, config.pushover_user)
        provider, model = extract_model_pattern(self.config.planner_llm_model)
        api_base, api_key_env = self.config.completion_endpoint_for(
            self.config.planner_llm_model
        )
        self.config_llm = LLMConfig(
            model=model,
            provider=provider,
            reasoning_effort=self.config.reasoning_effort,
            max_tokens=getattr(self.config, 'max_tokens', 8192),
            openrouter_provider=None,
            api_base=api_base,
            api_key_env=api_key_env,
            harness_auth_mode=self.config.harness_auth_mode,
        )
        self._workspace_files_before_step: set[str] = set()  # Track files before step execution
        self.visualizer: PlannerVisualizer | None = None
        self.visualizer_thread: threading.Thread | None = None
        self.use_visualization: bool = True  # Can be disabled if pygame not available
        self.is_macos: bool = sys.platform == "darwin"  # Detect macOS for threading workaround
        self.is_windows: bool = sys.platform == "win32"  # Detect Windows for path handling
        self.tts = create_tts_service() if enable_tts else None

    def _resolve_artifact_contract(
        self,
        contract: ArtifactContract | None = None,
        contract_path: str | Path | None = None,
    ) -> ArtifactContract | None:
        """Resolve an explicit contract or the configured trusted file."""
        if contract is not None and contract_path is not None:
            raise ContractValidationError("Pass a contract object or path, not both")
        if contract is not None:
            return contract
        selected = contract_path
        if selected is None:
            selected = getattr(self.config, "planner_contract_path", None)
        return load_artifact_contract(selected) if selected is not None else None

    @staticmethod
    def _contract_plan_prompt(goal: str, contract: ArtifactContract) -> str:
        return json.dumps(
            {
                "goal": goal,
                "authoritative_contract": contract.planner_projection(),
                "response_shape": {
                    "contract_digest": contract.digest,
                    "steps": [{"name": "canonical_step_name"}],
                },
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )

    def perspicacite_grounding(self, goal: str) -> str:
        """Query Perspicacite-AI for literature-grounded planning guidance.

        Args:
            goal: The task description for which scientific context is needed.

        Returns:
            The literature-grounded response text, or a fallback message when
            the service is unavailable or returns no relevant context.
        """
        # Agent-side grounding is opt-out for fair benchmarking (default web
        # search can surface the source paper); disable or point at a leak-free
        # KB via the perspicacite_agent_* config fields.
        if not self.config.perspicacite_agent_grounding_enabled:
            return "No relevant scientific context."
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
            response = query_perspicacite(
                prompt, kb_name=self.config.perspicacite_agent_kb_name
            ) or "No relevant scientific context."
            return response
        except Exception:
            return "Query failed. Unable to help with scientific litterature"

    def make_scientific_grounded_prompt(self, goal: str) -> str:
        """
        Create a scientific-grounded prompt by incorporating relevant scientific knowledge.
        Args:
            goal: The original goal description
        Returns:
            str: Enhanced prompt with scientific context
        """
        # Only announce the query when agent-side grounding is actually enabled;
        # perspicacite_grounding() itself returns a no-context sentinel when it
        # is disabled, so printing "Querying…" there would be misleading.
        grounding_on = getattr(self.config, "perspicacite_agent_grounding_enabled", True)
        if grounding_on:
            print_phase(
                f"🔬 Querying Perspicacite-AI for scientific context... (This can take several minutes)"
            )
        scientific_context = self.perspicacite_grounding(goal)
        if grounding_on:
            print(f"🔍 Scientific knowledge retrieved:\n{scientific_context[:2048]}...\n---")

        return f"""
You are a top-tier scientific in research. When generating the plan, please incorporate relevant scientific principles, theories, or findings that could inform the approach to achieving the goal. This will help ensure that the plan is not only practical but also grounded in scientific understanding.
Scientific context related to the goal:
{scientific_context}
You must generate a plan for goal:\n
{goal}\n
Important: Every task description should be very detailled and specific with the full path of all input output files specified.
"""

    def make_plan(
        self,
        system_prompt: str,
        goal_prompt: str,
        max_retries: int = 3,
        contract: ArtifactContract | None = None,
    ) -> Plan:
        """
        Generate a workflow plan using the LLM with retry logic and multiple parsing strategies.
        Args:
            system_prompt: The system prompt for plan generation
            goal_prompt: The goal description
            max_retries: Maximum number of retry attempts (default: 3)
        Returns:
            Plan: Validated plan object
        Raises:
            ValueError: If plan generation or validation fails after all retries
        """
        if not system_prompt or not isinstance(system_prompt, str):
            raise ValueError("❌ Planner: system_prompt must be a non-empty string")
        if not goal_prompt or not isinstance(goal_prompt, str):
            raise ValueError("❌ Planner: goal_prompt must be a non-empty string")

        last_error = None
        contract = self._resolve_artifact_contract(contract)
        if contract is not None:
            system_prompt = _CONTRACT_PLANNER_SYSTEM_PROMPT
            prompt = self._contract_plan_prompt(goal_prompt, contract)
        else:
            prompt = self.make_scientific_grounded_prompt(goal_prompt)
        for attempt in range(1, max_retries + 1):
            try:
                print_info(f"Plan generation attempt {attempt}/{max_retries}")

                memory_path = self.config.memory_dir
                raw_plan = LLMProvider("plan_creator", memory_path=memory_path, system_msg=system_prompt, config=self.config_llm, use_flat_cache=True)(
                    prompt, use_cache=contract is None
                )

                if not raw_plan or not isinstance(raw_plan, str):
                    raise ValueError("LLM returned empty or invalid response")

                print_info(f"Received plan response ({len(raw_plan)} characters)")
                plan_dict = self._extract_json_from_code_block(raw_plan)
                if plan_dict is None:
                    raise ValueError("Failed to extract valid JSON from LLM response\n")
                plan = self._parse_and_validate_plan(plan_dict, goal_prompt, contract)
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

                    if attempt > 1:
                        goal_prompt = self._enhance_prompt_with_error(goal_prompt, error_msg)
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

    def _parse_and_validate_plan(
        self,
        plan_dict: dict[str, Any],
        goal: str,
        contract: ArtifactContract | None = None,
    ) -> Plan:
        """
        Parse and validate a plan dictionary into a Plan object.

        Args:
            plan_dict: Dictionary containing plan data, with at least a
                non-empty ``"steps"`` list and optionally a ``"goal"`` string.
            goal: Fallback goal text used when ``plan_dict`` does not supply
                one, and as the canonical goal stored on the returned plan.

        Returns:
            Plan: Validated plan object.

        Raises:
            PlanValidationError: If plan validation fails.
        """
        if contract is not None:
            return contract.project_plan(plan_dict, goal)
        if not isinstance(plan_dict, dict):
            raise PlanValidationError("❌ Planner: Plan must be a JSON object")
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
        """Extract the plan object from an LLM response.

        Accepts a fenced json code block, a bare JSON response, or JSON
        preceded by a sentence of prose.

        Args:
            text: Raw LLM response.

        Returns:
            The decoded JSON object, or ``None`` when the response holds no
            parsable JSON at all.

        Raises:
            json.JSONDecodeError: If the block cannot be parsed even after
                repairing the defects LLMs emit (raw control characters,
                bare interior quotes, trailing prose).
        """
        code_blocks = []
        in_code_block = False

        for line in text.splitlines():
            line_stripped = line.strip()
            if line_stripped.startswith("```json") or line_stripped.startswith("```JSON"):
                in_code_block = True
                continue
            if line_stripped.startswith("```") and in_code_block:
                in_code_block = False
                continue
            if in_code_block:
                code_blocks.append(line)

        if code_blocks:
            json_str = "\n".join(code_blocks)
            # Tolerant parse: a model that pastes a multi-line span into a
            # string value produces "Invalid control character" and strict
            # parsing discards an otherwise complete plan. Valid JSON is
            # unaffected.
            return loads_llm_json(json_str)

        # No fence. A model told to answer in JSON frequently just answers in
        # JSON — observed against stealth/ox-alpha, which returned a valid
        # 7.5 kB plan object with no fence and had it discarded here. Try the
        # bare response, then from the first brace so a leading sentence
        # ("Here is the plan:") does not cost the plan either.
        brace = text.find("{")
        for candidate in (text, text[brace:] if brace != -1 else ""):
            if not candidate.strip():
                continue
            try:
                return loads_llm_json(candidate)
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _enhance_prompt_with_error(original_prompt: str, error_msg: str) -> str:
        """
        Enhance the prompt with error context for retry attempts.
        Args:
            original_prompt: Original goal prompt
            error_msg: Error message from previous attempt
        Returns:
            str: Enhanced prompt
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
        """
        Read the planner prompt from the configuration.
        Returns:
            str: The prompt content
        Raises:
            RuntimeError: If prompt file cannot be read
        """
        try:
            with open(self.config.prompt_planner, encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"❌ Planner: Error reading prompt: {str(e)}") from e

    def _build_knowledge_aware_task(self, task_description: str) -> str:
        """
        Create a task prompt that includes knowledge from previous tasks.
        Args:
            task_description: The current task description
        Returns:
            str: Enhanced task description with previous knowledge
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
        """Return True if the plan contains an explicit ``stop`` step.

        Args:
            plan: The plan whose steps are scanned for a sentinel ``stop`` name
                or task.

        Returns:
            True when any step's task or name (case-insensitive, stripped)
            equals ``"stop"``; False otherwise.
        """
        for step in plan.steps:
            if step.task.lower().strip() == "stop" or step.name.lower().strip() == "stop":
                return True
        return False

    def _display_plan(self, plan: Plan) -> None:
        """
        Display the plan steps in a readable format.
        Args:
            plan: The plan to display
        """
        print_phase("📋 EXECUTION PLAN")
        print(f"  {BOLD}Goal:{RESET} {plan.goal}\n")
        print(f"  {BOLD}Artifact contract:{RESET} {plan.contract_status}")
        if plan.contract_digest:
            print(f"  {DIM}Digest: {plan.contract_digest}{RESET}\n")
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
        """
        Request human validation of the generated plan.

        Args:
            plan: The plan presented to the user for review (used by the
                surrounding flow that calls :meth:`_display_plan` first).

        Returns:
            tuple[bool, str]: (is_approved, feedback)
                - is_approved: True if human pressed Enter (approve), False otherwise.
                - feedback: User's correction/feedback if plan not approved.
        """
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

    def _generate_plan_with_human_validation(
        self,
        goal: str,
        human_approve: bool = False,
        contract: ArtifactContract | None = None,
    ) -> Plan:
        """
        Generate a plan with iterative human validation and feedback loop.

        Args:
            goal: The goal description for the planner.
            human_approve: When True, prompt the user to approve or revise the
                generated plan before accepting it; when False, the first
                successfully generated plan is returned immediately.

        Returns:
            Plan: Human-approved plan object.

        Raises:
            ValueError: If plan generation fails.
        """
        system_prompt = self._read_prompt()
        plan_approved = False
        human_feedback = ""

        while not plan_approved:
            current_goal = goal
            if human_feedback:
                current_goal = f"{goal}\n\nHUMAN FEEDBACK ON PREVIOUS PLAN:\n{human_feedback}\n\nPlease address this feedback in the new plan."
            plan = self.make_plan(system_prompt, current_goal, contract=contract)
            if plan is None:
                raise ValueError("❌ Planner: Failed to generate a valid plan")
            self._display_plan(plan)
            if human_approve:
                plan_approved, human_feedback = self._request_human_plan_validation(plan)
            else:
                plan_approved, human_feedback = True, ""
        return plan

    def _init_visualization(self, plan: Plan) -> None:
        """
        Initialize the pygame visualization window.
        On macOS, pygame must run on the main thread due to Cocoa requirements.
        On Linux and Windows, it can run in a separate thread for better performance.

        Args:
            plan: The execution plan to visualize
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
        """
        Update the visualization with the current task states.
        On macOS, also handle events since we can't use a separate thread.

        Args:
            total_cost: The cumulative cost to display
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
        """
        Clean up and close the visualization window.
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
        """
        Get all files in the workspace directory recursively.
        Paths are always returned with forward slashes so that comparisons
        against plan-defined paths (which use forward slashes) work correctly
        on every platform including Windows.

        Returns:
            list[str]: List of relative file paths from workspace root
                       using forward slashes as separator.
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
        """
        Capture a snapshot of current workspace files before step execution.

        Returns:
            list[str]: The list of workspace files captured (also stored on
            ``self._workspace_files_before_step`` for later diffing).
        """
        self._workspace_files_before_step = self._get_workspace_files()
        print_info(f"Workspace snapshot: {len(self._workspace_files_before_step)} file(s)")
        return self._workspace_files_before_step

    def _verify_required_inputs(self, step: PlanStep) -> tuple[bool, list[str]]:
        """
        Verify that all required inputs for a step are available.
        Args:
            step: The plan step to verify
        Returns:
            Tuple[bool, List[str]]: (all_available, missing_inputs)
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
        """
        Verify that expected outputs files were produced.
        Args:
            step: The plan step
            produced_outputs: List of actually produced outputs
        Returns:
            Tuple[bool, List[str]]: (all_produced, missing_outputs)
        """
        missing_outputs = []
        workspace_files = self._get_workspace_files()

        for expected_output in step.expected_outputs:
            # A plan may declare a *directory* as an output ("/workspace/data/").
            # The workspace scan yields files only, so such an output could never
            # be matched and the step stayed permanently "missing outputs" —
            # which then blocked every dependent step. Observed on p_iimn
            # 2026-08-22: data_acquisition wrote nine files under data/ and was
            # still reported as missing /workspace/data/.
            if str(expected_output).rstrip().endswith(("/", "\\")):
                if self._directory_output_satisfied(expected_output, workspace_files):
                    continue
                missing_outputs.append(expected_output)
                continue
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

    @staticmethod
    def _directory_output_satisfied(expected_output: str, workspace_files: list[str]) -> bool:
        """True when any workspace file sits inside the declared directory.

        Matches on the trailing directory name rather than the full path: plans
        declare workspace-absolute paths ("/workspace/data/") while the scan
        returns paths relative to the workspace root ("data/features.csv").
        """
        name = PurePosixPath(str(expected_output).replace("\\", "/").rstrip("/")).name.lower()
        if not name:
            return False
        return any(
            name in [part.lower() for part in PurePosixPath(actual).parent.parts]
            for actual in workspace_files
        )

    def _can_execute_step(self, step: PlanStep) -> tuple[bool, list[str]]:
        """
        Check if a step can be executed based on its dependencies.
        Args:
            step: The plan step to check
        Returns:
            Tuple[bool, List[str]]: (can_execute, missing_dependencies)
        """
        missing_deps = []

        contract = getattr(self, "_active_contract", None)
        if contract is not None:
            if getattr(self, "current_plan", None) is None:
                raise ContractValidationError("Contract execution has no admitted plan")
            contract.revalidate_plan(self.current_plan)
            input_hashes = contract.validate_inputs(step, self.workspace_path)
            contract_tasks = getattr(self, "_contract_tasks", {})
            for dep_name in step.depends_on:
                dep_task = contract_tasks.get(dep_name)
                if dep_task is None or dep_task.status != TaskStatus.COMPLETED:
                    missing_deps.append(dep_name)
            spec = contract.steps[step.name]
            for artifact_id in spec.inputs:
                if artifact_id in contract.supplied:
                    continue
                producer = contract.producer_by_artifact[artifact_id]
                dep_task = contract_tasks.get(producer)
                if dep_task is None or dep_task.status != TaskStatus.COMPLETED:
                    continue
                expected = dep_task.output_artifact_sha256.get(artifact_id)
                if expected is None:
                    missing_deps.append(
                        f"{producer}[missing_artifact_receipt:{artifact_id}]"
                    )
                elif input_hashes[artifact_id] != expected:
                    missing_deps.append(
                        f"{producer}[artifact_changed:{artifact_id}]"
                    )
            return len(missing_deps) == 0, missing_deps

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
        """Ask whether to continue — but only when someone can answer.

        On a non-TTY this raises :class:`UserInterventionRequired` instead of
        reading stdin. The prompt was the last blocking ``input()`` on the
        benchmark path: on the p_iimn run of 2026-08-22 the planner reached
        step 4 of 6, asked "Continue ? (y/n)" into a redirected stdout, and
        died with ``EOF when reading a line`` — a message that names neither
        the question nor the step it was asked about.

        Raising rather than ``exit(1)`` is deliberate: the CSV harness counts
        the row as failed and still prints its summary, which a ``SystemExit``
        from inside the planner would skip. The same rule is already applied in
        ``pricing.py`` and ``csv_mode._prompt_with_default``.

        Args:
            msg: Message shown both in the Pushover notification body and
                printed to stdout before the prompt.
        """
        self.notifier.send_message(
            f"Mimosa is requesting exit:\n{msg}",
            title="Mimosa exit request."
        )
        print(msg)

        if not sys.stdin.isatty():
            self.logger.error("Intervention needed but stdin is not a TTY: %s", msg)
            raise UserInterventionRequired(
                f"{msg}\n(stdin is not a TTY — cannot ask whether to continue)"
            )

        choice = input("\nContinue ? (y(yes)/n(no))")
        if choice.lower() == "y" or choice.lower() == "yes":
            return
        print("\n---\nExited upon user request.\n---\n")
        exit(1)

    def _record_declared_outputs(self, step: Any, step_task: str) -> None:
        """Persist ``step.expected_outputs`` where the verifier can find them.

        Best-effort: a failure here must never fail the step, it only means the
        verifier scores as it did before this existed.
        """
        try:
            outputs = list(getattr(step, "expected_outputs", None) or [])
            if not outputs:
                return
            temp_root = (getattr(self.config, "temp_dir", None)
                         or Path(getattr(self.config, "workflow_dir", ".")) / "_verifier_tmp")
            if declared_outputs.record(temp_root, step_task, outputs):
                self.logger.info(
                    "Declared outputs recorded for step '%s': %s",
                    getattr(step, "name", "unknown"), ", ".join(outputs)
                )
        except Exception:
            self.logger.exception("Could not record declared outputs for the verifier")

    async def evolve_runs(
        self,
        task: str,
        judge: bool,
        cached_wf_allow: bool = True,
        original_task: str | None = None,
        reuse_workflows: bool = True,
    ) -> list[IndividualRun]:
        """
        Execute Iterative-Learning for a given task.

        Args:
            task: Task description string (may be knowledge-wrapped).
            judge: Whether to use judge evaluation.
            cached_wf_allow: Whether to allow reusing high-quality cached
                workflows discovered by the workflow selector.
            original_task: Original unwrapped task for similarity matching;
                used in preference to ``task`` for cache lookup.
            reuse_workflows: When false, skip both result-cache lookup and
                parent-workflow reuse for this evolution call.

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
            ) if cached_wf_allow and reuse_workflows else []

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

            evolution_kwargs = {
                "goal": task,
                "template_uuid": None,
                "judge": judge,
                "enable_evolution": True,
                "original_task": original_task,
            }
            if not reuse_workflows:
                evolution_kwargs["reuse_workflows"] = False
            runs = await self.evolve.start_workflow_evolution(**evolution_kwargs)

            if runs is None:
                print_warn("Runs is None, returning empty list")
                return []

            return runs

        except Exception as e:
            raise ValueError(f"❌ Planner: Evolution execution failed: {str(e)}") from e

    def _get_evolve_success(self, run: IndividualRun) -> bool:
        """Return the final success flag recorded in ``run.state_result``.

        Args:
            run: An individual evolution run whose ``state_result`` carries a
                ``"success"`` list of booleans.

        Returns:
            The last element of the ``success`` list, or ``False`` when the
            field is missing or malformed.
        """
        run_state_result = getattr(run, 'state_result', None) or {}
        success_list = run_state_result.get('success', [False]) if isinstance(run_state_result, dict) else [False]
        return success_list[-1]

    def _complete_contract_attempt(
        self,
        contract: ArtifactContract,
        step: PlanStep,
        task: Task,
        input_hashes: dict[str, str],
        attempt: int,
        max_attempts: int,
    ) -> bool:
        """Validate one contract attempt and record only fresh exact outputs."""
        try:
            observed_outputs = contract.snapshot_outputs(step, self.workspace_path)
            task.output_artifact_sha256 = {
                artifact_id: digest
                for artifact_id, digest in observed_outputs.items()
                if digest is not None
            }
            if contract.validate_inputs(step, self.workspace_path) != input_hashes:
                raise ArtifactValidationError(
                    "A canonical input changed during workflow execution"
                )
            output_hashes = contract.validate_outputs(step, self.workspace_path)
        except ArtifactValidationError as exc:
            task.status = TaskStatus.FAILED
            step.missing_outputs = list(step.expected_outputs)
            if attempt < max_attempts:
                contract.remove_outputs_for_retry(step, self.workspace_path)
                print_warn(
                    f"Task '{step.name}' violated its artifact contract: {exc} "
                    f"— retrying ({attempt}/{max_attempts})"
                )
            else:
                step.status = TaskStatus.FAILED
                print_err(
                    f"Task '{step.name}' exhausted {max_attempts} attempts with "
                    f"an invalid contract output: {exc}"
                )
            return False
        except ContractValidationError:
            task.status = TaskStatus.FAILED
            step.status = TaskStatus.FAILED
            raise

        task.status = TaskStatus.COMPLETED
        task.output_artifact_sha256 = output_hashes
        task.produced_outputs = list(step.expected_outputs)
        step.missing_outputs = []
        step.status = TaskStatus.COMPLETED
        contract_tasks = getattr(self, "_contract_tasks", None)
        if contract_tasks is None:
            contract_tasks = {}
            self._contract_tasks = contract_tasks
        contract_tasks[step.name] = task
        print_ok(f"Task '{step.name}' completed with validated artifacts")
        return True

    async def run_attempts(
        self,
        attempt_counts: dict[str, int],
        max_attempts: int,
        step: PlanStep,
        judge: bool,
    ) -> PlanStep:
        """
        Execute multiple attempts for a step with comprehensive error handling.
        Args:
            attempt_counts: Dictionary tracking attempt counts per step
            max_attempts: Maximum number of attempts allowed
            step: The plan step to execute
            judge: Whether to use judge evaluation
        Returns:
            PlanStep: The updated step with execution status
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
        task = getattr(step, 'task', '')
        contract = getattr(self, "_active_contract", None)
        if contract is not None:
            if getattr(self, "current_plan", None) is None:
                raise ContractValidationError("Contract execution has no admitted plan")
            can_execute, missing_dependencies = self._can_execute_step(step)
            if not can_execute:
                raise DependencyError(
                    f"Contract step {step_name!r} has unsatisfied dependencies: "
                    f"{missing_dependencies}"
                )
            existing_outputs = contract.snapshot_outputs(step, self.workspace_path)
            preexisting = [
                artifact_id
                for artifact_id, digest in existing_outputs.items()
                if digest is not None
            ]
            if preexisting:
                raise ArtifactValidationError(
                    "Canonical outputs exist before the first attempt and cannot earn "
                    "production credit: " + ", ".join(preexisting)
                )
            step_task = "Your task:" + task
            step_task += "\n---\nCanonical artifact interface (authoritative):\n" + json.dumps(
                contract.execution_projection(step),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        elif getattr(step, "contract_status", "legacy_unchecked") == "validated":
            raise ContractValidationError(
                "Validated step cannot execute without its active contract"
            )
        else:
            goal = getattr(step, 'goal_context', '')
            step_task = f"Broader context:{goal}\n---\nYour task:{task}"
        # Carry the plan's declared outputs to the verifier, which is handed a
        # uuid and would otherwise never see them (issue #196). Keyed on the
        # same task text the verifier keys its rubric cache on, so no signature
        # between here and there has to change.
        self._record_declared_outputs(step, step_task)
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
                input_hashes = (
                    contract.validate_inputs(step, self.workspace_path)
                    if contract is not None
                    else {}
                )
                if contract is not None:
                    remaining_outputs = contract.snapshot_outputs(
                        step, self.workspace_path
                    )
                    if any(digest is not None for digest in remaining_outputs.values()):
                        raise ArtifactValidationError(
                            "Canonical output remains before a retry attempt"
                        )
                enhanced_task = (
                    step_task
                    if contract is not None
                    else self._build_knowledge_aware_task(step_task)
                )
                # Pass both enhanced task and original task for proper workflow matching
                evolve_kwargs = {
                    "cached_wf_allow": attempt <= 1 and contract is None,
                    "original_task": step_task,
                }
                if contract is not None:
                    evolve_kwargs["reuse_workflows"] = False
                evolve_runs = await self.evolve_runs(
                    enhanced_task,
                    judge,
                    **evolve_kwargs,
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
                    status=(
                        TaskStatus.RUNNING
                        if contract is not None and evolve_success
                        else TaskStatus.COMPLETED if evolve_success else TaskStatus.FAILED
                    ),
                    depends_on=getattr(step, 'depends_on', []) or [],
                    required_inputs=getattr(step, 'required_inputs', []) or [],
                    expected_outputs=getattr(step, 'expected_outputs', []) or [],
                    complexity=getattr(step, 'complexity', 'medium'),
                    produced_outputs=(
                        []
                        if contract is not None
                        else [
                            f for f in self._get_workspace_files()
                            if f not in self._workspace_files_before_step
                        ]
                    ),
                    contract_status=getattr(step, "contract_status", "legacy_unchecked"),
                    contract_digest=getattr(step, "contract_digest", None),
                    input_artifact_ids=list(getattr(step, "input_artifact_ids", []) or []),
                    output_artifact_ids=list(getattr(step, "output_artifact_ids", []) or []),
                    supplied_artifact_ids=(
                        [item for item in getattr(step, "input_artifact_ids", []) if item in contract.supplied]
                        if contract is not None else []
                    ),
                    input_artifact_sha256=input_hashes,
                )

                self.task_history.append(task)

                if evolve_success and attempt_score >= 0.7:
                    time.sleep(10) # wait for files update
                    if contract is not None:
                        completed = self._complete_contract_attempt(
                            contract,
                            step,
                            task,
                            input_hashes,
                            attempt,
                            max_attempts,
                        )
                        if completed:
                            attempt = max_attempts
                        continue
                    outputs_produced, missing_outputs = self._verify_expected_outputs(step)
                    if outputs_produced:
                        step.status = TaskStatus.COMPLETED
                        print_ok(f"Task '{step_name}' completed successfully")
                        break
                    # The declared outputs are missing. Both branches used to mark
                    # the step COMPLETED and break, differing only in the log line,
                    # so a step that never produced its deliverable was recorded as
                    # a success and the failure surfaced one layer later at the next
                    # step's dependency gate (issue #196). Spend the remaining
                    # attempts on producing it instead of banking the miss.
                    step.missing_outputs = list(missing_outputs)
                    if attempt < max_attempts:
                        print_warn(
                            f"Task '{step_name}' scored {attempt_score} but did not produce "
                            f"its declared outputs: {missing_outputs} — retrying "
                            f"({attempt}/{max_attempts})"
                        )
                        continue
                    # Out of attempts. Keep COMPLETED so the dependency gate still
                    # reports precisely which output is missing for which step,
                    # rather than replacing that with a generic step failure.
                    step.status = TaskStatus.COMPLETED
                    print_err(
                        f"Task '{step_name}' exhausted {max_attempts} attempts with its "
                        f"declared outputs still missing: {missing_outputs}. Dependent "
                        f"steps cannot run."
                    )
                    break
                else:
                    if contract is not None:
                        task.status = TaskStatus.FAILED
                        observed_outputs = contract.snapshot_outputs(
                            step, self.workspace_path
                        )
                        task.output_artifact_sha256 = {
                            artifact_id: digest
                            for artifact_id, digest in observed_outputs.items()
                            if digest is not None
                        }
                        if attempt < max_attempts:
                            contract.remove_outputs_for_retry(step, self.workspace_path)
                    print_err(f"Task {step_name} (uuid: {final_uuid}) failed with score {attempt_score}")
                    if self.tts:
                        self.tts.speak(f"Task {step_name} failure, retrying...", voice_index=0)
                    continue

            except Exception as e:
                raise e

        if contract is not None and step.status != TaskStatus.COMPLETED:
            step.status = TaskStatus.FAILED
        step.cost = attempt_cost
        step.score = attempt_score
        self._narrate_step_completion(step_name, attempt_score, attempt_cost, final_answers)
        return step

    def _narrate_step_completion(
        self,
        step_name: str,
        attempt_score: float,
        attempt_cost: float,
        final_answers: list[Any],
    ) -> None:
        """Speak a step's outcome, without ever being able to fail the step.

        Two defects met here on a real run and cost it everything it had
        produced.

        ``final_answers`` is annotated ``list[str]`` but agents answer with a
        structured object: every entry of that run's ``state_result.json`` is a
        dict (``{"status": ..., "approach": ...}``). Slicing one raised
        ``TypeError: unhashable type: 'slice'`` on Python 3.11 — and on 3.12+,
        where slices became hashable, the same line degrades to a ``KeyError``
        instead. Every other consumer already coerces first
        (``planner.py`` line ~397, ``evolution_engine.py`` line ~245); this one
        did not.

        And the narration sat inside the step body, so a cosmetic summary
        propagated out as "Critical error in step execution" — reported after
        the step had already written its deliverable and its ASTRA capsule, and
        turning a scored run into a 0% success rate and a non-zero exit. What
        is spoken aloud must never decide whether the work counts.
        """
        if not self.tts:
            return
        try:
            answer = (
                '. '.join([str(x)[:128] for x in final_answers if x])
                if final_answers else "No answers produced."
            )
            tts_text = f"""
            Task completed. Score: {attempt_score}, Cost: {attempt_cost}. {answer}
            """
            self.tts.speak(tts_text, voice_index=0)
        except Exception:
            # Loud, but not fatal: the operator still learns narration broke.
            self.logger.exception("TTS narration failed for step '%s'", step_name)
            print_warn(f"Could not narrate completion of step '{step_name}'")

    async def start_planner(
        self,
        goal: str,
        judge: bool = True,
        max_task_retry: int = 5,
        contract_path: str | Path | None = None,
    ) -> list[Task]:
        """
        Start the planner with a given goal with comprehensive error handling.
        Args:
            goal: The goal description for the planner
            judge: Whether to use a judge for evaluation
            max_task_retry: Maximum number of retries for each task
        Returns:
            List[Task]: List of executed tasks
        Raises:
            ValueError: If goal is invalid or planning fails
        """
        if not goal or not isinstance(goal, str):
            raise ValueError("❌ Planner: Goal must be a non-empty string")

        self._active_contract = self._resolve_artifact_contract(contract_path=contract_path)
        self._contract_tasks = {}
        if self._active_contract is not None:
            self.task_history = []
        if self._active_contract is None:
            goal = "\nAvailable files:\n" + list_files(self.config.workspace_dir) + "\n" + goal
        print_info(f"Starting planner with goal: {goal[:80]}…")

        try:
            # Generate plan with human validation loop
            self.current_plan = self._generate_plan_with_human_validation(
                goal, contract=self._active_contract
            )

            if self.current_plan is None:
                raise ValueError("❌ Planner: Failed to generate a valid plan")
            if self._active_contract is not None:
                self._active_contract.revalidate_plan(self.current_plan)

            # Initialize visualization after plan is approved
            self._init_visualization(self.current_plan)
            # Check for stop condition
            if self._active_contract is None and self._check_stop_condition(self.current_plan):
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
                if self._active_contract is not None or lst_step:
                    can_execute, missing_deps = self._can_execute_step(step)
                    if not can_execute:
                        self.request_user_exit(f"Cannot execute step '{step_name}' — missing dependencies: {missing_deps}")

                # Execute the step with retry logic
                step.status = TaskStatus.RUNNING
                self._update_visualization(total_cost)  # Update to show running status
                max_attempts = max_task_retry

                try:
                    if self._active_contract is None:
                        self._capture_workspace_snapshot()  # Legacy output diff only.
                    else:
                        self._workspace_files_before_step = []
                    step = await self.run_attempts(attempt_counts, max_attempts, step, judge)
                    total_cost += step.cost
                    self._update_visualization(total_cost)  # Update after step completes
                except Exception as e:
                    step.status = TaskStatus.FAILED
                    self._update_visualization(total_cost)  # Update to show failed status
                    # Log the traceback before re-raising. `from e` preserves the
                    # chain for a Python caller, but the operator only ever sees
                    # the formatted message — so a bare TypeError like
                    # "unhashable type: 'slice'" arrives with no file or line and
                    # is effectively unattributable. Observed on a real run that
                    # had already produced its deliverable.
                    self.logger.exception(
                        "Step '%s' (%d/%d) failed", step_name, step_idx + 1,
                        len(self.current_plan.steps),
                    )
                    raise Exception(
                        f"❌ Critical error in step execution: {type(e).__name__}: {e}"
                    ) from e
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

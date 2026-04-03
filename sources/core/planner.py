import json
import time
import os
import re
import sys
import threading
from pathlib import Path
from .dgm import DarwinMachine
from .llm_provider import LLMProvider, LLMConfig, extract_model_pattern
from .schema import Task, Plan, PlanStep, TaskStatus, IndividualRun
from .workflow_selection import WorkflowSelector
from sources.utils.notify import PushNotifier
from sources.utils.planner_visualization import PlannerVisualizer
from sources.utils.list_files import list_files
from sources.extensibility.text_to_speech import create_tts_service

from sources.utils.perspicacite_client import (
    format_scientific_context,
    query_perspicacite,
)


class PlanValidationError(Exception):
    """Exception raised when plan validation fails."""
    pass


class DependencyError(Exception):
    """Exception raised when task dependencies are not satisfied."""
    pass


class Planner:
    """
    Enhanced planner class for long-term task planning with dependency management
    and input/output verification.
    """

    def __init__(self, config, enable_tts=True) -> None:
        if config is None:
            raise ValueError("❌ Planner: Configuration cannot be None")

        self.config = config
        self.workspace_path = config.workspace_dir
        self.dgm = DarwinMachine(config)
        self.task_history: list[Task] = []
        self.current_plan: Plan | None = None
        self.wf_selector = WorkflowSelector(self.config)
        self.notifier = PushNotifier(config.pushover_token, config.pushover_user)
        provider, model = extract_model_pattern(self.config.planner_llm_model)
        self.config_llm = LLMConfig(
            model=model,
            provider=provider,
            reasoning_effort=self.config.reasoning_effort,
            max_tokens=getattr(self.config, 'max_tokens', 8192)
        )
        self._workspace_files_before_step: set[str] = set()  # Track files before step execution
        self.visualizer: PlannerVisualizer | None = None
        self.visualizer_thread: threading.Thread | None = None
        self.use_visualization: bool = True  # Can be disabled if pygame not available
        self.is_macos: bool = sys.platform == "darwin"  # Detect macOS for threading workaround
        self.is_windows: bool = sys.platform == "win32"  # Detect Windows for path handling
        self.tts = create_tts_service() if enable_tts else None

    def make_scientific_grounded_prompt(self, goal: str) -> str:
        """
        Create a scientific-grounded prompt by incorporating relevant scientific knowledge.
        Args:
            goal: The original goal description
        Returns:
            str: Enhanced prompt with scientific context
        """
        scientific_context = query_perspicacite(goal) or "No relevant scientific context found."
        print(f"🔍 Scientific knowledge retrieved:\n{scientific_context[:500]}...\n---")

        return f"""
You are a top-tier scientific in research. When generating the plan, please incorporate relevant scientific principles, theories, or findings that could inform the approach to achieving the goal. This will help ensure that the plan is not only practical but also grounded in scientific understanding.
Scientific context related to the goal:
{scientific_context}
You must generate a plan for goal:\n
{goal}\n
Important: Every task description should be very detailled and specific with the full path of all input output files specified.
"""

    def make_plan(self, system_prompt: str, goal_prompt: str, max_retries: int = 3) -> Plan:
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

        prompt = self.make_scientific_grounded_prompt(goal_prompt)
        for attempt in range(1, max_retries + 1):
            try:
                print(f"🔄 Plan generation attempt {attempt}/{max_retries}")

                memory_path = getattr(self.config, 'memory_path', 'sources/memory')
                raw_plan = LLMProvider("plan_creator", memory_path=memory_path, system_msg=system_prompt, config=self.config_llm, use_flat_cache=True)(prompt, use_cache=True)

                if not raw_plan or not isinstance(raw_plan, str):
                    raise ValueError("LLM returned empty or invalid response")

                print(f"📝 Received plan response ({len(raw_plan)} characters)")
                print(raw_plan)
                print("---")
                plan_dict = self._extract_json_from_code_block(raw_plan)
                if plan_dict is None:
                    raise ValueError("Failed to extract valid JSON from LLM response\n")
                plan = self._parse_and_validate_plan(plan_dict, goal_prompt)
                print(f"✅ Successfully generated and validated plan with {len(plan.steps)} steps")
                return plan

            except (ValueError, PlanValidationError, json.JSONDecodeError) as e:
                last_error = e
                error_msg = str(e)
                # Check if this might be a truncation error (unterminated string at end of response)
                is_truncation = False
                if "Unterminated string" in error_msg or "Unexpected end of data" in error_msg:
                    is_truncation = True
                    error_msg = f"{error_msg} (This often indicates the response was truncated due to max_tokens limit)"
                print(f"⚠️ Attempt {attempt} failed: {error_msg}")
                if is_truncation:
                    print(f"💡 Tip: Consider increasing max_tokens in your config (current: {getattr(self.config, 'max_tokens', 'not set')})")

                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)

                    if attempt > 1:
                        goal_prompt = self._enhance_prompt_with_error(goal_prompt, error_msg)
                else:
                    print(f"❌ All {max_retries} attempts failed")

            except Exception as e:
                last_error = e
                print(f"❌ Unexpected error in attempt {attempt}: {str(e)}")
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

    def _parse_and_validate_plan(self, plan_dict: dict, goal: str) -> Plan:
        """
        Parse and validate a plan dictionary into a Plan object.
        Args:
            plan_dict: Dictionary containing plan data
        Returns:
            Plan: Validated plan object
        Raises:
            PlanValidationError: If plan validation fails
        """
        if "steps" not in plan_dict:
            raise PlanValidationError("❌ Planner: No steps found in the generated plan")

        if not isinstance(plan_dict["steps"], list):
            raise PlanValidationError("❌ Planner: Steps should be a list")

        if not plan_dict["steps"]:
            raise PlanValidationError("❌ Planner: Plan must contain at least one step")

        goal = plan_dict.get("goal", "")
        if not goal:
            raise PlanValidationError("❌ Planner: Plan must have a goal")

        steps = []
        for i, step_dict in enumerate(plan_dict["steps"]):
            try:
                step = PlanStep(
                    name=step_dict.get("name", f"step_{i}"),
                    task=step_dict.get("task", ""),
                    goal_context=goal,
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
    def _extract_json_from_code_block(text: str) -> dict | None:
        """Extract JSON from markdown code blocks (```json ... ```)"""
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
            return json.loads(json_str)
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
        print(f"\n📋 Execution Plan: {plan.goal}")
        print("=" * 80)
        for i, step in enumerate(plan.steps, 1):
            print(f"\n{i}. {step.name.upper()} [{step.complexity}]")
            print(f"   Task: {step.task}")
            if step.depends_on:
                print(f"   Dependencies: {', '.join(step.depends_on)}")
            if step.required_inputs:
                print(f"   Required Inputs: {', '.join(step.required_inputs)}")
            if step.expected_outputs:
                print(f"   Expected Outputs: {', '.join(step.expected_outputs)}")
        print("\n" + "=" * 80)

    def _request_human_plan_validation(self, plan: Plan) -> tuple[bool, str]:
        """
        Request human validation of the generated plan.
        Returns:
            tuple[bool, str]: (is_approved, feedback)
                - is_approved: True if human pressed Enter (approve), False otherwise
                - feedback: User's correction/feedback if plan not approved
        """
        print("\n" + "_" * 40)
        print("👤 HUMAN VALIDATION REQUIRED")
        print("_" * 40)
        print("\nPlease review the plan above.")
        print("\nOptions:")
        print("   • Press [ENTER] to approve and execute the plan")
        print("   • Type your corrections/feedback and press [ENTER] to regenerate")
        print("\n" + "─" * 80)

        user_input = input("\n👉 Your decision: ").strip()
        if not user_input:
            print("\n✅ Plan approved by human. Proceeding with execution...")
            return True, ""
        else:
            print(f"\n📝 Feedback received: {user_input}")
            print("🔄 Will regenerate plan based on your feedback...")
            return False, user_input

    def _generate_plan_with_human_validation(self, goal: str, human_approve = False) -> Plan:
        """
        Generate a plan with iterative human validation and feedback loop.
        Args:
            goal: The goal description for the planner
            max_attempts: Maximum number of plan generation attempts (default: 10)
        Returns:
            Plan: Human-approved plan object
        Raises:
            ValueError: If maximum attempts reached without approval or plan generation fails
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
            print("🎨 Visualization window initialized")

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
                print(f"🎨 Visualization running in separate thread ({platform_name})")
            else:
                print("🎨 Visualization will update from main thread (macOS)")

        except Exception as e:
            print(f"⚠️ Could not initialize visualization: {str(e)}")
            print("⚠️ Continuing without visualization...")
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
                print(f"⚠️ Error updating visualization: {str(e)}")

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
                print("🎨 Visualization window closed")
            except Exception as e:
                print(f"⚠️ Error closing visualization: {str(e)}")


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
                print(f"⚠️ Workspace path does not exist: {self.workspace_path}")
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
            print(f"⚠️ Error scanning workspace files: {str(e)}")

        return files

    def _capture_workspace_snapshot(self) -> None:
        """
        Capture a snapshot of current workspace files before step execution.
        """
        self._workspace_files_before_step = self._get_workspace_files()
        print(f"📸 Captured workspace snapshot: {len(self._workspace_files_before_step)} files")
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
        workspace_files = self._capture_workspace_snapshot()

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
        """
        Check if a step can be executed based on its dependencies.
        Args:
            step: The plan step to check
        Returns:
            Tuple[bool, List[str]]: (can_execute, missing_dependencies)
        """
        missing_deps = []

        for dep_name in step.depends_on:
            dep_task = next((task for task in self.task_history if task.name == dep_name), None)
            if dep_task is None or dep_task.status != TaskStatus.COMPLETED:
                missing_deps.append(dep_name)
        return len(missing_deps) == 0, missing_deps

    def request_user_exit(self, msg: str) -> None:
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

    async def evolve_runs(self, task, judge, max_evolve_iteration, cached_wf_allow=True, original_task=None):
        """
        Execute Iterative-Learning for a given task.
        Args:
            task: Task description string (may be knowledge-wrapped)
            judge: Whether to use judge evaluation
            max_evolve_iteration: Maximum iterations for Evolution
            cached_wf_allow: Whether to allow using cached workflows
            original_task: Original unwrapped task for similarity matching
        Returns:
            List[IndividualRun]: List of Evolution runs
        Raises:
            ValueError: If task is invalid or Evolution execution fails
        """
        if not task or not isinstance(task, str):
            raise ValueError("❌ Planner: Task must be a non-empty string")

        if max_evolve_iteration is None or max_evolve_iteration < 1:
            max_evolve_iteration = 1
            print(f"⚠️ Invalid max_evolve_iteration, using default: {max_evolve_iteration}")

        print(f"🎯 Starting Iterative-Learning for task: {task[:50]}...")

        try:
            # Use original_task for lookup to avoid knowledge wrapper interference
            lookup_task = original_task if original_task else task

            # Check for high-quality cached workflows
            past_wf_lookups = self.wf_selector.select_best_workflows(
                lookup_task, threshold_similary=0.8, threshod_score=0.85
            ) if cached_wf_allow else []

            if past_wf_lookups and len(past_wf_lookups) > 0:
                best_match = past_wf_lookups[0]
                if best_match is None:
                    print("⚠️ Best match is None, proceeding with new Evolution run")
                #elif self._get_evolve_success(best_match):
                elif best_match.is_success:
                    print(f"🔁 Using previously run workflow result with UUID: {getattr(best_match, 'uuid', 'N/A')}")

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
            print(f"🔄 No cached run found, starting task learning (max_iter: {max_evolve_iteration})")

            if self.dgm is None:
                raise ValueError("❌ Planner: instance is None")

            runs = await self.dgm.start_dgm(
                goal=task,
                template_uuid=None,
                judge=judge,
                learning_mode=True,
                max_iteration=max_evolve_iteration,
                original_task=original_task
            )

            if runs is None:
                print("⚠️ Runs is None, returning empty list")
                return []

            return runs

        except Exception as e:
            raise ValueError(f"❌ Planner: Evolution execution failed: {str(e)}") from e

    def _get_evolve_success(self, run: IndividualRun) -> bool:
        run_state_result = getattr(run, 'state_result', None) or {}
        success_list = run_state_result.get('success', [False]) if isinstance(run_state_result, dict) else [False]
        return success_list[-1]

    async def run_attempts(self, attempt_counts, max_attempts, step, judge, max_evolve_iteration):
        """
        Execute multiple attempts for a step with comprehensive error handling.
        Args:
            attempt_counts: Dictionary tracking attempt counts per step
            max_attempts: Maximum number of attempts allowed
            step: The plan step to execute
            judge: Whether to use judge evaluation
            max_evolve_iteration: Maximum learning iterations
        Returns:
            PlanStep: The updated step with execution status
        """
        if step is None:
            print("❌ Step is None, cannot execute")
            return step

        if attempt_counts is None:
            attempt_counts = {}

        if max_attempts is None or max_attempts < 1:
            max_attempts = 1
            print(f"⚠️ Invalid max_attempts, using default: {max_attempts}")

        step_name = getattr(step, 'name', 'unknown_step')
        goal = getattr(step, 'goal_context', '')
        task = getattr(step, 'task', '')
        step_task = f"Broader context:{goal}\n---\nYour task:{task}"
        attempt = attempt_counts.get(step_name, 0)
        attempt_cost = 0
        attempt_score = 0.0
        while attempt <= max_attempts:
            attempt += 1
            attempt_counts[step_name] = attempt

            print(f"🔄 Attempt {attempt}/{max_attempts} for task: {step_name}")
            if self.tts:
                self.tts.speak(f"now starting task {step_name}", voice_index=0)

            try:
                enhanced_task = self._build_knowledge_aware_task(step_task)
                # Pass both enhanced task and original task for proper workflow matching
                evolve_runs = await self.evolve_runs(
                    enhanced_task,
                    judge,
                    max_evolve_iteration,
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
                    produced_outputs=self._workspace_files_before_step
                )

                self.task_history.append(task)

                if evolve_success and attempt_score >= 0.7:
                    time.sleep(10) # wait for files update
                    outputs_produced, missing_outputs = self._verify_expected_outputs(step)
                    step.status = TaskStatus.COMPLETED
                    if outputs_produced:
                        print(f"✅ Task '{step_name}' completed successfully")
                        break
                    else:
                        print(f"⚠️ Task '{step_name}' completed but missing expected outputs: {missing_outputs}")
                        break
                else:
                    print(f"❌ Task {step_name} (uuid: {final_uuid}) failed with score {attempt_score}\n")
                    if self.tts:
                        self.tts.speak(f"Task {step_name} failure, retrying...", voice_index=0)
                    continue

            except Exception as e:
                raise e

        step.cost = attempt_cost
        step.score = attempt_score
        if self.tts:
            answer = '. '.join([x[:128] for x in final_answers])
            tts_text = f"""
            Task completed. Score: {attempt_score}, Cost: {attempt_cost}. {answer}
            """
            self.tts.speak(tts_text, voice_index=0)
        return step

    async def start_planner(
        self,
        goal: str,
        judge: bool = True,
        max_evolve_iteration: int = 1,
        max_task_retry: int = 5
    ) -> list[Task]:
        """
        Start the planner with a given goal with comprehensive error handling.
        Args:
            goal: The goal description for the planner
            judge: Whether to use a judge for evaluation
            max_evolve_iteration: Maximum number of Evolution improvement attempts per task
            max_task_retry: Maximum number of retries for each task
        Returns:
            List[Task]: List of executed tasks
        Raises:
            ValueError: If goal is invalid or planning fails
        """
        if not goal or not isinstance(goal, str):
            raise ValueError("❌ Planner: Goal must be a non-empty string")

        goal = "\nAvailable files:\n" + list_files(self.config.workspace_dir) + "\n" + goal
        print(f"▶ Starting planner with goal: {goal}")

        try:
            # Generate plan with human validation loop
            self.current_plan = self._generate_plan_with_human_validation(goal)

            if self.current_plan is None:
                raise ValueError("❌ Planner: Failed to generate a valid plan")

            # Initialize visualization after plan is approved
            self._init_visualization(self.current_plan)
            # Check for stop condition
            if self._check_stop_condition(self.current_plan):
                print("⏹️ Stop condition found in plan. Exiting.")
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
                    print(f"⚠️ Step {step_idx + 1} is None, skipping")
                    continue
                step_name = getattr(step, 'name', f'step_{step_idx}')
                print(f"\n{'='*60}")
                print(f"📋 Executing Step {step_idx + 1}/{len(self.current_plan.steps)}: {step_name}")
                print(f"{'='*60}")
                # Check if step can be executed (dependencies satisfied)
                if lst_step:
                    can_execute, missing_deps = self._can_execute_step(lst_step)
                    if not can_execute:
                        self.request_user_exit(f"⚠️ Cannot execute step '{step_name}' - missing dependencies: {missing_deps}")

                # Execute the step with retry logic
                step.status = TaskStatus.RUNNING
                self._update_visualization(total_cost)  # Update to show running status
                max_attempts = max_task_retry

                try:
                    step = await self.run_attempts(attempt_counts, max_attempts, step, judge, max_evolve_iteration)
                    total_cost += step.cost
                    self._update_visualization(total_cost)  # Update after step completes
                except Exception as e:
                    step.status = TaskStatus.FAILED
                    self._update_visualization(total_cost)  # Update to show failed status
                    raise Exception(f"❌ Critical error in step execution: {str(e)}") from e
                lst_step = step

                if step.status != TaskStatus.COMPLETED:
                    step.status = TaskStatus.FAILED
                    # Send notification for task failure
                    self.notifier.send_message(
                        f"Task '{step_name}' failed after {max_attempts} attempts\n"
                        f"Goal: {goal[:128]}...\n"
                        f"Step: {step_idx + 1}/{len(self.current_plan.steps)}",
                        title=f"Task '{step_name}' failed",
                        priority=1
                    )
                    raise Exception(f"❌ Giving up on task '{step_name}' after {max_attempts} attempts")

            print(f"\n🏁 Planner execution completed. Executed {len(self.task_history)} tasks. Cost: {total_cost}")

            # Send success notification
            completed_tasks = sum(1 for t in self.task_history if t.status == TaskStatus.COMPLETED)
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
            print(f"❌ Critical error in planner execution: {str(e)}")
            self.notifier.send_message(str(e), title="error during Mimosa execution.")
            self._cleanup_visualization()
            raise ValueError(f"❌ Planner: Execution failed: {str(e)}") from e

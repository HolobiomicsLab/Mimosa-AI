import json
import os
import re
import time
from pathlib import Path
from .dgm import GodelMachine
from .llm_provider import LLMProvider, LLMConfig
from .schema import Task, Plan, PlanStep, TaskStatus, GodelRun
from .workflow_selection import WorkflowSelector
from sources.utils.notify import PushNotifier
from sources.utils.transfer_toolomics import Transfer


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

    def __init__(self, config) -> None:
        if config is None:
            raise ValueError("❌ Planner: Configuration cannot be None")
        
        self.config = config
        self.workspace_path = config.workspace_dir
        self.dgm = GodelMachine(config)
        self.task_history: list[Task] = []
        self.current_plan: Plan | None = None
        self.wf_selector = WorkflowSelector(self.config)
        self.notifier = PushNotifier(config.pushover_token, config.pushover_user)
        self.config_llm = LLMConfig.from_dict({"model": "claude-3-7-sonnet-latest", "provider": "anthropic"})
        self._workspace_files_before_step: set[str] = set()  # Track files before step execution

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
        
        for attempt in range(1, max_retries + 1):
            try:
                print(f"🔄 Plan generation attempt {attempt}/{max_retries}")
                
                memory_path = getattr(self.config, 'memory_path', 'sources/memory')
                raw_plan = LLMProvider("plan_creator", memory_path=memory_path, system_msg=system_prompt, config=self.config_llm)(goal_prompt)
                
                if not raw_plan or not isinstance(raw_plan, str):
                    raise ValueError("LLM returned empty or invalid response")
                
                print(f"📝 Received plan response ({len(raw_plan)} characters)")
                plan_dict = self._extract_json_from_code_block(raw_plan)
                if plan_dict is None:
                    raise ValueError("Failed to extract valid JSON from LLM response")
                plan = self._parse_and_validate_plan(plan_dict)
                print(f"✅ Successfully generated and validated plan with {len(plan.steps)} steps")
                return plan
                
            except (ValueError, PlanValidationError, json.JSONDecodeError) as e:
                last_error = e
                error_msg = str(e)
                print(f"⚠️ Attempt {attempt} failed: {error_msg}")
                
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

    def _parse_and_validate_plan(self, plan_dict: dict) -> Plan:
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
IMPORTANT: Previous attempt failed with error: {error_msg}

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


    def _get_workspace_files(self) -> list[str]:
        """
        Get all files in the workspace directory recursively.
        Returns:
            set[str]: Set of relative file paths from workspace root
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
                            files.append(str(relative_path))
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
        
        for required_input in step.required_inputs:
            if required_input not in self._get_workspace_files():
                missing_inputs.append(required_input)
        return len(missing_inputs) == 0, missing_inputs

    def _verify_expected_outputs(self, step: PlanStep) -> tuple[bool, list[str]]:
        """
        Verify that expected outputs were produced.
        Args:
            step: The plan step
            produced_outputs: List of actually produced outputs
        Returns:
            Tuple[bool, List[str]]: (all_produced, missing_outputs)
        """
        missing_outputs = []
        
        workspace_files = self._capture_workspace_snapshot()
        for expected_output in step.expected_outputs:
            found = any(expected_output in produced for produced in workspace_files)
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
        print(msg)
        choice = input("\nContinue ? (y(yes)/n(no))")
        if choice.lower() == "y" or choice.lower() == "yes":
            return
        print("\n---\nExited upon user request.\n---\n")
        exit(1)
    

    async def dgm_runs(self, task, judge, max_dgm_iteration, cached_wf_allow=True):
        """
        Execute DGM runs for a given task with comprehensive error handling.
        Args:
            task: Task description string
            judge: Whether to use judge evaluation
            max_dgm_iteration: Maximum iterations for DGM
        Returns:
            List[GodelRun]: List of DGM runs
        Raises:
            ValueError: If task is invalid or DGM execution fails
        """
        if not task or not isinstance(task, str):
            raise ValueError("❌ Planner: Task must be a non-empty string")
        
        if max_dgm_iteration is None or max_dgm_iteration < 1:
            max_dgm_iteration = 2
            print(f"⚠️ Invalid max_dgm_iteration, using default: {max_dgm_iteration}")
        
        print(f"🎯 Starting DGM runs for task: {task[:50]}...")
        
        try:
            # Check for high-quality cached workflows
            past_wf_lookups = self.wf_selector.select_best_workflows(
                task, threshold_similary=0.8, threshod_score=0.0 # TODO change values
            ) if cached_wf_allow else []
            
            if past_wf_lookups and len(past_wf_lookups) > 0:
                best_match = past_wf_lookups[0]
                if best_match is None:
                    print("⚠️ Best match is None, proceeding with new DGM run")
                else:
                    print(f"🔁 Using previously run workflow result with UUID: {getattr(best_match, 'uuid', 'N/A')}")
                    
                    run = GodelRun(
                        goal=best_match.goal,
                        prompt=best_match.goal,
                        answers=best_match.answers,
                        state_result=best_match.state_result,
                        current_uuid=best_match.uuid,
                        reward=best_match.overall_score,
                        workflow_template=best_match.code
                    )
                    return [run]
            
            # Generate new workflows via DGM
            print(f"🔄 No cached run found, starting DGM task learning (max_iter: {max_dgm_iteration})")
            
            if self.dgm is None:
                raise ValueError("❌ Planner: DGM instance is None")
            
            runs = await self.dgm.start_dgm(
                goal=task,
                template_uuid=None,
                judge=judge,
                human_validation=False,
                max_iteration=max_dgm_iteration
            )
            
            if runs is None:
                print("⚠️ DGM returned None, returning empty list")
                return []

            return runs
            
        except Exception as e:
            print(f"❌ Error in dgm_runs: {str(e)}")
            raise ValueError(f"❌ Planner: DGM execution failed: {str(e)}") from e

    def _get_dgm_success(self, run: GodelRun) -> bool:
        run_state_result = getattr(run, 'state_result', None) or {}
        success_list = run_state_result.get('success', [False]) if isinstance(run_state_result, dict) else [False]
        return success_list[-1]

    async def run_attempts(self, attempt_counts, max_attempts, step, judge, max_dgm_iteration):
        """
        Execute multiple attempts for a step with comprehensive error handling.
        Args:
            attempt_counts: Dictionary tracking attempt counts per step
            max_attempts: Maximum number of attempts allowed
            step: The plan step to execute
            judge: Whether to use judge evaluation
            max_dgm_iteration: Maximum DGM iterations
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
        step_task = getattr(step, 'task', '')
        
        attempt = attempt_counts.get(step_name, 0)
        while attempt < max_attempts:
            attempt += 1
            attempt_counts[step_name] = attempt
            
            print(f"🔄 Attempt {attempt}/{max_attempts} for task: {step_name}")
            
            try:
                enhanced_task = self._build_knowledge_aware_task(step_task)
                dgm_runs = await self.dgm_runs(enhanced_task, judge, max_dgm_iteration, cached_wf_allow=(attempt<=1))
                
                last_run = dgm_runs[-1]
                
                final_answers = []
                final_uuid = None
                workflow_uuid = None
                
                if last_run is not None:
                    final_answers = getattr(last_run, 'answers', []) or []
                    final_uuid = getattr(last_run, 'current_uuid', None)
                    workflow_uuid = getattr(last_run, 'workflow_template', None)
                
                dgm_success = self._get_dgm_success(last_run)
                task = Task(
                    name=step_name,
                    description=step_task,
                    dgm_runs=dgm_runs or [],
                    final_answers=final_answers,
                    final_uuid=final_uuid,
                    workflow_uuid=workflow_uuid,
                    status=TaskStatus.COMPLETED if dgm_success else TaskStatus.FAILED,
                    depends_on=getattr(step, 'depends_on', []) or [],
                    required_inputs=getattr(step, 'required_inputs', []) or [],
                    expected_outputs=getattr(step, 'expected_outputs', []) or [],
                    complexity=getattr(step, 'complexity', 'medium'),
                    produced_outputs=self._workspace_files_before_step
                )
                
                self.task_history.append(task)
                
                if dgm_success:
                    outputs_produced, missing_outputs = self._verify_expected_outputs(step)
                    if outputs_produced:
                        print(f"✅ Task '{step_name}' completed successfully")
                    else:
                        print(f"⚠️ Task '{step_name}' completed but missing expected outputs: {missing_outputs}")
                    break
                else:
                    print(f"❌ Task '{step_name}' failed (attempt {attempt}/{max_attempts})")
                    if attempt < max_attempts:
                        print("🔄 Retrying...")
                    
            except Exception as e:
                raise e
        
        step.status = TaskStatus.COMPLETED
        return step

    async def start_planner(
        self,
        goal: str,
        judge: bool = True,
        max_dgm_iteration: int = 2,
        max_task_retry: int = 5
    ) -> list[Task]:
        """
        Start the planner with a given goal with comprehensive error handling.
        Args:
            goal: The goal description for the planner
            judge: Whether to use a judge for evaluation
            max_dgm_iteration: Maximum number of DGM improvement attempts per task
            max_task_retry: Maximum number of retries for each task
        Returns:
            List[Task]: List of executed tasks
        Raises:
            ValueError: If goal is invalid or planning fails
        """
        if not goal or not isinstance(goal, str):
            raise ValueError("❌ Planner: Goal must be a non-empty string")
        
        print(f"🚀 Starting planner with goal: {goal}")
        
        try:
            # Generate and validate plan
            system_prompt = self._read_prompt()
            self.current_plan = self.make_plan(system_prompt, goal)
            
            if self.current_plan is None:
                raise ValueError("❌ Planner: Failed to generate a valid plan")
            
            self._display_plan(self.current_plan)
            
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
                        continue
                
                # Execute the step with retry logic
                step.status = TaskStatus.RUNNING
                max_attempts = max_task_retry
                
                try:
                    step = await self.run_attempts(attempt_counts, max_attempts, step, judge, max_dgm_iteration)
                except Exception as e:
                    step.status = TaskStatus.FAILED
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
            
            print(f"\n🏁 Planner execution completed. Executed {len(self.task_history)} tasks.")
            
            # Send success notification
            completed_tasks = sum(1 for t in self.task_history if t.status == TaskStatus.COMPLETED)
            self.notifier.send_message(
                f"Planner completed successfully!\n"
                f"Goal: {goal[:128]}...\n"
                f"Completed: {completed_tasks}/{len(self.task_history)} tasks",
                title="Planner execution completed",
                priority=0
            )
            trs = Transfer(workspace_path=self.config.workspace_dir, runs_capsule_dir=self.config.runs_capsule_dir)
            trs.transfer_workspace_files_to_capsule(goal)
            return self.task_history
            
        except Exception as e:
            print(f"❌ Critical error in planner execution: {str(e)}")
            self.notifier.send_message(str(e), title="error during Mimosa execution.")
            raise ValueError(f"❌ Planner: Execution failed: {str(e)}") from e

import json
from .dgm import GodelMachine
from .llm_provider import LLMProvider
from .schema import Task, Plan, PlanStep, TaskStatus, GodelRun
from .workflow_selection import WorkflowSelector
from .workflow_info import WorkflowInfo
from sources.utils.notify import PushNotifier


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
        self.dgm = GodelMachine(config)
        self.task_history: list[Task] = []
        self.current_plan: Plan | None = None
        self.available_outputs: set[str] = set()  # Track all available outputs
        self.wf_selector = WorkflowSelector(self.config)
        self.notifier = PushNotifier(config.pushover_token, config.pushover_user)

    def make_plan(self, system_prompt: str, goal_prompt: str) -> Plan:
        """
        Generate a workflow plan using the LLM.
        Args:
            system_prompt: The system prompt for plan generation
            goal_prompt: The goal description
        Returns:
            Plan: Validated plan object
        Raises:
            ValueError: If plan generation or validation fails
        """
        if not system_prompt or not isinstance(system_prompt, str):
            raise ValueError("❌ Planner: system_prompt must be a non-empty string")
        if not goal_prompt or not isinstance(goal_prompt, str):
            raise ValueError("❌ Planner: goal_prompt must be a non-empty string")
        
        try:
            memory_path = getattr(self.config, 'memory_path', 'sources/memory')
            raw_plan = LLMProvider("plan_creator", memory_path=memory_path, system_msg=system_prompt)(goal_prompt)
        except Exception as e:
            raise ValueError(f"❌ Planner: Failed to generate plan from LLM: {str(e)}") from e
        
        if not raw_plan or not isinstance(raw_plan, str):
            raise ValueError("❌ Planner: LLM returned empty or invalid response")
        
        extracted_json = self._extract_json_code(raw_plan)
        if not extracted_json:
            print(f"Raw plan output:\n{raw_plan}")
            raise ValueError("❌ Planner: Failed to generate a plan from the LLM.")
        
        try:
            plan_dict = json.loads(extracted_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Planner: Invalid JSON: {str(e)}") from e
        
        return self._parse_and_validate_plan(plan_dict)

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
    def _extract_json_code(code: str) -> str:
        """
        Extract JSON code blocks from text.
        Args:
            code: Text potentially containing JSON code blocks
        Returns:
            str: Extracted JSON code
        """
        code_blocks = []
        in_code_block = False
        
        for line in code.splitlines():
            if line.strip().startswith("```json"):
                in_code_block = True
                continue
            if line.strip().startswith("```") and in_code_block:
                in_code_block = False
                continue
            if in_code_block:
                code_blocks.append(line)
        
        return "\n".join(code_blocks)

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
        """
        Check if the plan contains a 'stop' step.
        
        Args:
            plan: The plan to check
            
        Returns:
            bool: True if stop condition is found
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

    def _get_dgm_success(self, dgm_run: GodelRun | None) -> bool:
        """
        Check if a DGM run was successful.
        Args:
            dgm_run: The DGM run to check
        Returns:
            bool: True if the run was successful
        """
        if dgm_run is None:
            return False
        
        state_result = dgm_run.state_result or {}
        success_list = state_result.get('success', [False])
        return success_list[-1] if success_list else False

    def _extract_produced_outputs(self, dgm_runs: list[GodelRun]) -> list[str]:
        """
        Extract produced outputs from DGM run answers with comprehensive safety checks.
        Args:
            dgm_runs: List of DGM runs
        Returns:
            List[str]: List of produced output filenames
        """
        if not dgm_runs or not isinstance(dgm_runs, list):
            return []
        
        last_run = None
        for run in reversed(dgm_runs):
            if run is not None:
                last_run = run
                break
        
        if last_run is None:
            return []
        
        answers = getattr(last_run, 'answers', None)
        if not answers or not isinstance(answers, list):
            return []
        
        last_answer = ""
        for answer in reversed(answers):
            if answer is not None and isinstance(answer, str):
                last_answer = answer
                break
        
        if not last_answer:
            return []
        
        produced_outputs = []
        try:
            import re
            # Pattern for files with extensions
            file_patterns = [
                r'\b[\w\-_]+\.[a-zA-Z0-9]{1,5}\b',  # filename.ext
                r'\b[\w\-_]+/[\w\-_\.]+\b',         # directory/filename
                r'\.\/[\w\-_\/\.]+\b',              # ./path/filename
            ]
            for pattern in file_patterns:
                try:
                    matches = re.findall(pattern, last_answer)
                    if matches:
                        produced_outputs.extend(matches)
                except Exception as e:
                    print(f"⚠️ Error in regex pattern {pattern}: {str(e)}")
                    continue
        except Exception as e:
            print(f"⚠️ Error importing re module or processing patterns: {str(e)}")
            return []
        
        seen = set()
        unique_outputs = []
        for output in produced_outputs:
            if output and isinstance(output, str) and output not in seen:
                seen.add(output)
                unique_outputs.append(output)
        
        return unique_outputs

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
            if required_input not in self.available_outputs:
                missing_inputs.append(required_input)
        
        return len(missing_inputs) == 0, missing_inputs

    def _verify_expected_outputs(self, step: PlanStep, produced_outputs: list[str]) -> tuple[bool, list[str]]:
        """
        Verify that expected outputs were produced.
        Args:
            step: The plan step
            produced_outputs: List of actually produced outputs
        Returns:
            Tuple[bool, List[str]]: (all_produced, missing_outputs)
        """
        missing_outputs = []
        
        for expected_output in step.expected_outputs:
            found = any(expected_output in produced for produced in produced_outputs)
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
            # Find the dependency task in history
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
    

    async def dgm_runs(self, task, judge, max_dgm_iteration):
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
                task, threshold_similary=0.9, threshod_score=0.9 # TODO change values
            )
            
            if past_wf_lookups and len(past_wf_lookups) > 0:
                best_match = past_wf_lookups[0]
                if best_match is None:
                    print("⚠️ Best match is None, proceeding with new DGM run")
                else:
                    print(f"Using previously run workflow result with UUID: {getattr(best_match, 'uuid', 'N/A')}")
                    
                    state_result = getattr(best_match, 'state_result', None) or {}
                    run = GodelRun(
                        goal=getattr(best_match, 'goal', ''),
                        prompt=getattr(best_match, 'goal', ''),
                        answers=state_result.get('answers', []) if isinstance(state_result, dict) else [],
                        state_result=state_result,
                        current_uuid=getattr(best_match, 'uuid', None),
                        reward=getattr(best_match, 'overall_score', 0.0),
                        workflow_template=getattr(best_match, 'code', None)
                    )
                    
                    run_state_result = getattr(run, 'state_result', None) or {}
                    success_list = run_state_result.get('success', [False]) if isinstance(run_state_result, dict) else [False]
                    print(f"Run: success={success_list}\n")
                    return [run]
            
            # Generate new workflows via DGM
            print(f"🔄 No cached run found, starting DGM task learning (max_iter: {max_dgm_iteration})")
            
            if self.dgm is None:
                raise ValueError("❌ Planner: DGM instance is None")
            
            runs = await self.dgm.start_dgm(
                goal=task,
                template_uuid=None,
                judge=judge,
                answer=None,
                human_validation=False,
                max_iteration=max_dgm_iteration
            )
            
            if runs is None:
                print("⚠️ DGM returned None, returning empty list")
                return []
            
            if not isinstance(runs, list):
                print(f"⚠️ DGM returned non-list result: {type(runs)}, converting to list")
                runs = [runs] if runs else []
            
            print(f"DGM completed with {len(runs)} runs.")
            for i, run in enumerate(runs):
                if run is None:
                    print(f"   Run {i+1}: None run detected")
                    continue
                
                try:
                    run_state_result = getattr(run, 'state_result', None) or {}
                    success_list = run_state_result.get('success', [False]) if isinstance(run_state_result, dict) else [False]
                    print(f"   Run {i+1}: success={success_list}\n")
                except Exception as e:
                    print(str(e))
                    success_list = [True]
            return runs
            
        except Exception as e:
            print(f"❌ Error in dgm_runs: {str(e)}")
            raise ValueError(f"❌ Planner: DGM execution failed: {str(e)}") from e

    async def run_attempts(self, attempt_counts, max_attempts, step, judge, max_dgm_iteration, missing_inputs):
        """
        Execute multiple attempts for a step with comprehensive error handling.
        Args:
            attempt_counts: Dictionary tracking attempt counts per step
            max_attempts: Maximum number of attempts allowed
            step: The plan step to execute
            judge: Whether to use judge evaluation
            max_dgm_iteration: Maximum DGM iterations
            missing_inputs: List of missing inputs for this step
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
                dgm_runs = await self.dgm_runs(enhanced_task, judge, max_dgm_iteration)
                
                last_run = None
                if dgm_runs and isinstance(dgm_runs, list):
                    for run in reversed(dgm_runs):
                        if run is not None:
                            last_run = run
                            break
                
                dgm_success = self._get_dgm_success(last_run)
                produced_outputs = self._extract_produced_outputs(dgm_runs)
                
                # Safely extract data from last run
                final_answers = []
                final_uuid = None
                workflow_uuid = None
                
                if last_run is not None:
                    final_answers = getattr(last_run, 'answers', []) or []
                    final_uuid = getattr(last_run, 'current_uuid', None)
                    workflow_uuid = getattr(last_run, 'workflow_template', None)
                
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
                    produced_outputs=produced_outputs,
                    missing_inputs=missing_inputs or []
                )
                
                self.task_history.append(task)
                
                if dgm_success:
                    outputs_produced, missing_outputs = self._verify_expected_outputs(step, produced_outputs)
                    if outputs_produced:
                        print(f"✅ Task '{step_name}' completed successfully")
                        print(f"📄 Produced outputs: {produced_outputs}")
                    else:
                        print(f"⚠️ Task '{step_name}' completed but missing expected outputs: {missing_outputs}")
                        print(f"📄 Actual outputs: {produced_outputs}")
                    
                    # Safely update available outputs
                    if produced_outputs and isinstance(produced_outputs, list):
                        self.available_outputs.update(produced_outputs)
                    
                    step.status = TaskStatus.COMPLETED
                    break
                else:
                    print(f"❌ Task '{step_name}' failed (attempt {attempt}/{max_attempts})")
                    if attempt < max_attempts:
                        print("🔄 Retrying...")
                    
            except Exception as e:
                error_msg = f"Error executing task '{step_name}': {str(e)}, Retry task ? (already tried {attempt} time)"
                self.request_user_exit(error_msg)
                if attempt < max_attempts:
                    print("🔄 Retrying...")
        
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
            
            for step_idx, step in enumerate(self.current_plan.steps):
                if step is None:
                    print(f"⚠️ Step {step_idx + 1} is None, skipping")
                    continue
                
                step_name = getattr(step, 'name', f'step_{step_idx}')
                
                print(f"\n{'='*60}")
                print(f"📋 Executing Step {step_idx + 1}/{len(self.current_plan.steps)}: {step_name}")
                print(f"{'='*60}")
                
                # Check if step can be executed (dependencies satisfied)
                can_execute, missing_deps = self._can_execute_step(step)
                if not can_execute:
                    step.status = TaskStatus.SKIPPED
                    error_msg = f"⚠️ Cannot execute step '{step_name}' - missing dependencies: {missing_deps}"
                    self.request_user_exit(error_msg)
                    continue
                
                # Check required inputs
                inputs_available, missing_inputs = self._verify_required_inputs(step)
                if not inputs_available and step_idx > 0:
                    error_msg = f"⚠️ Missing required inputs for step '{step_name}': {missing_inputs}"
                    self.request_user_exit(error_msg)
                
                # Execute the step with retry logic
                step.status = TaskStatus.RUNNING
                max_attempts = max_task_retry
                
                try:
                    step = await self.run_attempts(attempt_counts, max_attempts, step, judge, max_dgm_iteration, missing_inputs)
                except Exception as e:
                    raise Exception(f"❌ Critical error in step execution: {str(e)}") from e
                    step.status = TaskStatus.FAILED
                
                if step.status != TaskStatus.COMPLETED:
                    step.status = TaskStatus.FAILED
                    raise Exception(f"❌ Giving up on task '{step_name}' after {max_attempts} attempts")
            
            print(f"\n🏁 Planner execution completed. Executed {len(self.task_history)} tasks.")
            return self.task_history
            
        except Exception as e:
            print(f"❌ Critical error in planner execution: {str(e)}")
            self.notifier.send_message(str(e), title="error during Mimosa execution.")
            raise ValueError(f"❌ Planner: Execution failed: {str(e)}") from e

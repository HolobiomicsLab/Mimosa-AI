import json
import os
from typing import List, Dict, Set, Optional, Tuple
from .dgm import GodelMachine
from .llm_provider import LLMProvider
from .schema import Task, Plan, PlanStep, TaskStatus, GodelRun


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
        self.config = config
        self.dgm = GodelMachine(config)
        self.task_history: List[Task] = []
        self.current_plan: Optional[Plan] = None
        self.available_outputs: Set[str] = set()  # Track all available outputs

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
        raw_plan = LLMProvider("plan_creator", system_msg=system_prompt)(goal_prompt)
        
        extracted_json = self._extract_json_code(raw_plan)
        if not extracted_json:
            print(f"Raw plan output:\n{raw_plan}")
            raise ValueError("❌ Planner: Failed to generate a plan from the LLM.")
        
        try:
            plan_dict = json.loads(extracted_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Planner: Invalid JSON: {str(e)}") from e
        
        return self._parse_and_validate_plan(plan_dict)

    def _parse_and_validate_plan(self, plan_dict: Dict) -> Plan:
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
            with open(self.config.prompt_planner, 'r', encoding='utf-8') as f:
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
                answers_text = '\n\t - '.join(task.final_answers)
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

    def _get_dgm_success(self, dgm_run: Optional[GodelRun]) -> bool:
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

    def _extract_produced_outputs(self, dgm_runs: List[GodelRun]) -> List[str]:
        """
        Extract produced outputs from DGM run answers.
        
        Args:
            dgm_runs: List of DGM runs
            
        Returns:
            List[str]: List of produced output filenames
        """
        if not dgm_runs or not dgm_runs[-1].answers:
            return []
        
        # Get the last answer from the last run
        last_answer = dgm_runs[-1].answers[-1] if dgm_runs[-1].answers else ""
        
        # Extract potential filenames (simple heuristic)
        produced_outputs = []
        
        # Look for common file patterns in the answer
        import re
        
        # Pattern for files with extensions
        file_patterns = [
            r'\b[\w\-_]+\.[a-zA-Z0-9]{1,5}\b',  # filename.ext
            r'\b[\w\-_]+/[\w\-_\.]+\b',         # directory/filename
            r'\.\/[\w\-_\/\.]+\b',              # ./path/filename
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, last_answer)
            produced_outputs.extend(matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_outputs = []
        for output in produced_outputs:
            if output not in seen:
                seen.add(output)
                unique_outputs.append(output)
        
        return unique_outputs

    def _verify_required_inputs(self, step: PlanStep) -> Tuple[bool, List[str]]:
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

    def _verify_expected_outputs(self, step: PlanStep, produced_outputs: List[str]) -> Tuple[bool, List[str]]:
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
            # Check if the expected output is mentioned in produced outputs
            found = any(expected_output in produced for produced in produced_outputs)
            if not found:
                missing_outputs.append(expected_output)
        
        return len(missing_outputs) == 0, missing_outputs

    def _can_execute_step(self, step: PlanStep) -> Tuple[bool, List[str]]:
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

    async def start_planner(
        self,
        goal: str,
        judge: bool = True,
        max_dgm_iteration: int = 2,
        max_task_retry: int = 2
    ) -> List[Task]:
        """
        Start the planner with a given goal.
        
        Args:
            goal: The goal description for the planner
            judge: Whether to use a judge for evaluation
            max_dgm_iteration: Maximum number of DGM improvement attempts per task
            max_task_retry: Maximum number of retries for each task
            
        Returns:
            List[Task]: List of executed tasks
        """
        print(f"🚀 Starting planner with goal: {goal}")
        
        # Generate and validate plan
        system_prompt = self._read_prompt()
        self.current_plan = self.make_plan(system_prompt, goal)
        self._display_plan(self.current_plan)
        
        # Check for stop condition
        if self._check_stop_condition(self.current_plan):
            print("⏹️ Stop condition found in plan. Exiting.")
            return self.task_history
        
        # Execute plan steps
        attempt_counts = {}
        
        for step_idx, step in enumerate(self.current_plan.steps):
            print(f"\n{'='*60}")
            print(f"📋 Executing Step {step_idx + 1}/{len(self.current_plan.steps)}: {step.name}")
            print(f"{'='*60}")
            
            # Check if step can be executed (dependencies satisfied)
            can_execute, missing_deps = self._can_execute_step(step)
            if not can_execute:
                print(f"⚠️ Cannot execute step '{step.name}' - missing dependencies: {missing_deps}")
                step.status = TaskStatus.SKIPPED
                continue
            
            # Check required inputs
            inputs_available, missing_inputs = self._verify_required_inputs(step)
            if not inputs_available:
                print(f"⚠️ Missing required inputs for step '{step.name}': {missing_inputs}")
                print("🔄 Proceeding anyway - agent may be able to handle missing inputs")
            
            # Execute the step with retry logic
            step.status = TaskStatus.RUNNING
            max_attempts = max_task_retry
            attempt = attempt_counts.get(step.name, 0)
            
            while attempt < max_attempts:
                attempt += 1
                attempt_counts[step.name] = attempt
                
                print(f"🔄 Attempt {attempt}/{max_attempts} for task: {step.name}")
                
                try:
                    # Execute DGM
                    enhanced_task = self._build_knowledge_aware_task(step.task)
                    dgm_runs = await self.dgm.start_dgm(
                        goal=enhanced_task,
                        template_uuid=None,
                        judge=judge,
                        answer=None,
                        human_validation=False,
                        max_iteration=max_dgm_iteration
                    )
                    
                    # Check success
                    dgm_success = self._get_dgm_success(dgm_runs[-1] if dgm_runs else None)
                    
                    # Extract produced outputs
                    produced_outputs = self._extract_produced_outputs(dgm_runs)
                    
                    # Create task record
                    task = Task(
                        name=step.name,
                        description=step.task,
                        dgm_runs=dgm_runs,
                        final_answers=dgm_runs[-1].answers if dgm_runs else [],
                        final_uuid=dgm_runs[-1].current_uuid if dgm_runs else None,
                        workflow_uuid=dgm_runs[-1].workflow_template if dgm_runs else None,
                        status=TaskStatus.COMPLETED if dgm_success else TaskStatus.FAILED,
                        depends_on=step.depends_on,
                        required_inputs=step.required_inputs,
                        expected_outputs=step.expected_outputs,
                        complexity=step.complexity,
                        produced_outputs=produced_outputs,
                        missing_inputs=missing_inputs
                    )
                    
                    self.task_history.append(task)
                    
                    if dgm_success:
                        # Verify expected outputs
                        outputs_produced, missing_outputs = self._verify_expected_outputs(step, produced_outputs)
                        
                        if outputs_produced:
                            print(f"✅ Task '{step.name}' completed successfully")
                            print(f"📄 Produced outputs: {produced_outputs}")
                        else:
                            print(f"⚠️ Task '{step.name}' completed but missing expected outputs: {missing_outputs}")
                            print(f"📄 Actual outputs: {produced_outputs}")
                        
                        # Add produced outputs to available outputs
                        self.available_outputs.update(produced_outputs)
                        step.status = TaskStatus.COMPLETED
                        break
                    else:
                        print(f"❌ Task '{step.name}' failed (attempt {attempt}/{max_attempts})")
                        if attempt < max_attempts:
                            print("🔄 Retrying...")
                        
                except Exception as e:
                    print(f"💥 Error executing task '{step.name}': {str(e)}")
                    if attempt < max_attempts:
                        print("🔄 Retrying...")
            
            if step.status != TaskStatus.COMPLETED:
                step.status = TaskStatus.FAILED
                print(f"❌ Giving up on task '{step.name}' after {max_attempts} attempts")
                
                # Decide whether to continue or stop
                print("❓ Do you want to continue with remaining tasks? (This may cause dependency issues)")
        
        print(f"\n🏁 Planner execution completed. Executed {len(self.task_history)} tasks.")
        return self.task_history

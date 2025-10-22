"""
Mock data generator for Mimosa AI schema classes.

This module provides utilities to generate mock data for testing and development
purposes, based on the schema defined in sources/core/schema.py.
"""

import random
import uuid
from typing import Any
from datetime import datetime

from sources.core.schema import (
    TaskStatus,
    TaskComplexity,
    GodelRun,
    PlanStep,
    Plan,
    Task,
)


class MockDataGenerator:
    """Generate mock data for all schema classes in the Mimosa AI system."""
    
    # Sample data pools for realistic mock generation
    SAMPLE_GOALS = [
        "Analyze protein structure and predict binding sites",
        "Train machine learning model for drug discovery",
        "Process genomic data and identify mutations",
        "Visualize scientific experiment results",
        "Extract information from research papers",
        "Reproduce results from scientific paper",
    ]
    
    SAMPLE_TASKS = [
        "Load and preprocess dataset",
        "Train neural network model",
        "Evaluate model performance",
        "Generate visualizations",
        "Extract features from data",
        "Perform statistical analysis",
        "Clone repository from GitHub",
        "Install dependencies",
        "Run experiments",
        "Compare results with baseline",
    ]
    
    SAMPLE_STEP_NAMES = [
        "doc_reader",
        "repo_cloner",
        "repo_analyzer",
        "dependency_miner",
        "doc_updater",
        "quality_judge",
        "data_loader",
        "model_trainer",
        "evaluator",
        "visualizer",
    ]
    
    SAMPLE_TOOLS = [
        "python_interpreter",
        "bash",
        "web_search",
        "final_answer",
        "file_reader",
        "file_writer",
        "code_executor",
    ]
    
    SAMPLE_PROMPTS = [
        "Read the documentation and extract key information",
        "Clone the repository and analyze its structure",
        "Install all required dependencies",
        "Train the model with provided data",
        "Evaluate model performance on test set",
    ]
    
    def __init__(self, seed: int | None = None):
        """
        Initialize the mock data generator.
        
        Args:
            seed: Random seed for reproducible mock data generation
        """
        if seed is not None:
            random.seed(seed)
    
    def generate_uuid(self) -> str:
        """Generate a workflow-style UUID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"{timestamp}_{unique_id}"
    
    def generate_state_result(
        self,
        num_steps: int = 6,
        workflow_uuid: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a mock state_result dictionary.
        
        Args:
            num_steps: Number of workflow steps to generate
            workflow_uuid: Optional workflow UUID, generated if not provided
            
        Returns:
            Dictionary matching the state_result.json structure
        """
        if workflow_uuid is None:
            workflow_uuid = self.generate_uuid()
        
        step_names = random.sample(self.SAMPLE_STEP_NAMES, min(num_steps, len(self.SAMPLE_STEP_NAMES)))
        
        # Generate actions for each step
        actions = []
        for _ in range(num_steps):
            num_tools = random.randint(3, 12)
            tools = [random.choice(self.SAMPLE_TOOLS) for _ in range(num_tools)]
            actions.append({"tool": tools})
        
        # Generate observations
        observations = [{"data": f"Observation data for step {i+1}"} for i in range(num_steps)]
        
        # Generate answers with realistic status messages
        answers = []
        for _, step_name in enumerate(step_names):
            status = "SUCCESS" if random.random() > 0.1 else "FAILED"
            message = f"Completed {step_name}: {random.choice(['processed data successfully', 'analysis complete', 'task finished', 'operation successful'])}"
            answer = f'{{"status": "{status}", "message": "{message}"}}'
            answers.append(answer)
        
        # Generate success flags
        success = [random.random() > 0.1 for _ in range(num_steps)]
        
        return {
            "workflow_uuid": workflow_uuid,
            "model_id": random.choice([
                "deepseek/deepseek-chat",
                "anthropic/claude-3-5-sonnet",
                "openai/gpt-4",
                "meta-llama/llama-3.1-70b",
            ]),
            "goal": random.choice(self.SAMPLE_GOALS),
            "step_name": step_names,
            "task_prompt": [random.choice(self.SAMPLE_PROMPTS) for _ in range(num_steps)],
            "actions": actions,
            "observations": observations,
            "answers": answers,
            "success": success,
        }
    
    def generate_godel_run(
        self,
        include_state_result: bool = True,
        num_answers: int = 3,
    ) -> GodelRun:
        """
        Generate a mock GodelRun instance.
        
        Args:
            include_state_result: Whether to include state_result data
            num_answers: Number of answers to generate
            
        Returns:
            Mock GodelRun instance
        """
        goal = random.choice(self.SAMPLE_GOALS)
        prompt = f"Complete the following task: {goal}"
        
        state_result = None
        if include_state_result:
            state_result = self.generate_state_result()
        
        return GodelRun(
            goal=goal,
            prompt=prompt,
            cost=round(random.uniform(0.01, 5.0), 3),
            reward=round(random.uniform(0.0, 1.0), 3),
            max_depth=random.randint(3, 10),
            iteration_count=random.randint(1, 20),
            judge=random.choice([True, False]),
            current_uuid=self.generate_uuid(),
            template_uuid=self.generate_uuid() if random.random() > 0.5 else None,
            workflow_template=random.choice(["standard", "papers_mode", "instruments_mode"]) if random.random() > 0.3 else None,
            scenario_id=f"scenario_{random.randint(1, 100)}" if random.random() > 0.5 else None,
            eval_type=random.choice(["exact_match", "similarity", "code_execution"]) if random.random() > 0.5 else None,
            answers=[f"Answer {i+1}: {random.choice(self.SAMPLE_TASKS)}" for i in range(num_answers)],
            state_result=state_result,
            plot=f"plot_data_{random.randint(1, 100)}.png" if random.random() > 0.5 else "",
            original_task=goal if random.random() > 0.5 else None,
        )
    
    def generate_plan_step(
        self,
        name: str | None = None,
        dependencies: list[str] | None = None,
    ) -> PlanStep:
        """
        Generate a mock PlanStep instance.
        
        Args:
            name: Optional step name, generated if not provided
            dependencies: Optional list of dependency names
            
        Returns:
            Mock PlanStep instance
        """
        if name is None:
            name = f"step_{random.randint(1, 100)}"
        
        if dependencies is None:
            dependencies = []
        
        return PlanStep(
            name=name,
            task=random.choice(self.SAMPLE_TASKS),
            cost=random.randint(1, 10),
            depends_on=dependencies,
            required_inputs=[f"input_{i}" for i in range(random.randint(0, 3))],
            expected_outputs=[f"output_{i}" for i in range(random.randint(1, 4))],
            complexity=random.choice([c.value for c in TaskComplexity]),
            status=random.choice(list(TaskStatus)),
        )
    
    def generate_plan(
        self,
        num_steps: int = 5,
        goal: str | None = None,
    ) -> Plan:
        """
        Generate a mock Plan instance with valid dependencies.
        
        Args:
            num_steps: Number of steps to generate
            goal: Optional plan goal, generated if not provided
            
        Returns:
            Mock Plan instance
        """
        if goal is None:
            goal = random.choice(self.SAMPLE_GOALS)
        
        steps = []
        step_names = [f"step_{i+1}" for i in range(num_steps)]
        
        for i, step_name in enumerate(step_names):
            # Only depend on previous steps
            possible_deps = step_names[:i]
            num_deps = random.randint(0, min(2, len(possible_deps)))
            dependencies = random.sample(possible_deps, num_deps) if possible_deps else []
            
            step = self.generate_plan_step(name=step_name, dependencies=dependencies)
            steps.append(step)
        
        return Plan(goal=goal, steps=steps)
    
    def generate_task(
        self,
        name: str | None = None,
        include_godel_runs: bool = True,
        num_godel_runs: int = 3,
    ) -> Task:
        """
        Generate a mock Task instance.
        
        Args:
            name: Optional task name, generated if not provided
            include_godel_runs: Whether to include godel run data
            num_godel_runs: Number of godel runs to generate
            
        Returns:
            Mock Task instance
        """
        if name is None:
            name = f"task_{random.randint(1, 100)}"
        
        dgm_runs = []
        if include_godel_runs:
            dgm_runs = [
                self.generate_godel_run(include_state_result=i == num_godel_runs - 1)
                for i in range(num_godel_runs)
            ]
        
        final_answers = [f"Final answer {i+1}" for i in range(random.randint(1, 3))]
        
        return Task(
            name=name,
            description=random.choice(self.SAMPLE_TASKS),
            run_id=random.randint(1, 100),
            dgm_runs=dgm_runs,
            final_answers=final_answers,
            cost=round(random.uniform(0.1, 20.0), 2),
            final_uuid=self.generate_uuid(),
            workflow_uuid=self.generate_uuid(),
            status=random.choice(list(TaskStatus)),
            depends_on=[f"task_{i}" for i in range(random.randint(0, 3))],
            required_inputs=[f"input_{i}" for i in range(random.randint(0, 3))],
            expected_outputs=[f"output_{i}" for i in range(random.randint(1, 4))],
            complexity=random.choice([c.value for c in TaskComplexity]),
            produced_outputs=[f"produced_output_{i}" for i in range(random.randint(0, 3))],
        )
    
    def generate_complete_workflow_example(self) -> dict[str, Any]:
        """
        Generate a complete workflow example with all schema types.
        
        Returns:
            Dictionary containing all generated mock data
        """
        plan = self.generate_plan(num_steps=5)
        tasks = [self.generate_task(name=step.name) for step in plan.steps]
        
        return {
            "plan": plan,
            "tasks": tasks,
            "godel_runs": [task.dgm_runs for task in tasks if task.dgm_runs],
            "state_results": [
                run.state_result
                for task in tasks
                for run in task.dgm_runs
                if run.state_result
            ],
        }


# Convenience function for quick mock data generation
def get_mock_data_generator(seed: int | None = None) -> MockDataGenerator:
    return MockDataGenerator(seed=seed)

"""
Schema for for storing and managing data in the Mimosa AI system. 
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class TaskStatus(Enum):
    """Enumeration for task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class TaskComplexity(Enum):
    """Enumeration for task complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# GodelRun class used in the Darwin Godel Machine for tracking run iterations.
@dataclass
class GodelRun:
    """Tracks information for a single Darwin Godel Machine run iteration."""
    goal: str
    prompt: str
    cost: float = 0.0
    reward: float = 0.0
    max_depth: int = 5
    iteration_count: int = 0
    answers: list[str] | None = field(default_factory=list)
    state_result: dict | None = None
    judge: bool = False
    need_human_validation: bool = False
    current_uuid: str | None = None
    template_uuid: str | None = None
    workflow_template: str | None = None
    scenario_id: str | None = None
    eval_type: str | None = None

@dataclass
class PlanStep:
    """Represents a single step in a plan with dependencies and I/O requirements."""
    name: str
    task: str
    depends_on: List[str] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    complexity: str = "medium"
    status: TaskStatus = TaskStatus.PENDING
    
    def __post_init__(self):
        """Validate the plan step after initialization."""
        if not self.name:
            raise ValueError("Plan step name cannot be empty")
        if not self.task:
            raise ValueError("Plan step task cannot be empty")
        if self.complexity not in [c.value for c in TaskComplexity]:
            raise ValueError(f"Invalid complexity: {self.complexity}")

@dataclass
class Plan:
    """Represents a complete execution plan with multiple steps."""
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate the plan after initialization."""
        if not self.goal:
            raise ValueError("Plan goal cannot be empty")
        if not self.steps:
            raise ValueError("Plan must contain at least one step")
        
        # Validate step names are unique
        step_names = [step.name for step in self.steps]
        if len(step_names) != len(set(step_names)):
            raise ValueError("Plan step names must be unique")
        
        # Validate dependencies exist
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_names:
                    raise ValueError(f"Step '{step.name}' depends on non-existent step '{dep}'")

# Task class used in the planner module for keeping track of tasks and their states.
@dataclass
class Task:
    name: str
    description: str
    run_id: int = 0
    dgm_runs: list[GodelRun] = field(default_factory=list) # godel run result for task
    final_answers: list[str] = field(default_factory=list) # last godel run answers
    final_uuid: str | None = None # last godel run uuid
    workflow_uuid: str | None = None # last workflow uuid
    status: TaskStatus = TaskStatus.PENDING
    depends_on: List[str] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    complexity: str = "medium"
    produced_outputs: List[str] = field(default_factory=list)  # Actual outputs produced
    missing_inputs: List[str] = field(default_factory=list)    # Missing required inputs

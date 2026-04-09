"""
Evaluation sub-package for Mimosa-AI.

Provides tools for assessing workflow outputs, scoring generated code,
detecting numerical inconsistencies, and running benchmark datasets.
"""

from .evaluator import (
    BaseEvaluator,
    GenericEvaluator,
    ScenarioEvaluator,
    WorkflowEvaluator,
    EvaluatorError,
    WorkflowDataError,
    ScenarioError,
    LLMEvaluationError,
    ScoreExtractionError,
)
from .scenario_loader import ScenarioLoader
from .science_agent_bench import ScienceAgentBenchLoader
from .capsule_evaluator import CapsuleEvaluator
from .execution_sandbox import ExecutionSandbox
from .bs_detection import BullshitDetectorNumerical, MemoryExtraction
from .csv_mode import CsvEvaluationMode
from .eval_workflow_generation import WorkflowEval

__all__ = [
    # Evaluators
    "BaseEvaluator",
    "GenericEvaluator",
    "ScenarioEvaluator",
    "WorkflowEvaluator",
    # Evaluator exceptions
    "EvaluatorError",
    "WorkflowDataError",
    "ScenarioError",
    "LLMEvaluationError",
    "ScoreExtractionError",
    # Loaders
    "ScenarioLoader",
    "ScienceAgentBenchLoader",
    # Capsule / sandbox
    "CapsuleEvaluator",
    "ExecutionSandbox",
    # Fraud / BS detection
    "BullshitDetectorNumerical",
    "MemoryExtraction",
    # Batch / CSV evaluation
    "CsvEvaluationMode",
    "WorkflowEval",
]

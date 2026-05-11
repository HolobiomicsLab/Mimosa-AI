"""
Evaluation sub-package for Mimosa-AI.

Provides tools for assessing workflow outputs, scoring generated code,
detecting numerical inconsistencies, and running benchmark datasets.
"""

from ..core.evaluators.evaluator import (
    BaseEvaluator,
    GenericEvaluator,
    ScenarioEvaluator,
    VerifierEvaluator,
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
from ..core.evaluators.bs_detection import BullshitDetectorNumerical, MemoryExtraction

__all__ = [
    # Evaluators
    "BaseEvaluator",
    "GenericEvaluator",
    "ScenarioEvaluator",
    "VerifierEvaluator",
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
]

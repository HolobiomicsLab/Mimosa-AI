"""
Core sub-package for Mimosa-AI.

Contains the central orchestration, planning, workflow management, LLM
provider abstraction, and evolutionary-improvement machinery that drives
the agent system.
"""

from .llm_provider import LLMConfig, LLMProvider
from .schema import (
    TaskStatus,
    TaskComplexity,
    SelectionLog,
    IndividualRun,
    PlanStep,
    Plan,
    Task,
)
from .workflow_info import WorkflowInfo
from .orchestrator import WorkflowOrchestrator
from .planner import Planner, PlanValidationError, DependencyError
from .factory import Factory
from .workflow_factory import WorkflowFactory
from .single_agent_factory import SingleAgentFactory
from .workflow_runner import WorkflowRunner, ExecutionStatus, ExecutionResult, RuntimeConfig
from .workflow_selection import WorkflowSelector
from .tools_manager import Tool, MCP, ToolManager
from .selection import (
    SelectionPressure,
    SelectionStrategy,
    PopulationMember,
)
from .evolution_engine import EvolutionEngine
from .variation_engine import VariationEngine

__all__ = [
    # LLM provider
    "LLMConfig",
    "LLMProvider",
    # Data schemas
    "TaskStatus",
    "TaskComplexity",
    "SelectionLog",
    "IndividualRun",
    "PlanStep",
    "Plan",
    "Task",
    # Workflow metadata
    "WorkflowInfo",
    # Orchestration & planning
    "WorkflowOrchestrator",
    "Planner",
    "PlanValidationError",
    "DependencyError",
    # Factories
    "Factory",
    "WorkflowFactory",
    "SingleAgentFactory",
    # Execution
    "WorkflowRunner",
    "ExecutionStatus",
    "ExecutionResult",
    "RuntimeConfig",
    # Selection
    "WorkflowSelector",
    # Tools / MCP
    "Tool",
    "MCP",
    "ToolManager",
    # Improvement / evolution
    "SelectionPressure",
    "SelectionStrategy",
    "PopulationMember",
    # Darwin / evolution engine
    "EvolutionEngine",
    "VariationEngine",
]

"""Core sub-package of Mimosa-AI: planning, crafting, running, evolving.

This package re-exports the central machinery of the layered system: the
planner and data schemas (Layer 0), MCP tool discovery (Layer 1), the
evolution engine with its selection and variation engines (Layer 2), and
the code factories plus workflow runner (Layer 3). Callers import these
names from ``sources.core`` instead of reaching into submodule files.
Importing the package only wires the submodules together; it runs no
agent logic itself.
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

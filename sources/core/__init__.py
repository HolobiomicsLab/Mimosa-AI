"""Core public interfaces, loaded only when their owning component is used.

Independent core modules must not initialize unrelated CLI, MCP or scientific
workflow dependencies as a side effect of importing their parent package.
"""

from importlib import import_module

_EXPORT_MODULES = {
    'LLMConfig': 'llm_provider',
    'LLMProvider': 'llm_provider',
    'TaskStatus': 'schema',
    'TaskComplexity': 'schema',
    'SelectionLog': 'schema',
    'IndividualRun': 'schema',
    'PlanStep': 'schema',
    'Plan': 'schema',
    'Task': 'schema',
    'WorkflowInfo': 'workflow_info',
    'WorkflowOrchestrator': 'orchestrator',
    'Planner': 'planner',
    'PlanValidationError': 'planner',
    'DependencyError': 'planner',
    'Factory': 'factory',
    'WorkflowFactory': 'workflow_factory',
    'SingleAgentFactory': 'single_agent_factory',
    'WorkflowRunner': 'workflow_runner',
    'ExecutionStatus': 'workflow_runner',
    'ExecutionResult': 'workflow_runner',
    'RuntimeConfig': 'workflow_runner',
    'WorkflowSelector': 'workflow_selection',
    'Tool': 'tools_manager',
    'MCP': 'tools_manager',
    'ToolManager': 'tools_manager',
    'SelectionPressure': 'selection',
    'SelectionStrategy': 'selection',
    'PopulationMember': 'selection',
    'EvolutionEngine': 'evolution_engine',
    'VariationEngine': 'variation_engine',
}

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


def __getattr__(name):
    """Resolve and cache an existing public export from its original module."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    """Expose the public interface to introspection without importing it."""
    return sorted(set(globals()) | set(__all__))

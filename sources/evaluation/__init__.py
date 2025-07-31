"""
Mimosa-AI Evaluation Framework

Standalone evaluation system for scientific workflows using unified scoring methodology.
Inspired by multiagent-collab-scenario-benchmark approach.
"""

from .scenario_loader import ScenarioLoader

# Import components that require dependencies lazily
try:
    from .simple_evaluator import Evaluator
except ImportError as e:
    print(
        f"Warning: Some evaluation components unavailable due to missing dependencies: {e}"
    )
    Evaluator = None

__version__ = "1.0.0"

__all__ = ["Evaluator", "ScenarioLoader"]

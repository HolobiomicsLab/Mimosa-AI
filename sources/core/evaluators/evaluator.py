import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any
import sys
import os

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .base import BaseEvaluator, EvaluatorError, WorkflowDataError
from .generic import GenericEvaluator, LLMEvaluationError, ScoreExtractionError
from .scenario import ScenarioEvaluator, ScenarioError

class WorkflowEvaluator:
    """Combined workflow evaluator with both judge and scenario-based evaluation capabilities.

    This is a facade class that delegates to GenericEvaluator and ScenarioEvaluator.
    """

    def __init__(self, config, scenarios_dir="datasets/scenarios"):
        """Initialize the WorkflowEvaluator with configuration.

        Args:
            config: Configuration object containing memory_dir, workflow_dir, model_pricing, and reasoning_effort

        Raises:
            EvaluatorError: If configuration is invalid or required directories don't exist
        """
        try:
            self.generic_evaluator = GenericEvaluator(config)
            self.scenario_evaluator = ScenarioEvaluator(config, scenarios_dir=scenarios_dir)
            self.logger = logging.getLogger(__name__)
            self.logger.info("WorkflowEvaluator initialized successfully")
        except Exception as e:
            if isinstance(e, EvaluatorError):
                raise
            raise EvaluatorError(f"Failed to initialize WorkflowEvaluator: {str(e)}") from e

    def evaluate(self, uuid: str, agent_answers: str = None, scenario_rubric: str = None) -> dict[str, Any]:
        """Evaluate the workflow results.

        Args:
            uuid: UUID of the workflow run to evaluate
            agent_answers: Optional list of answers from agents for evaluation
            scenario_rubric: Optional scenario ID for scenario-based evaluation

        Returns:
            Dictionary containing evaluation results

        Raises:
            EvaluatorError: If evaluation fails
        """
        try:
            # Validate inputs
            if not uuid or not isinstance(uuid, str):
                raise EvaluatorError("Invalid uuid: must be a non-empty string")

            # If scenario_rubric is provided, use scenario-based evaluation
            if scenario_rubric:
                try:
                    return self.scenario_evaluator.evaluate(uuid, scenario_rubric)
                except ScenarioError as e:
                    self.logger.error(f"Scenario evaluation failed for {uuid} with rubric {scenario_rubric}: {str(e)}")
                    self.generic_evaluator.evaluate(uuid, agent_answers)
                    return {'evaluation_type': 'generic', 'uuid': uuid}
            else:
                self.generic_evaluator.evaluate(uuid, agent_answers)
                return {'evaluation_type': 'generic', 'uuid': uuid}

        except (WorkflowDataError, ScenarioError, LLMEvaluationError) as _:
            # Re-raise specific evaluator errors
            raise
        except Exception as e:
            raise EvaluatorError(f"Evaluation failed for {uuid}: {str(e)}") from e


if __name__ == "__main__":
    """Manual testing of both evaluation modes."""
    from config import Config
    import dotenv
    dotenv.load_dotenv()
    config = Config()
    config.memory_dir = "../../sources/memory"
    config.workflow_dir = "../../sources/workflows"
    config.workflow_dir = "../../sources/workflows"
    evaluator = WorkflowEvaluator(config, scenarios_dir="../../datasets/scenarios")

    mock_uuid = "test-uuid-12345"
    try:
        result = evaluator.evaluate(mock_uuid, scenario_rubric="clintox_nn_rubric")
        print(f"✓ Scenario evaluation completed: {result}")
    except Exception as e:
        print(f"⚠ Unexpected error in scenario evaluation: {e}")


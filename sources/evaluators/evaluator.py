"""
Top level evaluator module for the workflow evaluation system.
"""

import logging
import os
import sys
from typing import Any

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .base import BaseEvaluator, EvaluatorError, WorkflowDataError  # noqa: F401
from .generic import (  # noqa: F401
    GenericEvaluator,
    LLMEvaluationError,
    ScoreExtractionError,
)
from .scenario import ScenarioError, ScenarioEvaluator
from .verifier import VerifierEvaluator

# Re-exported for `sources.evaluation.__init__`; touch with care.
__all__ = [
    "BaseEvaluator", "EvaluatorError", "WorkflowDataError",
    "GenericEvaluator", "LLMEvaluationError", "ScoreExtractionError",
    "ScenarioError", "ScenarioEvaluator",
    "VerifierEvaluator",
    "WorkflowEvaluator",
]


class WorkflowEvaluator:
    """Combined workflow evaluator: generic judge, scenario rubric, and verifier.

    Facade over GenericEvaluator, ScenarioEvaluator and VerifierEvaluator. The
    three evaluators are independent and can be invoked separately.
    """

    def __init__(self, config: "Config", scenarios_dir: str = "datasets/scenarios",
                 use_bs_penalty: bool = False, bs_fraud_threshold: float = 5.0,
                 verifier_workspace_dir: str | None = None) -> None:
        """Initialize the WorkflowEvaluator with configuration.

        Args:
            config: Configuration object containing memory_dir, workflow_dir,
                model_pricing, and reasoning_effort.
            scenarios_dir: Directory containing scenario rubric files.
            use_bs_penalty: Forwarded to GenericEvaluator — enables the
                BullshitDetectorNumerical penalty on the generic overall score.
            bs_fraud_threshold: Per-value fraud-score threshold (0-10) for the
                short fraud report.
            verifier_workspace_dir: Optional workspace directory for the
                verifier evaluator. If ``None``, the verifier uses its default.

        Raises:
            EvaluatorError: If configuration is invalid or required directories
                don't exist.
        """
        try:
            self.generic_evaluator = GenericEvaluator(
                config,
                use_bs_penalty=use_bs_penalty,
                bs_fraud_threshold=bs_fraud_threshold,
            )
            self.scenario_evaluator = ScenarioEvaluator(config, scenarios_dir=scenarios_dir)
            self.verifier_evaluator = VerifierEvaluator(config, workspace_dir=verifier_workspace_dir)
            self.logger = logging.getLogger(__name__)
            self.logger.info("WorkflowEvaluator initialized successfully")
        except Exception as e:
            if isinstance(e, EvaluatorError):
                raise
            raise EvaluatorError(f"Failed to initialize WorkflowEvaluator: {str(e)}") from e

    def evaluate(
        self,
        uuid: str,
        agent_answers: str | None = None,
        evaluator_type: str = "verifier",
        scenario_rubric: str | None = None,
        rubric_anchor_uuid: str | None = None,
    ) -> dict[str, Any]:
        """Route to the requested evaluator.

        Args:
            uuid: UUID of the workflow run to evaluate.
            agent_answers: Optional answers from agents (forwarded to generic).
            evaluator_type: One of {"generic", "scenario", "verifier"}.
                - "generic":  4-criterion LLM judge with optional bs penalty.
                - "scenario": rubric-based scoring; requires `scenario_rubric`.
                - "verifier": atomic-claim verification pipeline.
            scenario_rubric: Scenario ID, required when `evaluator_type="scenario"`.
            rubric_anchor_uuid: Optional ancestor UUID whose verifier cache
                (``_verifier_tmp/<id>/claims.json`` + ``verify_*.py``) should
                be reused for stable cross-generation scoring. Forwarded only
                to the verifier evaluator; ignored for generic/scenario.

        Returns:
            Dictionary containing evaluation results.

        Raises:
            EvaluatorError: If evaluation fails or arguments are invalid.
        """
        if not uuid or not isinstance(uuid, str):
            raise EvaluatorError("Invalid uuid: must be a non-empty string")

        evaluator_type = (evaluator_type or "generic").lower()
        if evaluator_type not in {"generic", "scenario", "verifier"}:
            raise EvaluatorError(
                f"Unknown evaluator_type '{evaluator_type}'. "
                "Expected one of: generic, scenario, verifier."
            )

        try:
            if evaluator_type == "scenario":
                if not scenario_rubric:
                    raise EvaluatorError(
                        "evaluator_type='scenario' requires scenario_rubric."
                    )
                try:
                    return self.scenario_evaluator.evaluate(uuid, scenario_rubric)
                except ScenarioError as e:
                    self.logger.error(
                        f"Scenario evaluation failed for {uuid} with rubric "
                        f"{scenario_rubric}: {str(e)} — falling back to generic."
                    )
                    self.generic_evaluator.evaluate(uuid, agent_answers)
                    return {"evaluation_type": "generic", "uuid": uuid}

            if evaluator_type == "verifier":
                result = self.verifier_evaluator.evaluate(
                    uuid, rubric_anchor_uuid=rubric_anchor_uuid
                )
                return {"evaluation_type": "verifier", "uuid": uuid, **result}

            # Default: generic
            self.generic_evaluator.evaluate(uuid, agent_answers)
            return {"evaluation_type": "generic", "uuid": uuid}

        except (EvaluatorError, WorkflowDataError, ScenarioError, LLMEvaluationError):
            raise
        except Exception as e:
            raise EvaluatorError(f"Evaluation failed for {uuid}: {str(e)}") from e


if __name__ == "__main__":
    """Manual testing of both evaluation modes."""
    import dotenv

    from config import Config
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


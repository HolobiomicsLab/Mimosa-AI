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

from sources.core.llm_provider import LLMConfig, LLMProvider
from sources.core.workflow_info import WorkflowInfo

class EvaluatorError(Exception):
    """Base exception for evaluator errors."""
    pass


class WorkflowDataError(EvaluatorError):
    """Exception raised when workflow data cannot be loaded or is invalid."""
    pass


class ScenarioError(EvaluatorError):
    """Exception raised when scenario data is invalid or cannot be processed."""
    pass


class LLMEvaluationError(EvaluatorError):
    """Exception raised when LLM evaluation fails."""
    pass


class ScoreExtractionError(EvaluatorError):
    """Exception raised when scores cannot be extracted from evaluation text."""
    pass


class BaseEvaluator:
    """Base evaluator with common functionality for workflow evaluation."""

    def __init__(self, config):
        """Initialize the BaseEvaluator with configuration.

        Args:
            config: Configuration object containing memory_dir, workflow_dir, model_pricing, and reasoning_effort

        Raises:
            EvaluatorError: If configuration is invalid or required directories don't exist
        """
        try:
            if not hasattr(config, 'memory_dir') or not config.memory_dir:
                raise EvaluatorError("Configuration must include 'memory_dir'")
            if not hasattr(config, 'workflow_dir') or not config.workflow_dir:
                raise EvaluatorError("Configuration must include 'workflow_dir'")
            if not hasattr(config, 'model_pricing'):
                raise EvaluatorError("Configuration must include 'model_pricing'")
            if not hasattr(config, 'reasoning_effort'):
                raise EvaluatorError("Configuration must include 'reasoning_effort'")

            self.memory_dir = Path(config.memory_dir)
            self.workflow_dir = Path(config.workflow_dir)
            self.model_pricing = config.model_pricing

            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self.workflow_dir.mkdir(parents=True, exist_ok=True)

            self.judge_model = config.judge_model
            try:
                provider, model = self.judge_model.split("/", 1) if "/" in self.judge_model else ("openai", self.judge_model)
                self.llm_config = LLMConfig().from_dict({
                    "model": model,
                    "provider": provider,
                    "reasoning_effort": config.reasoning_effort,
                    "max_tokens": getattr(config, 'max_tokens', 8192),
                    "openrouter_provider": config.openrouter_provider_for(self.judge_model),
                })
            except Exception as e:
                raise EvaluatorError(f"Failed to initialize LLM configuration: {str(e)}") from e

            self.logger = logging.getLogger(__name__)

        except Exception as e:
            if isinstance(e, EvaluatorError):
                raise
            raise EvaluatorError(f"Failed to initialize BaseEvaluator: {str(e)}") from e

    def _load_workflow_data(self, workflow_id: str) -> WorkflowInfo:
        """Load workflow execution data from UUID folder using WorkflowInfo.

        Args:
            workflow_id: UUID of the workflow to load

        Returns:
            WorkflowInfo instance containing workflow data

        Raises:
            WorkflowDataError: If workflow data cannot be loaded
        """
        if not workflow_id or not isinstance(workflow_id, str):
            raise WorkflowDataError("Invalid workflow_id: must be a non-empty string")

        workflow_path = Path(self.workflow_dir) / workflow_id

        if not workflow_path.exists():
            return WorkflowInfo(workflow_id, workflow_path)

        if not workflow_path.is_dir():
            raise WorkflowDataError(f"Workflow path is not a directory: {workflow_path}")

        try:
            workflow_info = WorkflowInfo(workflow_id, workflow_path)
            state_result = workflow_info.state_result
            code = workflow_info.code
            if state_result and not isinstance(state_result, dict):
                raise WorkflowDataError(f"state_result must be a dictionary, got {type(state_result)}")

            if not state_result:
                self.logger.warning(f"State result is empty for workflow {workflow_id}")
            if not code:
                self.logger.warning(f"Workflow code is empty for workflow {workflow_id}")

            return workflow_info

        except ValueError as e:
            raise WorkflowDataError(str(e)) from e
        except Exception as e:
            raise WorkflowDataError(f"Failed to load workflow data for {workflow_id}: {str(e)}") from e

    def workflow_execution_text(self, uuid: str) -> tuple[str, bool] | None:
        """Generate workflow execution text for evaluation using WorkflowInfo.
        Args:
            uuid: UUID of the workflow
        Returns:
            Formatted workflow execution text and execution success status
        Raises:
            WorkflowDataError: If workflow data is invalid
        """
        try:
            workflow_info = self._load_workflow_data(uuid)
            state_result = workflow_info.state_result
            workflow_code = workflow_info.code
            goal = workflow_info.goal or "Goal not specified"

            if not state_result and not workflow_code:
                return "workflow execution fully failed. report it.", False

            result = workflow_info.answers

            res = json.dumps(result, indent=2)
            success = not "[]" in res
            return f"""
                   GOAL:
                    The workflow's goal was to achieve the following scientific/research objective:
                   {goal}
                   FINAL ANSWER FROM AGENT(S) EXECUTION:
                   The final answer produced by the agent(s) at the end of the workflow execution was:
                   {res}
                   """, success
        except Exception as e:
            raise e

    def _save_results(self, scores: dict[str, float], uuid: str, eval_type: str) -> None:
        """Update the state result file with the evaluation scores.

        Args:
            scores: The scores to add to the state result
            uuid: UUID of the workflow run
            eval_type: Type of evaluation ('generic' or 'scenario')

        Raises:
            EvaluatorError: If results cannot be saved
        """
        if not uuid or not isinstance(uuid, str):
            raise EvaluatorError("Invalid uuid for saving results")

        if not isinstance(scores, dict):
            raise EvaluatorError("Scores must be a dictionary")

        if not eval_type or not isinstance(eval_type, str):
            raise EvaluatorError("Invalid eval_type for saving results")

        try:
            workflow_path = Path(self.workflow_dir) / uuid
            state_result_path = workflow_path / "state_result.json"

            # Load existing state result if it exists
            state_result = {}
            if state_result_path.exists():
                try:
                    with open(state_result_path, encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            state_result = json.loads(content)
                            if not isinstance(state_result, dict):
                                self.logger.warning("State result is not a dictionary, creating new one")
                                state_result = {}
                except (json.JSONDecodeError, OSError) as e:
                    self.logger.warning(f"Could not load existing state result: {str(e)}, creating new one")
                    state_result = {}
            else:
                self.logger.warning(f"State result file not found for UUID {uuid}, creating new one")

            # Add scores to state result
            if "evaluation" not in state_result:
                state_result["evaluation"] = {}

            state_result["evaluation"][eval_type] = scores

            # Write updated state result back to file
            workflow_path.mkdir(parents=True, exist_ok=True)
            with open(state_result_path, "w", encoding='utf-8') as f:
                json.dump(state_result, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Scores extracted and saved to state result for {uuid}")

        except OSError as e:
            raise EvaluatorError(f"OS error updating state result: {str(e)}") from e
        except json.JSONEncodeError as e:
            raise EvaluatorError(f"JSON encoding error updating state result: {str(e)}") from e
        except Exception as e:
            raise EvaluatorError(f"Unexpected error updating state result: {str(e)}") from e

    def _get_judge_system_prompt(self) -> str:
        """Get system prompt for LLM judge (keeping existing format exactly)."""
        # Preserving original prompt exactly
        return """You are an expert scientific researcher and rigorous multi-agent system evaluator. Your task is to assess whether a computational workflow achieved its intended goals through coordinated agent collaboration, while ensuring scientific validity and technical correctness.

    You will evaluate:
    System Description
    - The workflow's goal (scientific/research objective)
    - The agents involved, their roles, and expected behaviors
    - The workflow trace (inputs, outputs, execution steps)
    - The Python workflow implementation

    Multi-Agent System Evaluation Criteria
    - Role Consistency: Does each agent behave as expected given its role?
    - Logical Flow: Does each step follow coherently from the previous one?
    - Output Quality: Are outputs correct, useful, and free of errors?
    - Bottlenecks/Failures: Are there inefficiencies, misunderstandings, or failures?
    - Collaboration Effectiveness: Do agents work together optimally?
    - Goal Alignement: Did the execution achieve the defined objective?

    Scientific Research Evaluation Criteria
    -Result Accuracy: Were the requested scientific results/analysis produced correctly?
    - Research Question Addressed: Was the core problem adequately solved?
    - Tool Usage: Were tools (agents, algorithms, data) applied correctly and in sequence?
    - Error Handling: Did the system detect and manage errors appropriately?
    - Clarity & Professionalism: Are results presented clearly and in a usable format?"""

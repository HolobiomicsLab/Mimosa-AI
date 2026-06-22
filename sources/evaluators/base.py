"""
Base class for workflow evaluators, providing shared setup and helper methods.
"""

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


def extract_json_payload(text: str) -> str:
    """First balanced JSON object/array in *text*, tolerant of fences and prose.

    Args:
        text: Arbitrary string possibly containing a JSON document, with or
            without ``` fences and with prose around it.

    Returns:
        The substring spanning the first balanced ``{...}`` or ``[...]``
        block, or an empty string when none is found.
    """
    if not text:
        return ""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
    return ""

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
    """Base evaluator with common functionality for workflow evaluation.

    Provides shared setup (memory and workflow directories, LLM configuration,
    judge model selection) and helper methods used by concrete evaluators.
    """

    def __init__(self, config: "Config") -> None:
        """Initialize the BaseEvaluator with configuration.

        Args:
            config: Configuration object containing memory_dir, workflow_dir,
                model_pricing, reasoning_effort, judge_model, and OpenRouter
                provider/quantization lookup helpers.

        Raises:
            EvaluatorError: If configuration is invalid or required directories
                cannot be created/initialized.
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
                    "temperature": 0.2,
                    "reasoning_effort": config.reasoning_effort,
                    "max_tokens": getattr(config, 'max_tokens', 8192),
                    "openrouter_provider": config.openrouter_provider_for(self.judge_model),
                    "openrouter_quantizations": config.openrouter_quantizations_for(self.judge_model),
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
            uuid: UUID of the workflow.

        Returns:
            A tuple ``(execution_text, success)``: the formatted workflow
            execution text (goal + final answer) and a boolean indicating
            whether the execution produced non-empty answers.

        Raises:
            WorkflowDataError: If workflow data is invalid.
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
        """Return the system prompt used by the LLM judge.
        Prompt is quite generic because it is shared across verifiers evaluators call _call_judge_for_json.
        Used to extract claims, write verifiers program, select files and any other call that requires a judge.

        Returns:
            The fixed judge system prompt string, kept verbatim for
            reproducibility across evaluations.
        """
        # Preserving original prompt exactly
        return """You are an expert scientific researcher and rigorous evaluator.
 Your task is to assess whether a computational workflow achieved its intended goals ensuring goal alignment, scientific validity and technical correctness.
    """

    # ------------------------------------------------------------------
    # Judge call helpers (shared across evaluators)
    # ------------------------------------------------------------------

    _JSON_RETRY_FEEDBACK = (
        "Your previous response could not be parsed as JSON. Reply with ONLY "
        "the JSON object, no prose, no markdown fences, no commentary."
    )

    def _call_judge(self, uuid: str, agent_name: str, prompt: str) -> str:
        """One judge round-trip; raises whatever the LLM provider raises.

        Args:
            uuid: Workflow identifier (used to scope per-uuid memory).
            agent_name: Logical name for this judge call (used for memory file).
            prompt: User-side prompt to send to the judge.

        Returns:
            Raw text response from the LLM provider.
        """
        memory_path = Path(self.memory_dir) / uuid
        memory_path.mkdir(parents=True, exist_ok=True)
        provider = LLMProvider(
            agent_name=agent_name,
            memory_path=memory_path,
            system_msg=self._get_judge_system_prompt(),
            config=self.llm_config,
        )
        return provider(prompt)

    def _call_judge_for_json(
        self,
        uuid: str,
        agent_name: str,
        prompt: str,
    ) -> tuple[Any, str | None]:
        """Call the judge expecting JSON; one retry on parse failure.

        Both call exceptions and JSON-parse errors are absorbed so callers
        never have to wrap in try/except.

        Args:
            uuid: Workflow identifier.
            agent_name: Logical name for the call (a ``_retry`` suffix is added on retry).
            prompt: User-side prompt to send to the judge.

        Returns:
            ``(parsed, None)`` on success or ``(None, error_str)`` on terminal failure.
        """
        last_err: str | None = None
        cur_prompt = prompt
        for attempt in (1, 2):
            agent = agent_name if attempt == 1 else f"{agent_name}_retry"
            try:
                raw = self._call_judge(uuid, agent, cur_prompt)
            except Exception as e:
                last_err = f"judge call failed: {type(e).__name__}: {e}"
                self.logger.warning(f"[{agent}] {last_err}")
                # Retrying when the call itself raised is unlikely to help; bail.
                return None, last_err

            payload = extract_json_payload(raw or "")
            if payload:
                try:
                    return json.loads(payload), None
                except json.JSONDecodeError as e:
                    last_err = f"invalid JSON: {e}"
            else:
                last_err = "no JSON object found in response"

            if attempt == 1:
                self.logger.warning(
                    f"[{agent}] JSON parse failed ({last_err}); retrying once"
                )
                cur_prompt = (
                    f"{prompt}\n\nPREVIOUS ATTEMPT FAILED: {last_err}\n"
                    f"{self._JSON_RETRY_FEEDBACK}\n"
                )
        return None, last_err

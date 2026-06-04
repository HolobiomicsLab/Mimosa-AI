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
from sources.evaluation.scenario_loader import ScenarioLoader

from .base import *

class ScenarioEvaluator(BaseEvaluator):
    """Evaluator for scenario-based workflow evaluation.

    Loads scenario rubrics from disk and scores workflows either with the
    legacy assertion format or the ScienceAgentBench rubric format, delegating
    individual checks to an LLM judge.
    """

    def __init__(self, config: "Config", scenarios_dir: str = "datasets/scenarios") -> None:
        """Initialize the ScenarioEvaluator.

        Args:
            config: Configuration object forwarded to ``BaseEvaluator``.
            scenarios_dir: Directory containing scenario rubric files used by
                the underlying ``ScenarioLoader``.

        Raises:
            EvaluatorError: If the scenario loader cannot be initialized.
        """
        super().__init__(config)

        try:
            self.scenario_loader = ScenarioLoader(scenarios_dir=scenarios_dir)
        except Exception as e:
            raise EvaluatorError(f"Failed to initialize scenario loader: {str(e)}") from e

        self.logger.info("ScenarioEvaluator initialized successfully")

    def evaluate(self, uuid: str, scenario_rubric: str) -> dict[str, Any]:
        """Evaluate a workflow against a scenario with scoring.

        Dispatches to the rubric or legacy evaluation path depending on the
        scenario's structure. If ``uuid`` is ``None``, returns a zero-score
        placeholder result.

        Args:
            uuid: UUID of the workflow to evaluate.
            scenario_rubric: ID of the scenario to evaluate against.

        Returns:
            Dictionary containing scenario evaluation results
            (``earned_points``/``total_points``/``score``/``scenario_rubric``
            or the legacy equivalents).

        Raises:
            ScenarioError: If scenario evaluation fails.
        """
        try:
            self.logger.info(f"Evaluating workflow {uuid} against scenario {scenario_rubric}")

            # Handle null uuid case
            if uuid is None:
                return {
                    'earned_points': 0,
                    'total_points': 0,
                    'score': 0,
                    'scenario_rubric': scenario_rubric
                }

            # Validate inputs
            if not isinstance(uuid, str) or not uuid.strip():
                raise ScenarioError("Invalid uuid: must be a non-empty string")

            if not isinstance(scenario_rubric, str) or not scenario_rubric.strip():
                raise ScenarioError("Invalid scenario_rubric: must be a non-empty string")

            # Load scenario with error handling
            try:
                scenario = self.scenario_loader.load_scenario(scenario_rubric)
                if not scenario:
                    raise ScenarioError(f"Scenario {scenario_rubric} not found or is empty")

                if not isinstance(scenario, dict):
                    raise ScenarioError(f"Scenario {scenario_rubric} is not a valid dictionary")

            except Exception as e:
                if isinstance(e, ScenarioError):
                    raise
                raise ScenarioError(f"Failed to load scenario {scenario_rubric}: {str(e)}") from e

            # Check if this is the new rubric format or legacy format
            is_rubric_format = "total_points" in scenario

            if is_rubric_format:
                return self._evaluate_rubric_format(uuid, scenario_rubric, scenario)
            else:
                return self._evaluate_legacy_format(uuid, scenario_rubric, scenario)

        except ScenarioError:
            raise
        except Exception as e:
            raise ScenarioError(f"Scenario evaluation failed: {str(e)}") from e

    def _evaluate_legacy_format(self, uuid: str, scenario_rubric: str, scenario: dict[str, Any]) -> dict[str, Any]:
        """Evaluate workflow using legacy assertion format.

        Args:
            uuid: UUID of the workflow.
            scenario_rubric: ID of the scenario.
            scenario: Scenario dictionary with an ``"assertions"`` list.

        Returns:
            Dictionary with ``passed_assertions``, ``total_assertions``,
            ``score`` and ``scenario_rubric`` for evolution tracking.

        Raises:
            ScenarioError: If the scenario is missing or malformed.
        """
        if "assertions" not in scenario:
            raise ScenarioError(f"Scenario {scenario_rubric} missing 'assertions' field")

        assertions = scenario["assertions"]
        if not isinstance(assertions, list):
            raise ScenarioError(f"Scenario {scenario_rubric} 'assertions' must be a list")

        if not assertions:
            self.logger.warning(f"Scenario {scenario_rubric} has no assertions")

        # Evaluate all assertions
        assertion_results = []
        for i, assertion in enumerate(assertions):
            try:
                if not isinstance(assertion, dict):
                    raise ScenarioError(f"Assertion {i} is not a dictionary")

                result = self._evaluate_assertion(uuid, assertion)
                assertion_results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to evaluate assertion {i}: {str(e)}")
                # Add failed assertion result to maintain consistency
                assertion_results.append({
                    "id": assertion.get("id", f"assertion_{i}"),
                    "description": assertion.get("description", "Unknown assertion"),
                    "passed": False,
                    "evidence": f"Evaluation error: {str(e)}",
                    "confidence": 0.0,
                })

        # Calculate score
        passed_count = sum(int(result.get("passed", False)) for result in assertion_results)
        total_count = len(assertion_results)
        score = passed_count / total_count if total_count > 0 else 0.0
        print(f"Scenario evaluation completed for {scenario_rubric}: {passed_count}/{total_count} assertions passed, score: {score:.4f}")

        # Generate results
        results = {
            "scenario_rubric": scenario_rubric,
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "passed_assertions": passed_count,
            "total_assertions": total_count,
            "assertion_results": assertion_results,
            "judge_model": self.judge_model,
        }

        # Save results with error handling
        try:
            self._save_results(results, uuid, 'scenario')
        except EvaluatorError as e:
            self.logger.error(f"Failed to save scenario results: {str(e)}")

        # Save evaluation details to evaluation.txt
        try:
            evaluation_path = self.workflow_dir / uuid / "evaluation.txt"
            evaluation_path.parent.mkdir(parents=True, exist_ok=True)
            with open(evaluation_path, "w", encoding='utf-8') as file:
                file.write(f"Scenario Evaluation: {scenario_rubric}\n")
                file.write(f"Timestamp: {results['timestamp']}\n")
                file.write(f"Score: {score:.4f}\n")
                file.write(f"Passed Assertions: {passed_count}/{total_count}\n")
                file.write(f"Judge Model: {self.judge_model}\n")
                file.write("\n" + "="*60 + "\n\n")
                file.write("Assertion Results:\n\n")
                for result in assertion_results:
                    status = "✓ PASS" if result.get("passed", False) else "✗ FAIL"
                    file.write(f"[{status}] {result.get('id', 'unknown')}: {result.get('description', 'No description')}\n")
                    file.write(f"    Evidence: {result.get('evidence', 'No evidence')}\n")
                    file.write(f"    Confidence: {result.get('confidence', 0.0):.2f}\n\n")
            self.logger.info(f"Scenario evaluation saved to: {evaluation_path}")
        except OSError as e:
            self.logger.error(f"Failed to save evaluation to file: {str(e)}")

        # Return assertion metrics for Evolution tracking
        return {
            'passed_assertions': passed_count,
            'total_assertions': total_count,
            'score': score,
            'scenario_rubric': scenario_rubric
        }

    def _evaluate_rubric_format(self, uuid: str, scenario_rubric: str, scenario: dict[str, Any]) -> dict[str, Any]:
        """Evaluate workflow using ScienceAgentBench rubric format.

        Iterates over standard ScienceAgentBench categories (data loading,
        data processing, modeling/analysis/visualization, output formatting,
        output saving) and sums earned points across all rubric items.

        Args:
            uuid: UUID of the workflow.
            scenario_rubric: ID of the scenario.
            scenario: Scenario dictionary with ``total_points`` and rubric
                category lists.

        Returns:
            Dictionary with ``earned_points``, ``total_points``, ``score``
            (fraction of points earned) and ``scenario_rubric``.
        """
        total_possible_points = scenario.get("total_points", 0)

        # Standard ScienceAgentBench categories
        categories = [
            "data_loading",
            "data_processing",
            "modeling_or_analysis_or_visualization",
            "output_formatting",
            "output_saving"
        ]

        # Collect all rubric items from all categories
        all_items = []
        for category_name in categories:
            if category_name in scenario:
                category_items = scenario[category_name]
                if isinstance(category_items, list):
                    for item in category_items:
                        item_with_category = item.copy()
                        item_with_category["category"] = category_name
                        all_items.append(item_with_category)

        if not all_items:
            self.logger.warning(f"Scenario {scenario_rubric} has no rubric items")

        # Evaluate all rubric items
        item_results = []
        total_earned = 0

        for i, item in enumerate(all_items):
            try:
                result = self._evaluate_rubric_item(uuid, item)
                item_results.append(result)

                # Calculate earned points
                if result.get("passed", False):
                    earned = item.get("points", 0)
                else:
                    # No partial credit for failed rubric items
                    earned = 0

                total_earned += earned

            except Exception as e:
                self.logger.error(f"Failed to evaluate rubric item {i}: {str(e)}")
                # Add failed item result
                item_results.append({
                    "name": item.get("name", f"item_{i}"),
                    "category": item.get("category", "unknown"),
                    "description": item.get("description", "Unknown item"),
                    "possible_points": item.get("points", 0),
                    "earned_points": 0,
                    "passed": False,
                    "evidence": f"Evaluation error: {str(e)}",
                    "confidence": 0.0,
                })

        # Calculate score as percentage
        score = (total_earned / total_possible_points) if total_possible_points > 0 else 0.0

        # Generate results
        results = {
            "scenario_rubric": scenario_rubric,
            "timestamp": datetime.now().isoformat(),
            "score": score,
            "earned_points": total_earned,
            "total_points": total_possible_points,
            "item_results": item_results,
            "judge_model": self.judge_model,
            "format": "rubric"
        }

        # Save results
        try:
            self._save_results(results, uuid, 'scenario')
        except EvaluatorError as e:
            self.logger.error(f"Failed to save scenario results: {str(e)}")

        # Save evaluation details to evaluation.txt
        try:
            evaluation_path = self.workflow_dir / uuid / "evaluation.txt"
            evaluation_path.parent.mkdir(parents=True, exist_ok=True)
            with open(evaluation_path, "w", encoding='utf-8') as file:
                file.write(f"Scenario Evaluation (Rubric Format): {scenario_rubric}\n")
                file.write(f"Timestamp: {results['timestamp']}\n")
                file.write(f"Score: {score:.4f} ({total_earned:.1f}/{total_possible_points} points)\n")
                file.write(f"Judge Model: {self.judge_model}\n")
                file.write("\n" + "="*60 + "\n\n")
                file.write("Rubric Item Results:\n\n")
                for result in item_results:
                    status = "✓ PASS" if result.get("passed", False) else "✗ FAIL"
                    file.write(f"[{status}] {result.get('category', 'unknown')} - {result.get('name', 'unknown')}\n")
                    file.write(f"    Description: {result.get('description', 'No description')}\n")
                    file.write(f"    Points: {result.get('earned_points', 0):.1f}/{result.get('possible_points', 0)}\n")
                    file.write(f"    Evidence: {result.get('evidence', 'No evidence')}\n")
                    file.write(f"    Confidence: {result.get('confidence', 0.0):.2f}\n\n")
            self.logger.info(f"Scenario evaluation saved to: {evaluation_path}")
        except OSError as e:
            self.logger.error(f"Failed to save evaluation to file: {str(e)}")

        # Return metrics for Evolution tracking
        return {
            'earned_points': total_earned,
            'total_points': total_possible_points,
            'score': score,
            'scenario_rubric': scenario_rubric
        }

    def _evaluate_rubric_item(self, uuid: str, item: dict[str, Any]) -> dict[str, Any]:
        """Evaluate single rubric item using LLM.

        Args:
            uuid: UUID of the workflow.
            item: Rubric item dictionary with ``name``, ``description``,
                ``points`` and ``category`` keys.

        Returns:
            Dictionary with ``name``, ``category``, ``description``,
            ``possible_points``, ``earned_points``, ``passed``, ``evidence``,
            and ``confidence``.

        Raises:
            LLMEvaluationError: If rubric item evaluation fails.
        """
        try:
            # Validate item structure
            if not isinstance(item, dict):
                raise LLMEvaluationError("Rubric item must be a dictionary")

            item_name = item.get("name", "unknown")
            item_desc = item.get("description", "No description provided")
            item_points = item.get("points", 0)
            item_category = item.get("category", "unknown")

            # Build judge prompt
            try:
                judge_prompt = self._build_rubric_item_prompt(uuid, item)
            except Exception as e:
                raise LLMEvaluationError(f"Failed to build rubric item prompt: {str(e)}") from e

            # Use LLMProvider with error handling
            try:
                memory_path = self.memory_dir / uuid
                memory_path.mkdir(parents=True, exist_ok=True)

                llm_provider = LLMProvider(
                    agent_name=f"rubric_judge_{item_category}_{item_name}",
                    memory_path=memory_path,
                    system_msg=self._get_judge_system_prompt(),
                    config=self.llm_config,
                )

                judge_text = llm_provider(judge_prompt)

                if not judge_text or not isinstance(judge_text, str):
                    raise LLMEvaluationError("LLM returned empty or invalid response")

                judge_text = judge_text.strip()

            except Exception as e:
                raise LLMEvaluationError(f"LLM call failed for rubric item {item_name}: {str(e)}") from e

            # Parse judge response
            try:
                passed, evidence, confidence = self._parse_judge_response(judge_text)
            except Exception as e:
                self.logger.error(f"Failed to parse judge response for rubric item {item_name}: {str(e)}")
                passed, evidence, confidence = False, f"Parse error: {str(e)}", 0.0

            # Calculate earned points
            if passed:
                earned_points = item_points
            else:
                # Partial credit based on confidence
                earned_points = item_points * confidence

            return {
                "name": item_name,
                "category": item_category,
                "description": item_desc,
                "possible_points": item_points,
                "earned_points": earned_points,
                "passed": passed,
                "evidence": evidence,
                "confidence": confidence,
            }

        except LLMEvaluationError:
            raise
        except Exception as e:
            raise LLMEvaluationError(f"Rubric item evaluation failed: {str(e)}") from e

    def _build_rubric_item_prompt(self, uuid: str, item: dict[str, Any]) -> str:
        """Build judge prompt for rubric item evaluation.

        Args:
            uuid: UUID of the workflow.
            item: Rubric item dictionary.

        Returns:
            Formatted judge prompt string ready to be sent to the LLM.

        Raises:
            WorkflowDataError: If workflow execution text cannot be generated.
        """
        try:
            execution_text, _ = self.workflow_execution_text(uuid)
            if not execution_text:
                raise WorkflowDataError(f"Cannot generate execution text for workflow {uuid}")

            name = item.get("name", "Unknown")
            description = item.get("description", "No description provided")
            category = item.get("category", "unknown")
            points = item.get("points", 0)

            return f"""
{execution_text}

RUBRIC ITEM TO EVALUATE:
Category: {category}
Name: {name}
Description: {description}
Points: {points}

EVALUATION TASK:
Based on the complete execution state and workflow code above, determine if the
rubric item criteria has been met.
Focus on whether the workflow successfully completed the specific requirement described.
Analyze the full JSON state and workflow implementation to make your judgment.

Respond in this exact format:
{{
    "verdict": [true/false],
    "evidence": [Specific evidence from the execution that supports your verdict],
    "confidence": [0.0-1.0 confidence score]
}}
"""

        except WorkflowDataError:
            raise
        except Exception as e:
            raise WorkflowDataError(f"Failed to build rubric item prompt: {str(e)}") from e

    def _evaluate_assertion(self, uuid: str, assertion: dict[str, Any]) -> dict[str, Any]:
        """Evaluate single assertion using the existing LLM prompt format.

        Args:
            uuid: UUID of the workflow.
            assertion: Assertion dictionary to evaluate.

        Returns:
            Dictionary with ``id``, ``description``, ``passed``, ``evidence``,
            and ``confidence``.

        Raises:
            LLMEvaluationError: If assertion evaluation fails.
        """
        try:
            # Validate assertion structure
            if not isinstance(assertion, dict):
                raise LLMEvaluationError("Assertion must be a dictionary")

            assertion_id = assertion.get("id", "unknown")
            assertion_desc = assertion.get("description", "No description provided")

            # Build judge prompt using existing format (preserving original)
            try:
                judge_prompt = self._build_judge_prompt(uuid, assertion)
            except Exception as e:
                raise LLMEvaluationError(f"Failed to build judge prompt: {str(e)}") from e

            # Use LLMProvider with error handling
            try:
                memory_path = self.memory_dir / uuid
                memory_path.mkdir(parents=True, exist_ok=True)

                llm_provider = LLMProvider(
                    agent_name=f"scenario_judge_{assertion_id}",
                    memory_path=memory_path,
                    system_msg=self._get_judge_system_prompt(),
                    config=self.llm_config,
                )

                judge_text = llm_provider(judge_prompt)

                if not judge_text or not isinstance(judge_text, str):
                    raise LLMEvaluationError("LLM returned empty or invalid response")

                judge_text = judge_text.strip()

            except Exception as e:
                raise LLMEvaluationError(f"LLM call failed for assertion {assertion_id}: {str(e)}") from e

            # Parse judge response with error handling
            try:
                passed, evidence, confidence = self._parse_judge_response(judge_text)
            except Exception as e:
                self.logger.error(f"Failed to parse judge response for assertion {assertion_id}: {str(e)}")
                passed, evidence, confidence = False, f"Parse error: {str(e)}", 0.0

            return {
                "id": assertion_id,
                "description": assertion_desc,
                "passed": passed,
                "evidence": evidence,
                "confidence": confidence,
            }

        except LLMEvaluationError:
            raise
        except Exception as e:
            raise LLMEvaluationError(f"Assertion evaluation failed: {str(e)}") from e

    def _build_judge_prompt(self, uuid: str, assertion: dict[str, Any]) -> str:
        """Build judge prompt with workflow data.

        Args:
            uuid: UUID of the workflow.
            assertion: Assertion dictionary.

        Returns:
            Formatted judge prompt string ready to be sent to the LLM.

        Raises:
            WorkflowDataError: If workflow execution text cannot be generated.
        """
        try:
            execution_text, _ = self.workflow_execution_text(uuid)
            if not execution_text:
                raise WorkflowDataError(f"Cannot generate execution text for workflow {uuid}")

            criteria = assertion.get("evaluation_criteria", "Standard evaluation")
            description = assertion.get("description", "No description provided")

            # Build prompt using existing format (preserving original exactly)
            return f"""
{execution_text}

ASSERTION TO EVALUATE:
Description: {description}
Evaluation Criteria: {criteria}

EVALUATION TASK:
Based on the complete execution state and workflow code above, determine if the
assertion is true or false.
Focus on whether the workflow achieved the goals and execution was successful.
Analyze the full JSON state and workflow implementation to make your judgment.

Respond in this exact format:
{{
    "verdict": [true/false],
    "evidence": [Specific evidence from the execution that supports your verdict],
    "confidence": [0.0-1.0 confidence score]
}}
"""

        except WorkflowDataError:
            raise
        except Exception as e:
            raise WorkflowDataError(f"Failed to build judge prompt: {str(e)}") from e

    def _parse_judge_response(self, judge_text: str) -> tuple[bool, str, float]:
        """Parse LLM judge response from JSON format.

        Args:
            judge_text: Raw response text from the LLM.

        Returns:
            Tuple ``(verdict, evidence, confidence)`` where ``verdict`` is a
            boolean parsed from ``"TRUE"``/``"FALSE"``, ``evidence`` is the
            judge's free-text justification, and ``confidence`` is a float in
            ``[0.0, 1.0]``. On unexpected internal errors, returns a falsy
            placeholder triple instead of raising.

        Raises:
            ScoreExtractionError: If the response is empty/invalid or cannot
                be parsed as the expected JSON structure.
        """
        if not judge_text or not isinstance(judge_text, str):
            raise ScoreExtractionError("Judge response is empty or invalid")

        try:
            cleaned_text = judge_text.strip()

            # Handle cases where the LLM adds conversational fluff
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if not json_match:
                raise ScoreExtractionError("No JSON structure found in judge response")

            try:
                data = json.loads(json_match.group(0))
            except json.JSONDecodeError as e:
                raise ScoreExtractionError(f"Invalid JSON format in judge response: {str(e)}") from e

            if not isinstance(data, dict):
                raise ScoreExtractionError(f"Expected JSON object, got {type(data)}")

            # Validate required fields
            required_fields = ['verdict', 'evidence', 'confidence']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ScoreExtractionError(f"Missing required fields: {missing_fields}")

            # Convert verdict to boolean (case-insensitive)
            verdict_str = str(data['verdict']).upper()
            if verdict_str not in ['TRUE', 'FALSE']:
                raise ScoreExtractionError(f"Invalid verdict value: {data['verdict']}. Must be TRUE or FALSE")

            verdict = verdict_str == 'TRUE'

            # Validate confidence score
            try:
                confidence = float(data['confidence'])
                if not 0.0 <= confidence <= 1.0:
                    raise ScoreExtractionError("Confidence score must be between 0.0 and 1.0")
            except (TypeError, ValueError) as e:
                raise ScoreExtractionError(f"Invalid confidence score: {str(e)}") from e

            # Validate evidence
            evidence = str(data['evidence'])

            return verdict, evidence, confidence

        except ScoreExtractionError:
            raise
        except Exception as e:
            self.logger.error(f"Error parsing judge response: {str(e)}")
            return False, f"Parse error: {str(e)}", 0.0


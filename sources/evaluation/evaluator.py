"""
Unified evaluation system for Mimosa-AI workflows.
Combines WorkflowJudge and scenario-based Evaluator functionality.
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
from sources.evaluation.scenario_loader import ScenarioLoader
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

            self.judge_model = "deepseek/deepseek-chat"
            try:
                provider, model = self.judge_model.split("/", 1) if "/" in self.judge_model else ("openai", self.judge_model)
                self.llm_config = LLMConfig().from_dict({
                    "model": model,
                    "provider": provider,
                    "reasoning_effort": config.reasoning_effort
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
            raise WorkflowDataError(f"Workflow directory not found: {workflow_path}")

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

    def workflow_execution_text(self, uuid: str) -> str | None:
        """Generate workflow execution text for evaluation using WorkflowInfo.
        Args:
            uuid: UUID of the workflow
        Returns:
            Formatted workflow execution text or None if data cannot be loaded
        Raises:
            WorkflowDataError: If workflow data is invalid
        """
        try:
            workflow_info = self._load_workflow_data(uuid)
            
            state_result = workflow_info.state_result
            workflow_code = workflow_info.code
            goal = workflow_info.goal or "Goal not specified"
            
            if not state_result and not workflow_code:
                raise Exception(f"No workflow data available for {uuid}")

            result = workflow_info.answers

            return f"""You are evaluating a scientific workflow execution.
                   WORKFLOW GOAL:
                   {goal}
                   FULL WORKFLOW STATE RESULT (JSON):
                   {json.dumps(result, indent=2)}
                   WORKFLOW CODE:
                   ```python
                   {workflow_code or "# No workflow code available"}
                   ```"""
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


class GenericEvaluator(BaseEvaluator):
    """Evaluator for generic workflow evaluation using LLM judgment."""
    
    def __init__(self, config):
        """Initialize the GenericEvaluator.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.logger.info("GenericEvaluator initialized successfully")

    def evaluate(self, uuid: str, answer: str | None = None) -> None:
        """Perform generic evaluation of a workflow.
        
        Args:
            uuid: UUID of the workflow to evaluate
            answer: Optional expected answer for evaluation
            
        Raises:
            LLMEvaluationError: If LLM evaluation fails
            WorkflowDataError: If workflow data cannot be loaded
        """
        try:
            execution_text = self.workflow_execution_text(uuid)
            if not execution_text:
                raise WorkflowDataError(f"Cannot generate execution text for workflow {uuid}")

            prompt = f"""
    {execution_text}

    EVALUATION TASK:
    Based on the complete execution state and workflow code above, provide a score for each category.
    Give your score as a float on a scale of 0.0 to 1.0, where 0.0 means is a total failure, and 1.0 means a perfect execution.
    Analyse the full JSON state and workflow implementation to make your judgement.
    Please be objective, technical, and specific in your feedback.

    CRITERIA:
    1. GOAL ALIGNMENT
    Did the execution achieve the defined objective?
    Were all required steps completed?
    Were there unexpected deviations?

    2. AGENT COLLABORATION
    Did agents pass data correctly?
    Did agents handle failures gracefully?

    3. OUTPUT QUALITY
    Is the output complete?
    Is the output well-formatted?

    """ + (f"""4. ANSWER CORRECTNESS
    Answer should be : {answer}
    Is the answer factually correct?
    Is the answer logically consistent?
    Is the answer precise?
    """ if answer else "") + """

    Respond in this exact format:
    [
        {
            "category": "goal_alignment",
            "score": [0.0-1.0],
            "evidence": "[Specific JSON paths/logs proving objective fulfillment]"
        },
        {
            "category": "agent_collaboration", 
            "score": [0.0-1.0],
            "evidence": "[Message history snippets or error logs]"
        },
        {
            "category": "output_quality",
            "score": [0.0-1.0], 
            "evidence": "[Output validation errors or missing fields]"
        }""" + ("""
        ,{
            "category": "answer_correctness",
            "score": [0.0-1.0],
            "evidence": "[Factual accuracy verification]"
        }""" if answer else "") + """
    ]
    """

            self.logger.info(f"Evaluating workflow {uuid} with LLM judge")
            
            # Prepare memory path
            memory_path = Path(self.memory_dir) / uuid
            memory_path.mkdir(parents=True, exist_ok=True)

            # Call LLM with error handling
            try:
                llm_provider = LLMProvider(
                    "generic_judge",
                    memory_path,
                    system_msg=self._get_judge_system_prompt(),
                    config=self.llm_config,
                )
                output = llm_provider(prompt)
                
                if not output or not isinstance(output, str):
                    raise LLMEvaluationError("LLM returned empty or invalid response")

            except Exception as e:
                raise LLMEvaluationError(f"LLM evaluation failed: {str(e)}") from e

            # Save the evaluation to a file with error handling
            try:
                evaluation_path = self.workflow_dir / uuid / "evaluation.txt"
                evaluation_path.parent.mkdir(parents=True, exist_ok=True)
                with open(evaluation_path, "w", encoding='utf-8') as file:
                    file.write(output)
                self.logger.info(f"Evaluation completed for {uuid}. Results saved to: {evaluation_path}")
            except OSError as e:
                self.logger.error(f"Failed to save evaluation to file: {str(e)}")

            # Extract scores from the evaluation output
            try:
                scores = self._extract_scores(output)
                self._save_results(scores, uuid, 'generic')
            except ScoreExtractionError as e:
                self.logger.error(f"Failed to extract scores: {str(e)}")
                # Save empty scores to maintain consistency
                self._save_results({}, uuid, 'generic')

        except (WorkflowDataError, LLMEvaluationError):
            raise
        except Exception as e:
            raise LLMEvaluationError(f"Generic evaluation failed: {str(e)}") from e

    def _extract_scores(self, evaluation_text: str) -> dict[str, float]:
        """Extract scores from the evaluation text.

        Args:
            evaluation_text: The evaluation text containing the JSON scores

        Returns:
            Dictionary containing the extracted scores
            
        Raises:
            ScoreExtractionError: If scores cannot be extracted or are invalid
        """
        if not evaluation_text or not isinstance(evaluation_text, str):
            raise ScoreExtractionError("Evaluation text is empty or invalid")

        try:
            # Look for JSON array in the evaluation text
            cleaned_text = evaluation_text.strip()
            match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)

            if not match:
                raise ScoreExtractionError("No JSON array found in evaluation response")
            
            try:
                evaluations = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                raise ScoreExtractionError(f"Invalid JSON format in evaluation response: {str(e)}") from e
            
            if not isinstance(evaluations, list):
                raise ScoreExtractionError(f"Expected JSON array, got {type(evaluations)}")

            if not evaluations:
                raise ScoreExtractionError("Evaluation array is empty")
            # Validate evaluation structure
            required_fields = {'category', 'score', 'evidence'}
            valid_categories = {
                'goal_alignment',
                'agent_collaboration',
                'output_quality',
                'answer_correctness'
            }

            # Convert list of evaluations to a dictionary
            result = {}
            for i, eval_dict in enumerate(evaluations):
                if not isinstance(eval_dict, dict):
                    raise ScoreExtractionError(f"Evaluation entry {i} is not a dictionary")
                # Check required fields
                missing_fields = required_fields - set(eval_dict.keys())
                if missing_fields:
                    raise ScoreExtractionError(f"Missing fields {missing_fields} in evaluation entry {i}")
                # Validate category
                category = eval_dict['category']
                if not isinstance(category, str):
                    raise ScoreExtractionError(f"Category must be a string, got {type(category)} in entry {i}")
                if category not in valid_categories:
                    raise ScoreExtractionError(f"Invalid category '{category}' in entry {i}")
                # Validate score
                try:
                    score = float(eval_dict['score'])
                    if not 0.0 <= score <= 1.0:
                        raise ScoreExtractionError(f"Score must be between 0.0 and 1.0, got {score} for {category}")
                    result[category] = score
                except (TypeError, ValueError) as e:
                    raise ScoreExtractionError(f"Invalid score for {category}: {str(e)}") from e
                # Validate evidence
                if not isinstance(eval_dict['evidence'], str):
                    raise ScoreExtractionError(f"Evidence must be a string for {category}")
            # Calculate overall score
            score_keys = ['goal_alignment', 'agent_collaboration', 'output_quality']
            available_keys = [k for k in score_keys if k in result]
            
            if available_keys:
                result['overall_score'] = sum(result[k] for k in available_keys) / len(available_keys)
            else:
                self.logger.warning("No standard score categories found for overall score calculation")
            
            return result

        except ScoreExtractionError:
            raise
        except Exception as e:
            raise ScoreExtractionError(f"Unexpected error extracting scores: {str(e)}") from e


class ScenarioEvaluator(BaseEvaluator):
    """Evaluator for scenario-based workflow evaluation."""
    
    def __init__(self, config, scenarios_dir = "datasets/scenarios"):
        """Initialize the ScenarioEvaluator.
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        
        try:
            self.scenario_loader = ScenarioLoader(scenarios_dir=scenarios_dir)
        except Exception as e:
            raise EvaluatorError(f"Failed to initialize scenario loader: {str(e)}") from e
        
        self.logger.info("ScenarioEvaluator initialized successfully")

    def evaluate(self, uuid: str, scenario_id: str) -> dict[str, Any]:
        """Evaluate a workflow against a scenario with scoring.
        
        Args:
            uuid: UUID of the workflow to evaluate
            scenario_id: ID of the scenario to evaluate against
            
        Returns:
            Dictionary containing scenario evaluation results
            
        Raises:
            ScenarioError: If scenario evaluation fails
        """
        try:
            self.logger.info(f"Evaluating workflow {uuid} against scenario {scenario_id}")

            # Handle null uuid case
            if uuid is None:
                return {
                    'earned_points': 0,
                    'total_points': 0,
                    'score': 0,
                    'scenario_id': scenario_id
                }

            # Validate inputs
            if not isinstance(uuid, str) or not uuid.strip():
                raise ScenarioError("Invalid uuid: must be a non-empty string")
            
            if not isinstance(scenario_id, str) or not scenario_id.strip():
                raise ScenarioError("Invalid scenario_id: must be a non-empty string")

            # Load scenario with error handling
            try:
                scenario = self.scenario_loader.load_scenario(scenario_id)
                if not scenario:
                    raise ScenarioError(f"Scenario {scenario_id} not found or is empty")
                
                if not isinstance(scenario, dict):
                    raise ScenarioError(f"Scenario {scenario_id} is not a valid dictionary")

            except Exception as e:
                if isinstance(e, ScenarioError):
                    raise
                raise ScenarioError(f"Failed to load scenario {scenario_id}: {str(e)}") from e

            # Check if this is the new rubric format or legacy format
            is_rubric_format = "total_points" in scenario
            
            if is_rubric_format:
                return self._evaluate_rubric_format(uuid, scenario_id, scenario)
            else:
                return self._evaluate_legacy_format(uuid, scenario_id, scenario)

        except ScenarioError:
            raise
        except Exception as e:
            raise ScenarioError(f"Scenario evaluation failed: {str(e)}") from e

    def _evaluate_legacy_format(self, uuid: str, scenario_id: str, scenario: dict[str, Any]) -> dict[str, Any]:
        """Evaluate workflow using legacy assertion format.
        
        Args:
            uuid: UUID of the workflow
            scenario_id: ID of the scenario
            scenario: Scenario dictionary with assertions
            
        Returns:
            Dictionary containing evaluation results
        """
        if "assertions" not in scenario:
            raise ScenarioError(f"Scenario {scenario_id} missing 'assertions' field")
        
        assertions = scenario["assertions"]
        if not isinstance(assertions, list):
            raise ScenarioError(f"Scenario {scenario_id} 'assertions' must be a list")
        
        if not assertions:
            self.logger.warning(f"Scenario {scenario_id} has no assertions")

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
        passed_count = sum(1 for result in assertion_results if result.get("passed", False))
        total_count = len(assertion_results)
        score = passed_count / total_count if total_count > 0 else 0.0

        # Generate results
        results = {
            "scenario_id": scenario_id,
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
        
        # Return assertion metrics for DGM tracking
        return {
            'passed_assertions': passed_count,
            'total_assertions': total_count,
            'score': score,
            'scenario_id': scenario_id
        }
    
    def _evaluate_rubric_format(self, uuid: str, scenario_id: str, scenario: dict[str, Any]) -> dict[str, Any]:
        """Evaluate workflow using ScienceAgentBench rubric format.
        
        Args:
            uuid: UUID of the workflow
            scenario_id: ID of the scenario
            scenario: Scenario dictionary with rubric categories
            
        Returns:
            Dictionary containing evaluation results
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
            self.logger.warning(f"Scenario {scenario_id} has no rubric items")
        
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
                    # Partial credit based on confidence
                    confidence = result.get("confidence", 0.0)
                    earned = item.get("points", 0) * confidence
                
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
            "scenario_id": scenario_id,
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
        
        # Return metrics for DGM tracking
        return {
            'earned_points': total_earned,
            'total_points': total_possible_points,
            'score': score,
            'scenario_id': scenario_id
        }
    
    def _evaluate_rubric_item(self, uuid: str, item: dict[str, Any]) -> dict[str, Any]:
        """Evaluate single rubric item using LLM.
        
        Args:
            uuid: UUID of the workflow
            item: Rubric item dictionary with name, description, points, and category
            
        Returns:
            Dictionary containing rubric item evaluation results
            
        Raises:
            LLMEvaluationError: If rubric item evaluation fails
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
            uuid: UUID of the workflow
            item: Rubric item dictionary
            
        Returns:
            Formatted judge prompt string
            
        Raises:
            WorkflowDataError: If workflow execution text cannot be generated
        """
        try:
            execution_text = self.workflow_execution_text(uuid)
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
        """Evaluate single assertion using existing LLM prompt format.
        Args:
            uuid: UUID of the workflow
            assertion: Assertion dictionary to evaluate
        Returns:
            Dictionary containing assertion evaluation results
        Raises:
            LLMEvaluationError: If assertion evaluation fails
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
            uuid: UUID of the workflow
            assertion: Assertion dictionary
            
        Returns:
            Formatted judge prompt string
            
        Raises:
            WorkflowDataError: If workflow execution text cannot be generated
        """
        try:
            execution_text = self.workflow_execution_text(uuid)
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
            judge_text: Raw response text from LLM
            
        Returns:
            Tuple of (verdict, evidence, confidence)
            
        Raises:
            ScoreExtractionError: If response cannot be parsed
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

    def evaluate(self, uuid: str, answer: str = None, scenario_id: str = None) -> dict[str, Any]:
        """Evaluate the workflow results.

        Args:
            uuid: UUID of the workflow run to evaluate
            answer: Optional expected answer for evaluation
            scenario_id: Optional scenario ID for scenario-based evaluation
            
        Returns:
            Dictionary containing evaluation results
            
        Raises:
            EvaluatorError: If evaluation fails
        """
        try:
            # Validate inputs
            if not uuid or not isinstance(uuid, str):
                raise EvaluatorError("Invalid uuid: must be a non-empty string")

            # If scenario_id is provided, use scenario-based evaluation
            if scenario_id:
                return self.scenario_evaluator.evaluate(uuid, scenario_id)
            else:
                self.generic_evaluator.evaluate(uuid, answer)
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
        result = evaluator.evaluate(mock_uuid, scenario_id="clintox_nn_rubric")
        print(f"✓ Scenario evaluation completed: {result}")
    except Exception as e:
        print(f"⚠ Unexpected error in scenario evaluation: {e}")

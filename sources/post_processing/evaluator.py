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

from sources.core.llm_provider import LLMConfig, LLMProvider
from sources.utils.scenario_loader import ScenarioLoader


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


class WorkflowEvaluator:
    """Combined workflow evaluator with both judge and scenario-based evaluation capabilities."""

    def __init__(self, config):
        """Initialize the WorkflowEvaluator with configuration.
        
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

            try:
                self.scenario_loader = ScenarioLoader()
            except Exception as e:
                raise EvaluatorError(f"Failed to initialize scenario loader: {str(e)}") from e

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
            self.logger.info("WorkflowEvaluator initialized successfully")

        except Exception as e:
            if isinstance(e, EvaluatorError):
                raise
            raise EvaluatorError(f"Failed to initialize WorkflowEvaluator: {str(e)}") from e
    
    def _load_workflow_data(self, workflow_id: str) -> tuple[dict, str]:
        """Load workflow execution data from UUID folder.
        
        Args:
            workflow_id: UUID of the workflow to load
            
        Returns:
            Tuple of (state_result, workflow_code)
            
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

        state_result = None
        workflow_code = None

        state_result_path = workflow_path / "state_result.json"
        try:
            if not state_result_path.exists():
                self.logger.warning(f"state_result.json not found at {state_result_path}")
            else:
                with open(state_result_path, encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        self.logger.warning(f"state_result.json is empty at {state_result_path}")
                    else:
                        try:
                            state_result = json.loads(content)
                            if not isinstance(state_result, dict):
                                raise WorkflowDataError(f"state_result.json must contain a JSON object, got {type(state_result)}")
                        except json.JSONDecodeError as e:
                            raise WorkflowDataError(f"Invalid JSON in state_result.json: {str(e)}") from e
        except PermissionError as e:
            raise WorkflowDataError(f"Permission denied reading state_result.json: {str(e)}") from e
        except OSError as e:
            raise WorkflowDataError(f"OS error reading state_result.json: {str(e)}") from e

        workflow_code_path = workflow_path / f"workflow_code_{workflow_id}.py"
        try:
            if not workflow_code_path.exists():
                self.logger.warning(f"Workflow code not found at {workflow_code_path}")
            else:
                with open(workflow_code_path, encoding='utf-8') as f:
                    workflow_code = f.read()
                    if not workflow_code.strip():
                        self.logger.warning(f"Workflow code file is empty at {workflow_code_path}")
        except PermissionError as e:
            raise WorkflowDataError(f"Permission denied reading workflow code: {str(e)}") from e
        except OSError as e:
            raise WorkflowDataError(f"OS error reading workflow code: {str(e)}") from e

        # Validate that we have at least some data
        if state_result is None and workflow_code is None:
            raise WorkflowDataError(f"No valid workflow data found for {workflow_id}")

        return state_result, workflow_code

    def workflow_execution_text(self, uuid: str) -> str | None:
        """Generate workflow execution text for evaluation.
        Args:
            uuid: UUID of the workflow
        Returns:
            Formatted workflow execution text or None if data cannot be loaded
        Raises:
            WorkflowDataError: If workflow data is invalid
        """
        try:
            
            state_result, workflow_code = self._load_workflow_data(uuid)
            
            if not state_result and not workflow_code:
                self.logger.error(f"No workflow data available for {uuid}")
                return None

            # Extract goal with fallback
            goal = "Goal not specified"
            if state_result and isinstance(state_result, dict):
                goal = state_result.get("goal", goal)
                # Remove evaluation part for clarity (create a copy to avoid modifying original)
                state_result_copy = state_result.copy()
                state_result_copy.pop("evaluation", None)
            else:
                state_result_copy = {}

            # Format the execution text (preserving original prompt format)
            return f"""You are evaluating a scientific workflow execution.
                   WORKFLOW GOAL:
                   {goal}
                   FULL WORKFLOW STATE RESULT (JSON):
                   {json.dumps(state_result_copy, indent=2)}
                   WORKFLOW CODE:
                   ```python
                   {workflow_code or "# No workflow code available"}
                   ```"""
        except Exception as e:
            return f"""
            Worflow generation failed due to error:
            {str(e)}
            """

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
                return self.scenario(uuid, scenario_id)
            else:
                self.generic(uuid, answer)
                return {'evaluation_type': 'generic', 'uuid': uuid}

        except (WorkflowDataError, ScenarioError, LLMEvaluationError) as _:
            # Re-raise specific evaluator errors
            raise
        except Exception as e:
            raise EvaluatorError(f"Evaluation failed for {uuid}: {str(e)}") from e

    def generic(self, uuid: str, answer: str | None) -> None:
        """Perform generic evaluation of a workflow.
        
        Args:
            uuid: UUID of the workflow to evaluate
            answer: Optional expected answer for evaluation
            
        Raises:
            LLMEvaluationError: If LLM evaluation fails
            WorkflowDataError: If workflow data cannot be loaded
        """
        try:
            # Get workflow execution text
            execution_text = self.workflow_execution_text(uuid)
            if not execution_text:
                raise WorkflowDataError(f"Cannot generate execution text for workflow {uuid}")

            # Build evaluation prompt (preserving original format exactly)
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

    """ + ("""4. ANSWER CORRECTNESS
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
                # Don't raise here as the evaluation itself succeeded

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

    def scenario(self, uuid: str, scenario_id: str) -> dict[str, Any]:
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
                    'passed_assertions': 0,
                    'total_assertions': 0,
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
                
                if "assertions" not in scenario:
                    raise ScenarioError(f"Scenario {scenario_id} missing 'assertions' field")
                
                assertions = scenario["assertions"]
                if not isinstance(assertions, list):
                    raise ScenarioError(f"Scenario {scenario_id} 'assertions' must be a list")
                
                if not assertions:
                    self.logger.warning(f"Scenario {scenario_id} has no assertions")

            except Exception as e:
                if isinstance(e, ScenarioError):
                    raise
                raise ScenarioError(f"Failed to load scenario {scenario_id}: {str(e)}") from e

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
                # Don't raise here as the evaluation itself succeeded
            
            # Return assertion metrics for DGM tracking
            return {
                'passed_assertions': passed_count,
                'total_assertions': total_count,
                'score': score,
                'scenario_id': scenario_id
            }

        except ScenarioError:
            raise
        except Exception as e:
            raise ScenarioError(f"Scenario evaluation failed: {str(e)}") from e

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

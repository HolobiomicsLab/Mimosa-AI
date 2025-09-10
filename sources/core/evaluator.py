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


class WorkflowEvaluator:
    """Combined workflow evaluator with both judge and scenario-based evaluation capabilities."""

    def __init__(self, config):
        # Initialize with configuration from both original classes
        self.memory_dir = Path(config.memory_dir)
        self.workflow_dir = Path(config.workflow_dir)
        self.model_pricing = config.model_pricing

        # Initialize scenario loader and LLMProvider for scenario-based evaluation
        self.scenario_loader = ScenarioLoader()
        self.judge_model = "gpt-4o-mini"  # Default model, using gpt-4o-mini
        self.llm_config = LLMConfig().from_dict({
            "model": self.judge_model,
            "reasoning_effort": config.reasoning_effort
        })
        self.logger = logging.getLogger(__name__)
    
    def _load_workflow_data(self, workflow_id: str) -> dict[dict, str]:
        """Load workflow execution data from UUID folder."""
        workflow_path = Path(self.workflow_dir) / workflow_id

        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow {workflow_id} not found")

        # Load state_result.json
        state_result_path = workflow_path / "state_result.json"
        try:
            with open(state_result_path) as f:
                state_result = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load state_result.json: {e}")

        # Load workflow code
        workflow_code_path = workflow_path / f"workflow_code_{workflow_id}.py"
        try:
            with open(workflow_code_path) as f:
                workflow_code = f.read()
        except Exception as e:
            print(f"Warning: Could not load workflow code: {e}")

        return state_result, workflow_code

    def workflow_execution_text(self, uuid:str):
        state_result, workflow_code = self._load_workflow_data(uuid)
        goal = state_result.get("goal", "Goal not specified")
        state_result.pop("evaluation", None)  # Remove evaluation part for clarity
        return f"""You are evaluating a scientific workflow execution.

        WORKFLOW GOAL:
        {goal}

        FULL WORKFLOW STATE RESULT (JSON):
        {json.dumps(state_result, indent=2)}

        WORKFLOW CODE:
        ```python
        {workflow_code}
        ```"""

    def evaluate(self, uuid: str, answer: str = None, scenario_id: str = None) -> str:
        """Evaluate the workflow results.

        Args:
            uuid: UUID of the workflow run to evaluate
            short: Whether to use short evaluation format
            answer: Optional expected answer for evaluation
            scenario_id: Optional scenario ID for scenario-based evaluation
        """
        # If scenario_id is provided, use scenario-based evaluation
        if scenario_id:
            return self.scenario(uuid, scenario_id)
        else:
            self.generic(uuid, answer)
            return {'evaluation_type': 'generic'}

        
    def generic(self,uuid, answer):
        prompt = f"""
    {self.workflow_execution_text(uuid)}

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
        memory_path = Path(self.memory_dir) / uuid
        output = LLMProvider(
            "generic_judge",
            memory_path,
            system_msg=self._get_judge_system_prompt(),
            config=self.llm_config,
        )(prompt)

        # Save the evaluation to a file
        evaluation_path = self.workflow_dir / uuid / "evaluation.txt"
        with open(evaluation_path, "w") as file:
            file.write(output)
        self.logger.info(
            f"Evaluation completed for {uuid}. Results saved to: {evaluation_path}"
        )

        # Extract scores from the evaluation output
        scores = self._extract_scores(output)

        self._save_results(scores, uuid, 'generic')


    def _extract_scores(self, evaluation_text):
        """Extract scores from the evaluation text.

        Args:
            evaluation_text: The evaluation text containing the JSON scores

        Returns:
            dict: The extracted scores or empty dict if not found
        """
        try:
            # Look for JSON array in the evaluation text
            # This pattern matches both JSON arrays and objects with or without code blocks
            cleaned_text = evaluation_text.strip()
            match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)

            if not match:
                raise ValueError("No JSON array found in response")
            
            try:
                evaluations = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {str(e)}") from e
            
            required_fields = {'category', 'score', 'evidence'}
            valid_categories = {
                'goal_alignment',
                'agent_collaboration',
                'output_quality',
                'answer_correctness'
            }

            # Convert list of evaluations to a dictionary
            result = {}
            
            for eval_dict in evaluations:
                # Check required fields
                missing_fields = required_fields - set(eval_dict.keys())
                if missing_fields:
                    raise ValueError(f"Missing fields {missing_fields} in evaluation entry")
                
                # Validate category
                category = eval_dict['category']
                if category not in valid_categories:
                    raise ValueError(f"Invalid category '{category}'")
                
                # Validate score
                try:
                    score = float(eval_dict['score'])
                    if not 0.0 <= score <= 1.0:
                        raise ValueError("Score must be between 0.0 and 1.0")
                    # Add to result dictionary
                    result[category] = score
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Invalid score for {category}: {str(e)}") from e

            # Calculate overall score
            score_keys = ['goal_alignment', 'agent_collaboration', 'output_quality']
            available_keys = [k for k in score_keys if k in result]
            
            if available_keys:
                result['overall_score'] = sum(result[k] for k in available_keys) / len(available_keys)
            
            return result
        except Exception as e:
            print(f"❌ Error extracting scores: {str(e)}")
            return {}

    def _save_results(self, scores, uuid: str, type:str):
        """Update the state result file with the evaluation scores.

        Args:
            scores: The scores to add to the state result
            uuid: UUID of the workflow run
        """
        try:
            workflow_path = Path(self.workflow_dir) / uuid
            state_result_path = workflow_path / "state_result.json"

            # Load existing state result if it exists
            try:
                with open(state_result_path) as f:
                    state_result = json.load(f)
            except FileNotFoundError:
                print(f"⚠️ State result file not found for UUID {uuid}")
                return

            # Add scores to state result
            state_result.setdefault("evaluation", {})[type] = scores

            # Write updated state result back to file
            with open(state_result_path, "w") as f:
                json.dump(state_result, f, indent=2)

            print("Scores extracted and saved to state result.")

        except Exception as e:
            raise(f"❌ Error updating state result: {str(e)}") from e

    # Methods from scenario-based evaluator
    def scenario(self, uuid: str, scenario_id: str):
        """Evaluate a workflow against a scenario with scoring."""
        print(f"Evaluating workflow {uuid} against scenario {scenario_id}")

        # Load scenario and workflow data
        scenario = self.scenario_loader.load_scenario(scenario_id)
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")

        #workflow_data = self._load_workflow_data(workflow_id)

        # Evaluate all assertions
        assertion_results = []
        for assertion in scenario["assertions"]:
            result = self._evaluate_assertion(uuid, assertion)
            assertion_results.append(result)

        # Calculate score (only partial score)
        passed_count = sum(1 for result in assertion_results if result["passed"])
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

        # Save results
        self._save_results(results, uuid, 'scenario')
        
        # Return assertion metrics for DGM tracking
        return {
            'passed_assertions': passed_count,
            'total_assertions': total_count,
            'score': score,
            'scenario_id': scenario_id
        }


    def _evaluate_assertion(
        self, uuid: str, assertion: dict
    ) -> dict[str, Any]:
        """Evaluate single assertion using existing LLM prompt format."""
        # Build judge prompt using existing format
        judge_prompt = self._build_judge_prompt(uuid, assertion)

        try:
            # Use LLMProvider instead of direct OpenAI call
            llm_provider = LLMProvider(
                agent_name=f"scenario_judge_{assertion['id']}",
                memory_path=self.memory_dir / uuid,
                system_msg=self._get_judge_system_prompt(),
                config=self.llm_config,
            )

            # Call LLMProvider with the judge prompt
            judge_text = llm_provider(judge_prompt).strip()
            passed, evidence, confidence = self._parse_judge_response(judge_text)

            return {
                "id": assertion["id"],
                "description": assertion["description"],
                "passed": passed,
                "evidence": evidence,
                "confidence": confidence,
            }

        except Exception as e:
            print(f"Error evaluating assertion {assertion['id']}: {e}")
            return {
                "id": assertion["id"],
                "description": assertion["description"],
                "passed": False,
                "evidence": f"Evaluation error: {str(e)}",
                "confidence": 0.0,
            }

    def _build_judge_prompt(
        self, uuid: str, assertion: dict
    ) -> str:
        """Build judge prompt with workflow data."""
        criteria = assertion.get("evaluation_criteria", "Standard evaluation")

        return f"""
{self.workflow_execution_text(uuid)}

ASSERTION TO EVALUATE:
Description: {assertion["description"]}
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

    def _get_judge_system_prompt(self) -> str:
        """Get system prompt for LLM judge (keeping existing format)."""
        prompt = """You are an expert scientific researcher and rigorous multi-agent system evaluator. Your task is to assess whether a computational workflow achieved its intended goals through coordinated agent collaboration, while ensuring scientific validity and technical correctness.

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
        return prompt

    def _parse_judge_response(self, judge_text: str) -> tuple[bool, str, float]:
        """Parse LLM judge response from JSON format."""
        try:       
            cleaned_text = judge_text.strip()
    
            # Handle cases where the LLM adds conversational fluff
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON structure found in response")
            
            try:
                data = json.loads(json_match.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {str(e)}") from e
            
            # Validate required fields
            required_fields = ['verdict', 'evidence', 'confidence']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Convert verdict to boolean (case-insensitive)
            verdict_str = str(data['verdict']).upper()
            if verdict_str not in ['TRUE', 'FALSE']:
                raise ValueError(f"Invalid verdict value: {data['verdict']}. Must be TRUE or FALSE")
            
            verdict = verdict_str == 'TRUE'
            
            # Validate confidence score
            try:
                confidence = float(data['confidence'])
                if not 0.0 <= confidence <= 1.0:
                    raise ValueError("Confidence score must be between 0.0 and 1.0")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid confidence score: {str(e)}") from e
            
            return verdict, data['evidence'], confidence

        except Exception as e:
            print(f"Error parsing judge response: {e}")
            return False, f"Parse error: {str(e)}", 0.0
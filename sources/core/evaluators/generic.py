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
from sources.cli.pretty_print import (
    print_box,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)

from .base import *
from .grounding import get_perspicacite_grounding

class GenericEvaluator(BaseEvaluator):
    """Evaluator for generic workflow evaluation using LLM judgment."""

    def __init__(self, config):
        """Initialize the GenericEvaluator.

        Args:
            config: Configuration object
        """
        super().__init__(config)
        self.logger.info("GenericEvaluator initialized successfully")

    def _evaluate_single_criterion(self, uuid: str, execution_text: str, category: str,
                                    criterion_prompt: str) -> dict[str, Any]:
        """Evaluate a single criterion with a fresh LLM context to avoid inter-criteria bias.

        Args:
            uuid: UUID of the workflow to evaluate
            execution_text: The workflow execution text
            category: The category name (e.g. 'goal_alignment')
            criterion_prompt: The criterion-specific prompt section

        Returns:
            Dictionary with 'category', 'score', and 'evidence'

        Raises:
            LLMEvaluationError: If LLM evaluation fails
        """
        prompt = f"""
You are evaluating AI agent(s) performance on a computational task.
{execution_text}

EVALUATION TASK:
Based on the complete execution state and workflow code above, provide a score for the following criterion ONLY.
Give your score as a float on a scale of 0.0 to 1.0, where 0.0 means total failure, and 1.0 means perfect execution.
Analyse the full JSON state and workflow implementation to make your judgement.
Please be objective, technical, and specific in your feedback.

CRITERION: {category.upper().replace('_', ' ')}
{criterion_prompt}

Respond in this exact JSON format:
{{
    "category": "{category}",
    "score": [0.0-1.0],
    "evidence": "[Specific evidence supporting your score]"
}}
"""
        try:
            memory_path = Path(self.memory_dir) / uuid
            memory_path.mkdir(parents=True, exist_ok=True)

            llm_provider = LLMProvider(
                f"generic_judge_{category}",
                memory_path,
                system_msg=self._get_judge_system_prompt(),
                config=self.llm_config,
            )
            output = llm_provider(prompt)

            if not output or not isinstance(output, str):
                raise LLMEvaluationError(f"LLM returned empty or invalid response for {category}")

            return self._extract_single_score(output, category)

        except LLMEvaluationError:
            raise
        except Exception as e:
            raise LLMEvaluationError(f"LLM evaluation failed for {category}: {str(e)}") from e

    def _evaluate_goal_alignment(self, uuid: str, execution_text: str, litterature_grounding: str) -> dict[str, Any]:
        """Evaluate goal alignment criterion with a fresh LLM context.

        Args:
            uuid: UUID of the workflow to evaluate
            execution_text: The workflow execution text

        Returns:
            Dictionary with 'category', 'score', and 'evidence'
        """
        criterion_prompt = f"""Evaluate whether the multi-agent workflow achieved its stated scientific/research objective.

According to retrieved scientific litterature the task can be completed following these steps:
{litterature_grounding}
(ignore the litterature if irrelevant or failed to retrieve useful informations)

Consider the following:
- Did the orchestrated sequence of agents collectively fulfill the workflow's goal?
- Were all required processing stages (data retrieval, analysis, synthesis) completed by the assigned agents?
- Did any agent deviate from the expected plan, skip critical steps, or pursue irrelevant sub-tasks?
- If the goal required multiple sub-objectives, did the agent pipeline address all of them?
- Was the final answer produced by the last agent a direct and complete response to the original goal?"""
        return self._evaluate_single_criterion(uuid, execution_text, "goal_alignment", criterion_prompt)

    def _evaluate_agent_collaboration(self, uuid: str, execution_text: str) -> dict[str, Any]:
        """Evaluate agent collaboration criterion with a fresh LLM context.

        Args:
            uuid: UUID of the workflow to evaluate
            execution_text: The workflow execution text

        Returns:
            Dictionary with 'category', 'score', and 'evidence'
        """
        criterion_prompt = """Evaluate how effectively agents collaborated within the multi-agent workflow.

Consider the following:
- Was data passed between agents efficiently and without unnecessary loss of information?
- Would a downstream agent lack critical context or information due to poor handoff from an upstream agent?
- How well did the transitions between agents work? Were outputs from one agent well-suited as inputs for the next?
- Were errors encountered by any agent reported clearly and propagated correctly to subsequent agents?
- Did agents avoid redundant work, or did multiple agents repeat the same computation unnecessarily?
- If an agent failed or produced a partial result, did the next agent handle the degraded input gracefully?
- Was the overall agent orchestration logical, or were there bottlenecks, circular dependencies, or wasted steps?"""
        return self._evaluate_single_criterion(uuid, execution_text, "agent_collaboration", criterion_prompt)

    def _evaluate_output_quality(self, uuid: str, execution_text: str) -> dict[str, Any]:
        """Evaluate output quality criterion with a fresh LLM context.

        Args:
            uuid: UUID of the workflow to evaluate
            execution_text: The workflow execution text

        Returns:
            Dictionary with 'category', 'score', and 'evidence'
        """
        criterion_prompt = """Evaluate the quality and usability of the final output produced by the multi-agent workflow.

Consider the following:
- Is the final output complete, addressing all aspects of the original goal?
- Is the output well-structured, clearly formatted, and ready for consumption (e.g., valid data formats, readable text)?
- Does the output contain errors, inconsistencies, or artifacts from intermediate agent processing?
- Are results presented with appropriate precision, units, and context?
- If the workflow was supposed to produce files, plots, or structured data, were they generated correctly?
- Is the output self-contained enough to be understood without needing to re-read the full execution trace?
- Are error messages or warnings, if any, clear and actionable?"""
        return self._evaluate_single_criterion(uuid, execution_text, "output_quality", criterion_prompt)

    def _evaluate_answer_plausibility(self, uuid: str, execution_text: str,
                                       litterature_grounding: str) -> dict[str, Any]:
        """Evaluate answer plausibility criterion with a fresh LLM context,
        using scientific literature grounding from Perspicacite.

        Args:
            uuid: UUID of the workflow to evaluate
            execution_text: The workflow execution text
            litterature_grounding: Scientific literature grounding from Perspicacite

        Returns:
            Dictionary with 'category', 'score', and 'evidence'
        """
        prompt = f"""
You are evaluating AI agent(s) performance on a computational task.
{execution_text}

According to scientific literature grounding, the following information is relevant to assessing the workflow execution:
{litterature_grounding}

EVALUATION TASK:
Based on the complete execution state, workflow code, and the scientific literature grounding above,
provide a score for the following criterion ONLY.
Give your score as a float on a scale of 0.0 to 1.0, where 0.0 means total failure, and 1.0 means perfect execution.
Please be objective, technical, and specific in your feedback.
Use the literature grounding to assess whether the answer is scientifically supported.

CRITERION: ANSWER PLAUSIBILITY
Does the answer appear plausible given the workflow trace and available evidence?
Is the answer logically consistent?
Is the answer appropriately qualified rather than overstated?
Is the answer consistent with or contradicted by the scientific literature evidence provided?

Respond in this exact JSON format:
{{
    "category": "answer_plausibility",
    "score": [0.0-1.0],
    "evidence": "[Why the answer seems plausible or implausible based on both the execution trace and the scientific literature grounding]"
}}
"""
        try:
            memory_path = Path(self.memory_dir) / uuid
            memory_path.mkdir(parents=True, exist_ok=True)

            llm_provider = LLMProvider(
                "generic_judge_answer_plausibility",
                memory_path,
                system_msg=self._get_judge_system_prompt(),
                config=self.llm_config,
            )
            output = llm_provider(prompt)

            if not output or not isinstance(output, str):
                raise LLMEvaluationError("LLM returned empty or invalid response for answer_plausibility")

            return self._extract_single_score(output, "answer_plausibility")

        except LLMEvaluationError:
            raise
        except Exception as e:
            raise LLMEvaluationError(f"LLM evaluation failed for answer_plausibility: {str(e)}") from e

    def _extract_single_score(self, evaluation_text: str, expected_category: str) -> dict[str, Any]:
        """Extract a single criterion score from an LLM evaluation response.

        Args:
            evaluation_text: The evaluation text containing the JSON score
            expected_category: The expected category name

        Returns:
            Dictionary with 'category', 'score', and 'evidence'

        Raises:
            ScoreExtractionError: If the score cannot be extracted or is invalid
        """
        if not evaluation_text or not isinstance(evaluation_text, str):
            raise ScoreExtractionError(f"Evaluation text is empty or invalid for {expected_category}")

        try:
            cleaned_text = evaluation_text.strip()
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)

            if not json_match:
                raise ScoreExtractionError(f"No JSON object found in evaluation response for {expected_category}")

            try:
                data = json.loads(json_match.group(0))
            except json.JSONDecodeError as e:
                raise ScoreExtractionError(
                    f"Invalid JSON format in evaluation response for {expected_category}: {str(e)}"
                ) from e

            if not isinstance(data, dict):
                raise ScoreExtractionError(f"Expected JSON object, got {type(data)} for {expected_category}")

            # Validate required fields
            required_fields = {'category', 'score', 'evidence'}
            missing_fields = required_fields - set(data.keys())
            if missing_fields:
                raise ScoreExtractionError(
                    f"Missing fields {missing_fields} in evaluation response for {expected_category}"
                )

            # Validate and normalize category
            category = data['category']
            if category == 'answer_correctness':
                category = 'answer_plausibility'

            # Validate score
            try:
                score = float(data['score'])
                if not 0.0 <= score <= 1.0:
                    raise ScoreExtractionError(
                        f"Score must be between 0.0 and 1.0, got {score} for {expected_category}"
                    )
            except (TypeError, ValueError) as e:
                raise ScoreExtractionError(f"Invalid score for {expected_category}: {str(e)}") from e

            # Validate evidence
            if not isinstance(data['evidence'], str):
                raise ScoreExtractionError(f"Evidence must be a string for {expected_category}")

            return {
                'category': expected_category,
                'score': score,
                'evidence': data['evidence'],
            }

        except ScoreExtractionError:
            raise
        except Exception as e:
            raise ScoreExtractionError(
                f"Unexpected error extracting score for {expected_category}: {str(e)}"
            ) from e

    def evaluate(self, uuid: str, agent_answers: str | None = None) -> None:
        """Perform generic evaluation of a workflow.

        Each evaluation criterion is assessed independently with a fresh LLM context
        to avoid inter-criteria bias. The answer_plausibility criterion additionally
        uses scientific literature grounding from Perspicacite.

        Args:
            uuid: UUID of the workflow to evaluate
            agent_answers: Optional list of answers from agents for evaluation

        Raises:
            LLMEvaluationError: If LLM evaluation fails
            WorkflowDataError: If workflow data cannot be loaded
        """
        litterature_grounding = "No grounding available."
        try:
            execution_text, success = self.workflow_execution_text(uuid)
            if not execution_text:
                raise WorkflowDataError(f"Cannot generate execution text for workflow {uuid}")

            if success:
                litterature_grounding = self.get_perspicacite_grounding(execution_text)
                print_box(litterature_grounding, title=f"Perspicacite scientific grounding", color=GREEN)

            self.logger.info(f"Evaluating workflow {uuid} with independent LLM judges per criterion")

            # Evaluate each criterion independently with a fresh LLM context
            criteria_results = []
            all_outputs = []

            # 1. Goal Alignment
            try:
                result = self._evaluate_goal_alignment(uuid, execution_text, litterature_grounding)
                criteria_results.append(result)
                all_outputs.append(f"[GOAL ALIGNMENT] score={result['score']:.2f}\n{result['evidence']}")
                self.logger.info(f"Goal alignment evaluated: {result['score']:.2f}")
            except (LLMEvaluationError, ScoreExtractionError) as e:
                self.logger.error(f"Failed to evaluate goal_alignment: {str(e)}")
                criteria_results.append({'category': 'goal_alignment', 'score': 0.0, 'evidence': f'Evaluation failed: {str(e)}'})
                all_outputs.append(f"[GOAL ALIGNMENT] FAILED: {str(e)}")

            # 2. Agent Collaboration
            try:
                result = self._evaluate_agent_collaboration(uuid, execution_text)
                criteria_results.append(result)
                all_outputs.append(f"[AGENT COLLABORATION] score={result['score']:.2f}\n{result['evidence']}")
                self.logger.info(f"Agent collaboration evaluated: {result['score']:.2f}")
            except (LLMEvaluationError, ScoreExtractionError) as e:
                self.logger.error(f"Failed to evaluate agent_collaboration: {str(e)}")
                criteria_results.append({'category': 'agent_collaboration', 'score': 0.0, 'evidence': f'Evaluation failed: {str(e)}'})
                all_outputs.append(f"[AGENT COLLABORATION] FAILED: {str(e)}")

            # 3. Output Quality
            try:
                result = self._evaluate_output_quality(uuid, execution_text)
                criteria_results.append(result)
                all_outputs.append(f"[OUTPUT QUALITY] score={result['score']:.2f}\n{result['evidence']}")
                self.logger.info(f"Output quality evaluated: {result['score']:.2f}")
            except (LLMEvaluationError, ScoreExtractionError) as e:
                self.logger.error(f"Failed to evaluate output_quality: {str(e)}")
                criteria_results.append({'category': 'output_quality', 'score': 0.0, 'evidence': f'Evaluation failed: {str(e)}'})
                all_outputs.append(f"[OUTPUT QUALITY] FAILED: {str(e)}")

            # 4. Answer Plausibility (with literature grounding)
            try:
                result = self._evaluate_answer_plausibility(uuid, execution_text, litterature_grounding)
                criteria_results.append(result)
                all_outputs.append(f"[ANSWER PLAUSIBILITY] score={result['score']:.2f}\n{result['evidence']}")
                self.logger.info(f"Answer plausibility evaluated: {result['score']:.2f}")
            except (LLMEvaluationError, ScoreExtractionError) as e:
                self.logger.error(f"Failed to evaluate answer_plausibility: {str(e)}")
                criteria_results.append({'category': 'answer_plausibility', 'score': 0.0, 'evidence': f'Evaluation failed: {str(e)}'})
                all_outputs.append(f"[ANSWER PLAUSIBILITY] FAILED: {str(e)}")

            # Aggregate scores
            scores = {}
            for cr in criteria_results:
                scores[cr['category']] = cr['score']

            # Calculate overall score (excluding answer_plausibility, same as before)
            score_keys = ['goal_alignment', 'agent_collaboration', 'output_quality', 'answer_plausibility']
            available_keys = [k for k in score_keys if k in scores]
            if available_keys:
                scores['overall_score'] = sum(scores[k] for k in available_keys) / len(available_keys)
            else:
                self.logger.warning("No standard score categories found for overall score calculation")

            # Save the combined evaluation to a file
            try:
                evaluation_path = self.workflow_dir / uuid / "evaluation.txt"
                evaluation_path.parent.mkdir(parents=True, exist_ok=True)
                with open(evaluation_path, "w", encoding='utf-8') as file:
                    file.write("Generic Evaluation (Independent Criteria)\n")
                    file.write("=" * 60 + "\n\n")
                    for cr in criteria_results:
                        file.write(f"[{cr['category'].upper()}] Score: {cr['score']:.2f}\n")
                        file.write(f"  Evidence: {cr['evidence']}\n\n")
                    if 'overall_score' in scores:
                        file.write(f"Overall Score: {scores['overall_score']:.2f}\n")
                self.logger.info(f"Evaluation completed for {uuid}. Results saved to: {evaluation_path}")
            except OSError as e:
                self.logger.error(f"Failed to save evaluation to file: {str(e)}")

            # Save scores
            try:
                self._save_results(scores, uuid, 'generic')
            except ScoreExtractionError as e:
                self.logger.error(f"Failed to extract scores: {str(e)}")
                self._save_results({}, uuid, 'generic')

        except (WorkflowDataError, LLMEvaluationError):
            raise
        except Exception as e:
            raise LLMEvaluationError(f"Generic evaluation failed: {str(e)}") from e
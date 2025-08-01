"""
Unified evaluation system for Mimosa-AI workflows.
Combines WorkflowJudge and scenario-based Evaluator functionality.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from sources.core.llm_provider import LLMConfig, LLMProvider
from sources.evaluation.scenario_loader import ScenarioLoader


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
        self.llm_config = LLMConfig().from_dict({"model": self.judge_model})
        self.logger = logging.getLogger(__name__)

    def generate_text(self, uuid: str) -> str:
        """Generate formatted text for evaluation.

        Args:
            uuid: UUID of the workflow run
        """
        # Set paths with the provided UUID
        memory_path = Path(self.memory_dir) / uuid
        workflow_path = Path(self.workflow_dir) / uuid

        # Collect agent information
        agents = []
        workflow_steps = []

        for file in os.listdir(memory_path):
            if file.endswith(".json") and file.startswith("task_"):
                agent_name = (
                    file.removeprefix("task_").removesuffix(".json").replace("_", " ")
                )

                with open(os.path.join(memory_path, file)) as f:
                    steps = json.load(f)
                    start_time = int(steps[0]["timing"].get("start_time", ""))
                    user_task = steps[0]["model_input_messages"][1]["content"][0].get(
                        "text", ""
                    )

                    # Add agent to the list
                    agents.append(
                        {
                            "name": agent_name,
                            "task": user_task,
                            "start_time": start_time,
                        }
                    )

                    # Process steps for workflow
                    for step in steps:
                        error = step.get("error", None)
                        code_result = error if error else step.get("observations", "")
                        step_info = {
                            "agent": agent_name,
                            "start_time": step["timing"]["start_time"],
                            "step_number": step["step_number"],
                            "prompt": step["model_input_messages"][1]["content"][0][
                                "text"
                            ],
                            "reasoning": step["model_output"],
                            "code_action": step.get("code_action", ""),
                            "code_output": code_result,
                        }
                        workflow_steps.append(step_info)

        # Sort agents and workflow steps by start time
        workflow_steps.sort(key=lambda x: (x["start_time"], x["step_number"]))

        # Read the goal
        with open(workflow_path / "state_result.json") as f:
            goal = json.load(f).get("goal", "")

        # Format the text according to the new structure
        text = "--- GOAL ---\n"
        text += f"{goal}\n\n"

        text += "--- AGENTS ---\n"
        for agent in agents:
            text += f"{agent['name']}: {agent['task']}\n"
        text += "\n"

        text += "--- WORKFLOW ---\n"
        for step in workflow_steps:
            text += f"AGENT: {step['agent']}\n"
            text += f"STEP: {step['step_number']}\n"
            text += f"PROMPT: {step['prompt']}\n"
            text += f"REASONING {step['reasoning']}\n"
            text += f"OUTPUT: {step['code_output']}\n"
            text += "\n"

        text += "--- WORKFLOW CODE---\n"
        with open(workflow_path / f"workflow_code_{uuid}.py") as f:
            workflow_mermaid = f.read()
        text += f"{workflow_mermaid}\n"

        # Write the formatted text to file
        with open(memory_path / "formated.txt", "w") as file:
            file.write(text.strip())

        return text.strip()

    def long_prompt(self, include_answer_assessment=False):
        answer_assessment = (
            """
4. **Answer Correctness Assessment**
   - Evaluate whether the final answer produced by the system matches the expected answer
   - Analyze any discrepancies between the system's answer and the expected answer
   - Identify potential reasons for incorrect or incomplete answers

"""
            if include_answer_assessment
            else ""
        )

        return f"""
Please analyze the system with the following structure:

1. **Step-by-step Critique**
   - For each step in the workflow, indicate whether the output is relevant, valid, and logically derived from the input.
   - Highlight any inconsistencies, errors, redundant actions, or missed opportunities.

2. **Per-Agent Notes**
   - For each agent, evaluate:
     - Whether the agent's behavior aligns with its intended role
     - How well it contributes to the overall goal
     - Any limitations, misinterpretations, or inefficiencies observed
     - Suggestions for prompt improvements or role clarification

3. **Workflow Logic Assessment**
   - Assess the global coordination between agents:
     - Does the overall workflow make sense?
     - Are steps logically ordered?
     - Is there information loss or miscommunication between agents?
     - Is the final output aligned with the initial goal?
   - Suggest architectural or coordination-level improvements (e.g., adding intermediate validation, changing agent order, improving memory/context sharing)
{answer_assessment}
{4 + (1 if include_answer_assessment else 0)}. **Summary Judgment**
 """

    def evaluate(
        self, uuid: str, short: bool = True, answer: str = None, scenario_id: str = None
    ) -> str:
        """Evaluate the workflow results.

        Args:
            uuid: UUID of the workflow run to evaluate
            short: Whether to use short evaluation format
            answer: Optional expected answer for evaluation
            scenario_id: Optional scenario ID for scenario-based evaluation
        """
        # If scenario_id is provided, use scenario-based evaluation
        if scenario_id:
            self.evaluate_workflow(uuid, scenario_id)
            return "scenario"

        # Otherwise, use the original judge-based evaluation
        # Adjust system prompt based on whether an expected answer is provided
        answer_evaluation_task = ""
        if answer:
            answer_evaluation_task = "- Assess whether the final answer produced by the system matches the expected answer.\n"

        system_prompt = f"""You are a rigorous and objective evaluator of a multi-agent system designed to solve a complex goal through a coordinated workflow. You will be given:

1. A description of the system's **goal**.
2. A list of **agents**, each with their assigned roles.
3. The **workflow trace**, including the input and output of each step for every agent.
4. The **python workflow** that explain agent worflow.

Your task is to:
- Check if each agent behaves consistently with its role.
- Determine whether each step logically follows from the previous step.
- Identify whether outputs are appropriate, helpful, or erroneous.
- Pinpoint any bottlenecks, misunderstandings, or failures.
- Evaluate how well the agents are collaborating to reach the goal.
{answer_evaluation_task}- Suggest what changes could improve the system's reliability, performance, or alignment with the goal.

Be precise, constructive, and technical in your judgment."""
        # Prepare the expected answer information if available
        expected_answer_info = ""
        json_format = """
     ```json
     {{
        "goal_alignment": X,
        "agent_collaboration": Y,
        "output_quality": Z,
     }}
     ```"""

        if answer:
            expected_answer_info = f"\n\n--- EXPECTED ANSWER ---\n{answer}\n"
            json_format = """
     ```json
     {{
        "goal_alignment": X,
        "agent_collaboration": Y,
        "output_quality": Z,
        "answer_correctness": W
     }}
     ```"""

        prompt = f"""
You are provided with a multi-agent system designed to achieve a specific goal.
The system is composed of multiple specialized agents working in sequence or collaboration.

{self.generate_text(uuid)!r}{expected_answer_info}

--- EVALUATION REQUEST ---
{self.long_prompt(answer is not None) if not short else ""}
- Provide an overall score (1–10) for each category in the following JSON format:{json_format}
{"- The 'answer_correctness' score should evaluate how well the system's final answer matches the expected answer." if answer else ""}
- After the JSON, briefly justify your scores.

Please be objective, technical, and specific in your feedback.
"""
        self.logger.info(f"Evaluating workflow {uuid} with LLM judge")
        memory_path = Path(self.memory_dir) / uuid
        config_llm = LLMConfig().from_dict({"model": "gpt-4o-mini"})
        output = LLMProvider("judge", memory_path, system_prompt, config_llm)(prompt)

        # Save the evaluation to a file
        evaluation_path = self.workflow_dir / uuid / "evaluation.txt"
        with open(evaluation_path, "w") as file:
            file.write(output)
        self.logger.info(
            f"Evaluation completed for {uuid}. Results saved to: {evaluation_path}"
        )

        # Extract scores from the evaluation output
        scores = self._extract_scores(output)

        self._update_state_result(scores, uuid)
        print("Scores extracted and saved to state result.")

        return "generic"

    def _extract_scores(self, evaluation_text):
        """Extract scores from the evaluation text.

        Args:
            evaluation_text: The evaluation text containing the JSON scores

        Returns:
            dict: The extracted scores or empty dict if not found
        """
        try:
            # Look for JSON block in the evaluation text
            json_pattern = r"(?:```json\s*)?({[^`]*})(?:\s*```)?"
            match = re.search(json_pattern, evaluation_text)

            if match:
                json_str = match.group(1)
                scores = json.loads(json_str)
                # Calculate overall score including answer_correctness if available
                scores["overall_score"] = (
                    scores["goal_alignment"]
                    + scores["agent_collaboration"]
                    + scores["output_quality"]
                )
                scores["overall_score"] /= 3
                return scores
            else:
                print("⚠️ No JSON scores found in evaluation output")
                return {}
        except Exception as e:
            print(f"❌ Error extracting scores: {str(e)}")
            return {}

    def _update_state_result(self, scores, uuid: str):
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
            state_result.setdefault("evaluation", {})["generic"] = scores

            # Write updated state result back to file
            with open(state_result_path, "w") as f:
                json.dump(state_result, f, indent=2)

        except Exception as e:
            print(f"❌ Error updating state result: {str(e)}")

    def get_text(self, uuid: str):
        """Get the formatted benchmark text.

        Args:
            uuid: UUID of the workflow run

        Returns:
            str: The formatted benchmark text
        """
        formated_path = Path(self.memory_dir) / uuid / "formated.txt"
        if not formated_path.exists():
            self.generate_text(uuid)
        return open(formated_path).read()

    # Methods from scenario-based evaluator
    def evaluate_workflow(self, workflow_id: str, scenario_id: str):
        """Evaluate a workflow against a scenario with scoring."""
        print(f"Evaluating workflow {workflow_id} against scenario {scenario_id}")

        # Load scenario and workflow data
        scenario = self.scenario_loader.load_scenario(scenario_id)
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")

        workflow_data = self._load_workflow_data(workflow_id)

        # Evaluate all assertions
        assertion_results = []
        for assertion in scenario["assertions"]:
            result = self._evaluate_assertion(workflow_data, assertion)
            assertion_results.append(result)

        # Calculate score (only partial score)
        passed_count = sum(1 for result in assertion_results if result["passed"])
        total_count = len(assertion_results)
        score = passed_count / total_count if total_count > 0 else 0.0

        # Generate results
        results = {
            # "workflow_id": workflow_id,
            "scenario_id": scenario_id,
            "timestamp": datetime.now().isoformat(),
            # "goal": scenario.get("goal", ""),
            "score": score,
            "passed_assertions": passed_count,
            "total_assertions": total_count,
            "assertion_results": assertion_results,
            "judge_model": self.judge_model,
        }

        # Save results
        self._save_scenario_results(workflow_id, scenario_id, results)

    def _load_workflow_data(self, workflow_id: str) -> dict[str, Any]:
        """Load workflow execution data from UUID folder."""
        workflow_path = Path(self.workflow_dir) / workflow_id

        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow {workflow_id} not found")

        workflow_data = {
            "workflow_id": workflow_id,
            "state_result": {},
            "workflow_code": "",
        }

        # Load state_result.json
        state_result_path = workflow_path / "state_result.json"
        if state_result_path.exists():
            try:
                with open(state_result_path) as f:
                    workflow_data["state_result"] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load state_result.json: {e}")

        # Load workflow code
        workflow_code_path = workflow_path / f"workflow_code_{workflow_id}.py"
        if workflow_code_path.exists():
            try:
                with open(workflow_code_path) as f:
                    workflow_data["workflow_code"] = f.read()
            except Exception as e:
                print(f"Warning: Could not load workflow code: {e}")

        return workflow_data

    def _evaluate_assertion(
        self, workflow_data: dict[str, Any], assertion: dict
    ) -> dict[str, Any]:
        """Evaluate single assertion using existing LLM prompt format."""
        # Build judge prompt using existing format
        judge_prompt = self._build_judge_prompt(workflow_data, assertion)

        try:
            # Use LLMProvider instead of direct OpenAI call
            llm_provider = LLMProvider(
                agent_name="scenario_judge",
                memory_path=self.memory_dir / workflow_data["workflow_id"],
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
        self, workflow_data: dict[str, Any], assertion: dict
    ) -> str:
        """Build judge prompt with workflow data."""
        state_result = workflow_data.get("state_result", {})
        workflow_code = workflow_data.get("workflow_code", "")
        goal = state_result.get("goal", "Goal not specified")
        criteria = assertion.get("evaluation_criteria", "Standard evaluation")

        return f"""
You are evaluating a scientific workflow execution.

ASSERTION TO EVALUATE:
ID: {assertion["id"]}
Description: {assertion["description"]}
Evaluation Criteria: {criteria}

WORKFLOW GOAL:
{goal}

FULL WORKFLOW STATE RESULT (JSON):
{json.dumps(state_result, indent=2)}

WORKFLOW CODE:
```python
{workflow_code}
```

EVALUATION TASK:
Based on the complete execution state and workflow code above, determine if the 
assertion is TRUE or FALSE.
Focus on whether the workflow achieved the goals and execution was successful.
Analyze the full JSON state and workflow implementation to make your judgment.

Respond in this exact format:
VERDICT: [TRUE/FALSE]
EVIDENCE: [Specific evidence from the execution that supports your verdict]
CONFIDENCE: [0.0-1.0 confidence score]
"""

    def _get_judge_system_prompt(self) -> str:
        """Get system prompt for LLM judge (keeping existing format)."""
        prompt = "You are an expert scientific researcher evaluating whether "
        prompt += "a workflow achieved its intended goals. Focus on:\n"
        prompt += "- Did the workflow produce the requested results/analysis?\n"
        prompt += "- Are the scientific outputs accurate and useful?\n"
        prompt += "- Was the research question adequately addressed?\n"
        prompt += "- Were tools used correctly and in proper sequence?\n"
        prompt += "- Did the system handle errors appropriately?\n"
        prompt += "- Are results presented clearly and professionally?\n\n"
        prompt += "Evaluate based on available evidence, considering user "
        prompt += "satisfaction and system quality."
        return prompt

    def _parse_judge_response(self, judge_text: str) -> tuple[bool, str, float]:
        """Parse LLM judge response (keeping existing format)."""
        try:
            lines = judge_text.strip().split("\n")
            verdict = False
            evidence = "No evidence provided"
            confidence = 0.5

            for line in lines:
                if line.startswith("VERDICT:"):
                    verdict_str = line.split(":", 1)[1].strip().upper()
                    verdict = "TRUE" in verdict_str
                elif line.startswith("EVIDENCE:"):
                    evidence = line.split(":", 1)[1].strip()
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        confidence = 0.5

            return verdict, evidence, confidence

        except Exception as e:
            print(f"Error parsing judge response: {e}")
            return False, f"Parse error: {str(e)}", 0.0

    def _save_scenario_results(
        self, workflow_id: str, scenario_id: str, results: dict[str, Any]
    ):
        """Save evaluation results to workflow UUID directory."""
        # Save to workflow UUID directory instead of global results directory
        workflow_dir = Path(self.workflow_dir) / workflow_id

        if not workflow_dir.exists():
            print(
                f"Warning: Workflow directory {workflow_dir} does not exist. "
                f"Creating it."
            )
            workflow_dir.mkdir(parents=True, exist_ok=True)

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = "state_result.json"

        with open(workflow_dir / filename) as f:
            state_json = json.load(f)

        # Modify data
        state_json.setdefault("evaluation", {})["scenario"] = results

        # Then write back
        with open(workflow_dir / filename, "w") as f:
            json.dump(state_json, f, indent=2)

        print(f"Results saved to: {workflow_dir / filename}")

    def __str__(self):
        return "WorkflowEvaluator instance"

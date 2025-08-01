#!/usr/bin/env python3
"""
Evaluation engine for Mimosa-AI workflows.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import openai

from .scenario_loader import ScenarioLoader


class Evaluator:
    """Evaluator."""

    def __init__(self, judge_model: str = "gpt-4o"):
        self.judge_model = judge_model

        self.scenario_loader = ScenarioLoader()
        self.client = openai.OpenAI()

    def evaluate_workflow(self, workflow_id: str, scenario_id: str) -> dict[str, Any]:
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
            "workflow_id": workflow_id,
            "scenario_id": scenario_id,
            "timestamp": datetime.now().isoformat(),
            "goal": scenario.get("goal", ""),
            "score": score,
            "passed_assertions": passed_count,
            "total_assertions": total_count,
            "assertion_results": assertion_results,
            "judge_model": self.judge_model,
        }

        # Save results
        self._save_results(workflow_id, scenario_id, results)
        return results

    def _load_workflow_data(self, workflow_id: str) -> dict[str, Any]:
        """Load workflow execution data from UUID folder."""
        workflow_path = Path(f"sources/workflows/{workflow_id}")

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
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {"role": "system", "content": self._get_judge_system_prompt()},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0,
                # max_tokens=500
            )

            # Parse response
            judge_text = response.choices[0].message.content.strip()
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

    def _save_results(
        self, workflow_id: str, scenario_id: str, results: dict[str, Any]
    ):
        """Save evaluation results to workflow UUID directory."""
        # Save to workflow UUID directory instead of global results directory
        workflow_dir = Path(f"sources/workflows/{workflow_id}")

        if not workflow_dir.exists():
            print(
                f"Warning: Workflow directory {workflow_dir} does not exist. "
                f"Creating it."
            )
            workflow_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{scenario_id}_{timestamp}.json"

        with open(workflow_dir / filename, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {workflow_dir / filename}")


def main():
    """Command-line interface for workflow evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Mimosa-AI workflows")
    parser.add_argument(
        "--workflow_id", required=True, help="UUID of workflow to evaluate"
    )
    parser.add_argument("--scenario", required=True, help="Evaluation scenario ID")
    parser.add_argument(
        "--judge_model", default="gpt-4o", help="LLM model for evaluation"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    evaluator = Evaluator(judge_model=args.judge_model)

    try:
        results = evaluator.evaluate_workflow(args.workflow_id, args.scenario)

        if args.verbose:
            print(json.dumps(results, indent=2))
        else:
            print(f"Score: {results['score']:.2f}")
            print(
                f"Assertions: {results['passed_assertions']}/"
                f"{results['total_assertions']}"
            )

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

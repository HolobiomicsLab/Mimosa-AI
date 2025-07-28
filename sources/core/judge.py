import json
import os
from dataclasses import dataclass
from pathlib import Path
import re

from flask import Config

from sources.core.llm_provider import LLMConfig, LLMProvider


@dataclass
class TokenUsage:
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


class WorkflowJudge:
    def __init__(self, config):
        self.memory_dir = Path(config.memory_dir)
        self.workflow_dir = Path(config.workflow_dir)
        self.model_pricing = config.model_pricing

    def calculate_cost(self, uuid: str) -> float:
        """Calculate the cost of a workflow run based on token usage.

        Args:
            config: The configuration object
            uuid: Optional UUID of the workflow run to calculate cost for.
                If not provided, will try to find the most recent run.

        Returns:
            float: The total cost in USD
        """

        print("\n📊 Calculating final cost...")

        memory_path = Path(self.memory_dir) / uuid

        if not memory_path.exists():
            print(f"❌ Memory directory not found: {memory_path}")
            return 0.0

        llm_calls: list[TokenUsage] = []

        # Orchestrator and Judge LLM calls

        for call in ["workflow_creator", "judge"]:
            memory_file = memory_path / f"{call}.json"
            if not memory_file.exists():
                continue

            with open(memory_file) as f:
                json_call = json.load(f)
                llm_calls.append(
                    TokenUsage(
                        call,
                        json_call["model"],
                        json_call["usage"]["prompt_tokens"],
                        json_call["usage"]["completion_tokens"],
                        json_call["usage"]["total_tokens"],
                    )
                )

        workflow_path = Path(self.workflow_dir) / uuid

        if not workflow_path.exists():
            print(f"❌ Workflow directory not found: {workflow_path}")
            return 0.0

        try:
            with open(workflow_path / "state_result.json") as f:
                state_results = json.load(f)
                model_id = state_results.get("model_id", None)
        except FileNotFoundError:
            print(f"❌ State result file not found for UUID {uuid} in {workflow_path}.")
            return 0.0

        try:
            for file in os.listdir(memory_path):
                if file.startswith("task_") and file.endswith(".json"):
                    with open(memory_path / file) as f:
                        steps = json.load(f)
                        token_usage = {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        }
                        for step in steps:
                            step_usage = step.get("token_usage", None)
                            if token_usage:
                                token_usage = {
                                    key: token_usage[key] + step_usage[key]
                                    for key in step_usage
                                }
                        llm_calls.append(
                            TokenUsage(
                                file.replace("task_", "").replace(".json", ""),
                                model_id,
                                *token_usage.values(),
                            )
                        )
        except Exception as e:
            print(f"❌ Error reading workflow steps: {str(e)}")
            return 0.0

        total_cost = 0.0
        print("\n💰 Cost Breakdown:")
        print("=" * 60)
        for call in llm_calls:
            pricing = self.model_pricing.get(
                call.model,
                self.model_pricing.get("default", {"input": 0.70, "output": 2.50}),
            )
            cost = (
                call.input_tokens * pricing["input"]
                + call.output_tokens * pricing["output"]
            ) / 1_000_000
            print("Agent:", call.agent)
            print(f"  Model: {call.model}")
            print(f"  Tokens: {call.total_tokens:,}")
            print(f"  Cost: {cost:.3f} USD")
            print("-" * 40)
            total_cost += cost

        return total_cost

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
                            "start_time":step["timing"]["start_time"],
                            "step_number": step["step_number"],
                            "prompt": step["model_input_messages"][1]["content"][0]["text"],
                            "reasoning": step["model_output"],
                            "code_action" : step.get("code_action", ""),
                            "code_output": code_result
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

    def evaluate(self, uuid: str, short: bool = True, answer: str = None):
        """Evaluate the benchmark results.

        Args:
            uuid: UUID of the workflow run to evaluate
        """
        # Adjust system prompt based on whether an expected answer is provided
        answer_evaluation_task = ""
        if answer:
            answer_evaluation_task = "- Assess whether the final answer produced by the system matches the expected answer.\n"

        system_prompt = f"""You are a rigorous and objective evaluator of a multi-agent system designed to solve a complex goal through a coordinated workflow. You will be given:

1. A description of the system's **goal**.
2. A list of **agents**, each with their assigned roles.
3. The **workflow trace**, including the input and output of each step for every agent.
4. The **mermaid workflow** that explain agent worflow.

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

        prompt = f"""You are provided with a multi-agent system designed to achieve a specific goal. The system is composed of multiple specialized agents working in sequence or collaboration.

{self.generate_text(uuid)!r}{expected_answer_info}

--- EVALUATION REQUEST ---
{self.long_prompt(answer is not None) if not short else ""}
   - Provide an overall score (1–10) for each category in the following JSON format:{json_format}
   {"- The 'answer_correctness' score should evaluate how well the system's final answer matches the expected answer." if answer else ""}
   - After the JSON, briefly justify your scores.

Please be objective, technical, and specific in your feedback.
"""
        print("Calling LLMProvider to evaluate the workflow...")
        memory_path = Path(self.memory_dir) / uuid
        config_llm = LLMConfig().from_dict({"model": "o4-mini-2025-04-16"})
        output = LLMProvider("judge", memory_path, system_prompt, config_llm)(prompt)

        # Save the evaluation to a file
        evaluation_path = self.workflow_dir / uuid / "evaluation.txt"
        with open(evaluation_path, "w") as file:
            file.write(output)
        print("Evaluation completed. Results saved to:", evaluation_path)

        # Extract scores from the evaluation output
        scores = self._extract_scores(output)
            
        self._update_state_result(scores, uuid)
        print("Scores extracted and saved to state result.")

    def _extract_scores(self, evaluation_text):
        """Extract scores from the evaluation text.

        Args:
            evaluation_text: The evaluation text containing the JSON scores

        Returns:
            dict: The extracted scores or empty dict if not found
        """
        try:
            # Look for JSON block in the evaluation text
            import re

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
            state_result["evaluation_scores"] = scores

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

    def __str__(self):
        return "WorkflowJudge instance"

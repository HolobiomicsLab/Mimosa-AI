import json
import os
from dataclasses import dataclass
from pathlib import Path

from sources.core.llm_provider import LLMProvider


@dataclass
class TokenUsage:
    agent: str
    model: str
    tokens: int

class WorkflowJudge:
    def __init__(self, config):
        self.memory_dir = Path(config.memory_dir)
        self.workflow_dir = Path(config.workflow_dir)
        self.model_pricing = config.model_pricing

    def calculate_cost(self, uuid: str )-> float:
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
        with open(memory_path / "llms_call.json") as f:
            json_calls = json.load(f) 
            for call in json_calls:
                total_tokens = call["token_usage"]["total_tokens"]
                model =  call["model"]

                llm_calls.append(TokenUsage(
                    "orchestrator",
                    model,
                    total_tokens
                ))
        
        workflow_path= Path(self.workflow_dir) / uuid

        if not workflow_path.exists():
            print(f"❌ Workflow directory not found: {workflow_path}")
            return 0.0
        
        try:
            with open(workflow_path / f"state_result_{uuid}.json") as f:
                state_results = json.load(f)
                model_id = state_results.get("model_id", None)
        except FileNotFoundError:
            print(f"❌ State result file not found for UUID {uuid} in {workflow_path}.")
            return 0.0

            
        try:
            for file in os.listdir(memory_path):
                if file.startswith("task_") and file.endswith(".json"):
                    with open(workflow_path / file) as f:
                        steps = json.load(f)
                        total_tokens = 0
                        for step in steps:
                            total_tokens += step.get("token_usage", {}).get("total_tokens", 0)
                        llm_calls.append(TokenUsage(
                            file.replace("task_", "").replace(".json", ""),
                            model_id,
                            total_tokens
                        ))
        except Exception as e:
            print(f"❌ Error reading workflow steps: {str(e)}")
            return 0.0
        
        total_cost = 0.0
        print("\n💰 Cost Breakdown:")
        print("=" * 60)
        for call in llm_calls:
            pricing = self.model_pricing.get(call.model, self.model_pricing["default"])
            cost = call.tokens * pricing / 1_000_000 
            print("Agent:", call.agent)
            print(f"  Model: {call.model}")
            print(f"  Tokens: {call.tokens:,}")
            print(f"  Cost: {cost:.3f} USD")
            print("-" * 40)
            total_cost += cost
        
        return total_cost


    def generate_text(self, uuid: str):
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
                agent_name = file.removeprefix('task_').removesuffix('.json').replace('_', ' ')
                
                with open(os.path.join(memory_path, file), "r") as f:
                    steps = json.load(f)
                    start_time = int(steps[0]["timing"].get("start_time", ""))
                    user_task = steps[0]['model_input_messages'][1]['content'][0].get('text', '')
                    
                    # Add agent to the list
                    agents.append({
                        "name": agent_name,
                        "task": user_task,
                        "start_time": start_time
                    })
                    
                    # Process steps for workflow
                    for step in steps:
                        step_info = {
                            "agent": agent_name,
                            "step_number": step.get('step_number', ''),
                            "action": step.get('code_action', ''),
                            "start_time": start_time,
                            "error": step.get("error", ""),
                            "result": step.get('observations', '')
                        }
                        workflow_steps.append(step_info)
        
        # Sort agents and workflow steps by start time
        agents.sort(key=lambda x: x["start_time"])
        workflow_steps.sort(key=lambda x: (x["start_time"], x["step_number"]))
        
        # Read the goal
        with open(workflow_path / f"goal_{uuid}.txt") as f:
            goal = f.read()
        
        # Format the text according to the new structure
        text = "--- GOAL ---\n"
        text += f"{goal}\n\n"
        
        text += "--- AGENTS ---\n"
        for agent in agents:
            text += f"{agent['name']}: {agent['task']}\n"
        text += "\n"
        
        text += "--- WORKFLOW ---\n"
        for step in workflow_steps:
            text += f"Agent: {step['agent']}\n"
            text += f"Step: {step['step_number']}\n"
            text += f"Input: {step['action']}\n"
            if step['error']:
                text += f"Output: ERROR - {step['error']}\n"
            else:
                text += f"Output: {step['result']}\n"
            text += "\n"

        text += "--- WORKFLOW CODE ---\n"
        with open(workflow_path / f"workflow_code_{uuid}.py") as f:
            workflow_code = f.read()
        text += f"{workflow_code}\n"
        
        # Write the formatted text to file
        with open(memory_path / "formated.txt", "w") as file:
            file.write(text.strip())

    def evaluate(self, uuid: str):
        """Evaluate the benchmark results.
        
        Args:
            uuid: UUID of the workflow run to evaluate
        """
        system_prompt = """You are a rigorous and objective evaluator of a multi-agent system designed to solve a complex goal through a coordinated workflow. You will be given:

1. A description of the system's **goal**.
2. A list of **agents**, each with their assigned roles.
3. The **workflow trace**, including the input and output of each step for every agent.
4. The **workflow code** that orchestrates the agents.

Your task is to:
- Check if each agent behaves consistently with its role.
- Determine whether each step logically follows from the previous step.
- Identify whether outputs are appropriate, helpful, or erroneous.
- Pinpoint any bottlenecks, misunderstandings, or failures.
- Evaluate how well the agents are collaborating to reach the goal.
- Suggest what changes could improve the system's reliability, performance, or alignment with the goal.

Be precise, constructive, and technical in your judgment."""
        prompt = f"""You are provided with a multi-agent system designed to achieve a specific goal. The system is composed of multiple specialized agents working in sequence or collaboration.

{self.get_text(uuid)}

--- EVALUATION REQUEST ---
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

4. **Summary Judgment**
   - Provide an overall score (1–10) for each category in the following JSON format:
     ```json
     {
        "goal_alignment": X,
        "agent_collaboration": Y,
        "output_quality": Z,
     }
     ```
   - After the JSON, briefly justify your scores.

Please be objective, technical, and specific in your feedback.
"""
        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        output = LLMProvider().openai_completion(
            history=history,
            model = "o4-mini-2025-04-16"
        )

        # Save the evaluation to a file
        evaluation_path = Path(self.workflow_dir) / uuid / "evaluation.txt"
        with open(evaluation_path, "w") as file:
            file.write(output)
        print("Evaluation completed. Results saved to:", evaluation_path)
            
        # Extract scores from the evaluation output
        scores = self._extract_scores(output)
        
        # Update the state result file with the scores
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
            json_pattern = r'```json\s*({[^`]*})\s*```'
            match = re.search(json_pattern, evaluation_text)
            
            if match:
                json_str = match.group(1)
                scores = json.loads(json_str)
                scores["overall_score"] = scores["goal_alignment"] + scores["agent_collaboration"] + scores["output_quality"]
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
            state_result_path = workflow_path / f"state_result_{uuid}.json"
            
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
        memory_path = Path(self.memory_dir) / uuid
        formated_path = memory_path / "formated.txt"
        if not formated_path.exists():
            self.generate_text(uuid)
        return open(formated_path).read()

    def __str__(self):
        return "WorkflowJudge instance"

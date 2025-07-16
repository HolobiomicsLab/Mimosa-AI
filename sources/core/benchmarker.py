import json
import os
from pathlib import Path

from sources.core.llm_provider import LLMProvider


class Benchmarker:
    def __init__(self, config, uuid: str):
        self.uuid = uuid
        self.memory_path = Path(config.memory_dir) / self.uuid
        self.workflow_path = Path(config.workflow_dir) / self.uuid
        self.results = []

    def generate_text(self):
        # Collect agent information
        agents = []
        workflow_steps = []
        
        for file in os.listdir(self.memory_path):
            if file.endswith(".json") and file.startswith("task_"):
                agent_name = file.removeprefix('task_').removesuffix('.json').replace('_', ' ')
                
                with open(os.path.join(self.memory_path, file), "r") as f:
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
        with open(self.workflow_path / f"goal_{self.uuid}.txt") as f:
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
        with open(self.workflow_path / f"workflow_code_{self.uuid}.py") as f:
            workflow_code = f.read()
        text += f"{workflow_code}\n"
        
        # Write the formatted text to file
        with open(self.memory_path / "formated.txt", "w") as file:
            file.write(text.strip())

    def evaluate(self):
        """Evaluate the benchmark results."""
        system_prompt = """You are a rigorous and objective evaluator of a multi-agent system designed to solve a complex goal through a coordinated workflow. You will be given:

1. A description of the system’s **goal**.
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

{self.get_text()}

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
   - Provide an overall score (1–10) for:
     - Goal alignment
     - Agent collaboration
     - Output quality
   - Briefly justify your score.

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

        with open(self.workflow_path / "evaluation.txt", "w") as file:
            file.write(output)

    def get_text(self):
        return open(self.memory_path / "formated.txt").read()

    def __str__(self):
        return f"Benchmarker text:\n\n{self.get_text()}"

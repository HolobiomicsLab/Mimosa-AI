from smolagents import (
    CodeAgent,
    HfApiModel
)
import os

from schema_factory import WorkflowState, Action, Observation

class SmolAgentFactory:
    def __init__(self, model_id="Qwen/Qwen2.5-Coder-32B-Instruct", instruct_prompt = "", tools=None, max_steps=5):
        self.model_id = model_id
        self.token = os.getenv("HF_TOKEN")
        self.tools = tools or []
        self.instruct_prompt = instruct_prompt

        if not self.token:
            raise ValueError("Hugging Face token is required. Please set the HF_TOKEN environment variable or pass a token.")
        
        self.engine = HfApiModel(
            model_id=model_id,
            token=self.token,
            max_tokens=8096,
        )
        
        self.agent = CodeAgent(
            tools=self.tools,
            model=self.engine,
            name="agent",
            max_steps=max_steps,
        )
    
    def parse_tool_output(self, output: str):
        actions = []
        observations = []
        rewards = []
        
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('action:'):
                try:
                    action_str = line[7:].strip()
                    action_args = eval(action_str)
                    action = {
                        "tool": action_args[0].split('(')[0],
                        "inputs": {"args": action_args}
                    }
                    actions.append(action)
                except:
                    pass
            elif line.startswith('observation:'):
                obs_str = line[12:].strip()
                observation = {"data": obs_str}
                observations.append(observation)
            elif line.startswith('reward:'):
                try:
                    reward_str = line[7:].strip()
                    reward = float(reward_str)
                    rewards.append(reward)
                except:
                    pass
        return actions, observations, rewards
    
    def run(self, state: WorkflowState) -> dict:
        instructions_template = f"""
{{self.instruct_prompt}}
Current task: {{goal}}.
        """
        try:
            instructions = instructions_template.format(
                goal=state["goal"][-1] if state["goal"] else "complete the task"
            )
            result = self.agent.run(instructions)
            actions, observations, rewards = self.parse_tool_output(result)

            action: Action = {
                "tool": actions[-1]["tool"] if actions else "unknown",
                "inputs": actions[-1]["inputs"] if actions else {},
            }
            obs: Observation = {
                "data": result
            }
            reward = sum(rewards) if rewards else 0.0
            return {
                **state,
                "goal": state["goal"],
                "actions": state["actions"] + [action],
                "observations": state["observations"] + [obs],
                "rewards": state["rewards"] + [reward],
            }
        except Exception as e:
            return {
                **state,
                "goal": state["goal"],
                "actions": state["actions"] + [None],
                "observations": state["observations"] + [None],
                "rewards": state["rewards"] + [0],
            }

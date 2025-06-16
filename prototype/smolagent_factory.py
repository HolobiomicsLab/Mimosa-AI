
import os
from typing import Callable
from typing import TypedDict, List, Tuple, Any, Dict, Union, Optional, Callable
from smolagents import (
    CodeAgent,
    HfApiModel,
    InferenceClientModel,
    ActionStep,
    TaskStep
)

class Action(TypedDict):
    name: str
    inputs: dict

class Observation(TypedDict):
    data: str

class WorkflowState(TypedDict):
    goal: List[str]
    actions: List[Action]
    observations: List[Observation]
    rewards: List[float]
    success: List[bool]

class SmolAgentFactory:
    def __init__(self, instruct_prompt, tools, model_id="Qwen/Qwen2.5-Coder-32B-Instruct", max_steps=3):
        self.model_id = model_id
        self.token = os.getenv("HF_TOKEN")
        self.tools = tools or []
        self.instruct_prompt = instruct_prompt

        if not self.token:
            raise ValueError("Hugging Face token is required. Please set the HF_TOKEN environment variable or pass a token.")
        try:
            self.engine = InferenceClientModel(
                model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
                provider="nebius",
                token=self.token,
                max_tokens=5000,
            )

            self.agent = CodeAgent(
                tools=self.tools,
                model=self.engine,
                name="agent",
                max_steps=max_steps,
                additional_authorized_imports=["*"]
        )
        except Exception as e:
            raise ValueError(f"Error initializing SmolAgent: {e}") from e
        
    def build_worflow_step_prompt(self, state: WorkflowState) -> str:
        state_actions = state.get("actions", [])
        state_observations = state.get("observations", [])
        state_rewards = state.get("rewards", [])
        state_success = state.get("success", [])
        trajectories = zip(
            state_actions, 
            state_observations, 
            state_success
        )
        trajectories_prompt = "\n".join(
            f"Action: {action['tool']}, Observation: {observation['data'][:256]}"
            for action, observation, success in trajectories
        )
        return f"""
        You previously performed the following actions:
        {trajectories_prompt}
        The end goal is to:
        {state["goal"][-1] if len(state["goal"]) > 0 else "complete the task"}.
        Your need to follow instructions:
        {self.instruct_prompt}
        """
    
    def parse_tool_output(self, output: str):
        actions = []
        observations = []
        rewards = []
        success = []
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('action:'):
                action = line[7:].strip()
                actions.append(action)
            elif line.startswith('observation:'):
                obs_str = line[12:].strip()
                observations.append(obs_str)
            elif line.startswith('reward:'):
                reward_str = line[7:].strip()
                reward = float(reward_str)
                rewards.append(reward)
                success.append(reward > 0)
        return ('\n'.join(actions),
                '\n'.join(observations),
                (sum(rewards) / len(rewards)) if len(rewards) > 0 else sum(rewards),
                any(success)
        )

    def parse_memory_output(self):
        actions, observations, rewards, success = [], [], [], []
        for idx, step in enumerate(self.agent.memory.steps):
            if isinstance(step, ActionStep):
                error, feedback = step.error, step.observations
                step_output = error if error else feedback
                if not isinstance(step_output, str):
                    continue
                action_step, obs_step, reward_step, success_step = self.parse_tool_output(step_output)
                actions.append(action_step)
                observations.append(obs_step)
                rewards.append(reward_step)
                success.append(success_step)
        return actions, observations, rewards, success

    def run(self, state: WorkflowState) -> dict:
        instructions = self.build_worflow_step_prompt(state)
        result = self.agent.run(instructions)
        actions, observations, rewards, success = self.parse_memory_output()
        action: Action = {
            "tool": actions[-1] if actions else "unknown"
            #"inputs": actions[-1]["inputs"] if actions else {},
        }
        obs: Observation = {
            "data": observations
        }
        reward = sum(rewards) if rewards else 0.0
        success = any(success) if success else False
        return {
            **state,
            "goal": state["goal"],
            "actions": state["actions"] + [action],
            "observations": state["observations"] + [obs],
            "rewards": state["rewards"] + [reward],
            "success": state["success"] + [success],
        }

class WorkflowNodeFactory:
    @staticmethod
    def create_agent_node(agent_factory: SmolAgentFactory) -> Callable[[WorkflowState], dict]:
        def node_function(state: WorkflowState) -> dict:
            return agent_factory.run(state)
        return node_function


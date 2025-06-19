
import os
from typing import Callable
from typing import TypedDict, List, Tuple, Any, Dict, Union, Optional, Callable
from smolagents import (
    CodeAgent,
    HfApiModel,
    MLXModel,
    InferenceClientModel,
    ActionStep,
    TaskStep
)
from smolagents.local_python_executor import BASE_PYTHON_TOOLS, DANGEROUS_FUNCTIONS, DANGEROUS_MODULES
BASE_PYTHON_TOOLS["open"] = open
DANGEROUS_FUNCTIONS = {}
DANGEROUS_MODULES = {}

class Action(TypedDict):
    tool: str

class Observation(TypedDict):
    data: str

class WorkflowState(TypedDict):
    step_name: List[str]
    actions: List[Action]
    observations: List[Observation]
    rewards: List[float]
    answers: List[str]
    success: List[bool]

# good models:
#Qwen/Qwen2.5-72B-Instruct
#Qwen/Qwen2.5-Coder-32B-Instruct
# deepseek-ai/DeepSeek-V3
class SmolAgentFactory:

    def __init__(self, instruct_prompt, tools, model_id="deepseek-ai/DeepSeek-V3", engine_name="mlx", max_steps=5):
        self.model_id = model_id
        self.max_tokens = 1024
        self.provider = "novita"
        self.token = os.getenv("HF_TOKEN")
        self.tools = tools or []
        self.instruct_prompt = instruct_prompt
        self.local = False
        self.engine_name = engine_name
        self.engine = None

        if not self.token:
            raise ValueError("Hugging Face token is required. Please set the HF_TOKEN environment variable or pass a token.")
        try:
            self.engine = self.get_engine()
            self.agent = CodeAgent(
                tools=self.tools,
                model=self.engine,
                name="agent",
                max_steps=max_steps,
                additional_authorized_imports=["*"]
        )
        except Exception as e:
            raise ValueError(f"Error initializing SmolAgent: {e}") from e
    
    def get_engine(self):
        if self.engine_name == "mlx":
            self.local = True
            return MLXModel(
                model_id=self.model_id,
                max_tokens=self.max_tokens,
            )
        elif self.engine_name == "hf_api":
            return HfApiModel(
                model_id=self.model_id,
                provider=self.provider,
                token=self.token,
                max_tokens=self.max_tokens,
            )
        elif self.engine_name == "inference_client":
            return InferenceClientModel(
                model_id=self.model_id,
                provider=self.provider,
                token=self.token,
                max_tokens=self.max_tokens,
            )
        else:
            raise ValueError(f"Unknown engine name: {self.engine_name}. Supported engines are: mlx, hf_api, inference_client.")
        
    def build_worflow_step_prompt(self, state: WorkflowState) -> str:
        state_steps = state.get("step_name", [])
        state_actions = state.get("actions", [])
        state_observations = state.get("observations", [])
        state_success = state.get("success", [])
        state_rewards = state.get("rewards", [])
        trajectories = zip(
            state_actions, 
            state_observations, 
            state_success
        )
        state_answers = state.get("answers", [])
        prev_infos = state_answers[-1] if state_answers else "No information yet"
        return f"""
        You are an AI agent designed to assist with a specific task.
        Previous agents have provided the following information:
        {prev_infos}
        Your need to follow instructions:
        {self.instruct_prompt}
        Avoid making overly complex code for simple tasks. Be patient and thorough.
        Do not make assumptions about the data returned by the tools. Try a tool, see its output, then you might write code to process it.
        If encountering rate limits, timeout, or processing time issues, you might use a while loop with state checks, retries, or exponential backoff strategies.
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
        try:
            result = self.agent.run(instructions)
        except Exception as e:
            print(f"Error running agent: {e}")
            return {
                **state,
                "actions": state["actions"] + ["LLM request"],
                "observations": state["observations"] + [str(e)],
                "rewards": state["rewards"] + [0.0],
                "success": state["success"] + [False],
                "answers": state["answers"] + ["Error in step execution."],
            }
        actions, observations, rewards, success = self.parse_memory_output()
        action: Action = { # Only the last action matters for the state
            "tool": actions[-1] if actions else "No action",
        }
        obs: Observation = { # Only the last observation matters for the state
            "data": observations[-1] if observations else "No observation"
        }
        reward = sum(rewards) if rewards else 0.0
        success_bool = success[-1] if len(success) > 0 else True # return True if final answer was called (no tool called, so array is empty).
        return {
            **state,
            "actions": state["actions"] + [action],
            "observations": state["observations"] + [obs],
            "rewards": state["rewards"] + [reward],
            "success": state["success"] + [success_bool],
            "answers": state["answers"] + [result],
        }

class WorkflowNodeFactory:
    @staticmethod
    def create_agent_node(agent_factory: SmolAgentFactory) -> Callable[[WorkflowState], dict]:
        def node_function(state: WorkflowState) -> dict:
            return agent_factory.run(state)
        return node_function


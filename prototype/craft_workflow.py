import uuid

def get_tools_code() -> str:
    """Create tools setup code for the sandbox"""
    try:
        with open("tools_client/browser_tools.py", 'r') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Error reading tools setup: {str(e)}")

def craft_smolagent_node(goal_prompt: str) -> str:
    """Create SmolAgent node code for the workflow"""
    uuid_str = str(uuid.uuid4()).replace("-", "")
    tools_code = get_tools_code()
    smolagent_code = f"""
{tools_code}

from smolagents import CodeAgent, tool
from smolagents import (
    HfApiModel,
    DuckDuckGoSearchTool
)
from smolagents import CodeAgent, tool, HfApiModel

engine = HfApiModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token="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",
    max_tokens=5000,
)

agent = CodeAgent(
    tools=tools,
    model=engine,
    max_steps=10,
)
instruct = "You are a SmolAgent that can perform web searches, navigate pages, and fill forms. Use the tools provided to achieve your goal."
output = agent.run(f"Your goal is {goal_prompt}.")
print(output)
    """
    return smolagent_code, uuid_str

def craft_workflow(workflow_code: str, goal_prompt: str) -> tuple[str, str]:
    uuid_str = str(uuid.uuid4()).replace("-", "")
    tools_code = get_tools_code()
    complete_code = f"""
from smolagents import CodeAgent, tool
from smolagents import (
    HfApiModel,
    DuckDuckGoSearchTool
)
from smolagents import CodeAgent, tool, HfApiModel
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# pre-defined tools
{tools_code}

model = "Qwen/Qwen2.5-Coder-32B-Instruct"

class Action(TypedDict):
    tool: str
    inputs: dict

class Observation(TypedDict):
    data: Any

class WorkflowState(TypedDict):
    goal: str
    actions: List[Action]
    observations: List[Observation]
    rewards: List[float]

def parse_tool_output(output: str) -> Tuple[Optional[Action], Optional[Observation], Optional[float]]:
    action = None
    observation = None
    reward = None
    action_match = re.search(r'action:(\[.*?\])', output)
    obs_match = re.search(r'observation:(.*?)(?=\sreward:|$)', output)
    reward_match = re.search(r'reward:([0-9.]+)', output)
    if action_match:
        try:
            action_args = eval(action_match.group(1))
            action = {
                "tool": action_args[0].split('(')[0],
                "inputs": {"args": action_args}
            }
        except:
            pass
            
    if obs_match:
        observation = {"data": obs_match.group(1)}
        
    if reward_match:
        try:
            reward = float(reward_match.group(1))
        except:
            pass
    return action, observation, reward

def create_smolagent_node(state: WorkflowState) -> dict:
    instructions_template = '''
    You previously took the following actions: {{actions[:-1]}}.
    You last made the following observations: {{observations[:-1]}}.
    Current task: Continue working towards the goal using available tools.
    Goal: {{goal}}.
    '''

    agent = CodeAgent(
        tools=tools, # tools list declared in browser_tools.py
        model=model,
        name=state["steps"][-1] if state["steps"] else "unknown",
        max_steps=3,
    )
    try:
        # Format the template with current state
        instructions = instructions_template.format(
            goal=state["goal"][-1] if state["goal"] else "complete the task",
            actions=state["actions"][-1] if state["actions"] else ["none"],
            observations=state["observations"][-1] if state["observations"] else ["none"]
        )
        result = agent.run(instructions)
        # parse the str result to extract action, observation, and reward
        action, obs, reward = parse_tool_output(result)

        action: Action = {{
            "tool": tool_name,
            "inputs": inputs,
            "outputs": data
        }}
        obs: Observation = {{
            "data": result
        }}
        return {{
            "goal": state["goal"],
            "actions": state["actions"] + [action],
            "observations": state["observations"] + [obs],
            "rewards": state["rewards"] + [reward],
        }}
    except Exception as e:
        return {{
            "goal": state["goal"],
            "actions": state["actions"] + None,
            "observations": state["observations"] + None,
            "rewards": state["rewards"] + [0],
        }}

# LLM generated graph code
{workflow_code}

initial_state: WorkflowState = {{
    "goal": ["{goal_prompt}"],
    "actions": [],
    "observations": [],
    "rewards": [],
}}

result_state = app.invoke(initial_state)
print(result_state)
    """
    print("\nGenerated code for the workflow:")
    print("=" * 50)
    print(complete_code)
    print("=" * 50)
    print()
    return complete_code, uuid_str
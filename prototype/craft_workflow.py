import uuid

def get_codefile(path = "") -> str:
    """Create tools setup code for the sandbox"""
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Error reading file at {path}: {str(e)}")

def craft_workflow(workflow_code: str, goal_prompt: str) -> tuple[str, str]:
    uuid_str = str(uuid.uuid4()).replace("-", "")
    tools_code = get_codefile("tools_client/browser_tools.py")
    schema_code = get_codefile("./schema_factory.py")
    smolagent_code = get_codefile("./smolagent_factory.py")
    complete_code = f'''
from smolagents import CodeAgent, tool, HfApiModel
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Tuple, Any, Dict, Union, Optional, Callable
import json

# pre-defined tools
{tools_code}

# schema for the workflow state
{schema_code}

# smolagent factory
{smolagent_code}

# LLM generated logical multi-agent graph
{workflow_code}

initial_state: WorkflowState = {{
    "goal": ["{{goal_prompt}}"],
    "actions": [],
    "observations": [],
    "rewards": [],
}}

result_state = app.invoke(initial_state)
print(result_state)
    '''
    print("\nGenerated code for the workflow:")
    print("=" * 50)
    print(complete_code)
    print("=" * 50)
    print()
    return complete_code, uuid_str





##########
# Testing Functions
##########



def craft_smolagent(goal_prompt: str) -> str:
    """Create simple SmolAgent for testing purposes"""
    uuid_str = str(uuid.uuid4()).replace("-", "")
    tools_code = get_codefile()
    smolagent_code = f"""
from typing import List, Any
import requests
from smolagents import (
    HfApiModel,
    CodeAgent,
    Tool
)

model = "Qwen/Qwen2.5-Coder-32B-Instruct"

{tools_code}

engine = HfApiModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token="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",
    max_tokens=5000,
)
agent = CodeAgent(
    tools=tools, # tools list declared in browser_tools.py
    model=engine,
    name="agent",
    description="A code agent to assist with the task.",
    max_steps=3,
)
instructions = "Browse the web for cheap flights to Paris and summarize the findings."
result = agent.run(instructions)

    """
    return smolagent_code, uuid_str
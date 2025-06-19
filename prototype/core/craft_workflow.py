import uuid
import os
from core.provider import openai_fn, deepseek_fn

def get_system_prompt() -> str:
    try:
        with open("prompts/workflow_creator.md", 'r') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"get_system_prompt: Error:\n{str(e)}")

def llm_make_workflow(goal_prompt: str, existing_tool_prompt: str) -> str:
    """
    Use LLM to generate LangGraph workflow code with SmolAgent nodes
    """
    system_prompt = get_system_prompt()
    prompt = f"""
You are an expert in generating LangGraph workflows using SmolAgent nodes.

The following set of tools are availables for agents, it replaces EXISTING_TOOLS with more tailored package of tools.
A set of tools is a LIST of tools that could be used by an agent. Be careful not to put them within a list (list within list cause error). 

{existing_tool_prompt}

You could however combine tool set like so:
MY_TOOLS = COFFEE_MACHINE_TOOL + CLEANING_TOOL + DUMMY_TOOL

Your task is to create a LangGraph workflow that achieves the following goal:

{goal_prompt}
    """
    history = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
    return openai_fn(history)

def extract_python_code(code: str) -> str:
    """Extract Python code blocks from the given text."""
    code_blocks = []
    in_code_block = False
    for line in code.splitlines():
        if line.startswith("```python"):
            in_code_block = True
            continue
        if line.startswith("```") and in_code_block:
            in_code_block = False
            continue
        if in_code_block:
            code_blocks.append(line)
    return "\n".join(code_blocks)

def get_codefile(path = "") -> str:
    """Create tools setup code for the sandbox"""
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Error reading file at {path}: {str(e)}")

def load_tools_client() -> str:
    """Load the tools client code from all Python files in the tools_client directory"""
    tools_code = ""
    existing_tool_prompt = ""
    tools_client_dir = "core/tools_client"
    
    if not os.path.exists(tools_client_dir):
        return tools_code
    
    for filename in os.listdir(tools_client_dir):
        if not filename.endswith('.py'):
            continue
        filepath = os.path.join(tools_client_dir, filename)
        base_name = os.path.splitext(filename)[0]
        # Add the file content
        tools_code += get_codefile(filepath)
        # Add a tool variable for this file
        tool_var_name = f"{base_name.upper()}_TOOL"
        tools_code += f"\n{tool_var_name} = tools\n"
        existing_tool_prompt += f"{tool_var_name}\n"
    
    return tools_code, existing_tool_prompt

def load_factory_code() -> str:
    """Load the SmolAgent factory code"""
    return get_codefile("core/smolagent_factory.py")

def save_code_to_file(code: str, file_path: str = "workflow0.py") -> None:
    try:
        with open(file_path, 'w') as f:
            f.write(code)
    except Exception as e:
        raise ValueError(f"save_code_to_file: Error saving code to file '{file_path}': {str(e)}")
    print(f"✅ Code saved to {file_path}")

def create_workflow_code(goal_prompt, existing_tool_prompt) -> str:
    print("🧠 Generating workflow code with LLM...")
    llm_output = llm_make_workflow(goal_prompt, existing_tool_prompt)
    workflow_llm = extract_python_code(llm_output)
    if workflow_llm is None or workflow_llm.strip() == "":
        raise ValueError("LLM did not return any code")
    print("✅ LLM generated workflow code successfully.")
    return workflow_llm

def create_folder_structure(uuid_str: str) -> None:
    """Create the necessary folder structure for the workflow"""
    workflow_save_path = f"workflows/{uuid_str}/"
    os.makedirs("workflows", exist_ok=True)
    os.makedirs(f"workflows/{uuid_str}/", exist_ok=True)
    print(f"✅ Folder structure created: workflows/{uuid_str}/")
    return workflow_save_path

def craft_workflow(goal_prompt: str) -> tuple[str, str]:
    uuid_str = str(uuid.uuid4()).replace("-", "")
    path = create_folder_structure(uuid_str)
    tools_code, existing_tool_prompt = load_tools_client()
    factory_code = load_factory_code()
    workflow_code = create_workflow_code(goal_prompt, existing_tool_prompt)
    complete_code = f'''
from smolagents import CodeAgent, tool, HfApiModel
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Tuple, Any, Dict, Union, Optional, Callable
import json

# load tools client code
{tools_code}

# smolagent + state schema factory
{factory_code}

# LLM generated logical multi-agent graph
{workflow_code}

initial_state: WorkflowState = {{
    "step_name": ["Initial Step"],
    "actions": [],
    "observations": [],
    "rewards": [],
    "answers": [],
    "success": []
}}

try:
    png = app.get_graph().draw_mermaid_png()
    path_graph = os.path.join("./", "workflow_graph.png")
    with open(path_graph, "wb") as f:
        f.write(png)
except Exception as e:
    print(f"Could not save workflow graph.")

result_state = app.invoke(initial_state)
print(result_state)

path_json = os.path.join("{path}", "state_result.json")
try:
    with open(path_json, "w") as f:
        json.dump(result_state, f, indent=2)
except Exception as e:
    print(f"Could not save workflow data: {{e}}")
'''
    code_path = os.path.join(path, f"workflow_code.py")
    save_code_to_file(complete_code, code_path)
    return complete_code
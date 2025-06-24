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
Each set of tools is a LIST of tools that could be used by an agent. Be careful not to put them within a list (list within list cause error). 

{existing_tool_prompt}

You could however combine tool set like so:
AGENT_TOOLS = COFFEE_MACHINE_TOOL + CLEANING_TOOL + DUMMY_TOOL

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
    except FileNotFoundError as e:
        raise ValueError(f"File not found at {path}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error reading file at {path}: {str(e)}")

def load_tools_code(tools_dir) -> None:
    """Validate that the tools client directory exists and contains Python files"""
    if not os.path.exists(tools_dir):
        raise ValueError(f"Tools client directory '{tools_dir}' does not exist.")
    if not any(filename.endswith('.py') for filename in os.listdir(tools_dir)):
        raise ValueError(f"No Python files found in tools client directory '{tools_dir}'.")
    
    tools_code = []
    for filename in os.listdir(tools_dir):
        if not filename.endswith('.py'):
            continue
        filepath = os.path.join(tools_dir, filename)
        base_name = os.path.splitext(filename)[0]
        tools_code.append((base_name, get_codefile(filepath)))
    return tools_code

def load_tools() -> str:
    """Load the tools client code from all Python files in the tools directory"""
    tools_code = ""
    existing_tool_prompt = ""
    tools_dir = "modules/tools"

    loaded_tool = load_tools_code(tools_dir)
    print(f"✅ Loaded {len(loaded_tool)} tools from {tools_dir}")
    for base_name, code in loaded_tool:
        tools_code += code + '\n'
        # Add a special tool variable that copy the tools variable of the current file
        tool_var_name = f"{base_name.upper()}_TOOLS"
        tools_code += f"\n{tool_var_name} = tools\n"
        existing_tool_prompt += f"{tool_var_name}\n"
    
    return tools_code, existing_tool_prompt

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

def craft_workflow(goal_prompt: str, template_workflow=None, save_workflow=True) -> tuple[str, str]:
    path = ""
    uuid_str = str(uuid.uuid4()).replace("-", "")
    tools_code, existing_tool_prompt = load_tools()
    state_schema_code = get_codefile("modules/state_schema.py")
    smolagent_code = get_codefile("modules/smolagent_factory.py")
    if template_workflow is None:
        path = create_folder_structure(uuid_str)
        workflow_code = create_workflow_code(goal_prompt, existing_tool_prompt)
    else:
        workflow_code = template_workflow
        save_workflow = False
    complete_code = f'''
from smolagents import CodeAgent, tool, HfApiModel
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Tuple, Any, Dict, Union, Optional, Callable
import json

# load tools client code
{tools_code}

# load state schema code
{state_schema_code}

# smolagent factory code
{smolagent_code}

# LLM generated multi-agent workflow code
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
    path_graph = os.path.join("{path}", "workflow_{uuid_str}.png")
    with open(path_graph, "wb") as f:
        f.write(png)
except Exception as e:
    print(f"Could not save workflow graph:" + str(e))

result_state = app.invoke(initial_state)
print(result_state)

path_json = os.path.join("{path}", "state_result_{uuid_str}.json")
try:
    with open(path_json, "w") as f:
        json.dump(result_state, f, indent=2)
except Exception as e:
    print(f"Could not save workflow data:" + str(e))
'''
    code_path = os.path.join(path, f"workflow_code.py")
    save_code_to_file(complete_code, ".debug.py")
    if save_workflow:
        save_code_to_file(workflow_code, code_path)
    return complete_code
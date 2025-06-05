#!/usr/bin/env python3
"""
Meta-Agent Prototype: LLM-Generated LangGraph Workflows with SmolAgent Nodes
============================================================================

This prototype demonstrates the core concept of using an LLM to generate
LangGraph workflows that use SmolAgent instances as nodes.
"""

import os
import json
from typing import Dict, Any, List
from e2b_code_interpreter import Sandbox
from anthropic import Anthropic
import openai
from openai import OpenAI

from tools_client.browser_tools import tools_prompt
from craft_workflow import craft_workflow, craft_smolagent_node

model_choice = "openai"  # Change to "deepseek" or "anthropic" as needed

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

def deepseek_fn(history, verbose=False):
    """
    Use deepseek api to generate text.
    """
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=history,
            stream=False
        )
        thought = response.choices[0].messag.econtent
        if verbose:
            print(thought)
        return thought
    except Exception as e:
        raise Exception(f"Deepseek API error: {str(e)}") from e

def openai_fn(history, verbose=False):
    """
    Use openai to generate text.
    """
    client = OpenAI(api_key=openai_api_key)

    try:
        response = client.chat.completions.create(
            model="o3-2025-04-16",
            messages=history,
        )
        if response is None:
            raise Exception("OpenAI response is empty.")
        thought = response.choices[0].message.content
        if verbose:
            print(thought)
        return thought
    except Exception as e:
        raise Exception(f"OpenAI API error: {str(e)}") from e

def create_sandbox():
    """Create and configure the sandbox with all required dependencies"""
    print("📦 Setting up sandbox environment...")
    sandbox = Sandbox()
    requirements_file = "./sandbox_requirements.txt"
    if os.path.exists(requirements_file):
        with open(requirements_file, 'r') as f:
            requirements = f.read().strip().split('\n')
        for requirement in requirements:
            requirement = requirement.strip()
            if requirement and not requirement.startswith('#'):
                print(f"Installing {requirement}...")
                sandbox.commands.run(f"pip install {requirement}")
    else:
        raise FileNotFoundError(f"Requirements file '{requirements_file}' not found.")
    return sandbox

def run_code_sandbox(sandbox, code: str, verbose: bool = False) -> str:
    """Execute code in sandbox and raise errors if execution fails"""
    # Upload all files from tools folder to sandbox
    tools_folder = "tools"
    if os.path.exists(tools_folder):
        for root, dirs, files in os.walk(tools_folder):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        file_content = f.read()
                    # Create the same directory structure in sandbox
                    relative_path = os.path.relpath(file_path, '.')
                    sandbox.files.write(relative_path, file_content)
    execution = sandbox.run_code(
        code,
        envs={'HF_TOKEN': os.getenv('HF_TOKEN')}
    )
    if execution.error:
        execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
        logs = execution_logs
        logs += execution.error.traceback
        raise ValueError(logs)
    return "\n".join([str(log) for log in execution.logs.stdout])

def get_system_prompt() -> str:
    try:
        with open("prompts/workflow_creator.md", 'r') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Error reading system prompt: {str(e)}")

def llm_make_workflow(tools_prompt: str) -> str:
    """
    Use LLM to generate LangGraph workflow code with SmolAgent nodes
    """
    system_prompt = get_system_prompt()

    prompt = tools_prompt
    history = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}]
    if model_choice == "deepseek":
        return deepseek_fn(history)
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

def save_code_to_file(code: str, filename: str = "workflow0.py") -> None:
    folder_path = "generated/"
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'w') as f:
        f.write(code)

def main():
    """Main execution function"""
    print("🚀 Starting Meta-Agent Prototype...")
    
    goal_prompt = "Search the web for the latest news on AI advancements and summarize the findings."
    if not os.getenv('HF_TOKEN'):
        raise ValueError("HF_TOKEN environment variable is not set. Please set it to your Hugging Face token.")
    
    sandbox = create_sandbox()
    print("🧠 Generating workflow code with LLM...")
    try:
        # llm make langraph workflow given prompt workflow_creator.md and list of tools
        llm_output = "commented out for now, use llm_make_workflow(tools_prompt) to generate workflow code"
        #llm_output = llm_make_workflow(tools_prompt)
        workflow_llm = extract_python_code(llm_output)
        if workflow_llm is None or workflow_llm.strip() == "":
            raise ValueError("LLM did not return any code")
        print("\n🔧 Executing generated workflow in sandbox...")
        exec_code, uuid_str = craft_smolagent_node(
            goal_prompt
        )
        #exec_code, uuid_str = craft_workflow(
        #    workflow_llm,
        #    goal_prompt=goal_prompt
        #)
        save_code_to_file(exec_code, filename=f"workflow_{uuid_str}.py")
        execution_logs = run_code_sandbox(sandbox, exec_code, verbose=True)
        print("\n📊 Execution Results:")
        print(execution_logs)
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up sandbox...")
        try:
            sandbox.close()
        except:
            pass

if __name__ == "__main__":
    main()

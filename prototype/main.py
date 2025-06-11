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
from anthropic import Anthropic
import openai
from openai import OpenAI

from craft_workflow import craft_workflow, craft_smolagent
#from smolagents.local_python_executor import LocalPythonExecutor
import subprocess


model_choice = "deepseek"  # Change to "deepseek" or "anthropic" as needed

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
            model="deepseek-reasoner",
            messages=history,
            stream=False
        )
        thought = response.choices[0].message.content
        if verbose:
            print(thought)
        return thought
    except Exception as e:
        raise Exception(f"deepseek_fn: Deepseek API error: {str(e)}") from e

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
            raise Exception("openai_fn: OpenAI response is empty.")
        thought = response.choices[0].message.content
        if verbose:
            print(thought)
        return thought
    except Exception as e:
        raise Exception(f"openai_fn: OpenAI API error: {str(e)}") from e

def get_system_prompt() -> str:
    try:
        with open("prompts/workflow_creator.md", 'r') as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"get_system_prompt: Error:\n{str(e)}")

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
    try:
        with open(file_path, 'w') as f:
            f.write(code)
    except Exception as e:
        raise ValueError(f"save_code_to_file: Error saving code to file '{file_path}': {str(e)}")

def main():
    """Main execution function"""
    print("🚀 Starting Meta-Agent Prototype...")
    
    goal_prompt = "Goal: Search the web for the latest news on AI advancements and summarize the findings."
    if not os.getenv('HF_TOKEN'):
        raise ValueError("HF_TOKEN environment variable is not set. Please set it to your Hugging Face token.")

    print("🧠 Generating workflow code with LLM...")
    try:
        # llm make langraph workflow given prompt workflow_creator.md and list of tools
        llm_output = llm_make_workflow(goal_prompt)
        workflow_llm = extract_python_code(llm_output)
        print("\n💡 LLM :")
        print(llm_output)
        if workflow_llm is None or workflow_llm.strip() == "":
            raise ValueError("LLM did not return any code")
        print("\n🔧 Executing generated workflow in sandbox...")
        exec_code, uuid_str = craft_workflow(
            workflow_llm,
            goal_prompt=goal_prompt
        )
        result = subprocess.run(["python", "-c", exec_code], capture_output=True, text=True)
        save_code_to_file(exec_code, filename=f"workflow_{uuid_str}.py")
        print("\n📊 Execution Results:")
        print(result.stdout)
        print("\n🔍 Errors (if any):")
        print(result.stderr)
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up sandbox...")

if __name__ == "__main__":
    main()

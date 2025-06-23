#!/usr/bin/env python3
"""
Meta-Agent Prototype: LLM-Generated LangGraph Workflows with SmolAgent Nodes
============================================================================

This prototype demonstrates the core concept of using an LLM to generate
LangGraph workflows that use SmolAgent instances as nodes.
"""

import os, sys
from core.craft_workflow import craft_workflow
import subprocess

def executor(code: str) -> str:
    print("\n🔧 Executing generated workflow in sandbox...")
    
    # Create a process and stream output in real-time
    process = subprocess.Popen(
        ["python", "-c", code], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    print("\n📊 Execution Progress:")
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    stderr_output = process.stderr.read()
    process.wait()
    print("\nExited.\n🔍 Errors (if any):")
    print(stderr_output)
    return process.returncode

def select_workflow_template() -> str:
    workflow_dir = "workflows"
    if not os.path.exists(workflow_dir):
        return None
    workflows = [f for f in os.listdir(workflow_dir)]
    print("Available workflow templates:")
    for workflow in workflows:
        print(f"- {workflow}")
    workflow_uuid = input("Enter the workflow UUID to load (or press Enter to skip):").strip()
    try:
        with open(f"workflows/{workflow_uuid}/workflow_code.py", 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading workflow code: {e}")
        return None

def main():
    """Main execution function"""
    goal_prompt = """
Search how to install the gpaw sotware on a Macos system with M1 chip. Then proceed to install it.
    """
    if not os.getenv('HF_TOKEN'):
        raise ValueError("HF_TOKEN environment variable is not set. Please set it to your Hugging Face token.")
    try:
        exec_code = craft_workflow(
            goal_prompt,
            template_workflow=select_workflow_template()
        )
        executor(exec_code)
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up sandbox...")

if __name__ == "__main__":
    main()

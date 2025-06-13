#!/usr/bin/env python3
"""
Meta-Agent Prototype: LLM-Generated LangGraph Workflows with SmolAgent Nodes
============================================================================

This prototype demonstrates the core concept of using an LLM to generate
LangGraph workflows that use SmolAgent instances as nodes.
"""

import os
from craft_workflow import craft_workflow
import subprocess

def executor(code: str) -> str:
    print("\n🔧 Executing generated workflow in sandbox...")
    result = subprocess.run(["python", "-c", code], capture_output=True, text=True)
    print("\n📊 Execution Results:")
    print(result.stdout)
    print("\n🔍 Errors (if any):")
    print(result.stderr)

def main():
    """Main execution function"""
    goal_prompt = "Draft an agent flow for making a web agent that can search the web, navigate to a URL, and extract information from it."
    if not os.getenv('HF_TOKEN'):
        raise ValueError("HF_TOKEN environment variable is not set. Please set it to your Hugging Face token.")
    try:
        exec_code = craft_workflow(
            goal_prompt
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

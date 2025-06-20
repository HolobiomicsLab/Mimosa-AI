#!/usr/bin/env python3
"""
Meta-Agent Prototype: LLM-Generated LangGraph Workflows with SmolAgent Nodes
============================================================================

This prototype demonstrates the core concept of using an LLM to generate
LangGraph workflows that use SmolAgent instances as nodes.
"""

import os
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

def main():
    """Main execution function"""
    goal_prompt = """
Compile a CSV file listing upcoming community events in Austin, TX, for July and August 2025, suitable for a food truck vendor to participate in. For each event, include the event name, date, location, estimated attendance, vendor application deadline, application fee (if any), and a link to the application page. Collect data for at least 15 events. Add a column indicating whether the event is ‘High Potential’
    """
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

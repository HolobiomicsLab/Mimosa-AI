#!/usr/bin/env python3
"""
Mimosa - A AI Agent Framework for advancing scientific research
============================================================================
"""

import os, sys
import argparse
from core.craft_workflow import craft_workflow
from core.runner import runner as code_runner
from typing import Optional

def select_workflow_template(workflow_uuid: Optional[str] = None) -> str:
    workflow_dir = "workflows"
    if not os.path.exists(workflow_dir):
        return None
    workflows = [f for f in os.listdir(workflow_dir)]
    if not workflows:
        return None
    if workflow_uuid is None:
        # TODO implement a auto-selection mechanism for available workflows
        return None
    try:
        with open(f"workflows/{workflow_uuid}/workflow_code.py", 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"Workflow template for UUID {workflow_uuid} not found in {workflow_dir}.")
    except Exception as e:
        raise ValueError(f"Error reading workflow template: {str(e)}")

def orchestrate_workflow(goal_prompt: str, workflow_uuid: Optional[str] = None) -> str:
    try:
        exec_code = craft_workflow(
            goal_prompt,
            template_workflow=select_workflow_template(workflow_uuid=workflow_uuid),
        )
        code_runner(exec_code)
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up sandbox...")

def validate_environment() -> None:
    """Ensure the environment is set up correctly"""
    if not os.getenv('HF_TOKEN'):
        raise ValueError("HF_TOKEN environment variable is not set. Please set it to your Hugging Face token.")
    if not os.path.exists("modules/tools"):
        raise ValueError("Tools directory 'modules/' does not exist. Please ensure it is present.")
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it to your OpenAI API key.")

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description="Mimosa - A AI Agent Framework for advancing scientific research")
    parser.add_argument("--goal", required=True, type=str, help="Goal prompt for the workflow")
    parser.add_argument("--load_template", type=str, help="Optional workflow UUID to load")
    
    args = parser.parse_args()
    
    validate_environment()
    orchestrate_workflow(goal_prompt=args.goal, workflow_uuid=args.load_template)

if __name__ == "__main__":
    main()
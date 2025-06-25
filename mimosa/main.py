#!/usr/bin/env python3
"""
Mimosa - A AI Agent Framework for advancing scientific research
============================================================================
"""

import os
import sys
import argparse
import requests
from typing import Optional

from core.craft_workflow import WorkflowCrafting
from core.runner import WorkflowRunner

class MimosaApp:
    """Main application class for the Mimosa AI Agent Framework.
    
    Attributes:
        workflow_dir (str): Directory containing workflow templates
    """
    
    def __init__(self, workflow_dir: str = "workflows") -> None:
        """Initialize the Mimosa application.
        
        Args:
            workflow_dir: Path to directory containing workflow templates
        """
        self.workflow_dir = workflow_dir

    def select_workflow_template(self, workflow_uuid: Optional[str] = None) -> str:
        """Select and load a workflow template by UUID.
        
        Args:
            workflow_uuid: Optional UUID of workflow template to load
        Returns:
            str: The workflow template content if found, None otherwise
        """
        if not os.path.exists(self.workflow_dir):
            return None
        workflows = [f for f in os.listdir(self.workflow_dir)]
        if not workflows:
            return None
        if workflow_uuid is None:
            # TODO implement a auto-selection mechanism for available workflows
            return None
        try:
            with open(f"{self.workflow_dir}/{workflow_uuid}/workflow_code_{workflow_uuid}.py", 'r') as f:
                return f.read()
        except FileNotFoundError:
            raise ValueError(f"❌ Workflow template for UUID {workflow_uuid} not found in {self.workflow_dir}.")
        except Exception as e:
            raise ValueError(f"❌ Error reading workflow template: {str(e)}")

    def orchestrate_workflow(self, goal_prompt: str,
                                   workflow_uuid: Optional[str] = None,
                                   python_version: str = "3.10") -> str:
        """Execute a workflow with the given goal prompt.
        
        Args:
            goal_prompt: The goal description for the workflow
            workflow_uuid: Optional UUID of a workflow template to load
        Returns:
            str: Execution status message
        """
        workflow_runner = WorkflowRunner(python_version=python_version)
        workflow_crafter = WorkflowCrafting(tools_dir="modules/tools", workflow_dir=self.workflow_dir)
        try:
            workflow_code = workflow_crafter.craft_workflow(
                goal_prompt,
                template_workflow=self.select_workflow_template(workflow_uuid=workflow_uuid),
                save_workflow=(workflow_uuid is not None),
            )
            workflow_runner.run(workflow_code)
            return "Workflow executed successfully"
        except Exception as e:
            print(f"❌ Error during execution: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Workflow execution failed: {str(e)}")
        finally:
            print("\nCleaning up sandbox...")
    
    def ping_mcp_server(self) -> None:
        """Ping the MCP server to ensure it is running.
        """
        try:
            response = requests.get("http://localhost:5000/health", timeout=5)
            if response.status_code != 200:
                raise RuntimeError("\n❌ MCP server is not reachable. Please start the server.")
        except (requests.ConnectionError, requests.Timeout, requests.RequestException) as e:
            raise RuntimeError(f"❌ Failed to connect to Tools MCP server. Please ensure it is running.")
        except ConnectionRefusedError as e:
            raise RuntimeError(f"❌ Connection refused when trying to reach MCP server: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"❌ An unexpected error occurred while pinging MCP server: {str(e)}")

    def install_dependencies(self, python_version: str = "3.10", requirement_path: str = "sandbox_requirement.txt") -> None:
        """Install required dependencies for python code runner."""
        print("\n🔧 Installing dependencies for workflow code runner...")
        try:
            from subprocess import call
            call([f"python{python_version}", "-m", "pip", "install", "-r", requirement_path])
            print("✅ Dependencies installed successfully.")
        except Exception as e:
            raise RuntimeError(f"❌ Failed to install dependencies: {str(e)}")

    def validate_environment(self) -> None:
        """Validate required environment configuration.
        """
        if not os.getenv('HF_TOKEN'):
            raise ValueError("⚠️ HF_TOKEN environment variable is not set. Please set it to your Hugging Face token.")
        if not os.path.exists("modules/tools"):
            raise ValueError("❌ Tools directory 'modules/' does not exist. Please ensure it is present.")
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("⚠️ OPENAI_API_KEY environment variable is not set. Please set it to your OpenAI API key.")
        self.ping_mcp_server()

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Mimosa - A AI Agent Framework for advancing scientific research")
    parser.add_argument("--goal", required=True, type=str, help="Goal prompt for the workflow")
    parser.add_argument("--load_template", type=str, help="Optional workflow UUID to load")
    
    args = parser.parse_args()
    
    app = MimosaApp()
    app.validate_environment()
    app.install_dependencies()
    app.orchestrate_workflow(goal_prompt=args.goal, workflow_uuid=args.load_template)


if __name__ == "__main__":
    main()

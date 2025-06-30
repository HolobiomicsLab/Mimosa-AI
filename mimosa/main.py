#!/usr/bin/env python3
"""
Mimosa - A AI Agent Framework for advancing scientific research
============================================================================
"""

import os
import asyncio
import argparse
import requests
import dotenv

from orchestrator import WorkflowOrchestrator
from config import Config

dotenv.load_dotenv()
    
def ping_mcp_server(endpoint: str) -> None:
    """Ping the MCP server to ensure it is running.
    """
    try:
        response = requests.get(endpoint, timeout=5)
        if response.status_code != 200:
            raise RuntimeError("\n❌ MCP server is not reachable. Please start the server.")
    except (requests.ConnectionError, requests.Timeout, requests.RequestException) as e:
        raise RuntimeError(f"❌ Failed to connect to Tools MCP server. Please ensure it is running.")
    except ConnectionRefusedError as e:
        raise RuntimeError(f"❌ Connection refused when trying to reach MCP server: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"❌ An unexpected error occurred while pinging MCP server: {str(e)}")
    print(" ✅ Connected to Tools MCP server successfully.")

def validate_environment() -> None:
    """Validate required environment configuration.
    """
    if not os.getenv('HF_TOKEN'):
        raise ValueError("⚠️ HF_TOKEN environment variable is not set. Please set it to your Hugging Face token.")
    if not os.path.exists("modules/tools"):
        raise ValueError("❌ Tools directory 'modules/' does not exist. Please ensure it is present.")
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("⚠️ OPENAI_API_KEY environment variable is not set. Please set it to your OpenAI API key.")

async def main():
    """Main execution function"""
    config = Config()
    parser = argparse.ArgumentParser(description="Mimosa - A AI Agent Framework for advancing scientific research")
    parser.add_argument("--goal", required=True, type=str, help="Goal prompt for the workflow")
    parser.add_argument("--load_template", type=str, help="Optional workflow UUID to load")
    args = parser.parse_args()

    validate_environment()
    ping_mcp_server(endpoint=config.mcp_health_endpoint)
    config.validate_paths()
    
    orchestrator = WorkflowOrchestrator(config)
    #await orchestrator.orchestrate_workflow(goal_prompt=args.goal, template_uuid=args.load_template)
    await orchestrator.recursive_self_improvement(goal_prompt=args.goal, template_uuid=args.load_template)

if __name__ == "__main__":
    asyncio.run(main())

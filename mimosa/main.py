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

from config import Config
from fastmcp import Client
import asyncio

from core.dgm import GodelMachine

dotenv.load_dotenv()

async def discover_mcp_servers():
    """Discover MCP servers on ports 5000-5050 and list their tools."""
    print("🔍 Discovering MCP servers on ports 5000-5050...")
    found_servers = False
    for port in range(5000, 5051):
        try:
            async with Client(f"http://localhost:{port}/mcp") as client:
                tools = await client.list_tools()
                if tools:
                    print(f"✅ Found MCP server on port {port}")
                    print(f"📋 Available tools: {[tool.name for tool in tools]}")
                    found_servers = True
        except Exception as e:
            continue
    if not found_servers:
        print("❌ No MCP servers found on ports 5000-5050. Please ensure at least one server is running.")
        raise RuntimeError("No MCP servers found. Please start Toolomics MCP server.")
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
    config.validate_paths()
    
    dgm = GodelMachine(config)
    await discover_mcp_servers()
    await dgm.recursive_self_improvement(goal_prompt=args.goal, template_uuid=args.load_template)

if __name__ == "__main__":
    asyncio.run(main())

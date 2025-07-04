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
from typing import Optional, List

from core.dgm import GodelMachine

dotenv.load_dotenv()

async def discover_mcp_servers() -> List[int]:
    """Discover MCP servers on ports 5000-5050 and list their tools."""
    print("🔍 Discovering MCP servers on ports 5000-5250...")
    found_servers = False
    ports = []
    for port in range(5000, 5251):
        try:
            async with Client(f"http://localhost:{port}/mcp") as client:
                tools = await client.list_tools()
                if tools:
                    print(f"✅ Found MCP server on port {port}")
                    print(f"📋 Available tools: {[tool.name for tool in tools]}")
                    found_servers = True
                    ports.append(port)
        except Exception as _:
            continue
    if not found_servers:
        print("❌ No MCP servers found on ports 5000-5100. Please ensure at least one server is running.")
        raise RuntimeError("No MCP servers found. Please start Toolomics MCP server.")
    print(" ✅ Connected to Tools MCP server successfully.")
    return ports

def verify_tools_module_ports(tools_dir: str, ports: List[int]) -> None:
    """Verify that all tools in the specified directory are compatible with the discovered MCP ports."""
    print("🔍 Verifying tools module compatibility with MCP ports...")
    for tool_file in os.listdir(tools_dir):
        if tool_file.endswith('.py'):
            try:
                content = open(os.path.join(tools_dir, tool_file), 'r').read()
            except Exception as e:
                print(f"❌ Error reading tool file {tool_file}: {str(e)}")
                continue
            lines = content.splitlines()
            port_lines = [line for line in lines if 'localhost' in line]
            port_line = port_lines[0]
            if not any(f"http://localhost:{port}" in port_line for port in ports):
                raise ValueError(f"❌ A Tool is using a port not in the discovered MCP ports: {port_line}\n")

    print("✅ All tools are compatible with the discovered MCP ports.")

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
    ports = await discover_mcp_servers()
    verify_tools_module_ports(config.tools_dir, ports)
    await dgm.start_dgm(goal_prompt=args.goal, template_uuid=args.load_template)

if __name__ == "__main__":
    asyncio.run(main())

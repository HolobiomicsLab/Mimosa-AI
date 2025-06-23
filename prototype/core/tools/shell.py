'''
This module provides a set of tools for interacting with a CSV management API.
It is loaded in a sandbox python environment as part of the crafted workflow.
'''

from typing import List, Any, Dict
import asyncio
import json
from fastmcp import Client

from smolagents import Tool

API_SHELL_TOOLS_URL = 'http://localhost:5003'

def build_formatted_output(action: str, observation: str, reward: float) -> str:
    action_formatted = action[:256].strip().replace('\n', ' - ')
    observation_formatted = observation[:2048].strip().replace('\n', ' - ')
    return f"""
action: {action_formatted}
observation: {observation_formatted}
reward: {reward}
"""

async def _async_tool_call(tool_name: str, params: dict) -> dict:
    async with Client(f"{API_SHELL_TOOLS_URL}/mcp") as client:
        tools = await client.list_tools()
        assert tool_name in tools, "Fatal Error: " + tool_name + " not in tools list for mcp at " + API_SHELL_TOOLS_URL
        buffer = await client.call_tool(tool_name, params)
        return json.loads(buffer[0].text)

class ExecuteBashCommand(Tool):
    name = "execute_bash_command"
    description = "Execute a bash command."
    inputs = {"command": {"type": "string", "description": "The bash command to execute."}}
    output_type = "string"

    def forward(self, command: str) -> str:
        action = f"execute_bash_command({command})"
        obs = ''
        reward = 0.0
        
        try:
            result = _async_tool_call("execute_command", {"command": command})
            
            if result and result.get('status') == 'success' and 'stdout' in result:
                obs = result.get('stdout', 'No output') 
                reward = 1.0
            else:
                obs = result.get('stderr', 'Command execution failed')
                reward = 0.0
        except Exception as e:
            obs = str(e)
        return build_formatted_output(action, obs, reward)

# Tool instances
create_csv_tool = ExecuteBashCommand()

tools = [
    create_csv_tool
]

tools_name = [tool.name for tool in tools]
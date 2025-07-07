'''
This module provides a set of tools for interacting with a CSV management API.
It is loaded in a sandbox python environment as part of the crafted workflow.
'''

from typing import List, Any, Dict
import asyncio
import json
from fastmcp import Client

from smolagents import Tool

API_SHELL_TOOLS_URL = 'http://localhost:5102'

class ShellTool(Tool):
    def __init__(self):
        super().__init__()

    def build_formatted_output(self, action: str, observation: str, reward: float) -> str:
        output = {
            "action": action[:256].strip().replace('\n', ' - '),
            "observation": observation[:4096],
            "reward": reward
        }
        return f"\n```json\n{json.dumps(output, indent=2)}\n```\n"

    async def _async_shell_tool_call(self, tool_name: str, params: dict) -> dict:
        async with Client(f"{API_SHELL_TOOLS_URL}/mcp") as client:
            tools = await client.list_tools()
            tool_names = [tool.name for tool in tools]
            assert tool_name in tool_names, "Fatal Error: " + tool_name + " not in tools list for mcp at " + API_SHELL_TOOLS_URL
            buffer = await client.call_tool(tool_name, params, timeout=1800)
            return json.loads(buffer[0].text)

class ExecuteBashCommand(ShellTool):
    name = "execute_bash_command"
    description = "Execute a bash command."
    inputs = {"command": {"type": "string", "description": "The bash command to execute. Do not specify interpreter or shell, just the command itself."}}
    output_type = "string"

    def forward(self, command: str) -> str:
        import asyncio
        action = f"execute_bash_command({command})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(self._async_shell_tool_call("execute_command", {"command": command}))
            
            if result and result.get('status') == 'success' and 'stdout' in result:
                stdout = result.get('stdout', '') 
                stderr = result.get('stderr', '')
                obs = stdout if stdout else stderr
                reward = 1.0
            else:
                obs = result.get('stderr', 'Command execution failed')
                reward = 0.0
        except Exception as e:
            obs = str(e)
        return self.build_formatted_output(action, obs, reward)

# Tool instances
create_csv_tool = ExecuteBashCommand()

tools = [
    create_csv_tool
]

tools_name = [tool.name for tool in tools]
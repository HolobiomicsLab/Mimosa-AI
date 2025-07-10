'''
This module provides a set of tools for interacting with a Shell management API.
It is loaded in a sandbox python environment as part of the crafted workflow.
The shell commands should not be use to execute R scripts, use the dedicated R script tool instead.
'''

API_SHELL_TOOLS_URL = 'http://localhost:5102'

from smolagents import MCPClient

param1 = {"url": f"{API_SHELL_TOOLS_URL}/mcp", "transport": "streamable-http"}
client = MCPClient(param1)
tools = client.get_tools()
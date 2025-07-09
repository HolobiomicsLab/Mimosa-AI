'''
This module provides a set of tools for interacting with a web browser instance.
It is loaded in a sandbox python environment as part of the crafted workflow.
'''

from smolagents import MCPClient

API_BROWSER_TOOLS_URL = 'http://localhost:5000'

param1 = {"url": f"{API_BROWSER_TOOLS_URL}/mcp", "transport": "streamable-http"}
client = MCPClient(param1)
tools = client.get_tools()
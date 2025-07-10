'''
This module provides a set of tools for interacting with a R script management API.
'''

API_R_TOOLS_URL = 'http://localhost:5001'

from smolagents import MCPClient

param1 = {"url": f"{API_R_TOOLS_URL}/mcp", "transport": "streamable-http"}
client = MCPClient(param1)
tools = client.get_tools()
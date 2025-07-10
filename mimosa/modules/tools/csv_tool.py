'''
This module provides a set of tools for interacting with a CSV management API.
'''

API_CSV_TOOLS_URL = 'http://localhost:5101'

from smolagents import MCPClient

param1 = {"url": f"{API_CSV_TOOLS_URL}/mcp", "transport": "streamable-http"}
client = MCPClient(param1)
tools = client.get_tools()
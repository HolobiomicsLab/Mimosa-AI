

import asyncio

from fastmcp import Client
import asyncio
from typing import Optional, List

class MCP:
    def __init__(self, name: Optional[str] = None, tools: List[str] = None, address: Optional[str] = None, port: Optional[int] = None):
        self.name = name
        self.tools = tools if tools is not None else []
        self.address = address
        self.port = port 

class ToolManager:
    """Manager for MCP tools discovery and management."""

    def __init__(self, mcps: Optional[List[MCP]] = None):
        self.mcps = mcps

    async def discover_mcp_servers(self, port_min: int = 5000, port_max: int = 5250, timeout: float = 2.0) -> List[int]:
        """Discover MCP servers on ports range with timeout handling."""
        print(f"🔍 Discovering MCP servers on ports {port_min}-{port_max}...")
        found_servers = False
        mcps = []

        for port in range(port_min, port_max + 1):
            try:
                async with Client(f"http://localhost:{port}/mcp", timeout=3.0) as client:
                    found_servers = True
                    tools = await client.list_tools()
                    name = None
                    try:
                        name = await client.call_tool("get_mcp_name", {})
                    except Exception as e:
                        print(f"⚠️ Failed to get name for MCP server on port {port}: {e}")
                    if tools:
                        print(f"✅ Found MCP server on port {port}")
                        print(f"📋 Available tools: {[tool.name for tool in tools]}")
                        mcps.append(MCP(name=name, tools=[tool.name for tool in tools], address="localhost", port=port))
            except asyncio.TimeoutError:
                print(f"❌ MCP server on port {port} timed out after {timeout}s")
                continue
            except Exception as e:
                continue
            
        if not found_servers:
            print(f"❌ No MCP servers found on ports {port_min}-{port_max}. Please ensure at least one server is running.")
            raise RuntimeError("No MCP servers found. Please start Toolomics MCP server.")

        print(f"✅ Connected to {len(mcps)} MCP server(s) successfully.")
        self.mcps = mcps
        return mcps

    def _get_client_variable_name(self, mcp: MCP) -> str:
        """Generate a variable name for the MCP client based on its name."""
        name = mcp.name if mcp.name else "Unknown"
        if hasattr(name, 'content') and name.content:
            name = name.content[0].text
        name = name.replace(" ", "_").upper()
        return name + "_TOOLS"

    def get_client_prompt(self, mcp: MCP) -> str:
        """Generate a prompt for the MCP client."""
        if not mcp.address or not mcp.port:
            raise ValueError("MCP address and port must be set.")
        if not mcp.tools:
            raise ValueError("MCP tools list cannot be empty.")
        tool_list_str = ", ".join(mcp.tools)
        name = self._get_client_variable_name(mcp)
        return f'''
        Tool {name} is a collection of tools with the following capabilities:
        {tool_list_str}
        '''
    
    def get_client_code(self, mcp: MCP) -> str:
        """Generate client code for a specific MCP server."""
        if not mcp.address or not mcp.port:
            raise ValueError("MCP address and port must be set.")
        if not mcp.tools:
            raise ValueError("MCP tools list cannot be empty.")
        api_url = f"http://{mcp.address}:{mcp.port}/mcp"
        name = self._get_client_variable_name(mcp)
        return f'''
from smolagents import MCPClient
params = {{"url": "{api_url}", "transport": "streamable-http"}}
client = MCPClient(params)
tools = client.get_tools()
{name} = tools
'''

if __name__ == "__main__":
    # Example usage
    tool_manager = ToolManager()
    mcps = asyncio.run(tool_manager.discover_mcp_servers())
    for mcp in mcps:
        client_code = tool_manager.get_client_code(mcp)
        print(client_code)  # This would print the generated client code for the MCP server
        client_prompt = tool_manager.get_client_prompt(mcp)
        print(client_prompt)  # This would print the generated client prompt for the MCP server
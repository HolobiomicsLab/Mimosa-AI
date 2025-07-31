"""
This class manages the discovery and interaction with MCP tools.
"""

import asyncio

from fastmcp import Client


class MCP:
    def __init__(
        self,
        name: str | None = None,
        tools: list[str] = None,
        address: str | None = None,
        port: int | None = None,
    ):
        self.name = name
        self.tools = tools if tools is not None else []
        self.address = address
        self.port = port


class ToolManager:
    """Manager for MCP tools discovery and management."""

    def __init__(self, config, mcps: list[MCP] | None = None):
        self.discovery_address = config.discovery_addresses
        self.mcps = mcps if mcps is not None else []

    async def discover_mcp_at_address(
        self,
        address: str,
        port_min: int = 5000,
        port_max: int = 5250,
        timeout: float = 2.0,
    ) -> list[int]:
        """Discover MCP servers on address and ports range with timeout handling."""
        found_servers = False
        mcps = []

        for port in range(port_min, port_max + 1):
            try:
                async with Client(
                    f"http://{address}:{port}/mcp", timeout=3.0
                ) as client:
                    found_servers = True
                    tools = await client.list_tools()
                    name = None
                    try:
                        resp = await client.call_tool("get_mcp_name", {})
                        if "content" in resp and resp.content:
                            name = resp.content[0].text
                        else:  # fallback because it randomly change ????
                            name = resp[0].text if resp else None
                    except Exception as e:
                        print(
                            f"⚠️ Failed to get name for MCP server on port {port}: {e}"
                        )
                        name = f"mcp_{port}"
                    assert name, "MCP name must be set"
                    if tools:
                        print(f"✅ Found MCP server on port {port} with name {name}")
                        print(f"📋 Available tools: {[tool.name for tool in tools]}")
                        mcps.append(
                            MCP(
                                name=name,
                                tools=[tool.name for tool in tools],
                                address=address,
                                port=port,
                            )
                        )
            except asyncio.TimeoutError:
                print(f"❌ MCP server on port {port} timed out after {timeout}s")
                continue
            except Exception:
                continue

        if not found_servers:
            print(
                f"❌ No MCP servers found on ports {port_min}-{port_max}. \
                Please ensure toolomics MCPs server is running."
            )
            raise RuntimeError(
                "No MCP servers found. Please start Toolomics MCP server."
            )
        self.mcps.extend(mcps)
        return mcps

    async def discover_mcp_servers(self, timeout: float = 2.0) -> list[MCP]:
        for addr in self.discovery_address:
            print(f"🔍 Discovering MCP servers at {addr.ip}...")
            try:
                mcps = await self.discover_mcp_at_address(
                    addr.ip, addr.port_min, addr.port_max, timeout
                )
                if mcps:
                    print(f"✅ Found {len(mcps)} MCP server(s) at {addr.ip}.")
                    return mcps
            except Exception as e:
                raise ValueError(
                    f"❌ Error discovering MCP servers at {addr.ip}, no MCPs found."
                ) from e

    def _get_client_variable_name(self, mcp: MCP) -> str:
        """Generate a variable name for the MCP client based on its name."""
        name = mcp.name
        name = name.replace(" ", "_").upper()
        return name + "_TOOLS"

    def get_client_prompt(self, mcp: MCP) -> str:
        """Generate a prompt for the MCP client."""
        assert isinstance(mcp, MCP), "Expected MCP instance"
        if not mcp.address or not mcp.port:
            raise ValueError("MCP address and port must be set.")
        if not mcp.tools:
            raise ValueError("MCP tools list cannot be empty.")
        tool_list_str = ", ".join(mcp.tools)
        name = self._get_client_variable_name(mcp)
        return f"""
        Tool {name} is a collection of tools with the following capabilities:
        {tool_list_str}
        """

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
        print(
            client_code
        )  # This would print the generated client code for the MCP server
        client_prompt = tool_manager.get_client_prompt(mcp)
        print(
            client_prompt
        )  # This would print the generated client prompt for the MCP server

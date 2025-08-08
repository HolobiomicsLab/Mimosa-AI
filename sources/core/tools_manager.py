"""
This class manages the discovery and interaction with MCP tools.
"""

import asyncio
import json
import subprocess
import re
from typing import Optional, Dict, Any

from fastmcp import Client


class MCP:
    def __init__(
        self,
        name: str | None = None,
        tools: list[str] = None,
        address: str | None = None,
        port: int | None = None,
        toolhive_name: str | None = None,
        transport: str = "streamable-http",
        discovery_url: str | None = None,
        client_url: str | None = None,
    ):
        self.name = name
        self.tools = tools if tools is not None else []
        self.address = address
        self.port = port
        self.toolhive_name = toolhive_name  # Store toolhive server name for management
        self.transport = transport  # Transport type: streamable-http, sse, stdio
        self.discovery_url = discovery_url  # URL used for discovery/testing
        self.client_url = client_url  # URL used for client connections


class ToolManager:
    """Manager for MCP tools discovery and management."""

    def __init__(self, config, mcps: list[MCP] | None = None):
        self.discovery_address = config.discovery_addresses
        self.mcps = mcps if mcps is not None else []
        self.use_toolhive = self._check_toolhive_available()
        
    def _check_toolhive_available(self) -> bool:
        """Check if ToolHive is available on the system."""
        try:
            result = subprocess.run(['thv', 'version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            return False
    
    async def discover_toolhive_servers(self) -> list[MCP]:
        """Discover all MCP servers running via ToolHive."""
        try:
            # Get list of running servers from ToolHive
            result = subprocess.run(['thv', 'list', '--format', 'json'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                print(f"❌ Failed to get ToolHive server list: {result.stderr}")
                return []
            
            # Parse JSON output
            servers_data = json.loads(result.stdout)
            mcps = []
            
            for server in servers_data:
                name = server.get('name', '')
                status = server.get('status', '')
                
                if status != 'running':
                    print(f"⚠️ ToolHive server {name} is not running (status: {status})")
                    continue
                
                # Extract connection info
                url = server.get('url', '')
                if not url:
                    print(f"⚠️ No URL found for ToolHive server {name}")
                    continue
                
                # Parse URL to extract address and port
                # URL format: http://address:port/sse#server-name
                import urllib.parse
                parsed = urllib.parse.urlparse(url)
                address = parsed.hostname or 'localhost'
                port = parsed.port
                
                if not port:
                    print(f"⚠️ Could not extract port from URL {url} for server {name}")
                    continue
                
                # For ToolHive SSE servers, we need to use the SSE endpoint for MCP communication
                # The url already contains the SSE endpoint
                sse_url = f"http://{address}:{port}/sse"
                
                # Try to connect and get tools
                try:
                    async with Client(sse_url, timeout=5.0) as client:
                        tools = await client.list_tools()
                        
                        # Get server name
                        server_name = None
                        try:
                            resp = await client.call_tool("get_mcp_name", {})
                            if "content" in resp and resp.content:
                                server_name = resp.content[0].text
                            else:
                                server_name = resp[0].text if resp else None
                        except Exception as e:
                            print(f"⚠️ Failed to get name for server {name}: {e}")
                            # Fallback to a readable name based on toolhive server name
                            server_name = name.replace('-', ' ').title() + ' MCP'
                        
                        if tools:
                            print(f"✅ Found ToolHive MCP server {name} ({server_name})")
                            print(f"📋 Available tools: {[tool.name for tool in tools]}")
                            
                            # For ToolHive servers, both discovery and client use the SSE endpoint
                            mcps.append(
                                MCP(
                                    name=server_name,
                                    tools=[tool.name for tool in tools],
                                    address=address,
                                    port=port,
                                    toolhive_name=name,
                                    transport="sse",  # ToolHive uses SSE transport
                                    discovery_url=sse_url,  # /sse for discovery
                                    client_url=sse_url  # /sse for client connections
                                )
                            )
                
                except Exception as e:
                    print(f"❌ Failed to connect to ToolHive server {name} at {url}: {e}")
                    continue
            
            return mcps
            
        except subprocess.TimeoutExpired:
            print("❌ ToolHive command timed out")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse ToolHive JSON output: {e}")
            return []
        except Exception as e:
            print(f"❌ Error discovering ToolHive servers: {e}")
            return []

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
                        
                        # Original servers use streamable-http with /mcp endpoint
                        mcp_endpoint = f"http://{address}:{port}/mcp"
                        
                        mcps.append(
                            MCP(
                                name=name,
                                tools=[tool.name for tool in tools],
                                address=address,
                                port=port,
                                transport="streamable-http",  # Original transport
                                discovery_url=mcp_endpoint,  # /mcp for discovery
                                client_url=mcp_endpoint  # /mcp for client too
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
            toolhive_hint = " Or start with ToolHive: ./start-toolhive.sh" if self.use_toolhive else ""
            raise RuntimeError(
                f"No MCP servers found. Please start Toolomics MCP server.{toolhive_hint}"
            )
        self.mcps.extend(mcps)
        return mcps

    async def discover_mcp_servers(self, timeout: float = 2.0) -> list[MCP]:
        # ToolHive-only discovery
        if not self.use_toolhive:
            raise RuntimeError("ToolHive is required. Please install ToolHive: curl -sSL https://get.toolhive.dev | sh")
        
        print("🔍 Discovering MCP servers via ToolHive...")
        try:
            mcps = await self.discover_toolhive_servers()
            if mcps:
                print(f"✅ Found {len(mcps)} MCP server(s) via ToolHive.")
                self.mcps.extend(mcps)
                return mcps
            else:
                print("⚠️ No running MCP servers found via ToolHive.")
                return []
        except Exception as e:
            print(f"❌ ToolHive discovery failed: {e}")
            raise RuntimeError(f"Failed to discover MCP servers via ToolHive: {e}") from e

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
        """Generate transport-aware client code for a specific MCP server."""
        if not mcp.address or not mcp.port:
            raise ValueError("MCP address and port must be set.")
        if not mcp.tools:
            raise ValueError("MCP tools list cannot be empty.")
        
        # Use client_url if available, otherwise fallback to discovery_url or construct one
        if mcp.client_url:
            api_url = mcp.client_url
        elif mcp.discovery_url:
            api_url = mcp.discovery_url
        else:
            # Fallback for backward compatibility
            api_url = f"http://{mcp.address}:{mcp.port}/mcp"
        
        name = self._get_client_variable_name(mcp)
        transport = mcp.transport
        
        # Generate transport-specific client code
        if transport == "sse":
            return f'''
from smolagents import MCPClient
params = {{"url": "{api_url}", "transport": "sse"}}
client = MCPClient(params)
tools = client.get_tools()
{name} = tools
'''
        else:  # streamable-http (default/fallback)
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

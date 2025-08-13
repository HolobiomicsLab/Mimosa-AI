"""
This class manages the discovery and interaction with MCP tools.
"""

import asyncio
import json
import subprocess
from typing import Any
from urllib.parse import urlparse, urlunparse

from fastmcp import Client


def normalize_mcp_endpoint(
    raw_url: str, transport: str
) -> tuple[str, str, dict[str, Any]]:
    """
    Normalize MCP endpoint URL and transport for consistent client usage.

    Returns (url, transport, extra_params)
    - Strips URL fragments (not sent over HTTP)
    - Forces /mcp/ endpoint path for streamable-http servers
    - Preserves /sse endpoint path for SSE servers

    Args:
        raw_url: Raw URL from ToolHive (may include fragment)
        transport: Transport type (sse, streamable-http)

    Returns:
        Tuple of (normalized_url, normalized_transport, extras_dict)

    Raises:
        ValueError: If transport type is unsupported
    """
    extras = {}

    parsed = urlparse(raw_url)
    # Capture fragment (container name) if present
    container = parsed.fragment if parsed.fragment else None

    # Strip fragment (HTTP clients won't send it anyway)
    parsed = parsed._replace(fragment="")

    # Normalize transport string
    t = transport.lower().strip()
    if t in ("stdio", "sse"):
        t = "sse"  # Keep SSE transport for ToolHive default servers
        # For SSE servers, keep original path (/sse) - don't force /mcp
        # SSE servers from ToolHive use /sse endpoint, not /mcp
    elif t in ("http", "streamable-http", "streamable"):
        t = "streamable-http"  # Use full name for Smolagents compatibility
        # Ensure /mcp path exists, but preserve original trailing slash preference
        if not parsed.path.endswith("/mcp") and not parsed.path.endswith("/mcp/"):
            # Only add /mcp if it's missing entirely
            base_path = parsed.path.rstrip("/")
            path = f"{base_path}/mcp" if base_path else "/mcp"
            parsed = parsed._replace(path=path)
        # Otherwise, keep the original path as-is (including trailing slash preference)
    else:
        raise ValueError(
            f"Unsupported transport: {transport}. Supported transports: 'sse', 'streamable-http'"
        )

    url = urlunparse(parsed)

    # Store container hint for logging/debugging
    if container:
        extras["container_hint"] = container
        print(
            f"⚠️ ToolHive container fragment '{container}' detected but fragments are not sent over HTTP. "
            f"Container selection may be ignored unless handled by the MCP server."
        )

    return url, t, extras


class Tool:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description


class MCP:
    def __init__(
        self,
        name: str | None = None,
        tools: list[Tool] = None,
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

    @property
    def tool_names(self) -> list[str]:
        """Get list of tool names for backwards compatibility."""
        return [tool.name for tool in self.tools]


class ToolManager:
    """Manager for MCP tools discovery and management."""

    def __init__(self, config, mcps: list[MCP] | None = None):
        self.discovery_address = config.discovery_addresses
        self.mcps = mcps if mcps is not None else []
        self.use_toolhive = self._check_toolhive_available()

    def _check_toolhive_available(self) -> bool:
        """Check if ToolHive is available on the system."""
        try:
            result = subprocess.run(
                ["thv", "version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.CalledProcessError,
        ):
            return False

    def _get_tools_with_descriptions(self, server_url: str) -> list[Tool]:
        """Get tools with descriptions using thv mcp list tools command."""
        try:
            result = subprocess.run(
                [
                    "thv",
                    "mcp",
                    "list",
                    "tools",
                    "--server",
                    server_url,
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                print(f"❌ Failed to get tools from {server_url}: {result.stderr}")
                return []

            # Parse JSON output
            try:
                data = json.loads(result.stdout)
                tools = []
                for tool_data in data.get("tools", []):
                    name = tool_data.get("name", "")
                    description = tool_data.get("description", "")
                    if name:  # Only add tools with valid names
                        tools.append(Tool(name, description))
                return tools
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON response from {server_url}: {e}")
                return []

        except subprocess.TimeoutExpired:
            print(f"❌ Timeout getting tools from {server_url}")
            return []
        except Exception as e:
            print(f"❌ Error getting tools from {server_url}: {e}")
            return []

    async def discover_toolhive_servers(self) -> list[MCP]:
        """Discover all MCP servers running via ToolHive."""
        try:
            # Get list of running servers from ToolHive
            result = subprocess.run(
                ["thv", "list", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                print(f"❌ Failed to get ToolHive server list: {result.stderr}")
                return []

            # Parse JSON output
            servers_data = json.loads(result.stdout)
            mcps = []

            for server in servers_data:
                name = server.get("name", "")
                status = server.get("status", "")

                if status != "running":
                    print(f"⚠️ ToolHive server {name} is not running (status: {status})")
                    continue

                # Extract connection info
                url = server.get("url", "")
                if not url:
                    print(f"⚠️ No URL found for ToolHive server {name}")
                    continue

                # Parse URL to extract address and port
                # URL format: http://address:port/sse#server-name
                import urllib.parse

                parsed = urllib.parse.urlparse(url)
                address = parsed.hostname or "localhost"
                port = parsed.port

                if not port:
                    print(f"⚠️ Could not extract port from URL {url} for server {name}")
                    continue

                # For ToolHive servers, auto-detect transport from URL path
                # The url from ToolHive may contain fragments, normalize it for consistent client usage
                raw_url = (
                    url  # Use the full URL from ToolHive (includes fragment if present)
                )

                # Auto-detect transport type from URL path
                parsed_url = urllib.parse.urlparse(raw_url)
                if "/sse" in parsed_url.path:
                    detected_transport = "sse"
                elif "/mcp" in parsed_url.path:
                    detected_transport = "streamable-http"
                else:
                    # Default based on URL pattern - if has fragment, likely SSE
                    detected_transport = (
                        "sse" if parsed_url.fragment else "streamable-http"
                    )

                normalized_url, normalized_transport, extras = normalize_mcp_endpoint(
                    raw_url, detected_transport
                )

                # Get tools with descriptions using thv command
                tools = self._get_tools_with_descriptions(raw_url)

                if tools:
                    # Get server name using FastMCP client (fallback for server name only)
                    server_name = None
                    try:
                        async with Client(normalized_url, timeout=5.0) as client:
                            resp = await client.call_tool("get_mcp_name", {})
                            if "content" in resp and resp.content:
                                server_name = resp.content[0].text
                            else:
                                server_name = resp[0].text if resp else None
                    except Exception as e:
                        print(f"⚠️ Failed to get name for server {name}: {e}")
                        # Fallback to a readable name based on toolhive server name
                        server_name = name.replace("-", " ").title() + " MCP"

                    print(f"✅ Found ToolHive MCP server {name} ({server_name})")
                    print(
                        f"📋 Available tools: {[tool.name for tool in tools]} with descriptions"
                    )
                    if extras.get("container_hint"):
                        print(f"🏷️ Container: {extras['container_hint']}")

                    # For ToolHive servers, store normalized URLs for consistent client usage
                    mcps.append(
                        MCP(
                            name=server_name,
                            tools=tools,  # Now storing Tool objects with descriptions
                            address=address,
                            port=port,
                            toolhive_name=name,
                            transport=normalized_transport,  # Normalized transport
                            discovery_url=normalized_url,  # Normalized URL for discovery
                            client_url=normalized_url,  # Normalized URL for client connections
                        )
                    )
                else:
                    print(f"⚠️ No tools found for ToolHive server {name}")

                # Remove the try-catch that was wrapping the tool discovery
                # since we're no longer using async FastMCP for tool listing

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
                # Try MCP endpoint with streamable-http transport
                raw_mcp_url = f"http://{address}:{port}"
                normalized_url, normalized_transport, extras = normalize_mcp_endpoint(
                    raw_mcp_url, "streamable-http"
                )

                # First try to get tools with descriptions via thv
                server_url = normalized_url
                tools = self._get_tools_with_descriptions(server_url)

                if not tools:
                    # Fallback to FastMCP client if thv fails
                    try:
                        async with Client(normalized_url, timeout=3.0) as client:
                            fastmcp_tools = await client.list_tools()
                            tools = [
                                Tool(tool.name, getattr(tool, "description", ""))
                                for tool in fastmcp_tools
                            ]
                    except Exception as e:
                        print(f"❌ Failed to get tools from {server_url}: {e}")
                        continue

                if tools:
                    found_servers = True
                    # Get server name
                    name = None
                    try:
                        async with Client(normalized_url, timeout=3.0) as client:
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
                    print(f"✅ Found MCP server on port {port} with name {name}")
                    print(
                        f"📋 Available tools: {[tool.name for tool in tools]} with descriptions"
                    )

                    mcps.append(
                        MCP(
                            name=name,
                            tools=tools,  # Now storing Tool objects with descriptions
                            address=address,
                            port=port,
                            transport=normalized_transport,  # Normalized transport
                            discovery_url=normalized_url,  # Normalized URL for discovery
                            client_url=normalized_url,  # Normalized URL for client connections
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
            toolhive_hint = (
                " Or start with ToolHive: ./start-toolhive.sh"
                if self.use_toolhive
                else ""
            )
            raise RuntimeError(
                f"No MCP servers found. Please start Toolomics MCP server.{toolhive_hint}"
            )
        self.mcps.extend(mcps)
        return mcps

    async def discover_mcp_servers(self, timeout: float = 2.0) -> list[MCP]:
        # ToolHive-only discovery
        if not self.use_toolhive:
            raise RuntimeError(
                "ToolHive is required. Please install ToolHive: curl -sSL https://get.toolhive.dev | sh"
            )

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
            raise RuntimeError(
                f"Failed to discover MCP servers via ToolHive: {e}"
            ) from e

    def _get_client_variable_name(self, mcp: MCP) -> str:
        """Generate a variable name for the MCP client based on its name."""
        name = mcp.name
        name = name.replace(" ", "_").upper()
        return name + "_TOOLS"

    def get_client_prompt(self, mcp: MCP) -> str:
        """Generate a prompt for the MCP client with tool descriptions."""
        assert isinstance(mcp, MCP), "Expected MCP instance"
        if not mcp.tools:
            raise ValueError("MCP tools list cannot be empty.")

        # Create detailed tool descriptions
        tool_descriptions = []
        for tool in mcp.tools:
            if tool.description:
                tool_descriptions.append(f"  - {tool.name}: {tool.description}")
            else:
                tool_descriptions.append(f"  - {tool.name}")

        tool_list_str = "\n".join(tool_descriptions)
        name = self._get_client_variable_name(mcp)
        return f"""
        Tool {name} is a collection of tools with the following capabilities:
{tool_list_str}
        """

    def get_client_code(self, mcp: MCP) -> str:
        """Generate transport-aware client code for a specific MCP server."""
        if not mcp.tools:
            raise ValueError("MCP tools list cannot be empty.")

        # Handle different transport types according to SmolAgents MCPClient documentation
        name = self._get_client_variable_name(mcp)
        transport = mcp.transport

        if transport == "stdio":
            # For stdio transport - this would need additional configuration
            # For now, raise an error as stdio servers aren't discovered by thv list
            raise ValueError(
                "Stdio transport not supported in current ToolHive-based discovery"
            )

        # For HTTP-based transports (sse, streamable-http)
        # Use client_url if available, otherwise fallback to discovery_url or construct one
        if mcp.client_url:
            api_url = mcp.client_url
        elif mcp.discovery_url:
            api_url = mcp.discovery_url
        else:
            # Fallback for backward compatibility
            if not mcp.address or not mcp.port:
                raise ValueError(
                    "MCP address and port must be set for HTTP transports."
                )
            api_url = f"http://{mcp.address}:{mcp.port}/mcp"

        # Generate transport-specific client code for workflow execution
        # The client connection must remain active throughout the workflow
        return f'''
from smolagents import MCPClient
params = {{"url": "{api_url}", "transport": "{transport}"}}
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

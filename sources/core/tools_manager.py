"""MCP tool discovery and registration for the Mimosa-AI workflow stack.
    This module owns Layer 1 tool discovery: it finds MCP servers through
    ToolHive or by probing network addresses, normalizes their endpoint
    URLs, and captures each server's tool list. The workflow runner
    (Layer 3) imports ToolManager at start-up, discovers servers, and asks
    it for the client prompt and client code that each agent embeds.
    Discovery runs once per workflow; the generated client snippets then
    run inside the sandboxed workflow.

"""

import asyncio
import json
import time
import subprocess
from typing import Any
from urllib.parse import urlparse, urlunparse
from fastmcp import Client

from config import Config
from sources.cli.pretty_print import (
    print_ok, print_warn, print_err, print_info,
    print_phase, print_section,
    print_iteration_header, print_box,
    print_summary, print_agent_answers,
    CYAN, GREEN, YELLOW, RED, DIM, RESET, BOLD,
)

def normalize_mcp_endpoint(
    raw_url: str, transport: str
) -> tuple[str, str, dict[str, Any]]:
    """Normalize an MCP endpoint URL and transport for client connections.
        ToolHive discovery calls it with every server URL before building an
        MCP descriptor, so all clients connect the same way. It strips the URL
        fragment (keeping the container name as a hint), maps transport
        aliases to canonical names, and appends the ``/mcp`` path for
        streamable-http servers while keeping the original path for SSE.

        Args:
            raw_url (str): Server URL from the ToolHive listing, possibly
                carrying a fragment.
            transport (str): Transport name from discovery or
                auto-detection. Accepted: ``"stdio"``, ``"sse"``, ``"http"``,
                ``"streamable-http"``, ``"streamable"``.

        Returns:
            tuple[str, str, dict[str, Any]]: Normalized URL, canonical
            transport (``"sse"`` or ``"streamable-http"``), and an extras
            dict holding ``container_hint`` when the URL carried a fragment.

        Raises:
            ValueError: When the transport is not one of the accepted
            names.

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
    else:
        raise ValueError(
            f"Unsupported transport: {transport}. Supported transports: 'sse', 'streamable-http'"
        )

    url = urlunparse(parsed)
    # Store container hint for logging/debugging
    if container:
        extras["container_hint"] = container
        print_warn(
            f"⚠️ ToolHive container fragment '{container}' detected but fragments are not sent over HTTP."
        )

    return url, t, extras


class Tool:
    """Name and describe one tool exposed by an MCP server.
        ToolManager builds Tool instances while parsing a server's tool list,
        and the client generators embed these fields into prompts and code.
        Invariant: ``name`` is non-empty; ``description`` may be blank.

    """

    def __init__(self, name: str, description: str = "") -> None:
        """Store the tool's name and description.
            ToolManager calls it with fields parsed from a server's JSON tool
            listing.

            Args:
                name (str): Tool identifier as exposed by the MCP server.
                description (str): Human-readable tool text from the listing;
                    defaults to empty.

            Returns:
                None.

        """
        self.name = name
        self.description = description

    def __repr__(self) -> str:
        """Return a one-line summary of the tool.
            Logs and debugging code use it to show a tool compactly.

            Returns:
                str: ``"<name>: <description>"``.

        """
        return f"""{self.name}: {self.description}"""

class MCP:
    """Describe one discovered MCP server and its tools.
        ToolManager builds an MCP for every server found during discovery and
        keeps the instances in its registry for later prompt and code
        generation. Invariant: ``tools`` is always a list (never ``None``).

    """

    def __init__(
        self,
        name: str | None = None,
        tools: list[Tool] | None = None,
        address: str | None = None,
        port: int | None = None,
        toolhive_name: str | None = None,
        transport: str = "streamable-http",
        discovery_url: str | None = None,
        client_url: str | None = None,
    ) -> None:
        """Store the server descriptor fields.
            ToolManager calls it during discovery with values from the ToolHive
            listing or from a network probe.

            Args:
                name (str | None): Display name for the server, derived from the
                    ToolHive name or from the probed port.
                tools (list[Tool] | None): Tools exposed by the server; ``None``
                    becomes an empty list.
                address (str | None): Host name or IP the server listens on.
                port (int | None): TCP port of the server.
                toolhive_name (str | None): ToolHive server name used by
                    lifecycle commands (stop, restart); ``None`` for
                    network-discovered servers.
                transport (str): Transport name: ``"streamable-http"``,
                    ``"sse"``, or ``"stdio"``. Defaults to streamable-http.
                discovery_url (str | None): Normalized URL used during
                    discovery and testing.
                client_url (str | None): Normalized URL used by generated
                    clients at runtime.

            Returns:
                None.

        """
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
        """Expose the tool names without the Tool objects.
            The client generators and tool verification use this to inspect a
            server's tool set by name.

            Returns:
                list[str]: Name of each tool in ``tools``; empty when the server
                exposes none.

        """
        return [tool.name for tool in self.tools]

    def __repr__(self) -> str:
        """Return a one-line summary of the server.
            Logs and debugging code use it to show a server compactly.

            Returns:
                str: Server name, tool count, and ``address:port``.

        """
        return f"""{self.name}: {len(self.tools)} tools available | running on {self.address}:{self.port}"""


class ToolManager:
    """Discover MCP servers and generate client prompts and code for them.
        The workflow runner (Layer 3) creates one manager at start-up, runs
        ``discover_mcp_servers``, and renders each found server into agent
        client snippets. Two discovery backends exist: ToolHive lifecycle
        commands and a direct network scan. Because ToolHive servers showed
        latency and freezing in practice, the ToolHive branch is currently
        disabled inside ``discover_mcp_servers``; the network scan stays live
        and ToolHive support remains ready for future use. Invariant:
        discovered servers accumulate in the ``mcps`` registry.

    """

    def __init__(self, config: "Config | None" = None, mcps: list[MCP] | None = None) -> None:
        """Create the manager and probe for ToolHive availability.
            The workflow runner constructs it once per session, optionally
            passing its Config and a pre-filled server list. It stores both and
            records whether the ``thv`` command works.

            Args:
                config (Config | None): Application configuration object from
                    the caller; a default Config is created when ``None``.
                mcps (list[MCP] | None): Known servers from the caller; ``None``
                    becomes an empty list.

            Returns:
                None.

        """
        self.mcps = mcps if mcps is not None else []
        self.use_toolhive = self._check_toolhive_available()
        self.config = config if config is not None else Config()

    def _check_toolhive_available(self) -> bool:
        """Probe whether the ToolHive CLI is installed and working.
            The constructor calls it once to set the ToolHive-usage flag. It runs
            ``thv version`` with a five-second timeout and checks the exit code.

            Returns:
                bool: ``True`` when ``thv version`` exits zero; ``False`` when
                the command fails, times out, or is missing.

        """
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

    def _attempt_mcp_stop(self, mcp_name: str) -> bool:
        """Stop one ToolHive server by name.
            The full-restart helper calls it when a server exposes no tools and
            must be recovered. It runs ``thv stop`` with a five-second timeout.

            Args:
                mcp_name (str): ToolHive server name from the discovery
                    listing.

            Returns:
                bool: ``True`` when ``thv stop`` exits zero; ``False`` on
                failure, timeout, or missing binary.

        """
        try:
            result = subprocess.run(
                ["thv",
                "stop", mcp_name],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.CalledProcessError,
        ):
            return False

    def _attempt_mcp_restart(self, mcp_name: str) -> bool:
        """Restart one ToolHive server by name.
            The full-restart helper calls it after a stop; ToolHive has no plain
            start command, so restart is the only start path. It runs ``thv
            restart`` with a five-second timeout.

            Args:
                mcp_name (str): ToolHive server name from the discovery
                    listing.

            Returns:
                bool: ``True`` when ``thv restart`` exits zero; ``False`` on
                failure, timeout, or missing binary.

        """
        try:
            result = subprocess.run(
                ["thv",
                "restart", mcp_name], # no start command, use restart
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.CalledProcessError,
        ):
            return False

    def _attempt_full_mcp_restart(self, mcp_name: str) -> bool:
        """Stop and then restart one ToolHive server.
            Discovery calls it when a server answers but exposes no tools,
            because a plain restart does not always recover a hanging server.

            Args:
                mcp_name (str): ToolHive server name from the discovery
                    listing.

            Returns:
                bool: ``True`` when both the stop and the restart succeed;
                ``False`` when either step fails.

        """
        if not self._attempt_mcp_stop(mcp_name):
            return False
        return self._attempt_mcp_restart(mcp_name)

    def _get_tools_with_descriptions(self, server_url: str) -> list[Tool]:
        """Fetch one server's tool list through the ToolHive CLI.
            ToolHive discovery calls it with each server URL to fill MCP
            descriptors. It runs ``thv mcp list tools`` in JSON mode with a
            ten-second timeout, parses the listing, and skips tools whose fields
            cannot be read.

            Args:
                server_url (str): Server URL from the discovery listing.

            Returns:
                list[Tool]: Parsed tools with names and descriptions; empty when
                the command fails, times out, or the JSON cannot be parsed.

        """
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
                print_err(f"❌ Failed to get tools from {server_url}: {result.stderr}")
                return []

            try:
                data = json.loads(result.stdout)
                tools = []
                for tool_data in data.get("tools", []):
                    try:
                        name = tool_data.get("name", "")
                    except Exception as e:
                        print_err(f"❌ Cannot get name from {server_url}: {e}")
                        continue
                    try:
                        description = tool_data.get("description", "")
                    except Exception as e:
                        print_err(f"❌ Cannot get description from {server_url}: {e}")
                        continue
                    if name:
                        tools.append(Tool(name, description))
                return tools
            except json.JSONDecodeError as e:
                print_err(f"❌ Failed to parse JSON response from {server_url}: {e}")
                return []

        except subprocess.TimeoutExpired:
            print_err(f"❌ Timeout getting tools from {server_url}")
            return []
        except Exception as e:
            print_err(f"❌ Error getting tools from {server_url}: {e}")
            return []

    async def discover_toolhive_servers(self) -> list[MCP]:
        """Find every running ToolHive server and its tools.
            ``discover_mcp_servers`` normally skips this method, but the manager
            keeps it for ToolHive deployments. It lists servers with ``thv list``,
            skips entries that are not running or lack a URL, auto-detects the
            transport from the URL path, normalizes the endpoint, and probes each
            server for tools. A server without tools is stopped and restarted
            once, and discovery then reruns from the start.

            Returns:
                list[MCP]: Descriptors for running servers that expose tools;
                empty when the listing fails or no server qualifies.

        """
        try:
            result = subprocess.run(
                ["thv", "list", "--format", "json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                print_err(f"❌ Failed to get ToolHive server list: {result.stderr}")
                return []

            servers_data = json.loads(result.stdout)
            mcps = []

            for server in servers_data:
                name = server.get("name", "")
                status = server.get("status", "")

                if status != "running":
                    print_warn(f"⚠️ ToolHive server {name} is not running (status: {status})")
                    continue

                # Extract connection info
                url = server.get("url", "")
                if not url:
                    print_warn(f"⚠️ No URL found for ToolHive server {name}")
                    continue

                # Parse URL to extract address and port
                # URL format: http://address:port/sse#server-name
                import urllib.parse

                parsed = urllib.parse.urlparse(url)
                address = parsed.hostname or "localhost"
                port = parsed.port

                if not port:
                    print_warn(f"⚠️ Could not extract port from URL {url} for server {name}")
                    continue

                # For ToolHive servers, auto-detect transport from URL path
                raw_url = (url)

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

                tools = self._get_tools_with_descriptions(raw_url)
                if tools:
                    # Use readable name based on toolhive server name - no need for MCP client
                    server_name = name.replace("-", " ").title() + " MCP"

                    print_ok(f"Found ToolHive MCP server {name} ({server_name})")
                    print_info(
                        f"Available tools: {[tool.name for tool in tools]} with descriptions"
                    )
                    if extras.get("container_hint"):
                        print_info(f"🏷️ Container: {extras['container_hint']}")

                    mcps.append(
                        MCP(
                            name=server_name,
                            tools=tools,
                            address=address,
                            port=port,
                            toolhive_name=name,
                            transport=normalized_transport,
                            discovery_url=normalized_url,
                            client_url=normalized_url,
                        )
                    )
                else:
                    print_warn(f"⚠️ Attempting full restart for ToolHive server {name}...")
                    if self._attempt_full_mcp_restart(name):
                        print_ok(f"✅ Successfully restarted ToolHive server {name}")
                        return await self.discover_toolhive_servers()
                    else:
                        print_err(f"❌ Failed to restart ToolHive server {name}")
            return mcps

        except subprocess.TimeoutExpired:
            print_err("❌ ToolHive command timed out")
            return []
        except json.JSONDecodeError as e:
            print_err(f"❌ Failed to parse ToolHive JSON output: {e}")
            return []
        except Exception as e:
            print_err(f"❌ Error discovering ToolHive servers: {e}")
            return []

   ###################################
   #   METHODS FOR NETWORK BASED MCP #
   ###################################

    async def discover_mcp_at_address(
        self,
        address: str,
        port_min: int = 5000,
        port_max: int = 5250,
        timeout: float = 2.0,
    ) -> list[MCP]:
        """Probe one address across a port range for MCP servers.
            Network discovery calls it for each configured discovery address. For
            every port in the range it tries an MCP client connection to
            ``http://<address>:<port>/mcp`` and records the servers that answer
            with tools. The client connection itself uses a fixed three-second
            timeout; the declared ``timeout`` argument is only reported in the
            timeout log line.

            Args:
                address (str): Host name or IP to probe, from the config.
                port_min (int): First port to probe, inclusive; config default
                    5000.
                port_max (int): Last port to probe, inclusive; config default
                    5250.
                timeout (float): Declared per-port probe timeout from the
                    config, echoed in log output.

            Returns:
                list[MCP]: One descriptor per port that served tools, named
                ``mcp_<port>``; empty when no port responds.

        """
        mcps = []

        for port in range(port_min, port_max + 1):
            try:
                async with Client(
                    f"http://{address}:{port}/mcp", timeout=3.0
                ) as client:
                    tools = await client.list_tools()
                    name = f"mcp_{port}"
                    if tools:
                        print_ok(f"Found MCP server on {address} port {port} with name {name}")
                        print_info(f"Available tools: {[tool.name for tool in tools]}")
                        mcps.append(
                            MCP(
                                name=name,
                                tools=[tool for tool in tools],
                                address=address,
                                port=port,
                            )
                        )
            except asyncio.TimeoutError:
                print_err(f"❌ MCP server on port {port} timed out after {timeout}s")
                continue
            except Exception:
                continue
        return mcps

    async def discover_network_mcp_servers(self) -> list[MCP]:
        """Scan all configured discovery addresses for MCP servers.
            ``discover_mcp_servers`` calls it as the live discovery path. It
            walks the discovery addresses from the config, aggregates the
            per-address probe results, and skips addresses that fail with a
            logged error.

            Returns:
                list[MCP]: Descriptors from every address that served tools;
                empty when none respond or the config lists no addresses.

        """
        mcps = []
        for addr in self.config.discovery_addresses:
            try:
                mcps_server = await self.discover_mcp_at_address(addr.ip, addr.port_min, addr.port_max)
                if mcps_server:
                    mcps.extend(mcps_server)
            except Exception as e:
                print_err(f"❌ Failed to discover MCP servers on {addr}: {e}")
                continue
        return mcps

    async def verify_tools(self) -> bool:
        """Check that at least one server exposes the shell tool.
            The workflow runner calls it after discovery to confirm that agents
            can execute commands. It scans the registered servers for a tool
            named ``execute_command`` and waits two seconds before reporting
            failure so the warning stays readable.

            Returns:
                bool: ``True`` when any registered server exposes
                ``execute_command``; ``False`` when none does or when a tool
                object lacks the expected fields.

        """
        bash_found = False
        for mcp in self.mcps:
            tools = '\n'.join(mcp.tool_names)
            try:
                if "execute_command" in tools:
                    bash_found = True
            except AttributeError as e:
                return False
        if not bash_found:
            print_warn("\n⚠️ No execute_command for shell use found among MCP servers.\nPlease deploy shell MCP.")
            time.sleep(2)
            return False
        return True

    async def discover_mcp_servers(self) -> list[MCP]:
        """Discover MCP servers through the live network path.
            The workflow runner calls this entry point at start-up before
            building agent clients. The ToolHive branch is currently disabled in
            code, so only the network scan runs; found servers are appended to
            the registry. When nothing is found it warns and returns an empty
            list.

            Returns:
                list[MCP]: The registry after discovery; empty when no server
                was found.

            Raises:
                RuntimeError: When discovery raises an unexpected error.

        """
        try:
            print_info("🔍 Discovering MCP servers via ToolHive...")
            #mcps_thv = await self.discover_toolhive_servers()
            mcps_thv = []
            print_info("🔍 Discovering MCP servers on network...")
            mcps_net = await self.discover_network_mcp_servers()

            if mcps_net or mcps_thv:
                print_ok(f"Found {len(mcps_thv)} MCP server(s) via ToolHive.")
                self.mcps.extend(mcps_thv)
                print_ok(f"Found {len(mcps_net)} MCP server(s) via network scan.")
                self.mcps.extend(mcps_net)
                return self.mcps
            else:
                print_warn("⚠️ No running MCP servers found via ToolHive.")
                return []
        except Exception as e:
            print_err(f"❌ ToolHive discovery failed: {e}")
            raise RuntimeError(
                f"Failed to discover MCP servers via ToolHive: {e}"
            ) from e

    def _get_client_variable_name(self, mcp: MCP) -> str:
        """Derive the Python variable name for one server's tool collection.
            The prompt and code generators call it so every generated snippet
            binds the tools to a stable name. It upper-cases the server name,
            replaces spaces with underscores, and appends ``_TOOLS``.

            Args:
                mcp (MCP): Server descriptor built during discovery.

            Returns:
                str: Variable name such as ``"MY_SERVER_TOOLS"``.

        """
        name = mcp.name
        name = name.replace(" ", "_").upper()
        return name + "_TOOLS"

    def get_client_prompt(self, mcp: MCP) -> str:
        """Render a text prompt describing one server's tools.
            Agent builders embed this prompt in the model context so the model
            knows which tool collection to call. It lists each tool name with the
            first line of its description and binds the collection to the derived
            variable name.

            Args:
                mcp (MCP): Server descriptor holding at least one tool.

            Returns:
                str: Multi-line prompt text naming the tool collection.

            Raises:
                AssertionError: When ``mcp`` is not an MCP instance.
                ValueError: When the server exposes no tools.
                AttributeError: When a tool lacks a usable name.

        """
        assert isinstance(mcp, MCP), "Expected MCP instance"
        if not mcp.tools:
            raise ValueError("MCP tools list cannot be empty.")

        # Create detailed tool descriptions
        tool_descriptions = []
        for tool in mcp.tools:
            try:
                name = getattr(tool, 'name', None)
                if not name:
                    raise AttributeError(f"Tool name is empty or None: {tool}")
                if hasattr(tool, 'description') and getattr(tool, 'description', None):
                    description = tool.description.split('\n')[0]
                    tool_descriptions.append(f"\n\n  - {name}: {description}")
                else:
                    tool_descriptions.append(f"\n\n  - {name}: No description available.")
            except AttributeError as e:
                raise e

        tool_list_str = "\n".join(tool_descriptions)
        name = self._get_client_variable_name(mcp)
        return f"""
        Tool {name} is a collection of tools with the following capabilities:
{tool_list_str}
        """

    def get_client_code(self, mcp: MCP) -> str:
        """Generate Python client code that binds one server's tools.
            The workflow runner embeds this snippet in generated workflows so the
            code can call the server's tools at runtime. It picks the endpoint
            URL (client URL, then discovery URL, then ``address:port``
            construction) and emits an smolagents MCPClient setup bound to the
            derived variable name.

            Args:
                mcp (MCP): Server descriptor built during discovery.

            Returns:
                str: Python source snippet that creates the client and binds its
                tools.

            Raises:
                ValueError: When the server exposes no tools, when the transport
                is ``"stdio"``, or when no URL can be built.

        """
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

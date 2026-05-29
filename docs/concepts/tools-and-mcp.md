# Tool discovery & MCP

Mimosa-AI does not ship a fixed catalogue of scientific tools. Instead, it
discovers them at run time through the
[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) — a standard
JSON-RPC interface for exposing tools to LLM agents.

Mimosa is **agnostic about where MCP servers come from**. Anything that
exposes a streamable-HTTP or SSE MCP endpoint on a reachable address and
port is discoverable.

[Toolomics](toolomics.md) is the companion platform we ship and document
alongside Mimosa, but it is not a hard requirement — see
[Operating without Toolomics](#operating-without-toolomics) below.

## How discovery works

The [`ToolManager`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/tools_manager.py)
discovers MCP servers through two complementary paths:

1. **Network scan** — for each `AddressMCP` in `Config.discovery_addresses`
   (default `AddressMCP("0.0.0.0", 5000, 5100)`), every port in the range
   is probed with an MCP handshake against `http://<ip>:<port>/mcp`.
2. **ToolHive enumeration** — if the [`thv`](https://github.com/stacklok/toolhive)
   CLI is present on the host, `thv list` is queried and every running
   server (streamable-HTTP or SSE) is registered.

   !!! note "ToolHive path currently disabled by default"
       The ToolHive code path is implemented and tested but commented out
       in `discover_mcp_servers()` because of latency / freezing observed
       on some hosts. Re-enable it in `tools_manager.py` if you want
       Mimosa to use a local ToolHive deployment.

For every discovered server, Mimosa records the URL, transport
(`streamable-http` or `sse`), and the list of tools with their schemas.

## How tools are wired into a workflow

Discovered tools become available in two places:

- **Workflow factory** — the synthesized workflow includes a header block
  emitted by `ToolManager.get_client_code()` that re-binds the MCP client.
  Every agent in the generated workflow sees **every discovered tool**;
  Mimosa does not filter by tool tags (the MCP protocol itself does not
  expose tags, and Mimosa does not invent any).
- **Workflow runner** — the generated workflow is plain Python, so the
  tools are called like ordinary functions inside the sandbox.

Example of the auto-generated client block (transport-aware):

```python
from smolagents import MCPClient
params = {"url": "http://localhost:5042/mcp", "transport": "streamable-http"}
client = MCPClient(params)
tools = client.get_tools()
MCP_5042_TOOLS = tools
```

## Operating without Toolomics

You **can** run Mimosa without Toolomics. The `ToolManager` only cares
that MCP servers answer on the configured discovery addresses. Concretely:

- Point `Config.discovery_addresses` at any IP / port range where you
  expose MCP servers — your own `fastmcp` scripts, third-party MCP
  containers, or anything else speaking streamable-HTTP MCP.
- Or install ToolHive separately and re-enable the ToolHive branch in
  `tools_manager.py`; Mimosa will then pick up every running ToolHive
  server regardless of port range.
- Or mix: run some servers via Toolomics on `5000–5100` and others as
  standalone processes on a different IP / port.

What you give up by skipping Toolomics:

- The shared workspace volume management Toolomics provides for
  multi-instance / multi-user setups.
- The opinionated registration / Dockerization workflow for adding new
  tools.
- The default workspace layout `Config.workspace_dir` expects (you'll
  need to set `workspace_dir` to a directory that all MCP servers can
  read and write).

In other words, Toolomics is **packaging and isolation around MCP**, not
a protocol Mimosa depends on. See [Toolomics](toolomics.md) for what it
buys you.

## Shared workspace

Whatever discovery layer you use, Mimosa expects a **single shared
workspace directory** at `Config.workspace_dir`. All generated artefacts —
intermediate files, scripts, downloads, plots — go through this
directory. The workspace gives Mimosa:

- A single filesystem-level rendezvous between agents and MCP servers.
- A predictable path for the verifier to recompute claims against.
- An archive surface for `runs_capsule/` snapshots.

If you use Toolomics, the workspace is the directory you passed to
`./start.sh`. Otherwise it's any path you choose, as long as the MCP
servers exposing file I/O tools see the same path.

See [Workspace & audit trail](../usage/workspace.md) for how the workspace
flows into the audit trail.

## Adding a new tool

1. Write your tool as a `fastmcp` MCP server (or any other MCP-compatible
   implementation).
2. Expose it on an IP / port that falls inside
   `Config.discovery_addresses`, or register it with ToolHive.
3. Restart Mimosa. The next discovery cycle picks the tool up.

If you use Toolomics, follow its [Adding a new MCP via Toolomics](toolomics.md#adding-a-new-mcp-via-toolomics)
section — it handles port assignment and the optional Docker wrapper
automatically.

## See also

- [Toolomics](toolomics.md) — the companion platform and what it adds on
  top of plain MCP discovery.
- [Architecture](architecture.md) — where the `ToolManager` fits in.
- [Scientific grounding](grounding.md) — what Perspicacité adds on top.
- [Workspace & audit trail](../usage/workspace.md) — the filesystem side.

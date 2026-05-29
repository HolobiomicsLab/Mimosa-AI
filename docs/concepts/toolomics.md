# Toolomics

[Toolomics](https://github.com/HolobiomicsLab/Toolomics) is Mimosa's
**companion platform for MCP server management and workspace-isolated
scientific tool execution**. It is not a Mimosa subsystem and not a
protocol Mimosa depends on — it is a standalone tool that packages
scientific MCP servers in a way that plays well with Mimosa (and any
other MCP-compatible agent).

If you are looking at how MCP discovery itself works on Mimosa's side, see
[Tool discovery & MCP](tools-and-mcp.md). This page is about the
companion platform.

## Why use it

Toolomics gives you four things that bare MCP servers do not:

| | What you get | Why it matters for Mimosa |
| --- | ------------ | ------------------------- |
| **Auto-discovery of MCPs to host** | Scans `mcp_host/<tool>/server.py` and `mcp_host/<tool>/docker-compose.yml` for tools to expose. | Add a new MCP by dropping files in a folder — Mimosa picks it up on the next run. |
| **Port assignment** | Each MCP gets a port in a configurable range, recorded in `config_<instance_id>.json`. | Mimosa's `discovery_addresses` scans that range and finds everything at once. |
| **Shared workspace** | All MCP servers read and write from the same `workspace/` directory. | Files produced by one MCP (e.g. a download) are immediately visible to others (e.g. a PDF parser) and to Mimosa's verifier. |
| **Multi-instance isolation** | Each workspace gets an 8-char instance ID; Docker containers, volumes, and auxiliary ports are suffixed with it. | You can run two Mimosa users / two benchmarks side by side on the same host without port or volume collisions. |

## How Mimosa interacts with Toolomics

Toolomics runs independently. Mimosa never starts, stops, or queries it
directly — they communicate only through MCP:

```
Toolomics                             Mimosa
─────────                             ──────
mcp_host/pdf/        ─► port 5002  ◄─ ToolManager scans
mcp_host/image/      ─► port 5006  ◄─ Config.discovery_addresses
mcp_host/shell/      ─► port 5012  ◄─ (default 5000–5100)
…
workspace/           ◄────────────►   Config.workspace_dir
```

Concretely, Mimosa cares about exactly two configuration values when
running on top of Toolomics:

- `Config.discovery_addresses` — must include the port range Toolomics
  is using. Default `5000–5100` matches Toolomics' default `./start.sh`.
- `Config.workspace_dir` — must point at the same `workspace/` directory
  Toolomics is using.

Both can be edited in `config.py` or via a JSON config file. See
[Configuration reference](../reference/configuration.md).

## Typical deployment

The simplest Toolomics deployment is:

```bash
git clone https://github.com/HolobiomicsLab/Toolomics.git
cd Toolomics
./start.sh                          # uses workspace/ and ports 5000–5200
```

For a different workspace or port range:

```bash
./start.sh 5000 5099 workspace_mimosa
```

On first run Toolomics enumerates `mcp_host/*/`, allocates a port to each
enabled service, and writes `config_<instance_id>.json` listing them.
Newly discovered services start with `"enabled": false` — flip them to
`true` and re-run `./start.sh` to bring them up.

Then start Mimosa as usual. The `ToolManager` will scan the port range
and find every enabled Toolomics service automatically.

## Multi-instance setups

A scenario Toolomics makes possible: two researchers, one host.

```bash
# Terminal A (Martin)
./start.sh 5000 5099 workspace_martin

# Terminal B (John), at the same time
./start.sh 5100 5199 workspace_john
```

Each instance gets a different workspace, a different port range, and
instance-suffixed Docker resources. Two Mimosa sessions can then point at
`5000–5099` and `5100–5199` respectively and never see each other's
files or tools.

This is also the mechanism behind safe concurrent benchmark runs — pair
each `csv_runs_limit` shard with its own Toolomics instance.

## Adding a new MCP via Toolomics

1. Create `mcp_host/<tool_name>/server.py` reading the port from
   `MCP_PORT` / `FASTMCP_PORT` env vars or `sys.argv[1]`.
2. *(optional)* Add a `Dockerfile` + `docker-compose.yml` in the same
   directory if the tool needs isolated dependencies. Toolomics will
   build and run the container instead of executing `server.py`
   directly.
3. Run `./start.sh`. Toolomics discovers the new service, assigns it a
   port, and writes it into `config_<instance_id>.json` with
   `"enabled": false`.
4. Flip the entry to `"enabled": true` and re-run `./start.sh`.

The Toolomics README has the full Dockerization checklist, including the
project-root build context required to access `shared.py`.

## When you don't want Toolomics

You can run Mimosa against any MCP servers you bring yourself. See
[Operating without Toolomics](tools-and-mcp.md#operating-without-toolomics)
in the tool-discovery page for the trade-offs (you keep the same Mimosa
behaviour; you take ownership of workspace layout, port assignment, and
isolation).

## See also

- [Tool discovery & MCP](tools-and-mcp.md) — Mimosa's discovery side.
- [Workspace & audit trail](../usage/workspace.md) — how Mimosa uses the
  shared workspace.
- [Toolomics on GitHub](https://github.com/HolobiomicsLab/Toolomics) —
  source, README, and the canonical reference for the platform itself.

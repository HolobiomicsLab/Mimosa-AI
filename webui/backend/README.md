# Mimosa Observatory — backend

Read-only FastAPI observability API over the artifacts a Mimosa-AI run writes
to disk (`sources/workflows`, `sources/memory`, the toolomics workspace),
plus a small setup/launch bridge into the real Mimosa install. See
[`../README.md`](../README.md) for what the Observatory is and the full API
surface; this file covers only running the backend itself.

The service is intentionally lightweight and standalone: it does not import
`torch`, `smolagents`, or Mimosa itself for the observability endpoints — it
only reads files off disk.

## Prerequisites

- Python >= 3.11
- [`uv`](https://docs.astral.sh/uv/) as the dependency manager

## Install

```bash
cd webui/backend
uv sync
```

## Run

```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8848
```

The app object is `app.main:app`. It serves on port 8848 by default; pair it
with the frontend dev server (see [`../frontend/README.md`](../frontend/README.md)),
which proxies `/api` to this port.

## Tests

```bash
uv run pytest
```

## Environment

All variables are optional — sensible defaults point at a local Mimosa
checkout. See [`../README.md`](../README.md#deployment) for the full table and
what each path is used for.

## Setup & launch require a real Mimosa install

The read-only observability endpoints (runs, tree, series, artifacts, memory,
workspace, live feed) work with only the dependencies installed above — no
Mimosa checkout is required to be functional, just present on disk to read.

The setup and launch endpoints are different: objective refinement
(`/api/assist/refine`), goal/task classification (`/api/assist/classify`),
and run launching (`/api/launches`) shell out to the real Mimosa install via
`MIMOSA_PYTHON`, a Python interpreter that can `import` Mimosa. That means
those features additionally require:

- A working Mimosa install with its own `.venv` (see the root
  [`README.md`](../../README.md)).
- Provider API keys available to that install, either in the project `.env`
  or `~/.config/mimosa/.env`.

If `MIMOSA_PYTHON` doesn't point at a usable interpreter, the setup page
(`/api/setup`, `/api/setup/config`) still works for viewing and editing
config — but refine, classify, and launch are disabled, and the UI reports
this.

## Production notes

There is no Dockerfile, systemd unit, or CI in this repo for the backend.
Running `uv run uvicorn ...` as shown above is the only supported way to
start it today. See the root [`webui/README.md`](../README.md#deployment)
for the current state of frontend/backend production packaging — in short,
none is shipped, and this is a single-operator, localhost tool with no
authentication that should not be exposed on a shared or public network.

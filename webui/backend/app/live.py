"""Filesystem-derived live event feed over the workflows directory.

Phase 1 is read-only: with no event bus inside Mimosa yet, we watch the
workflow_dir and translate file changes into semantic events (a new run dir, an
iteration's ``run_metrics.json`` landing, the evolution tree PNG refreshing, the
QD archive growing). Clients receive these over a WebSocket and refetch what
they care about. When the in-process event bus lands, its events layer on top of
this same broadcast channel.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from watchfiles import awatch

from .settings import get_settings

_RUN_ID = re.compile(r"(\d{8}_\d{6}_[0-9a-f]{8})")


def _classify(path: str) -> dict[str, Any] | None:
    name = Path(path).name
    m = _RUN_ID.search(path)
    run_id = m.group(1) if m else None
    if name == "run_metrics.json":
        etype = "iteration_complete"
    elif name == "state_result.json":
        etype = "execution_complete"
    elif name == "evolution_tree.png":
        etype = "tree_updated"
    elif name == "reward_progress.png":
        etype = "run_finished"
    elif name == "qd_archive.jsonl":
        etype = "archive_appended"
    elif name.startswith("workflow_genotype_"):
        etype = "workflow_crafted"
    else:
        return None
    return {"type": etype, "run_id": run_id, "filename": name}


class LiveHub:
    """Fans filesystem-change events out to all connected WebSocket clients."""

    def __init__(self) -> None:
        self._clients: set[Any] = set()
        self._task: asyncio.Task | None = None

    async def register(self, ws: Any) -> None:
        self._clients.add(ws)

    def unregister(self, ws: Any) -> None:
        self._clients.discard(ws)

    async def _broadcast(self, event: dict[str, Any]) -> None:
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def _run(self) -> None:
        root = get_settings().workflow_dir
        if not root.exists():
            return
        async for changes in awatch(root, recursive=True):
            seen: set[tuple[str, str | None]] = set()
            for _change, path in changes:
                event = _classify(path)
                if not event:
                    continue
                key = (event["type"], event["run_id"])
                if key in seen:
                    continue
                seen.add(key)
                await self._broadcast(event)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None


hub = LiveHub()

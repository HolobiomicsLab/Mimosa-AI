"""Filesystem-derived live event feed over Mimosa's artifact roots.

Phase 1 is read-only: with no event bus inside Mimosa yet, we watch the
directories Mimosa writes and translate file changes into semantic events —
a new run dir, an iteration's ``run_metrics.json`` landing, a memory step
appended mid-run, the textual gradient being written, an ASTRA capsule
(decision layer) or asb_eval evaluation capsule appearing. Clients receive
these over a WebSocket and refetch what they care about. When the in-process
event bus lands, its events layer on top of this same broadcast channel.

Watched roots (those that exist at startup): ``workflow_dir`` (run
artifacts), ``memory_dir`` (per-agent step traces — this is what makes the
replay live while a run executes), ``capsule_dir`` (ASTRA capsules the
transparency exporter writes), ``eval_dir`` (asb_eval evaluation capsules).
A root created after startup is picked up on the next backend restart.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from watchfiles import awatch

from .settings import get_settings

_RUN_ID = re.compile(r"((?:single_agent_)?\d{8}_\d{6}_[0-9a-f]{8})")


def _classify(path: str, memory_root: str) -> dict[str, Any] | None:
    p = Path(path)
    name = p.name
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
    elif name == "textual_gradient.txt":
        etype = "gradient_updated"
    elif name == "evaluation.txt":
        etype = "evaluation_updated"
    elif name == "astra.yaml":
        # The transparency exporter's capsule — the ASTRA decision layer.
        etype = "astra_updated"
    elif name == "eval_astra.yaml":
        # asb_eval's independent evaluation capsule; its tree carries no run
        # id in the path, so run_id is usually None here.
        etype = "evaluation_capsule_updated"
    elif name.startswith("task_") and p.suffix == ".json" and path.startswith(memory_root):
        # A smolagents step trace growing under memory_dir/<run_id>/.
        etype = "step_appended"
    elif p.suffix == ".json" and path.startswith(memory_root):
        # Any other per-run JSON under memory_dir is a single LLM call logged
        # (workflow_creator, verifier_*, judge_*). Scoped to the memory root
        # on purpose: run dirs under workflow_dir also churn cache JSONs
        # (rubric_cache_*, claim_cache_*) that are not calls.
        etype = "llm_call_logged"
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
        s = get_settings()
        roots = [
            r for r in (s.workflow_dir, s.memory_dir, s.capsule_dir, s.eval_dir)
            if r.exists()
        ]
        if not roots:
            return
        memory_root = str(s.memory_dir)
        async for changes in awatch(*roots, recursive=True):
            seen: set[tuple[str, str | None]] = set()
            for _change, path in changes:
                event = _classify(path, memory_root)
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

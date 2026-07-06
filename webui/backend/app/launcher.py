"""Spawn and track Mimosa runs as detached subprocesses in the Mimosa venv.

A launch runs ``mimosa_bridge.py run`` with the run's output dirs forced to the
ones the Observatory watches, so its artifacts appear live through the existing
run endpoints and the filesystem WebSocket. The subprocess is put in its own
session so it can be signalled as a group on cancel. This is deliberately
single-process/in-memory bookkeeping — fine for one operator driving runs.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .bridge import BRIDGE_SCRIPT, bridge_available
from .settings import get_settings

_LOG_DIR = Path(tempfile.gettempdir()) / "mimosa_observatory_launches"

# Mimosa's console output is ANSI-coloured; strip escapes for the web log view.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class Launch:
    def __init__(self, launch_id: str, proc: subprocess.Popen, log_path: Path, meta: dict[str, Any]):
        self.id = launch_id
        self.proc = proc
        self.log_path = log_path
        self.meta = meta
        self.started_at = meta.get("started_at")

    def status(self) -> dict[str, Any]:
        rc = self.proc.poll()
        return {
            "id": self.id,
            "pid": self.proc.pid,
            "running": rc is None,
            "returncode": rc,
            **self.meta,
        }


class Launcher:
    def __init__(self) -> None:
        self._launches: dict[str, Launch] = {}
        self._counter = 0

    def launch(
        self, objective: str, mode: str, learn: bool, judge: bool, started_at: str
    ) -> dict[str, Any]:
        ok, err = bridge_available()
        if not ok:
            return {"ok": False, "error": err}
        settings = get_settings()
        params = {
            "objective": objective,
            "mode": mode,
            "learn": learn,
            "judge": judge,
            "workflow_dir": str(settings.workflow_dir),
            "memory_dir": str(settings.memory_dir),
            "workspace_dir": str(settings.workspace_dir),
        }
        self._counter += 1
        launch_id = f"L{self._counter}"
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _LOG_DIR / f"{launch_id}.log"
        log_fh = log_path.open("w")
        proc = subprocess.Popen(
            [str(settings.mimosa_python), str(BRIDGE_SCRIPT), "run", json.dumps(params)],
            cwd=str(settings.root),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        meta = {
            "objective": objective, "mode": mode, "learn": learn, "judge": judge,
            "started_at": started_at, "log": str(log_path),
        }
        self._launches[launch_id] = Launch(launch_id, proc, log_path, meta)
        return {"ok": True, **self._launches[launch_id].status()}

    def status(self, launch_id: str) -> dict[str, Any] | None:
        launch = self._launches.get(launch_id)
        if not launch:
            return None
        st = launch.status()
        st["log_tail"] = self._tail(launch.log_path)
        return st

    def list(self) -> list[dict[str, Any]]:
        return [ln.status() for ln in self._launches.values()]

    def cancel(self, launch_id: str) -> dict[str, Any] | None:
        launch = self._launches.get(launch_id)
        if not launch:
            return None
        if launch.proc.poll() is None:
            try:
                os.killpg(os.getpgid(launch.proc.pid), signal.SIGTERM)
                time.sleep(0.2)
                if launch.proc.poll() is None:
                    os.killpg(os.getpgid(launch.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        return launch.status()

    @staticmethod
    def _tail(path: Path, limit: int = 4000) -> str:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return _ANSI_RE.sub("", text)[-limit:]


launcher = Launcher()

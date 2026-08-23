"""Call the Mimosa-venv bridge for short LLM tasks (refine, classify).

These are synchronous, bounded subprocess calls: the backend writes a JSON
payload to the bridge's stdin and reads the single ``@@RESULT@@`` line back.
Long-running launches go through ``launcher.py`` instead.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .settings import get_settings

RESULT_MARKER = "@@RESULT@@"
BRIDGE_SCRIPT = Path(__file__).resolve().parent.parent / "mimosa_bridge.py"


def bridge_available() -> tuple[bool, str | None]:
    py = get_settings().mimosa_python
    if not py.exists():
        return False, f"Mimosa venv python not found at {py} (set MIMOSA_PYTHON)"
    if not BRIDGE_SCRIPT.exists():
        return False, f"bridge script missing at {BRIDGE_SCRIPT}"
    return True, None


def bridge_env() -> dict[str, str]:
    """Process env for bridge subprocesses.

    The bridge script lives in webui/backend, so Python puts *that* dir on
    sys.path, not the Mimosa root — and ``sources`` isn't installed into the
    venv. Put the root on PYTHONPATH so ``import sources`` works (as the
    bridge docstring promises: "with MIMOSA_ROOT on the path").
    """
    env = dict(os.environ)
    root = str(get_settings().root)
    env["PYTHONPATH"] = (
        root + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else root
    )
    return env


def _call(cmd: str, payload: dict[str, Any], timeout: int = 90) -> dict[str, Any]:
    ok, err = bridge_available()
    if not ok:
        return {"ok": False, "error": err, "available": False}
    settings = get_settings()
    try:
        proc = subprocess.run(
            [str(settings.mimosa_python), str(BRIDGE_SCRIPT), cmd],
            input=json.dumps(payload),
            cwd=str(settings.root),
            env=bridge_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"bridge '{cmd}' timed out", "available": True}
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            try:
                data = json.loads(line[len(RESULT_MARKER):])
                data["available"] = True
                return data
            except json.JSONDecodeError:
                break
    tail = (proc.stderr or proc.stdout or "").strip()[-500:]
    return {"ok": False, "error": f"bridge '{cmd}' produced no result. {tail}", "available": True}


def refine(objective: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return _call("refine", {"objective": objective, "history": history or []})


def classify(objective: str) -> dict[str, Any]:
    return _call("classify", {"objective": objective})


def objective_history() -> dict[str, Any]:
    return _call("objective_history", {})

"""Read/write the Mimosa config JSON and report setup readiness.

The setup page edits a curated subset of Mimosa's config (workspace, the model
slots, learning knobs). Writes merge into the existing file so every other key
(provider lists, runner requirements, prompt paths) is preserved verbatim.
API-key values are never returned — only which keys are present.
"""

from __future__ import annotations

import json
import os
import re
import socket
import tempfile
from pathlib import Path
from typing import Any

from .settings import get_settings

# The subset the setup UI edits. Everything else in the file is preserved.
EDITABLE_KEYS = {
    "workspace_dir": str,
    "planner_llm_model": str,
    "workflow_llm_model": str,
    "smolagent_model_id": str,
    "capsule_namer_model": str,
    "judge_model": str,
    "reasoning_effort": str,
    "max_tokens": int,
    "learned_score_threshold": float,
    "max_learning_evolve_iterations": int,
    "export_astra": bool,
}

# Mimosa's Config.from_json resets discovery_addresses to [] when the key is
# absent, so a freshly created partial config must carry the default range.
DEFAULT_DISCOVERY = {"ip": "0.0.0.0", "port_min": 5000, "port_max": 5200}

# Suggested model ids per role, grouped by provider, for the picker.
MODEL_PRESETS = {
    "orchestration": [
        "anthropic/claude-opus-4-8", "openrouter/z-ai/glm-5.2",
        "mistral/mistral-medium-3-5", "openai/gpt-5",
    ],
    "agent": [
        "deepseek/deepseek-v4-flash", "openrouter/deepseek/deepseek-v4-flash",
        "anthropic/claude-sonnet-5", "mistral/mistral-medium-3-5",
    ],
    "judge": [
        "openrouter/z-ai/glm-5.2", "anthropic/claude-opus-4-8",
        "deepseek/deepseek-v4-flash",
    ],
}


def config_path() -> Path:
    return get_settings().config_path


def _atomic_write(path: Path, text: str, mode: int = 0o644) -> None:
    """Write *text* to *path* via tmp-file + rename so a crash can't truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.chmod(tmp, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_config() -> dict[str, Any]:
    path = config_path()
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def editable_view() -> dict[str, Any]:
    """Return only the editable subset, plus a couple of read-only context keys."""
    cfg = read_config()
    view = {k: cfg.get(k) for k in EDITABLE_KEYS}
    view["_config_path"] = str(config_path())
    view["_config_exists"] = config_path().is_file()
    workspace = cfg.get("workspace_dir")
    view["_workspace_exists"] = bool(workspace) and Path(str(workspace)).is_dir()
    view["_discovery_addresses"] = cfg.get("discovery_addresses", [])
    return view


def _coerce(key: str, value: Any) -> Any:
    """Coerce *value* to the declared type for *key*; raise ValueError if unfit."""
    expected = EDITABLE_KEYS[key]
    if expected is bool:
        if not isinstance(value, bool):
            raise ValueError(f"invalid value for {key!r}: expected bool")
        return value
    if isinstance(value, bool):  # bool is an int subclass — reject for numbers/strings
        raise ValueError(f"invalid value for {key!r}: expected {expected.__name__}")
    if expected is str:
        if not isinstance(value, str):
            raise ValueError(f"invalid value for {key!r}: expected str")
        return value
    if expected is int and isinstance(value, float) and not value.is_integer():
        raise ValueError(f"invalid value for {key!r}: expected int (got a fraction)")
    try:
        return expected(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid value for {key!r}: expected {expected.__name__}")


def update_config(patch: dict[str, Any]) -> dict[str, Any]:
    """Validate + merge *patch* into the config file, preserving all other keys."""
    cfg = read_config()
    # Mimosa's from_json resets discovery_addresses to [] when the key is
    # absent, so any config we write must carry an explicit range.
    cfg.setdefault("discovery_addresses", [DEFAULT_DISCOVERY])
    applied: dict[str, Any] = {}
    for key, value in patch.items():
        if key not in EDITABLE_KEYS or value is None:
            continue  # ignore unknown/non-editable keys
        cfg[key] = applied[key] = _coerce(key, value)
    _atomic_write(config_path(), json.dumps(cfg, indent=2) + "\n")
    return applied


def _keys_in_env_files() -> set[str]:
    found: set[str] = set()
    pat = re.compile(r"^\s*(?:export\s+)?([A-Z0-9_]+)\s*=")
    for env_file in get_settings().env_files:
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                m = pat.match(line)
                if m:
                    found.add(m.group(1))
        except OSError:
            continue
    return found


def key_status() -> dict[str, Any]:
    """Which known API keys are present in env or the dotenv files (no values)."""
    import os

    file_keys = _keys_in_env_files()
    settings = get_settings()
    statuses = []
    for name in settings.known_key_names:
        in_env = bool(os.environ.get(name))
        in_file = name in file_keys
        statuses.append(
            {"name": name, "present": in_env or in_file,
             "source": "env" if in_env else ("dotenv" if in_file else None)}
        )
    return {"keys": statuses, "any": any(s["present"] for s in statuses)}


def _env_file_for(name: str) -> Path:
    """Pick the dotenv file to write *name* into.

    The bridge loads the project ``.env`` before the XDG one and dotenv keeps
    the first value it sees, so a key already defined in the project file must
    be updated there; everything else goes to ``~/.config/mimosa/.env`` (the
    file the CLI onboarding writes).
    """
    project_env, xdg_env = get_settings().env_files
    pat = re.compile(rf"^\s*(?:export\s+)?{re.escape(name)}\s*=")
    try:
        if any(pat.match(line) for line in project_env.read_text(encoding="utf-8").splitlines()):
            return project_env
    except OSError:
        pass
    return xdg_env


def save_key(name: str, value: str) -> dict[str, Any]:
    """Upsert one API key into a dotenv file; the value is never echoed back."""
    if name not in get_settings().known_key_names:
        raise ValueError(f"unknown key name: {name!r}")
    value = value.strip()
    if not value or any(ch in value for ch in "\r\n\"'"):
        raise ValueError("invalid key value")
    target = _env_file_for(name)
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    pat = re.compile(rf"^\s*(?:export\s+)?{re.escape(name)}\s*=")
    entry = f"{name}={value}"
    replaced = False
    for i, line in enumerate(lines):
        if pat.match(line):
            lines[i] = entry
            replaced = True
            break
    if not replaced:
        lines.append(entry)
    _atomic_write(target, "\n".join(lines) + "\n", mode=0o600)
    return {"name": name, "saved_to": str(target)}


def _port_open(host: str, port: int, timeout: float = 0.15) -> bool:
    probe = "127.0.0.1" if host in ("0.0.0.0", "") else host
    try:
        with socket.create_connection((probe, port), timeout=timeout):
            return True
    except OSError:
        return False


def mcp_health(max_ports: int = 220) -> dict[str, Any]:
    """TCP-probe the configured MCP discovery range; report responsive ports."""
    cfg = read_config()
    addresses = cfg.get("discovery_addresses") or [
        {"ip": "0.0.0.0", "port_min": 5000, "port_max": 5200}
    ]
    open_ports: list[dict[str, Any]] = []
    scanned = 0
    for addr in addresses:
        host = addr.get("ip", "0.0.0.0")
        lo, hi = int(addr.get("port_min", 5000)), int(addr.get("port_max", 5200))
        for port in range(lo, hi + 1):
            if scanned >= max_ports:
                break
            scanned += 1
            if _port_open(host, port):
                open_ports.append({"host": host, "port": port})
    return {
        "reachable": bool(open_ports),
        "open_ports": open_ports,
        "scanned": scanned,
        "note": None if open_ports
        else "No MCP servers responding — start toolomics before launching runs.",
    }

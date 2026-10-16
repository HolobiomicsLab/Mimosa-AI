"""Read per-agent memory traces for the post-run replay view.

Two file shapes live in ``sources/memory/<uuid>/``:
- ``task_<name>.json`` — a list of smolagents ActionStep dicts (the workflow
  agents; these are the replayable step timelines). Can be multi-MB.
- every other ``*.json`` — a single LLM call (workflow_creator, verifier_*,
  judge_*) with prompt/response/usage.cost.

Timeline ordering is by ``timing.start_time`` (epoch), NOT filename: agents
interleave, and alphabetical filename order misrepresents true execution order.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .settings import get_settings

_MAX = 4000  # hard char cap on any single text field returned to the client
_CACHE: dict[tuple[str, int, int], Any] = {}


def memory_path(run_id: str) -> Path:
    return get_settings().memory_dir / run_id


def _load_cached(path: Path) -> Any | None:
    """Parse JSON with an (path, mtime, size) cache to avoid re-reading 5 MB files."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), int(st.st_mtime), st.st_size)
    if key in _CACHE:
        return _CACHE[key]
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if len(_CACHE) > 64:
        _CACHE.clear()
    _CACHE[key] = data
    return data


def _trim(text: Any, limit: int = _MAX) -> str:
    if text is None:
        return ""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + " …[truncated]"


def _extract_code(step: dict[str, Any]) -> str:
    code = step.get("code_action")
    if isinstance(code, str) and code.strip():
        return code.strip()
    for call in step.get("tool_calls") or []:
        if isinstance(call, dict):
            args = (call.get("function") or {}).get("arguments")
            if isinstance(args, str) and args.strip():
                return args.strip()
            if isinstance(args, dict) and args.get("code"):
                return str(args["code"]).strip()
    return ""


def _output_text(step: dict[str, Any]) -> str:
    msg = step.get("model_output_message") or {}
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        joined = "\n".join(
            c.get("text", "") for c in content if isinstance(c, dict)
        ).strip()
        if joined:
            return joined
    return str(step.get("model_output", "") or "")


def _observations(step: dict[str, Any]) -> str:
    obs = step.get("observations")
    if obs in (None, "None"):
        return ""
    return obs if isinstance(obs, str) else json.dumps(obs)


def _is_task_file(path: Path) -> bool:
    return path.name.startswith("task_")


def _agent_name(path: Path) -> str:
    return path.stem.removeprefix("task_")


def list_memory(run_id: str) -> dict[str, Any] | None:
    """Summarise the agents and LLM calls recorded for a run."""
    mem = memory_path(run_id)
    if not mem.is_dir():
        return None
    agents: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for path in sorted(mem.glob("*.json")):
        data = _load_cached(path)
        if _is_task_file(path) and isinstance(data, list):
            steps = [s for s in data if isinstance(s, dict)]
            start = next(
                (s["timing"]["start_time"] for s in steps if s.get("timing")), None
            )
            tokens = sum((s.get("token_usage") or {}).get("total_tokens", 0) for s in steps)
            agents.append(
                {
                    "name": _agent_name(path),
                    "kind": "agent",
                    "steps": len(steps),
                    "errored": any(s.get("error") for s in steps),
                    "start_time": start,
                    "total_tokens": tokens,
                }
            )
        elif isinstance(data, dict) and "choices" in data:
            usage = data.get("usage") or {}
            calls.append(
                {
                    "name": path.stem,
                    "kind": "llm_call",
                    "model": data.get("model"),
                    "created": data.get("created"),
                    "cost_usd": usage.get("cost"),
                    "total_tokens": usage.get("total_tokens"),
                }
            )
    agents.sort(key=lambda a: (a["start_time"] is None, a["start_time"]))
    calls.sort(key=lambda c: (c["created"] is None, c["created"]))
    return {"run_id": run_id, "agents": agents, "calls": calls}


def timeline(run_id: str) -> dict[str, Any] | None:
    """Ordered, compact agent steps across all ``task_*`` files.

    Sorted by ``timing.start_time`` so cross-agent order is faithful. Heavy
    fields (full message history) are excluded here; fetch them per-step.
    """
    mem = memory_path(run_id)
    if not mem.is_dir():
        return None
    rows: list[dict[str, Any]] = []
    for path in sorted(mem.glob("task_*.json")):
        data = _load_cached(path)
        if not isinstance(data, list):
            continue
        agent = _agent_name(path)
        for idx, step in enumerate(data):
            if not isinstance(step, dict):
                continue
            timing = step.get("timing") or {}
            error = step.get("error")
            rows.append(
                {
                    "agent": agent,
                    "index": idx,
                    "step_number": step.get("step_number"),
                    "start_time": timing.get("start_time"),
                    "duration": timing.get("duration"),
                    "is_final_answer": bool(step.get("is_final_answer")),
                    "error_type": (error or {}).get("type") if error else None,
                    "code": _trim(_extract_code(step), 2000),
                    "output_text": _trim(_output_text(step), 1500),
                    "observations": _trim(_observations(step), 2000),
                    "tokens": step.get("token_usage") or {},
                }
            )
    rows.sort(key=lambda r: (r["start_time"] is None, r["start_time"] or 0))
    for order, row in enumerate(rows):
        row["order"] = order
    return {"run_id": run_id, "steps": rows}


def step_detail(run_id: str, agent: str, index: int) -> dict[str, Any] | None:
    """Full content of one agent step, including the model message history."""
    path = memory_path(run_id) / f"task_{agent}.json"
    data = _load_cached(path)
    if not isinstance(data, list) or index < 0 or index >= len(data):
        return None
    step = data[index]
    if not isinstance(step, dict):
        return None
    messages = []
    for msg in step.get("model_input_messages") or []:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            content = "\n".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        messages.append({"role": msg.get("role"), "content": _trim(content, 8000)})
    action_output = step.get("action_output")
    return {
        "agent": agent,
        "index": index,
        "step_number": step.get("step_number"),
        "timing": step.get("timing"),
        "is_final_answer": bool(step.get("is_final_answer")),
        "error": step.get("error"),
        "code": _extract_code(step),
        "output_text": _trim(_output_text(step), 12000),
        "observations": _trim(_observations(step), 12000),
        "action_output": _trim(json.dumps(action_output, default=str), 8000)
        if action_output is not None
        else None,
        "tokens": step.get("token_usage"),
        "input_messages": messages,
    }


def call_detail(run_id: str, name: str) -> dict[str, Any] | None:
    """Full prompt/response/usage of one single-shot LLM call."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        return None
    path = memory_path(run_id) / f"{name}.json"
    data = _load_cached(path)
    if not isinstance(data, dict):
        return None
    usage = data.get("usage") or {}
    messages = [
        {"role": m.get("role"), "content": _trim(m.get("content"), 12000)}
        for m in data.get("message") or []
        if isinstance(m, dict)
    ]
    return {
        "name": name,
        "model": data.get("model"),
        "provider": data.get("provider"),
        "temperature": data.get("temperature"),
        "reasoning_effort": data.get("reasoning_effort"),
        "created": data.get("created"),
        "cost_usd": usage.get("cost"),
        "tokens": {
            "prompt": usage.get("prompt_tokens"),
            "completion": usage.get("completion_tokens"),
            "total": usage.get("total_tokens"),
        },
        "messages": messages,
        "response": _trim(data.get("response"), 16000),
    }

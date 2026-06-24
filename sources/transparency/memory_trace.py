"""Canonical extraction of executed code and reasoning from a run's memory.

This is the single source of truth for reading smolagents ``ActionStep`` dumps
(``sources/memory/<uuid>/task_*.json``). Both the ASTRA exporter and the
interactive memory chat (``sources/cli/memory_chat_cli.py``) build on these
primitives so the two never drift on what counts as "the code the agent ran".

Field precedence for code, mirroring smolagents' own ActionStep layout:
1. ``code_action`` — the parsed Python a CodeAgent executed (most reliable)
2. ``tool_calls[*].function.arguments`` — the python_interpreter tool payload
3. a fenced/``<code>`` block inside the model output (last-resort text scrape)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


RECIPE_FILENAME = "recipe.py"


def trim(text: str, limit: int) -> str:
    """Hard-truncate *text* to *limit* characters with an ellipsis marker."""
    if not text:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + " …[truncated]"


def extract_code(step: dict[str, Any]) -> str:
    """Return the executed code for one step (best-effort, never raises)."""
    code = step.get("code_action")
    if isinstance(code, str) and code.strip():
        return code.strip()
    for call in step.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        args = (call.get("function") or {}).get("arguments")
        if isinstance(args, str) and args.strip():
            return args.strip()
        if isinstance(args, dict) and args.get("code"):
            return str(args["code"]).strip()
    return _scrape_code_block(extract_output_text(step))


def extract_output_text(step: dict[str, Any]) -> str:
    """Return the model's reasoning/output text for one step."""
    msg = step.get("model_output_message") or {}
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        chunks = [c.get("text", "") for c in content if isinstance(c, dict)]
        joined = "\n".join(t for t in chunks if t)
        if joined.strip():
            return joined
    return str(step.get("model_output", "") or "")


def extract_observations(step: dict[str, Any]) -> str:
    """Return the observation text returned to the agent after the step."""
    obs = step.get("observations") or step.get("observation") or ""
    return obs if isinstance(obs, str) else json.dumps(obs)


def load_raw_steps(memory_dir: Path) -> list[dict[str, Any]]:
    """Load every ``task_*.json`` trace under *memory_dir*, in run order.

    Files are read name-sorted (multi-agent traces are deterministic); steps
    within a file keep their saved order. Unreadable files are skipped.
    """
    steps: list[dict[str, Any]] = []
    for path in sorted(memory_dir.glob("task_*.json")):
        try:
            with path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        agent = path.stem.removeprefix("task_")
        for step in payload if isinstance(payload, list) else []:
            if isinstance(step, dict):
                step.setdefault("_agent_name", agent)
                steps.append(step)
    return steps


def reconstruct_recipe(memory_dir: Path) -> str:
    """Rebuild the run's executed code as one annotated script string.

    Returns an empty string when no step carried code. The result is a faithful
    transcript, not a guaranteed-runnable script: per-step in-memory state is
    not serialised, so blocks may depend on variables defined earlier.
    """
    blocks: list[str] = []
    for position, step in enumerate(load_raw_steps(memory_dir), start=1):
        code = extract_code(step)
        if not code:
            continue
        agent = step.get("_agent_name", "agent")
        number = step.get("step_number", position)
        blocks.append(f"# ── step {number} · {agent} ──\n{code}")
    if not blocks:
        return ""
    header = (
        "# ASTRA recipe — reconstructed from the best run's agent trace.\n"
        "# Each block is one executed step, in order. In-memory state shared\n"
        "# between steps is not serialised, so this is a transcript of what ran,\n"
        "# not a guaranteed standalone script.\n"
    )
    return header + "\n\n".join(blocks) + "\n"


def _scrape_code_block(text: str) -> str:
    """Pull a fenced or ``<code>`` block out of free-form model output."""
    match = re.search(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


if __name__ == "__main__":
    import tempfile

    code_step = {
        "step_number": 1,
        "code_action": "from scipy import stats\nstats.ttest_ind(a, b, equal_var=False)",
        "model_output": "Use Welch's t-test.",
        "observations": "pvalue=0.02",
    }
    tool_step = {
        "step_number": 2,
        "tool_calls": [{"function": {"name": "python_interpreter",
                                     "arguments": "model.fit(X, y)"}}],
        "observations": "fitted",
    }
    assert extract_code(code_step).startswith("from scipy"), extract_code(code_step)
    assert extract_code(tool_step) == "model.fit(X, y)", extract_code(tool_step)
    assert extract_output_text(code_step) == "Use Welch's t-test."
    assert trim("abcdef", 3) == "abc …[truncated]"

    with tempfile.TemporaryDirectory() as tmp:
        mem = Path(tmp)
        (mem / "task_single_agent.json").write_text(json.dumps([code_step, tool_step]))
        recipe = reconstruct_recipe(mem)
        assert "from scipy" in recipe and "model.fit" in recipe, recipe
        assert "step 1 · single_agent" in recipe, recipe
        assert reconstruct_recipe(Path(tmp) / "empty") == ""
    print("[OK] memory_trace smoke check passed")

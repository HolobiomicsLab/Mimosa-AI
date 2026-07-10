"""Read the final context length of each agent from a run's saved memory.

The SmolAgent factory writes one JSON file per agent into
``<memory_dir>/<workflow_uuid>/``, holding the ordered steps of that agent.
Each step records its ``token_usage``; the input tokens of the last step are
the context the agent was carrying when it finished.
"""

import json
from pathlib import Path

# Both factories save agent memory as ``task_{agent_name}.json``; the
# single-agent run saves itself as ``task_single_agent.json``.
_AGENT_FILE_PREFIX = "task_"


def _final_input_tokens(steps: list) -> int:
    """Input tokens of the last step that recorded any, ``0`` when none did.

    Args:
        steps: Ordered agent memory steps, as loaded from the memory file.

    Returns:
        The final step's ``token_usage.input_tokens``, or ``0``.
    """
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        usage = step.get("token_usage") or {}
        tokens = usage.get("input_tokens")
        if isinstance(tokens, int) and tokens > 0:
            return tokens
    return 0


def read_agent_context_lengths(memory_dir: str, workflow_uuid: str) -> list[int]:
    """Final context length of every agent that ran under ``workflow_uuid``.

    Missing directories, unreadable files and agents with no recorded token
    usage are skipped rather than raised, because this feeds a ranking term
    and must never abort an evolution iteration.

    Args:
        memory_dir: Root directory holding per-run agent memory folders.
        workflow_uuid: UUID naming this run's memory folder.

    Returns:
        One positive length per agent, ordered by memory filename.
    """
    if not memory_dir or not workflow_uuid:
        return []

    run_memory = Path(memory_dir) / workflow_uuid
    if not run_memory.is_dir():
        return []

    lengths = []
    for path in sorted(run_memory.glob("*.json")):
        if not path.name.startswith(_AGENT_FILE_PREFIX):
            continue
        try:
            steps = json.loads(path.read_text())
        except Exception:
            # A ranking term must never abort an iteration; a memory file that
            # is missing, truncated, or too large to parse simply does not vote.
            continue
        if not isinstance(steps, list):
            continue
        tokens = _final_input_tokens(steps)
        if tokens:
            lengths.append(tokens)
    return lengths


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        run = Path(root) / "uuid-1"
        run.mkdir()
        (run / "task_a.json").write_text(json.dumps([{"token_usage": {"input_tokens": 10}}]))
        (run / "task_b.json").write_text(json.dumps([{"token_usage": {"input_tokens": 90}}]))
        assert read_agent_context_lengths(root, "uuid-1") == [10, 90]
        assert read_agent_context_lengths(root, "missing") == []
    print("agent_context smoke check passed")

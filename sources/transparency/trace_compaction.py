"""Compact a smolagents memory trace down to the bytes worth shipping to an LLM.

The raw ``task_<agent>.json`` files include ``model_input_messages`` on every
step, which re-serializes the entire prior conversation. That field carries
no new signal — the per-step novelty lives in ``model_output_message`` and
``observations``. Stripping it typically drops the trace to under 20% of its
original size before any further filtering.

A heuristic prefilter then skips steps whose code is obviously mechanical
(file I/O, plotting, env probing) so they never reach the LLM extractor.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

from sources.transparency.memory_trace import (
    extract_code,
    extract_observations,
    extract_output_text,
    load_raw_steps,
)


_OBSERVATION_CHAR_LIMIT = 1200
_MECHANICAL_CODE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.MULTILINE) for p in (
        r"^\s*import\s",
        r"^\s*from\s+\S+\s+import\s",
        r"\bos\.(listdir|path\.|makedirs|getcwd)\b",
        r"\bpd\.read_(csv|excel|parquet|json|table|hdf)\b",
        r"\bopen\s*\(",
        r"\bplt\.(show|savefig|figure|subplot|plot|bar|scatter|hist|imshow)\b",
        r"\bprint\s*\(",
        r"\bsubprocess\.(run|Popen|check_output)\b",
        r"^\s*!\s*pip\s+install",
    )
)


def load_trace(memory_dir: Path) -> list[dict[str, Any]]:
    """Load and concatenate every ``task_*.json`` agent trace in ``memory_dir``.

    Thin wrapper over :func:`sources.transparency.memory_trace.load_raw_steps`,
    kept so existing callers/tests import the loader from here.
    """
    return load_raw_steps(memory_dir)


def compact_step(step: dict[str, Any], index: int) -> dict[str, Any]:
    """Strip the bulky fields and keep only what an extractor LLM needs."""
    return {
        "index": index,
        "reasoning": extract_output_text(step),
        "code": extract_code(step),
        "observation": _truncate(extract_observations(step), _OBSERVATION_CHAR_LIMIT),
    }


def is_methodological_candidate(compact: dict[str, Any]) -> bool:
    """Heuristic prefilter — return False for steps the LLM would reject anyway."""
    code = (compact.get("code") or "").strip()
    if not code:
        return bool((compact.get("reasoning") or "").strip())
    non_mechanical_lines = _count_non_mechanical_lines(code)
    return non_mechanical_lines >= 1


def compact_trace(raw_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return prefiltered, compact step dicts for downstream extraction."""
    compact = [compact_step(s, i) for i, s in enumerate(raw_steps)]
    return [c for c in compact if is_methodological_candidate(c)]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n…[truncated]…\n" + text[-limit // 2 :]


def _count_non_mechanical_lines(code: str) -> int:
    lines = [line for line in code.splitlines() if line.strip()]
    mechanical = sum(
        1 for line in lines if any(p.search(line) for p in _MECHANICAL_CODE_PATTERNS)
    )
    return len(lines) - mechanical


if __name__ == "__main__":
    sample = [
        {
            "model_output_message": {"content": "We pick Welch's t-test because variances differ."},
            "code_action": "from scipy import stats\nresult = stats.ttest_ind(a, b, equal_var=False)",
            "observations": "Ttest_indResult(statistic=2.31, pvalue=0.022)",
            "model_input_messages": [{"role": "system", "content": "x" * 50_000}],
        },
        {
            "model_output_message": {"content": "List files."},
            "code_action": "import os\nos.listdir('.')",
            "observations": "['a.csv']",
        },
    ]
    kept = compact_trace(sample)
    assert len(kept) == 1, f"Expected 1 step, got {len(kept)}"
    assert "Welch" in kept[0]["reasoning"], kept[0]
    assert "ttest_ind" in kept[0]["code"], kept[0]
    print(f"[OK] trace_compaction smoke check passed (kept {len(kept)} / {len(sample)} steps)")

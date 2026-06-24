"""Surface methodological decisions from a compact agent trace via the LLM.

Strategy: one LLM call per surviving step (after :mod:`trace_compaction`).
Each call returns either ``null`` or a single ASTRA decision JSON object.
Decisions are aggregated by ``id`` — the first occurrence wins so later
re-justifications of the same choice don't shadow the originating step.

Per-step calls (vs. one whole-trace dump) keep the prompt-window cost flat,
let the project's LLMProvider cache hit on re-runs, and give each decision
a concrete provenance index (the step it surfaced from).
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

# Lazy-imported inside _extract_one to keep this module import-cheap and to
# avoid pulling in litellm during pure-Python unit tests / smoke checks.


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "decision_extraction.md"


@dataclass(frozen=True)
class Decision:
    """One ASTRA decision surfaced from a single trace step."""

    id: str
    label: str
    rationale: str
    option_id: str
    option_label: str
    option_description: str
    source_step: int


def extract_decisions(
    steps: list[dict[str, Any]],
    goal: str,
    memory_path: Path,
    llm_config: Any | None = None,
) -> list[Decision]:
    """Run the per-step extractor over ``steps`` and dedupe by decision id."""
    template = _PROMPT_PATH.read_text()
    seen: dict[str, Decision] = {}
    for step in steps:
        decision = _extract_one(step, goal, template, memory_path, llm_config)
        if decision is not None and decision.id not in seen:
            seen[decision.id] = decision
    return list(seen.values())


def _extract_one(
    step: dict[str, Any],
    goal: str,
    template: str,
    memory_path: Path,
    llm_config: Any | None,
) -> Decision | None:
    from sources.core.llm_provider import LLMConfig, LLMProvider
    prompt = template.format(
        goal=goal,
        step_index=step["index"],
        reasoning=step.get("reasoning") or "(none)",
        code=step.get("code") or "(none)",
        observation=step.get("observation") or "(none)",
    )
    provider = LLMProvider(
        agent_name=f"astra_decision_step_{step['index']}",
        memory_path=str(memory_path),
        system_msg=None,
        config=llm_config or LLMConfig(),
    )
    try:
        raw = provider(prompt, use_cache=True)
    except Exception:
        return None
    return _parse_response(raw, step["index"])


def _parse_response(raw: str, source_step: int) -> Decision | None:
    body = _strip_fences(raw).strip()
    if not body or body.lower() == "null":
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    return _validate(payload, source_step)


def _validate(payload: Any, source_step: int) -> Decision | None:
    if not isinstance(payload, dict):
        return None
    required = ("id", "label", "rationale", "option_id", "option_label", "option_description")
    if not all(isinstance(payload.get(k), str) and payload[k].strip() for k in required):
        return None
    if not _ID_PATTERN.match(payload["id"]) or not _ID_PATTERN.match(payload["option_id"]):
        return None
    return Decision(
        id=payload["id"],
        label=payload["label"],
        rationale=payload["rationale"],
        option_id=payload["option_id"],
        option_label=payload["option_label"],
        option_description=payload["option_description"],
        source_step=source_step,
    )


def _strip_fences(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text


if __name__ == "__main__":
    good = '{"id": "fit_method", "label": "Fitting method", "rationale": "Outliers bias OLS.", "option_id": "ols", "option_label": "Ordinary least squares", "option_description": "Minimises squared residuals."}'
    assert _parse_response(good, 3).id == "fit_method"
    assert _parse_response("null", 0) is None
    assert _parse_response("not json at all", 0) is None
    bad_id = '{"id": "Fit-Method", "label": "x", "rationale": "x", "option_id": "ols", "option_label": "x", "option_description": "x"}'
    assert _parse_response(bad_id, 0) is None
    fenced = "```json\n" + good + "\n```"
    assert _parse_response(fenced, 1).option_id == "ols"
    print("[OK] decision_extractor smoke check passed")

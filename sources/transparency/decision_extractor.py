"""Surface methodological decisions from a compact agent trace via the LLM.

Strategy: one LLM call per surviving step (after :mod:`trace_compaction`).
Each call returns either ``null`` or a single ASTRA decision JSON object
carrying the chosen option plus any alternatives the agent weighed. Steps are
extracted concurrently but reduced in trace order, so the dedup below is
deterministic regardless of which call finishes first.

Decisions are aggregated by ``id``: the first occurrence wins for the core
fields (label, rationale, chosen option, provenance), and options surfaced by
later re-justifications of the same choice are merged in rather than dropped.

Per-step calls (vs. one whole-trace dump) keep the prompt-window cost flat,
let the project's LLMProvider cache hit on re-runs (each step gets its own
cache file), and give each decision a concrete provenance index.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

# Lazy-imported inside _extract_one to keep this module import-cheap and to
# avoid pulling in litellm during pure-Python unit tests / smoke checks.


_LOGGER = logging.getLogger(__name__)
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "decision_extraction.md"
_MAX_WORKERS = 6

# Sentinel: the LLM call for this step raised. Distinct from ``None`` (a clean
# ``null`` / unparseable response) so a broken judge config — every call
# failing — is visible in the logs instead of masquerading as "no decisions".
_EXTRACT_FAILED = object()


@dataclass(frozen=True)
class Option:
    """One option of an ASTRA decision — chosen or a considered alternative."""

    id: str
    label: str
    description: str


@dataclass(frozen=True)
class Decision:
    """One ASTRA decision surfaced from a single trace step."""

    id: str
    label: str
    rationale: str
    chosen_option_id: str
    options: tuple[Option, ...]
    source_step: int


def extract_decisions(
    steps: list[dict[str, Any]],
    goal: str,
    memory_path: Path,
    llm_config: Any | None = None,
    max_workers: int = _MAX_WORKERS,
) -> list[Decision]:
    """Run the per-step extractor over ``steps`` and dedupe by decision id.

    Steps are extracted concurrently (bounded by ``max_workers``) but reduced
    in original trace order so the first-occurrence-wins merge is deterministic.
    A warning is logged when calls fail, so a broken judge model surfaces
    instead of silently yielding zero decisions.
    """
    if not steps:
        return []
    template = _PROMPT_PATH.read_text()
    results: list[Any] = [None] * len(steps)
    workers = max(1, min(max_workers, len(steps)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_pos = {
            pool.submit(_extract_one, step, goal, template, memory_path, llm_config): pos
            for pos, step in enumerate(steps)
        }
        for future in as_completed(future_to_pos):
            pos = future_to_pos[future]
            try:
                results[pos] = future.result()
            except Exception as exc:  # defensive — _extract_one should not raise
                _LOGGER.warning("ASTRA extraction task crashed at step %s: %s", pos, exc)
                results[pos] = _EXTRACT_FAILED

    _warn_on_failures(sum(1 for r in results if r is _EXTRACT_FAILED), len(steps))

    merged: dict[str, Decision] = {}
    for result in results:
        if not isinstance(result, Decision):
            continue
        existing = merged.get(result.id)
        merged[result.id] = _merge_options(existing, result) if existing else result
    return list(merged.values())


def _warn_on_failures(errors: int, total: int) -> None:
    """Log a diagnostic when extraction calls failed (aggregate, once)."""
    if not errors:
        return
    if errors == total:
        _LOGGER.warning(
            "ASTRA: all %d decision-extraction calls failed — the judge model "
            "is likely misconfigured (see AstraExporter._build_llm_config).",
            errors,
        )
    else:
        _LOGGER.warning(
            "ASTRA: %d/%d decision-extraction calls failed; those steps were skipped.",
            errors,
            total,
        )


def _merge_options(existing: Decision, later: Decision) -> Decision:
    """Fold a later step's alternative options into the first occurrence.

    First-occurrence-wins for the core fields (label, rationale, chosen option,
    provenance); options are unioned by id so alternatives raised when the same
    choice is re-justified later aren't lost.
    """
    known = {o.id for o in existing.options}
    extra = tuple(o for o in later.options if o.id not in known)
    if not extra:
        return existing
    return replace(existing, options=existing.options + extra)


def _extract_one(
    step: dict[str, Any],
    goal: str,
    template: str,
    memory_path: Path,
    llm_config: Any | None,
) -> Any:
    """Extract one step's decision. Returns a Decision, None, or _EXTRACT_FAILED."""
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
    except Exception as exc:
        _LOGGER.warning(
            "ASTRA decision extraction failed at step %s: %s", step["index"], exc
        )
        return _EXTRACT_FAILED
    return _parse_response(raw, step["index"])


def _parse_response(raw: str, source_step: int) -> Decision | None:
    body = _strip_fences(raw).strip()
    if not body or body.lower() == "null":
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        _LOGGER.debug("ASTRA step %s: response was not valid JSON.", source_step)
        return None
    decision = _validate(payload, source_step)
    if decision is None:
        _LOGGER.debug("ASTRA step %s: JSON did not match the decision schema.", source_step)
    return decision


def _validate(payload: Any, source_step: int) -> Decision | None:
    if not isinstance(payload, dict):
        return None
    base = ("id", "label", "rationale")
    if not all(isinstance(payload.get(k), str) and payload[k].strip() for k in base):
        return None
    if not _ID_PATTERN.match(payload["id"]):
        return None
    options = _parse_options(payload)
    if not options:
        return None
    valid_ids = {o.id for o in options}
    chosen = payload.get("chosen_option_id")
    if not (isinstance(chosen, str) and chosen in valid_ids):
        # Fall back to the first listed option (the prompt lists the chosen one
        # first) so a missing/typo'd chosen_option_id doesn't drop the decision.
        chosen = options[0].id
    return Decision(
        id=payload["id"],
        label=payload["label"],
        rationale=payload["rationale"],
        chosen_option_id=chosen,
        options=options,
        source_step=source_step,
    )


def _parse_options(payload: dict[str, Any]) -> tuple[Option, ...]:
    """Parse the ``options`` array; fall back to the legacy single-option shape."""
    raw = payload.get("options")
    if isinstance(raw, list):
        parsed: list[Option] = []
        seen: set[str] = set()
        for item in raw:
            opt = _parse_option(item)
            if opt is not None and opt.id not in seen:
                parsed.append(opt)
                seen.add(opt.id)
        return tuple(parsed)
    legacy = _parse_option(
        {
            "id": payload.get("option_id"),
            "label": payload.get("option_label"),
            "description": payload.get("option_description"),
        }
    )
    return (legacy,) if legacy is not None else ()


def _parse_option(item: Any) -> Option | None:
    if not isinstance(item, dict):
        return None
    oid, label, desc = item.get("id"), item.get("label"), item.get("description")
    if not all(isinstance(v, str) and v.strip() for v in (oid, label, desc)):
        return None
    if not _ID_PATTERN.match(oid):
        return None
    return Option(id=oid, label=label, description=desc)


def _strip_fences(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text


if __name__ == "__main__":
    good = (
        '{"id": "fit_method", "label": "Fitting method", "rationale": "Outliers bias OLS.",'
        ' "chosen_option_id": "robust", "options": ['
        '{"id": "robust", "label": "Robust regression", "description": "Down-weights outliers."},'
        '{"id": "ols", "label": "Ordinary least squares", "description": "Minimises squared residuals."}'
        "]}"
    )
    parsed = _parse_response(good, 3)
    assert parsed.id == "fit_method", parsed
    assert parsed.chosen_option_id == "robust", parsed
    assert {o.id for o in parsed.options} == {"robust", "ols"}, parsed

    # chosen_option_id not among options -> falls back to the first listed option.
    bad_chosen = (
        '{"id": "d", "label": "x", "rationale": "x", "chosen_option_id": "missing",'
        ' "options": [{"id": "a", "label": "A", "description": "d"}]}'
    )
    assert _parse_response(bad_chosen, 0).chosen_option_id == "a"

    # Legacy single-option shape still parses.
    legacy = (
        '{"id": "fit_method", "label": "Fit", "rationale": "r", "option_id": "ols",'
        ' "option_label": "OLS", "option_description": "Minimises squared residuals."}'
    )
    legacy_parsed = _parse_response(legacy, 1)
    assert legacy_parsed.chosen_option_id == "ols", legacy_parsed
    assert legacy_parsed.options[0].label == "OLS", legacy_parsed

    assert _parse_response("null", 0) is None
    assert _parse_response("not json at all", 0) is None
    bad_id = (
        '{"id": "Fit-Method", "label": "x", "rationale": "x",'
        ' "options": [{"id": "ols", "label": "x", "description": "x"}]}'
    )
    assert _parse_response(bad_id, 0) is None
    fenced = "```json\n" + good + "\n```"
    assert _parse_response(fenced, 1).chosen_option_id == "robust"

    # Option merge keeps first-occurrence core fields, unions alternatives.
    first = _parse_response(good, 3)
    later = Decision(
        id="fit_method", label="ignored", rationale="ignored",
        chosen_option_id="ols",
        options=(Option(id="theil_sen", label="Theil-Sen", description="Median slope."),),
        source_step=9,
    )
    merged = _merge_options(first, later)
    assert merged.label == "Fitting method", merged
    assert {o.id for o in merged.options} == {"robust", "ols", "theil_sen"}, merged
    print("[OK] decision_extractor smoke check passed")

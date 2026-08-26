"""Surface methodological decisions from a compact agent trace via the LLM.

Strategy: one LLM call per surviving step (after :mod:`trace_compaction`).
Each call is *expected* to return either ``null`` or a single ASTRA decision
JSON object carrying the chosen option plus any alternatives the agent
weighed; anything else (prose, truncated JSON, schema violations) is counted
as malformed in the returned :class:`ExtractionResult` rather than silently
treated as "no decision". Steps are extracted concurrently but reduced in
trace order, so the dedup below is deterministic regardless of which call
finishes first.

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
# \Z, not $: $ matches before a trailing newline, letting ids like "a\n"
# leak verbatim into the exported YAML.
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*\Z")
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "decision_extraction.md"
_MAX_WORKERS = 6

# Sentinels, both distinct from ``None`` (the model's legitimate "no decision
# here"): _EXTRACT_FAILED marks a step whose LLM call raised; _MALFORMED marks
# a response that was neither a valid decision JSON nor the literal ``null``
# (prose, truncated JSON, schema violations). Keeping them apart from ``None``
# means a degraded extraction is counted and surfaced instead of masquerading
# as an absence of decisions.
_EXTRACT_FAILED = object()
_MALFORMED = object()


@dataclass(frozen=True)
class Option:
    """One option of an ASTRA decision — chosen or a considered alternative."""

    id: str
    label: str
    description: str


@dataclass(frozen=True)
class Decision:
    """One ASTRA decision surfaced from one or more trace steps.

    ``model`` is the model id that produced the source step (trace
    provenance, not LLM output); "" when the trace predates the field.
    ``source_step`` is the first contributing step (kept as the scalar
    accessor for existing readers); ``source_steps`` lists EVERY
    contributing step in trace order, deduped — the merge path appends the
    steps of later re-justifications of the same decision id.
    """

    id: str
    label: str
    rationale: str
    chosen_option_id: str
    options: tuple[Option, ...]
    source_step: int
    model: str = ""
    source_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Default ``source_steps`` to the scalar so both stay consistent."""
        if not self.source_steps:
            object.__setattr__(self, "source_steps", (self.source_step,))


@dataclass(frozen=True)
class ExtractionResult:
    """Extracted decisions plus the extraction-health counters.

    ``crashed`` counts steps whose LLM call raised; ``malformed`` counts steps
    whose response was neither a decision JSON nor the literal ``null``.
    Either way that step's decision, if it had one, is missing from
    ``decisions`` — the counters make the gap visible to the capsule reader.
    """

    decisions: tuple[Decision, ...]
    steps_total: int
    crashed: int
    malformed: int


def extract_decisions(
    steps: list[dict[str, Any]],
    goal: str,
    memory_path: Path,
    llm_config: Any | None = None,
    max_workers: int = _MAX_WORKERS,
) -> ExtractionResult:
    """Run the per-step extractor over ``steps`` and dedupe by decision id.

    Steps are extracted concurrently (bounded by ``max_workers``) but reduced
    in original trace order so the first-occurrence-wins merge is deterministic.
    Crashed calls and malformed responses are counted in the returned
    :class:`ExtractionResult` and logged, so a broken or sloppy judge model
    surfaces instead of silently yielding zero decisions.
    """
    if not steps:
        return ExtractionResult(decisions=(), steps_total=0, crashed=0, malformed=0)
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

    crashed = sum(1 for r in results if r is _EXTRACT_FAILED)
    malformed = sum(1 for r in results if r is _MALFORMED)
    _warn_on_failures(crashed, malformed, len(steps))

    merged: dict[str, Decision] = {}
    for result in results:
        if not isinstance(result, Decision):
            continue
        existing = merged.get(result.id)
        merged[result.id] = _merge_decisions(existing, result) if existing else result
    return ExtractionResult(
        decisions=tuple(merged.values()),
        steps_total=len(steps),
        crashed=crashed,
        malformed=malformed,
    )


def _warn_on_failures(crashed: int, malformed: int, total: int) -> None:
    """Log a diagnostic when extraction results were lost (aggregate, once)."""
    if not crashed and not malformed:
        return
    if crashed + malformed == total:
        _LOGGER.warning(
            "ASTRA: all %d decision-extraction calls were lost (%d crashed, "
            "%d malformed) — the judge model is likely misconfigured "
            "(see AstraExporter._build_llm_config).",
            total,
            crashed,
            malformed,
        )
        return
    _LOGGER.warning(
        "ASTRA: of %d extraction calls, %d crashed and %d returned malformed "
        "output; any decision in those steps is missing from the export.",
        total,
        crashed,
        malformed,
    )


def _merge_decisions(existing: Decision, later: Decision) -> Decision:
    """Fold a later step's options AND provenance into the first occurrence.

    First-occurrence-wins for the core fields (label, rationale, chosen
    option, scalar ``source_step``); options are unioned by id so
    alternatives raised when the same choice is re-justified later aren't
    lost, and ``source_steps`` collects every contributing step (trace
    order, deduped) so no re-justifying step is dropped from provenance.
    """
    known_options = {o.id for o in existing.options}
    extra_options = tuple(o for o in later.options if o.id not in known_options)
    known_steps = set(existing.source_steps)
    extra_steps = tuple(s for s in later.source_steps if s not in known_steps)
    if not extra_options and not extra_steps:
        return existing
    return replace(
        existing,
        options=existing.options + extra_options,
        source_steps=existing.source_steps + extra_steps,
    )


def _extract_one(
    step: dict[str, Any],
    goal: str,
    template: str,
    memory_path: Path,
    llm_config: Any | None,
) -> Any:
    """Extract one step's decision.

    Returns a Decision, None (model said ``null``), _MALFORMED (unusable
    response), or _EXTRACT_FAILED (the LLM call raised).
    """
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
    decision = _parse_response(raw, step["index"])
    model = step.get("model")
    if isinstance(decision, Decision) and isinstance(model, str) and model.strip():
        decision = replace(decision, model=model.strip())
    return decision


def _parse_response(raw: Any, source_step: int) -> Any:
    """Parse one extraction response.

    Returns a :class:`Decision`, ``None`` for the literal ``null`` (the model
    judged the step non-methodological), or :data:`_MALFORMED` for everything
    else — empty or non-text output, prose, truncated JSON, schema violations.
    """
    if not isinstance(raw, str):
        # litellm yields None content for empty/reasoning-only completions —
        # an unusable response, not a crashed call.
        _LOGGER.debug(
            "ASTRA step %s: non-text extraction response (%s).",
            source_step,
            type(raw).__name__,
        )
        return _MALFORMED
    body = _strip_fences(raw).strip()
    if body.lower() == "null":
        return None
    if not body:
        _LOGGER.debug("ASTRA step %s: empty extraction response.", source_step)
        return _MALFORMED
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        _LOGGER.debug("ASTRA step %s: response was not valid JSON.", source_step)
        return _MALFORMED
    decision = _validate(payload, source_step)
    if decision is None:
        _LOGGER.debug("ASTRA step %s: JSON did not match the decision schema.", source_step)
        return _MALFORMED
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
        # No resolvable chosen_option_id: never guess. Guessing could record
        # a rejected alternative as the realised decision — a confidently
        # wrong provenance record. The caller counts this as malformed.
        if not _sole_option_as_listed(payload, options):
            return None
        chosen = options[0].id
    return Decision(
        id=payload["id"],
        label=payload["label"],
        rationale=payload["rationale"],
        chosen_option_id=chosen,
        options=options,
        source_step=source_step,
    )


def _sole_option_as_listed(
    payload: dict[str, Any], options: tuple[Option, ...]
) -> bool:
    """True when the response as given listed exactly one option and it parsed.

    Resolving a missing/typo'd ``chosen_option_id`` is only safe when the
    model never named an alternative. One *surviving* option is not enough:
    ``_parse_options`` drops invalid entries and dedupes ids, so a
    multi-option response whose stated chosen option was dropped would
    otherwise have the surviving rejected alternative promoted to "chosen".
    The legacy single-option shape (no ``options`` list) qualifies by
    construction.
    """
    if len(options) != 1:
        return False
    raw = payload.get("options")
    return not isinstance(raw, list) or len(raw) == 1


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

    # Missing/typo'd chosen_option_id with a SINGLE option is unambiguous.
    bad_chosen = (
        '{"id": "d", "label": "x", "rationale": "x", "chosen_option_id": "missing",'
        ' "options": [{"id": "a", "label": "A", "description": "d"}]}'
    )
    assert _parse_response(bad_chosen, 0).chosen_option_id == "a"

    # With SEVERAL options an unresolvable chosen_option_id is never guessed.
    ambiguous_chosen = (
        '{"id": "d", "label": "x", "rationale": "x", "chosen_option_id": "typo",'
        ' "options": [{"id": "a", "label": "A", "description": "d"},'
        ' {"id": "b", "label": "B", "description": "d"}]}'
    )
    assert _parse_response(ambiguous_chosen, 0) is _MALFORMED

    # The chosen option was LISTED but dropped by option validation (missing
    # description): the surviving rejected alternative must not be promoted.
    dropped_chosen = (
        '{"id": "d", "label": "x", "rationale": "x", "chosen_option_id": "a",'
        ' "options": [{"id": "a", "label": "A"},'
        ' {"id": "b", "label": "B", "description": "d"}]}'
    )
    assert _parse_response(dropped_chosen, 0) is _MALFORMED

    # Legacy single-option shape still parses.
    legacy = (
        '{"id": "fit_method", "label": "Fit", "rationale": "r", "option_id": "ols",'
        ' "option_label": "OLS", "option_description": "Minimises squared residuals."}'
    )
    legacy_parsed = _parse_response(legacy, 1)
    assert legacy_parsed.chosen_option_id == "ols", legacy_parsed
    assert legacy_parsed.options[0].label == "OLS", legacy_parsed

    # A legitimate `null` and unusable output are DISTINCT outcomes.
    assert _parse_response("null", 0) is None
    assert _parse_response("not json at all", 0) is _MALFORMED
    assert _parse_response("", 0) is _MALFORMED
    assert _parse_response(None, 0) is _MALFORMED
    bad_id = (
        '{"id": "Fit-Method", "label": "x", "rationale": "x",'
        ' "options": [{"id": "ols", "label": "x", "description": "x"}]}'
    )
    assert _parse_response(bad_id, 0) is _MALFORMED
    fenced = "```json\n" + good + "\n```"
    assert _parse_response(fenced, 1).chosen_option_id == "robust"

    # Decision merge keeps first-occurrence core fields, unions alternatives,
    # and collects every contributing step in source_steps.
    first = _parse_response(good, 3)
    assert first.source_steps == (3,), first
    later = Decision(
        id="fit_method", label="ignored", rationale="ignored",
        chosen_option_id="ols",
        options=(Option(id="theil_sen", label="Theil-Sen", description="Median slope."),),
        source_step=9,
    )
    merged = _merge_decisions(first, later)
    assert merged.label == "Fitting method", merged
    assert {o.id for o in merged.options} == {"robust", "ols", "theil_sen"}, merged
    assert merged.source_step == 3, merged
    assert merged.source_steps == (3, 9), merged
    print("[OK] decision_extractor smoke check passed")

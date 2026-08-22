"""Tolerant JSON parsing for LLM responses.

Models asked to emit JSON around copied source spans routinely emit JSON that
is *almost* valid, and a strict ``json.loads`` throws the whole answer away.
Three defects account for nearly all of it:

* **Raw control characters** — a multi-line code listing or a pasted paragraph
  dropped into a string value with its newlines intact. Observed against
  ``stealth/ox-alpha`` while running ASB capsules: the planner failed all three
  of its attempts with ``Invalid control character at …`` and the task died.
* **Bare inner quotes** — ``{"code": "df = pd.read_csv("data.csv")"}``. Also
  reproduced against ox-alpha.
* **Trailing prose** after an otherwise complete object, which strict parsing
  rejects wholesale ("Extra data: line 119").

The repair is only attempted *after* a strict parse fails, and valid JSON is
returned byte-for-byte unchanged: in valid JSON every interior quote is already
escaped, every terminator is followed by one of ``_JSON_AFTER_STRING``, and no
string holds a raw control character. When nothing parses, the **original**
``JSONDecodeError`` is raised so callers still see the true defect.

Ported from AgenticScienceBuilder's ``llm_pipeline.loads_llm_json``, which
solved the same problem against the same model family.
"""

import json
from typing import Any

# Characters that may legally follow a JSON string once it has closed. A quote
# followed by anything else is a quote the model forgot to escape.
_JSON_AFTER_STRING = frozenset(",}]:")

# JSON escapes for the control characters a model most often pastes in raw.
_CONTROL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}

_NO_VALUE = object()


def strip_json_fence(text: str | None) -> str:
    """Drop a surrounding Markdown code fence, with or without a language tag.

    ``None`` maps to ``""`` — a reasoning model that spends its whole token
    budget before answering returns a null message content, not a string.
    """
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.split("\n")
    end = len(lines) - 1
    while end > 0 and not lines[end].strip().startswith("```"):
        end -= 1
    return "\n".join(lines[1:end]).strip()


def _closes_string(text: str, quote_index: int) -> bool:
    """True when the quote at ``quote_index`` terminates its JSON string."""
    rest = text[quote_index + 1:].lstrip()
    return not rest or rest[0] in _JSON_AFTER_STRING


def repair_json_strings(text: str) -> str:
    """Escape characters that are illegal raw inside a JSON string."""
    out: list[str] = []
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            if not in_string:
                in_string = True
            elif _closes_string(text, i):
                in_string = False
            else:
                out.append("\\")  # an inner quote the model left bare
        elif in_string and ch < " ":
            out.append(_CONTROL_ESCAPES.get(ch, f"\\u{ord(ch):04x}"))
            continue
        out.append(ch)
    return "".join(out)


def strip_trailing_commas(text: str) -> str:
    """Drop a comma that is followed only by ``}`` or ``]``.

    JSON forbids the trailing comma every model has seen a million times in
    JavaScript and Python. It surfaces as ``Expecting property name enclosed in
    double quotes``, which reads like a quoting problem and is not — observed
    six times in a single failed planner run against ``stealth/ox-alpha``.

    Commas inside string values are left alone, so the walk tracks string state
    the same way :func:`repair_json_strings` does.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\" and in_string:
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string if (not in_string or _closes_string(text, i)) else in_string
            out.append(ch)
            continue
        if ch == "," and not in_string:
            rest = text[i + 1:].lstrip()
            if rest[:1] in ("}", "]"):
                continue  # drop it
        out.append(ch)
    return "".join(out)


def _decode_leading_value(text: str) -> Any:
    """Decode the first complete JSON value, ignoring anything after it.

    Returns ``_NO_VALUE`` when nothing decodes, so a legitimate ``None`` result
    stays distinguishable from a failure.
    """
    try:
        value, _ = json.JSONDecoder().raw_decode(text.lstrip())
    except json.JSONDecodeError:
        return _NO_VALUE
    return value


def loads_llm_json(raw: str | None) -> Any:
    """Parse JSON from an LLM response, repairing the defects models emit.

    Raises the original :class:`json.JSONDecodeError` when no attempt succeeds.
    """
    stripped = strip_json_fence(raw)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as first_error:
        # Each repair targets a different observed defect, and they compose:
        # a response can carry both a pasted multi-line span and a trailing
        # comma. Try the cheapest first and the combination last.
        repaired = repair_json_strings(stripped)
        candidates = (
            stripped,
            repaired,
            strip_trailing_commas(stripped),
            strip_trailing_commas(repaired),
        )
        for candidate in candidates:
            value = _decode_leading_value(candidate)
            if value is not _NO_VALUE:
                return value
        raise first_error from None

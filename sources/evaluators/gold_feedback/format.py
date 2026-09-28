"""Gold-gradient text: the benchmark grader's feedback in the E33 format.

Adapted from the E33 'TRUMAN' transcript format
(``experiments_verifiers/harness/e33_truman.py`` ``gold_gradient_text``):
the decisive execution outcome comes first, then the grader verdict
verbatim, with its specific values / columns / figure critiques. For
production the provenance paragraph and campaign header are dropped (no
information, they eat the budget) and the text is compacted to
``MAX_GRADIENT_CHARS`` so the mutator never clips the decisive part.

No oracle label is put in the text itself: the label would change the
directive LLM's behaviour compared with the other gradient channels. The
oracle flags live in ``state_result.json``, ``evaluation.txt`` and the logs.
"""

from __future__ import annotations

import ast
import re
from typing import Any

MAX_GRADIENT_CHARS = 2000

EXEC_FAILURE = "exec_failure"
FIGURE_JUDGE = "figure_judge"
EVAL_SCRIPT_ERROR = "eval_script_error"
PROG_CHECK_FAIL = "prog_check_fail"
PROG_CHECK_SUCCESS = "prog_check_success"

_UNINFORMATIVE_NOTE = "Note: the grader gave no detail beyond pass/fail."
_NOT_GRADED = (
    "Not graded: the grader checks the deliverable only after a clean re-run "
    "of the delivered script, so SR is false."
)
_ERROR_LINE = re.compile(
    r"^\s*(?:[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Fault)"
    r"|AssertionError|KeyboardInterrupt)\b.*"
)
_RATIO = re.compile(r"^\s*\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\s*$")
_SCORE_LINE = re.compile(r"\[FINAL SCORE\]\s*:?\s*[^\n]*", re.IGNORECASE)


def parse_critiques(message: Any) -> list[str] | None:
    """Return the figure-judge critiques when *message* is a list of strings.

    The visual judge's SR message is the ``str()`` of a list of critiques.

    Args:
        message: The raw ``SR_message`` (str or list).

    Returns:
        The critiques, or ``None`` when *message* is not such a list.
    """
    value = message
    if isinstance(message, str):
        try:
            value = ast.literal_eval(message.strip())
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
    if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
        return value
    return None


def is_uninformative(message: Any) -> bool:
    """True when a grader message carries no detail beyond pass/fail.

    Covers the by-design cases seen in the frozen ablations: empty, ``N/A``,
    bare ``a / b`` ratios, bare booleans and the "no result tuple" parse
    failure.

    Args:
        message: A ``VER_message`` or ``SR_message``.

    Returns:
        ``True`` if the message is uninformative.
    """
    text = str(message if message is not None else "").strip()
    if not text:
        return True
    if text.lower() in {"n/a", "na", "none", "true", "false", "0", "1"}:
        return True
    if _RATIO.match(text):
        return True
    return text.startswith("No (status, message) result tuple in eval output")


def decisive_error_line(text: str) -> str | None:
    """Return the decisive error line of a traceback-like message.

    Args:
        text: A VER or SR message (may hold a full traceback).

    Returns:
        The LAST line that looks like a raised exception, else the timeout /
        missing-output sentence, else ``None``.
    """
    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        if _ERROR_LINE.match(line):
            return line
    for line in lines:
        low = line.lower()
        if "timeout" in low or "not created" in low or "no python file" in low:
            return line
    return None


def classify(grade: dict[str, Any]) -> str:
    """Feedback kind of one graded generation (E33 kinds + eval-script error).

    Args:
        grade: Normalised grade (``VER``, ``SR``, ``SR_message`` ...).

    Returns:
        One of ``exec_failure``, ``figure_judge``, ``eval_script_error``,
        ``prog_check_fail``, ``prog_check_success``.
    """
    if not grade.get("VER"):
        return EXEC_FAILURE
    sr_msg = grade.get("SR_message")
    if parse_critiques(sr_msg):
        return FIGURE_JUDGE
    text = str(sr_msg or "")
    if not grade.get("SR") and (
        text.startswith(
            ("Eval script failed", "Evaluation error", "Evaluation timeout")
        )
    ):
        return EVAL_SCRIPT_ERROR
    return PROG_CHECK_SUCCESS if grade.get("SR") else PROG_CHECK_FAIL


def clip(text: str, budget: int, head_share: float = 0.6) -> str:
    """Clip *text* to at most *budget* chars, keeping head and tail.

    Args:
        text: Text to clip.
        budget: Maximum length of the result (marker included).
        head_share: Share of the kept chars taken from the head.

    Returns:
        *text* unchanged when it fits, else head + ``…`` marker + tail.
    """
    text = str(text or "").strip()
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    marker = f"\n… ({len(text)} chars, clipped) …\n"
    keep = budget - len(marker)
    if keep < 20:
        return text[:budget]
    head = int(keep * head_share)
    tail = keep - head
    return text[:head] + marker + (text[-tail:] if tail > 0 else "")


def _figure_block(critiques: list[str], budget: int) -> list[str]:
    """Score lines of every critique first, then critique bodies in order."""
    out: list[str] = []
    for i, crit in enumerate(critiques, 1):
        score = _SCORE_LINE.search(crit)
        out.append(
            f"grader critique {i}: {score.group(0).strip() if score else '(no score line)'}"
        )
    remaining = budget - sum(len(x) + 1 for x in out)
    for i, crit in enumerate(critiques, 1):
        header = f"### grader critique {i} (verbatim)"
        share = remaining // max(1, len(critiques) - i + 1)
        body = clip(crit, share - len(header) - 2, head_share=0.5)
        if not body:
            break
        out.append(header)
        out.append(body)
        remaining -= len(header) + len(body) + 2
    return out


def build_gold_gradient(
    grade: dict[str, Any], max_chars: int = MAX_GRADIENT_CHARS
) -> str:
    """Assemble the gold gradient, decisive content first, ``<= max_chars``.

    Args:
        grade: Normalised grade from ``snapshot_grading.grade_directory``
            (``VER``, ``VER_message``, ``SR``, ``SR_message``, ``CBS``) with
            ``status="evaluated"``.
        max_chars: Hard length cap of the returned text.

    Returns:
        The gradient text: execution outcome section, then grader verdict
        section. Never longer than *max_chars*.
    """
    kind = classify(grade)
    ver_msg = str(grade.get("VER_message") or "").strip()
    sr_msg = str(grade.get("SR_message") or "").strip()
    lines: list[str] = []

    if kind == EXEC_FAILURE:
        lines.append(
            "## Execution outcome (benchmark re-run of the delivered script): FAILED"
        )
        decisive = decisive_error_line(ver_msg)
        if decisive:
            lines.append(f"Decisive error: {clip(decisive, 400)}")
        if is_uninformative(ver_msg):
            lines.append(_UNINFORMATIVE_NOTE)
        tail_lines = [
            "## Grader verdict against the gold reference",
            _NOT_GRADED,
        ]
        fixed = (
            sum(len(x) + 1 for x in lines + tail_lines)
            + len("Execution output (verbatim excerpt):\n")
            + 1
        )
        if ver_msg and ver_msg != decisive:
            lines.append("Execution output (verbatim excerpt):")
            # Tracebacks end with the decisive frames: keep more tail.
            lines.append(clip(ver_msg, max_chars - fixed, head_share=0.2))
        lines.extend(tail_lines)
    else:
        lines.append(
            "## Execution outcome (benchmark re-run of the delivered script): OK"
        )
        verdict = "PASS" if grade.get("SR") else "FAIL"
        lines.append(f"## Grader verdict against the gold reference: {verdict}")
        note = [_UNINFORMATIVE_NOTE] if is_uninformative(sr_msg) else []
        budget = max_chars - sum(len(x) + 1 for x in lines + note)
        if kind == FIGURE_JUDGE:
            lines.extend(_figure_block(parse_critiques(sr_msg) or [], budget))
        elif kind == EVAL_SCRIPT_ERROR:
            decisive = decisive_error_line(sr_msg)
            if decisive:
                lines.append(
                    f"Grader program error on the deliverable: {clip(decisive, 400)}"
                )
                budget -= len(lines[-1]) + 1
            lines.append("Grader output (verbatim excerpt):")
            lines.append(clip(sr_msg, budget - 40, head_share=0.2))
        else:
            lines.append("Grader message (verbatim):")
            lines.append(clip(sr_msg or "(empty)", budget - 30, head_share=0.7))
        lines.extend(note)

    text = "\n".join(x for x in lines if x is not None).strip()
    if len(text) > max_chars:
        marker = "\n… (clipped)"
        text = text[: max_chars - len(marker)] + marker
    return text


def censored_note(reason: str, max_chars: int = 300) -> str:
    """One-paragraph note for a censored (grader unavailable) generation.

    Args:
        reason: The infra reason (exclusion message or harness error).
        max_chars: Cap on the quoted reason.

    Returns:
        A short note to put above the fallback (hybrid) gradient.
    """
    return (
        "Grader feedback unavailable for this generation: the grading "
        f"environment failed (result censored: {clip(reason or 'unknown', max_chars)}). "
        "The feedback below comes from the standard verifier."
    )

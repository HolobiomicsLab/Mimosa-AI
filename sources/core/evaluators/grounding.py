
import re

from sources.utils.perspicacite_client import query_perspicacite


_CSV_TOKEN_PATTERN = re.compile(r"^[A-Za-z_][\w]{0,40}$")
_MIN_CSV_IDENT_TOKENS = 2
_MAX_CSV_TOKENS = 30

_IDENTIFIER_PATTERN = re.compile(
    r"`([^`\s]{2,})`"                      # backticked token
    r"|\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b"  # ALL_CAPS_WITH_UNDERSCORE
    r"|\b([A-Z][A-Z0-9]{3,})\b"            # short ALL-CAPS (>=4 chars)
    r"|\b([A-Z][a-z]+(?:[A-Z][a-z0-9]+){1,})\b"  # CamelCase
)


def _extract_cues(goal: str) -> dict[str, list[str]]:
    """Extract literature-search cues from a workflow goal.

    Returns a dict with two lists. ``header_line`` holds at most one CSV-like
    header line found verbatim inside the goal (empty when none was visible);
    ``identifiers`` holds distinct capitalised or backticked code-identifier
    tokens in first-seen order. Both lists may be empty — never ``None``.
    """
    return {
        "header_line": [line] if (line := _first_header_line(goal)) else [],
        "identifiers": _capitalised_identifiers(goal),
    }


def _first_header_line(goal: str) -> str:
    """Return the first CSV-like header line in *goal*, or ``""`` if absent."""
    for raw in (goal or "").splitlines():
        line = raw.strip()
        if "," not in line:
            continue
        tokens = [t.strip() for t in line.split(",")]
        if not (_MIN_CSV_IDENT_TOKENS <= len(tokens) <= _MAX_CSV_TOKENS):
            continue
        ident_count = sum(1 for t in tokens if _CSV_TOKEN_PATTERN.match(t))
        if ident_count >= _MIN_CSV_IDENT_TOKENS:
            return line
    return ""


def _capitalised_identifiers(goal: str) -> list[str]:
    """Distinct capitalised / backticked identifier tokens, first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _IDENTIFIER_PATTERN.finditer(goal or ""):
        token = next((g for g in match.groups() if g), "")
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def get_perspicacite_grounding(goal: str) -> str:
    """Retrieve a literature grounding block for a workflow goal.

    Delegates to the local Perspicacite agentic retriever. Augments the
    primary goal-text query with two cue-derived sub-search seeds
    (verbatim example-data header line, capitalised identifiers) so
    dataset-canonical papers — whose fingerprint is in the example
    data — have a chance of surfacing alongside generic background.
    """
    cues = _extract_cues(goal)
    header_text = cues["header_line"][0] if cues["header_line"] else "(no example-data preview detected)"
    ids_text = ", ".join(cues["identifiers"]) if cues["identifiers"] else "(no capitalised identifiers detected)"
    prompt = (
        "TASK: Provide scientific literature evidence for the goal below.\n\n"
        "SCIENCE GOAL:\n"
        f"{goal}\n\n"
        "ADDITIONAL FINGERPRINTS to drive your literature search:\n"
        f"- Example-data header line (verbatim): {header_text}\n"
        f"- Capitalised identifiers worth a direct lookup: {ids_text}\n\n"
        "SUB-SEARCHES YOU MUST RUN:\n"
        "1. The goal text as written (broad context).\n"
        "2. The example-data header line as a verbatim query — if it\n"
        "   uniquely identifies a published dataset or methodology,\n"
        "   return the canonical paper that introduced those columns\n"
        "   AND any quantitative thresholds, cutoffs, or hyperparameters\n"
        "   that paper specifies (binary activity cutoffs, AUC bars,\n"
        "   train/test split conventions, recommended featurisers).\n"
        "3. Each capitalised identifier as a query, scoped to the\n"
        "   goal's domain.\n\n"
        "INSTRUCTIONS:\n"
        "1. INFER SUCCESS INDICATORS — what does the literature say\n"
        "   about how to measure success on this task? Cite specific\n"
        "   papers.\n"
        "2. SEARCH — run the sub-searches above. Prefer Semantic Scholar\n"
        "   + OpenAlex + PubMed; require at least one peer-reviewed\n"
        "   source.\n"
        "3. PROVIDE EVIDENCE — quote relevant passages, give DOIs.\n"
        "4. SUMMARIZE — a structured grounding block consumable by\n"
        "   downstream claim extractors. If the literature is silent\n"
        "   on a specific threshold or hyperparameter, say so —\n"
        "   do not invent a number.\n"
    )
    try:
        response = query_perspicacite(prompt)
        return response
    except Exception:
        return "Perspicacite query failed, unable to provide grounded expectations. Proceeding without external grounding."

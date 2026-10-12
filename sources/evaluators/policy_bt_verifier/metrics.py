"""Deterministic metric ledger for the policy_bt verifier (review §4.1).

One source of truth for primary-metric values. The recon supplies, per
metric, a stable ``id`` plus ``aliases`` — the EXACT digest keys it has
seen in the generation slots (the semantic matching that lexical code
cannot do, because key drift between LLM-written workflows is semantic,
not lexical). This module then resolves each metric to a value
deterministically and NEVER guesses: several candidates with different
values are reported ``ambiguous``.

The ledger replaces three independent metric-comparison implementations
that could silently disagree (the LLM policy's free-form key search, the
gradient fold's tie regex, and the evaluator's fuzzy lookup — review
§2.2): the policy, the fold and any exploration logic all read
``digest["primary_values"]``.

Resolution order per metric (review §4.1):

1. Exact alias hits (file + key), preferring a file that is
   ``produced_this_run=True``, then a non-verification file (not
   ``independent_*``/``verif*``/``check*``), then the alias order.
2. Strict lexical fallback only when no alias hits: EVERY core query
   token (from the metric id) must be present in the candidate key,
   verification-echo keys are excluded, and ``log_metrics`` are searched
   too.
3. Several candidates with different values → ``ambiguous`` (never guess).
4. Nothing → ``missing``.
"""

from __future__ import annotations

import math
import re
from typing import Any

#: Verifier-written echo FILES: they restate solver metrics (possibly
#: stale or recomputed under a different convention), so they lose every
#: alias preference and are excluded from the lexical fallback entirely.
_VERIF_FILE_RE = re.compile(r"(?:^|/)(?:independent_|verif|check)", re.I)

#: Verifier-echo KEY prefixes inside any file (``verification.elapsed``,
#: ``solver_claimed.optimal_count_raw``, ``reported_energy`` ...).
_VERIF_KEY_RE = re.compile(
    r"(?:^|\.)(?:verification|verifier|solver_claimed|reported"
    r"|recomputed|independent)",
    re.I,
)

#: English function words that never identify a quantity. Deliberately
#: MINIMAL: ``count``/``number``/``total`` stay tokens — they are generic
#: but not stopwords, and the every-token-required rule tolerates them.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "for", "in", "to", "and", "or", "is",
        "are", "on", "at", "by", "with", "this", "that", "from", "as",
    }
)

#: Cap on candidates listed in an ambiguous result (audit trail, not a
#: file manifest).
_MAX_CANDIDATES = 6


def _tokens(text: Any) -> list[str]:
    """Distinct query tokens of a name/id: lowercase alphanumerics.

    English stopwords and single characters drop out; a trailing ``s``
    is stemmed (``contacts`` → ``contact``) so singular/plural drift
    between generations still matches.
    """
    out: list[str] = []
    for tok in re.split(r"[^a-z0-9]+", str(text or "").lower()):
        if len(tok) < 2 or tok in _STOPWORDS:
            continue
        if len(tok) > 3 and tok.endswith("s"):
            tok = tok[:-1]
        if tok and tok not in out:
            out.append(tok)
    return out


def _tok_match(query: str, cand: str) -> bool:
    """Token equality, or a prefix match with the shorter side >= 3 chars.

    ``min`` ≈ ``minimum``, ``saw`` ≈ ``saws`` (stemming usually covers
    the plural; prefixes cover abbreviations).
    """
    if query == cand:
        return True
    short, long_ = (query, cand) if len(query) <= len(cand) else (cand, query)
    return len(short) >= 3 and long_.startswith(short)


def _scalar(value: Any) -> float | None:
    """Finite numeric value of a digest leaf (booleans/markers excluded)."""
    if isinstance(value, bool | dict):
        return None
    if isinstance(value, int | float) and math.isfinite(value):
        return float(value)
    return None


def _is_verif_file(fname: str) -> bool:
    """True for verifier-echo file names (``independent_*``/``verif*``/``check*``)."""
    stem = str(fname).rsplit("/", 1)[-1]
    return bool(_VERIF_FILE_RE.match(stem))


def _is_verif_key(key: str) -> bool:
    """True for keys naming a verification echo inside any file."""
    return bool(_VERIF_KEY_RE.search(str(key)))


def _query_tokens(metric: dict[str, Any]) -> list[str]:
    """Core query tokens: the metric ``id`` (fallback: the ``key``)."""
    return _tokens(metric.get("id") or metric.get("key") or "")


def _freshness_map(digest: dict[str, Any]) -> dict[str, bool | None]:
    """``deliverables[].path -> produced_this_run`` (True/False/None)."""
    out: dict[str, bool | None] = {}
    for dv in digest.get("deliverables") or []:
        if isinstance(dv, dict) and dv.get("path"):
            fresh = dv.get("produced_this_run")
            out[str(dv["path"])] = (
                bool(fresh) if isinstance(fresh, bool) else None
            )
    return out


def _rank(fresh: bool | None, verif: bool) -> tuple[int, int]:
    """Sort rank: fresh file first, then non-verification file.

    Unknown freshness ranks between fresh and stale (the file was not in
    the deliverables inventory at all).
    """
    return (
        0 if fresh is True else (2 if fresh is False else 1),
        1 if verif else 0,
    )


def _result(
    status: str,
    value: float | None = None,
    source: str | None = None,
    fresh: bool | None = None,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One metric's ledger entry."""
    return {
        "status": status,
        "value": value,
        "source": source,
        "fresh": fresh,
        "candidates": candidates or [],
    }








def extract_text_metrics_llm(
    text_lines: list[str],
    metric_keys: list[str],
    llm_json: Any = None,
) -> dict[str, float | None]:
    """Extract numeric metric values from raw text via a cached LLM call.

    This replaces ALL regex-based text scraping (operator directive:
    no regex, no brittle patterns, no task-specific patterns — the LLM
    reads the program output and reports what it finds).

    Args:
        text_lines: Raw output lines (exec.log_tail + text file contents).
        metric_keys: The metric names/keys to look for (from the recon).
        llm_json: Callable ``(prompt, max_tokens, role) -> parsed JSON or
            None``. When None, returns all-missing (deterministic fallback).

    Returns:
        ``{metric_key: float | None}`` — the extracted value or None.
    """
    if not llm_json or not text_lines or not metric_keys:
        return {k: None for k in metric_keys}

    text = "\n".join(text_lines[:100])  # cap at 100 lines
    keys_str = ", ".join(f'"{k}"' for k in metric_keys)
    prompt = (
        "You are a precise data extraction instrument. Given the raw output"
        " of a scientific program below, extract the FINAL numeric value"
        " for each named metric. Report only numbers that appear in the"
        " output text — never compute, estimate, or recall from memory.\n\n"
        f"# Metrics to extract: [{keys_str}]\n\n"
        f"# Program output:\n{text}\n\n"
        "Return STRICT JSON: {\"<metric_name>\": <number or null>, ...}"
        " — one key per metric, null when not found in the output."
    )
    parsed = llm_json(prompt, 2000, "metric_text_extract")
    if isinstance(parsed, dict):
        return {
            k: float(parsed[k]) if isinstance(parsed.get(k), int | float)
            and not isinstance(parsed.get(k), bool) else None
            for k in metric_keys
        }
    return {k: None for k in metric_keys}


def _resolve_one(
    metric: dict[str, Any],
    json_metrics: dict[str, dict[str, Any]],
    log_metrics: list[dict[str, Any]],
    freshness: dict[str, bool | None],
    text_lines: list[str] | None = None,
    llm_extract: Any = None,
) -> dict[str, Any]:
    """Resolve one metric through the resolution ladder.

    Order: (1) json_metrics alias hits, (1b) LLM text extraction from
    the re-execution output when aliases fail, (2) strict lexical
    fallback in json_metrics, (3) log_metrics context matching.
    """

    # --- 1. exact alias hits, ranked fresh > non-verification > order ---
    hits: list[tuple[tuple[int, int], int, str, float, bool | None]] = []
    for ai, alias in enumerate(metric.get("aliases") or []):
        if not isinstance(alias, dict):
            continue
        fname = str(alias.get("file") or "")
        key = str(alias.get("key") or "")
        if not key:
            continue
        value = _scalar((json_metrics.get(fname) or {}).get(key))
        if value is None:
            continue
        fresh = freshness.get(fname)
        hits.append((_rank(fresh, _is_verif_file(fname)), ai, fname, value, fresh))
    if hits:
        hits.sort(key=lambda h: (h[0], h[1]))
        best_rank, _, best_file, best_value, best_fresh = hits[0]
        same_rank_vals = {h[3] for h in hits if h[0] == best_rank}
        cands = [
            {"source": f"{h[2]}:{_alias_key(metric, h[1])}", "value": h[3]}
            for h in hits[:_MAX_CANDIDATES]
        ]
        if len(same_rank_vals) == 1:
            return _result("resolved", best_value, f"{best_file}:"
                           f"{_alias_key(metric, hits[0][1])}", best_fresh, cands)
        # Equal-preference aliases disagree → never guess.
        return _result("ambiguous", candidates=cands)

    # --- 1b. LLM text extraction (replaces ALL regex — operator directive) ---
    # When json_metrics aliases fail, ask the LLM to read the raw program
    # output and extract the metric values. Generic, no patterns.
    if text_lines and llm_extract:
        mid_key = str(metric.get("id") or metric.get("key") or "").strip()
        alias_keys = [str(a.get("key") or "") for a in
                      (metric.get("aliases") or []) if isinstance(a, dict)]
        search_keys = [k for k in [mid_key] + alias_keys if k]
        if search_keys:
            extracted = llm_extract(text_lines, search_keys)
            for k, v in extracted.items():
                if v is not None:
                    return _result("resolved", v, f"llm:{k}", None, [])

    # --- 2. strict lexical fallback (no alias hit anything) ---
    query = _query_tokens(metric)
    if not query:
        return _result("missing")
    cands: list[dict[str, Any]] = []
    for fname in sorted(json_metrics):
        if _is_verif_file(fname):
            continue
        key_tokens_cache: dict[str, list[str]] = {}
        for key, value in sorted((json_metrics.get(fname) or {}).items()):
            if _scalar(value) is None or _is_verif_key(key):
                continue
            if key not in key_tokens_cache:
                key_tokens_cache[key] = _tokens(key)
            ktok = key_tokens_cache[key]
            if all(any(_tok_match(q, c) for c in ktok) for q in query):
                cands.append(
                    {"source": f"{fname}:{key}", "value": float(value)}
                )
    for entry in log_metrics:
        if not isinstance(entry, dict):
            continue
        value = _scalar(entry.get("value"))
        name = str(entry.get("metric") or "")
        if value is None or _is_verif_key(name):
            continue
        mtok = _tokens(name)
        if mtok and all(any(_tok_match(q, c) for c in mtok) for q in query):
            src = str(entry.get("source") or "log")
            cands.append({"source": f"{src}:{name}", "value": value})
    if not cands:
        return _result("missing")
    vals = {c["value"] for c in cands}
    if len(vals) == 1:
        first = cands[0]
        fname = first["source"].split(":", 1)[0]
        return _result(
            "resolved", first["value"], first["source"],
            freshness.get(fname), cands[:_MAX_CANDIDATES],
        )
    return _result("ambiguous", candidates=cands[:_MAX_CANDIDATES])


def _alias_key(metric: dict[str, Any], alias_index: int) -> str:
    """The exact key of alias *alias_index* (for the ``source`` string)."""
    aliases = metric.get("aliases") or []
    if 0 <= alias_index < len(aliases) and isinstance(aliases[alias_index], dict):
        return str(aliases[alias_index].get("key") or "")
    return ""


def resolve_primary_values(
    digest: dict[str, Any] | None,
    primary_metrics: list[Any] | None,
    llm_extract: Any = None,
) -> dict[str, dict[str, Any]]:
    """Resolve every recon primary metric against one generation's digest.

    Searches THREE sources in order: json_metrics (structured), exec.log_tail
    + text files (the re-execution output), and log_metrics (scraped lines).
    """
    digest = digest or {}
    json_metrics = {
        str(f): (m if isinstance(m, dict) else {})
        for f, m in (digest.get("json_metrics") or {}).items()
    }
    log_metrics = [
        e for e in (digest.get("log_metrics") or []) if isinstance(e, dict)
    ]
    # Raw text from the re-execution output + deliverable file contents
    # (HP-folding fix: "Minimum energy found: -9" lives HERE, not in json)
    exec_tail = (digest.get("exec") or {}).get("log_tail") or []
    if isinstance(exec_tail, str):
        exec_tail = exec_tail.splitlines()
    text_lines = [str(ln) for ln in exec_tail]
    # Also include scraped log_metrics raw lines
    text_lines.extend(str(e.get("raw_line", "")) for e in log_metrics)
    freshness = _freshness_map(digest)
    # Raw text from exec.log_tail for LLM extraction
    exec_tail = (digest.get("exec") or {}).get("log_tail") or []
    if isinstance(exec_tail, str):
        exec_tail = exec_tail.splitlines()
    text_lines = [str(ln) for ln in exec_tail if ln.strip()]
    out: dict[str, dict[str, Any]] = {}
    for i, metric in enumerate(primary_metrics or []):
        if not isinstance(metric, dict):
            continue
        mid = str(metric.get("id") or "").strip()
        if not mid:
            key = str(metric.get("key") or "").strip()
            mid = "_".join(_tokens(key))[:40] or f"metric_{i}"
        if mid in out:
            mid = f"{mid}_{i}"
        out[mid] = _resolve_one(
            metric, json_metrics, log_metrics, freshness,
            text_lines=text_lines, llm_extract=llm_extract,
        )
    return out

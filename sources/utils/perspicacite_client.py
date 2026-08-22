"""
Perspicacite-AI client for Mimosa-AI.

Queries the Perspicacite-AI REST API to retrieve grounded scientific knowledge
that can inform workflow generation. If Perspicacite is offline or unreachable,
a warning is logged and execution continues normally.

**Important – timeouts**:  The agentic RAG pipeline performs multiple LLM calls,
literature searches across several APIs, PDF downloads, relevance scoring, and
answer synthesis.  A single request can easily take 1-5+ minutes.  This client
therefore uses *streaming* by default (SSE) so intermediate "thinking" events
keep the connection alive, and sets generous read timeouts.
"""

import hashlib
import json
import logging
import os
import pathlib
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request cache
# ---------------------------------------------------------------------------
_CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "memory" / "perspicacite_requests"

def _cache_key(science_query: str, mode: str) -> str:
    """Return a deterministic hex digest for a (query, mode) pair."""
    blob = json.dumps({"query": science_query, "mode": mode}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _read_cache(science_query: str, mode: str) -> Optional[str]:
    """Return the cached answer for *science_query* + *mode*, or ``None``."""
    key = _cache_key(science_query, mode)
    cache_file = _CACHE_DIR / f"{key}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        logger.info(
            "[Perspicacite] Cache HIT for query (mode=%s): %s…",
            mode, science_query[:80],
        )
        return data.get("answer")
    except Exception:
        logger.debug("[Perspicacite] Failed to read cache file %s", cache_file)
        return None


def _write_cache(science_query: str, mode: str, answer: str) -> None:
    """Persist *answer* to disk so future identical requests are served from cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(science_query, mode)
    cache_file = _CACHE_DIR / f"{key}.json"
    payload = {
        "query": science_query,
        "mode": mode,
        "answer": answer,
    }
    try:
        cache_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("[Perspicacite] Cached answer → %s", cache_file.name)
    except Exception as exc:
        logger.debug("[Perspicacite] Failed to write cache file: %s", exc)

# Default Perspicacite API base URL (can be overridden by env var)
PERSPICACITE_BASE_URL = os.environ.get(
    "PERSPICACITE_API_URL", "http://localhost:8000"
)

# ---------------------------------------------------------------------------
# Runtime settings + telemetry
# ---------------------------------------------------------------------------
# Grounding degrades silently by design: every failure path returns ``None``
# and each caller substitutes a "no context" string, so a run in which every
# grounding call failed produces the same artifacts as a fully grounded one.
# The ledger below makes the difference observable — ``grounding_stats()`` is
# folded into ``run_metrics.json`` so a run can be shown to have been grounded
# rather than assumed to have been.
_GROUNDING_LEDGER: list[dict] = []

# ``kb_name=None`` sends the query to Perspicacite's web-search pipeline
# (literature APIs + PDF download + synthesis). Naming a local knowledge base
# instead scopes retrieval to that corpus, which is both far faster and
# reproducible. Left at None so default behaviour is unchanged.
_SETTINGS: dict = {"kb_name": None, "mode": "agentic", "max_papers": 5}


def configure(
    kb_name: str | None = None,
    mode: str | None = None,
    max_papers: int | None = None,
) -> None:
    """Set process-wide grounding defaults (call once, after config load)."""
    if kb_name is not None:
        _SETTINGS["kb_name"] = kb_name
    if mode is not None:
        _SETTINGS["mode"] = mode
    if max_papers is not None:
        _SETTINGS["max_papers"] = int(max_papers)
    logger.info("[Perspicacite] settings: %s", _SETTINGS)


def configure_from_config(config) -> None:
    """Apply ``perspicacite_*`` fields from a Mimosa ``Config`` object."""
    configure(
        kb_name=getattr(config, "perspicacite_kb_name", None),
        mode=getattr(config, "perspicacite_mode", None),
        max_papers=getattr(config, "perspicacite_max_papers", None),
    )


def _record(outcome: str, seconds: float, chars: int, kb_name: str | None) -> None:
    """Append one grounding attempt to the in-process ledger."""
    _GROUNDING_LEDGER.append({
        "outcome": outcome,
        "seconds": round(seconds, 3),
        "answer_chars": chars,
        "kb_name": kb_name,
        "mode": _SETTINGS["mode"],
    })


def grounding_stats() -> dict:
    """Summarise grounding attempts made so far in this process.

    ``grounded`` is the count of attempts that returned usable context —
    whether freshly retrieved or served from the on-disk cache. A run with
    ``attempts > 0`` and ``grounded == 0`` completed with no literature
    grounding at all, which is otherwise invisible in the artifacts.
    """
    total = len(_GROUNDING_LEDGER)
    by = {}
    for e in _GROUNDING_LEDGER:
        by[e["outcome"]] = by.get(e["outcome"], 0) + 1
    grounded = by.get("ok", 0) + by.get("cache_hit", 0)
    secs = sum(e["seconds"] for e in _GROUNDING_LEDGER)
    return {
        "attempts": total,
        "grounded": grounded,
        "hit_rate": round(grounded / total, 4) if total else None,
        "by_outcome": by,
        "total_seconds": round(secs, 3),
        "mean_seconds": round(secs / total, 3) if total else None,
        "kb_name": _SETTINGS["kb_name"],
        "mode": _SETTINGS["mode"],
    }


def reset_grounding_stats() -> None:
    """Clear the ledger (used between tasks in batch evaluation)."""
    _GROUNDING_LEDGER.clear()

# Timeout configuration.
# The agentic pipeline involves multiple LLM calls, literature searches, paper
# downloads, and answer synthesis — easily taking 1-5+ minutes.
# - connect timeout:  how long to wait for the TCP connection (short is fine)
# - read timeout:     how long to wait *between* chunks of data.  For streaming
#                     this is the gap between SSE events; for non-streaming it
#                     is the total wall-clock time until the full response body
#                     arrives.
_CONNECT_TIMEOUT = 30       # seconds – TCP connect
_READ_TIMEOUT = 600         # seconds – between data chunks (10 min)
# NOTE: httpx has no single "overall" cap; the read timeout above bounds the
# gap between SSE events, which is the operative limit for the streaming path.


def _build_httpx_timeout():
    """Build an ``httpx.Timeout`` with separate connect / read / write caps."""
    import httpx
    return httpx.Timeout(
        connect=_CONNECT_TIMEOUT,
        read=_READ_TIMEOUT,
        write=60.0,
        pool=30.0,
    )


def query_perspicacite(
    science_query: str,
    mode: str = "agentic",
    base_url: str = PERSPICACITE_BASE_URL,
) -> Optional[str]:
    """Query Perspicacite-AI for scientific knowledge relevant to a task.

    Uses the ``/api/chat`` endpoint in **streaming** mode (SSE) by default.
    Streaming is strongly preferred because intermediate "thinking" events
    keep the HTTP connection alive even while the server performs long-running
    operations (literature search, PDF download, LLM synthesis, etc.).

    Falls back to non-streaming JSON mode only if streaming fails.

    Args:
        science_query: The scientific question / research goal.
        mode:     RAG mode to use in Perspicacite (``"basic"``, ``"advanced"``,
                  ``"profound"`` or ``"agentic"``).  Defaults to ``"agentic"``
                  for a good balance between speed and quality.
        base_url: Base URL of the Perspicacite web application.

    Returns:
        A string containing the scientific context retrieved from Perspicacite,
        or ``None`` if the service is unavailable or the query fails.
    """
    import time as _time

    kb_name = _SETTINGS["kb_name"]
    started = _time.perf_counter()

    # ---- check the on-disk cache first ----
    cached = _read_cache(science_query, mode)
    if cached is not None:
        _record("cache_hit", _time.perf_counter() - started, len(cached), kb_name)
        return cached

    # Try streaming first (preferred — keeps connection alive during long ops)
    result = _query_perspicacite_streaming(science_query, mode, base_url)
    if result:
        _write_cache(science_query, mode, result)
        _record("ok", _time.perf_counter() - started, len(result), kb_name)
        return result

    # Fall back to non-streaming JSON if streaming failed
    logger.info("[Perspicacite] Streaming failed; trying non-streaming fallback.")
    result = _query_perspicacite_non_streaming(science_query, mode, base_url)
    if result:
        _write_cache(science_query, mode, result)
        _record("ok", _time.perf_counter() - started, len(result), kb_name)
    else:
        _record("failed", _time.perf_counter() - started, 0, kb_name)
    return result


def _query_perspicacite_streaming(
    science_query: str,
    mode: str,
    base_url: str,
) -> Optional[str]:
    """Query Perspicacite using SSE streaming.

    Streaming is the preferred transport because:
    - The server emits "thinking" / "status" events every few seconds, which
      reset the read-timeout counter → the connection stays alive even when a
      single internal step takes a long time.
    - We get visibility into server progress in the log.
    """
    import base64
    import json

    try:
        import httpx
    except ImportError:
        logger.warning(
            "⚠️ [Perspicacite] httpx not available – cannot query Perspicacite API."
        )
        return None

    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "query": science_query,
        "mode": mode,
        "stream": True,           # SSE streaming
        # None → Perspicacite's web-search pipeline; a KB name scopes
        # retrieval to that local corpus instead (see configure()).
        "kb_name": _SETTINGS["kb_name"],
        "max_papers": _SETTINGS["max_papers"],
        "databases": ["semantic_scholar", "openalex", "pubmed"],
    }

    timeout = _build_httpx_timeout()
    full_answer = ""

    try:
        logger.info(
            f"[Perspicacite] Querying {url} (streaming, mode={mode}, "
            f"query={science_query[:80]}...)"
        )
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    # Log intermediate progress so the caller can see activity
                    if event_type in ("thinking", "status"):
                        msg = event.get("message", "")
                        logger.debug(f"[Perspicacite] {event_type}: {msg}")

                    elif event_type == "answer":
                        # Answer may be base64-encoded (to avoid mid-chunk
                        # JSON breakage over chunked HTTP)
                        content_b64 = event.get("content_b64")
                        if content_b64:
                            full_answer = base64.b64decode(content_b64).decode(
                                "utf-8", errors="replace"
                            )
                        elif "content" in event:
                            full_answer = str(event["content"])

                    elif event_type == "token":
                        # Live token delta (base64-encoded) — accumulate
                        delta_b64 = event.get("delta_b64")
                        if delta_b64:
                            full_answer += base64.b64decode(delta_b64).decode(
                                "utf-8", errors="replace"
                            )

                    elif event_type == "done":
                        break

                    elif event_type == "error":
                        err_msg = event.get("message", "Unknown server error")
                        logger.warning(f"⚠️ [Perspicacite] Server error: {err_msg}")
                        return None

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"⚠️ [Perspicacite] Streaming query failed: {exc}. "
            "Workflow generation will proceed without scientific grounding."
        )
        return None

    if not full_answer:
        logger.warning(
            "⚠️ [Perspicacite] Streaming response yielded no answer. "
            "Workflow generation will proceed without scientific grounding."
        )
        return None

    logger.info(
        "[Perspicacite] Successfully retrieved scientific context (streaming)."
    )
    return full_answer


def _query_perspicacite_non_streaming(
    science_query: str,
    mode: str,
    base_url: str,
) -> Optional[str]:
    """Query Perspicacite using a plain (non-streaming) JSON POST.

    **Warning**: The server must complete the *entire* RAG pipeline before it
    can return a response.  For ``agentic`` mode this easily takes 1-5+ minutes.
    The generous ``_READ_TIMEOUT`` (default 600 s) should cover most cases, but
    streaming is always preferred.
    """
    try:
        import httpx
    except ImportError:
        return None

    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "query": science_query,
        "mode": mode,
        "stream": False,
        "kb_name": _SETTINGS["kb_name"],
        "max_papers": _SETTINGS["max_papers"],
        "databases": ["semantic_scholar", "openalex", "pubmed"],
    }

    timeout = _build_httpx_timeout()

    try:
        logger.info(
            f"[Perspicacite] Querying {url} (non-streaming, mode={mode}, "
            f"query={science_query[:80]}...)"
        )
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
    except httpx.ConnectError:
        logger.warning(
            f"⚠️ [Perspicacite] Service unreachable at {base_url}. "
            "Workflow generation will proceed without scientific grounding."
        )
        return None
    except httpx.TimeoutException:
        logger.warning(
            f"⚠️ [Perspicacite] Request timed out (read_timeout={_READ_TIMEOUT}s). "
            "The agentic pipeline may need more time. "
            "Workflow generation will proceed without scientific grounding."
        )
        return None
    except httpx.HTTPStatusError as exc:
        logger.warning(
            f"⚠️ [Perspicacite] HTTP error {exc.response.status_code}: "
            f"{exc.response.text[:200]}. "
            "Workflow generation will proceed without scientific grounding."
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"⚠️ [Perspicacite] Unexpected error: {exc}. "
            "Workflow generation will proceed without scientific grounding."
        )
        return None

    try:
        data = response.json()
    except Exception:
        logger.warning(
            "⚠️ [Perspicacite] Could not parse JSON response. "
            "Workflow generation will proceed without scientific grounding."
        )
        return None

    # Server might signal that non-streaming is not supported
    if isinstance(data, dict) and "error" in data:
        logger.warning(
            f"⚠️ [Perspicacite] Server returned error: {data['error']}"
        )
        return None

    # Extract answer from non-streaming response
    answer = data.get("answer") or data.get("content") or data.get("message")
    if not answer:
        logger.warning(
            "⚠️ [Perspicacite] Response contained no extractable answer. "
            "Workflow generation will proceed without scientific grounding."
        )
        return None

    logger.info("[Perspicacite] Successfully retrieved scientific context.")
    return str(answer)


def format_scientific_context(
    task: str,
    scientific_context: str,
) -> str:
    """Wrap *scientific_context* in a structured section to be prepended to
    the workflow-generation instructions.

    This makes it easy for the LLM to locate and use the retrieved knowledge.

    Args:
        task:               The original task description.
        scientific_context: Raw text returned by Perspicacite.

    Returns:
        A formatted string ready to be prepended to ``craft_instructions``.
    """
    return (
        f"# SCIENTIFIC LITERATURE CONTEXT  (retrieved from Perspicacite-AI)\n"
        f"Scientific Knowledge:\n\n"
        f"{scientific_context.strip()}\n"
    )

if __name__ == "__main__":
    """test Perspicacite server."""
    import time
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    test_query = (
        "What are the main bioinformatics tools used for 16S rRNA amplicon "
        "sequence analysis in microbial ecology?"
    )

    t0 = time.time()
    result = query_perspicacite(test_query, mode="agentic")
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'='*70}")
    if result:
        print(f"\n🔍 Scientific knowledge retrieved ({len(result)} chars):\n")
        # Print first 1000 chars as preview
        print(result[:1000])
        if len(result) > 1000:
            print(f"\n... [{len(result) - 1000} more characters]")
    else:
        print("\n❌ No result returned (None)")
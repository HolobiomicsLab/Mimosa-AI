"""
Perspicacite-AI client for Mimosa-AI.

Queries the Perspicacite-AI REST API to retrieve grounded scientific knowledge
that can inform workflow generation. If Perspicacite is offline or unreachable,
a warning is logged and execution continues normally.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Default Perspicacite API base URL (can be overridden by env var)
PERSPICACITE_BASE_URL = os.environ.get(
    "PERSPICACITE_API_URL", "http://localhost:8000"
)

# Timeout in seconds for each request
_REQUEST_TIMEOUT = 30


def _build_science_query(task: str) -> str:
    """Build a focused scientific-literature query from a task description.

    We keep the query concise so Perspicacite can return relevant results
    without being confused by implementation details.
    """
    # Take the first 500 characters to avoid overwhelming the query
    return task[:500].strip()


def query_perspicacite(
    task: str,
    mode: str = "agentic",
    base_url: str = PERSPICACITE_BASE_URL,
    timeout: int = _REQUEST_TIMEOUT,
) -> Optional[str]:
    """Query Perspicacite-AI for scientific knowledge relevant to *task*.

    Uses the ``/api/chat`` endpoint with non-streaming mode so we can read
    the full answer synchronously.  Perspicacite is queried in web-search
    mode (no knowledge-base required) so no KB setup is needed on the
    Perspicacite side.

    Args:
        task:     The scientific task / research goal given to Mimosa.
        mode:     RAG mode to use in Perspicacite (``"basic"``, ``"advanced"``,
                  ``"profound"`` or ``"agentic"``).  Defaults to ``"agentic"``
                  for a good balance between speed and quality.
        base_url: Base URL of the Perspicacite web application.
        timeout:  HTTP request timeout in seconds.

    Returns:
        A string containing the scientific context retrieved from Perspicacite,
        or ``None`` if the service is unavailable or the query fails.
    """
    try:
        import httpx  # httpx is already a dependency of both projects
    except ImportError:
        logger.warning(
            "⚠️ [Perspicacite] httpx not available – cannot query Perspicacite API."
        )
        return None

    science_query = _build_science_query(task)
    url = f"{base_url.rstrip('/')}/api/chat"

    # Non-streaming request so we get a simple JSON response
    payload = {
        "query": science_query,
        "mode": mode,
        "stream": False,
        "kb_name": None,          # use web-search only, no specific KB needed
        "max_papers": 5,
        "databases": ["semantic_scholar", "openalex", "pubmed"],
    }

    try:
        logger.info(
            f"[Perspicacite] Querying {url} for scientific context "
            f"(mode={mode}, query={science_query[:80]}...)"
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
            f"⚠️ [Perspicacite] Request timed out after {timeout}s. "
            "Workflow generation will proceed without scientific grounding."
        )
        return None
    except httpx.HTTPStatusError as exc:
        logger.warning(
            f"⚠️ [Perspicacite] HTTP error {exc.response.status_code}: {exc.response.text[:200]}. "
            "Workflow generation will proceed without scientific grounding."
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            f"⚠️ [Perspicacite] Unexpected error: {exc}. "
            "Workflow generation will proceed without scientific grounding."
        )
        return None

    # ------------------------------------------------------------------ #
    # Parse the response.  The /api/chat endpoint returns SSE when stream=True
    # but since we set stream=False it returns a plain JSON object.
    # The web_app_full.py non-streaming branch returns:
    #   {"error": "Non-streaming not supported. Use stream=true"}
    # which means we must use streaming.  Handle both cases gracefully.
    # ------------------------------------------------------------------ #
    try:
        data = response.json()
    except Exception:
        logger.warning(
            "⚠️ [Perspicacite] Could not parse JSON response. "
            "Workflow generation will proceed without scientific grounding."
        )
        return None

    # If the server tells us streaming is required, fall back to streaming
    if isinstance(data, dict) and "error" in data:
        logger.info(
            "[Perspicacite] Non-streaming not supported by server; "
            "falling back to streaming mode."
        )
        return _query_perspicacite_streaming(
            science_query, mode, base_url, timeout
        )

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


def _query_perspicacite_streaming(
    science_query: str,
    mode: str,
    base_url: str,
    timeout: int,
) -> Optional[str]:
    """Internal helper that queries Perspicacite using streaming SSE.

    This is called as a fallback when the non-streaming path is unsupported.
    It consumes the SSE stream and extracts the final answer from the
    ``"answer"`` event type.
    """
    import base64
    import json

    try:
        import httpx
    except ImportError:
        return None

    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "query": science_query,
        "mode": mode,
        "stream": True,
        "kb_name": None,
        "max_papers": 5,
        "databases": ["semantic_scholar", "openalex", "pubmed"],
    }

    full_answer = ""
    try:
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
                    if event_type == "answer":
                        # The answer may be base64-encoded
                        content_b64 = event.get("content_b64")
                        if content_b64:
                            full_answer = base64.b64decode(content_b64).decode(
                                "utf-8", errors="replace"
                            )
                        elif "content" in event:
                            full_answer = str(event["content"])
                    elif event_type == "done":
                        break
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
    separator = "=" * 72
    return (
        f"{separator}\n"
        f"# SCIENTIFIC LITERATURE CONTEXT  (retrieved from Perspicacite-AI)\n"
        f"{separator}\n"
        f"The following scientific knowledge was retrieved for the task:\n"
        f"\"{task[:200].strip()}\"\n\n"
        f"{scientific_context.strip()}\n"
        f"{separator}\n\n"
    )

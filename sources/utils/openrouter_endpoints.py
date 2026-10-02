"""Discover which providers actually serve a given OpenRouter model.

Avoids the 'probe every known provider, 404 most of them' anti-pattern by
calling OpenRouter's per-model endpoints API up front.

API:  GET https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints
"""
from __future__ import annotations

import logging
import os

import requests

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Higher rank = higher precision. Used to pick the best endpoint when a
# provider exposes multiple quantizations, and to sort the final probe list.
_QUANT_RANK: dict[str, int] = {
    "fp32": 5,
    "bf16": 4, "fp16": 4,
    "fp8": 3,
    "int8": 2,
    "fp6": 1,
    "fp4": 0, "int4": 0,
    "unknown": -1,
}

# Routing slugs known to be the model creator's own serving endpoint, not a
# community reseller. Sorted ahead of community providers within a tier:
# even when they expose `quantization: "unknown"` (which ranks -1), they
# should win over an fp8 community endpoint because they serve the reference
# weights at intended precision.
FIRST_PARTY_SLUGS: frozenset[str] = frozenset({
    "anthropic", "openai", "google-vertex", "google-ai-studio",
    "xai", "deepseek", "mistral", "cohere", "moonshotai",
    "z-ai", "alibaba", "minimax", "perplexity", "sakana"
})

# Mapping from model-slug author (the segment before '/') to the OpenRouter
# provider slugs that are the official first-party serving endpoint for that
# author's models.  When a provider IS the model creator, its ``unknown``
# quantization tag is trusted as full-precision reference weights and should
# not be penalised.  Community resellers with ``unknown`` quant are demoted
# below fp8 because ``unknown`` usually means a low quantization.
_MODEL_AUTHOR_PROVIDERS: dict[str, frozenset[str]] = {
    "anthropic":   frozenset({"anthropic"}),
    "cohere":      frozenset({"cohere"}),
    "deepseek":    frozenset({"deepseek"}),
    "google":      frozenset({"google-vertex", "google-ai-studio"}),
    "meta-llama":  frozenset(),          # Meta doesn't self-host on OpenRouter
    "minimax":     frozenset({"minimax"}),
    "mistralai":   frozenset({"mistral"}),
    "moonshotai":  frozenset({"moonshotai"}),
    "openai":      frozenset({"openai"}),
    "perplexity":  frozenset({"perplexity"}),
    "qwen":        frozenset({"alibaba"}),
    "x-ai":        frozenset({"xai"}),
    "sakana":        frozenset({"sakana"})
}

logger = logging.getLogger(__name__)


def quant_rank(q: str) -> int:
    return _QUANT_RANK.get((q or "unknown").lower(), -1)


def is_first_party(slug: str) -> bool:
    return (slug or "").lower() in FIRST_PARTY_SLUGS


def is_model_creator(slug: str, model_slug: str) -> bool:
    """True when *slug* is the first-party creator / official host of *model_slug*.

    ``model_slug`` is the bare OpenRouter model id, e.g.
    ``deepseek/deepseek-v3.2`` — the author is the first ``/``-segment.

    Community resellers (e.g. ``alibaba`` serving a DeepSeek model) return
    False even though they appear in :data:`FIRST_PARTY_SLUGS`, because they
    are only first-party for their *own* models.
    """
    author = model_slug.split("/", 1)[0].lower() if model_slug else ""
    provider = (slug or "").lower()
    known = _MODEL_AUTHOR_PROVIDERS.get(author)
    if known is not None:
        return provider in known
    # Fallback for authors not yet in the map: exact slug match + must be a
    # recognised first-party slug so we don't accidentally promote randoms.
    return provider == author and provider in FIRST_PARTY_SLUGS


def fetch_endpoints(model_id: str, timeout: int = 30) -> list[dict]:
    """Raw endpoint records for `model_id` (e.g. 'deepseek/deepseek-v3.2').
    Returns [] on any error so callers can fall back to their configured list.
    """
    key = os.getenv("OPENROUTER_API_KEY", "")
    url = f"{OPENROUTER_BASE}/models/{model_id}/endpoints"
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        logger.warning("OpenRouter endpoint discovery network error for %s: %s", model_id, e)
        return []
    if r.status_code != 200:
        logger.warning(
            "OpenRouter endpoint discovery failed for %s: %s %s",
            model_id, r.status_code, r.text[:200],
        )
        return []
    try:
        body = r.json()
    except ValueError:
        return []
    data = body.get("data") or {}
    return data.get("endpoints", []) or []


def providers_for_model(model_id: str) -> dict[str, str]:
    """Map provider_name -> highest-precision quantization for `model_id`.

    Example: {'parasail': 'fp8', 'friendli': 'bf16', 'novita': 'fp8'}.
    Returns {} on discovery failure.
    """
    out: dict[str, str] = {}
    for ep in fetch_endpoints(model_id):
        # Skip endpoints OpenRouter has flagged as not currently serving.
        # Observed values: 0 = active, -2 = deprecated/down. Probing a
        # negative-status endpoint reliably returns 404 or rate-limits, so
        # it just burns a probe slot.
        if (ep.get("status") or 0) < 0:
            continue
        # `tag` looks like "baidu/fp8" or just "friendli"; the prefix before
        # the slash is the lowercase routing slug used by extra_body.provider.
        # `provider_name` is display-cased ("Baidu") and NOT usable for routing.
        tag = ep.get("tag") or ""
        slug = tag.split("/", 1)[0] if tag else ""
        if not slug:
            # Last-resort: derive a slug from the display name.
            pn = ep.get("provider_name") or ""
            slug = pn.lower().replace(" ", "-")
        if not slug:
            continue
        quant = (ep.get("quantization") or "unknown").lower()
        existing = out.get(slug)
        if existing is None or quant_rank(quant) > quant_rank(existing):
            out[slug] = quant
    return out


if __name__ == "__main__":
    import json as _json
    import sys

    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))

    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek/deepseek-v3.2"
    print(f"Discovering OpenRouter endpoints for: {model}\n")

    raw = fetch_endpoints(model)
    print(f"{len(raw)} raw endpoint record(s).")
    if raw:
        print("\nFirst record (shape sanity-check):")
        print(_json.dumps(raw[0], indent=2)[:1200])

    agg = providers_for_model(model)
    print(f"\nAggregated providers ({len(agg)}):")
    for p, q in sorted(agg.items(), key=lambda kv: (-quant_rank(kv[1]), kv[0])):
        print(f"  {p:<20} {q}")

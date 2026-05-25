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

logger = logging.getLogger(__name__)


def quant_rank(q: str) -> int:
    return _QUANT_RANK.get((q or "unknown").lower(), -1)


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
        # OpenRouter field names have varied historically; try the likely ones.
        name = ep.get("provider_name") or ep.get("provider") or ep.get("name") or ""
        quant = (ep.get("quantization") or "unknown").lower()
        if not name:
            continue
        existing = out.get(name)
        if existing is None or quant_rank(quant) > quant_rank(existing):
            out[name] = quant
    return out

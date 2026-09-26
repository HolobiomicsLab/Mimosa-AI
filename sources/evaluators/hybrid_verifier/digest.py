"""Deterministic format digests of deliverable files (pre-policy stage).

ONE LLM call per task reads deterministic sampled sections of the
workspace's deliverable-ish files (seeded head/middle/tail slices, ≤
``max_files`` files, bounded bytes per slice) and writes a SHORT format
digest per file — column names, log line grammar, value formats. The
digest is cached in the per-task registry and injected into every
policy-writer prompt, so the code-writing LLM sees the real file FORMATS,
not just their names.

Sampling is fully deterministic: slice offsets derive from the file size
and a per-path seed (sha256 of the relative path) — no global RNG state,
identical slices on every machine and re-run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sources.evaluators.base import extract_json_payload

from . import inventory as inv_mod

# Sampling budgets (bounded bytes per slice: head / middle / tail).
SLICE_BYTES = 1200
MAX_FILES_DEFAULT = 8


def sample_sections(path: Path, slice_bytes: int = SLICE_BYTES) -> str:
    """Deterministic head/middle/tail text slices of one file.

    Offsets depend only on the file size, so the same file always yields
    the same sample (missing/unreadable files yield an empty string).
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            head = fh.read(slice_bytes)
            mid_off = max(0, (size - slice_bytes) // 2)
            fh.seek(mid_off)
            mid = fh.read(slice_bytes)
            fh.seek(max(0, size - slice_bytes))
            tail = fh.read(slice_bytes)
    except OSError:
        return ""
    decode = lambda b: b.decode("utf-8", errors="replace")  # noqa: E731
    parts = [
        f"--- head (offset 0) ---\n{decode(head)}",
        f"--- middle (offset {mid_off}) ---\n{decode(mid)}",
        f"--- tail (offset {max(0, size - slice_bytes)}) ---\n{decode(tail)}",
    ]
    return "\n".join(parts)


def digest_prompt(
    samples: dict[str, str],
    signals: dict[str, str],
) -> str:
    """Build the one-shot format-digest prompt for a task's files."""
    blocks = []
    for rel, body in samples.items():
        sig = f" ({signals[rel]})" if signals.get(rel) else ""
        blocks.append(f"### {rel}{sig}\n{body}")
    return f"""You write SHORT FORMAT DIGESTS of files so a code-writing model can parse \
these files correctly. Below are deterministic head/middle/tail samples of this task's \
deliverable-ish files. For EACH file output a digest of its FORMAT ONLY:

- CSV/TSV: exact column names in order, delimiter, quoting, row grammar \
(one row per molecule?), value formats (float precision, ints, strings).
- Logs: the line grammar (timestamp/epoch/loss/accuracy fields, separators), \
where the FINAL and BEST metrics appear, which line to regex LAST.
- JSON/YAML/other: top-level keys/schema, value types.

Rules: describe only what the samples show; never invent fields; keep each \
digest <= 40 words. Output strict JSON only:
{{"digests": [{{"path": "<file>", "digest": "<format digest>"}}]}}\n
Sampled sections:
{chr(10).join(blocks)}"""


def parse_digests(raw: str) -> dict[str, str]:
    """Parse the digest JSON response into a path -> digest map."""
    import json

    payload = extract_json_payload(raw or "")
    if not payload:
        return {}
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict) or not isinstance(obj.get("digests"), list):
        return {}
    out: dict[str, str] = {}
    for d in obj["digests"]:
        if not isinstance(d, dict):
            continue
        p = str(d.get("path") or "").strip()
        text = str(d.get("digest") or "").strip()
        if p and text:
            out[p] = text[:400]
    return out


def render_digests(digests: dict[str, str]) -> str:
    """Render the cached digests block for policy-writer prompts."""
    if not digests:
        return "(no format digests cached for this task)"
    lines = [f"- {p}: {d}" for p, d in sorted(digests.items())]
    return "\n".join(lines)


def ensure_digests(
    workspace: Path,
    inventory: dict[str, dict[str, Any]],
    cached: dict[str, str],
    llm_text: Callable[[str, str], str],
    uuid: str,
    max_files: int = MAX_FILES_DEFAULT,
    logger: logging.Logger | None = None,
) -> dict[str, str]:
    """Return the task's format digests, computing them once if empty.

    Args:
        workspace: Workspace root being scored.
        inventory: Scanned inventory of that workspace.
        cached: Digests already persisted in the registry; non-empty
            caches are returned unchanged (one LLM call per task).
        llm_text: ``(agent_name, prompt) -> raw text`` judge call.
        uuid: Workflow identifier (memory scoping for the LLM call).
        max_files: Cap on sampled files (``hybrid_verifier_digest_max_files``).
        logger: Optional logger for soft-failure warnings.

    Returns:
        ``{rel_path: digest}``; empty when the call fails (soft-fail —
        policy prompts then fall back to the union inventory alone).
    """
    if cached:
        return cached
    cands = sorted(
        (
            (rel, info)
            for rel, info in inventory.items()
            if inv_mod._deliverable_rank(rel, info) < 9
        ),
        key=lambda kv: (
            inv_mod._deliverable_rank(kv[0], kv[1]),
            -kv[1].get("bytes", 0),
        ),
    )[: max(1, max_files)]
    samples: dict[str, str] = {}
    signals: dict[str, str] = {}
    for rel, info in cands:
        body = sample_sections(workspace / rel)
        if body:
            samples[rel] = body
            if info.get("signal"):
                signals[rel] = info["signal"]
    if not samples:
        return {}
    try:
        raw = llm_text("hybrid_format_digest", digest_prompt(samples, signals))
    except Exception as e:  # noqa: BLE001 — digest is an accelerator, not a dependency
        if logger:
            logger.warning(f"[{uuid}] format digest call failed: {e}")
        return {}
    digests = parse_digests(raw)
    if not digests and logger:
        logger.warning(
            f"[{uuid}] format digest response unparsable; continuing without"
        )
    return digests

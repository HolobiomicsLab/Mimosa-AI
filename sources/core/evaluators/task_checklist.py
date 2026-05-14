"""Per-task verification checklist, fixed once before evolution begins.

Generated from the task spec + literature grounding, *before* any candidate
workflow runs. The verifier then derives its claims from this checklist
rather than from the workflow's self-narration — closing the loop where
the workflow authors its own exam (a workflow that prints
``"loaded ClinTox dataset"`` ends up evaluated against the existence of
that exact phrase, which is satisfied by any agent that knows to print it).

Items describe *kinds* of artifact and *spaces* of acceptable variation,
not concrete file names; the verifier-generation stage maps each item onto
whatever the workspace actually contains.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from sources.core.llm_provider import LLMConfig, LLMProvider

from .grounding import get_perspicacite_grounding

_CHECKLIST_VERSION = 1
_LOG = logging.getLogger(__name__)

_SYSTEM = """You are designing a verification checklist for a scientific \
computing task. A separate system will later run automated checks against \
each item. You see the task spec and peer-reviewed literature grounding — \
you do NOT see any candidate solution."""

_PROMPT_TEMPLATE = """TASK SPECIFICATION:
{task}

LITERATURE GROUNDING (peer-reviewed evidence; may be absent):
{grounding}

Produce a typed checklist of properties any correct solution to this task
MUST satisfy. Each item must be checkable against on-disk artifacts the
workflow produces (files in the workspace) or against instrumented
execution behavior. Do NOT include items checkable only against the
workflow's self-report ("the agent claims it loaded data" is forbidden;
"a non-empty predictions table exists with one probability column per
class" is fine).

Where legitimate variation exists, describe the SPACE of acceptable
artifacts rather than hard-coding a single shape. Example:
  GOOD: "produces a predictions table with one probability column per
         class label found in the source CSV"
  BAD:  "produces predictions.csv with columns prob_0, prob_1"

Return STRICT JSON only:
{{
  "items": [
    {{
      "id": "<short_slug>",
      "description": "<one sentence — what must be true>",
      "criticality": "hard" | "soft",
      "checkable_via": "file" | "execution" | "both",
      "expected_artifact_kind": "<short noun phrase: 'predictions_table', 'trained_model_weights', ...>",
      "acceptable_variation": "<one sentence — what counts as a legitimate alternative>"
    }},
    ...
  ]
}}

Prefer 5–12 items. Mark as "hard" only items the literature treats as
load-bearing for this task; "soft" for supporting context. The checklist
is FIXED for the duration of evolution — do not include items whose
correctness depends on the candidate solution's implementation choices.
"""


def task_hash(task: str) -> str:
    """Stable 16-char hex hash of the task text — used as cache key."""
    return hashlib.sha256((task or "").strip().encode("utf-8")).hexdigest()[:16]


def _extract_json(text: str) -> str:
    if not text:
        return ""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return ""


def _parse_items(raw: str) -> list[dict[str, Any]]:
    payload = _extract_json(raw)
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    items = data.get("items", []) if isinstance(data, dict) else []
    cleaned: list[dict[str, Any]] = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict) or "description" not in it:
            continue
        cleaned.append({
            "id": str(it.get("id") or f"chk_{idx}"),
            "description": str(it["description"]).strip(),
            "criticality": "hard" if it.get("criticality") == "hard" else "soft",
            "checkable_via": str(it.get("checkable_via") or "file"),
            "expected_artifact_kind": str(it.get("expected_artifact_kind") or "").strip(),
            "acceptable_variation": str(it.get("acceptable_variation") or "").strip(),
        })
    return cleaned


class TaskChecklistBuilder:
    """Builds (and caches on disk) a per-task verification checklist."""

    def __init__(
        self,
        cache_dir: Path | str,
        llm_config: LLMConfig,
        memory_dir: Path | str,
        use_grounding: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.llm_config = llm_config
        self.memory_dir = Path(memory_dir)
        self.use_grounding = use_grounding

    def build(self, task: str) -> dict[str, Any]:
        """Return ``{"task_hash", "task_text", "version", "items"[, "error"]}``.

        Cached on disk under ``{cache_dir}/task_checklist_{hash}.json`` so
        repeat runs of the same task reuse the same rubric (the checklist
        being task-locked is the whole point — regenerating it per run
        would re-open the loop the checklist is meant to close).
        """
        task = (task or "").strip()
        if not task:
            return {
                "task_hash": "",
                "task_text": "",
                "version": _CHECKLIST_VERSION,
                "items": [],
            }
        h = task_hash(task)
        cache_path = self.cache_dir / f"task_checklist_{h}.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                if isinstance(cached, dict) and cached.get("items"):
                    _LOG.info(f"task checklist cache hit: {cache_path}")
                    return cached
            except (json.JSONDecodeError, OSError) as e:
                _LOG.warning(f"could not load cached checklist {cache_path}: {e}")

        grounding = ""
        if self.use_grounding:
            try:
                grounding = get_perspicacite_grounding(task)
            except Exception as e:
                _LOG.warning(f"grounding lookup failed for task checklist: {e}")
                grounding = ""

        memory_path = self.memory_dir / "_task_checklists"
        memory_path.mkdir(parents=True, exist_ok=True)
        provider = LLMProvider(
            agent_name=f"task_checklist_{h}",
            memory_path=memory_path,
            system_msg=_SYSTEM,
            config=self.llm_config,
        )
        prompt = _PROMPT_TEMPLATE.format(
            task=task,
            grounding=(grounding or "(unavailable)").strip(),
        )
        try:
            raw = provider(prompt) or ""
        except Exception as e:
            _LOG.error(f"task checklist LLM call failed: {e}")
            return {
                "task_hash": h,
                "task_text": task,
                "version": _CHECKLIST_VERSION,
                "items": [],
                "error": f"{type(e).__name__}: {e}",
            }

        items = _parse_items(raw)
        record = {
            "task_hash": h,
            "task_text": task,
            "version": _CHECKLIST_VERSION,
            "items": items,
        }
        if not items:
            record["error"] = "could not parse any items from LLM response"
            _LOG.warning(f"task checklist parse returned no items for {h}")

        try:
            with open(cache_path, "w") as f:
                json.dump(record, f, indent=2)
        except OSError as e:
            _LOG.warning(f"could not persist checklist {cache_path}: {e}")
        return record

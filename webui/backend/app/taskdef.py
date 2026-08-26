"""The task a run was asked to do, joined back from the run dir outward.

Four layers, each honest about absence (a missing layer carries a reason,
never a silent null):

* verbatim prompt — the newest ``original_task_*.txt`` in the run dir, full
  text: this is what the agent actually saw;
* task ref — ``task_ref.json`` when csv_mode stamped one, else the structured
  ``[ASB …]``/``(ASB …)`` tag regex-parsed out of the prompt. Parsed
  structurally (tag shape, not corpus vocabulary), and never guessed;
* grounding — the ``grounding`` block of ``run_metrics.json``, verbatim;
* card — the ASB card ``<corpus>/<challenge>/cards/<task_id>.json``, verbatim
  (schema-driven display is the frontend's job), when ``MIMOSA_CORPUS_DIR``
  is set and a task ref resolved.

Everything follows store.py's defensive contract: partial or missing
artifacts yield ``None`` plus a reason, never an exception.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .settings import get_settings
from .store import run_path

# The structured task tag benchmark prompts carry, in its on-disk variants:
#   [ASB benchmark, challenge p_iimn, task_001]
#   [ASB benchmark p_iimn, task_001]
#   (ASB Metabolomics challenge q_haffner, task_001)
#   [ASB Metabolomics — challenge: q_haffner, task: task_001]
# Structural, not vocabulary-bound: a bracketed/parenthesised span opening
# with "ASB", an optional descriptor, an optional "challenge" keyword, the
# challenge token, an optional "task" keyword, and a task_<id> token.
_TASK_TAG = re.compile(
    r"[\[(]ASB[\w ]*?[,\s—–-]+(?:challenge[:\s]+)?([A-Za-z0-9_-]+)"
    r"[,\s]+(?:task[:\s]+)?(task_[A-Za-z0-9_-]+)[\])]"
)

# Challenge/task-id values are used as path components under the corpus root;
# reject anything that could escape it (separators, leading dots).
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _read_json(path: Path) -> Any | None:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _newest_prompt_file(run_dir: Path) -> Path | None:
    """The newest ``original_task_*.txt`` (filenames embed the timestamp)."""
    files = sorted(run_dir.glob("original_task_*.txt"))
    return files[-1] if files else None


def parse_task_tag(text: str) -> dict[str, str] | None:
    """``{challenge, task_id}`` from the first structured ASB tag, or None."""
    match = _TASK_TAG.search(text)
    if not match:
        return None
    return {"challenge": match.group(1), "task_id": match.group(2)}


def _task_ref(run_dir: Path, prompt: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """(task ref, absent-reason). ``task_ref.json`` wins over the prompt tag."""
    ref_path = run_dir / "task_ref.json"
    stamped_but_unreadable = False
    if ref_path.is_file():
        data = _read_json(ref_path)
        if isinstance(data, dict):
            return {
                "challenge": data.get("challenge") or None,
                "task_id": data.get("task_id") or None,
                "csv_row": data.get("csv_row"),
                "source": "task_ref.json",
            }, None
        stamped_but_unreadable = True
    if prompt:
        parsed = parse_task_tag(prompt)
        if parsed:
            return {**parsed, "csv_row": None, "source": "prompt_tag"}, None
    prefix = "task_ref.json unreadable; " if stamped_but_unreadable else ""
    if prompt is None:
        return None, prefix + "no readable original_task_*.txt in the run dir"
    return None, prefix + "no structured ASB tag in the task prompt"


def _grounding(run_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    """(grounding block of run_metrics.json verbatim, absent-reason)."""
    metrics = _read_json(run_dir / "run_metrics.json")
    if not isinstance(metrics, dict):
        return None, "run_metrics.json missing or unreadable"
    grounding = metrics.get("grounding")
    if not isinstance(grounding, dict):
        return None, "no grounding block recorded in run_metrics.json"
    return grounding, None


def _card(task_ref: dict[str, Any] | None) -> tuple[Any, str | None]:
    """(the ASB card verbatim, absent-reason). Never projects fields away."""
    corpus = get_settings().corpus_dir
    if corpus is None:
        return None, "MIMOSA_CORPUS_DIR is not set"
    if not corpus.is_dir():
        return None, "corpus dir does not exist"
    if task_ref is None:
        return None, "no task ref to join on"
    challenge, task_id = task_ref.get("challenge"), task_ref.get("task_id")
    if not challenge or not task_id:
        return None, "task ref lacks a challenge or task id"
    if not (_SAFE_COMPONENT.match(str(challenge))
            and _SAFE_COMPONENT.match(str(task_id))):
        return None, "task ref contains unsafe path components"
    rel = f"{challenge}/cards/{task_id}.json"
    card_path = corpus / challenge / "cards" / f"{task_id}.json"
    if not card_path.is_file():
        return None, f"card not found in corpus: {rel}"
    card = _read_json(card_path)
    if card is None:
        return None, f"card unreadable: {rel}"
    return card, None


def task_view(run_id: str) -> dict[str, Any]:
    """Everything the Task panel shows for one run (see module docstring)."""
    run_dir = run_path(run_id)
    prompt_path = _newest_prompt_file(run_dir)
    prompt = _read_text(prompt_path) if prompt_path else None
    task_ref, ref_reason = _task_ref(run_dir, prompt)
    grounding, grounding_reason = _grounding(run_dir)
    card, card_reason = _card(task_ref)
    return {
        "run_id": run_id,
        "prompt": prompt,
        "prompt_file": prompt_path.name if prompt_path else None,
        "prompt_absent_reason": (
            None if prompt is not None
            else "prompt file unreadable" if prompt_path
            else "no original_task_*.txt in the run dir"),
        "task_ref": task_ref,
        "task_ref_absent_reason": ref_reason,
        "grounding": grounding,
        "grounding_absent_reason": grounding_reason,
        "card": card,
        "card_absent_reason": card_reason,
    }


__all__ = ["task_view", "parse_task_tag"]

"""Complete artifact inventory and explicit, bounded inspection for the policy LLM."""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import bounded_listing, decode_object, inventory, read_page, safe_path
from .criteria_guard import (
    GuardExhausted,
    guard_plan,
    input_files,
    input_heads,
    shown_heads,
)
from .placeholders import input_profile, one_value_columns, scan_tables
from .prompts import RECON, evidence_prompt

MAX_INSPECTION_ROUNDS = 16
MAX_REQUESTS_PER_ROUND = 16
# Per-generation listing caps; small tasks keep them.
INITIAL_PREVIEW_BYTES = 65536
MAX_LISTED_ENTRIES = 400
# Prompt-wide totals, shared by all generations.
PREVIEW_BUDGET_BYTES = 3 * INITIAL_PREVIEW_BYTES
LISTED_ENTRIES_BUDGET = 3 * MAX_LISTED_ENTRIES
# Floors that keep every late generation visible.
MIN_PREVIEW_BYTES = 4096
MIN_LISTED_ENTRIES = 40
MAX_RETAINED_PAGES = 48
# Who decides a criterion: adapter code or the vision tie-break.
JUDGE_CODE = "code"
JUDGE_VISION = "vision"
# Provenance keys the model sees; the pristine data path stays internal.
PROVENANCE_SHOWN = ("status", "tampered")


class ReconError(ValueError):
    """The instrument could not establish a complete, valid schema plan.

    Raised by reconnaissance and its helpers on malformed model answers,
    missing criteria, or invalid visual selections. The evaluator's error
    boundary turns it into ``verifier_instrument_error``, never a score.
    """


def describe_generation(generation: dict, share: int) -> dict:
    """Expose inventory rows while bounding preview bytes and row count.

    ``_open_transcript`` calls it once per generation of one recon prompt.
    The prompt-wide budgets are split over ``share`` generations, so the
    prompt stays bounded when the session has many generations. Rows are
    chosen shallow paths first (``bounded_listing``); omitted rows are
    counted per directory, and a directory page lists their names.

    Args: generation (dict): UUID, owned snapshot path, execution facts and
        the optional ``input_provenance`` of its task inputs; share (int):
        number of generations described in the same prompt.
    Returns: dict with metadata, explicitly omitted previews, omitted-row
        counts, and the provenance status and changed input paths.
    """
    entries = inventory(Path(generation["snapshot"]))
    share = max(1, share)
    limit = max(MIN_LISTED_ENTRIES, min(MAX_LISTED_ENTRIES, LISTED_ENTRIES_BUDGET // share))
    listed, unlisted = bounded_listing(entries, limit)
    _bound_previews(listed, max(MIN_PREVIEW_BYTES, min(INITIAL_PREVIEW_BYTES, PREVIEW_BUDGET_BYTES // share)))
    provenance = generation.get("input_provenance") or {}
    return {"uuid": generation["uuid"], "inventory": listed,
            "unlisted_entries": len(entries) - len(listed),
            "unlisted_directories": unlisted,
            "unlisted_note": "Request a page of a directory path to list its entries; unlisted entries still exist." if unlisted else "",
            "execution": generation.get("execution", {}),
            "input_provenance": {key: provenance[key] for key in PROVENANCE_SHOWN if key in provenance}}


def _bound_previews(entries: list[dict], budget: int) -> None:
    """Blank the previews that do not fit one generation's byte budget.

    Called by ``describe_generation`` on the listed rows, in listing order.

    Args: entries (list[dict]): Listed inventory rows, changed in place;
        budget (int): Preview bytes this generation may show.
    Returns: None.
    """
    for entry in entries:
        preview = entry.get("preview", "")
        if not isinstance(preview, str):
            continue
        size = len(preview.encode("utf-8"))
        if size > budget:
            entry["preview"] = ""
            entry["preview_omitted"] = "Initial preview byte budget; use inspect"
        else:
            budget -= size


def code_task(task: dict) -> dict:
    """Return the task without the criteria only the vision model judges.

    The policy engine calls it before it writes or runs adapter and
    comparison code, and the gradient fold calls it before it reports
    unmeasured values. A criterion only an image shows (a plotted element,
    legibility) has ``"judge": "vision"``: adapters cannot read pixels, so it
    goes only to the visual tie-break (Elk/Toronto run 2026-10-20).

    Args:
        task (dict): Frozen task or bundle with ``criteria`` and
            ``representation``.

    Returns:
        dict: A shallow copy whose criteria and representation hold only
        code-judged entries.
    """
    vision = {c["id"] for c in task.get("criteria") or [] if c.get("judge") == JUDGE_VISION}
    return {**task, "criteria": [c for c in task.get("criteria") or [] if c["id"] not in vision],
            "representation": {k: v for k, v in (task.get("representation") or {}).items() if k not in vision}}


def _check_criteria(criteria: object) -> None:
    """Reject criterion lists the instrument cannot compare.

    Called by ``_check_complete`` with the model's proposed criteria.

    Args:
        criteria (object): Proposed ``criteria`` value from the plan.

    Returns:
        None; raises ReconError for an invalid list.
    """
    if not isinstance(criteria, list) or not criteria:
        raise ReconError("At least one goal-grounded criterion is required")
    ids = [item.get("id") for item in criteria if isinstance(item, dict)]
    if len(ids) != len(criteria) or any(not isinstance(key, str) or not key for key in ids):
        raise ReconError("Criterion IDs must be nonempty strings")
    if len(set(ids)) != len(ids):
        raise ReconError("Criterion IDs must be unique")
    for item in criteria:
        if type(item.get("stage")) is not int or item["stage"] < 1:
            raise ReconError("Criterion stages must be positive integers")
        if item.get("direction") not in {"higher", "lower", "satisfy"}:
            raise ReconError("Criterion direction must be higher, lower or satisfy")
        if item.get("judge", JUDGE_CODE) not in {JUDGE_CODE, JUDGE_VISION}:
            raise ReconError('Criterion judge must be "code" or "vision"')
    if all(item.get("judge") == JUDGE_VISION for item in criteria):
        raise ReconError('At least one criterion must be judged by "code" (for example output presence)')


def _check_complete(result: dict, generations: list[dict]) -> None:
    """Reject missing generation coverage and invalid criterion definitions.

    Args: result (dict): model plan; generations (list[dict]): supplied records.
    Returns: None; raises ReconError for an incomplete response.
    """
    _check_criteria(result.get("criteria"))
    if not isinstance(result.get("task_representation"), str):
        raise ReconError("Task representation must be a string")
    if not isinstance(result.get("representation"), dict):
        raise ReconError("Normalized representation must be an object")
    for generation in generations:
        uuid = generation["uuid"]
        if not isinstance(result.get("plans", {}).get(uuid), dict):
            raise ReconError(f"No adapter plan for {uuid}")
        paths = result.get("visuals", {}).get(uuid, [])
        if not isinstance(paths, list):
            raise ReconError(f"Visual paths must be a list for {uuid}")
        for path in paths:
            if not safe_path(Path(generation["snapshot"]), path).is_file():
                raise ReconError(f"Selected visual artifact is missing: {uuid}/{path}")


def reconnaissance(goal: str, generations: list[dict], context: dict) -> dict:
    """Let the model inspect files before producing a schema plan.

    A malformed or incomplete answer is returned to the model as feedback and
    costs one round; it never ends the evaluation by itself.

    Args: goal (str): user goal; generations (list[dict]): owned snapshots;
        context (dict): injected ask callback and optional frozen task/adapters.
    Returns: dict with criteria, schema plans, figures and inspection transcript.
    """
    evidence, context = _open_transcript(goal, generations, context)
    for round_index in range(MAX_INSPECTION_ROUNDS):
        evidence["rounds_left"] = MAX_INSPECTION_ROUNDS - 1 - round_index
        try:
            response = decode_object(context["ask"](evidence_prompt(RECON, evidence)))
            if response.get("action") == "complete":
                return _accept_plan(response, generations, context, evidence)
            _inspect(response, generations, evidence)
            evidence.pop("previous_answer_error", None)
        except GuardExhausted as error:
            raise ReconError(str(error)) from error
        except ValueError as error:
            evidence["previous_answer_error"] = f"{type(error).__name__}: {error}. Answer again with one valid JSON object."
    raise ReconError("Inspection budget exhausted; no partial criteria were accepted")


def _open_transcript(goal: str, generations: list[dict], context: dict) -> tuple[dict, dict]:
    """Build the first recon transcript and the context the guard needs.

    Called once by ``reconnaissance``. For an unfrozen task the inputs are
    found once (``_input_source``); their heads are read (complete for the
    guard, bounded for the model prompt), and every input table is read
    completely for two facts: ``input_profile`` and ``one_value_columns``
    (hidden labels).

    Args:
        goal (str): User goal.
        generations (list[dict]): Owned snapshots, oldest first.
        context (dict): Injected ask callback and optional frozen task/adapters.

    Returns:
        tuple[dict, dict]: The transcript and the extended context.
    """
    evidence = {"goal": goal, "generations": [describe_generation(g, len(generations)) for g in generations],
                "frozen_task": context.get("frozen_task"),
                "adapters": context.get("adapters", {}), "inspections": []}
    if context.get("frozen_task"):
        return evidence, context
    source = _input_source(generations)
    heads = input_heads(goal, source)
    tables = scan_tables(Path(source["snapshot"]), input_files(source), goal)
    placeholders = one_value_columns(tables, goal)
    evidence.update(input_heads=shown_heads(heads), input_profile=input_profile(tables, goal), one_value_columns=placeholders)
    return evidence, {**context, "generation": source, "input_heads": heads, "one_value_columns": placeholders}


def _input_source(generations: list[dict]) -> dict:
    """Choose where the task inputs are read, for any task layout.

    The evaluator keeps the workspace as it was before the first run (its
    path is ``input_provenance`` ``data``). Every file there is an input,
    and no run could rewrite it (HIV run_oct_18: the snapshot's test labels
    were replaced by real ones). Without that copy (standalone verifier,
    older registries) the first snapshot is used, and ``input_files`` keeps
    only the files its run did not produce.

    Args:
        generations (list[dict]): Generation records, oldest first.

    Returns:
        dict: A record with ``snapshot`` for ``input_heads`` and
        ``input_files``: the baseline (``is_baseline`` True) or the first
        generation record with its ``execution``.
    """
    copies = [(g.get("input_provenance") or {}).get("data") for g in generations]
    baseline = next((path for path in copies if path and Path(path).is_dir()), None)
    return {"snapshot": baseline, "execution": {}, "is_baseline": True} if baseline else generations[0]


def _accept_plan(response: dict, generations: list[dict], context: dict, evidence: dict) -> dict:
    """Validate a complete plan and attach the inspection transcript.

    When a task is already frozen, its criteria are restored verbatim from the
    registry: the model only contributes adapter plans and visual selections,
    so a paraphrase can never redefine success or fail the evaluation. New
    criteria first pass ``criteria_guard.guard_plan``.

    Args: response (dict): Model "complete" answer; generations (list[dict]);
        context (dict): Holds the optional frozen task; evidence (dict): Transcript.
    Returns: dict: The accepted plan; raises ReconError when it is invalid.
    """
    frozen = context.get("frozen_task")
    response.update(frozen or {})
    _check_complete(response, generations)
    if not frozen:
        guard_plan(response, context, evidence)
        # A guard drop may leave only vision criteria.
        _check_criteria(response["criteria"])
    evidence.pop("previous_answer_error", None)
    response["inspection"] = evidence
    return response


def _inspect(response: dict, generations: list[dict], evidence: dict) -> None:
    """Serve one round of byte-page requests into the bounded transcript.

    Args: response (dict): Model "inspect" answer; generations (list[dict]):
        Owned snapshots; evidence (dict): Transcript updated in place.
    Returns: None; raises ReconError for an invalid request shape.
    """
    roots = {item["uuid"]: Path(item["snapshot"]) for item in generations}
    requests = response.get("requests")
    if response.get("action") != "inspect" or not isinstance(requests, list) or not 1 <= len(requests) <= MAX_REQUESTS_PER_ROUND:
        raise ReconError("Expected complete or a bounded nonempty inspect request")
    for request in requests:
        if not isinstance(request, dict) or request.get("uuid") not in roots:
            raise ReconError("Inspection requested an unknown generation")
        evidence["inspections"].append(read_page(roots[request["uuid"]], request))
    dropped = len(evidence["inspections"]) - MAX_RETAINED_PAGES
    if dropped > 0:
        evidence["inspections"] = evidence["inspections"][dropped:]
        evidence["inspection_note"] = "Oldest inspected pages were dropped; re-request a page if still needed."


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(describe_generation({"uuid": "preview", "snapshot": str(args.root)}, 1), indent=2))

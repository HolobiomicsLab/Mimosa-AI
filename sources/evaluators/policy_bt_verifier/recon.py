"""Complete artifact inventory and explicit, bounded inspection for the policy LLM."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from .artifacts import decode_object, inventory
from .criteria_guard import GuardExhausted, guard_plan, input_heads, shown_heads
from .placeholders import input_profile, one_value_columns, scan_tables, table_keys
from .prompts import RECON, evidence_prompt

PAGE_BYTES = 8192
MAX_INSPECTION_ROUNDS = 16
MAX_REQUESTS_PER_ROUND = 16
INITIAL_PREVIEW_BYTES = 65536
MAX_LISTED_ENTRIES = 400
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


def safe_path(root: Path, relative: str) -> Path:
    """Keep artifact reads inside a real directory without symlink traversal.

    Args: root (Path): owned artifact root; relative (str): untrusted file name.
    Returns: Path confined below root; raises on unsafe paths.
    """
    if not isinstance(relative, str) or not relative:
        raise ValueError("Artifact path must be a nonempty relative string")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Absolute and parent paths are forbidden")
    candidate = root.resolve()
    for part in path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("Symlink artifact access is forbidden")
    candidate.resolve().relative_to(root.resolve())
    return candidate


def inspect_page(root: Path, request: dict) -> dict:
    """Read an explicit byte page for reconnaissance.

    Args: root (Path): owned root; request (dict): path, offset and length.
    Returns: dict with bytes, format, offsets and remaining-byte count.
    """
    offset, length = request.get("offset", 0), request.get("length", PAGE_BYTES)
    if type(offset) is not int or type(length) is not int:
        raise ReconError("Inspection offset and length must be integers")
    if offset < 0 or not 1 <= length <= PAGE_BYTES:
        raise ReconError(f"Inspection length must be 1..{PAGE_BYTES}")
    path = safe_path(root, request["path"])
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read(length)
    try:
        content, encoding = data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        content, encoding = base64.b64encode(data).decode("ascii"), "base64"
    return {**request, "content": content, "encoding": encoding,
            "next_offset": offset + len(data),
            "remaining_bytes": max(0, path.stat().st_size - offset - len(data))}


def describe_generation(generation: dict) -> dict:
    """Expose inventory rows while bounding preview bytes and row count.

    Args: generation (dict): UUID, owned snapshot path, execution facts and
        the optional ``input_provenance`` of its task inputs.
    Returns: dict with metadata, explicitly omitted previews, an omitted-row
        count, and the provenance status and changed input paths.
    """
    entries = inventory(Path(generation["snapshot"]))
    remaining = INITIAL_PREVIEW_BYTES
    for entry in entries:
        preview = entry.get("preview", "")
        if not isinstance(preview, str):
            continue
        encoded = preview.encode("utf-8")
        if len(encoded) > remaining:
            entry["preview"] = ""
            entry["preview_omitted"] = "Initial preview byte budget; use inspect"
        else:
            remaining -= len(encoded)
    listed = entries[:MAX_LISTED_ENTRIES]
    provenance = generation.get("input_provenance") or {}
    return {"uuid": generation["uuid"], "inventory": listed,
            "unlisted_entries": len(entries) - len(listed),
            "unlisted_note": "Request a byte page of any path; unlisted entries still exist." if len(entries) > len(listed) else "",
            "execution": generation.get("execution", {}),
            "input_provenance": {key: provenance[key] for key in PROVENANCE_SHOWN if key in provenance}}


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

    Called once by ``reconnaissance``. For an unfrozen task the input heads
    are read once (complete for the guard, bounded for the model prompt),
    and every input table is read completely once for two facts:
    ``input_profile`` and ``one_value_columns`` (hidden labels).

    Args:
        goal (str): User goal.
        generations (list[dict]): Owned snapshots, oldest first.
        context (dict): Injected ask callback and optional frozen task/adapters.

    Returns:
        tuple[dict, dict]: The transcript and the extended context.
    """
    evidence = {"goal": goal, "generations": [describe_generation(g) for g in generations],
                "frozen_task": context.get("frozen_task"),
                "adapters": context.get("adapters", {}), "inspections": []}
    if context.get("frozen_task"):
        return evidence, context
    heads = input_heads(goal, generations[0])
    tables = _input_tables(generations, heads)
    placeholders = one_value_columns(tables, goal)
    evidence.update(input_heads=shown_heads(heads), input_profile=input_profile(tables, goal), one_value_columns=placeholders)
    return evidence, {**context, "generation": generations[0], "input_heads": heads, "one_value_columns": placeholders}


def _input_tables(generations: list[dict], heads: dict[str, str]) -> dict[str, dict]:
    """Read the task's input tables, from the pristine copy when one exists.

    A generation can rewrite an input before its snapshot (HIV run_oct_18:
    the hidden test labels were replaced by real ones), so the transfer-time
    copy the evaluator keeps in the registry comes first. Without it (no
    benchmark transfer, older records) the goal-named snapshot inputs are read.

    Args:
        generations (list[dict]): Generation records; ``input_provenance``
            ``data`` names the pristine copy.
        heads (dict[str, str]): Goal-named input files of the first snapshot.

    Returns:
        dict[str, dict]: Scanned tables from ``placeholders.scan_tables``.
    """
    copies = [(g.get("input_provenance") or {}).get("data") for g in generations]
    pristine = next((Path(path) for path in copies if path and Path(path).is_dir()), None)
    if pristine is not None:
        return scan_tables(pristine, table_keys(pristine))
    return scan_tables(Path(generations[0]["snapshot"]), list(heads))


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
    requests = response.get("requests", [])
    if response.get("action") != "inspect" or not 1 <= len(requests) <= MAX_REQUESTS_PER_ROUND:
        raise ReconError("Expected complete or a bounded nonempty inspect request")
    for request in requests:
        if request.get("uuid") not in roots:
            raise ReconError("Inspection requested an unknown generation")
        try:
            page = inspect_page(roots[request["uuid"]], request)
        except (OSError, ValueError, KeyError) as error:
            page = {**request, "error": str(error)}
        evidence["inspections"].append(page)
    dropped = len(evidence["inspections"]) - MAX_RETAINED_PAGES
    if dropped > 0:
        evidence["inspections"] = evidence["inspections"][dropped:]
        evidence["inspection_note"] = "Oldest inspected pages were dropped; re-request a page if still needed."


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(describe_generation({"uuid": "preview", "snapshot": str(args.root)}), indent=2))

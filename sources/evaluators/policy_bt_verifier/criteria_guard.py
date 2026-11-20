"""Freeze-time checks that keep verification criteria aligned with the goal.

Reconnaissance proposes the criteria a task is judged on, and they stay frozen
for the whole session. Two checks run before freezing: a deterministic
grounding check (every exact string a criterion requires must come from the
goal or an input file) and one LLM audit per plan. The audit keeps or rejects
each criterion against the goal, and, when an input column holds one
placeholder value (``one_value_columns``) that the goal asks to predict, it
names the held-out quality criterion the plan lacks. Findings go back to
reconnaissance for a revision; when no revision is left, rejected criteria
are dropped and a missing quality criterion is recorded as ``quality_gap``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .artifacts import decode_object
from .prompts import CRITERIA_GUARD, evidence_prompt

# Rejected plans sent back to recon before rejected criteria are dropped.
MAX_PLAN_REVISIONS = 3
# Attempts for one well-formed guard answer before failing open.
GUARD_ANSWER_ATTEMPTS = 2
MAX_FILE_HEADS = 80
FILE_HEAD_LINES = 2
# Grounding reads complete first lines up to this size (wide CSV headers).
FILE_HEAD_BYTES = 256 * 1024
# Head size shown to a model; grounding always uses the complete head.
SHOWN_HEAD_CHARS = 2000
TRUNCATION_NOTE = " …[head truncated for display; a cut head does not prove a name is absent]"
# Literals shorter than this must match a whole token.
SHORT_LITERAL_CHARS = 3
# Characters stripped around goal words before comparing file names.
WORD_PUNCTUATION = "'\"`()[]{},;:|*<>.?!"
# Agent-written source never grounds a name (its docstring could say anything).
SOURCE_SUFFIXES = {".py", ".sh", ".ipynb", ".r", ".jl", ".js"}
PACKAGING_PATHS = (".policy_bt_reexec", "requirements.txt")
PRUNED_DIRS = {"__pycache__", ".git", "node_modules", ".ipynb_checkpoints"}
VENV_MARKER = "pyvenv.cfg"


class GuardRevision(ValueError):
    """Reconnaissance must revise the plan (``rejected_criteria``, ``missing_criteria``).

    Raised by ``_request_revision``. The recon loop catches it like any
    malformed answer and returns it to the model as ``previous_answer_error``.
    """


class GuardExhausted(Exception):
    """Every criterion is still rejected after the last revision.

    Raised by ``_drop_criteria``. It is not a ValueError, so the recon loop
    stops at once instead of asking again with no revision left.
    """


def guard_plan(response: dict, context: dict, evidence: dict) -> None:
    """Ground and audit new criteria, asking for revisions or dropping.

    Called by reconnaissance for every complete plan of an unfrozen task. The
    code check runs first and, while it fails, the LLM is not called. Each
    revision request is recorded under ``plan_revisions``; when no revision
    is left (``MAX_PLAN_REVISIONS`` or the recon round budget), rejected
    criteria are dropped (``dropped_criteria``) and a missing quality
    criterion is recorded under ``quality_gap``; the plan is still accepted.

    Args:
        response (dict): Validated recon plan; criteria may be removed.
        context (dict): Holds ``ask``, ``generation`` and, from recon, the
            complete ``input_heads`` and ``one_value_columns``.
        evidence (dict): Recon transcript with ``goal`` and ``rounds_left``.

    Returns:
        None; raises GuardRevision to request a revised plan.
    """
    revisions = evidence.setdefault("plan_revisions", [])
    can_revise = len(revisions) < MAX_PLAN_REVISIONS and evidence.get("rounds_left", 1) > 0
    heads = context.get("input_heads")
    heads = input_heads(evidence["goal"], context["generation"]) if heads is None else heads
    findings = {"rejected": _unsourced_literals(response, evidence["goal"], heads)}
    if findings["rejected"] and not can_revise:
        _drop_criteria(response, findings["rejected"], evidence)
        findings["rejected"] = {}
    if not findings["rejected"]:
        findings["rejected"], findings["missing_quality_criterion"] = _guard_review(response, {**context, "input_heads": heads}, evidence)
    missing = findings.get("missing_quality_criterion")
    if (findings["rejected"] or missing) and can_revise:
        _request_revision(response, findings, evidence)
    if findings["rejected"]:
        _drop_criteria(response, findings["rejected"], evidence)
    if missing and missing not in evidence.setdefault("quality_gap", []):
        evidence["quality_gap"].append(missing)
    evidence.pop("rejected_criteria", None)
    evidence.pop("missing_criteria", None)


def _request_revision(response: dict, findings: dict, evidence: dict) -> None:
    """Record one revision request and send the plan back to reconnaissance.

    Called by ``guard_plan`` while a revision is left. The next recon prompt
    shows ``rejected_criteria`` (each with its reason) and ``missing_criteria``
    (the quality criterion to add). A round where only the grounding check
    ran has no guard judgment, so it keeps the previous ``missing_criteria``.

    Args:
        response (dict): Plan whose criteria were checked.
        findings (dict): ``rejected`` (criterion id to reason) and, when the
            guard judged the plan, ``missing_quality_criterion`` (str or None).
        evidence (dict): Recon transcript, updated in place.

    Returns:
        None; always raises GuardRevision.
    """
    evidence["plan_revisions"].append(findings)
    rejected = findings["rejected"]
    evidence["rejected_criteria"] = [{**c, "reason": rejected[c["id"]]} for c in response["criteria"] if c["id"] in rejected]
    if findings.get("missing_quality_criterion"):
        evidence["missing_criteria"] = [findings["missing_quality_criterion"]]
    elif "missing_quality_criterion" in findings:
        evidence.pop("missing_criteria", None)
    raise GuardRevision("The criteria review asks for a revised plan: revise or remove each of "
                        "rejected_criteria and add each of missing_criteria.")


def input_heads(goal: str, generation: dict) -> dict[str, str]:
    """Read the complete first lines of the task input files.

    Input headers decide which names an output may use. ``guard_plan`` calls
    it per plan and reconnaissance once to show the model what grounds a name.
    In the pre-run baseline (``is_baseline``) every file is an input, so a
    ``--task`` goal need not name its files. In a generation snapshot a file
    counts only when a goal word is its file name and the run did not produce
    it: the goal also names the output path, and an output's own header must
    never ground the names it is checked against. Goal-named files come first
    under the ``MAX_FILE_HEADS`` bound.

    Args:
        goal (str): Task goal text with its dataset structure.
        generation (dict): Record with a ``snapshot`` path, its reproduction
            ``execution`` record, and ``is_baseline`` for the pre-run copy.

    Returns:
        dict[str, str]: Relative path to its first lines (complete lines).
    """
    root, names = Path(generation["snapshot"]), _goal_file_names(goal)
    files = input_files(generation)
    if not generation.get("is_baseline"):
        files = [relative for relative in files if Path(relative).name in names]
    ranked = sorted(files, key=lambda relative: Path(relative).name not in names)
    heads = {relative: _read_head(root / relative) for relative in ranked[:MAX_FILE_HEADS]}
    return {key: value for key, value in heads.items() if value}


def input_files(generation: dict) -> list[str]:
    """List the files of a snapshot that can be task inputs.

    Reconnaissance calls it on the pre-run baseline, where every file is an
    input, or, without a baseline, on the first generation's snapshot, where
    a file counts only when the run did not produce it. Left out: contract
    outputs, files the rerun created or changed, the script, requirements,
    the verifier log folder, agent-written source (.py, .sh, ...) and links.
    Environments and caches are not walked.

    Args:
        generation (dict): Record with a ``snapshot`` path and an optional
            reproduction ``execution`` record.

    Returns:
        list[str]: Sorted snapshot-relative POSIX paths.
    """
    root, produced, found = Path(generation["snapshot"]), _produced_paths(generation), []
    for directory, subdirs, files in os.walk(root):
        # Never walk environments or caches (thousands of package files).
        subdirs[:] = sorted(d for d in subdirs if d not in PRUNED_DIRS and not (Path(directory) / d / VENV_MARKER).exists())
        relatives = ((Path(directory) / name).relative_to(root).as_posix() for name in files)
        found += [relative for relative in relatives if _is_input(root / relative, relative, produced)]
    return sorted(found)


def _is_input(path: Path, relative: str, produced: list[str]) -> bool:
    """Tell whether one walked file can be a task input.

    Called by ``input_files`` for each file.

    Args:
        path (Path): File inside the snapshot.
        relative (str): Its snapshot-relative POSIX path.
        produced (list[str]): Paths from ``_produced_paths``.

    Returns:
        bool: False for source code, links, and produced or packaging paths.
    """
    return path.suffix.lower() not in SOURCE_SUFFIXES and not path.is_symlink() and not _is_produced(relative, produced)


def _is_produced(relative: str, produced: list[str]) -> bool:
    """Tell whether a path is, or lies under, a produced or packaging path.

    Args:
        relative (str): Snapshot-relative POSIX path.
        produced (list[str]): Paths from ``_produced_paths``.

    Returns:
        bool: True when the file is not a task input.
    """
    return any(relative == out or relative.startswith(out.rstrip("/") + "/") for out in produced)


def shown_heads(heads: dict[str, str]) -> dict[str, str]:
    """Bound input heads for a model prompt, marking any cut head.

    Args:
        heads (dict[str, str]): Complete heads from ``input_heads``.

    Returns:
        dict[str, str]: Heads cut to ``SHOWN_HEAD_CHARS`` with a note.
    """
    return {key: value if len(value) <= SHOWN_HEAD_CHARS else value[:SHOWN_HEAD_CHARS] + TRUNCATION_NOTE
            for key, value in heads.items()}


def _read_head(path: Path) -> str:
    """Read the first text lines of one file, or nothing for binary data.

    Args:
        path (Path): Regular file inside the snapshot.

    Returns:
        str: Up to ``FILE_HEAD_LINES`` lines, or ``""`` when unreadable or binary.
    """
    try:
        with path.open("rb") as stream:
            data = stream.read(FILE_HEAD_BYTES)
    except OSError:
        return ""
    if b"\x00" in data:
        return ""
    return "\n".join(data.decode("utf-8", errors="replace").splitlines()[:FILE_HEAD_LINES])


def _goal_file_names(goal: str) -> set[str]:
    """Collect the file names the goal mentions, word by word.

    Whole words keep "test.csv" from matching "train_test.csv".

    Args:
        goal (str): Task goal text.

    Returns:
        set[str]: Base names of the goal's words.
    """
    return {Path(word.strip(WORD_PUNCTUATION)).name for word in goal.split() if word.strip(WORD_PUNCTUATION)}


def _produced_paths(generation: dict) -> list[str]:
    """List paths a generation produced or packages, which are never inputs.

    Args:
        generation (dict): Record with the reexec ``execution`` record.

    Returns:
        list[str]: Contract outputs, files the rerun created or changed, the
        script, requirements and the verifier log directory.
    """
    execution = generation.get("execution") or {}
    contract, files = execution.get("contract") or {}, execution.get("files") or {}
    produced = [str(path) for path in contract.get("generated_outputs") or []]
    for key in ("created", "changed", "regenerated"):
        produced += [str(path) for path in files.get(key) or []]
    return produced + [str(contract.get("script") or "run.sh"), *PACKAGING_PATHS]


def _unsourced_literals(response: dict, goal: str, heads: dict[str, str]) -> dict[str, str]:
    """Find criteria whose exact required strings have no source.

    A literal is an exact string a criterion makes an output contain (a path,
    a column name, a label). It must appear in the goal or in an input head
    (live case Ames 2026-10-18: header "SMILES,Probability" while the input
    header is Drug,Y). No model call is needed.

    Args:
        response (dict): Recon plan with criteria and their ``literals``.
        goal (str): Task goal text.
        heads (dict[str, str]): Complete input heads.

    Returns:
        dict[str, str]: Criterion id to rejection reason; raises ValueError
        for a malformed ``literals`` field.
    """
    text = " ".join([goal, *heads.values()])
    sources, tokens = " ".join(text.split()), set(_tokens(text))
    rejected = {}
    for criterion in response["criteria"]:
        literals = criterion.setdefault("literals", [])
        if not isinstance(literals, list) or not all(isinstance(item, str) for item in literals):
            raise ValueError(f"Criterion {criterion['id']}: literals must be a list of strings")
        missing = [item for item in literals if not _is_sourced(item, sources, tokens)]
        if missing:
            rejected[criterion["id"]] = (f"literals {missing} appear neither in the goal nor in an input file's "
                                         "first lines; use the goal's or the input's exact names, or drop the requirement")
    return rejected


def _is_sourced(literal: str, sources: str, tokens: set[str]) -> bool:
    """Tell whether one literal appears in the grounding text.

    A short literal ("Y", "id") would match inside any word, so it must be a
    whole token; a longer one may match as a substring (paths, phrases).

    Args:
        literal (str): Exact string a criterion requires.
        sources (str): Goal plus input heads, whitespace collapsed.
        tokens (set[str]): Alphanumeric tokens of the same text.

    Returns:
        bool: True when the literal is grounded.
    """
    compact = " ".join(literal.split())
    if len(compact) < SHORT_LITERAL_CHARS:
        return compact in tokens
    return compact in sources


def _tokens(text: str) -> list[str]:
    """Split text into alphanumeric tokens (underscore kept), without regex.

    Args:
        text (str): Goal or file head text.

    Returns:
        list[str]: Tokens in order.
    """
    return "".join(char if char.isalnum() or char == "_" else " " for char in text).split()


def _guard_review(response: dict, context: dict, evidence: dict) -> tuple[dict[str, str], str | None]:
    """Ask the text model to audit each criterion and the plan's quality coverage.

    Called by ``guard_plan`` once the grounding check passes. A malformed
    answer goes back to the guard once; a second one fails open (no
    rejection, no gap), because reconnaissance is not at fault. Every verdict
    list is kept under ``criteria_guard`` in the transcript, followed by the
    guard's missing quality criterion when it names one. That criterion
    counts only when the code found ``one_value_columns``: without a
    placeholder column there is no gap.

    Args:
        response (dict): Plan with criteria, representation and adapter plans.
        context (dict): Holds the ``ask`` text-model callback, the complete
            ``input_heads`` (shown bounded) and the ``one_value_columns``.
        evidence (dict): Recon transcript holding the goal.

    Returns:
        tuple[dict[str, str], str | None]: Rejected criterion id to the
        guard's reason, and the missing quality criterion or None.
    """
    placeholders = context.get("one_value_columns") or {}
    payload = {**_plan_view(response), "goal": evidence["goal"], "input_heads": shown_heads(context["input_heads"]),
               "one_value_columns": placeholders}
    record = evidence.setdefault("criteria_guard", [])
    for _ in range(GUARD_ANSWER_ATTEMPTS):
        try:
            answer = decode_object(context["ask"](evidence_prompt(CRITERIA_GUARD, payload)))
            verdicts, missing = _valid_verdicts(answer, response), _missing_quality(answer)
        except ValueError as error:
            payload["previous_answer_error"] = f"{type(error).__name__}: {error}"
            continue
        record.append(verdicts + ([{"missing_quality_criterion": missing}] if missing else []))
        rejected = {item["id"]: item["reason"] for item in verdicts if item["verdict"] == "reject"}
        return rejected, missing if placeholders else None
    record.append([{"error": payload["previous_answer_error"]}])
    return {}, None


def _plan_view(response: dict) -> dict:
    """Select the plan fields the guard audits.

    Called by ``_guard_review`` to build the guard payload.

    Args:
        response (dict): Recon plan.

    Returns:
        dict: Task representation, criteria, value representation and the
        adapter plan instructions per generation.
    """
    return {"task_representation": response["task_representation"], "criteria": response["criteria"],
            "representation": response["representation"],
            "plans": {uuid: plan.get("instructions") for uuid, plan in response["plans"].items() if isinstance(plan, dict)}}


def _valid_verdicts(answer: dict, response: dict) -> list[dict]:
    """Check that a guard answer gives one valid verdict per criterion.

    Args:
        answer (dict): Decoded guard response.
        response (dict): Plan whose criteria were audited.

    Returns:
        list[dict]: Verdicts ordered like the criteria; raises ValueError.
    """
    verdicts = {item.get("id"): item for item in answer.get("verdicts") or [] if isinstance(item, dict)}
    ordered = [verdicts.get(criterion["id"]) for criterion in response["criteria"]]
    if any(item is None or item.get("verdict") not in {"keep", "reject"} for item in ordered):
        raise ValueError("Give exactly one keep or reject verdict per criterion id")
    if any(item["verdict"] == "reject" and not str(item.get("reason") or "").strip() for item in ordered):
        raise ValueError("A reject verdict needs a reason that reconnaissance can act on")
    return [{"id": item["id"], "verdict": item["verdict"], "reason": str(item.get("reason", ""))} for item in ordered]


def _missing_quality(answer: dict) -> str | None:
    """Read the guard's missing quality criterion, if it names one.

    Called by ``_guard_review``; a ValueError sends the answer back once.

    Args:
        answer (dict): Decoded guard response.

    Returns:
        str | None: One sentence describing the criterion to add, or None
        for null, absent or blank; raises ValueError for another type.
    """
    value = answer.get("missing_quality_criterion")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("missing_quality_criterion must be null or one sentence")
    return value.strip() or None


def _drop_criteria(response: dict, rejected: dict[str, str], evidence: dict) -> None:
    """Remove criteria still rejected after the last revision.

    Args:
        response (dict): Plan updated in place (criteria and representation).
        rejected (dict[str, str]): Rejected criterion id to its reason.
        evidence (dict): Transcript; the drop is kept under ``dropped_criteria``.

    Returns:
        None; raises GuardExhausted when no criterion would remain.
    """
    evidence.setdefault("dropped_criteria", []).append(rejected)
    kept = [criterion for criterion in response["criteria"] if criterion["id"] not in rejected]
    if not kept:
        raise GuardExhausted("Every proposed criterion was still rejected after the last revision")
    response["criteria"] = kept
    for key in rejected:
        response["representation"].pop(key, None)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "test.csv").write_text(",Drug_ID,Drug,Y\n0,D1,CCO,0\n")
        (Path(folder) / "preds.csv").write_text("SMILES,Probability\n")
        found = input_heads('Use test.csv; save "preds.csv".', {"snapshot": folder, "execution": {"contract": {"generated_outputs": ["preds.csv"]}}})
        plan = {"criteria": [{"id": "c", "literals": ["Drug", "Probability"]}]}
        print(json.dumps({"heads": found, "rejected": _unsourced_literals(plan, "Use test.csv", found)}, indent=1))

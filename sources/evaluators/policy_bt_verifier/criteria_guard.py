"""Freeze-time checks that keep verification criteria aligned with the goal.

Reconnaissance proposes the criteria a task is judged on, and they stay frozen
for the whole session. Two checks run before freezing: a deterministic
grounding check (every exact string a criterion requires must come from the
goal or an input file) and one LLM audit per plan (keep or reject each
criterion against the goal). Rejections go back to reconnaissance for a
revision; when no revision is left, rejected criteria are dropped.
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
    """Reconnaissance must revise the criteria named in ``rejected_criteria``.

    Raised by ``guard_plan``. The recon loop catches it like any malformed
    answer and returns it to the model as ``previous_answer_error``.
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
    rejected plan is recorded under ``plan_revisions``; when no revision is
    left (``MAX_PLAN_REVISIONS`` or the recon round budget), rejected criteria
    are dropped and recorded under ``dropped_criteria``.

    Args:
        response (dict): Validated recon plan; criteria may be removed.
        context (dict): Holds ``ask``, ``generation`` and optional complete
            ``input_heads``.
        evidence (dict): Recon transcript with ``goal`` and ``rounds_left``.

    Returns:
        None; raises GuardRevision to request a revised plan.
    """
    revisions = evidence.setdefault("plan_revisions", [])
    can_revise = len(revisions) < MAX_PLAN_REVISIONS and evidence.get("rounds_left", 1) > 0
    heads = context.get("input_heads")
    heads = input_heads(evidence["goal"], context["generation"]) if heads is None else heads
    rejected = _unsourced_literals(response, evidence["goal"], heads)
    if rejected and not can_revise:
        _drop_criteria(response, rejected, evidence)
        rejected = {}
    rejected = rejected or _guard_rejections(response, context, evidence, heads)
    if rejected and can_revise:
        revisions.append(rejected)
        evidence["rejected_criteria"] = [{**c, "reason": rejected[c["id"]]} for c in response["criteria"] if c["id"] in rejected]
        raise GuardRevision("Criteria rejected for misalignment with the goal (see rejected_criteria): revise or remove them.")
    if rejected:
        _drop_criteria(response, rejected, evidence)
    evidence.pop("rejected_criteria", None)


def input_heads(goal: str, generation: dict) -> dict[str, str]:
    """Read the complete first lines of each task input file the goal names.

    Input headers decide which names an output may use. ``guard_plan`` calls
    it per plan and reconnaissance once to show the model what grounds a name.
    A file is an input when a goal word is its file name and it is not one of
    the generation's outputs, run logs or packaging files: the goal also names
    the output path, and an output's own header must never ground the names it
    is checked against.

    Args:
        goal (str): Task goal text with its dataset structure.
        generation (dict): Generation record with an owned ``snapshot`` path
            and its reproduction ``execution`` record.

    Returns:
        dict[str, str]: Relative path to its first lines (complete lines).
    """
    root, names, produced, heads = Path(generation["snapshot"]), _goal_file_names(goal), _produced_paths(generation), {}
    for directory, subdirs, files in os.walk(root):
        # Never walk environments or caches (thousands of package files).
        subdirs[:] = sorted(d for d in subdirs if d not in PRUNED_DIRS and not (Path(directory) / d / VENV_MARKER).exists())
        for name in sorted(set(files) & names):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if len(heads) < MAX_FILE_HEADS and path.suffix.lower() not in SOURCE_SUFFIXES and not path.is_symlink() and not _is_produced(relative, produced):
                heads[relative] = _read_head(path)
    return {key: value for key, value in heads.items() if value}


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


def _guard_rejections(response: dict, context: dict, evidence: dict, heads: dict[str, str]) -> dict[str, str]:
    """Ask the text model to keep or reject each criterion against the goal.

    A malformed answer goes back to the guard once; a second one fails open
    (no rejection), because reconnaissance is not at fault. Every outcome is
    kept under ``criteria_guard`` in the transcript.

    Args:
        response (dict): Plan with criteria, representation and adapter plans.
        context (dict): Holds the ``ask`` text-model callback.
        evidence (dict): Recon transcript holding the goal.
        heads (dict[str, str]): Complete input heads (shown bounded).

    Returns:
        dict[str, str]: Rejected criterion id to the guard's reason.
    """
    payload = {"goal": evidence["goal"], "input_heads": shown_heads(heads),
               "task_representation": response["task_representation"],
               "criteria": response["criteria"], "representation": response["representation"],
               "plans": {uuid: plan.get("instructions") for uuid, plan in response["plans"].items() if isinstance(plan, dict)}}
    record = evidence.setdefault("criteria_guard", [])
    for _ in range(GUARD_ANSWER_ATTEMPTS):
        try:
            verdicts = _valid_verdicts(decode_object(context["ask"](evidence_prompt(CRITERIA_GUARD, payload))), response)
        except ValueError as error:
            payload["previous_answer_error"] = f"{type(error).__name__}: {error}"
            continue
        record.append(verdicts)
        return {item["id"]: item["reason"] for item in verdicts if item["verdict"] == "reject"}
    record.append([{"error": payload["previous_answer_error"]}])
    return {}


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

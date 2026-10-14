"""Read the Policy-BT verifier verdict of one run for the Verification tab.

The verifier (``sources/evaluators/policy_bt_verifier/results.py``) writes
``evaluation.txt`` as the header line ``Policy-BT verifier`` followed by a JSON
object: reward at arrival, tournament version, the reproduction/execution
record, and every pair verdict that involves this generation. Pair verdicts
name their sides ``a``/``b``; this module relabels them ``self``/``opponent``
from the run's point of view, joins the live ratings that the verifier
refreshes in ``state_result.json``, and the frozen comparison criteria from the
task session's ``<workflow_dir>/.policy_bt/<hash>/tournament.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPORT_HEADER = "Policy-BT verifier"
REGISTRY_DIRNAME = ".policy_bt"

# Scalar fields of the execution record shown as-is.
_EXECUTION_FIELDS = (
    "eligible",
    "status",
    "reason",
    "enabled",
    "attempted",
    "script",
    "returncode",
    "timed_out",
    "duration_s",
    "preserved_outputs",
    "symlink_gaps",
)
_PROCESS_FIELDS = ("argv", "returncode", "timed_out", "error", "duration_s")


def _load_json(path: Path) -> Any | None:
    """Load a JSON file and ignore missing or broken files.

    The run-detail reader calls this for state and registry files.
    It gives back None when the file is absent or is not valid JSON.

    Args:
        path (Path): JSON file inside the workflow directory.

    Returns:
        Any | None: The decoded value, or None.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _read_report(path: Path) -> dict[str, Any] | None:
    """Decode a Policy-BT ``evaluation.txt`` into its JSON object.

    ``read_policy_bt_report`` calls this with the run's evaluation file.
    Files of other verifiers (other header line) give back None.

    Args:
        path (Path): The run's ``evaluation.txt``.

    Returns:
        dict | None: The report object, or None for other formats.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    header, _, body = text.partition("\n")
    if header.strip() != REPORT_HEADER:
        return None
    try:
        report = json.loads(body)
    except json.JSONDecodeError:
        return None
    return report if isinstance(report, dict) else None


def _live_verifier(run_dir: Path) -> dict[str, Any]:
    """Get the verifier block that holds the current tournament ratings.

    The verifier rewrites ``state_result.json`` of every generation after each
    evaluation, so it holds fresher ratings than ``evaluation.txt``.

    Args:
        run_dir (Path): One generation folder under the workflow directory.

    Returns:
        dict: ``evaluation.verifier`` for a policy_bt run, else an empty dict.
    """
    state = _load_json(run_dir / "state_result.json")
    evaluation = state.get("evaluation") if isinstance(state, dict) else None
    verifier = evaluation.get("verifier") if isinstance(evaluation, dict) else None
    if not isinstance(verifier, dict) or verifier.get("verifier_kind") != "policy_bt":
        return {}
    return verifier


def _process_view(record: Any) -> dict[str, Any] | None:
    """Keep the readable parts of one provision or run subprocess record.

    Args:
        record (Any): ``execution.provision`` or ``execution.run`` (dict or None).

    Returns:
        dict | None: Exit facts plus stdout/stderr tails, or None.
    """
    if not isinstance(record, dict):
        return None
    view = {key: record.get(key) for key in _PROCESS_FIELDS}
    for stream in ("stdout", "stderr"):
        captured = record.get(stream)
        view[stream] = captured.get("tail") if isinstance(captured, dict) else None
    return view


def _files_view(files: Any) -> dict[str, Any] | None:
    """Summarize the workspace file diff of the reproduction rerun.

    The ``unchanged`` list can hold thousands of paths; only its size is kept.

    Args:
        files (Any): ``execution.files`` (dict of path lists, or None).

    Returns:
        dict | None: Path lists without ``unchanged`` plus ``unchanged_count``.
    """
    if not isinstance(files, dict):
        return None
    view = {key: value for key, value in files.items() if key != "unchanged"}
    view["unchanged_count"] = len(files.get("unchanged") or [])
    return view


def summarize_execution(execution: dict[str, Any]) -> dict[str, Any]:
    """Build the reproduction-contract view of one execution record.

    Args:
        execution (dict): ``execution`` object of the Policy-BT report.

    Returns:
        dict: Gate verdict, contract, file diff, provision and run records.
    """
    view = {key: execution.get(key) for key in _EXECUTION_FIELDS}
    view["contract"] = execution.get("contract")
    view["files"] = _files_view(execution.get("files"))
    view["provision"] = _process_view(execution.get("provision"))
    view["run"] = _process_view(execution.get("run"))
    return view


def _relabel(entry: dict[str, Any], own: str) -> dict[str, Any]:
    """Replace an evidence side ``a``/``b`` with ``self``/``opponent``.

    Args:
        entry (dict): One pair evidence entry from the verifier.
        own (str): The side letter of the viewed run, ``a`` or ``b``.

    Returns:
        dict: A copy of the entry with the relabeled ``side``.
    """
    side = "self" if entry.get("side") == own else "opponent"
    return {**entry, "side": side}


def _pair_view(
    pair: dict[str, Any], own: str, ratings: dict[str, Any]
) -> dict[str, Any]:
    """Express one pair verdict from the viewed run's point of view.

    Execution-gate evidence repeats the full execution records; it is dropped
    because both feedback strings already carry the gate reasons.

    Args:
        pair (dict): One verdict of ``report["pairs"]``.
        own (str): The side letter of the viewed run, ``a`` or ``b``.
        ratings (dict): Current rating per generation uuid.

    Returns:
        dict: Opponent, outcome, stage, channel, both feedbacks, evidence, values.
    """
    other = "b" if own == "a" else "a"
    winner = pair.get("winner")
    outcome = "tie" if winner == "tie" else "win" if winner == own else "loss"
    evidence = [
        _relabel(item, own)
        for item in pair.get("evidence") or []
        if isinstance(item, dict) and "execution" not in item
    ]
    normalized = (pair.get(f"normalized_{own}"), pair.get(f"normalized_{other}"))
    values = None
    if all(isinstance(side, dict) for side in normalized):
        values = {
            "self": normalized[0].get("values"),
            "opponent": normalized[1].get("values"),
        }
    return {
        "opponent": pair.get(other),
        "opponent_reward": ratings.get(pair.get(other)),
        "outcome": outcome,
        "stage": pair.get("stage"),
        "channel": pair.get("channel"),
        "policy_version": pair.get("policy_version"),
        "feedback": pair.get(f"feedback_{own}"),
        "opponent_feedback": pair.get(f"feedback_{other}"),
        "evidence": evidence,
        "values": values,
    }


def _session_criteria(run_dir: Path) -> list[dict[str, Any]] | None:
    """Find the frozen comparison criteria of the run's task session.

    Each session registry keeps a ``snapshots/<uuid>`` copy per generation;
    that folder identifies the session without reading every registry.

    Args:
        run_dir (Path): One generation folder under the workflow directory.

    Returns:
        list | None: Criteria sorted by stage, or None before a policy exists.
    """
    registry_root = run_dir.parent / REGISTRY_DIRNAME
    for snapshot in sorted(registry_root.glob(f"*/snapshots/{run_dir.name}")):
        state = _load_json(snapshot.parent.parent / "tournament.json")
        bundle = state.get("bundle") if isinstance(state, dict) else None
        criteria = bundle.get("criteria") if isinstance(bundle, dict) else None
        if isinstance(criteria, list):
            return sorted(
                (c for c in criteria if isinstance(c, dict)),
                key=lambda c: c.get("stage", 0),
            )
    return None


def read_policy_bt_report(run_dir: Path) -> dict[str, Any] | None:
    """Build the Verification tab payload for one generation.

    ``store.run_detail`` calls this with a run folder of the workflow directory.
    It reads ``evaluation.txt``, the live ratings and the session criteria.

    Args:
        run_dir (Path): One generation folder; its name is the generation uuid.

    Returns:
        dict | None: Reward, execution gate, pairs and criteria, or None when
        the run has no Policy-BT report.
    """
    report = _read_report(run_dir / "evaluation.txt")
    if report is None:
        return None
    uuid = run_dir.name
    live = _live_verifier(run_dir)
    ratings = live.get("tournament_rewards") or {}
    pairs = [
        _pair_view(pair, "a" if pair.get("a") == uuid else "b", ratings)
        for pair in report.get("pairs") or []
        if isinstance(pair, dict) and uuid in (pair.get("a"), pair.get("b"))
    ]
    return {
        "reward": live.get("reward", report.get("reward")),
        "arrival_reward": live.get("arrival_reward"),
        "tournament_version": live.get(
            "tournament_version", report.get("tournament_version")
        ),
        "ratings": ratings,
        "execution": summarize_execution(report.get("execution") or {}),
        "pairs": pairs,
        "criteria": _session_criteria(run_dir),
    }


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    print(json.dumps(read_policy_bt_report(target), indent=2)[:4000])

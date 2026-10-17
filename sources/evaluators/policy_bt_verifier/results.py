"""Publish current tournament ratings and evidence-grounded mutation feedback."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from .registry import atomic_json

NO_PREDECESSOR_GRADIENT = (
    "No predecessor comparison is available. Preserve reproducibility and satisfy "
    "the task's requested outputs; no quality advantage has been established."
)
ALL_TIED_GRADIENT = (
    "Every comparison with other runs tied, and no deficiency was measured or "
    "observed in your outputs."
)
MAX_FIGURE_DEFICIENCIES = 6
FIGURE_REVIEW_HEADER = "Figure review by the vision model (observations, not measurements):"
GRADIENT_FILE = "textual_gradient.txt"
GRADIENT_HISTORY_FILE = "textual_gradient_history.jsonl"


def generation_pairs(state: dict, uuid: str) -> list[dict]:
    """Find every current comparison involving a generation.

    Args:
        state: Committed or pending registry state.
        uuid: Generation identifier from the workflow engine.

    Returns:
        Pair records with the generation on either side.
    """
    return [pair for pair in state["pairs"] if uuid in (pair["a"], pair["b"])]


def fold_gradient(generation: dict, pairs: list[dict], criteria: list[dict] | None = None) -> str:
    """Fold every current pair of one generation into mutation feedback.

    The next mutation reads this text as the parent's diagnosis, so it is
    refolded each time the tournament changes. ``build_result`` calls it
    with the generation record, all its current pairs and the frozen
    criteria. Decisive pairs give their feedback, losses first and earliest
    stage first. Then come the absolute deficiencies measured on this run
    (failed gates, values that rivals provide but this run does not), except
    criteria at a stage already lost decisively, and the figure deficiencies
    the vision model observed. Tied pairs never add "you match" text.

    Args:
        generation (dict): Registry snapshot record with ``uuid`` and the
            ``execution`` outcome from reexec.
        pairs (list[dict]): Current pair verdicts involving it, from
            ``generation_pairs``.
        criteria (list[dict] | None): Frozen bundle criteria with ``id``,
            ``stage``, ``description`` and ``direction``; ``None`` before
            freezing.

    Returns:
        str: Mutation feedback without tournament scores or policy source.
    """
    if not generation["execution"]["eligible"]:
        return "Reproducibility failure: " + generation["execution"]["reason"]
    uuid = generation["uuid"]
    decisive = sorted((p for p in pairs if _outcome(p, uuid) != "tie"),
                      key=lambda pair: (_outcome(pair, uuid) != "loss", pair["stage"]))
    lost = {pair["stage"] for pair in decisive if _outcome(pair, uuid) == "loss"}
    open_criteria = [c for c in criteria or [] if isinstance(c, dict) and c.get("stage") not in lost]
    lines = [pair["feedback_" + _side(pair, uuid)] for pair in decisive]
    lines += _criterion_deficiencies(uuid, pairs, open_criteria) + _figure_deficiencies(uuid, pairs)
    if lines:
        return "\n".join(dict.fromkeys(lines))
    return ALL_TIED_GRADIENT if pairs else NO_PREDECESSOR_GRADIENT


def _side(pair: dict, uuid: str) -> str:
    """Name the side of a pair that one generation occupies.

    The gradient fold reads side-labelled fields (feedback, normalized
    values, visual deficiencies). The fold helpers call it with a pair
    verdict and a uuid.

    Args:
        pair (dict): Pair verdict with ``a`` and ``b`` uuids.
        uuid (str): Generation on one side of the pair.

    Returns:
        str: ``"a"`` or ``"b"``.
    """
    return "a" if pair["a"] == uuid else "b"


def _criterion_deficiencies(uuid: str, pairs: list[dict], criteria: list[dict]) -> list[str]:
    """State the absolute criterion deficiencies measured on one run.

    Tied pairs say nothing about what is still wrong, so the fold reads the
    adapter values directly. ``fold_gradient`` calls it with the run's pairs
    and the open criteria. A failed ``satisfy`` gate is a deficiency. A
    null value is one only when some rival run provides that value, because
    null alone can be a verifier limit (for example hidden test labels).

    Args:
        uuid (str): Generation whose values are read.
        pairs (list[dict]): Its current pair verdicts; code pairs carry
            ``normalized_a``/``normalized_b`` adapter outputs.
        criteria (list[dict]): Frozen criteria not lost decisively.

    Returns:
        list[str]: One feedback line per deficiency, in criterion order.
    """
    other = {"a": "b", "b": "a"}
    sides = [(pair.get("normalized_" + _side(pair, uuid)), pair.get("normalized_" + other[_side(pair, uuid)]))
             for pair in pairs]
    measured = [(own, rival) for own, rival in sides if isinstance(own, dict)]
    if not measured:
        return []
    rivals = [rival.get("values") or {} for _, rival in measured if isinstance(rival, dict)]
    lines = [_deficiency(criterion, measured[0][0], rivals) for criterion in criteria]
    return [line for line in lines if line]


def _deficiency(criterion: dict, normalized: dict, rivals: list[dict]) -> str:
    """Describe one criterion deficiency, or nothing when none exists.

    ``_criterion_deficiencies`` calls it once per open criterion.

    Args:
        criterion (dict): Frozen criterion with ``id``, ``description``,
            ``direction``.
        normalized (dict): This run's adapter output: ``values`` by
            criterion id and ``evidence`` entries with the observed
            ``detail``.
        rivals (list[dict]): Adapter values of the runs it was compared to.

    Returns:
        str: A feedback line, or ``""``.
    """
    key = criterion["id"]
    value = (normalized.get("values") or {}).get(key)
    if criterion.get("direction") == "satisfy" and value is False:
        details = [e.get("detail") for e in normalized.get("evidence") or [] if e.get("criterion") == key and e.get("detail")]
        return f"Your run does not satisfy '{key}': " + (details[0] if details else criterion.get("description", ""))
    if value is None and any(rival.get(key) is not None for rival in rivals):
        return f"'{key}' ({criterion.get('description', '')}) could not be measured from your outputs, while another run's outputs provide it."
    return ""


def _figure_deficiencies(uuid: str, pairs: list[dict]) -> list[str]:
    """Collect the absolute figure deficiencies the vision model observed.

    The visual channel records, per side, which goal-named elements each
    figure lacks, even when the pair ties. ``fold_gradient`` calls it with
    the run's pairs. The lines are labelled as model observations so the
    mutator does not take them for measurements.

    Args:
        uuid (str): Generation whose figures were reviewed.
        pairs (list[dict]): Current pair verdicts; visual reviews sit in
            ``visual_deficiencies`` as ``{"a": [...], "b": [...]}``.

    Returns:
        list[str]: A header plus distinct deficiency lines (capped), or [].
    """
    found = []
    for pair in pairs:
        found += (pair.get("visual_deficiencies") or {}).get(_side(pair, uuid)) or []
    distinct = list(dict.fromkeys(item.strip() for item in found if isinstance(item, str) and item.strip()))
    if not distinct:
        return []
    return [FIGURE_REVIEW_HEADER] + [f"- {item}" for item in distinct[:MAX_FIGURE_DEFICIENCIES]]


def _outcome(pair: dict, uuid: str) -> str:
    """Translate side vocabulary into a generation's outcome.

    Args:
        pair: Validated pair verdict.
        uuid: UUID of one of its participants.

    Returns:
        win, loss, or tie.
    """
    if pair["winner"] == "tie":
        return "tie"
    return "win" if pair[pair["winner"]] == uuid else "loss"


def build_result(state: dict, generation: dict) -> dict:
    """Build the public engine result from a complete tournament state.

    Args:
        state: Registry with current ratings and pair evidence.
        generation: Participant record for the result being published.

    Returns:
        JSON-compatible evaluation.verifier payload.
    """
    uuid = generation["uuid"]
    pairs = generation_pairs(state, uuid)
    outcomes = {pair["b"] if pair["a"] == uuid else pair["a"]: _outcome(pair, uuid) for pair in pairs}
    return {"verifier_kind": "policy_bt", "reward": state["ratings"][uuid],
            "overall_score": state["ratings"][uuid], "reward_fallback": False,
            "arrival_reward": generation["arrival_reward"], "n_pairs": len(pairs),
            "pair_outcomes": outcomes, "tournament_rewards": state["ratings"],
            "tournament_outcomes": _tournament_outcomes(state),
            "tournament_version": state["version"], "execution": generation["execution"],
            "abstracted_textual_gradient": fold_gradient(generation, pairs, (state.get("bundle") or {}).get("criteria"))}


def _tournament_outcomes(state: dict) -> dict[str, dict[str, str]]:
    """Tabulate every generation's current outcome against every rival.

    The engine keeps old run records in memory; it copies this table into
    each record with the ratings, so capsule tie-breaks read current pair
    outcomes, not the ones seen at arrival. ``build_result`` calls it with
    the committed tournament state.

    Args:
        state (dict): Registry with ``generations`` and current ``pairs``.

    Returns:
        dict[str, dict[str, str]]: uuid → rival uuid → win/loss/tie.
    """
    table = {item["uuid"]: {} for item in state["generations"]}
    for pair in state["pairs"]:
        table[pair["a"]][pair["b"]] = _outcome(pair, pair["a"])
        table[pair["b"]][pair["a"]] = _outcome(pair, pair["b"])
    return table


def publish_results(root: Path, state: dict, current: str) -> dict:
    """Refresh selectable rewards and every generation's current gradient.

    Args:
        root: Workflow artifact root owned by the evolution engine.
        state: Atomically committed tournament state.
        current: UUID whose new evaluation and gradient are being emitted.

    Returns:
        Current generation's public result.
    """
    current_result = None
    for generation in state["generations"]:
        uuid = generation["uuid"]
        folder = root / uuid
        result = build_result(state, generation)
        _publish_state(folder, result, uuid == current)
        if uuid == current:
            _publish_evidence(folder, state, generation)
            current_result = result
    if current_result is None:
        raise ValueError("Current generation absent from tournament")
    return current_result


def _publish_state(folder: Path, result: dict, is_current: bool) -> None:
    """Update one generation's live rating and refolded gradient.

    Parent selection reads the gradient from ``state_result.json``, so a
    parent always serves the fold of all its current pairs. The first
    gradient stays as ``arrival_textual_gradient``. A torn historical file is
    left untouched: rewriting it would erase that generation's other
    evidence, and the registry still holds its rating.

    Args:
        folder: One generation's engine artifact directory.
        result: Current public result for that generation.
        is_current: Whether this is the newly evaluated generation.

    Returns:
        None.
    """
    path = folder / "state_result.json"
    try:
        state = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError) as error:
        if is_current:
            raise
        logging.getLogger(__name__).warning("Skipping unreadable %s: %s", path, error)
        return
    old = state.setdefault("evaluation", {}).get("verifier", {})
    gradient = result["abstracted_textual_gradient"]
    arrival = old.get("arrival_textual_gradient") or old.get("abstracted_textual_gradient") or gradient
    state["evaluation"]["verifier"] = {**result, "arrival_textual_gradient": arrival}
    atomic_json(path, state)
    if gradient != old.get("abstracted_textual_gradient"):
        _record_gradient(folder, gradient, result["tournament_version"])


def _record_gradient(folder: Path, gradient: str, version: int) -> None:
    """Write a changed gradient and append it to the gradient history.

    Auditors need the text a child's mutation received, which can differ
    from the parent's latest fold. ``_publish_state`` calls it when a fold
    changes. ``textual_gradient.txt`` always holds the current fold;
    ``textual_gradient_history.jsonl`` keeps every version with its time.

    Args:
        folder (Path): Generation artifact directory.
        gradient (str): The new folded gradient.
        version (int): Tournament version that produced it.

    Returns:
        None.
    """
    (folder / GRADIENT_FILE).write_text(gradient)
    entry = {"time": datetime.now().isoformat(timespec="seconds"), "tournament_version": version, "gradient": gradient}
    with (folder / GRADIENT_HISTORY_FILE).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _publish_evidence(folder: Path, state: dict, generation: dict) -> None:
    """Write the new generation's evaluation evidence (gradient: ``_publish_state``).

    Args:
        folder: Generation artifact directory created by the engine.
        state: Complete committed tournament state.
        generation: Current generation snapshot record.

    Returns:
        None.
    """
    result = build_result(state, generation)
    pairs = generation_pairs(state, generation["uuid"])
    report = {"reward": result["reward"], "tournament_version": state["version"],
              "execution": generation["execution"], "pairs": pairs}
    (folder / "evaluation.txt").write_text("Policy-BT verifier\n" + json.dumps(report, indent=2, ensure_ascii=False))
    atomic_json(folder / "policy_bt_evidence.json", report)


if __name__ == "__main__":
    print(fold_gradient({"uuid": "smoke", "execution": {"eligible": False, "reason": "Missing reproducibility .sh script."}}, []))
    tie = {"a": "x", "b": "y", "winner": "tie", "stage": 0, "feedback_a": "match", "feedback_b": "match",
           "normalized_a": {"values": {"ok": False, "auc": None}, "evidence": [{"criterion": "ok", "detail": "header is ['id']"}]},
           "normalized_b": {"values": {"ok": True, "auc": 0.8}, "evidence": []},
           "visual_deficiencies": {"a": ["no legend"], "b": []}}
    crit = [{"id": "ok", "description": "header", "direction": "satisfy"},
            {"id": "auc", "description": "AUC", "direction": "higher"}]
    folded = fold_gradient({"uuid": "x", "execution": {"eligible": True}}, [tie], crit)
    assert "match" not in folded and "header is ['id']" in folded and "- no legend" in folded, folded
    print(folded)

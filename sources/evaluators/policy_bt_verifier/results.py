"""Publish current tournament ratings and evidence-grounded mutation feedback."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .registry import atomic_json


def generation_pairs(state: dict, uuid: str) -> list[dict]:
    """Find every current comparison involving a generation.

    Args:
        state: Committed or pending registry state.
        uuid: Generation identifier from the workflow engine.

    Returns:
        Pair records with the generation on either side.
    """
    return [pair for pair in state["pairs"] if uuid in (pair["a"], pair["b"])]


def fold_gradient(generation: dict, pairs: list[dict]) -> str:
    """Collect measured pair diagnoses with earliest losses first.

    Args:
        generation: Snapshot record with execution outcome and UUID.
        pairs: Current pair evidence involving this generation.

    Returns:
        Mutation feedback without tournament scores or policy source.
    """
    if not generation["execution"]["eligible"]:
        return "Reproducibility failure: " + generation["execution"]["reason"]
    uuid = generation["uuid"]
    ordered = sorted(pairs, key=lambda pair: (_outcome(pair, uuid) != "loss", pair["stage"]))
    feedback = [pair["feedback_a" if pair["a"] == uuid else "feedback_b"] for pair in ordered]
    return "\n".join(dict.fromkeys(feedback)) or "No predecessor comparison is available. Preserve reproducibility and satisfy the task's requested outputs; no quality advantage has been established."


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
            "tournament_version": state["version"], "execution": generation["execution"],
            "abstracted_textual_gradient": fold_gradient(generation, pairs)}


def publish_results(root: Path, state: dict, current: str) -> dict:
    """Refresh selectable rewards while preserving historical feedback files.

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
    """Update live ratings without rewriting consumed historical gradients.

    A torn historical file is left untouched: rewriting it would erase that
    generation's other evidence, and the registry still holds its rating.

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
    if not is_current and old.get("abstracted_textual_gradient"):
        result = {**result, "abstracted_textual_gradient": old["abstracted_textual_gradient"]}
    state["evaluation"]["verifier"] = result
    atomic_json(path, state)


def _publish_evidence(folder: Path, state: dict, generation: dict) -> None:
    """Write the new generation's evaluation evidence and mutation gradient.

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
    (folder / "textual_gradient.txt").write_text(result["abstracted_textual_gradient"])
    atomic_json(folder / "policy_bt_evidence.json", report)


if __name__ == "__main__":
    print(fold_gradient({"uuid": "smoke", "execution": {"eligible": False, "reason": "Missing reproducibility .sh script."}}, []))

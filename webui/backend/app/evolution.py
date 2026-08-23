"""One payload for the animated evolution replay: the family, fully annotated.

The desktop animator (workflow_evolution_anim.py) reads only lineage +
evaluation.txt and leaves the most telling signals on the floor: cost, QD /
novelty scores, the selection log, and the textual gradient that steered each
mutation. This view joins all of it per family member so the frontend can
narrate the evolution — which run mutated from which, steered by what
feedback, at what cost, with what outcome — without a request per node.

Follows store.py's defensive contract: absent files yield ``None`` fields,
never an exception.
"""

from __future__ import annotations

from typing import Any

from . import lineage, store

GRADIENT_SNIPPET_CHARS = 800


def _claim_counts(run_id: str) -> dict[str, int] | None:
    claims = store.read_evaluation_claims(run_id)
    if claims is None:
        return None
    counts = {"passed": 0, "failed": 0, "error": 0, "unsure": 0}
    for c in claims:
        status = str(c.get("status", ""))
        key = {"pass": "passed", "fail": "failed"}.get(status, status)
        if key in counts:
            counts[key] += 1
    return counts


def _gradient_snippet(run_id: str) -> str | None:
    text = store._read_text(store.run_path(run_id) / "textual_gradient.txt")
    if not text:
        return None
    text = text.strip()
    if len(text) > GRADIENT_SNIPPET_CHARS:
        return text[:GRADIENT_SNIPPET_CHARS].rstrip() + " …"
    return text


def family_evolution(run_id: str) -> dict[str, Any] | None:
    """Tree + per-node metrics/claims/gradient for *run_id*'s family."""
    tree = lineage.tree(run_id)
    if tree is None:
        return None
    idx_parents = {
        e["target"]: [] for e in tree["edges"]
    }
    for e in tree["edges"]:
        idx_parents[e["target"]].append(e["source"])

    nodes = []
    for n in tree["nodes"]:
        uuid = n["id"]
        metrics = store.read_run_metrics(uuid) or {}
        selection = metrics.get("selection_log")
        selection = selection if isinstance(selection, dict) else {}
        nodes.append({
            **n,
            "parents": idx_parents.get(uuid, []),
            "score_uncapped": metrics.get("overall_score_uncapped"),
            "qd_score": metrics.get("qd_score"),
            "novelty_score": metrics.get("novelty_score"),
            "iteration_cost_usd": metrics.get("iteration_cost_usd"),
            "cumulative_cost_usd": metrics.get("cumulative_cost_usd"),
            "wall_time_s": metrics.get("iteration_wall_time_s"),
            "on_error": metrics.get("on_error"),
            "claims": _claim_counts(uuid),
            "gradient_snippet": _gradient_snippet(uuid),
            "selection": {
                "improvement_type": selection.get("improvement_type"),
                "delta_reward": selection.get("delta_reward"),
                "is_validated": selection.get("is_validated"),
                "confidence": selection.get("confidence"),
                "admit_rejected": selection.get("admit_rejected"),
            } if selection else None,
        })
    return {"focus": run_id, "nodes": nodes, "edges": tree["edges"]}

"""Gradient v3 (E19b-V5 decisive-first) + ``evaluation.txt`` rendering.

The E35 PRIME gradient format: DECISIVE LOSSES lead ("WHAT TO FIX
FIRST" — each row carries the goal-anchored requirement, the measured
score and the scorer's evidence), then the decisive wins, the E24
execution facts, the E26 visual evidence for figure tasks, the dead
claims report, the all-claims transcript and the pairwise record.
NO elimination-point framing — E19c measured it significantly worse
(0.553 vs 0.672 relevance); every comparative row is re-anchored to the
goal requirement it implies (E32). Faithful transcript of measured
facts; no editorializing, no labels.
"""

from __future__ import annotations

from typing import Any

from .claims import STAGES

_HEADER_BAR = "=" * 60
_TRANSCRIPT_CAP = 4000


def _sign(sa: float | None, sb: float | None) -> str:
    """+ / - / = / n/a for one claim between two generations."""
    if sa is None or sb is None:
        return "n/a"
    return "+" if sa > sb else ("-" if sa < sb else "=")





def build_gradient(
    uuid: str,
    goal: str,
    now_scores: dict[str, float | None],
    evidence: dict[str, str],
    claims: list[dict[str, Any]],
    surviving: list[str],
    pair_records: list[dict[str, Any]],
    reward: float,
    win_rate: float,
    mean_score: float,
    dead_claims: list[dict[str, Any]],
    exec_facts: dict[str, Any] | None = None,
    stages: tuple[str, ...] = STAGES,
) -> str:
    """Assemble the E35-PRIME (V5 decisive-first) gradient for one generation.

    Args:
        exec_facts: The execution-gate record (``{"status", "runtime_s",
            "cap"}`` from ``ExecutionGateLayer.last_facts``) or None when
            the gate is disabled / has no data for this generation.
        stages: The ladder order (``FIGURE_STAGES`` for figure tasks —
            visual rows rank earliest).
    """
    wins = sum(1 for p in pair_records if p["outcome"] == "win")
    losses = sum(1 for p in pair_records if p["outcome"] == "loss")
    ties = sum(1 for p in pair_records if p["outcome"] == "tie")
    order = {s: i for i, s in enumerate(stages)}

    # Decisive losses/wins grouped by stage, temporal order within stage;
    # every comparative row re-anchored to the goal requirement (E32).
    lost_rows: list[tuple[int, int, str]] = []
    won_rows: list[tuple[int, int, str]] = []
    claim_lines: list[str] = []
    for c in claims:
        cid = c["id"]
        if cid not in surviving and cid not in now_scores:
            continue
        s = now_scores.get(cid)
        ev = evidence.get(cid, "")
        stage = c.get("stage", "result")
        rank = order.get(stage, len(stages))
        tidx = int(c.get("temporal_index", 0))
        if s is None:
            claim_lines.append(
                f"- [{cid}] (stage {stage}) {c.get('statement', '')}\n"
                f"  score: NOT SCORED ({ev or 'scorer failed'})"
            )
            continue
        rivals = [p.get("prev_scores", {}).get(cid) for p in pair_records]
        scored = [r for r in rivals if r is not None]
        if scored:
            rank_n = 1 + sum(1 for r in scored if r > s)
            stats = (
                f"rivals min {min(scored):.3f}, median "
                f"{sorted(scored)[len(scored) // 2]:.3f}, "
                f"max {max(scored):.3f} (rank {rank_n}/{len(scored) + 1})"
            )
        else:
            stats = "no scored predecessor yet"
        claim_lines.append(
            f"- [{cid}] (stage {stage}) {c.get('statement', '')}\n"
            f"  target: {c.get('target', '')}\n"
            f"  score: {s:.3f} | {stats}\n"
            f"  evidence: {ev}"
        )
        if not rivals:
            continue
        bal = sum(((s > r) - (s < r)) for r in scored)
        row = (
            f"- [{cid}] (stage {stage}) REQUIREMENT (from the task goal): "
            f"{c.get('statement', '')}\n"
            f"  this workspace: {s:.3f} (measured evidence: {ev[:160]})\n"
            f"  decisive balance vs predecessors: {bal:+d} ("
            + ("LOSING" if bal < 0 else "WINNING")
            + " this claim)"
        )
        (lost_rows if bal < 0 else won_rows if bal > 0 else []).append(
            (rank, tidx, row)
        )
    lost_rows.sort(key=lambda t: (t[0], t[1]))
    won_rows.sort(key=lambda t: (t[0], t[1]))

    transcript_lines = []
    for p in sorted(pair_records, key=lambda p: -abs(p.get("d", 0))):
        parts = [
            f"{cid}{_sign(now_scores.get(cid), p.get('prev_scores', {}).get(cid))}"
            for cid in surviving
        ]
        detail = p.get("detail") or {}
        where = (
            f"decided at stage {detail.get('stage')}"
            if detail.get("stage")
            else f"d={p.get('d', 0):+d}"
        )
        transcript_lines.append(
            f"- vs {p.get('prev_uuid', '?')} [{p['outcome']}; {where}] "
            + " ".join(parts)
        )
    transcript = "\n".join(transcript_lines)[:_TRANSCRIPT_CAP]

    dropped = (
        "\n".join(
            f"- [{c['id']}] (stage {c.get('stage', 'result')}) "
            f"{c.get('statement', '')[:130]} "
            f"(dropped: {c.get('drop_reason', 'unknown')})"
            for c in dead_claims
        )
        or "(none)"
    )

    lost_block = (
        "\n".join(r for _, _, r in lost_rows[:5]) or "(none — no decisive losses)"
    )
    won_block = "\n".join(r for _, _, r in won_rows[:5]) or "(none yet)"
    claim_block = "\n".join(claim_lines) or "(none)"

    # E24 execution facts (stage 5 gate): status, wall-clock, cap applied.
    status = (exec_facts or {}).get("status")
    runtime = (exec_facts or {}).get("runtime_s")
    cap = (exec_facts or {}).get("cap")
    exec_line = (
        f"- status: {status}, wall-clock {runtime:.1f}s"
        if runtime is not None
        else f"- status: {status if status is not None else 'no re-execution evidence'}"
    )
    if exec_facts is None:
        cap_note = "gate disabled for this run (hybrid_verifier_execution_gate=False)"
    elif cap is None:
        cap_note = "none (clean re-execution recovery or no evidence)"
    else:
        cap_note = f"reward capped at {cap:.1f} ({status} re-execution)"
    gate_line = f"- gate consequence: {cap_note}"

    # E26 visual evidence: what the figures showed against the frozen
    # goal-anchored criteria (figure tasks only).
    vis_claims = [c for c in claims if c.get("stage") == "visual"]
    vis_lines = []
    for c in vis_claims:
        s = now_scores.get(c["id"])
        ev = evidence.get(c["id"], "")
        s_txt = f"{s:.3f}" if s is not None else "not scored"
        vis_lines.append(
            f"- [{c['id']}] REQUIREMENT (from the task goal): "
            f"{c.get('statement', '')}\n"
            f"  visual score: {s_txt} — what the figures show: {ev}"
        )
    vis_block = (
        "\n".join(vis_lines)
        or "(no visual criteria for this task — non-figure deliverable)"
    )

    return f"""# Hybrid verifier gradient v3 (PRIME) — {uuid}
Task goal: {goal[:500]}
Generated: verifier v3 PRIME — deterministic per-claim policy scorers (the SAME
script for every workspace) + goal-anchored visual rung for figure tasks
(criteria frozen before scoring) + execution gate; decisive-first assembly.
Faithful measured transcript; NO labels used or shown.

Bottom line: reward {reward:.4f} | win-rate (ties=0.5) {win_rate:.3f} over
{len(pair_records)} comparisons ({wins}W/{losses}L/{ties}T) | {len(surviving)} of
{len(claims)} claims discriminate the candidates ({len(dead_claims)} dropped as
non-discriminative) | mean claim score {mean_score:.3f}.

## 1. WHAT TO FIX FIRST — decisive lost claims (earliest ladder stage first)
{lost_block}
## 2. WHAT ALREADY WORKS — decisive wins (keep these properties)
{won_block}
## 3. Execution facts — re-execution gate (crash/no-entry/timeout -> cap 0.0; divergent -> cap 0.5)
{exec_line}
{gate_line}
## 4. Visual evidence — figure deliverable judged against goal-anchored visual criteria
{vis_block}
## 5. Dead claims report (non-discriminative; one line each)
{dropped}
## 6. All claim scores for this workspace (0..1; evidence = scorer output, verbatim)
{claim_block}
## 7. Comparison transcript (per-claim sign vs every previous generation: + win, - loss, = tie)
{transcript or "(no previous generations to compare yet)"}
"""


def build_evaluation_report(
    uuid: str,
    scores: dict[str, Any],
    claims: list[dict[str, Any]],
    now_scores: dict[str, float | None],
    evidence: dict[str, str],
    surviving: list[str],
    pair_records: list[dict[str, Any]],
) -> str:
    """Render the human-readable ``evaluation.txt`` (stage headers)."""
    lines = [
        "Hybrid Verifier Evaluation (E35 PRIME, temporal ladder)",
        _HEADER_BAR,
        (
            f"Claims: {scores.get('n_claims', 0)}  "
            f"surviving={scores.get('n_surviving', 0)}  "
            f"dropped={scores.get('n_dropped', 0)}  "
            f"scored={scores.get('n_scored', 0)}  "
            f"scorer_failures={scores.get('n_scorer_failures', 0)}"
        ),
        (
            f"Reward: {scores.get('overall_score', 0.0):.3f} "
            f"(pairwise {scores.get('pairwise_mode', 'temporal')}), "
            f"mean claim score {scores.get('mean_claim_score', 0.0):.3f}, "
            f"win_rate {scores.get('win_rate', 0.0):.3f} "
            f"over {scores.get('n_pairs', 0)} previous generations "
            f"({scores.get('n_wins', 0)}W/{scores.get('n_losses', 0)}L/"
            f"{scores.get('n_ties', 0)}T)"
        ),
        (
            f"Refinement: {scores.get('n_replacements', 0)} replacement claims "
            f"this generation"
        ),
        "",
        "Per-claim results (temporal order):",
    ]
    current_stage = None
    for c in claims:
        stage = c.get("stage", "result")
        if stage != current_stage:
            current_stage = stage
            lines.append(f"--- stage: {stage.upper()} ---")
        cid = c["id"]
        s = now_scores.get(cid)
        marker = "SURVIVING" if cid in surviving else "DROPPED"
        score_txt = "n/a (scorer failed)" if s is None else f"{s:.3f}"
        lines.append(
            f"[{cid}] ({c.get('category', 'other')}; {marker}) {c.get('statement', '')}"
        )
        lines.append(f"  score: {score_txt}  (pass >= 0.5)")
        ev = evidence.get(cid, "")
        if ev:
            lines.append(f"  evidence: {ev}")
        if cid not in surviving and c.get("drop_reason"):
            lines.append(f"  dropped: {c['drop_reason']}")
        lines.append("")
    if pair_records:
        lines.append("Pairwise record vs previous generations:")
        for p in pair_records:
            detail = p.get("detail") or {}
            where = (
                f"decided at stage {detail.get('stage')}, claim "
                f"{detail.get('claim_id')}"
                if detail.get("stage")
                else f"d={p.get('d', 0):+d}, dm={p.get('dm', 0):+.3f}, "
                f"k={p.get('k_eff', 0)}"
            )
            lines.append(f"  vs {p.get('prev_uuid', '?')}: {p['outcome']} ({where})")
    else:
        lines.append(
            "Pairwise record: (first generation of this task — "
            "reward is the mean claim score fallback)"
        )
    lines.append("")
    return "\n".join(lines)

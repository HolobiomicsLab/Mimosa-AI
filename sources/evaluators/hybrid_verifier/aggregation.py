"""Zero-variance filter, pairwise comparison modes and reward.

Aggregation modes (``hybrid_verifier_pairwise_mode``):

- ``temporal`` (DEFAULT — operator semantics, temporal elimination): the
  pairwise winner is decided at the EARLIEST ladder stage where the two
  generations differ. Within a stage: higher pass-count wins (a claim
  passes at ``score >= PASS_THRESHOLD``); tie → that stage's mean
  score-diff; still tied → next stage; all equal → tie. Operator rule:
  "a lower-level claim not passing for A that B passes = B wins (A lost
  already); whoever makes it longer in the temporal order wins" — a
  script-stage failure dominates any result-stage advantage. The ladder
  ORDER is configurable per task (E26/E35): ``STAGES`` for text tasks,
  ``FIGURE_STAGES`` (visual first) for figure tasks — a visual-stage
  failure dominates every later one.
- ``temporal_strict``: claim-level first-difference lexicographic — the
  first claim (in temporal order) whose scores differ decides the pair
  outright (higher score wins).
- ``sign_sum`` / ``mean_diff`` / ``escalation``: the E19b flat modes,
  kept as options (see ``experiments_verifiers/harness/e19b_dryrun.py``).

Also ports E19's ``claim_stats`` variance filter: a claim with no
observed scores is dead (``all_fail``); a claim whose observed scores are
all equal (with ≥ 2 observations) is dead (``zero_variance``).
"""

from __future__ import annotations

import math
import statistics
from typing import Any

# Modes accepted by winner_of / win_rate_reward.
PAIRWISE_MODES: tuple[str, ...] = (
    "temporal",
    "temporal_strict",
    "mean_diff",
    "sign_sum",
    "escalation",
)
DEFAULT_PAIRWISE_MODE = "temporal"

# Ladder stage order, earliest first (must match claims.STAGES /
# claims.FIGURE_STAGES). STAGES is the default text ladder; FIGURE_STAGES
# inserts the E26 visual rung FIRST for figure tasks (E35 production
# default passes the order explicitly per task).
STAGES: tuple[str, ...] = ("script", "log", "result")
FIGURE_STAGES: tuple[str, ...] = ("visual", "script", "log", "result")

# A claim "passes" its rung at this score (operator: pass-counts per stage).
PASS_THRESHOLD = 0.5

# Minimum observations before the zero-variance verdict is trusted.
MIN_OBS_FOR_VARIANCE = 2


def claim_stats(observed: list[float | None]) -> dict[str, Any]:
    """Per-claim observation stats + drop decision (E19 semantics)."""
    obs = [float(s) for s in observed if s is not None]
    if not obs:
        return {
            "n_obs": 0,
            "variance": None,
            "dropped": True,
            "drop_reason": "all_fail",
        }
    if len(obs) < MIN_OBS_FOR_VARIANCE:
        return {
            "n_obs": len(obs),
            "variance": None,
            "dropped": False,
            "drop_reason": None,
        }
    var = statistics.pvariance(obs)
    if var == 0.0:
        return {
            "n_obs": len(obs),
            "variance": 0.0,
            "dropped": True,
            "drop_reason": "zero_variance",
        }
    return {"n_obs": len(obs), "variance": var, "dropped": False, "drop_reason": None}


def decisive(
    now: dict[str, float | None],
    prev: dict[str, float | None],
    claim_ids: list[str],
) -> tuple[int, float, int]:
    """Flat claim-summed comparison (sign-sum, mean-diff, k)."""
    d = dm = 0
    k = 0
    for cid in claim_ids:
        sa = now.get(cid)
        sb = prev.get(cid)
        if sa is None or sb is None:
            continue
        k += 1
        d += (sa > sb) - (sa < sb)
        dm += sa - sb
    return d, dm, k


def winner_of(d: int, dm: float, mode: str) -> int:
    """Flat-mode winner from A's (current generation's) view: +1/-1/0."""
    if mode == "sign_sum":
        return (d > 0) - (d < 0)
    if mode == "mean_diff":
        return (dm > 0) - (dm < 0)
    if mode == "escalation":
        if d:
            return (d > 0) - (d < 0)
        return (dm > 0) - (dm < 0)
    raise ValueError(f"unknown pairwise mode {mode!r}")


def temporal_winner(
    now: dict[str, float | None],
    prev: dict[str, float | None],
    claims: list[dict[str, Any]],
    mode: str = "temporal",
    stages: tuple[str, ...] = STAGES,
) -> tuple[int, dict[str, Any]]:
    """Temporal-elimination winner between two generations.

    Args:
        now: Current generation's per-claim scores.
        prev: Previous generation's per-claim scores.
        claims: Claims in TEMPORAL order, each carrying ``id`` and
            ``stage``; claims unmeasured on either side are skipped.
        mode: ``temporal`` (stage pass-count elimination) or
            ``temporal_strict`` (first differing claim wins outright).
        stages: The ladder order, earliest first — ``STAGES`` for text
            tasks, ``FIGURE_STAGES`` (visual first) for figure tasks.
            Stages not in the order never decide.

    Returns:
        ``(winner, detail)`` — winner is +1 (now) / -1 (prev) / 0 (tie);
        detail describes the deciding stage/claim for gradients.
    """

    def _score(gen: dict[str, float | None], cid: str) -> float | None:
        return gen.get(cid)

    if mode == "temporal_strict":
        for c in claims:
            sa = _score(now, c["id"])
            sb = _score(prev, c["id"])
            if sa is None or sb is None:
                continue
            if sa != sb:
                return (1 if sa > sb else -1), {
                    "kind": "claim",
                    "stage": c.get("stage", "result"),
                    "claim_id": c["id"],
                    "now": sa,
                    "prev": sb,
                }
        return 0, {"kind": "all_equal"}

    # temporal: stage-by-stage elimination, earliest stage wins.
    for stage in stages:
        stage_claims = [c for c in claims if c.get("stage", "result") == stage]
        pairs = [
            (c["id"], _score(now, c["id"]), _score(prev, c["id"])) for c in stage_claims
        ]
        pairs = [(cid, a, b) for cid, a, b in pairs if a is not None and b is not None]
        if not pairs:
            continue
        pa = sum(1 for _, a, _ in pairs if a >= PASS_THRESHOLD)
        pb = sum(1 for _, _, b in pairs if b >= PASS_THRESHOLD)
        if pa != pb:
            winner = 1 if pa > pb else -1
            # first failing (or passing) claim of the losing side at this stage
            cid, sa, sb = next(
                (
                    t
                    for t in pairs
                    if (t[1] >= PASS_THRESHOLD) != (t[2] >= PASS_THRESHOLD)
                ),
                pairs[0],
            )
            return winner, {
                "kind": "stage",
                "stage": stage,
                "claim_id": cid,
                "now_passes": pa,
                "prev_passes": pb,
                "now": sa,
                "prev": sb,
            }
        dm = sum(a - b for _, a, b in pairs)
        if dm != 0:
            cid, sa, sb = max(pairs, key=lambda t: abs(t[1] - t[2]))
            return (1 if dm > 0 else -1), {
                "kind": "stage",
                "stage": stage,
                "claim_id": cid,
                "now_passes": pa,
                "prev_passes": pb,
                "now": sa,
                "prev": sb,
                "tiebreak": "mean_diff",
            }
    return 0, {"kind": "all_equal"}

def win_rate_reward(
    now: dict[str, float | None],
    previous: list[dict[str, float | None]],
    claims: list[dict[str, Any]],
    mode: str = DEFAULT_PAIRWISE_MODE,
    stages: tuple[str, ...] = STAGES,
) -> dict[str, Any]:
    """Mean win-rate reward against every previous generation.

    Args:
        now: Current generation's score vector.
        previous: Score vectors of every previous generation of the task.
        claims: Claims (with ``id``/``stage``) in temporal order — used
            by the temporal modes as the ordered ladder and by the flat
            modes as the surviving-claim id list.
        mode: One of ``PAIRWISE_MODES`` (default ``temporal``).
        stages: Ladder order for the temporal modes (``FIGURE_STAGES``
            for figure tasks — the visual rung decides first).

    Returns:
    """
    if mode not in PAIRWISE_MODES:
        raise ValueError(f"unknown pairwise mode {mode!r}")
    claim_ids = [c["id"] for c in claims]
    if not previous:
        vals = [now[c] for c in claim_ids if now.get(c) is not None]
        reward = sum(vals) / len(vals) if vals else 0.0
        return {
            "reward": reward,
            "n_pairs": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "fallback": "mean_claim",
            "pairs": [],
        }
    wins = losses = ties = 0
    pair_records: list[dict[str, Any]] = []
    for idx, prev in enumerate(previous):
        if mode in ("temporal", "temporal_strict"):
            w, detail = temporal_winner(now, prev, claims, mode, stages)
            rec = {
                "prev_index": idx,
                "outcome": "win" if w > 0 else ("loss" if w < 0 else "tie"),
                "detail": detail,
            }
        else:
            d, dm, k = decisive(now, prev, claim_ids)
            w = winner_of(d, dm, mode)
            rec = {
                "prev_index": idx,
                "d": d,
                "dm": round(dm, 6),
                "k_eff": k,
                "outcome": "win" if w > 0 else ("loss" if w < 0 else "tie"),
            }
        if w > 0:
            wins += 1
        elif w < 0:
            losses += 1
        else:
            ties += 1
        pair_records.append(rec)
    n = len(previous)
    return {
        "reward": (wins + 0.5 * ties) / n,
        "n_pairs": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "fallback": None,
        "pairs": pair_records,
    }


def _sigmoid(x: float) -> float:
    """Numerically tolerant logistic function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve a small dense linear system by Gaussian elimination."""
    n = len(b)
    m = [row[:] + [bi] for row, bi in zip(a, b, strict=True)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            m[col][col] += 1e-9
            piv = col
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] / pv
            if f:
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / m[r][r]
    return x

def bradley_terry_reward(
    now: dict[str, float | None],
    previous: list[dict[str, float | None]],
    claims: list[dict[str, Any]],
    mode: str = DEFAULT_PAIRWISE_MODE,
    stages: tuple[str, ...] = STAGES,
    ridge: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> dict[str, Any]:
    """Bradley–Terry reward: logistic-MLE strengths over ALL pairwise
    outcomes among the task's generations (current + every previous),
    evaluated under the configured pairwise policy.

    Unlike :func:`win_rate_reward` (a raw majority score), the BT model
    weighs each win by opponent strength: beating a strong earlier
    generation contributes more than beating a weak one. Matches among
    PREVIOUS generations are recomputed from their stored score vectors
    so opponent strengths are estimable at scoring time; past recorded
    rewards are never revised (online/causal: only already-known matches
    enter the fit). Ties contribute half a win to each side. The ridge
    (L2 on strengths, default 1.0) matches the measured experiment
    convention (``experiments_verifiers/harness/e15_bt_eval.py``) and
    keeps strengths finite under perfect separation.

    Reward = sigmoid(beta_now − mean(beta)) — the E15/E19 experiment
    convention (``logistic(beta − mean beta)``).

    Args:
        now: Current generation's score vector.
        previous: Score vectors of every previous generation of the task.
        mode: One of ``PAIRWISE_MODES`` (default ``temporal``).
        stages: Ladder order for the temporal modes (``FIGURE_STAGES``
            for figure tasks — the visual rung decides first).
        ridge: L2 regularisation strength on the strengths.
        max_iter: Newton iteration cap.
        tol: Convergence tolerance on the update size.

    Returns:
        Dict with ``reward`` (0..1), ``strength_now``, ``strengths``
        (centered, current generation first), match counts, the current
        generation's raw ``win_rate`` diagnostic, and convergence info.
        First generation (no previous) falls back to the mean claim
        score exactly like :func:`win_rate_reward`.
    """
    if mode not in PAIRWISE_MODES:
        raise ValueError(f"unknown pairwise mode {mode!r}")
    claim_ids = [c["id"] for c in claims]
    if not previous:
        vals = [now[c] for c in claim_ids if now.get(c) is not None]
        reward = sum(vals) / len(vals) if vals else 0.0
        return {
            "reward": reward,
            "n_pairs": 0,
            "strength_now": 0.0,
            "strengths": [],
            "win_rate": None,
            "iterations": 0,
            "converged": True,
            "fallback": "mean_claim",
        }

    def outcome(a: dict[str, float | None], b: dict[str, float | None]) -> int:
        if mode in ("temporal", "temporal_strict"):
            w, _ = temporal_winner(a, b, claims, mode, stages)
            return w
        d, dm, _ = decisive(a, b, claim_ids)
        return winner_of(d, dm, mode)

    players = [now, *previous]
    n = len(players)
    wins = [[0.0] * n for _ in range(n)]
    games = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            w = outcome(players[i], players[j])
            games[i][j] = games[j][i] = 1
            if w > 0:
                wins[i][j] += 1.0
            elif w < 0:
                wins[j][i] += 1.0
            else:
                wins[i][j] += 0.5
                wins[j][i] += 0.5

    beta = [0.0] * n
    converged = False
    iterations = 0
    for _it in range(1, max_iter + 1):
        iterations = _it
        grad = [0.0] * n
        hess = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j or not games[i][j]:
                    continue
                s = _sigmoid(beta[i] - beta[j])
                grad[i] += wins[i][j] - games[i][j] * s
                hess_v = games[i][j] * s * (1.0 - s)
                hess[i][i] -= hess_v
                hess[i][j] += hess_v
            grad[i] -= ridge * beta[i]
            hess[i][i] -= ridge
        step = _solve_linear(hess, grad)
        mean_step = sum(step) / n
        # Hessian is negative-definite: Newton ASCENT is beta - H^-1 g.
        beta = [b - (d - mean_step) for b, d in zip(beta, step, strict=True)]
        if max(abs(d - mean_step) for d in step) < tol:
            converged = True
            break
    mean_beta = sum(beta) / n
    beta = [b - mean_beta for b in beta]

    my_wins = sum(wins[0][j] for j in range(1, n))
    return {
        "reward": _sigmoid(beta[0]),
        "n_pairs": n - 1,
        "strength_now": round(beta[0], 6),
        "strengths": [round(b, 6) for b in beta],
        "win_rate": my_wins / (n - 1),
        "iterations": iterations,
        "converged": converged,
        "fallback": None,
    }


def mean_claim_score(
    now: dict[str, float | None],
    claim_ids: list[str],
) -> float:
    """Mean observed score over the surviving claims (absolute quality)."""
    vals = [now[c] for c in claim_ids if now.get(c) is not None]
    return sum(vals) / len(vals) if vals else 0.0

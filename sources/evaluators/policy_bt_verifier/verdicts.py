"""Pair verdicts and tournament rewards for the policy_bt verifier.

Ports from the offline harness (E44 ``e44_bt.py``):

- ``bt_fit_weighted`` — the E15/E14 Bradley-Terry logistic MLE with ridge
  1.0 (copied verbatim; ties enter as two half-weight matches);
- ``expit`` ratings = ``sigmoid(beta - mean beta)`` (production reward
  convention);
- win-rate reward over arrival-moment outcomes (the DEFAULT — E19c
  measured BT-reward worse: 0.643 vs 0.742).

Production additions:

- :func:`crash_rung` — the deterministic exec-crash rung applied to BOTH
  channels before any policy runs (safeguard 8: identical error
  signatures tie — rule-correct, chem clintox lesson; one-sided crashes
  are decisive for the clean side; both-crash with clearly different
  signatures is decided by partial evidence, else an honest tie);
- :func:`code_pair_verdict` — both-orders policy execution with the
  runtime antisymmetry assertion (safeguard 5: disagreement -> tie +
  WARNING, never a coin flip).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

RIDGE = 1.0


def expit(x):
    """Numerically safe logistic."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def bt_fit_weighted(w_idx, l_idx, wgt, n, iters=25, tol=1e-10):
    """Bradley-Terry weighted logistic MLE (verbatim from e15_bt_eval.py).

    Args:
        w_idx: Winner indices per match.
        l_idx: Loser indices per match.
        wgt: Match weights (ties = two matches at 0.5 — E14/E15).
        n: Number of players.
        iters: Newton step cap.
        tol: Convergence tolerance on the step norm.

    Returns:
        Strength vector ``theta`` (len *n*).
    """
    w_idx = np.asarray(w_idx, dtype=int)
    l_idx = np.asarray(l_idx, dtype=int)
    wgt = np.asarray(wgt, dtype=float)
    th = np.zeros(n)
    if len(w_idx) == 0:
        return th
    for _ in range(iters):
        p = expit(th[w_idx] - th[l_idx])
        w = wgt * p * (1 - p)
        g = np.zeros(n)
        np.add.at(g, w_idx, wgt * (1 - p))
        np.add.at(g, l_idx, -wgt * (1 - p))
        g -= RIDGE * th
        H = np.eye(n) * RIDGE
        np.add.at(H, (w_idx, w_idx), w)
        np.add.at(H, (l_idx, l_idx), w)
        np.add.at(H, (w_idx, l_idx), -w)
        np.add.at(H, (l_idx, w_idx), -w)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        th = th + step
        if np.max(np.abs(step)) < tol:
            break
    return th


# ------------------------------------------------------------------
# Deterministic crash rung (safeguard 8)
# ------------------------------------------------------------------


def _crashed(ledger: dict[str, Any] | None) -> bool:
    """A generation crashed (or timed out, or never launched) on re-exec.

    P2.8: a run whose source workspace was MUTATED during re-execution
    (``source_manifest_intact=False``) counts as crashed — its evidence
    was produced against a workspace the guard can no longer vouch for.
    """
    if not ledger or ledger.get("launch_error"):
        return True
    if ledger.get("source_manifest_intact") is False:
        return True
    return ledger.get("exit_code") != 0 or bool(ledger.get("timed_out"))


def _fresh_count(ledger: dict[str, Any] | None) -> int:
    """Number of artifacts the fresh run produced (partial evidence)."""
    return len((ledger or {}).get("fresh_artifacts") or {})


def _crash_signature(ledger: dict[str, Any] | None) -> list[str]:
    """Sorted error signature PLUS crash-class tokens (P2.5).

    A timeout, a launch failure and an exit-1 crash are DIFFERENT failure
    classes: without class tokens a timed-out side and an exit-1 side
    with equal (possibly empty) error signatures compared as "identical
    signature" — a confidence-1.0 tie that hid a real difference. The
    ``exit_code`` token joins so exits of different severity never merge.
    """
    lg = ledger or {}
    sig = sorted(str(s) for s in lg.get("error_signature") or [])
    if lg.get("timed_out"):
        sig.append("class:timed_out")
    if lg.get("launch_error"):
        sig.append("class:launch_error")
    if lg.get("source_manifest_intact") is False:
        sig.append("class:manifest_mutated")
    ec = lg.get("exit_code")
    if ec is not None:
        sig.append(f"class:exit_code={ec}")
    return sig


def crash_rung(
    uuid_a: str,
    ledger_a: dict[str, Any] | None,
    uuid_b: str,
    ledger_b: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Exec-crash rung for one pair; ``None`` = both clean, channel decides.

    Rules (plan §4 8 / §4b ladder position):

    - one side crashed, the other clean -> the clean side wins (decisive);
    - both crashed with IDENTICAL error signature AND crash class -> tie
      (rule-correct: an environment regression is not a quality
      difference — chem clintox lesson);
    - both crashed with clearly different signatures/classes -> decided
      by partial evidence (fresh artifact counts); equal evidence -> tie.

    P2.4: when either ledger is a BOOTSTRAP (digest rebuilt from a
    snapshot, no fresh execution facts) this rung returns ``None`` —
    "unknown", not "crashed" — and the comparison channel decides.
    """
    if (ledger_a or {}).get("bootstrap") or (ledger_b or {}).get("bootstrap"):
        return None
    ca, cb = _crashed(ledger_a), _crashed(ledger_b)
    if not ca and not cb:
        return None
    if ca and not cb:
        return _rung_verdict(uuid_b, uuid_a, ledger_b, ledger_a)
    if cb and not ca:
        return _rung_verdict(uuid_a, uuid_b, ledger_a, ledger_b)
    sig_a = _crash_signature(ledger_a)
    sig_b = _crash_signature(ledger_b)
    if sig_a == sig_b:
        sig = sig_a[0] if sig_a else "no error signature"
        return {
            "winner": "tie",
            "channel": "crash_rung",
            "rung": "both_crashed_identical_signature",
            "deciding": ["exec.error_signature"],
            "confidence": 1.0,
            "feedback_winner": (
                "both sides crash with the identical error signature "
                f"({sig}) on fresh re-execution — tie is rule-correct"
            ),
            "feedback_looser": (
                "both sides crash with the identical error signature "
                f"({sig}) on fresh re-execution — tie is rule-correct"
            ),
        }
    fa, fb = _fresh_count(ledger_a), _fresh_count(ledger_b)
    if fa == fb:
        return {
            "winner": "tie",
            "channel": "crash_rung",
            "rung": "both_crashed_different_signature_equal_evidence",
            "deciding": ["exec.error_signature", "exec.fresh_artifacts"],
            "confidence": 0.5,
            "feedback_winner": (
                f"both sides crash (signatures differ) with equal partial "
                f"evidence: {fa} fresh artifacts each — honest tie"
            ),
            "feedback_looser": (
                f"both sides crash (signatures differ) with equal partial "
                f"evidence: {fa} fresh artifacts each — honest tie"
            ),
        }
    winner, loser = (uuid_a, uuid_b) if fa > fb else (uuid_b, uuid_a)
    wl, ll = (ledger_a, ledger_b) if fa > fb else (ledger_b, ledger_a)
    return _rung_verdict(winner, loser, wl, ll)


def _exit_line(ledger: dict[str, Any] | None) -> str:
    """One measured exec fact line for rung feedback."""
    if not ledger:
        return "no re-execution ledger"
    if ledger.get("timed_out"):
        return f"timed out after {ledger.get('wall_time_s')}s"
    if ledger.get("launch_error"):
        return f"never launched ({ledger['launch_error']})"
    sig = (ledger.get("error_signature") or ["nonzero exit"])[0]
    return f"exit {ledger.get('exit_code')} ({sig})"


def _rung_verdict(
    winner: str,
    loser: str,
    ledger_w: dict[str, Any] | None,
    ledger_l: dict[str, Any] | None,
) -> dict[str, Any]:
    """One-sided crash verdict: clean side wins, with measured facts."""
    return {
        "winner": winner,
        "channel": "crash_rung",
        "rung": "one_side_crashed",
        "deciding": ["exec.exit_code"],
        "confidence": 1.0,
        "feedback_winner": (
            f"fresh re-execution {_exit_line(ledger_w)} with "
            f"{_fresh_count(ledger_w)} fresh artifacts vs predecessor "
            f"{_exit_line(ledger_l)}"
        ),
        "feedback_looser": (
            f"fresh re-execution {_exit_line(ledger_l)} while the other "
            f"side ran clean with {_fresh_count(ledger_w)} fresh artifacts; "
            f"re-run the deliverable script so it exits 0 and rewrites the "
            f"goal's deliverables"
        ),
    }


def code_pair_verdict(
    run_fn: Callable[[str, dict, dict], dict],
    policy_src: str,
    uuid_a: str,
    dA: dict[str, Any],
    uuid_b: str,
    dB: dict[str, Any],
    log: Any = None,
) -> dict[str, Any]:
    """Compare two digests with the task policy in BOTH orders.

    Runs ``compare(dA, dB)`` and ``compare(dB, dA)`` in subprocesses;
    agreement on the mirrored winner produces the verdict, disagreement
    is an instrument failure -> tie + WARNING (safeguard 5; offline
    E44-E48 saw 0 in ~700 pairs, any occurrence is noise, never a coin
    flip). Execution errors on both orders also degrade to a tie with a
    warning (a crashing policy must not decide pairs).
    """
    v_ab = run_fn(policy_src, dA, dB)
    v_ba = run_fn(policy_src, dB, dA)
    if "error" in v_ab or "error" in v_ba:
        err = v_ab.get("error") or v_ba.get("error")
        if log is not None:
            log.warn(f"policy pair ({uuid_a} vs {uuid_b}) errored: {str(err)[:200]}")
        return {
            "winner": "tie",
            "channel": "code_policy",
            "error": str(err)[:2048],
            "feedback_winner": "",
            "feedback_looser": "",
            "mirrored": None,
        }
    w_ab, w_ba = v_ab.get("winner"), v_ba.get("winner")
    mirrored = (
        (w_ab == "tie" and w_ba == "tie")
        or (w_ab == "A" and w_ba == "B")
        or (w_ab == "B" and w_ba == "A")
    )
    if not mirrored:
        if log is not None:
            log.warn(
                f"antisymmetry disagreement on pair ({uuid_a} vs {uuid_b}): "
                f"(A,B)->{w_ab} but (B,A)->{w_ba}; recording tie (instrument "
                f"noise, not a coin flip)"
            )
        return {
            "winner": "tie",
            "channel": "code_policy",
            "antisymmetry": "disagreement",
            "feedback_winner": "",
            "feedback_looser": "",
            "mirrored": False,
        }
    winner = "tie" if w_ab == "tie" else (uuid_a if w_ab == "A" else uuid_b)
    verdict = {
        "winner": winner,
        "channel": "code_policy",
        "confidence": v_ab.get("confidence"),
        "deciding": v_ab.get("deciding"),
        "rungs": v_ab.get("rungs"),
        "feedback_winner": v_ab.get("feedback_winner") or "",
        "feedback_looser": v_ab.get("feedback_looser") or "",
        "mirrored": True,
    }
    if v_ab.get("schema_error") and v_ba.get("schema_error"):
        # Both orders signal unknown digest structure (E48 SchemaError
        # contract): honest tie + the flag, so the evaluator can cache
        # it, refresh its recon and re-evaluate this pair. A one-sided
        # signal stays an antisymmetry tie WITHOUT the flag (the branch
        # above) — one order reading novel keys is instrument noise,
        # not a knowledge gap.
        verdict["schema_error"] = v_ab.get("schema_error")
    return verdict


# ------------------------------------------------------------------
# Rewards (arrival-moment; history never rewritten — E19c convention)
# ------------------------------------------------------------------


def win_rate_reward(outcomes: list[str]) -> dict[str, Any]:
    """Raw win-rate vs predecessors (ties count half; default mode)."""
    n = len(outcomes)
    if n == 0:
        return {"reward": 0.5, "wins": 0, "losses": 0, "ties": 0, "n_pairs": 0}
    wins = sum(1 for o in outcomes if o == "win")
    losses = sum(1 for o in outcomes if o == "loss")
    ties = sum(1 for o in outcomes if o == "tie")
    reward = (wins + 0.5 * ties) / n
    return {
        "reward": reward,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n_pairs": n,
    }


def bradley_terry_reward(
    uuids: list[str], pair_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """BT fit over the prefix {1..k} (the E44 replay semantics).

    Args:
        uuids: Arrival-ordered uuid prefix of the task.
        pair_records: Verdict records with ``winner`` (uuid or ``tie``)
            over pairs inside the prefix.

    Returns:
        Ratings per uuid, the fitted pick (argmax, latest tie-break) and
        convergence info.
    """
    pidx = {u: k for k, u in enumerate(uuids)}
    w_idx: list[int] = []
    l_idx: list[int] = []
    wgt: list[float] = []
    for rec in pair_records:
        a, b, w = rec.get("a"), rec.get("b"), rec.get("winner")
        if a not in pidx or b not in pidx or not w or w == "error":
            continue
        ia, ib = pidx[a], pidx[b]
        if w == "tie":
            w_idx += [ia, ib]
            l_idx += [ib, ia]
            wgt += [0.5, 0.5]
        else:
            wi, li = (ia, ib) if w == a else (ib, ia)
            w_idx.append(wi)
            l_idx.append(li)
            wgt.append(1.0)
    th = bt_fit_weighted(w_idx, l_idx, wgt, len(uuids))
    ratings = expit(th - th.mean())
    best = max(range(len(uuids)), key=lambda k: (ratings[k], k))
    return {
        "ratings": {
            u: round(float(r), 4) for u, r in zip(uuids, ratings, strict=False)
        },
        "pick": uuids[best],
        "theta": th.tolist(),
    }

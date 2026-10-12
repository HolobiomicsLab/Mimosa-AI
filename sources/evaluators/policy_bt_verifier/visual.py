"""Kimi per-PAIR visual channel for the policy_bt verifier (plan §4b).

One kimi-k3 call per (workspace, workspace) PAIR — identical in shape to
the code policy so the tournament and gradient layers stay
channel-agnostic:

- input: goal excerpt + the figures of BOTH sides, clearly labeled
  ("Candidate A: <images>", "Candidate B: <images>"), up to 2 largest
  pred PNGs per side, <=6 MB each, base64 data URLs;
- output (strict JSON): ``{winner: A|B|tie, confidence, figure_notes,
  feedback_winner, feedback_looser}`` — the feedback strings feed the
  gradient fold exactly like the code policy's.

Symmetry protocol (E14 convention): every pair is judged in BOTH orders —
``(A,B)`` and ``(B,A)`` are separate cached calls; agreement on the
mirrored winner produces the verdict, disagreement -> tie + WARNING (the
same antisymmetry assertion the code policy runs).

Safeguard 6 (E48 kimi fixes): retry-once on empty/parse-fail before
caching, and the cache lookup is a self-heal scan — cached parse-fails
never satisfy a hit; any later healed entry for the same key wins.

Cost discipline: verdicts cached forever by ``(prompt_version, uuid_a,
uuid_b, order, image hashes)``; ``visual_max_pairs_per_gen`` (0 =
unlimited) caps fresh pair-calls per generation — overflow pairs fall to
the code-policy path and the log says so. Kimi does image comparison
ONLY, never text judgments.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Up to 2 largest pred figures per side (plan §4b contract).
MAX_FIGS_PER_SIDE = 2
MAX_IMAGE_BYTES = 6 << 20

PAIR_PROMPT = """You are a strict visual comparison judge for a scientific \
computing task. Below are the figure deliverables of TWO candidate \
solutions, labeled Candidate A and Candidate B. Judge which candidate's \
figures better satisfy the task goal's visual requirements (correct plot \
type, required elements such as axes/labels/legends/color conventions the \
goal names, apparent correctness of the depicted data relationships). \
Ignore resolution unless unreadable.

# TASK GOAL (excerpt)
{goal}

# CANDIDATE A FIGURES
{figs_a}

# CANDIDATE B FIGURES
{figs_b}

Reply with ONLY a JSON object:
{{"winner": "A"|"B"|"tie", "confidence": 0.0-1.0,
  "figure_notes": "<=30 words: the decisive visual difference, with the \
measured observation (what each side shows)",
  "feedback_winner": "<=50 words of measured visual facts for the winner",
  "feedback_looser": "<=50 words of measured visual facts: the decisive \
visual deficiency of the loser + the goal's visual contract"}}"""

#: Prompt-hash version: the visual verdict cache keys on it.
VISUAL_PROMPT_VERSION = "vis-v1-" + hashlib.sha256(
    PAIR_PROMPT.encode("utf-8")
).hexdigest()[:12]


def select_figures(digest: dict[str, Any], root: Path) -> list[Path]:
    """Up to :data:`MAX_FIGS_PER_SIDE` largest readable pred figures.

    P3.4: size is NOT a presence filter here — the deliverable rung
    decides on digest figure records, and the 6 MB cap only filters
    uploads in the evaluator's vision call.

    Args:
        digest: The generation's digest (figure records, largest first
            by file size via the deliverables bytes).
        root: Directory the digest's paths are relative to.

    Returns:
        Figure file paths (may be empty when the side has no figures).
    """
    root = Path(root)
    bytes_of = {
        dv.get("path"): dv.get("bytes", 0) for dv in digest.get("deliverables") or []
    }
    figs = sorted(
        digest.get("figures") or [],
        key=lambda fg: -int(bytes_of.get(fg.get("path"), 0)),
    )
    out: list[Path] = []
    for fg in figs:
        if len(out) >= MAX_FIGS_PER_SIDE:
            break
        p = root / str(fg.get("path"))
        try:
            if p.exists():
                out.append(p)
        except OSError:
            continue
    return out


def images_digest(paths: list[Path]) -> str:
    """Stable hash of one side's figure set (cache key component)."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode("utf-8"))
        h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:12]


def build_pair_prompt(goal: str, names_a: list[str], names_b: list[str]) -> str:
    """Assemble the per-pair judge prompt (both sides clearly labeled)."""
    def _side(names: list[str]) -> str:
        return "\n".join(f"- {n}" for n in names) or "- (no figures)"

    return PAIR_PROMPT.format(
        goal=goal[:1500],
        figs_a=_side(names_a),
        figs_b=_side(names_b),
    )


def _parse_verdict(text: str) -> dict[str, Any] | None:
    """Strict-ish JSON parse of one kimi pair response (or None)."""
    t = (text or "").strip()
    if not t:
        return None
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        j = json.loads(t)
    except json.JSONDecodeError:
        return None
    if not isinstance(j, dict) or j.get("winner") not in {"A", "B", "tie"}:
        return None
    return j


class VisualChannel:
    """Per-pair kimi comparison with retry/self-heal and both orders.

    Args:
        vision_call: ``(prompt, figs_a, figs_b, max_tokens, role) ->
            raw text`` — one vision-LLM round-trip with EACH candidate's
            figure files attached (P3.2: per-side image lists so the
            call can label them). *role* is a per-attempt suffix
            (``r1_600``) the caller appends to the provider agent name
            so a retry can NEVER be served the cached first attempt
            (P3.1).
        cache_get: ``key -> parsed verdict | None`` (self-heal scan done
            by the caller-provided store: any non-parse-fail entry for
            the key satisfies the hit).
        cache_put: ``(key, entry)`` append (parse-fails stored too, for
            audit; they never satisfy a later hit).
        log: Optional StageLogger for WARNING lines.
    """

    def __init__(
        self,
        vision_call: Callable[[str, list[Path], list[Path], int, str], str],
        cache_get: Callable[[str], dict[str, Any] | None],
        cache_put: Callable[[str, dict[str, Any]], None],
        log: Any = None,
    ) -> None:
        self.vision_call = vision_call
        self.cache_get = cache_get
        self.cache_put = cache_put
        self.log = log
        self.fresh_calls = 0

    # ------------------------------------------------------------------

    def one_order(
        self,
        goal: str,
        uuid_a: str,
        figs_a: list[Path],
        uuid_b: str,
        figs_b: list[Path],
    ) -> dict[str, Any] | None:
        """One kimi call for one order ``(A, B)`` (cached, self-healing).

        Returns the parsed verdict dict or None on terminal parse fail
        (after the E48 retry-once). The retry passes a DIFFERENT role
        suffix (attempt + max_tokens) so the provider cache cannot
        re-serve the failed attempt verbatim (P3.1).
        """
        names_a = [p.name for p in figs_a]
        names_b = [p.name for p in figs_b]
        key = "|".join(
            (
                VISUAL_PROMPT_VERSION,
                uuid_a,
                uuid_b,
                "AB",
                images_digest(figs_a) + "_" + images_digest(figs_b),
            )
        )
        hit = self.cache_get(key)
        if hit is not None:
            return hit
        prompt = build_pair_prompt(goal, names_a, names_b)
        parsed: dict[str, Any] | None = None
        # E48 fix: retry once (bumped budget) on empty/parse-fail.
        for attempt, max_tokens in enumerate((600, 800), start=1):
            self.fresh_calls += 1
            role = f"r{attempt}_{max_tokens}"
            raw = self.vision_call(prompt, figs_a, figs_b, max_tokens, role)
            parsed = _parse_verdict(raw)
            if parsed is not None:
                break
            if self.log is not None and attempt == 1:
                self.log.detail(
                    f"kimi pair call parse fail (order AB {uuid_a[-8:]} vs "
                    f"{uuid_b[-8:]}); retrying once with max_tokens={max_tokens + 200}"
                )
        self.cache_put(key, parsed if parsed is not None else {"parse_fail": True})
        return parsed

    def judge_pair(
        self,
        goal: str,
        uuid_a: str,
        figs_a: list[Path],
        uuid_b: str,
        figs_b: list[Path],
    ) -> dict[str, Any] | None:
        """Both-orders kimi judgement of one pair (E14 protocol).

        Returns a code-policy-shaped verdict dict (winner uuid or tie),
        or ``None`` when either order terminally failed to parse (caller
        falls to the code policy).
        """
        v_ab = self.one_order(goal, uuid_a, figs_a, uuid_b, figs_b)
        v_ba = self.one_order(goal, uuid_b, figs_b, uuid_a, figs_a)
        if v_ab is None or v_ba is None:
            if self.log is not None:
                self.log.warn(
                    f"kimi pair ({uuid_a[-8:]} vs {uuid_b[-8:]}) parse-failed "
                    f"after retry; falling to the code policy for this pair"
                )
            return None
        w_ab, w_ba = v_ab.get("winner"), v_ba.get("winner")
        mirrored = (w_ab == "tie" and w_ba == "tie") or (
            w_ab == "A" and w_ba == "B"
        ) or (w_ab == "B" and w_ba == "A")
        if not mirrored:
            if self.log is not None:
                self.log.warn(
                    f"kimi antisymmetry disagreement on pair ({uuid_a[-8:]} vs "
                    f"{uuid_b[-8:]}): (A,B)->{w_ab} but (B,A)->{w_ba}; "
                    f"recording tie (instrument noise, not a coin flip)"
                )
            return {
                "winner": "tie",
                "channel": "kimi_pair",
                "antisymmetry": "disagreement",
                "mirrored": False,
                "confidence": 0.0,
                "figure_notes": (v_ab.get("figure_notes") or "")[:200],
                "feedback_winner": "",
                "feedback_looser": "",
            }
        winner = "tie" if w_ab == "tie" else (uuid_a if w_ab == "A" else uuid_b)
        return {
            "winner": winner,
            "channel": "kimi_pair",
            "confidence": v_ab.get("confidence"),
            "figure_notes": (v_ab.get("figure_notes") or "")[:200],
            "feedback_winner": v_ab.get("feedback_winner") or "",
            "feedback_looser": v_ab.get("feedback_looser") or "",
            "mirrored": True,
        }


def figure_rung(
    uuid_a: str,
    n_figs_a: int,
    uuid_b: str,
    n_figs_b: int,
) -> dict[str, Any] | None:
    """Deliverable rung for figure-dominant tasks (plan §4b).

    P3.4: the rung decides on DIGEST figure-record counts — whether the
    side PRODUCED figures — not on the upload list (a >6 MB figure is
    still a produced figure; the size cap only filters uploads). One
    side without figures loses here; when BOTH sides lack figures the
    code policy decides (returns None).
    """
    if bool(n_figs_a) == bool(n_figs_b):
        return None
    winner = uuid_a if n_figs_a else uuid_b
    n = n_figs_a or n_figs_b
    return {
        "winner": winner,
        "channel": "figure_rung",
        "rung": "one_side_no_figures",
        "deciding": ["figures"],
        "confidence": 1.0,
        "feedback_winner": (
            f"produces {n} figure deliverable(s) vs none on the other side"
        ),
        "feedback_looser": (
            "produced no figure deliverables while the other side produced "
            f"{n}; the task's deliverables are figure-dominant — produce the "
            "goal's figures into pred_results"
        ),
    }

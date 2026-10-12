"""LLM-written deterministic pairwise policy (plan §1 P3, §4 safeguards 1-3).

Port of the E45 v3 policy writer with ALL E48 fixes baked in:

- **Token cap + truncation rejection** (safeguard 1): policy-writing calls
  use ``max_tokens`` 9000 with escalation to 12000; a non-dict / truncated
  JSON response is NEVER used as code — it fails loudly into the repair
  loop as a screen-failing sentinel (the E48 regression where a truncated
  raw JSON text became the policy source).
- **Real repair plumbing** (safeguard 2): every retry passes the exact
  screen/runtime errors in the prompt AND uses a cache-busting agent role
  (``_fix<n>``) so a cached broken artifact can never be re-served.
- **Runtime smoke before acceptance** (safeguard 3): a policy must
  correctly compare 2 real digest pairs (both orders, mirrored winners)
  before it is used.
- **AST screen** (safeguard 4): stdlib import allowlist + banned builtin
  names; execution happens in a subprocess with a timeout, cwd = temp dir.

Policy versioning (operator decision 3): ``POLICY_SHA`` version = sha256
of the writer prompt; verdict caches are keyed by it, so a prompt change
invalidates old verdicts cleanly.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .recon import RECON_PROMPT_VERSION

ALLOWED_IMPORTS = {
    "json",
    "math",
    "statistics",
    "re",
    "itertools",
    "collections",
    "functools",
    "copy",
    "typing",
}
BANNED_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "breakpoint",
    "exit",
    "quit",
    "globals",
    "locals",
    "vars",
}
#: N12/P2.11: reflective attribute access defeats every static ban
#: (``getattr(__builtins__, "__imp"+"ort__")("os")``) — banned outright,
#: as is ANY dunder name/attribute (``__class__``, ``__globals__`` ...).
REFLECTIVE_CALLS = {"getattr", "setattr", "delattr"}


def _is_dunder(name: str) -> bool:
    """True for ``__dunder__`` identifiers (never needed by a policy)."""
    return len(name) > 4 and name.startswith("__") and name.endswith("__")

#: First-attempt / escalated generation budgets (E48: 6000 regressed;
#: >=9000 with 12000 escalation measured clean at 19/19 coverage).
MAX_TOKENS_FIRST = 12000
MAX_TOKENS_ESCALATED = 32768
#: Runtime-smoke cap on distinct digests (every pair, both orders).
SMOKE_MAX_DIGESTS = 6
#: N10: per-call smoke subprocess timeout — the old 60 s multiplied by
#: up to 32 pairs on EVERY cache hit (gen-4 rewrite: 920 s / $0.13).
SMOKE_TIMEOUT_S = 30

#: N10: pair budget for CACHE-HIT re-smokes (3 pairs ≈ 6 subprocesses);
#: fresh writes and repairs keep the full distinct-pair lattice.
SMOKE_MAX_PAIRS_CACHE_HIT = 3

MAIN_STANZA = '''

if __name__ == "__main__":
    import json as _json
    import sys as _sys
    payload = _json.load(_sys.stdin)
    try:
        verdict = compare(payload["A"], payload["B"])
    except Exception as _e:
        if _e.__class__.__name__ == "SchemaError":
            verdict = {"schema_error": str(_e), "winner": "tie"}
        else:
            raise
    print(_json.dumps(verdict))
'''

V3_PROMPT = """You are writing a DETERMINISTIC pairwise comparison policy for a \
workspace verifier. You get: (1) the verbatim task goal, (2) a reconnaissance \
TASK CONFIG, (3) the digest schema. You write ONE Python file: compare(dA, dB) \
-> dict. Hard requirements:

- PURE function; stdlib imports from {{json, math, statistics, re, itertools,
  collections, functools, copy, typing}} ONLY; .get()-safe everywhere;
  compare(d, d) -> tie; mirrored winners + swapped feedbacks under argument
  swap (verified at runtime).
- DECISION LADDER (decisive-rung floor included):
  R1 crash/timeout vs clean — decisive. If BOTH sides crash, compare
     error_signature families and partial evidence (environment/import vs
     data-shape vs timeout; produced_pred_results) — decisive if clearly
     different; tie only on identical signature with no partial evidence.
  R2 deliverable freshness — decisive ONLY as a contract check: a
     goal-named contract file or a primary-metric carrier that is
     missing or stale (produced_this_run=false / bytes=0 / absent) on
     ONE side while present-and-fresh on the other is a decisive
     violation. A raw COUNT difference of fresh deliverables (3 vs 2,
     produced_this_run true on both sides) is NOT evidence of quality —
     never decisive; mention it as a caveat note at most.
  R3 VALUE DETERMINANTS (when both sides are clean): id_order_match vs the
     workspace test input; prediction-distribution stats vs train-target stats
     (mean/std shifts); goal-stated dynamic-range coverage; degeneracy
     (n_unique) of value columns. Decisive when clearly different.
  R4 schema/columns/row-counts vs the goal contract.
  R5 primary metrics — ONLY through the pre-resolved METRIC LEDGER: each
     digest carries primary_values{{id: {{"status": "resolved"|"ambiguous"|
     "missing", "value": number|null, "source": "file:key"|null}}}} built
     from TASK CONFIG.primary_metrics by a deterministic resolver. Compare
     primary_values[<id>].value ONLY when BOTH sides have status ==
     "resolved", with an explicit caveat when the recon marks contexts
     incomparable. NEVER auto-tie merely because contexts differ in
     wording; same family (e.g. both report cross-validated MAE) =>
     compare with caveat.
  VALUE-RUNG CONSTRAINT (binding): rungs R3/R5 compare ONLY the METRIC
     LEDGER above. Rules:
     * use the metric's stated direction (higher/lower is better;
       "equal" metrics must match exactly).
     * metrics whose primary_metrics role is "invariant" or "report"
       are NEVER decisive evidence — cite their values as context notes
       at most.
     * a metric that is "missing" or "ambiguous" on either side is NOT
       comparable: say so in the feedback. NEVER search
       json_metrics/log_metrics for a similar-looking key yourself —
       cross-generation key drift is handled by the recon aliases in
       the ledger, not by the policy.
     * `deciding` must name the ledger ids you compared, as
       primary_values[<id>] entries taken from primary_metrics.
     Keys outside the ledger — booleans (json_flags), coordinate/
     conformation arrays (shape/len markers), degeneracy or count keys
     unless listed — are NEVER decisive evidence; mention them as
     caveat notes at most.
  If after R1-R5 nothing differs, tie.
- SCHEMA ERROR SIGNALLING (binding): define `class SchemaError(Exception): pass`
  at the top of your file. When you encounter a digest whose json_metrics,
  tables, or contract_files contain keys/paths you cannot interpret — keys
  that don't match any TASK CONFIG alias or known pattern — raise
  `SchemaError("brief description of what key/path is unrecognised")`.
  Do NOT silently tie on unknown schemas; the caller uses this signal to
  refresh its knowledge and re-evaluate. Only raise for genuinely novel
  structures, not for absent (missing/null/empty) values — those are
  handled by the ladder.
- FEEDBACK = MEASURED ASSESSMENTS ONLY (an auditor scores only stated
  measurements/comparisons; naming surfaces without assessing scores zero):
  * every line states a FACT WITH VALUES: "K_VRH mean 132.4 vs train 135.6
    (-2.4%)", "id order matches test input", "n_unique=157 of 292".
  * feedback_looser: the decisive deficiency + the goal contract restated
    ONCE in full sentences with the goal's own words for paths/columns.
  * feedback_winner: measured advantages only. Column/path names may be cited
    as contract ONLY if they appear VERBATIM in the goal text — EXACT token
    equality: singular/plural variants (name vs names) are NOT contract
    matches; otherwise label
    them "observed schema (not goal-verified)". NEVER bless an observed schema
    as goal-matching. NEVER use constants copied from peer workspaces as
    expected targets; provenance/parameter values may be compared only
    side-vs-side or against goal-stated values, never against absolute
    numbers from the config.
  * both values + delta for every compared number. No ellipses, no truncation.
  * TIE FEEDBACK (binding): when the verdict is tie because primary metrics
    are EQUAL, feedback_looser MUST state the tied values for every primary
    metric resolved on both sides: "minimum_energy -2 vs -2 (tied); raw SAW
    count 16268 vs 16268 (tied)". Metric comparisons are ALWAYS more
    important than file-presence details — lead with metrics, files second.
  * NO coverage lists, NO "not assessed" inventories, NO boilerplate — those
    live elsewhere and score zero here.
- Output shape (STRICT):
  {{"winner": "A"|"B"|"tie", "confidence": 0.0-1.0,
    "rungs": [{{"name": "...", "verdict": "...", "note": "<=20 words with values"}}],
    "deciding": ["digest.field.path ..."],
    "feedback_winner": "<=70 words of measured facts",
    "feedback_looser": "<=70 words of measured facts + contract"}}

# 1. TASK GOAL (verbatim)

{goal}

# 2. TASK CONFIG (reconnaissance; treat its column/path names as OBSERVED
unless verbatim in the goal above)

{config}
- exec: {{exit_code, wall_time_s, timed_out, error_signature, produced_pred_results, log_tail}}
- primary_values: {{metric_id: {{"status": "resolved"|"ambiguous"|"missing",
  "value": number|null, "source": "file:key"|null, "fresh": bool|null}}}} —
  the pre-resolved METRIC LEDGER for primary_metrics (the ONLY sanctioned
  source of metric values; see VALUE-RUNG CONSTRAINT)
- deliverables: [{{path, bytes, kind, produced_this_run}}]
- tables: [{{path, n_rows, columns, num_stats{{col: {{min,max,mean,std,n_nan,n_unique}}}},
  first_col_head, id_order_match}}]
- arrays: [{{path, shape, dtype, stats}}]; figures: [{{path, width_px, height_px, mode}}]
- contract_files: [{{path, present, bytes, sha256, head}}] — files the GOAL
  names verbatim; presence is judged from the workspace snapshot. A goal-named
  file with present=false is a contract violation (decisive at R4); present
  infra-ish files (run.sh) are contract evidence, not deliverables.
- json_metrics: {{filename: {{dotted.key: number | {{shape, len}} marker}}}};
  json_flags: {{filename: {{dotted.key: bool}}}} (flags are never metrics)
- inputs: {{train_csvs: {{path: {{col: stats}}}}, test_csvs: {{path: {{n_rows, columns}}}}}}
- train_used: {{path, n_rows}}|null; uuid: str (identifier only)

Return STRICT JSON: {{"policy_py": "<complete Python source>"}}
"""

#: Prompt-hash policy version: verdict caches key on this so a prompt
#: improvement invalidates old verdicts cleanly (operator decision 3).
POLICY_SHA = hashlib.sha256(
    (V3_PROMPT + RECON_PROMPT_VERSION).encode("utf-8")
).hexdigest()[:12]

#: Screen-failing sentinel returned when the LLM payload is unusable
#: (truncated/unparseable JSON must NEVER become policy source — E48).
TRUNCATED_SENTINEL = "raise SyntaxError('policy JSON truncated or unparseable')"


def static_screen(src: str) -> list[str]:
    """AST screen: compare() present, imports allowlisted, names banned.

    N12/P2.11 additions: reflective calls (``getattr``/``setattr``/
    ``delattr``) and ANY ``__dunder__`` name or attribute are banned —
    they are the standard escape hatch around a static banlist.

    Args:
        src: Candidate policy source.

    Returns:
        Empty list when clean; one message per violation otherwise.
    """
    errs: list[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    if not any(
        isinstance(n, ast.FunctionDef) and n.name == "compare"
        for n in ast.walk(tree)
    ):
        errs.append("missing def compare(dA, dB)")
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                root = a.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    errs.append(f"banned import: {a.name}")
        elif isinstance(n, ast.ImportFrom):
            root = (n.module or "").split(".")[0]
            if root and root not in ALLOWED_IMPORTS:
                errs.append(f"banned import-from: {n.module}")
        elif (
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Load)
            and n.id in BANNED_NAMES
        ):
            errs.append(f"banned name: {n.id}")
        elif (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in REFLECTIVE_CALLS
        ):
            errs.append(f"banned reflective call: {n.func.id}")
        elif isinstance(n, ast.Name) and _is_dunder(n.id):
            errs.append(f"banned dunder name: {n.id}")
        elif isinstance(n, ast.Attribute) and _is_dunder(n.attr):
            errs.append(f"banned dunder attribute: {n.attr}")
    return errs


def write_prompt(goal: str, task_config: dict[str, Any], repair: str | None = None) -> str:
    """Assemble the policy-writer prompt (optionally with repair errors)."""
    prompt = V3_PROMPT.format(
        goal=goal, config=json.dumps(task_config or {}, indent=1, sort_keys=True)
    )
    if repair:
        prompt += (
            f"\n\n# REPAIR REQUIRED — your previous attempt failed "
            f"the screen or the runtime smoke:\n{repair}\nReturn the FULL "
            f"corrected file; do NOT import anything outside the allowlist "
            f"(remove any import of sys entirely; never reference sys)."
        )
    return prompt


def policy_source(parsed: Any) -> str:
    """Policy source from an LLM payload, or the screen-failing sentinel.

    A non-dict payload or a dict without a ``policy_py`` string is NEVER
    used as code (E48 truncation regression): the sentinel fails the AST
    screen and drives the repair loop instead.
    """
    if isinstance(parsed, dict):
        src = parsed.get("policy_py")
        if isinstance(src, str) and src.strip():
            return src
    return TRUNCATED_SENTINEL


def run_policy(
    policy_src: str, dA: dict[str, Any], dB: dict[str, Any], timeout: int = 60
) -> dict[str, Any]:
    """Execute ``compare(dA, dB)`` in a subprocess; verdict dict or error.

    N12/P2.11: the subprocess runs with PATH ONLY in its environment (no
    API keys or operator secrets) inside a FRESH empty temporary cwd.

    Args:
        policy_src: Policy source INCLUDING the ``__main__`` stanza.
        dA: Digest of candidate A.
        dB: Digest of candidate B.
        timeout: Subprocess timeout seconds.

    Returns:
        The policy's verdict dict (plus ``wall_ms``), or
        ``{"error": ..., "wall_ms": ...}`` on crash/timeout/bad output.
    """
    t0 = time.monotonic()
    with tempfile.NamedTemporaryFile(
        "w", suffix="_policy_bt.py", delete=False, encoding="utf-8"
    ) as f:
        f.write(policy_src)
        path = f.name
    try:
        with tempfile.TemporaryDirectory(prefix="policy_bt_run_") as run_cwd:
            r = subprocess.run(
                [sys.executable, path],
                cwd=run_cwd,
                env={"PATH": os.environ.get("PATH", os.defpath)},
                input=json.dumps({"A": dA, "B": dB}),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        wall = round((time.monotonic() - t0) * 1000)
        if r.returncode != 0:
            tail = (r.stderr or "").strip().splitlines()
            err_line = tail[-1][:200] if tail else ""
            return {
                "error": (
                    f"exit {r.returncode}: {err_line} | {r.stderr[-300:]}"
                ),
                "wall_ms": wall,
            }
        try:
            v = json.loads(r.stdout)
        except json.JSONDecodeError as e:
            return {
                "error": f"bad json: {e}",
                "wall_ms": wall,
            }
        if not isinstance(v, dict):
            return {
                "error": f"non-dict verdict: {type(v).__name__}",
                "wall_ms": wall,
            }
        if v.get("schema_error"):
            # Schema mismatch signalled by the policy — distinct from a
            # crash; the evaluator uses this to trigger a recon refresh.
            v["wall_ms"] = wall
            return v
        if v.get("winner") not in {"A", "B", "tie"}:
            return {"error": f"bad winner: {v.get('winner')!r}", "wall_ms": wall}
        v["wall_ms"] = wall
        return v
    except subprocess.TimeoutExpired:
        return {
            "error": "timeout",
            "wall_ms": round((time.monotonic() - t0) * 1000),
        }
    finally:
        with contextlib.suppress(OSError):
            Path(path).unlink(missing_ok=True)


def mirrored(v_ab: dict[str, Any], v_ba: dict[str, Any]) -> bool:
    """True when the two orderings agree on the mirrored winner (E14)."""
    w_ab, w_ba = v_ab.get("winner"), v_ba.get("winner")
    return (w_ab == "tie" and w_ba == "tie") or (
        w_ab == "A" and w_ba == "B"
    ) or (w_ab == "B" and w_ba == "A")


def smoke_test(
    policy_src: str,
    digests: list[dict[str, Any]],
    timeout: int = SMOKE_TIMEOUT_S,
    run_fn: Callable[[str, dict, dict], dict] | None = None,
    max_pairs: int = 0,
) -> tuple[bool, str | None]:
    """Runtime smoke: distinct digest pairs, both orders (cap 6 digests).

    A policy that only fails on ONE digest shape (HP gen 4: KeyError on
    every gen 1-3 pair while the first pair self-compared fine) must
    fail the smoke before it is ever used. The loop STOPS at the first
    failure (N10); *max_pairs* > 0 caps how many pairs run — cache-hit
    re-smokes pass ``3`` so the full lattice is only paid on fresh
    writes and repairs.

    Args:
        policy_src: Candidate policy source (with ``__main__`` stanza).
        digests: Real digests of this task's generations (>=1; a single
            digest self-compares).
        timeout: Per-call subprocess timeout (N10: default 30 s).
        run_fn: Execution override (tests); defaults to :func:`run_policy`.
        max_pairs: Cap on executed pairs (0 = no cap); N10.

    Returns:
        ``(True, None)`` when the smoke passes, else ``(False, reason)``.
    """
    run = run_fn or run_policy
    seen: set[str] = set()
    ds: list[dict[str, Any]] = []
    for d in digests:
        if not isinstance(d, dict):
            continue
        key = json.dumps(d, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        ds.append(d)
        if len(ds) >= SMOKE_MAX_DIGESTS:
            break
    if not ds:
        ds = [{}]
    pairs: list[tuple[dict[str, Any], dict[str, Any], str]] = [
        (ds[i], ds[j], f"pair {i}x{j}")
        for i in range(len(ds))
        for j in range(i + 1, len(ds))
    ] + [(ds[0], ds[0], "self-pair")]
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    for dA, dB, label in pairs:
        v_ab = run(policy_src, dA, dB)
        if "error" in v_ab:
            return (
                False,
                f"runtime failure on a real digest pair ({label}, A->B): "
                f"{v_ab['error'][:400]}",
            )
        v_ba = run(policy_src, dB, dA)
        if "error" in v_ba:
            return (
                False,
                f"runtime failure on a real digest pair ({label}, B->A): "
                f"{v_ba['error'][:400]}",
            )
        if dA is dB:
            if v_ab.get("winner") != "tie":
                return (
                    False,
                    f"self-comparison must tie: compare(d, d) -> "
                    f"{v_ab.get('winner')!r}",
                )
            continue
        if not mirrored(v_ab, v_ba):
            return (
                False,
                f"antisymmetry violation on {label}: compare(dA,dB)->"
                f"{v_ab.get('winner')} but compare(dB,dA)->{v_ba.get('winner')}",
            )
    return True, None


def ensure_policy(
    goal: str,
    task_config: dict[str, Any],
    smoke_digests: list[dict[str, Any]],
    llm_json: Callable[[str, int, str], Any],
    repair_rounds: int = 3,
    smoke_timeout_s: int = SMOKE_TIMEOUT_S,
    log: Any = None,
) -> tuple[str | None, dict[str, Any]]:
    """Write -> screen -> smoke -> repair a policy until it passes.

    Implements the full E48 loop: token-cap escalation (9000 -> 12000) on
    an unparseable payload, sentinel rejection of truncated JSON, exact
    error feedback on every repair, and a cache-busting role per attempt
    so the LLM helper can never re-serve the cached broken artifact.

    Args:
        goal: Verbatim task goal.
        task_config: Recon TASK CONFIG for the task.
        smoke_digests: Real digests used by the runtime smoke.
        llm_json: ``(prompt, max_tokens, role) -> parsed JSON or None``.
        repair_rounds: Max repair attempts after the first write.
        smoke_timeout_s: Per-call smoke subprocess timeout.
        log: Optional StageLogger (verbose per-repair lines).

    Returns:
        ``(policy_source_with_stanza, metadata)`` on success, or
        ``(None, metadata)`` when every attempt failed (metadata carries
        the last error; the caller degrades loudly).
    """
    meta: dict[str, Any] = {
        "version": POLICY_SHA,
        "attempts": 0,
        "repairs": 0,
        "escalations": 0,
        "smoke": "pending",
        "last_error": None,
    }
    repair_note: str | None = None
    attempt = 0
    max_attempts = 1 + max(0, repair_rounds)
    while attempt < max_attempts:
        attempt += 1
        meta["attempts"] = attempt
        role = "policy_write_v3" if attempt == 1 else f"policy_write_fix{attempt}"
        prompt = write_prompt(goal, task_config, repair=repair_note)
        parsed = llm_json(prompt, MAX_TOKENS_FIRST, role)
        src = policy_source(parsed)
        if src == TRUNCATED_SENTINEL and attempt < max_attempts:
            # Safeguard 1: escalate the token budget once before burning a
            # repair round — the E48 regression was a length truncation.
            meta["escalations"] += 1
            role = f"policy_write_fix{attempt}_{MAX_TOKENS_ESCALATED}"
            esc_prompt = (
                write_prompt(goal, task_config, repair=repair_note)
                + "\n\nYour previous response was NOT valid JSON with a "
                'policy_py string (truncated or unparseable). Return STRICT '
                'JSON: {"policy_py": "<complete Python source>"} — nothing else.'
            )
            parsed = llm_json(esc_prompt, MAX_TOKENS_ESCALATED, role)
            src = policy_source(parsed)
        errs = static_screen(src)
        if errs:
            repair_note = "; ".join(errs[:5])
            meta["last_error"] = f"screen: {repair_note}"
            meta["repairs"] += 1
            if log is not None:
                log.detail(f"policy attempt {attempt}: screen FAIL: {repair_note}")
            continue
        full_src = src + MAIN_STANZA
        ok, err = smoke_test(full_src, smoke_digests, timeout=smoke_timeout_s)
        if ok:
            meta["smoke"] = "pass"
            meta["last_error"] = None
            return full_src, meta
        meta["smoke"] = "fail"
        meta["repairs"] += 1
        repair_note = (
            f"{err}. Fix the bug; keep the same contract; do not hardcode "
            f"peer-workspace constants."
        )
        meta["last_error"] = repair_note
        if log is not None:
            log.detail(f"policy attempt {attempt}: smoke FAIL: {str(err)[:200]}")
    return None, meta

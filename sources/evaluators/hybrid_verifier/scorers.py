"""Deterministic per-claim scorer generation, screening and execution.

Ports the measured E19b V4 scorer contract from
``experiments_verifiers/harness/e19b_extract.py``
(``SCORER_TEMPLATE_V2`` / ``REPAIR_TEMPLATE_V2``) and the execution
guards of ``e18_scripted_bt.py`` / ``e19_hybrid_bt.py`` (static policy
screening, single-JSON-line output parsing). Execution itself runs
through the repo's pinned-subprocess pattern (``WorkflowRunner`` with
``python_executable=sys.executable``, cwd = the workspace being scored).

Contract adaptation (documented deviation from the harness): the repo
runner passes no argv, so the workspace root and the claim are injected
as predefined module constants (``WORKSPACE_ROOT``, ``CLAIM``) in a
launcher prefix instead of ``argv[1]`` / ``argv[2]``.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .claims import _flag, _screen_call

# Allowed imports for scorer scripts: stdlib + numpy + pandas + PIL (E18 set).
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "json",
        "sys",
        "os",
        "re",
        "math",
        "csv",
        "io",
        "pathlib",
        "collections",
        "itertools",
        "statistics",
        "warnings",
        "datetime",
        "time",
        "typing",
        "functools",
        "string",
        "base64",
        "hashlib",
        "struct",
        "textwrap",
        "ast",
        "glob",
        "fnmatch",
        "operator",
        "heapq",
        "bisect",
        "copy",
        "numpy",
        "pandas",
        "PIL",
    }
)

# Network / process / randomness / destruction APIs (E18 BANNED_SNIPPETS).
_BANNED_SNIPPETS: tuple[str, ...] = (
    "subprocess",
    "socket",
    "urllib",
    "requests",
    "shutil",
    "os.system",
    "popen",
    "rmtree",
    "os.remove",
    "os.unlink",
    "unlink(",
    "os.rmdir",
    "random.",
    "numpy.random",
)

# Write APIs invisible to import screening (E19 EXTRA_BANNED; note the
# pilot-1 lesson — bare "json.dump" would also ban json.dumps, so the two
# dumper calls are matched with an explicit open-paren).
_EXTRA_BANNED: tuple[str, ...] = (
    "to_csv",
    "to_excel",
    "to_pickle",
    "to_parquet",
    "to_hdf",
    "to_sql",
    "savefig",
    "np.save",
    "numpy.save",
    "write_text",
    "write_bytes",
    "os.mkdir",
    "os.makedirs",
    "Path.mkdir",
)
_EXTRA_BANNED_RE: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bjson\.dump\s*\("),
    re.compile(r"\bpickle\.dump\s*\("),
)

_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", re.M)
_WRITE_MODE_OPEN_RE = re.compile(r"open\([^)]*['\"][waxb+]['\"]")

MAX_CODE_ATTEMPTS = 3  # initial + 2 repairs (E18 protocol 4bis)


@dataclass
class ExecOutcome:
    """Result of one scorer-script execution."""

    rc: int
    stdout: str = ""
    stderr: str = ""
    timeout: bool = False


@dataclass
class ScorerAttempt:
    """Telemetry for one generate/repair attempt on a scorer."""

    attempt: int
    parse: str | None = None
    static_violations: list[str] = field(default_factory=list)
    executed: bool = False
    ok: bool = False
    feedback: str = ""


def scorer_prompt(
    goal: str,
    claim: dict[str, Any],
    union_inventory: str,
    digests_block: str = "",
) -> str:
    """Build the policy-script prompt for one ladder rung (claim).

    The per-claim script IS the policy for its rung on the temporal
    ladder; the prompt frames that role, injects the cached format
    digests (real file FORMATS, not just names) and cites the operator's
    policy examples.
    """
    stage = claim.get("stage", "result")
    digest_section = (
        "# Format digests of this task's deliverable files (deterministic samples; "
        "use these FORMATS, not guesses)\n" + digests_block + "\n"
        if digests_block
        else ""
    )
    return f"""You are writing ONE self-contained deterministic Python script that \
scores ONE candidate agent workspace on ONE verification claim. The SAME script runs \
unchanged against every candidate workspace of the task, so it must locate files by \
pattern, never hardcode one candidate's layout.

This claim is a RUNG on a TEMPORAL LADDER of stages: script (delivered code exists, \
is structurally valid, reads the goal's inputs, implements the intended pipeline) -> \
log (execution/training dynamics parsed from logs) -> result (artifact fidelity). \
This claim's rung is stage "{stage}". Your measured score decides ELIMINATION at \
stage {stage}: a candidate that fails a lower rung loses regardless of later rungs, \
and policy examples are — lower loss in the log wins; better accuracy at training \
step X wins; a prediction CSV whose columns match the ORIGINAL file's column names \
exactly (no useless `_pred` suffix added) wins; more complete results (one row per \
input row) win.

# Task goal (context only)
{goal[:2500]}

# Claim to implement exactly (JSON)
{json.dumps(claim, indent=1)}

# File paths that appear across this task's candidate workspaces — RELATIVE TO THE \
WORKSPACE ROOT (a given workspace may lack some; the claim's convention is: target \
absent -> score 0)
{union_inventory}
{digest_section}# Script contract — MUST follow exactly
- The script runs with TWO predefined constants: WORKSPACE_ROOT (the absolute path \
string of the workspace root; the process working directory is also that root) and \
CLAIM (the claim as a dict, same as above). Do not redefine them.
- Locate the claim's target file(s) under WORKSPACE_ROOT (os.walk / glob / fnmatch). \
Open files READ-ONLY. Never write, never use network/subprocess/randomness — fully \
deterministic.
- Allowed imports: Python standard library + numpy + pandas + PIL ONLY.
- MEASURE A QUANTITY, do not just detect presence: count rows/items, compute the \
fraction satisfying the check, read values and compare to the plausible range, parse \
log lines for loss/accuracy at a step, or read the script's code for the required \
method — then map that measured quantity to a CONTINUOUS 0..1 score implementing the \
scoring_rule EXACTLY (partial credit per the rule). Do NOT collapse to 0/1 unless the \
property is truly binary. Absent/unreadable target -> 0.
- When the claim names EXACT column names, header fields, file names or output paths, \
your script MUST verify them with EXACT equality — set equality such as \
set(df.columns) == {{"gene_id", "expression_value", "p_value"}} or a direct == \
comparison. Never use substring matching (`"X" in c`), case-insensitive matching, or \
partial names for an exact-name requirement: a column named PRED_score does NOT \
satisfy a requirement for PRED, and an exact-set requirement also fails when extra \
or renamed columns are present. Fuzzy/substring matching is ONLY for LOCATING a \
candidate file or column, never for verifying an exact-name claim.
- Print EXACTLY ONE line to stdout: a single JSON object
  {{"claim_id": "<claim id>", "score": <number 0..1>, "evidence": "<=40 words quoting \
the measured numbers (counts, values, ranges) you actually read"}}
- Exit with code 0. Diagnostics go to stderr, never stdout. Handle malformed/short \
files gracefully (score them down per the rule, do not crash).

Output ONLY the Python source code inside one ```python fenced block."""


def repair_prompt(feedback: str, prev_code: str) -> str:
    """Build the scorer repair prompt (execution-failure feedback)."""
    return f"""Your previous scoring script failed. Fix it.

# Failure report (last output lines)
{feedback[:4000]}

# Previous script
{prev_code}

# Reminder of the contract
- Predefined constants: WORKSPACE_ROOT (absolute workspace root path string) and CLAIM \
(the claim dict); the working directory is the workspace root.
- Locate targets by pattern under WORKSPACE_ROOT; open READ-ONLY; stdlib + numpy + \
pandas + PIL only; no network/writes/randomness; deterministic.
- MEASURE the quantity the scoring_rule names and map it to continuous partial credit \
on 0..1 (never a bare 0/1 unless truly binary); absent target -> 0.
- Print EXACTLY ONE JSON line: {{"claim_id": "...", "score": <0..1>, "evidence": \
"<=40 words with the measured numbers"}}; exit 0.
- Common pitfalls: workspace lacks the target entirely (must print score 0, not \
crash); CSV delimiter/encoding variance (try engine fallbacks, dtype=str first); \
column names vary across workspaces (locate a column fuzzily if you must, but when \
the claim demands exact column/file names verify them with EXACT set equality — \
never substring or case-insensitive matching, PRED_score is NOT PRED); numbers \
embedded in logs (regex the LAST occurrence); NaN handling; printing anything else \
to stdout.

Output ONLY the corrected Python source code in one ```python fenced block."""


def parse_code(text: str) -> str | None:
    """Longest fenced Python block in *text*; long bare text accepted."""
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text or "", flags=re.S)
    if m:
        return max(m, key=len).strip()
    t = (text or "").strip()
    return t if t and len(t) > 80 else None


def _pair_block(
    claims: list[dict[str, Any]], scripts: dict[str, str]
) -> list[dict[str, str]]:
    """Claim/script pairs as one strict-JSON line for a screen prompt."""
    out: list[dict[str, str]] = []
    for c in claims:
        cid = str(c.get("id", ""))
        code = scripts.get(cid)
        if not cid or not code:
            continue
        out.append(
            {
                "claim_id": cid,
                "statement": str(c.get("statement", "")),
                "target": str(c.get("target", "")),
                "scoring_rule": str(c.get("scoring_rule", "")),
                "script": code,
            }
        )
    return out


def exact_name_violations(
    claims: list[dict[str, Any]],
    scripts: dict[str, str],
    llm_text: Callable[[str, str, str], str],
    uuid: str = "",
) -> dict[str, str]:
    """ONE batched LLM judgment: which exact-name scorers match leniently.

    A claim that specifies exact column names, header fields, file names
    or output paths must be verified by its scorer script with EXACT
    equality — set equality (``set(df.columns) == expected_set``) or a
    direct ``==`` comparison. Lenient matching (substring ``in``,
    ``.str.contains()``, a ``.lower()`` before the comparison, partial
    names) lets a near-miss name such as ``PRED_score`` satisfy a
    requirement for ``PRED`` and scores the wrong artifact 1.0. All of
    the task's claim/script pairs go to the judge in a single call
    (batched, never per-claim), reusing the claims screens' retry
    transport. Returns ``{claim_id: reason}`` naming only the lenient
    scripts; verdicts on ids outside the screened pairs are ignored, and
    a failed judge call degrades open (nothing flagged).
    """
    pairs = _pair_block(claims, scripts)
    if not pairs:
        return {}
    prompt = f"""# EXACT-NAME SCORER SCREEN
You are auditing the scorer scripts of ONE scientific verification rubric. Each claim \
below specifies exact column names, header fields, file names or output paths, and is \
verified by ONE Python script. EXACT equality means the script compares names with \
`set(df.columns) == expected_set`, `col == "name"` or an equivalent direct equality \
(including each name inside a set/list literal compared with ==). LENIENT matching \
means substring membership (`"X" in c`, `any(n in col for col in cols)`), \
`.str.contains("X")`, a `.lower()` call before the comparison, or any other partial \
or case-insensitive name test. Lenient matching lets near-miss names pass — a column \
named PRED_score does NOT satisfy a requirement for PRED — so a script that verifies \
exact names leniently is defective even if it also checks other things loosely.

CLAIMS + SCRIPTS (JSON):
{json.dumps(pairs)}

For EACH pair above, judge: does the script verify the claim's exact names with EXACT \
equality, or does it use lenient matching? Reply with ONLY a strict JSON array, \
exactly one entry per pair and nothing else:
[{{"claim_id": "<claim id>", "lenient": true, "line_evidence": "<the specific \
script line>", "reason": "<one sentence>"}}]
(entries for exact scripts use "lenient": false)."""
    verdicts = _screen_call(llm_text, uuid, "hybrid_screen_exact_name", prompt)
    ids = {p["claim_id"] for p in pairs}
    flagged: dict[str, str] = {}
    for v in verdicts or []:
        if not isinstance(v, dict):
            continue
        cid = str(v.get("claim_id", ""))
        if cid not in ids or not _flag(v, "lenient"):
            continue
        reason = str(v.get("reason") or "lenient name matching").strip()
        ev = str(v.get("line_evidence") or "").strip()
        flagged[cid] = f"{reason} — line: {ev}" if ev else reason
    return flagged


def static_violations(code: str) -> list[str]:
    """Static STRUCTURAL policy screen for a scorer script (E18+E19).

    Allowed imports, banned APIs, write-mode opens — pure structure, no
    content judgment. The exact-name content screen is LLM-powered
    (``exact_name_violations``) and runs on the host's judge call in the
    scorer build loop.
    """
    v: list[str] = []
    for mod in _IMPORT_RE.findall(code):
        root = mod.split(".")[0]
        if root not in ALLOWED_IMPORTS:
            v.append(f"import not allowed: {mod}")
    for pat in _BANNED_SNIPPETS + _EXTRA_BANNED:
        if pat in code:
            v.append(f"banned pattern: {pat}")
    for rx in _EXTRA_BANNED_RE:
        if rx.search(code):
            v.append(f"banned pattern: {rx.pattern}")
    if "open(" in code and _WRITE_MODE_OPEN_RE.search(code):
        v.append("write-mode open() detected")
    return v


def build_launcher(workspace: Path, claim: dict[str, Any], body: str) -> str:
    """Wrap a scorer body with the WORKSPACE_ROOT / CLAIM constants."""
    ws_lit = json.dumps(str(workspace))
    claim_lit = json.dumps(json.dumps(claim))
    header = (
        "# hybrid-verifier scorer launcher (deterministic, read-only)\n"
        f"import json as _json\n"
        f"WORKSPACE_ROOT = {ws_lit}\n"
        f"CLAIM = _json.loads({claim_lit})\n"
        "# --- scorer body follows ---\n"
    )
    return header + body


def parse_scorer_output(stdout: str) -> dict[str, Any] | None:
    """Parse the one-JSON-line scorer contract; None when violated.

    Accepts the last stdout line or the whole stdout as the payload,
    mirroring E19's ``parse_scorer_output``; scores are clamped to [0, 1]
    and non-finite values are rejected.
    """
    lines = [ln for ln in (stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None
    for text in (lines[-1], "\n".join(lines)):
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        s = d.get("score")
        if isinstance(s, bool) or not isinstance(s, int | float):
            return None
        s = float(s)
        if not math.isfinite(s):
            return None
        return {
            "claim_id": str(d.get("claim_id", ""))[:60],
            "score": min(1.0, max(0.0, s)),
            "evidence": str(d.get("evidence", ""))[:300],
        }
    return None


def feedback_lines(outcome: ExecOutcome, timeout_s: int) -> str:
    """Render repair feedback: exit status + last ~30 output lines (E19)."""
    lines = (outcome.stderr + "\n" + outcome.stdout).splitlines()
    tail = [ln for ln in lines if ln.strip()][-30:]
    extra = (
        f"[timeout after {timeout_s}s]"
        if outcome.timeout
        else f"[exit code {outcome.rc}]"
    )
    return extra + "\n" + "\n".join(tail)

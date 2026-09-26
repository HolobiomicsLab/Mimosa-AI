"""Temporal-ladder key-claim extraction, validation and refinement (E19b-v2 + operator temporal spec).

Claims form a LADDER of stages in temporal order — ``script`` → ``log`` →
``result``: a generation must first deliver structurally valid code that
reads the goal's inputs (script stage), then show healthy execution
dynamics in its logs (log stage), before its result artifacts are scored
(result stage). Aggregation resolves pairwise winners at the EARLIEST
stage where two generations differ (see ``aggregation``), so a
script-stage failure dominates any result-stage advantage.

Ports the E19b-v2 prompt rules (task-defining / decidable / decisive /
graded, ``experiments_verifiers/harness/e19b_extract.py``) plus the
operator's temporal-ladder claim spec: pred-CSV schema must match the
ORIGINAL file's column names exactly (no spurious ``_pred`` suffixes),
log claims parse training dynamics, script claims check the delivered
code itself.
"""

from __future__ import annotations

import collections
import json
import re
from collections.abc import Callable
from typing import Any

from sources.evaluators.base import extract_json_payload
CLAIM_CATEGORIES: tuple[str, ...] = (
    "deliverable_path",
    "output_schema",
    "prediction_sanity",
    "metric_quality",
    "completeness",
    "method_implementation",
    # E30 content lever: the two claim families the E16 audit found most
    # under-covered (online 0.742 -> 0.759 when mandated).
    "core_computation",
    "method_identity",
    "other",
)

# Temporal ladder stages, earliest first. A claim's stage fixes its rung.
# STAGES is the standard text ladder; FIGURE_STAGES inserts the E26 visual
# rung FIRST for figure tasks (visual-only 0.833 vs 0.646 as 4th rung).
STAGES: tuple[str, ...] = ("script", "log", "result")
FIGURE_STAGES: tuple[str, ...] = ("visual", "script", "log", "result")
STAGE_RANK: dict[str, int] = {s: i for i, s in enumerate(FIGURE_STAGES)}

# E30 content mandate: >=2 core_computation + >=2 method_identity claims.
LEVER_MIN_COUNTS: dict[str, int] = {"core_computation": 2, "method_identity": 2}
# E16 key_missing fidelity families the extraction prompt also demands.
FIDELITY_CATEGORIES: tuple[str, ...] = (
    "output_schema",
    "prediction_sanity",
    "deliverable_path",
)

# Claim fields every extracted/replacement claim must carry (non-empty str).
_REQUIRED_FIELDS: tuple[str, ...] = ("statement", "target", "scoring_rule")

CUT


def extract_claims_prompt(
    goal: str,
    union_inventory: str,
    previews: str,
    lo: int,
    hi: int,
) -> str:
    """Build the temporal-ladder claim-extraction prompt for one task."""
    return f"""You are designing a DECISIVE, TEMPORALLY-ORDERED verification rubric for ONE \
scientific task solved by MANY candidate agents. You define the KEY CLAIMS that separate a \
CORRECT workspace from a PLAUSIBLE-BUT-WRONG one — properties a deterministic Python script \
can grade on ANY single workspace on a 0..1 scale with GRADED partial credit.

The claims form a TEMPORAL LADDER of three stages, in this order:
- "script" (earliest): the delivered CODE exists, is structurally valid, \
reads the goal's input files correctly, and implements the intended \
pipeline (checkable by READING the candidate's own scripts).
- "log" (middle): execution / training dynamics parsed from the run's \
logs — loss decreasing, best/final loss value, accuracy at training step \
X, convergence, no NaN or crash markers.
- "result" (latest): artifact fidelity — the prediction CSV's schema \
matches the ORIGINAL input file's column names EXACTLY (no spurious \
`_pred` suffixes added to columns), one row per input row (completeness), \
prediction sanity (probabilities in [0,1], non-negative counts).

# Task goal
{goal[:4000]}

# Union of the candidate workspaces' file inventories — paths RELATIVE TO EACH WORKSPACE \
ROOT (size; CSV rows x cols / PNG dims; how many workspaces contain each path)
{union_inventory}
# Previews of up to 3 deliverable-ish files from the most complete workspace
{previews or "(no deliverable-ish files found)"}

# Hard rules for EVERY claim (all four required)
1. TASK-DEFINING: only properties the goal's OWN words specify or imply — deliverable \
paths/file kinds it names, schema/columns it names, counts (one row per input item), \
value sanity it implies, the method/model/featurization it explicitly requires, \
completeness vs the provided input. NEVER invent deliverables, metrics or file names the \
goal does not mention: a demanded-but-unmeasured artifact is a hallucinated requirement.
2. DECIDABLE: gradeable by deterministic Python from THIS workspace's files ALONE (no \
gold outputs, no reference labels, no network). You may grade the candidate's SCRIPT \
contents (imports, class names, featurization, train/predict structure) for script-stage \
claims, and parse log files for log-stage claims, as well as output artifacts for \
result-stage claims.
3. DECISIVE: a claim that EVERY workspace will likely pass — or that none can pass — is \
WORTHLESS; it will be dropped and wasted. Prefer checks that separate a correct from a \
plausible-but-wrong deliverable: exact required row counts, column sets matching the \
ORIGINAL input column names (no `_pred` suffix inventions), fraction of values in a \
plausible numeric range, presence of the required method in code, decreasing loss across \
epochs in the log. Ask: "how would a lazy agent fake the deliverable, and what would \
STILL catch it?"
4. GRADED: the scoring rule must MEASURE A QUANTITY (a count, a fraction of rows/items \
satisfying the check, a loss/accuracy value at a parsed step, a distance from the expected \
count) and map it to continuous partial credit on 0..1 with the mapping stated. NEVER a \
bare boolean unless the property is truly binary. State the measured quantity explicitly.

# T1 FIREWALL — workspace-verifiability (hard constraint, E29)
Claims must reference WORKSPACE-VERIFIABLE properties only — what files should exist, \
what their schemas should contain, what values should be in range. NEVER encode a \
specific numeric answer, file hash, or computed result as a claim requirement. If you \
find yourself wanting to assert "the answer is X", instead assert "the output file must \
contain a value consistent with the method described in the goal". A claim that \
hard-codes a specific answer value is unfalsifiable from the workflow side and will be \
rejected. Concretely: no claim may demand a numeric value, count, score or threshold \
constant that the goal text itself does not state (a range like [0,1] for probabilities \
is fine; "the minimum energy must be -2" when the goal never says -2 is NOT).

# SELF-CHECK — judge every claim you emit, and output that judgment
For each claim, also output a "self_check" field on your OWN claim:
"self_check": {{"generic": false, "smuggled_answer": false, "reason": "<one sentence>"}}
- "generic": true when ANY workspace would trivially satisfy the claim — bare file \
existence, "the script runs", workspace tidiness; it measures nothing and separates nobody.
- "smuggled_answer": true when the claim encodes a specific computed result (a number, \
hash, or output value) that the goal does not state and that a workflow cannot verify \
without already knowing the answer.
NEVER emit a claim whose self_check flags it — REWRITE the claim first; a flagged claim \
is your own admission that it is worthless or unfalsifiable.

# CONTENT PRIORITIES (E16 audit mandate — the under-covered claim families, E30)
An audit of verifier feedback on this benchmark found the graded properties most often \
IGNORED are (a) the task's CORE COMPUTATIONAL CHAIN, (b) the IDENTITY of the method \
the goal names, and (c) the deliverable's schema / value sanity / path. In addition to \
any other checks, your rubric MUST include:
- AT LEAST 2 claims of category "core_computation": the delivered code reproduces the \
goal's computation END-TO-END — named inputs -> core transformation with goal-stated \
parameters -> deliverable produced from it. Every stage the goal implies, in order: \
reading the goal's named input files/data, performing the core transformation / \
modeling / analysis the goal demands (with the parameters, thresholds, subsets or \
options it states), and producing the goal's named deliverable FROM that computation. \
Grade the FRACTION of required chain elements that are present in the delivered code \
AND mutually consistent (later stages consume earlier ones); every element name comes \
only from the goal's own words.
- AT LEAST 2 claims of category "method_identity": the code invokes EXACTLY the method, \
model, featurizer, solver, metric or library routine the goal names or uniquely implies, \
applied to the data the goal names — a lookalike substitute must score 0; a bare import \
is not identity. Establish identity from the class/function/algorithm signature in the \
delivered code. Grade the FRACTION of identity elements confirmed in the code. The >=2 \
mandate is UNCONDITIONAL: when the goal names one headline method, split its identity \
into distinct checkable elements — e.g. the exact model/class instantiated with its \
goal-stated parameters, the exact featurizer/preprocessing applied to the goal-named \
input, and the exact train/fit->predict call structure consuming it — one claim each. \
Two claims may not be near-duplicates; each must check a DIFFERENT identity element.
- AT LEAST ONE claim in EACH of the categories "output_schema", "prediction_sanity" and \
"deliverable_path" where the goal's deliverable is a data file whose columns / values / \
location the goal names or clearly implies (e.g. prediction CSV columns must match the \
input's names exactly; probabilities must lie in [0,1]; the deliverable must exist at \
the goal-named path). If the goal truly implies none of the three, say so with claims \
in the other categories instead.
These claims obey the same four hard rules AND the T1 firewall (task-defining, \
decidable, decisive, graded, workspace-verifiable).

Output EXACTLY {lo} to {hi} claims (as few as carry the task; prefer the middle of the \
range; shallow checks die, depth is the point). Composition: at least 2 \
"core_computation" AND at least 2 "method_identity"; at least one "script" claim; at \
least one "log" claim (when the task implies training/execution logs); the remaining \
claims on result-stage deliverable fidelity. Cover ALL THREE stages. Strict JSON only:
{{
 "claims": [
  {{
   "id": "C1",
   "stage": "script" | "log" | "result",
   "temporal_index": 1,
   "category": "deliverable_path | output_schema | prediction_sanity | metric_quality | \
completeness | method_implementation | core_computation | method_identity | other",
   "statement": "<one sentence, task-defining property>",
   "target": "<which workspace files this claim applies to>",
   "scoring_rule": "<what quantity to measure on ONE workspace and how it maps to 0..1 \
partial credit, with the absent-target convention (absent -> 0)>",
   "self_check": {{"generic": false, "smuggled_answer": false, "reason": "<one sentence>"}}
  }}
 ]
}}"""


def refinement_prompt(
    goal: str,
    union_inventory: str,
    outcome_report: str,
    dead_block: str,
    alive_block: str,
    n_new: int,
    next_id: str,
) -> str:
    """Build the temporal-ladder refinement prompt (replacement claims)."""
    return f"""You are refining a DECISIVE verification rubric for ONE scientific \
task solved by MANY candidate agents. The current claims were scored on every candidate \
workspace; some turned out NON-DISCRIMINATIVE (all workspaces equal, or scorer failed). \
Replace the DEAD claims with NEW ones designed to actually SEPARATE the candidates.
The claims form a TEMPORAL LADDER of stages — "script" (code exists/valid/reads inputs), \
"log" (training dynamics from logs), "result" (artifact fidelity: pred CSV columns must \
match the ORIGINAL input's column names exactly, no spurious `_pred` suffixes; \
completeness; prediction sanity). Replace dead claims at the stage where discrimination \
is missing.

# Task goal
{goal[:4000]}

# Union of the candidate workspaces' file inventories — paths RELATIVE TO EACH WORKSPACE \
ROOT (size; CSV rows x cols / PNG dims; how many workspaces contain each path)
{union_inventory}

# Current rubric outcome on every workspace (label-free measurements)
{outcome_report}

# DEAD claims to replace (verbatim statements + what happened)
{dead_block}

# ALIVE claims (discriminating already — do NOT restate or duplicate them)
{alive_block or "(none)"}

# What to output
{n_new} REPLACEMENT claims only, meeting the same four hard rules:
1. TASK-DEFINING — only properties the goal's OWN words specify or imply (never invent \
deliverables/metrics the goal does not mention).
2. DECIDABLE — deterministic Python on this workspace's files alone (scripts' code \
included: method/model/featurization requirements are checkable by reading the code; \
log dynamics are checkable by regex-parsing the logs).
3. DECISIVE — aim at the DIFFERENCES visible above: if all candidates have the output \
file, target what differs (row counts vs the input inventory, exact column sets matching \
the original naming, value plausibility/ranges, the method the goal requires in code, \
completeness fractions, loss/accuracy trends in logs). A claim all candidates pass is \
worthless.
4. GRADED — measure a QUANTITY and map it to continuous partial credit on 0..1; never \
a bare boolean unless truly binary.
Each replacement claim must carry its "stage" ("script" | "log" | "result") and a \
"temporal_index" continuing the ladder. New claim ids must be FRESH (continue numbering: \
{next_id}, ...); never reuse a dead or alive id. Strict JSON only:
{{"claims": [{{"id": "...", "stage": "...", "temporal_index": <int>, "category": "...", \
"statement": "...", "target": "...", "scoring_rule": "..."}}]}}"""


# ------------------------------------------------ LLM content screens ----
# Operator mandate: NO regex/whitelist content judgment. Whether a claim
# is GENERIC and whether it SMUGGLES a numeric answer is decided by the
# judge LLM — ONE batched call per task's claim set (never per-claim) on
# the host's ``llm_text`` transport (judge model, temperature 0), with
# the same one-repair-round JSON retry as the pipeline's other judge
# calls. The extraction prompt's SELF-CHECK instructions are the first
# line of defense; these screens are the belt-and-braces post-filter.


def _flag(verdict: dict[str, Any], key: str) -> bool:
    """Coerce a model-supplied boolean-ish flag (true/"true"/1) to bool."""
    raw = verdict.get(key)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() == "true"


def _screen_call(
    llm_text: Callable[[str, str, str], str],
    uuid: str,
    agent: str,
    prompt: str,
) -> list[dict[str, Any]] | None:
    """One screening judge call parsed as a JSON list of verdict objects.

    Same retry pattern as the pipeline's other judge calls: on a parse
    failure the call is re-issued once with the failure fed back. A
    failed call returns ``None`` — callers degrade open, keeping the
    prompt-level SELF-CHECK as the primary screen.
    """
    cur = prompt
    for attempt in (1, 2):
        name = agent if attempt == 1 else f"{agent}_retry"
        try:
            raw = llm_text(uuid, name, cur)
        except Exception:  # noqa: BLE001 — degrade open per screen
            return None
        payload = extract_json_payload(raw or "")
        if payload:
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return parsed
        cur = (
            f"{prompt}\n\nPREVIOUS ATTEMPT FAILED: the reply was not the "
            "strict JSON array requested. Reply with ONLY that array — no "
            "prose, no markdown fences, no commentary."
        )
    return None


def _claim_block(claims: list[dict[str, Any]]) -> str:
    """The task's claims as one strict-JSON line inside a screen prompt."""
    return json.dumps(
        [
            {
                "id": c.get("id", ""),
                "statement": c.get("statement", ""),
                "target": c.get("target", ""),
                "scoring_rule": c.get("scoring_rule", ""),
            }
            for c in claims
        ]
    )


def is_generic_claim(
    claims: list[dict[str, Any]],
    llm_text: Callable[[str, str, str], str],
    uuid: str = "",
) -> dict[str, str]:
    """ONE batched LLM judgment: which of a task's claims are generic.

    A generic claim is one any workspace would trivially satisfy — bare
    file existence, "the script runs", workspace tidiness — so it
    measures nothing and cannot separate a correct workspace from a
    plausible-but-wrong one. All of the task's claims go to the judge in
    a single call (batched, never per-claim). Returns ``{claim_id:
    reason}`` naming only the generic claims; verdicts on ids outside
    the claim set are ignored, and a failed judge call degrades open
    (nothing flagged).
    """
    if not claims:
        return {}
    prompt = f"""# GENERIC-CLAIM SCREEN
You are auditing the acceptance claims of ONE scientific verification rubric. A GENERIC \
claim is one any workspace would trivially satisfy — bare file existence, "the script \
runs", workspace tidiness — so it measures nothing and cannot separate a correct \
workspace from a plausible-but-wrong one. A claim that measures a real quantity (row \
counts vs the input inventory, a fraction of values in a stated range, the presence of \
the required method in the delivered code) is NOT generic.

CLAIMS (JSON):
{_claim_block(claims)}

For EACH claim above, judge: is it generic/trivially-satisfiable? Reply with ONLY a \
strict JSON array, exactly one entry per claim and nothing else:
[{{"id": "<claim id>", "generic": true, "reason": "<one sentence>"}}]
(entries for non-generic claims use "generic": false)."""
    verdicts = _screen_call(llm_text, uuid, "hybrid_screen_generic", prompt)
    ids = {str(c.get("id", "")) for c in claims}
    flagged: dict[str, str] = {}
    for v in verdicts or []:
        if not isinstance(v, dict):
            continue
        cid = str(v.get("id", ""))
        if cid in ids and _flag(v, "generic"):
            flagged[cid] = str(v.get("reason") or "generic claim")
    return flagged


def t1_violations(
    claims: list[dict[str, Any]],
    goal: str,
    llm_text: Callable[[str, str, str], str],
    uuid: str = "",
) -> dict[str, list[str]]:
    """ONE batched LLM judgment: which claims smuggle an answer (E29 T1).

    A claim violates the firewall when its statement / scoring_rule /
    target hard-codes a specific numeric answer or computed result that
    the task goal does NOT state — an answer key a workflow cannot
    verify without already knowing the answer. Constants the goal
    itself states, and constants structural to scoring or plotting
    (ranges like [0,1], partial-credit levels, bin/grid counts, schema
    sentinels), are verifiable, never smuggled. All of the task's claims
    go to the judge in a single call (batched, never per-claim).
    Returns per-claim violation strings keyed by claim id — every
    screened id is present, clean claims mapping to ``[]``; a failed
    judge call degrades open (all clean).
    """
    if not claims:
        return {}
    prompt = f"""# SMUGGLED-ANSWER SCREEN (T1 firewall, E29)
You are enforcing a WORKSPACE-VERIFIABILITY firewall on the acceptance claims of ONE \
scientific task. A claim SMUGGLES an answer when its statement, scoring_rule or target \
hard-codes a specific numeric answer or computed result that the task goal does NOT \
state — a value a workflow cannot verify without already knowing the answer. A claim \
that says "the output CSV must have exactly 293 rows" when the goal never mentions 293 \
is smuggling an answer. A claim that says "probabilities must be in [0,1]" is \
structural, not smuggled. Constants the goal itself states are verifiable, never \
smuggled.

# Task goal
{goal[:4000]}

CLAIMS (JSON):
{_claim_block(claims)}

For EACH claim above, judge: does it hard-code a specific numeric answer or computed \
result the goal does not state? Reply with ONLY a strict JSON array, exactly one entry \
per claim and nothing else:
[{{"id": "<claim id>", "smuggled": true, "violating_field": "statement", \
"violating_value": "<the specific number/text>", "reason": "<one sentence>"}}]
(entries for clean claims use "smuggled": false with "violating_field": null and \
"violating_value": null)."""
    verdicts = _screen_call(llm_text, uuid, "hybrid_screen_t1", prompt)
    ids = {str(c.get("id", "")) for c in claims}
    why: dict[str, list[str]] = {cid: [] for cid in ids}
    for v in verdicts or []:
        if not isinstance(v, dict):
            continue
        cid = str(v.get("id", ""))
        if cid not in ids or not _flag(v, "smuggled"):
            continue
        field = str(v.get("violating_field") or "").strip()
        value = str(v.get("violating_value") or "").strip()
        if field and value:
            why[cid].append(f"{field}:{value}")
        else:
            why[cid].append(str(v.get("reason") or "smuggled answer"))
    return why


def composition_counts(claims: list[dict[str, Any]]) -> dict[str, int]:
    """E30 content-lever counts: lever categories + fidelity families."""
    cats = collections.Counter(c.get("category", "other") for c in claims)
    return {
        "core_computation": int(cats.get("core_computation", 0)),
        "method_identity": int(cats.get("method_identity", 0)),
        "fidelity_families": int(
            sum(1 for f in FIDELITY_CATEGORIES if cats.get(f, 0) >= 1)
        ),
    }


def composition_error(claims: list[dict[str, Any]]) -> str | None:
    """Error naming any unmet E30 mandate, or None when satisfied."""
    cc = composition_counts(claims)
    shortfall = [
        f"{cat}={cc[cat]} (need >= {need})"
        for cat, need in LEVER_MIN_COUNTS.items()
        if cc[cat] < need
    ]
    return "composition mandate unmet: " + "; ".join(shortfall) if shortfall else None


def _normalize_stage(raw: Any) -> str:
    """Coerce a model-supplied stage label to a canonical stage."""
    s = str(raw or "result").strip().lower()
    return s if s in FIGURE_STAGES else "result"



def _normalize_claim(c: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    """Normalize one raw model claim into the canonical ladder shape."""
    stage = _normalize_stage(c.get("stage", c.get("rung", "result")))
    try:
        tidx = int(c.get("temporal_index", fallback_index))
    except (TypeError, ValueError):
        tidx = fallback_index
    cat = str(c.get("category") or "other").strip()
    return {
        "id": str(c.get("id") or f"C{fallback_index}").strip(),
        "stage": stage,
        "temporal_index": tidx,
        "category": cat if cat in CLAIM_CATEGORIES else "other",
        "statement": str(c["statement"]).strip(),
        "target": str(c["target"]).strip(),
        "scoring_rule": str(c["scoring_rule"]).strip(),
    }


def validate_claims(
    obj: Any,
    lo: int,
    hi: int,
    drop_generic: bool = True,
    require_composition: bool = False,
    llm_text: Callable[[str, str, str], str] | None = None,
    uuid: str = "",
) -> tuple[list[dict[str, Any]], str | None]:
    """Validate a parsed claims JSON object against the ladder contract.

    Claims are returned in TEMPORAL ORDER (stage rank, then
    ``temporal_index``, then input order). The stage field is lenient
    (unknown/missing → ``result``) but shape and the count contract are
    strict.

    Args:
        obj: Parsed JSON object with a ``claims`` list.
        lo: Minimum accepted claim count.
        hi: Maximum accepted claim count.
        drop_generic: Drop generic (non-measuring) claims, judged by ONE
            batched LLM call over all claims at once. The screen runs
            only when ``llm_text`` is given; without it validation is
            shape-only (the extraction flow always passes the judge
            transport).
        require_composition: Enforce the E30 content mandate — at least 2
            ``core_computation`` AND at least 2 ``method_identity``
            claims. An unmet mandate returns the valid claims together
            with a ``composition mandate unmet: ...`` error, so the caller
            may retry with feedback and still accept the best effort once
            its retry budget is spent (E35 semantics: 18/19 groups met
            the mandate on prompt power alone; the remainder is disclosed,
            not rejected).
        llm_text: ``(uuid, agent_name, prompt) -> raw text`` — the host
            evaluator's judge call (temperature 0, cached, retried).
        uuid: Workflow uuid passed through to the judge call.
    """
    if not isinstance(obj, dict) or not isinstance(obj.get("claims"), list):
        return [], "top-level JSON must be an object with a 'claims' list"
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for k, c in enumerate(obj["claims"], 1):
        if not isinstance(c, dict):
            return out, f"claim #{k} is not an object"
        cid = str(c.get("id") or f"C{k}").strip()
        if cid in seen:
            return out, f"duplicate claim id {cid}"
        seen.add(cid)
        for field in _REQUIRED_FIELDS:
            if not isinstance(c.get(field), str) or not c[field].strip():
                return out, f"claim {cid}: missing/empty '{field}'"
        out.append(_normalize_claim(c, k))
    generic: dict[str, str] = {}
    if drop_generic and llm_text is not None and out:
        generic = is_generic_claim(out, llm_text, uuid)
        out = [c for c in out if c["id"] not in generic]
    n = len(out)
    if not lo <= n <= hi:
        detail = f"exactly {lo}-{hi} non-generic claims required, got {n}"
        if generic:
            detail += f" (generic-rejected: {sorted(generic)})"
        return out, detail
    ordered = sort_claims_temporally(out)
    if require_composition:
        err = composition_error(ordered)
        if err is not None:
            return ordered, err
    return ordered, None

def sort_claims_temporally(
    claims: list[dict[str, Any]], stages: tuple[str, ...] = FIGURE_STAGES
) -> list[dict[str, Any]]:
    """Order claims by (stage rank, temporal_index, original position).

    ``stages`` fixes the ladder order — ``STAGES`` for text tasks,
    ``FIGURE_STAGES`` (visual first) for figure tasks; a claim whose stage
    is not in the given order sorts last.
    """
    order = {s: i for i, s in enumerate(stages)}
    keyed = sorted(
        enumerate(claims),
        key=lambda t: (
            order.get(t[1].get("stage", "result"), len(stages)),
            t[1].get("temporal_index", 0),
            t[0],
        ),
    )
    return [c for _, c in keyed]


def next_claim_id(claims: list[dict[str, Any]]) -> str:
    """Fresh claim id continuing the numeric sequence (C14, C15, ...)."""
    return f"C{_max_claim_no(claims) + 1}"


def _max_claim_no(claims: list[dict[str, Any]]) -> int:
    mx = 0
    for c in claims:
        m = re.search(r"(\d+)$", str(c.get("id", "")))
        if m:
            mx = max(mx, int(m.group(1)))
    return mx


def dedupe_claim_ids(
    new_claims: list[dict[str, Any]],
    existing: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Re-id replacement claims that collide with live/dead ids (E19b guard)."""
    taken = {c["id"] for c in existing}
    next_no = _max_claim_no(existing) + 1
    out: list[dict[str, Any]] = []
    for c in new_claims:
        claim = dict(c)
        while claim["id"] in taken:
            claim["id"] = f"C{next_no}"
            next_no += 1
        taken.add(claim["id"])
        out.append(claim)
    return out


def outcome_report(
    claims: list[dict[str, Any]],
    generations: list[dict[str, Any]],
) -> str:
    """Label-free per-claim outcome over every scored generation."""
    lines: list[str] = []
    for c in claims:
        vals: list[str] = []
        scores: list[float] = []
        for gen in generations:
            s = (gen.get("scores") or {}).get(c["id"])
            if s is None:
                continue
            vals.append(f"{gen.get('uuid', '?')[:12]}={s:.2f}")
            scores.append(s)
        stage = c.get("stage", "result")
        if not vals:
            lines.append(
                f"- [{c['id']}] (stage {stage}) {c['statement']}\n"
                f"  outcome: NO SCORES (scorer failed everywhere)"
            )
            continue
        distinct = sorted(set(scores))
        verdict = (
            "ALL EQUAL (non-discriminative)"
            if len(distinct) <= 1
            else f"{len(distinct)} distinct values"
        )
        ev = next(
            (
                (gen.get("evidence") or {}).get(c["id"])
                for gen in generations
                if (gen.get("evidence") or {}).get(c["id"])
            ),
            "",
        )
        lines.append(
            f"- [{c['id']}] (stage {stage}; {c.get('category', 'other')}) {c['statement']}\n"
            f"  per-workspace: {' '.join(vals)} -> {verdict}"
            + (f"\n  sample evidence: {str(ev)[:200]}" if ev else "")
        )
    return "\n".join(lines) if lines else "(no claims yet)"

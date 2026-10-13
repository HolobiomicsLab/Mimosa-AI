"""Gradient fold + optional E47 refiner for the policy_bt verifier.

The raw gradient is the elimination-point-first template assembly of the
generation's pairwise feedbacks (production convention, E44 ``e44_bt.py``):
decisive LOSSES first (sorted by descending confidence), then ties, then
wins. Feedback strings already carry measured facts with values (policy
content rules); the fold adds no coverage footers inside pairwise
feedback — humility is confined to ONE gradient-tail line carrying the
recon coverage footer.

The optional refiner (E47, default ON) takes the raw gradient as ground
truth plus deterministic FILE PREVIEWS of the generation's workspace and
returns an evidence-checked refinement: verified issues with evidence
quotes, labelled hypotheses (never solutions), missing files, unverified
raw claims and corrections where the raw gradient was wrong about a file.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: Hard cap on the gradient-tail coverage footer: it is one humility
#: line, not a file manifest (2026-10-07: unbounded footers dominated the
#: raw gradient and the refiner echoed them as the whole gradient).
COVERAGE_FOOTER_CAP = 512
REFINER_PROMPT = """You are a gradient refiner for an evolution loop. You \
receive (1) the RAW GRADIENT produced by a pairwise tournament verifier for \
generation N — treat it as GROUND TRUTH, the verified verdict record; and \
(2) FILE PREVIEWS of generation N's workspace. Your job is to produce ONE \
refined gradient that a mutator engineer can act on: clean, faithful, \
evidence-checked.

RULES (binding):
- VERIFY: check each raw claim against the previews — does the file exist, \
does the script contain what is claimed, is the reported error real? Quote \
the matching evidence.
- MISSING FILES: list deliverable files the raw gradient expects that are \
absent from the workspace inventory. A file absent from the deliverables \
inventory is NOT absent from the workspace — check the FULL file listing \
and the goal-contract records before declaring any file missing.
- ISSUES + LIKELY EXPLANATIONS ONLY. Never propose solutions, fixes, or \
next steps. Explanations are HYPOTHESES, labeled as such with their basis.
- FAITHFUL: no new claims, thresholds, or requirements that are not in the \
raw gradient or the previews. No gold/benchmark knowledge. No opinions. If \
the raw gradient is wrong about a file, say so under CORRECTIONS with the \
evidence.
- UNCERTAINTY: anything you cannot check goes to UNVERIFIED — do not guess.

Output STRICT JSON:
{{"refined_gradient": "markdown: ## Verified issues (with evidence quotes) ## \
Likely explanations (hypotheses + basis) ## Missing files ## Unverified from \
raw ## Corrections to raw (if any)",
 "verified_issues": [{{"issue": "...", "evidence": "...", "confidence": 0.0}}],
 "likely_explanations": [{{"hypothesis": "...", "basis": "...", "confidence":
   0.0}}],
 "missing_files": ["..."],
 "unverified_from_raw": ["..."],
 "corrections": [{{"raw_claim": "...", "what_files_show": "..."}}]}}


# RAW GRADIENT (ground truth, generation {uuid})

{gradient}

# FILE PREVIEWS (generation {uuid} workspace)

{preview}"""

#: Output keys the refiner contract requires (tests assert the schema).
REFINER_KEYS = (
    "refined_gradient",
    "verified_issues",
    "likely_explanations",
    "missing_files",
    "unverified_from_raw",
    "corrections",
)


def fold_gradient(
    uuid: str,
    pair_records: list[dict[str, Any]],
    coverage_footer: str = "",
    ledger: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Elimination-first fold of one generation's pairwise feedbacks.

    Args:
        uuid: The generation the gradient is for.
        pair_records: One record per predecessor pair. Two spellings are
            accepted: ``other``/``rel`` in {won, lost, tie} (unit tests)
            or ``prev_uuid``/``outcome`` (the evaluator's pair_records).
            Both carry ``feedback`` (measured facts) and ``confidence``.
        coverage_footer: Recon coverage footer — used for exactly ONE
            tail line when something was genuinely left unassessed (the
            v2b lesson: humility lives here, never inside the pairwise
            feedback).
        ledger: The metric ledger (review §4.2): ``current`` =
            primary_values of this generation, ``preds`` = {uuid:
            primary_values} per predecessor, ``metrics`` = the recon
            primary_metrics entries. When present, a deterministic
            ``METRICS`` section renders measured values and tie
            verdicts FROM THE DATA — never regex-parsed from the
            LLM-written feedback (the 2026-10-09 Fix-A regex missed its
            own motivating single-digit tie and matched non-metrics).

    Returns:
        ``(text, meta)`` where meta = ``{"tied": [metric ids], "not_assessed":
        [metric ids], "footer_appended": bool}`` — the log reports what the
        fold ACTUALLY appended (S5: the old log recomputed a stale
        condition instead).
    """
    def _rel(f: dict[str, Any]) -> Any:
        return f.get("rel", f.get("outcome"))

    def _other(f: dict[str, Any]) -> Any:
        return f.get("other", f.get("prev_uuid", "?"))

    def _conf(f: dict[str, Any]) -> float:
        """N13/P2.2: string confidence ("high") must not raise here."""
        try:
            return float(f.get("confidence") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    losses = [f for f in pair_records if _rel(f) == "lost"]
    ties = [f for f in pair_records if _rel(f) == "tie"]
    wins = [f for f in pair_records if _rel(f) == "won"]
    lines = [
        f"GEN {uuid} — verdicts vs predecessors ({len(pair_records)} pairs: "
        f"{len(wins)}W/{len(losses)}L/{len(ties)}T):",
        "",
    ]
    lines.append("WHAT TO FIX FIRST (decisive losses):")
    if losses:
        for f in sorted(losses, key=_conf, reverse=True):
            lines.append(f"- vs {_other(f)}: {f['feedback']}")
    if ties:
        lines.append("")
        lines.append("UNRESOLVED (ties — evidence invisible or balanced):")
        for f in ties:
            fb = f.get("feedback") or "tie — no decisive evidence"
            lines.append(f"- vs {_other(f)}: {fb}")
    if wins:
        lines.append("")
        lines.append("KEEP (what beats predecessors):")
        for f in wins:
            lines.append(f"- vs {_other(f)}: {f.get('feedback') or ''}")

    # ---- deterministic METRICS section from the ledger (review §4.2) ----
    # The coverage footer is appended ONLY when something was genuinely
    # left unassessed: a degenerate no-pairs generation, or unresolved
    # ties the ledger could not cover with a single resolved metric.
    # All-decisive folds (any mix of wins/losses, 0 ties) mean every
    # pair produced a verdict, so no footer is needed.
    tied: list[str] = []
    not_assessed: list[str] = []
    ledger_covers = False
    metrics = (ledger or {}).get("metrics") or []
    current_pv = (ledger or {}).get("current") or {}
    preds_pv = (ledger or {}).get("preds") or {}
    if metrics:
        metric_lines = _metrics_section(metrics, current_pv, preds_pv, tied, not_assessed)
        if metric_lines:
            lines.append("")
            lines.extend(metric_lines)
        ledger_covers = any(
            (current_pv.get(_metric_id(m)) or {}).get("status") == "resolved"
            for m in metrics
            if isinstance(m, dict)
        )

    footer = _clean_footer(coverage_footer)
    footer_appended = bool(footer) and (
        not pair_records or (ties and not ledger_covers)
    )
    if footer_appended:
        lines.append("")
        lines.append(f"NOT ASSESSED (grading bars unknown): {footer}")
    meta = {
        "tied": tied,
        "not_assessed": not_assessed,
        "footer_appended": footer_appended,
    }
    return "\n".join(lines), meta


def _metric_id(metric: dict[str, Any]) -> str:
    """Stable ledger id of one primary_metrics entry."""
    return str(metric.get("id") or metric.get("key") or "").strip()


def _fmt_value(value: float) -> str:
    """Compact deterministic number format (47 not 47.0, -2 not -2.0)."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _metrics_section(
    metrics: list[Any],
    current_pv: dict[str, Any],
    preds_pv: dict[str, dict[str, Any]],
    tied: list[str],
    not_assessed: list[str],
) -> list[str]:
    """Render the deterministic METRICS block (review §4.2).

    One bullet per metric: id, direction, role, current value and every
    predecessor value (with missing/ambiguous counts). Current-gen
    missing/ambiguous metrics land under NOT ASSESSED with the reason
    instead — they were NOT measured, so no value is invented for them.
    """
    n_preds = len(preds_pv)
    lines = [
        f"METRICS (measured, vs {n_preds} predecessor"
        f"{'s' if n_preds != 1 else ''}):"
    ]
    unassessed: list[str] = []
    bullets = 0
    for m in metrics:
        if not isinstance(m, dict):
            continue
        mid = _metric_id(m)
        if not mid:
            continue
        cur = current_pv.get(mid) or {}
        if cur.get("status") != "resolved":
            unassessed.append(
                f"NOT ASSESSED: {mid} — {_not_assessed_reason(cur)}"
            )
            not_assessed.append(mid)
            continue
        role = str(m.get("role") or "objective")
        direction = str(m.get("direction") or "higher")
        dir_label = {
            "higher": "higher is better",
            "lower": "lower is better",
            "equal": "equal expected",
        }.get(direction, "higher is better")
        value = float(cur.get("value"))
        value_counts: dict[float, int] = {}
        n_missing = n_amb = 0
        for pv in preds_pv.values():
            entry = pv.get(mid) or {}
            status = entry.get("status")
            if status == "resolved":
                pv_val = float(entry.get("value"))
                value_counts[pv_val] = value_counts.get(pv_val, 0) + 1
            elif status == "ambiguous":
                n_amb += 1
            else:
                n_missing += 1
        n_value = sum(value_counts.values())
        pred_parts = [
            f"{_fmt_value(v)} ×{value_counts[v]}"
            for v in sorted(value_counts)
        ]
        if n_missing:
            pred_parts.append(f"missing ×{n_missing}")
        if n_amb:
            pred_parts.append(f"ambiguous ×{n_amb}")
        preds_txt = (
            "; ".join(pred_parts) if pred_parts else "none comparable"
        )
        is_tied = (
            n_value > 0
            and list(value_counts) == [value]
        )
        tie_txt = ""
        if is_tied:
            tied.append(mid)
            tie_txt = (
                " → tied with all" if not (n_missing or n_amb) else " → tied"
            )
        lines.append(
            f"- {mid} ({dir_label}, {role}): {_fmt_value(value)} here; "
            f"predecessors {preds_txt}{tie_txt}"
        )
        bullets += 1
    if not bullets and not unassessed:
        return []  # nothing measured and nothing to declare: no section
    lines.extend(unassessed)
    return lines


def _not_assessed_reason(entry: dict[str, Any]) -> str:
    """Why one metric was not measured in this generation."""
    if entry.get("status") == "ambiguous":
        cands = entry.get("candidates") or []
        parts = [
            f"{c.get('source')}={_fmt_value(float(c.get('value')))}"
            for c in cands[:4]
            if isinstance(c, dict) and c.get("value") is not None
        ]
        listed = ", ".join(parts)
        if len(cands) > 4:
            listed += ", …" if listed else "…"
        return f"ambiguous in this generation (candidates: {listed})"
    return "missing in this generation (no alias hit, no lexical match)"


def _clean_footer(coverage_footer: str) -> str:
    """One capped humility line, without the trailing grading-bars note."""
    footer = " ".join(str(coverage_footer or "").split())
    footer = re.sub(
        r"\s*\(\s*grading bars unknown\s*\)\s*:?\s*$", "", footer,
        flags=re.IGNORECASE,
    ).strip()
    if len(footer) > COVERAGE_FOOTER_CAP:
        footer = footer[: COVERAGE_FOOTER_CAP - 1].rstrip() + "…"
    return footer


def build_preview(
    ws: Path,
    digest: dict[str, Any],
    ledger: dict[str, Any] | None,
    limit: int = 6000,
) -> str:
    """Cheap deterministic file previews for the refiner (port of E47).

    Args:
        ws: The generation's workspace (sandbox copy / snapshot / live).
        digest: Its digest (deliverables inventory).
        ledger: Its re-execution ledger (exec facts).
        limit: Soft character cap.

    Returns:
        Preview text (script head/tail, deliverables inventory, log tail,
        pred CSV headers).
    """
    ws = Path(ws)
    out: list[str] = []
    ledger = ledger or {}
    out.append(
        f"exec: exit={ledger.get('exit_code')} timed_out="
        f"{ledger.get('timed_out')} wall={ledger.get('wall_time_s')}s "
        f"error_signature={ledger.get('error_signature')}"
    )
    out.append("deliverables inventory (path | bytes | fresh_this_run):")
    for dv in (digest.get("deliverables") or [])[:12]:
        out.append(
            f"  {dv.get('path')} | {dv.get('bytes')} | "
            f"{dv.get('produced_this_run')}"
        )
    # P1.6: the goal contract records + a FULL top-level listing — the
    # refiner must never infer absence from the (filtered) inventory.
    cfs = digest.get("contract_files") or []
    if cfs:
        out.append("goal-contract files (path | present | bytes | head):")
        for cf in cfs[:16]:
            head = " ".join(str(cf.get("head") or "").split())[:120]
            out.append(
                f"  {cf.get('path')} | {cf.get('present')} | "
                f"{cf.get('bytes')} | {head}"
            )
    if ws.exists():
        out.append("FULL top-level file listing (path | bytes) — ALL suffixes:")
        for p in sorted(ws.iterdir()):
            if p.is_file():
                try:
                    out.append(f"  {p.name} | {p.stat().st_size}")
                except OSError:
                    out.append(f"  {p.name} | ?")
    if ws.exists():
        scripts = sorted(
            [
                p
                for p in ws.glob("*.py")
                if not p.name.startswith(("test_", "verify_", "probe_"))
            ],
            key=lambda p: -p.stat().st_size,
        )[:1]
        for sc in scripts:
            txt = sc.read_text(errors="replace").splitlines()
            out.append(f"--- script {sc.name} ({len(txt)} lines) head ---")
            out += txt[:60]
            out.append("--- script tail ---")
            out += txt[-25:]
        log = ws / "output.log"
        if log.exists():
            out.append("--- output.log tail ---")
            out += log.read_text(errors="replace").splitlines()[-25:]
        pred = ws / "pred_results"
        if pred.exists():
            for p in sorted(pred.iterdir())[:4]:
                if p.suffix.lower() == ".csv" and p.stat().st_size < 5 << 20:
                    lines = p.read_text(errors="replace").splitlines()[:6]
                    out.append(f"--- {p.name} header+5rows ---")
                    out += lines
    text = "\n".join(out)
    if len(text) > limit * 2:
        return text[:limit] + "\n... (preview truncated)"
    return text


def refine_gradient(
    llm_json: Callable[[str, int, str], Any],
    uuid: str,
    raw_gradient: str,
    preview: str,
) -> dict[str, Any] | None:
    """One E47 refiner call; None on unparseable output.

    Args:
        llm_json: ``(prompt, max_tokens, role) -> parsed JSON or None``.
        uuid: The generation being refined.
        raw_gradient: The fold output (ground truth).
        preview: File previews from :func:`build_preview`.

    Returns:
        The refiner payload when it carries ``refined_gradient`` and the
        schema keys, else None (caller keeps the raw gradient).
    """
    prompt = REFINER_PROMPT.format(
        uuid=uuid, gradient=raw_gradient, preview=preview
    )
    parsed = llm_json(prompt, 2500, f"policy_bt_refiner_{uuid[-8:]}")
    if not isinstance(parsed, dict):
        return None
    if not parsed.get("refined_gradient"):
        return None
    for key in REFINER_KEYS:
        parsed.setdefault(key, [] if key != "refined_gradient" else "")
    return parsed


def refined_text(raw_gradient: str, refinement: dict[str, Any] | None) -> str:
    """Choose the final gradient text (refined when available).

    P1.7: the refined text KEEPS the fold's header line (``GEN … (N
    pairs: xW/yL/zT)``) and the decisive-loss block prepended — the
    tournament's measured verdict summary must survive refinement (the
    HP gen-4 final gradient lost its ``3W/0L/0T`` header entirely).
    §4.2: the deterministic ``METRICS`` block (measured values and tie
    verdicts from the ledger) is kept verbatim as well — the refiner is
    an LLM and must never be able to drop or reword measured values.
    """
    if not (refinement and refinement.get("refined_gradient")):
        return raw_gradient
    lines = raw_gradient.splitlines()
    head = lines[0] if lines else ""
    block_lines: list[str] = []
    in_block = False
    for ln in lines[1:]:
        if ln.startswith("WHAT TO FIX FIRST"):
            in_block = True
            block_lines.append(ln)
            continue
        if in_block:
            if not ln.strip() or len(block_lines) > 12:
                break  # the block ends at the first blank line
            block_lines.append(ln)
    metrics_block = ""
    for i, ln in enumerate(lines):
        if ln.startswith("METRICS ("):
            metrics_block = "\n".join(lines[i:]).rstrip()
            break
    refined = str(refinement["refined_gradient"])
    parts = [
        p
        for p in (head, "\n".join(block_lines), refined, metrics_block)
        if p
    ]
    return "\n\n".join(parts)


def refiner_stats(refinement: dict[str, Any] | None) -> dict[str, int]:
    """Counts for the GRADIENT stage line (issues/missing/corrections)."""
    if not refinement:
        return {"issues": 0, "missing": 0, "corr": 0}
    return {
        "issues": len(refinement.get("verified_issues") or []),
        "missing": len(refinement.get("missing_files") or []),
        "corr": len(refinement.get("corrections") or []),
    }

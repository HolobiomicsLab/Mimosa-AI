"""Per-task reconnaissance for the policy_bt verifier (plan §1 P2b).

ONE judge-model call per task: the verbatim goal plus the deterministic
digests of the generations seen so far go in, a TASK CONFIG comes out —
what the deliverables are, which metric contexts are comparable across
generations, value-sanity rules derivable from goal + in-workspace
inputs, tie policy, pitfalls and the coverage footer used (sparingly)
by the gradient tail. Cached per task in the registry (goal-hash keyed);
regenerated only when the goal or the prompt version changes.

Gold, labels, rewards and grader code never enter the prompt (gold
firewall — audited PASS in E45).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

#: Bump to invalidate every recon cache cleanly (operator decision 3).
#: v2 (2026-10-09): primary_metrics gained the metric-ledger schema
#: (id/role/aliases/bounds/integer/convention — review §4.1).
RECON_PROMPT_VERSION = (
    "recon-v2-ledger-"
    + hashlib.sha256(b"policy_bt_recon_v2_metric_ledger").hexdigest()[:12]
)

RECON_PROMPT = """You are the reconnaissance analyst for a pairwise workspace verifier.
Below: (1) the verbatim goal of a scientific-computing task, and (2) one DIGEST
per generation of an evolution run that attempted it. All digests use the same
deterministic schema:

- exec: fresh re-execution facts (exit_code, wall_time_s, timed_out,
  error_signature, produced_pred_results, log_tail lines)
- deliverables: [{{path, bytes, kind, produced_this_run}}] (produced_this_run=false
  means the file is STALE — present in the workspace but not written by a fresh run)
- contract_files: [{{path, present, bytes, sha256, head}}] — files the goal
  names verbatim; present=false is a contract violation regardless of freshness
- tables: parsed CSVs [{{path, n_rows, columns, num_stats{{col:{{min,max,mean,std,
  n_nan,n_unique}}}}, first_col_head, id_order_match{{test_file,test_col,pred_col,match}}}}]
- arrays: npy [{{path, shape, dtype, stats}}]
- figures: png [{{path, width_px, height_px, mode}}]
- log_metrics: metric lines scraped from logs [{{metric, value(null=nan), context,
  raw_line, source}}]
- json_metrics: {{filename: {{dotted.key: numeric_value}}}}
- inputs: in-workspace train/test input stats (train target distributions!)
- train_used: rows of training data actually consumed

Your job: write the TASK CONFIG that a second engineer (who writes a
deterministic pairwise comparison script) will follow. Ground every rule in
fields that actually exist in these digests. Never invent thresholds that are
not derivable from the goal text or the digest values. If generations disagree
on metric semantics (different splits/contexts), say so and mark those
comparisons discounted or incomparable.

primary_metrics (binding): for each metric list under "aliases" EVERY exact
digest key that carried the quantity in ANY slot above, including drifted
spellings across generations (e.g. results.json:minimum_energy in slot 1 and
results.json:min_energy in slot 2 are ONE metric with two aliases) — a
deterministic resolver matches aliases exactly, so a missed alias means an
unresolved metric. "bounds" gets [lo, hi] ONLY when derivable from the goal
text (e.g. [0,1] for AUC); null otherwise. "integer" is true for counts.
"role": objective (a real optimization target), invariant (a fact every
correct answer shares, e.g. degeneracy or total-enumeration counts) or
report (descriptive only).

# 1. TASK GOAL (verbatim)

{goal}

# 2. GENERATION DIGESTS (slot order = evolution order)

{digests}

# 3. YOUR OUTPUT — strict JSON, nothing else

{{
 "schema_map": "<=120 words: what the deliverable(s) of THIS task are, which
   digest fields carry them, and how generations differ structurally>",
 "deliverable_contract": {{"expected_paths": ["..."],
   "expected_schema": "<=50 words: the exact columns/format/order the GOAL TEXT
   demands for each deliverable, quoted or paraphrased from the goal only",
   "format_notes": "<=60 words"}},
 "graded_surface_vocabulary": ["<=12 words each: properties the goal text
   EXPLICITLY demands and a strict checker would verify — named methods to use,
   exact paths, columns, row counts, orders, value ranges, required components"],
 "primary_metrics": [
   {{"id": "<stable snake_case id naming the QUANTITY, e.g. min_energy,
     hh_contacts — used verbatim by the policy and the gradient fold>",
     "field": "log_metrics|json_metrics|tables.num_stats|arrays.stats",
     "key": "<which metric/column/key, incl. file prefix if fixed>",
     "direction": "higher|lower|equal",
     "role": "objective|invariant|report — objective = a real
       optimization target; invariant = a fact every correct answer
       shares (degeneracy/total-count); report = descriptive only",
     "aliases": [{{"file": "<filename>", "key": "<EXACT dotted key as seen
       in the digests above>"}}],
     "bounds": null,
     "integer": false,
     "convention": "<e.g. full directed walks, not first-step-fixed>",
     "comparable_contexts": ["..."], "discounted_contexts": ["..."],
     "notes": "<=40 words"}},
 ],
 "value_sanity": ["<=25 words each: dynamic-range / distribution /
   degeneracy rules derivable from goal or inputs stats (e.g. probability
   columns must span [0,1] with n_unique >> 2; predictions should not be
   shifted far from train target stats without cause)"],
 "execution_rules": ["<=25 words each: how crash/timeout/stale artifacts /
   produced_this_run should decide or discount for THIS task"],
 "tie_policy": "<=60 words: when the honest verdict is tie (e.g. both
   structurally identical, or the deciding difference is invisible in digests)",
 "pitfalls": ["<=25 words each: comparability traps specific to this task's
   generations (metric semantics drift, split differences, degenerate
   self-metrics)"],
 "feedback_focus": "<=50 words: what feedback_for_loser should target first
   for this task",
 "coverage_footer": "<=45 words: ONE line naming every deliverable path and
   metric family from the goal that ANY pairwise feedback for this task must
   either assess or explicitly declare unassessed (never silently omit);
   append '(grading bars unknown)' verbatim"
}}"""
# NOTE: the JSON-brace template above is hand-escaped for .format(); keep as-is.


def build_prompt(goal: str, digests: list[dict[str, Any]]) -> str:
    """Assemble the recon prompt from the goal and generation digests.

    Digest ``uuid`` and derived ledger fields are stripped (identifiers
    and resolver output are never evidence for the recon).
    """
    parts = []
    for i, d in enumerate(digests, 1):
        d = {
            k: v
            for k, v in d.items()
            if k not in ("uuid", "primary_values", "primary_values_version")
        }
        parts.append(f"## slot_{i}\n" + json.dumps(d, indent=1, sort_keys=True))
    return RECON_PROMPT.format(goal=goal, digests="\n".join(parts))


#: The three roles a primary metric can play (review §4.1). Only
#: ``objective`` metrics are ever optimization targets.
_METRIC_ROLES = ("objective", "invariant", "report")
#: ``equal`` covers invariants every correct answer shares.
_METRIC_DIRECTIONS = ("higher", "lower", "equal")


def _metric_id(key: str, taken: set[str]) -> str:
    """Stable snake_case id derived from a metric key (legacy payloads).

    The recon supplies explicit ids; this only covers cached configs
    from before the schema extension (or degenerate payloads).
    """
    toks = re.split(r"[^a-z0-9]+", key.lower())
    keep = [
        t
        for t in toks
        if len(t) > 1
        and t
        not in (
            "the",
            "a",
            "an",
            "of",
            "for",
            "in",
            "to",
            "and",
            "or",
            "is",
        )
    ]
    mid = "_".join(keep)[:40] or "metric"
    base, n = mid, 2
    while mid in taken:
        mid = f"{base}_{n}"
        n += 1
    taken.add(mid)
    return mid


def _normalize_primary_metrics(entries: Any) -> list[dict[str, Any]]:
    """Coerce recon ``primary_metrics`` entries to the §4.1 schema.

    Every entry gains defaults for the new fields (``id``, ``role``,
    ``aliases``, ``bounds``, ``integer``, ``convention``) so downstream
    consumers (the resolver, the fold, the policy prompt) can always
    ``.get()`` safely; ``direction``/``role`` values outside their
    vocabularies fall back to the neutral defaults.
    """
    out: list[dict[str, Any]] = []
    taken: set[str] = set()
    for m in entries or []:
        if not isinstance(m, dict):
            continue
        key = str(m.get("key") or "").strip()
        if not key:
            continue
        mid = str(m.get("id") or "").strip() or _metric_id(key, taken)
        taken.add(mid)
        direction = str(m.get("direction") or "higher").strip().lower()
        if direction not in _METRIC_DIRECTIONS:
            direction = "higher"
        role = str(m.get("role") or "objective").strip().lower()
        if role not in _METRIC_ROLES:
            role = "objective"
        aliases: list[dict[str, str]] = []
        for a in m.get("aliases") or []:
            if isinstance(a, dict) and str(a.get("key") or "").strip():
                aliases.append(
                    {
                        "file": str(a.get("file") or "").strip(),
                        "key": str(a["key"]).strip(),
                    }
                )
        bounds = m.get("bounds")
        if isinstance(bounds, list | tuple) and len(bounds) == 2:
            try:
                bounds = [float(bounds[0]), float(bounds[1])]
            except (TypeError, ValueError):
                bounds = None
        else:
            bounds = None
        out.append(
            {
                **m,
                "id": mid,
                "key": key,
                "direction": direction,
                "role": role,
                "aliases": aliases,
                "bounds": bounds,
                "integer": bool(m.get("integer", False)),
                "convention": str(m.get("convention") or ""),
            }
        )
    return out


def ensure_shape(parsed: Any) -> dict[str, Any]:
    """Coerce a parsed recon payload into a usable TASK CONFIG.

    Non-dict payloads or dicts missing every well-known key yield an
    empty config (the caller regenerates); missing individual keys get
    neutral defaults so the policy writer can always ``.get()`` safely.
    """
    if not isinstance(parsed, dict):
        return {}
    known = {
        "schema_map",
        "deliverable_contract",
        "graded_surface_vocabulary",
        "primary_metrics",
        "value_sanity",
        "execution_rules",
        "tie_policy",
        "pitfalls",
        "feedback_focus",
        "coverage_footer",
    }
    if not known & set(parsed):
        return {}
    cfg = dict(parsed)
    cfg.setdefault("deliverable_contract", {})
    cfg.setdefault("graded_surface_vocabulary", [])
    cfg["primary_metrics"] = _normalize_primary_metrics(cfg["primary_metrics"])
    cfg.setdefault("value_sanity", [])
    cfg.setdefault("execution_rules", [])
    cfg.setdefault("pitfalls", [])
    cfg.setdefault("schema_map", "")
    cfg.setdefault("tie_policy", "")
    cfg.setdefault("feedback_focus", "")
    cfg.setdefault("coverage_footer", "")
    return cfg


# ---------------------------------------------------------------------------
# Infrastructure-file filtering (gradient presentation only)
# ---------------------------------------------------------------------------

#: Build/run plumbing that is NEVER a graded deliverable: the pairwise
#: digests cannot see these files (digest.py excludes .sh/.py/infra), so a
#: coverage footer naming them turns every gradient into "NOT ASSESSED"
#: noise about files the policy can never check (2026-10-07 HP run).
INFRA_FILENAMES = {
    "run.sh",
    "requirements.txt",
    "requirement.txt",
    "constraints.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "environment.yml",
    "environment.yaml",
    "makefile",
    "pipfile",
    "pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "conda.yaml",
    "output.log",
    "provenance_notes.txt",
    "environment_report.json",
    "environment_summary.txt",
}
_INFRA_SUFFIXES = (".pyc", ".pyo", ".pyd")
_INFRA_PARTS = ("__pycache__", ".venv", "venv", ".ipynb_checkpoints")
#: Harness bookkeeping markers (``marker_<hex>.*``) are never deliverables
# (P1.5: they showed up as stale deliverables in the HP run).
_INFRA_PREFIXES = ("marker_",)
_INFRA_WORD_RE = re.compile(
    r"\b[\w./-]*"
    + r"(?:"
    + "|".join(re.escape(n) for n in sorted(INFRA_FILENAMES))
    + r")\b"
    + r"|\b[\w./-]*__pycache__[\w./-]*\b"
    + r"|\b[\w./-]+\.(?:pyc|pyo|pyd)\b"
    + r"|\b[\w./-]*marker_[\w-]*\b",
    re.IGNORECASE,
)


def is_infra_name(token: str) -> bool:
    """True when *token* names an infrastructure file/dir, not a deliverable."""
    t = token.strip().strip("`'\"*.,;:()[]{}").lower()
    if not t:
        return False
    base = t.rsplit("/", 1)[-1]
    return (
        t in INFRA_FILENAMES
        or base in INFRA_FILENAMES
        or any(part == p for part in t.split("/") for p in _INFRA_PARTS)
        or base.startswith(_INFRA_PREFIXES)
        or t.endswith(_INFRA_SUFFIXES)
    )


def strip_infra_mentions(text: str) -> str:
    """Remove infrastructure file mentions from a coverage-footer line.

    Deterministic scrub: infra names (and path tokens ending in them) are
    removed from comma/semicolon-separated lists, emptied list items are
    dropped, and separator/spacing debris is tidied. Pure presentation —
    the stored recon config and the policy inputs are untouched.
    """
    if not text:
        return ""
    pieces = re.split(r"([,;])", text)
    kept: list[str] = []
    for piece in pieces:
        if piece in (",", ";"):
            kept.append(piece)
            continue
        cleaned = _INFRA_WORD_RE.sub("", piece)
        # drop trailing list connectors left behind ("and ", "or ")
        cleaned = re.sub(r"\s*\b(?:and|or|plus)\b\s*$", "", cleaned)
        cleaned = cleaned.strip(" ,;")
        kept.append(f" {cleaned}" if cleaned else "")
    out = "".join(kept)
    out = _INFRA_WORD_RE.sub("", out)  # mentions that were not list items
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,;])", r"\1", out)
    out = re.sub(r"(?:[,;]\s*){2,}", ", ", out)  # collapse emptied items
    out = re.sub(r"^[\s,;]+", "", out)
    return out.strip(" ,;")


def coverage_footer(config: dict[str, Any] | None) -> str:
    """The task's coverage footer (gradient-tail humility line only).

    Infrastructure files (run.sh, requirements.txt, __pycache__, *.pyc,
    ...) are stripped: they are reproducibility plumbing the pairwise
    digests never see, so demanding their assessment only manufactures
    "NOT ASSESSED" noise in every gradient.
    """
    return strip_infra_mentions(str((config or {}).get("coverage_footer") or ""))


def _run_recon_judge(
    prompt: str, mimosa_config: Path | None, max_tokens: int
) -> dict[str, Any]:
    """Perform the real recon judge call (the evaluator's own plumbing).

    Same call contract as ``evaluator.py::_ensure_recon``: judge model
    from the Mimosa config, temperature 0, ``reasoning_effort="low"``,
    strict-JSON response normalized by :func:`ensure_shape`. Requires
    an API key (repo ``.env`` is loaded like ``main.py`` does). The raw
    provider dump lands in ``<tmp>/policy_bt_recon_cli/``.

    Args:
        prompt: Assembled recon prompt (goal + digests).
        mimosa_config: Optional Mimosa config JSON path (``judge_model``
            override); ``None`` keeps the built-in defaults.
        max_tokens: Token cap for the judge response.

    Returns:
        The normalized TASK CONFIG (empty dict when unparseable).
    """
    try:
        from sources.evaluators.policy_bt_verifier.evaluator import _SYSTEM_PROMPT
    except ImportError:  # direct script execution (uv run recon.py --run)
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from sources.evaluators.policy_bt_verifier.evaluator import _SYSTEM_PROMPT
    import dotenv

    from config import Config
    from sources.core.llm_provider import LLMConfig, LLMProvider
    from sources.evaluators.base import extract_json_payload
    from sources.utils import paths as paths_mod

    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    dotenv.load_dotenv(paths_mod.user_env_file())
    mimosa = Config()
    if mimosa_config is not None:
        mimosa.load(str(mimosa_config))
    provider_name, model = (
        mimosa.judge_model.split("/", 1)
        if "/" in mimosa.judge_model
        else ("openai", mimosa.judge_model)
    )
    cfg = LLMConfig().from_dict(
        {
            "model": model,
            "provider": provider_name,
            "temperature": 0.0,
            "reasoning_effort": "low",
            "max_tokens": max(0, int(max_tokens)),
            "openrouter_provider": mimosa.openrouter_provider_for(mimosa.judge_model),
            "openrouter_quantizations": mimosa.openrouter_quantizations_for(
                mimosa.judge_model
            ),
        }
    )
    memory_path = Path(tempfile.gettempdir()) / "policy_bt_recon_cli"
    memory_path.mkdir(parents=True, exist_ok=True)
    provider = LLMProvider(
        agent_name="policy_bt_recon_cli",
        memory_path=memory_path,
        system_msg=_SYSTEM_PROMPT,
        config=cfg,
    )
    raw = provider(prompt)
    payload = extract_json_payload(raw or "")
    parsed = None
    if payload:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = None
    if parsed is None:
        print(
            f"recon judge response unparseable; raw dump: "
            f"{memory_path / 'policy_bt_recon_cli.json'}",
            file=sys.stderr,
        )
    return ensure_shape(parsed)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: assemble the recon prompt for one workspace folder.

    Builds the folder's digest (no LLM call) and prints the exact prompt
    the judge model would receive, so the recon input can be inspected
    offline. Run from this package directory::

        uv run recon.py <workspace> [--uuid FAKE]
            [--goal TEXT | --goal-file PATH] [--out prompt.txt]

    With ``--run`` it instead performs the REAL recon judge call (same
    plumbing as ``evaluator.py::_ensure_recon``: judge model from the
    Mimosa config, temperature 0, effort low; an API key must be in the
    environment) and prints the normalized TASK CONFIG::

        uv run recon.py <workspace> --run [--goal ...]
            [--mimosa-config my_config.json] [--max-tokens 4000]

    or as a module from the repository root::

        uv run python -m sources.evaluators.policy_bt_verifier.recon ...

    Args:
        argv: CLI arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success).
    """
    # Local import: digest.py imports this module (is_infra_name), so a
    # module-level import here would be circular under ``python -m``.
    try:
        from .digest import build_digest
    except ImportError:  # direct script execution (uv run recon.py <folder>)
        from digest import build_digest

    ap = argparse.ArgumentParser(
        description=("Assemble the policy_bt recon prompt for one workspace folder."),
    )
    ap.add_argument("folder", type=Path, help="workspace folder to digest")
    ap.add_argument(
        "--uuid",
        default="cli-recon-00000000",
        help="fake generation uuid for the digest slot (default: %(default)s)",
    )
    ap.add_argument(
        "--goal", default="", help="verbatim task goal text (prompt section 1)"
    )
    ap.add_argument(
        "--goal-file", type=Path, default=None, help="read the goal text from a file"
    )
    ap.add_argument("--out", type=Path, default=None, help="also write the output here")
    ap.add_argument(
        "--run",
        action="store_true",
        help="perform the real recon judge call (needs an API key in the "
        "environment) and print the normalized TASK CONFIG instead of the prompt",
    )
    ap.add_argument(
        "--mimosa-config",
        type=Path,
        default=None,
        help="Mimosa config JSON choosing the judge model "
        "(default: built-in config defaults)",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="judge response token cap (default: %(default)s, the evaluator's "
        "recon cap)",
    )
    args = ap.parse_args(argv)

    ws = args.folder.resolve()
    if not ws.is_dir():
        ap.error(f"not a directory: {ws}")
    goal = args.goal
    if args.goal_file is not None:
        goal = args.goal_file.read_text(encoding="utf-8", errors="replace")

    digest = build_digest(args.uuid, ws, None, goal=goal)
    prompt = build_prompt(goal, [digest])
    if args.run:
        task_cfg = _run_recon_judge(prompt, args.mimosa_config, args.max_tokens)
        text = json.dumps(task_cfg, indent=1, sort_keys=True)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
        print(text)
        print(
            f"recon TASK CONFIG: {len(task_cfg.get('primary_metrics') or [])} "
            f"primary metrics, {len(task_cfg)} top-level keys "
            f"(prompt {len(prompt)} chars)",
            file=sys.stderr,
        )
        return 0
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(prompt + "\n", encoding="utf-8")
    print(prompt)
    print(
        f"recon prompt: {len(prompt)} chars "
        f"(goal {len(goal)} chars, 1 digest slot, uuid {args.uuid})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

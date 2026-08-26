"""The run's ASTRA record and its independent evaluations, read back for humans.

Two artifact families feed this view, and neither is written by the
Observatory:

* ``<capsule_dir>/<uuid>/astra.yaml`` — the transparency exporter's ASTRA
  capsule for a family's *best* run: the workflow's decisions with their
  alternatives and rationale, plus the realised universe. Only the best run of
  a family has one, so for every other run we point at the family members
  that do.
* ``<eval_dir>/**/eval_astra.yaml`` — ASB-as-evaluator capsules produced by
  ``asb_eval`` against a run's workspace: executor verdicts over the ASB
  card's own criteria, workspace provenance flags, and the pinned-instrument
  judge layer. These are *independent* of the run's self-assessment — the
  evaluator never reads ``state_result.json`` or ``evaluation.txt`` — which
  is exactly why they belong beside it in the UI.

Everything here follows store.py's defensive contract: partial or missing
artifacts yield ``None``/empty, never an exception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import lineage
from .settings import get_settings


def _read_yaml(path: Path) -> Any | None:
    try:
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        return None


# ── dual-shape ASTRA reading ─────────────────────────────────────────────────
# SOURCE OF TRUTH: AgenticScienceBuilder,
# src/agentic_science_builder/astra_render/render.py (``analysis_bodies`` and
# ``_universe_selection``; ``analysis_body`` is not needed here). Vendored so
# nested ``analyses:``-shaped documents (ASB ground truth, astra_export
# output) resolve without importing the ASB package — shape convergence
# happens at readers, never writers. Keep in lock-step with the renderer.


def _dict(value: Any) -> dict[str, Any]:
    """The value as a mapping, or empty — never an AttributeError downstream."""
    return value if isinstance(value, dict) else {}


def analysis_bodies(doc: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every analysis in the document, root first.

    A nested document holds the run as the ROOT analysis plus one inline
    sub-analysis per capsule — dropping the root or the later siblings would
    silently hide most of the record. A flat legacy document (no ``analyses``
    container) is just its own single body.
    """
    doc = _dict(doc)
    analyses = _dict(doc.get("analyses"))
    if not analyses:
        return [(doc.get("id", ""), doc)]
    bodies: list[tuple[str, dict[str, Any]]] = []
    root = {k: v for k, v in doc.items() if k != "analyses"}
    if any(_dict(root.get(k)) for k in ("decisions", "findings", "evaluation")):
        bodies.append(("(root)", root))
    bodies += [(slug, _dict(body)) for slug, body in analyses.items()]
    return bodies


def _universe_selection(universe: dict[str, Any]) -> dict[str, Any]:
    """Option selection of a universe in either shape (nested merged, or flat)."""
    analyses = _dict(universe.get("analyses"))
    if not analyses:
        return _dict(universe.get("decisions"))
    merged: dict[str, Any] = {}
    for body in analyses.values():
        merged.update(_dict(_dict(body).get("decisions")))
    return merged


# ── the run's own ASTRA capsule ──────────────────────────────────────────────

def capsule_path(run_id: str) -> Path:
    return get_settings().capsule_dir / run_id


def _with_parsed_tags(decision: dict[str, Any]) -> dict[str, Any]:
    """The decision plus derived ``source_steps`` and ``model``.

    New-generation capsules carry ``trace_step:<N>`` and ``model:<id>`` tags;
    old capsules carry a plain ``model`` key and no tags — both are accepted,
    and the original keys are passed through untouched.
    """
    out = dict(decision)
    steps: list[int] = []
    model = decision.get("model")
    tags = decision.get("tags")
    for tag in tags if isinstance(tags, list) else []:
        if not isinstance(tag, str):
            continue
        prefix, _, value = tag.partition(":")
        if prefix == "trace_step" and value.isdigit():
            steps.append(int(value))
        elif prefix == "model" and value:
            model = value
    out["source_steps"] = steps
    out["model"] = model
    return out


def _merged_decisions(bodies: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Decisions across every analysis body, ids kept, collisions slug-prefixed."""
    merged: dict[str, Any] = {}
    for slug, body in bodies:
        for did, dec in _dict(body.get("decisions")).items():
            key = did if did not in merged else f"{slug}/{did}"
            merged[key] = _with_parsed_tags(dec) if isinstance(dec, dict) else dec
    return merged


def _collected_ports(doc: dict[str, Any], key: str) -> list[Any]:
    """The union of ``inputs``/``outputs`` across the document, root first.

    Unlike ``analysis_bodies``'s root gate (which keys on decision-ish
    content), the root ALWAYS counts here: a nested export declares the run's
    ports on the root analysis even when the root carries no decisions.
    """
    analyses = _dict(doc.get("analyses"))
    bodies: list[dict[str, Any]] = [doc] + [_dict(b) for b in analyses.values()]
    out: list[Any] = []
    for body in bodies:
        value = body.get(key)
        if isinstance(value, list):
            out.extend(value)
    return out


def _decisions_era(decisions: dict[str, Any], extraction: Any) -> str | None:
    """Why the decision layer is empty, when it is.

    ``predates_extractor`` — the capsule was written before the decision
    extractor existed (no ``extraction`` block either); ``extracted_none`` —
    the extractor ran and recorded nothing. None when decisions exist.
    """
    if decisions:
        return None
    return "extracted_none" if isinstance(extraction, dict) else "predates_extractor"


def read_astra_capsule(run_id: str) -> dict[str, Any] | None:
    """The transparency exporter's capsule for *run_id*, or None.

    Reads both capsule generations (flat 0.1 with a per-decision ``model``
    key, and 0.0.12 with tags) and both document shapes (flat, nested
    ``analyses:``). Every field of the old payload is preserved; ``inputs``,
    ``extraction``, ``decisions_era``, per-decision ``source_steps``/``model``
    and per-universe merged selections are additive.
    """
    doc = _read_yaml(capsule_path(run_id) / "astra.yaml")
    if not isinstance(doc, dict):
        return None
    universes = []
    udir = capsule_path(run_id) / "universes"
    if udir.is_dir():
        for upath in sorted(udir.glob("*.yaml")):
            u = _read_yaml(upath)
            if isinstance(u, dict):
                universes.append({**u, "decisions": _universe_selection(u)})
    bodies = analysis_bodies(doc)
    decisions = _merged_decisions(bodies)
    extraction = doc.get("extraction")
    return {
        "name": doc.get("name"),
        "description": doc.get("description"),
        "version": doc.get("version"),
        "inputs": _collected_ports(doc, "inputs"),
        "decisions": decisions,
        "decisions_era": _decisions_era(decisions, extraction),
        "outputs": _collected_ports(doc, "outputs"),
        "extraction": extraction if isinstance(extraction, dict) else None,
        "universes": universes,
    }


def family_capsule_ids(run_id: str) -> list[str]:
    """Family members of *run_id* that have an ASTRA capsule on disk.

    The exporter writes one capsule per evolution family (the best run), so a
    run without its own capsule should still lead the reader to the family's
    record instead of a dead end.
    """
    tree = lineage.tree(run_id)
    members = ({n.get("id") for n in tree.get("nodes", [])}
               if isinstance(tree, dict) else {run_id})
    return sorted(
        m for m in members
        if isinstance(m, str) and (capsule_path(m) / "astra.yaml").is_file()
    )


# ── independent evaluation capsules (asb_eval) ───────────────────────────────

def _summarise_eval(doc: dict[str, Any], source: Path) -> dict[str, Any] | None:
    ev = doc.get("evaluation")
    if not isinstance(ev, dict):
        return None
    judge = ev.get("judge") if isinstance(ev.get("judge"), dict) else None
    return {
        "source": str(source),
        "name": doc.get("name"),
        "criteria_source": ev.get("criteria_source"),
        "independent_of_subject": ev.get("independent_of_subject"),
        "summary": ev.get("summary") if isinstance(ev.get("summary"), dict) else {},
        "verdicts": ev.get("verdicts") if isinstance(ev.get("verdicts"), list) else [],
        "workspace_flags": (ev.get("workspace_flags")
                            if isinstance(ev.get("workspace_flags"), list) else []),
        "target_conflicts": (ev.get("target_conflicts")
                             if isinstance(ev.get("target_conflicts"), list) else []),
        "asb_workflow_rubrics": (ev.get("asb_workflow_rubrics")
                                 if isinstance(ev.get("asb_workflow_rubrics"), dict)
                                 else None),
        "result_evaluations": (ev.get("result_evaluations")
                               if isinstance(ev.get("result_evaluations"), dict)
                               else None),
        "judge": judge,
    }


def _eval_match(doc: dict[str, Any], run_id: str) -> str | None:
    """How this eval document names *run_id*: "subject", "substring", or None.

    Structured-first: a document carrying ``evaluation.subject.run_id`` is
    matched on equality alone — a structured id naming a DIFFERENT run never
    falls back to text matching. Legacy documents (no subject id) match on
    the run id appearing in the name or an input's description/source, since
    the eval output tree's layout (<capsule>/<task>/) carries no run id of
    its own.
    """
    subject_id = _dict(_dict(doc.get("evaluation")).get("subject")).get("run_id")
    if isinstance(subject_id, str) and subject_id:
        return "subject" if subject_id == run_id else None
    name = str(doc.get("name") or "")
    inputs_text = " ".join(
        str(i.get("description") or "") + str(i.get("source") or "")
        for i in (doc.get("inputs") or []) if isinstance(i, dict)
    )
    return "substring" if run_id in name + inputs_text else None


def find_evaluations(run_id: str) -> list[dict[str, Any]]:
    """Every asb_eval capsule under ``eval_dir`` that names *run_id*.

    Each summary records which path matched under ``matched_by``
    ("subject" = structured ``evaluation.subject.run_id`` equality,
    "substring" = legacy document-text fallback).
    """
    eval_dir = get_settings().eval_dir
    if not eval_dir.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for path in sorted(eval_dir.rglob("eval_astra.yaml")):
        doc = _read_yaml(path)
        if not isinstance(doc, dict):
            continue
        matched_by = _eval_match(doc, run_id)
        if matched_by is None:
            continue
        summary = _summarise_eval(doc, path.relative_to(eval_dir))
        if summary is not None:
            summary["matched_by"] = matched_by
            found.append(summary)
    return found


def provenance(run_id: str) -> dict[str, Any]:
    """Everything the Provenance tab shows for one run."""
    capsule = read_astra_capsule(run_id)
    return {
        "run_id": run_id,
        "astra": capsule,
        # Only meaningful when this run has no capsule of its own.
        "family_capsules": [] if capsule else family_capsule_ids(run_id),
        "evaluations": find_evaluations(run_id),
    }


__all__ = ["provenance", "read_astra_capsule", "family_capsule_ids",
           "find_evaluations"]

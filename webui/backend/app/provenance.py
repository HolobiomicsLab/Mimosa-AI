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


# ── the run's own ASTRA capsule ──────────────────────────────────────────────

def capsule_path(run_id: str) -> Path:
    return get_settings().capsule_dir / run_id


def read_astra_capsule(run_id: str) -> dict[str, Any] | None:
    """The transparency exporter's capsule for *run_id*, or None."""
    doc = _read_yaml(capsule_path(run_id) / "astra.yaml")
    if not isinstance(doc, dict):
        return None
    universes = []
    udir = capsule_path(run_id) / "universes"
    if udir.is_dir():
        for upath in sorted(udir.glob("*.yaml")):
            u = _read_yaml(upath)
            if isinstance(u, dict):
                universes.append(u)
    decisions = doc.get("decisions")
    return {
        "name": doc.get("name"),
        "description": doc.get("description"),
        "version": doc.get("version"),
        "decisions": decisions if isinstance(decisions, dict) else {},
        "outputs": doc.get("outputs") if isinstance(doc.get("outputs"), list) else [],
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
        "judge": judge,
    }


def find_evaluations(run_id: str) -> list[dict[str, Any]]:
    """Every asb_eval capsule under ``eval_dir`` that names *run_id*.

    The evaluator stamps the run id into the capsule's ``name`` ("ASB criteria
    evaluation of Mimosa run <id> on <task>") and into the workspace input's
    description; matching on the document text is deliberate — the eval output
    tree's layout (<capsule>/<task>/) carries no run id of its own.
    """
    eval_dir = get_settings().eval_dir
    if not eval_dir.is_dir():
        return []
    found: list[dict[str, Any]] = []
    for path in sorted(eval_dir.rglob("eval_astra.yaml")):
        doc = _read_yaml(path)
        if not isinstance(doc, dict):
            continue
        name = str(doc.get("name") or "")
        inputs_text = " ".join(
            str(i.get("description") or "") + str(i.get("source") or "")
            for i in (doc.get("inputs") or []) if isinstance(i, dict)
        )
        if run_id not in name + inputs_text:
            continue
        summary = _summarise_eval(doc, path.relative_to(eval_dir))
        if summary is not None:
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

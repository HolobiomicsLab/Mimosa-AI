"""Build and write ASTRA YAML — the analysis file and one universe per export.

ASTRA fields used (see https://astra-spec.org/latest/specification/#decisions):
- ``decisions[*].label``, ``rationale``, ``options``
- ``options[*].label``, ``description``
- ``inputs[*]``, ``outputs[*]``
- universe file with ``id``, ``description``, ``decisions``

Provenance extensions beyond the spec:
- each decision carries ``tags``: one ``trace_step:<N>`` per contributing
  trace step (the join key to ``sources/memory/<uuid>/astra_decision_step_N``
  evidence) plus ``model:<id>`` — the model that produced the source trace
  step — when the saved memory recorded it. Capsules written before version
  0.0.12 carried the model as a per-decision ``model`` key instead; readers
  accept both generations.
- the analysis carries an ``extraction`` block with the decision-extraction
  health counters, so a capsule produced from a degraded extraction (crashed
  LLM calls, malformed responses) is self-describing rather than silently
  thin.
- analysis-level ``tags`` state what is NOT tracked instead of implying it
  is: ``mimosa:output_attribution=untracked`` (per-output decision lists are
  honest-empty ``[]`` — the old writer stamped every decision id on every
  output, which was attribution-shaped noise),
  ``mimosa:decisions=none (extractor produced no decisions)`` when the
  extractor ran and yielded nothing, so future empty capsules are
  distinguishable from pre-extractor ones, and
  ``mimosa:outputs=none (no artefacts captured)`` when the snapshot held no
  files — the outputs list stays honestly empty (pre-0.0.12 capsules carried
  a synthetic ``(none captured)`` output instead; readers accept both).

The recipe field is left intentionally minimal: Mimosa runs Python inside
smolagents rather than a single shell command, so we point reviewers at the
saved trace rather than fabricating a fake command line.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

from sources.transparency.decision_extractor import Decision, ExtractionResult, Option

_ASTRA_VERSION = "0.0.12"
_DEFAULT_UNIVERSE_ID = "best"

_OUTPUT_ATTRIBUTION_TAG = "mimosa:output_attribution=untracked"
_NO_DECISIONS_TAG = "mimosa:decisions=none (extractor produced no decisions)"
_NO_OUTPUTS_TAG = "mimosa:outputs=none (no artefacts captured)"


_RECIPE_FALLBACK_COMMAND = "see agent trace in sources/memory/<run_uuid>/task_*.json"


def build_analysis(
    goal: str,
    best_uuid: str,
    workspace_files: list[str],
    decisions: list[Decision],
    recipe_command: str = _RECIPE_FALLBACK_COMMAND,
    extraction: ExtractionResult | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the dict that will be dumped to ``astra.yaml``.

    ``recipe_command`` is the POSIX command ASTRA stores per output. Callers
    pass ``python recipe.py`` once the run's code has been reconstructed; the
    default is a pointer used only when no executable code was recovered.
    ``extraction`` adds the extraction-health block (see module docstring);
    the exporter always passes it, ``None`` merely keeps old callers working.
    ``environment`` adds the orchestrator-environment block assembled by
    :mod:`sources.transparency.env_capture` (same regime as ``extraction``).
    """
    analysis = {
        "version": _ASTRA_VERSION,
        "name": f"Mimosa best run {best_uuid}",
        "description": (
            "ASTRA export of the best-performing evolved workflow. "
            "Decisions reconstructed post-run from the agent memory trace."
        ),
        "tags": _build_analysis_tags(decisions, workspace_files),
        "inputs": _build_inputs(goal),
        "outputs": _build_outputs(workspace_files, recipe_command),
        "decisions": _build_decisions(decisions),
    }
    if extraction is not None:
        analysis["extraction"] = {
            "steps_considered": extraction.steps_total,
            "decisions_recorded": len(decisions),
            "llm_call_failures": extraction.crashed,
            "malformed_responses": extraction.malformed,
        }
    if environment is not None:
        analysis["environment"] = environment
    return analysis


def build_universe(decisions: list[Decision], best_uuid: str) -> dict[str, Any]:
    """Assemble the universe-file dict pinning every decision to its chosen option."""
    return {
        "id": _DEFAULT_UNIVERSE_ID,
        "description": f"Configuration realised by Mimosa run {best_uuid}.",
        "decisions": {d.id: d.chosen_option_id for d in decisions},
    }


def write_export(
    workspace_dir: Path,
    analysis: dict[str, Any],
    universe: dict[str, Any],
) -> Path:
    """Write ``astra.yaml`` + ``universes/best.yaml`` under ``workspace_dir``."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = workspace_dir / "astra.yaml"
    universe_dir = workspace_dir / "universes"
    universe_dir.mkdir(exist_ok=True)
    universe_path = universe_dir / f"{universe['id']}.yaml"
    _dump(analysis_path, analysis)
    _dump(universe_path, universe)
    return analysis_path


def _build_analysis_tags(
    decisions: list[Decision], workspace_files: list[str]
) -> list[str]:
    """Analysis-level honest-empty tags (see module docstring)."""
    tags = [_OUTPUT_ATTRIBUTION_TAG]
    if not decisions:
        tags.append(_NO_DECISIONS_TAG)
    if not workspace_files:
        tags.append(_NO_OUTPUTS_TAG)
    return tags


def _build_inputs(goal: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "task_description",
            "type": "data",
            "source": "user_goal",
            "description": goal.strip()[:500] or "(empty goal)",
        }
    ]


def _build_outputs(
    workspace_files: list[str],
    recipe_command: str,
) -> list[dict[str, Any]]:
    # An empty snapshot yields an empty outputs list — the analysis-level
    # ``mimosa:outputs=none`` tag carries the reason. The pre-0.0.12 writer
    # synthesized a "(none captured)" output here, which dangled: the manifest
    # never carried a matching entry. Readers still accept those capsules.
    return [
        {
            "id": output_id,
            "type": "data",
            "description": f"Artefact produced by the best run: {f}",
            "inputs": ["task_description"],
            # Per-output decision attribution is not tracked; an honest empty
            # list (plus the analysis-level tag) beats stamping every decision
            # id on every output as the pre-0.0.12 writer did.
            "decisions": [],
            "recipe": {"command": recipe_command},
        }
        for output_id, f in zip(
            unique_output_ids(workspace_files), workspace_files, strict=True
        )
    ]


def _build_decisions(decisions: list[Decision]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for d in decisions:
        tags = [f"trace_step:{n}" for n in d.source_steps]
        if d.model:
            tags.append(f"model:{d.model}")
        entries[d.id] = {
            "label": d.label,
            "rationale": d.rationale,
            "default": d.chosen_option_id,
            "options": {
                o.id: {"label": o.label, "description": o.description}
                for o in d.options
            },
            "tags": tags,
        }
    return entries


def safe_output_id(filename: str, index: int) -> str:
    """Slugify ``filename`` into a stable ASTRA output id (also the manifest key)."""
    import re
    stem = re.sub(r"[^a-z0-9_]+", "_", filename.lower()).strip("_")
    if not stem or not stem[0].isalpha():
        stem = f"output_{index}"
    return stem[:48]


def unique_output_ids(filenames: list[str]) -> list[str]:
    """One collision-free output id per filename, in order.

    ``safe_output_id`` can collapse distinct filenames onto one slug
    (punctuation squashing, 48-char truncation). A colliding slug would make
    one output id silently carry another file's digest in the manifest — the
    exact misattribution a content-attestation artefact exists to prevent —
    so collisions get a ``_<n>`` suffix (trimmed to stay within 48 chars).
    This is THE shared id source: ``_build_outputs`` and the exporter's
    ``outputs_manifest.json`` both call it, keeping astra.yaml output ids and
    manifest keys unique and in lock-step.
    """
    ids: list[str] = []
    taken: set[str] = set()
    for index, name in enumerate(filenames):
        slug = safe_output_id(name, index)
        candidate, n = slug, index
        while candidate in taken:
            tail = f"_{n}"
            candidate = slug[: 48 - len(tail)] + tail
            n += 1
        taken.add(candidate)
        ids.append(candidate)
    return ids


def _dump(path: Path, data: dict[str, Any]) -> None:
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    import tempfile
    decisions = [
        Decision(
            id="fit_method", label="Fit", rationale="r",
            chosen_option_id="ols",
            options=(
                Option(id="ols", label="OLS", description="Minimises squared residuals."),
                Option(id="robust", label="Robust regression", description="Down-weights outliers."),
            ),
            source_step=2,
            model="openrouter/qwen/qwen3.7-plus",
        ),
    ]
    extraction = ExtractionResult(
        decisions=tuple(decisions), steps_total=5, crashed=1, malformed=2
    )
    analysis = build_analysis(
        "Predict X.", "abc-123", ["model.pkl", "report.md"], decisions,
        extraction=extraction,
    )
    universe = build_universe(decisions, "abc-123")
    with tempfile.TemporaryDirectory() as tmp:
        out = write_export(Path(tmp), analysis, universe)
        assert out.exists() and out.name == "astra.yaml"
        loaded = yaml.safe_load(out.read_text())
        assert loaded["version"] == "0.0.12", loaded["version"]
        entry = loaded["decisions"]["fit_method"]
        assert entry["default"] == "ols"
        assert set(entry["options"]) == {"ols", "robust"}
        # Provenance now travels as tags; the legacy per-decision "model" key
        # is no longer written (readers still accept it on old capsules).
        assert entry["tags"] == ["trace_step:2", "model:openrouter/qwen/qwen3.7-plus"]
        assert "model" not in entry
        assert loaded["inputs"][0]["type"] == "data"
        for output in loaded["outputs"]:
            assert output["type"] == "data" and output["decisions"] == []
        assert loaded["tags"] == ["mimosa:output_attribution=untracked"]
        assert loaded["extraction"] == {
            "steps_considered": 5,
            "decisions_recorded": 1,
            "llm_call_failures": 1,
            "malformed_responses": 2,
        }, loaded["extraction"]
        uni = yaml.safe_load((Path(tmp) / "universes" / "best.yaml").read_text())
        assert uni["decisions"]["fit_method"] == "ols"

    empty = build_analysis("goal", "abc-123", [], [])
    assert _NO_DECISIONS_TAG in empty["tags"], empty["tags"]
    assert _NO_OUTPUTS_TAG in empty["tags"], empty["tags"]
    assert empty["outputs"] == [], empty["outputs"]

    colliding = unique_output_ids(["a-b.csv", "a b.csv"])
    assert colliding == ["a_b_csv", "a_b_csv_1"], colliding
    assert len(set(unique_output_ids(["a_b_csv_1", "a-b.csv", "a b.csv"]))) == 3
    print("[OK] yaml_writer smoke check passed")

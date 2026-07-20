"""Build and write ASTRA YAML — the analysis file and one universe per export.

ASTRA fields used (see https://astra-spec.org/latest/specification/#decisions):
- ``decisions[*].label``, ``rationale``, ``options``
- ``options[*].label``, ``description``
- ``inputs[*]``, ``outputs[*]``
- universe file with ``id``, ``description``, ``decisions``

Two provenance extensions beyond the spec:
- each decision carries ``model`` — the model id that produced the source
  trace step — when the saved memory recorded it (older traces predate the
  field and simply omit it);
- the analysis carries an ``extraction`` block with the decision-extraction
  health counters, so a capsule produced from a degraded extraction (crashed
  LLM calls, malformed responses) is self-describing rather than silently
  thin.

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


_ASTRA_VERSION = "0.1"
_DEFAULT_UNIVERSE_ID = "best"


_RECIPE_FALLBACK_COMMAND = "see agent trace in sources/memory/<run_uuid>/task_*.json"


def build_analysis(
    goal: str,
    best_uuid: str,
    workspace_files: list[str],
    decisions: list[Decision],
    recipe_command: str = _RECIPE_FALLBACK_COMMAND,
    extraction: ExtractionResult | None = None,
) -> dict[str, Any]:
    """Assemble the dict that will be dumped to ``astra.yaml``.

    ``recipe_command`` is the POSIX command ASTRA stores per output. Callers
    pass ``python recipe.py`` once the run's code has been reconstructed; the
    default is a pointer used only when no executable code was recovered.
    ``extraction`` adds the extraction-health block (see module docstring);
    the exporter always passes it, ``None`` merely keeps old callers working.
    """
    analysis = {
        "version": _ASTRA_VERSION,
        "name": f"Mimosa best run {best_uuid}",
        "description": (
            "ASTRA export of the best-performing evolved workflow. "
            "Decisions reconstructed post-run from the agent memory trace."
        ),
        "inputs": _build_inputs(goal),
        "outputs": _build_outputs(workspace_files, decisions, recipe_command),
        "decisions": _build_decisions(decisions),
    }
    if extraction is not None:
        analysis["extraction"] = {
            "steps_considered": extraction.steps_total,
            "decisions_recorded": len(decisions),
            "llm_call_failures": extraction.crashed,
            "malformed_responses": extraction.malformed,
        }
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


def _build_inputs(goal: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "task_description",
            "type": "text",
            "source": "user_goal",
            "description": goal.strip()[:500] or "(empty goal)",
        }
    ]


def _build_outputs(
    workspace_files: list[str],
    decisions: list[Decision],
    recipe_command: str,
) -> list[dict[str, Any]]:
    decision_ids = [d.id for d in decisions]
    if not workspace_files:
        workspace_files = ["(none captured)"]
    return [
        {
            "id": _safe_output_id(f, i),
            "type": "file",
            "description": f"Artefact produced by the best run: {f}",
            "inputs": ["task_description"],
            "decisions": decision_ids,
            "recipe": {"command": recipe_command},
        }
        for i, f in enumerate(workspace_files)
    ]


def _build_decisions(decisions: list[Decision]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for d in decisions:
        entry: dict[str, Any] = {
            "label": d.label,
            "rationale": d.rationale,
            "default": d.chosen_option_id,
            "options": {
                o.id: {"label": o.label, "description": o.description}
                for o in d.options
            },
        }
        if d.model:
            entry["model"] = d.model
        entries[d.id] = entry
    return entries


def _safe_output_id(filename: str, index: int) -> str:
    import re
    stem = re.sub(r"[^a-z0-9_]+", "_", filename.lower()).strip("_")
    if not stem or not stem[0].isalpha():
        stem = f"output_{index}"
    return stem[:48]


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
        assert loaded["decisions"]["fit_method"]["default"] == "ols"
        assert set(loaded["decisions"]["fit_method"]["options"]) == {"ols", "robust"}
        assert loaded["decisions"]["fit_method"]["model"] == "openrouter/qwen/qwen3.7-plus"
        assert loaded["extraction"] == {
            "steps_considered": 5,
            "decisions_recorded": 1,
            "llm_call_failures": 1,
            "malformed_responses": 2,
        }, loaded["extraction"]
        uni = yaml.safe_load((Path(tmp) / "universes" / "best.yaml").read_text())
        assert uni["decisions"]["fit_method"] == "ols"
    print("[OK] yaml_writer smoke check passed")

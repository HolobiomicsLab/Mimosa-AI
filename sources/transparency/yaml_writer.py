"""Build and write ASTRA YAML — the analysis file and one universe per export.

ASTRA fields used (see https://astra-spec.org/latest/specification/#decisions):
- ``decisions[*].label``, ``rationale``, ``options``
- ``options[*].label``, ``description``
- ``inputs[*]``, ``outputs[*]``
- universe file with ``id``, ``description``, ``decisions``

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

from sources.transparency.decision_extractor import Decision


_ASTRA_VERSION = "0.1"
_DEFAULT_UNIVERSE_ID = "best"


_RECIPE_FALLBACK_COMMAND = "see agent trace in sources/memory/<run_uuid>/task_*.json"


def build_analysis(
    goal: str,
    best_uuid: str,
    workspace_files: list[str],
    decisions: list[Decision],
    recipe_command: str = _RECIPE_FALLBACK_COMMAND,
) -> dict[str, Any]:
    """Assemble the dict that will be dumped to ``astra.yaml``.

    ``recipe_command`` is the POSIX command ASTRA stores per output. Callers
    pass ``python recipe.py`` once the run's code has been reconstructed; the
    default is a pointer used only when no executable code was recovered.
    """
    return {
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


def build_universe(decisions: list[Decision], best_uuid: str) -> dict[str, Any]:
    """Assemble the universe-file dict pinning every decision to its chosen option."""
    return {
        "id": _DEFAULT_UNIVERSE_ID,
        "description": f"Configuration realised by Mimosa run {best_uuid}.",
        "decisions": {d.id: d.option_id for d in decisions},
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
    if not decisions:
        return {}
    return {
        d.id: {
            "label": d.label,
            "rationale": d.rationale,
            "default": d.option_id,
            "options": {
                d.option_id: {
                    "label": d.option_label,
                    "description": d.option_description,
                }
            },
        }
        for d in decisions
    }


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
            option_id="ols", option_label="OLS", option_description="d",
            source_step=2,
        ),
    ]
    analysis = build_analysis("Predict X.", "abc-123", ["model.pkl", "report.md"], decisions)
    universe = build_universe(decisions, "abc-123")
    with tempfile.TemporaryDirectory() as tmp:
        out = write_export(Path(tmp), analysis, universe)
        assert out.exists() and out.name == "astra.yaml"
        loaded = yaml.safe_load(out.read_text())
        assert loaded["decisions"]["fit_method"]["default"] == "ols"
        uni = yaml.safe_load((Path(tmp) / "universes" / "best.yaml").read_text())
        assert uni["decisions"]["fit_method"] == "ols"
    print("[OK] yaml_writer smoke check passed")

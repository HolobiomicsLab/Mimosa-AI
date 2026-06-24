"""Orchestrate the post-run ASTRA export for the best evolved workflow.

Pipeline
--------
1. Locate the best run's memory directory (``<config.memory_dir>/<uuid>``).
2. Load the smolagents trace and strip the bloat — see
   :mod:`sources.transparency.trace_compaction`.
3. Run a per-step decision-extraction LLM pass — see
   :mod:`sources.transparency.decision_extractor`.
4. Assemble the analysis + universe dicts and write them under the restored
   workspace as ``astra.yaml`` and ``universes/best.yaml``.

The exporter is wired into :func:`start_workflow_evolution` immediately
after ``workspace_mgr.restore_best`` so the YAML lands next to the
artefacts of the same best run.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

from sources.transparency.decision_extractor import extract_decisions
from sources.transparency.trace_compaction import compact_trace, load_trace
from sources.transparency.yaml_writer import (
    build_analysis,
    build_universe,
    write_export,
)


class AstraExporter:
    """Build an ASTRA-spec YAML pair from the best run's saved memory."""

    def __init__(self, config, logger: logging.Logger | None = None) -> None:
        """Bind to the running :class:`config.Config` and the engine's logger."""
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def export(self, best_uuid: str, goal: str) -> Path | None:
        """Run the pipeline and return the analysis path (None on failure).

        Args:
            best_uuid: UUID of the run selected as best by the evolution loop.
            goal: User-provided goal text — supplied to the LLM as context.
        """
        from sources.cli.pretty_print import (
            print_info, print_ok, print_section, print_warn,
        )
        print_section("ASTRA EXPORT")
        memory_path = Path(self.config.memory_dir) / best_uuid
        workspace_dir = Path(self.config.workspace_dir)
        if not memory_path.is_dir():
            print_warn(f"Memory directory missing: {memory_path}; export skipped.")
            return None

        raw_steps = load_trace(memory_path)
        compact = compact_trace(raw_steps)
        print_info(
            f"Trace compacted: {len(raw_steps)} raw → {len(compact)} candidate steps."
        )

        llm_config = self._build_llm_config()
        decisions = extract_decisions(compact, goal, memory_path, llm_config)
        print_info(f"Decisions extracted: {len(decisions)}.")

        workspace_files = self._list_workspace_files(workspace_dir)
        analysis = build_analysis(goal, best_uuid, workspace_files, decisions)
        universe = build_universe(decisions, best_uuid)
        path = write_export(workspace_dir, analysis, universe)
        print_ok(f"ASTRA analysis written to {path}")
        return path

    def _build_llm_config(self):
        """Reuse the project's judge model for cheap structured extraction."""
        from sources.core.llm_provider import LLMConfig
        model = getattr(self.config, "judge_model", None) or "anthropic/claude-sonnet-4-5"
        provider = model.split("/", 1)[0] if "/" in model else "anthropic"
        return LLMConfig(
            model=model,
            provider=provider,
            temperature=0.0,
            openrouter_provider=self.config.openrouter_provider_for(model),
        )

    def _list_workspace_files(self, workspace_dir: Path) -> list[str]:
        """Top-level files in the restored workspace, sorted, excluding our own output."""
        if not workspace_dir.is_dir():
            return []
        excluded = {"astra.yaml", "universes"}
        return sorted(
            p.name for p in workspace_dir.iterdir()
            if p.is_file() and p.name not in excluded
        )


if __name__ == "__main__":
    import sys
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as tmp:
        memory_dir = Path(tmp) / "memory"
        workspace_dir = Path(tmp) / "workspace"
        run_uuid = "smoke-uuid"
        (memory_dir / run_uuid).mkdir(parents=True)
        workspace_dir.mkdir()
        (workspace_dir / "model.pkl").write_text("fake")
        (memory_dir / run_uuid / "task_single_agent.json").write_text(json.dumps([
            {
                "model_output_message": {"content": "Plot."},
                "action_output": "import matplotlib.pyplot as plt\nplt.savefig('x.png')",
                "observations": "saved",
            },
        ]))
        config = SimpleNamespace(
            memory_dir=str(memory_dir),
            workspace_dir=str(workspace_dir),
            judge_model="anthropic/claude-sonnet-4-5",
            openrouter_provider_for=lambda _m: None,
        )
        exporter = AstraExporter(config)
        compact = compact_trace(load_trace(memory_dir / run_uuid))
        assert compact == [], f"Mechanical step should be filtered, got {compact}"
        analysis = build_analysis("smoke goal", run_uuid, ["model.pkl"], [])
        universe = build_universe([], run_uuid)
        out = write_export(workspace_dir, analysis, universe)
        assert out.exists()
        print("[OK] astra_exporter smoke check passed")
    sys.exit(0)

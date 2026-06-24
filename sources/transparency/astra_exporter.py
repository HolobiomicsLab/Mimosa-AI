"""Orchestrate the post-run ASTRA export for the best evolved workflow.

Pipeline
--------
1. Locate the best run's memory directory (``<config.memory_dir>/<uuid>``).
2. Load the smolagents trace and strip the bloat — see
   :mod:`sources.transparency.trace_compaction`.
3. Run a per-step decision-extraction LLM pass — see
   :mod:`sources.transparency.decision_extractor`.
4. Assemble the analysis + universe dicts and write them into the run's
   capsule subfolder as ``astra.yaml`` and ``universes/best.yaml``.

Output target: ``<config.runs_capsule_dir>/<best_uuid>/``. The best run's
UUID is used as the capsule subfolder name (stable, deterministic, mirrors
the ``sources/memory/<uuid>`` convention) — it does NOT depend on the
LLM-named goal capsule produced later by ``LocalTransfer``.

The exporter is wired into :func:`start_workflow_evolution` immediately
after ``workspace_mgr.restore_best`` and is gated on ``config.export_astra``.
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

    # WorkspaceManager snapshots best-run artefacts under this root with the
    # naming pattern ``mimosa_run_<session_id>_<run_uuid>``. Overridable for
    # tests so the lookup can target a sandbox rather than the real ``/tmp``.
    _SNAPSHOT_ROOT: Path = Path("/tmp")

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
        capsule_dir = Path(self.config.runs_capsule_dir) / best_uuid
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

        artefacts_dir = self._resolve_artefacts_dir(best_uuid, workspace_dir)
        if artefacts_dir != workspace_dir:
            print_info(f"Reading artefacts from /tmp snapshot: {artefacts_dir}")
        workspace_files = self._list_workspace_files(artefacts_dir)
        analysis = build_analysis(goal, best_uuid, workspace_files, decisions)
        universe = build_universe(decisions, best_uuid)
        path = write_export(capsule_dir, analysis, universe)
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

    def _resolve_artefacts_dir(self, best_uuid: str, workspace_dir: Path) -> Path:
        """Prefer the run's saved /tmp snapshot, fall back to the live workspace.

        The snapshot is the canonical source — ``WorkspaceManager.restore_best``
        copies it into ``workspace_dir`` just before the export runs. Falling
        back keeps standalone mode functional after ``cleanup()`` has wiped
        ``/tmp``.
        """
        pattern = f"mimosa_run_*_{best_uuid}"
        for snapshot in self._SNAPSHOT_ROOT.glob(pattern):
            if snapshot.is_dir():
                return snapshot
        return workspace_dir

    def _list_workspace_files(self, workspace_dir: Path) -> list[str]:
        """Top-level files in the restored workspace, sorted, excluding our own output."""
        if not workspace_dir.is_dir():
            return []
        excluded = {"astra.yaml", "universes"}
        return sorted(
            p.name for p in workspace_dir.iterdir()
            if p.is_file() and p.name not in excluded
        )


def _resolve_goal(memory_dir: Path, uuid: str, override: str | None) -> str:
    """Read the goal from ``state_result.json`` unless an override is given."""
    if override:
        return override
    state_path = memory_dir / uuid / "state_result.json"
    if not state_path.exists():
        return "(unknown goal — pass --goal to override)"
    try:
        return json.loads(state_path.read_text()).get("goal") or "(unknown goal)"
    except (OSError, json.JSONDecodeError):
        return "(unknown goal — state_result.json unreadable)"


def _run_standalone(uuid: str, goal_override: str | None) -> int:
    """Export ASTRA for a real memory UUID using the project's Config."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from config import Config
    config = Config()
    goal = _resolve_goal(Path(config.memory_dir), uuid, goal_override)
    path = AstraExporter(config).export(uuid, goal)
    if path is None:
        print("[FAIL] export returned None — see warnings above.")
        return 1
    print(f"[OK] ASTRA export wrote {path}")
    return 0


def _run_smoke_check() -> None:
    """Self-contained sanity check — sandboxed in /var/folders, leaves nothing behind."""
    import tempfile
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory() as tmp:
        memory_dir = Path(tmp) / "memory"
        workspace_dir = Path(tmp) / "workspace"
        capsule_dir = Path(tmp) / "runs_capsule"
        run_uuid = "smoke-uuid"
        (memory_dir / run_uuid).mkdir(parents=True)
        workspace_dir.mkdir()
        capsule_dir.mkdir()
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
            runs_capsule_dir=str(capsule_dir),
            judge_model="openrouter/deepseek/deepseek-v4-flash",
            openrouter_provider_for=lambda _m: None,
        )
        AstraExporter(config)  # construction smoke
        compact = compact_trace(load_trace(memory_dir / run_uuid))
        assert compact == [], f"Mechanical step should be filtered, got {compact}"
        analysis = build_analysis("smoke goal", run_uuid, ["model.pkl"], [])
        universe = build_universe([], run_uuid)
        out = write_export(capsule_dir / run_uuid, analysis, universe)
        assert out.exists() and out.parent == capsule_dir / run_uuid, out
        print(f"[OK] astra_exporter smoke check passed (wrote {out})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "ASTRA exporter. Without args, runs a self-cleaning smoke check. "
            "With --uuid, exports a real run's memory under sources/memory/<uuid> "
            "to runs_capsule/<uuid>/."
        )
    )
    parser.add_argument("--uuid", help="Run UUID under config.memory_dir to export.")
    parser.add_argument("--goal", help="Override goal text (default: read from state_result.json).")
    args = parser.parse_args()
    if args.uuid:
        sys.exit(_run_standalone(args.uuid, args.goal))
    _run_smoke_check()
    sys.exit(0)

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

import hashlib
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
from sources.transparency.env_capture import capture_environment
from sources.transparency.memory_trace import RECIPE_FILENAME, reconstruct_recipe
from sources.transparency.trace_compaction import compact_trace, load_trace
from sources.transparency.yaml_writer import (
    build_analysis,
    build_universe,
    safe_output_id,
    write_export,
)

OUTPUTS_MANIFEST_FILENAME = "outputs_manifest.json"


# Names never worth listing as scientific outputs.
_ARTEFACT_IGNORE = {".DS_Store", "Thumbs.db", ".gitkeep", RECIPE_FILENAME}


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
            print_info,
            print_ok,
            print_section,
            print_warn,
        )
        print_section("ASTRA EXPORT")
        memory_path = Path(self.config.memory_dir) / best_uuid
        workspace_dir = Path(self.config.workspace_dir)
        capsule_dir = Path(self.config.runs_capsule_dir) / best_uuid
        if not memory_path.is_dir():
            print_warn(f"Memory directory missing: {memory_path}; export skipped.")
            return None

        raw_steps = load_trace(memory_path)
        if not raw_steps:
            print_warn(f"Empty trace for {best_uuid}; ASTRA export skipped.")
            return None
        compact = compact_trace(raw_steps)
        print_info(
            f"Trace compacted: {len(raw_steps)} raw → {len(compact)} candidate steps."
        )

        llm_config = self._build_llm_config()
        extraction = extract_decisions(compact, goal, memory_path, llm_config)
        decisions = list(extraction.decisions)
        print_info(f"Decisions extracted: {len(decisions)}.")
        if extraction.crashed or extraction.malformed:
            print_warn(
                f"Extraction degraded: of {extraction.steps_total} steps, "
                f"{extraction.crashed} LLM calls crashed and "
                f"{extraction.malformed} returned malformed output; "
                "counts are recorded in astra.yaml under 'extraction'."
            )

        artefacts_dir = self._resolve_artefacts_dir(best_uuid, workspace_dir)
        if artefacts_dir != workspace_dir:
            print_info(f"Reading artefacts from /tmp snapshot: {artefacts_dir}")
        workspace_files = self._list_workspace_files(artefacts_dir)

        recipe_command = self._write_recipe(capsule_dir, memory_path)
        environment = capture_environment(
            self.config, memory_path, self._run_metrics_path(best_uuid)
        )
        analysis = build_analysis(
            goal, best_uuid, workspace_files, decisions, recipe_command,
            extraction=extraction, environment=environment,
        )
        universe = build_universe(decisions, best_uuid)
        path = write_export(capsule_dir, analysis, universe)
        self._write_outputs_manifest(capsule_dir, artefacts_dir, workspace_files)
        print_ok(f"ASTRA analysis written to {path}")
        return path

    def _write_outputs_manifest(
        self, capsule_dir: Path, artefacts_dir: Path, workspace_files: list[str]
    ) -> Path:
        """Write ``outputs_manifest.json`` beside ``astra.yaml``.

        Pure projection of the artefact snapshot: one entry per exported
        output, keyed by the same slug the analysis uses as output id, with
        the file's relative name, byte size and sha256 content digest.
        Fail-soft per file — an unreadable artefact becomes an entry with an
        ``error`` reason (no absolute paths in it), never a crash. An empty
        snapshot writes ``{}``, which is itself the honest record.
        """
        manifest: dict[str, dict] = {}
        for index, name in enumerate(workspace_files):
            manifest[safe_output_id(name, index)] = self._manifest_entry(
                artefacts_dir / name, name
            )
        manifest_path = capsule_dir / OUTPUTS_MANIFEST_FILENAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest_path

    @staticmethod
    def _manifest_entry(path: Path, relative_name: str) -> dict:
        """One manifest entry; ``error`` instead of digest when unreadable."""
        try:
            data = path.read_bytes()
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            return {"path": relative_name, "error": f"unreadable ({reason})"}
        return {
            "path": relative_name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def _write_recipe(self, capsule_dir: Path, memory_path: Path) -> str:
        """Reconstruct the run's code into ``recipe.py`` and return its command.

        Falls back to a pointer command when no executable code was recovered,
        so the analysis stays valid for runs that produced no code steps.
        """
        from sources.transparency.yaml_writer import _RECIPE_FALLBACK_COMMAND
        code = reconstruct_recipe(memory_path)
        if not code.strip():
            return _RECIPE_FALLBACK_COMMAND
        capsule_dir.mkdir(parents=True, exist_ok=True)
        (capsule_dir / RECIPE_FILENAME).write_text(code)
        return f"python {RECIPE_FILENAME}"

    def _build_llm_config(self):
        """Reuse the project's judge model for cheap structured extraction.

        Splits ``judge_model`` into ``(provider, bare_model)`` the way every
        other consumer does (e.g. ``sources/cli/memory_chat_cli._build_llm``).
        This matters: ``LLMProvider.__call__`` re-prefixes the model as
        ``f"{provider}/{model}"``, so passing the full ``provider/model``
        string as ``model`` double-prefixes it (``openrouter/openrouter/...``)
        and every extraction call fails. The OpenRouter routing + quantization
        filter are looked up per-model so routing matches the rest of the run.
        """
        from sources.core.llm_provider import LLMConfig, extract_model_pattern
        judge = getattr(self.config, "smolagent_model_id", None) or "openrouter/deepseek/deepseek-v4-flash"
        # `smolagent_model_id` may be a list of candidate models; the judge is a
        # single model, so resolve to the primary (first).
        if isinstance(judge, list):
            judge = judge[0] if judge else "openrouter/deepseek/deepseek-v4-flash"
        provider, model = extract_model_pattern(judge)
        return LLMConfig(
            model=model,
            provider=provider,
            temperature=0.0,
            openrouter_provider=self.config.openrouter_provider_for(judge),
            openrouter_quantizations=self.config.openrouter_quantizations_for(judge),
        )

    def _run_metrics_path(self, best_uuid: str) -> Path | None:
        """``sources/workflows/<uuid>/run_metrics.json``; None when unconfigured."""
        workflow_dir = getattr(self.config, "workflow_dir", None)
        if not workflow_dir:
            return None
        return Path(workflow_dir) / best_uuid / "run_metrics.json"

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
        """Top-level artefact files, sorted; drops our own output and OS junk."""
        if not workspace_dir.is_dir():
            return []
        excluded = {"astra.yaml", "universes"} | _ARTEFACT_IGNORE
        return sorted(
            p.name for p in workspace_dir.iterdir()
            if p.is_file() and p.name not in excluded and not p.name.startswith(".")
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
        (workspace_dir / ".DS_Store").write_text("junk")
        (memory_dir / run_uuid / "task_single_agent.json").write_text(json.dumps([
            {
                "step_number": 1,
                "model_output_message": {"content": "Plot."},
                "code_action": "import matplotlib.pyplot as plt\nplt.savefig('x.png')",
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
        exporter = AstraExporter(config)
        compact = compact_trace(load_trace(memory_dir / run_uuid))
        assert compact == [], f"Mechanical step should be filtered, got {compact}"
        files = exporter._list_workspace_files(workspace_dir)
        assert files == ["model.pkl"], f".DS_Store should be dropped, got {files}"
        cmd = exporter._write_recipe(capsule_dir / run_uuid, memory_dir / run_uuid)
        assert cmd == "python recipe.py", cmd
        recipe = (capsule_dir / run_uuid / "recipe.py").read_text()
        assert "plt.savefig" in recipe and "step 1 · single_agent" in recipe, recipe
        analysis = build_analysis("smoke goal", run_uuid, files, [], cmd)
        universe = build_universe([], run_uuid)
        out = write_export(capsule_dir / run_uuid, analysis, universe)
        assert out.exists() and out.parent == capsule_dir / run_uuid, out
        manifest_path = exporter._write_outputs_manifest(
            capsule_dir / run_uuid, workspace_dir, files
        )
        manifest = json.loads(manifest_path.read_text())
        assert manifest["model_pkl"]["bytes"] == len("fake"), manifest
        assert len(manifest["model_pkl"]["sha256"]) == 64, manifest
        environment = capture_environment(config, memory_dir / run_uuid, None)
        assert environment["runner_env"].startswith("partial"), environment
        assert environment["model_roles"]["judge_model"], environment
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

"""
Workspace lifecycle management for multi-agent evolution sessions.

Provides a stateful WorkspaceManager that keeps the workspace consistent
across evolution iterations:
  1. Snapshots the user-provided workspace before evolution starts.
  2. Resets the workspace to that pristine state before every workflow run.
  3. Saves each run's output to /tmp so the best result can be restored.
  4. Restores the workspace to the best-performing run's snapshot.
  5. Deletes all /tmp artefacts at the end of the session.
"""

import logging
import shutil
import uuid as _uuid_mod
from pathlib import Path

from sources.utils.transfer_toolomics import LocalTransfer
from sources.cli.pretty_print import print_info, print_ok, print_warn, print_section


class WorkspaceManager:
    """
    Manages workspace state across a single evolution session.

    Typical call sequence
    ---------------------
    mgr = WorkspaceManager(config)
    mgr.begin_session()              # snapshot + clean + restore initial state

    # inside evolve_generation loop:
    mgr.reset_for_run()              # restore initial state before each run
    ...execute workflow...
    mgr.save_run_snapshot(uuid)      # snapshot result for this run

    # in start_workflow_evolution after all iterations:
    mgr.restore_best(best_uuid)      # put best result into workspace
    mgr.cleanup()                    # delete all /tmp artefacts
    """

    def __init__(self, config, logger: logging.Logger = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._transfer: LocalTransfer | None = None
        self._session_id: str | None = None
        self._initial_backup: Path | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def begin_session(self) -> None:
        """
        Start a new evolution session:
          - Generate a unique session ID.
          - Create a LocalTransfer bound to the workspace.
          - Snapshot the current (user-provided) workspace to /tmp.
          - Clean the workspace and restore it from that snapshot so agents
            always start from a known-good baseline.
        """
        self._session_id = _uuid_mod.uuid4().hex[:12]
        self._transfer = LocalTransfer(self.config, self.config.workspace_dir)

        print_section("WORKSPACE SETUP")
        self._initial_backup = self._backup_initial_workspace()
        self._restore_from(self._initial_backup)
        print_ok("Workspace reset to initial user-provided state.")

    def reset_for_run(self) -> None:
        """
        Reset the workspace to the pristine initial state before a workflow run.
        Must be called after begin_session().
        """
        self._assert_session()
        print_info("Resetting workspace to initial state before workflow execution …")
        self._restore_from(self._initial_backup)

    def save_run_snapshot(self, run_uuid: str) -> Path:
        """
        Snapshot the workspace output produced by *run_uuid* into /tmp.
        Returns the snapshot path.
        Must be called after begin_session().
        """
        self._assert_session()
        snapshot = Path(f"/tmp/mimosa_run_{self._session_id}_{run_uuid}")
        workspace = Path(self.config.workspace_dir)
        if workspace.exists() and workspace.is_dir():
            snapshot.mkdir(parents=True, exist_ok=True)
            self._transfer.copy_files_recursive(workspace, snapshot)
            self.logger.info(f"[WORKSPACE] Run {run_uuid} snapshot saved to {snapshot}")
        return snapshot

    def restore_best(self, best_uuid: str) -> None:
        """
        Wipe the workspace and restore it from the snapshot of *best_uuid*.
        Falls back to the initial user-provided state if the snapshot is missing.
        Must be called after begin_session().
        """
        self._assert_session()
        best_snapshot = Path(f"/tmp/mimosa_run_{self._session_id}_{best_uuid}")
        print_section("WORKSPACE RESTORE")
        if best_snapshot.exists() and best_snapshot.is_dir():
            print_info(f"Restoring workspace from best run: {best_uuid}")
            self._restore_from(best_snapshot)
            print_ok("Workspace restored to best-performing workflow output.")
        else:
            print_warn(
                f"Snapshot for best run {best_uuid} not found; "
                "restoring initial user-provided state."
            )
            self._restore_from(self._initial_backup)

    def cleanup(self) -> None:
        """
        Delete every /tmp directory created for this session.
        Safe to call even if begin_session() was never called.
        """
        if self._session_id is None:
            return

        candidates = [Path(f"/tmp/mimosa_initial_{self._session_id}")]
        candidates.extend(
            Path("/tmp").glob(f"mimosa_run_{self._session_id}_*")
        )
        for p in candidates:
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
                self.logger.info(f"[WORKSPACE] Cleaned up tmp directory: {p}")
        print_info(f"Temporary evolution directories for session {self._session_id} removed.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _assert_session(self) -> None:
        if self._session_id is None or self._transfer is None:
            raise RuntimeError(
                "WorkspaceManager.begin_session() must be called before this operation."
            )

    def _backup_initial_workspace(self) -> Path:
        """Copy workspace → /tmp/mimosa_initial_<session_id>/ and return that path."""
        backup = Path(f"/tmp/mimosa_initial_{self._session_id}")
        workspace = Path(self.config.workspace_dir)
        backup.mkdir(parents=True, exist_ok=True)
        if workspace.exists() and workspace.is_dir() and any(workspace.iterdir()):
            self._transfer.copy_files_recursive(workspace, backup)
            self.logger.info(f"[WORKSPACE] Initial state backed up to {backup}")
            print_info(f"Initial workspace backed up to: {backup}")
        else:
            self.logger.info("[WORKSPACE] Workspace is empty; created empty initial backup.")
            print_info("Workspace is empty; initial backup directory created.")
        return backup

    def _restore_from(self, source: Path) -> None:
        """Wipe the workspace and populate it from *source*."""
        self._transfer.clean_workspace()
        if source.exists() and source.is_dir() and any(source.iterdir()):
            workspace = Path(self.config.workspace_dir)
            self._transfer.copy_files_recursive(source, workspace)
            self.logger.info(f"[WORKSPACE] Restored from {source}")
        else:
            self.logger.info(f"[WORKSPACE] Source {source} is empty – workspace left clean.")

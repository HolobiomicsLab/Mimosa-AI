"""Benchmark task context and private-copy grading for the gold verifier.

Grading itself is NOT implemented here: it goes through the shared
``sources.benchmark_evaluation.snapshot_grading.grade_directory`` — the same
``CapsuleEvaluator`` call the per-generation snapshot ablations use (VER
re-execution 900 s, SR eval program 300 s, CodeBERT CBS, infra exclusion).
This module only (1) resolves the benchmark task row, (2) makes a private
copy of the generation workspace, because the sandbox copies
``pred_results/`` back into the graded directory, and (3) turns any harness
fault into a censored ``excluded`` result so the verifier never crashes.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sources.benchmark_evaluation.snapshot_grading import grade_directory

_REQUIRED_ROW_KEYS = ("gold_program_name", "eval_script_name")
_logger = logging.getLogger(__name__)


@dataclass
class TaskContext:
    """Benchmark task context needed to grade one generation.

    Attributes:
        row: The benchmark CSV row (``instance_id``, ``gold_program_name``,
            ``output_fname``, ``eval_script_name`` ...).
        sab_loader: ``ScienceAgentBenchLoader`` for eval program / gold files.
    """

    row: dict
    sab_loader: Any


def resolve_task_context(config: Any) -> tuple[TaskContext | None, str | None]:
    """Read the benchmark task context that csv_mode puts on the config.

    ``CsvEvaluationMode`` sets ``config.gold_feedback_task_row`` (and
    ``config.gold_feedback_sab_loader``) per task when
    ``verifier_kind == "gold"``. Plain ``--task`` / ``--goal`` runs have none.

    Args:
        config: The run config (``Config`` or a stand-in namespace).

    Returns:
        ``(context, None)`` when grading is possible, else
        ``(None, reason)`` with a human-readable reason.
    """
    row = getattr(config, "gold_feedback_task_row", None)
    if not isinstance(row, dict) or not row:
        return None, (
            "no benchmark task row on the config (gold_feedback_task_row is "
            "unset): gold feedback needs a --science_agent_bench run"
        )
    missing = [k for k in _REQUIRED_ROW_KEYS if not str(row.get(k) or "").strip()]
    if missing:
        return None, f"benchmark task row lacks {', '.join(missing)}"
    loader = getattr(config, "gold_feedback_sab_loader", None)
    if loader is None:
        from sources.benchmark_evaluation.science_agent_bench import (
            ScienceAgentBenchLoader,
        )

        loader = ScienceAgentBenchLoader()
    return TaskContext(row=row, sab_loader=loader), None


def make_private_copy(workspace: str | Path) -> Path:
    """Copy *workspace* into a fresh temp dir (``mimosa_gold_*``).

    Args:
        workspace: The live generation workspace (never modified).

    Returns:
        Path of the copy (``<mkdtemp>/workspace``).

    Raises:
        OSError: If the workspace does not exist or cannot be copied.
    """
    src = Path(workspace)
    if not src.is_dir():
        raise FileNotFoundError(f"workspace not found: {src}")
    root = Path(tempfile.mkdtemp(prefix="mimosa_gold_"))
    dest = root / "workspace"
    try:
        shutil.copytree(src, dest, ignore_dangling_symlinks=True)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return dest


def discard_private_copy(copy_dir: str | Path | None) -> None:
    """Delete a copy made by :func:`make_private_copy` (best effort).

    Args:
        copy_dir: The path returned by :func:`make_private_copy`, or ``None``.
    """
    if copy_dir is None:
        return
    root = Path(copy_dir).parent
    if root.name.startswith("mimosa_gold_"):
        shutil.rmtree(root, ignore_errors=True)


class CopyHandoff:
    """Lock-protected ownership of a private copy between caller and grader worker.

    Exactly one side deletes the copy. The worker calls :meth:`worker_done`
    when it ends; the caller calls :meth:`caller_reclaims` once after its
    ``join(timeout)``. Whichever call comes second learns the other side's
    state under the same lock, so there is no window where both or neither
    delete it.

    Known gap: the worker is a daemon thread. If the interpreter exits while
    an abandoned worker still runs, the worker is killed before it can
    delete its ``mimosa_gold_*`` copy, which then stays in the system temp
    dir (``tempfile.gettempdir()``) until the OS or an operator removes it.
    """

    def __init__(self) -> None:
        """Start with neither side finished."""
        self._lock = threading.Lock()
        self._done = False
        self._abandoned = False

    def worker_done(self) -> bool:
        """Mark the worker finished.

        Returns:
            ``True`` when the caller already abandoned the worker — the
            worker must then delete the copy itself.
        """
        with self._lock:
            if self._abandoned:
                return True
            self._done = True
            return False

    def caller_reclaims(self) -> bool:
        """Caller's decision after ``join(timeout)``.

        Returns:
            ``True`` when the worker already finished — the caller keeps
            ownership (and the worker's result). ``False`` when the worker is
            still running — it is now abandoned and owns the copy.
        """
        with self._lock:
            if self._done:
                return True
            self._abandoned = True
            return False


def _grade_once(
    copy_dir: Path, context: TaskContext, grade_fn: Callable[..., dict[str, Any]]
) -> dict[str, Any]:
    """One grading call; a harness exception becomes a censored result."""
    try:
        return dict(
            grade_fn(
                Path(copy_dir), task_row=context.row, sab_loader=context.sab_loader
            )
        )
    except Exception as e:  # noqa: BLE001 — harness fault = censored, never a crash
        _logger.error(f"[GOLD FEEDBACK] grader harness error: {e}", exc_info=True)
        return _censored(f"Unexpected harness error: {e}")


def _censored(reason: str) -> dict[str, Any]:
    """Normalised ``excluded`` grade carrying *reason*."""
    return {
        "VER": None,
        "SR": None,
        "CBS": None,
        "cost": 0.0,
        "status": "excluded",
        "infra_error": reason,
    }


def grade_private_copy(
    copy_dir: str | Path,
    context: TaskContext,
    grade_fn: Callable[..., dict[str, Any]] = grade_directory,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """Grade a private workspace copy with the shared ablation grader.

    Never raises: a harness exception, or a grading run longer than
    *timeout_s*, becomes a censored ``excluded`` result. With a timeout the
    grading runs in a daemon worker thread. When the cap is hit the worker
    is abandoned (its sandbox subprocesses still stop at their own VER/SR
    timeouts) and it deletes *copy_dir* itself when it ends; the result then
    carries ``copy_owned_by_worker=True`` so the caller must not delete it.
    Ownership is decided under a lock (:class:`CopyHandoff`). A daemon
    worker killed at interpreter exit can leave its ``mimosa_gold_*`` copy
    in the system temp dir.

    Args:
        copy_dir: Private copy from :func:`make_private_copy`.
        context: Task row + loader.
        grade_fn: Grading function (default: the shared ``grade_directory``);
            injectable for offline tests.
        timeout_s: Wall-clock cap in seconds; ``None`` or ``<= 0`` = no cap.

    Returns:
        The normalised grade (see ``grade_directory``) plus ``wall_s``.
    """
    t0 = time.time()
    if not timeout_s or timeout_s <= 0:
        grade = _grade_once(Path(copy_dir), context, grade_fn)
        grade["wall_s"] = round(time.time() - t0, 3)
        return grade

    holder: dict[str, Any] = {}
    handoff = CopyHandoff()

    def _work() -> None:
        try:
            holder["grade"] = _grade_once(Path(copy_dir), context, grade_fn)
        finally:
            if handoff.worker_done():
                discard_private_copy(copy_dir)  # caller gave up: worker owns it

    worker = threading.Thread(target=_work, name="gold-feedback-grader", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if handoff.caller_reclaims():
        # Worker finished (possibly right after the join timed out): its
        # result is valid and the caller keeps ownership of the copy.
        grade = holder.get("grade") or _censored("grader worker returned no result")
    else:
        _logger.error(
            f"[GOLD FEEDBACK] grading exceeded the {timeout_s:.0f}s wall-clock cap; "
            "result censored (worker abandoned)"
        )
        grade = _censored(
            f"grading exceeded the gold_feedback_timeout_s wall-clock cap ({timeout_s:.0f}s)"
        )
        grade["copy_owned_by_worker"] = True
    grade["wall_s"] = round(time.time() - t0, 3)
    return grade

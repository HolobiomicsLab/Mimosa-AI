"""``GoldFeedbackEvaluator`` — ORACLE / BENCHMARK-LEAKING verifier control.

Composition, not a new verifier: the evaluator holds

* an inner ``HybridVerifierEvaluator`` that still produces the reward,
  QD score, early-stop signal and capsule selection (default mode), and
* the shared benchmark grader (``snapshot_grading.grade_directory`` through
  :mod:`.grader`), i.e. the same ``CapsuleEvaluator`` call the snapshot
  ablations use.

Per generation it replaces ONLY the steering text
(``abstracted_textual_gradient`` + ``textual_gradient.txt``) with the
grader's own feedback (VER / SR messages in the E33 format, see
:mod:`.format`). With ``config.gold_feedback_reward = True`` (full oracle)
the reward also comes from the grader (:func:`.reward.oracle_reward`).

Every artifact carries the leakage warning. Never report scores from a run
that used this mode.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sources.core.workflow_info import WorkflowInfo, is_oracle_generation
from sources.evaluators.base import BaseEvaluator, EvaluatorError

from . import format as fmt
from .grader import (
    TaskContext,
    discard_private_copy,
    grade_private_copy,
    make_private_copy,
    resolve_task_context,
)
from .leak import LEAK_WARNING, leak_banner
from .reward import oracle_reward

STATUS_GOLD = "gold"
STATUS_NO_TASK = "no_task_context"
STATUS_SHORT_CIRCUIT = "skipped_short_circuit"
STATUS_CENSORED = "censored_infra"
STATUS_COPY_FAILED = "censored_copy_failed"

_SHORT_CIRCUIT_REASONS = {"workflow_generation_or_execution_failed", "no_goal_recorded"}
_MSG_KEEP = 20000  # verbatim grader messages kept in state_result / sidecar
_REPORT_MARKER = "---- hybrid verifier report (unchanged) ----\n"
#: Full-oracle reward of a censored generation: "no oracle measurement".
#: 0.0 cannot reach learned_score_threshold (early stop) nor admit_threshold
#: (QD archive), and ``reward_fallback="oracle_censored"`` keeps it out of
#: the capsule argmax.
CENSORED_ORACLE_REWARD = 0.0
GOLD_REGISTRY_SUFFIX = "_gold"


def gold_hybrid_config(config: Any) -> Any:
    """Shallow config copy that gives the inner hybrid its OWN registry dir.

    The hybrid keeps its per-task registry (claims, scorers, the previous
    generations it computes win-rates against) under
    ``config.temp_dir or workflow_dir/_verifier_tmp``. A gold run sharing
    ``workflow_dir`` and goal with honest runs would otherwise add
    gold-steered generations to the honest registry. The copy points
    ``temp_dir`` at the same base with a ``_gold`` suffix; the caller's
    config object is not modified, so non-gold runs are unaffected.

    Args:
        config: The run config.

    Returns:
        A ``copy.copy`` of *config* with ``temp_dir`` set to the gold root.
    """
    base = (
        getattr(config, "temp_dir", None) or Path(config.workflow_dir) / "_verifier_tmp"
    )
    isolated = copy.copy(config)
    isolated.temp_dir = str(Path(str(base) + GOLD_REGISTRY_SUFFIX))
    return isolated


class GoldFeedbackEvaluator:
    """Hybrid reward + benchmark-grader gradient (optionally grader reward).

    Duck-types the verifier interface used by ``WorkflowEvaluator``:
    ``evaluate(uuid) -> dict`` and persistence under ``evaluation.verifier``.
    """

    # Reuse the shared persistence helper (state_result.json merge-write);
    # it only needs ``self.workflow_dir`` and ``self.logger``.
    _save_results = BaseEvaluator._save_results

    def __init__(
        self,
        config: Any,
        workspace_dir: str | Path | None = None,
        hybrid: Any | None = None,
        grade_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        """Build the inner hybrid verifier and wire the grader.

        Args:
            config: Run config. Reads ``workflow_dir``, ``workspace_dir``,
                ``gold_feedback_reward`` (bool, default False) and, at
                evaluate time, the runtime-only ``gold_feedback_task_row`` /
                ``gold_feedback_sab_loader`` set by csv_mode.
            workspace_dir: Agents' workspace override (as for the hybrid).
            hybrid: Inner verifier; ``None`` builds a
                ``HybridVerifierEvaluator`` (injectable for tests).
            grade_fn: Grading function; ``None`` uses the shared
                ``grade_directory`` (injectable for tests).
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        if hybrid is None:
            from sources.evaluators.hybrid_verifier import HybridVerifierEvaluator

            hybrid = HybridVerifierEvaluator(
                gold_hybrid_config(config), workspace_dir=workspace_dir
            )
        self.hybrid = hybrid
        self.workflow_dir = Path(config.workflow_dir)
        self.workspace_dir = Path(
            workspace_dir
            if workspace_dir is not None
            else getattr(hybrid, "workspace_dir", None)
            or getattr(config, "workspace_dir", ".")
        )
        self.grade_fn = grade_fn
        self.full_oracle = bool(getattr(config, "gold_feedback_reward", False))
        self.timeout_s = float(getattr(config, "gold_feedback_timeout_s", 1800) or 0)
        self.logger.warning(
            f"{leak_banner()}\nGoldFeedbackEvaluator initialised "
            f"(reward source: {self.reward_source}, workspace={self.workspace_dir}, "
            f"grading cap={self.timeout_s:.0f}s)"
        )
        self._warn_mixed_workflow_dir()

    @property
    def reward_source(self) -> str:
        """``benchmark_grader`` in full-oracle mode, else ``hybrid_verifier``."""
        return "benchmark_grader" if self.full_oracle else "hybrid_verifier"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(self, uuid: str) -> dict[str, Any]:
        """Run the hybrid verifier, grade a private workspace copy, swap the gradient.

        Args:
            uuid: Workflow identifier to evaluate.

        Returns:
            The hybrid result dict, updated with the gold gradient, the
            oracle flags and the ``gold_feedback`` summary (and the grader
            reward in full-oracle mode).

        Raises:
            EvaluatorError: When ``uuid`` is invalid or the hybrid raises it.
        """
        if not uuid or not isinstance(uuid, str):
            raise EvaluatorError("Invalid uuid: must be a non-empty string")
        self.logger.warning(f"[GOLD FEEDBACK] {uuid}: {LEAK_WARNING}")

        context, no_context_reason = resolve_task_context(self.config)
        copy_dir: Path | None = None
        copy_error: str | None = None
        has_generation = context is not None and self._has_generation(uuid)
        # Private copy BEFORE the hybrid runs: the grader re-executes the
        # script and writes pred_results/ into the directory it grades.
        if has_generation:
            try:
                copy_dir = make_private_copy(self.workspace_dir)
            except Exception as e:  # noqa: BLE001 — censored, never a crash
                copy_error = f"could not copy the workspace for grading: {e}"
        grade: dict[str, Any] | None = None
        try:
            hybrid_result = self.hybrid.evaluate(uuid)
            block = self._verifier_block(uuid, hybrid_result)
            hybrid_gradient = str(block.get("abstracted_textual_gradient") or "")

            if context is None:
                status, reason = STATUS_NO_TASK, no_context_reason
            elif (
                not has_generation
                or block.get("skipped_reason") in _SHORT_CIRCUIT_REASONS
            ):
                status, reason = (
                    STATUS_SHORT_CIRCUIT,
                    (
                        "hybrid short-circuit (no code / no state_result): nothing to grade"
                    ),
                )
            elif copy_dir is None:
                status, reason = STATUS_COPY_FAILED, copy_error
            else:
                grade = self._grade(copy_dir, context)
                if grade.get("status") == "excluded":
                    status, reason = (
                        STATUS_CENSORED,
                        str(grade.get("infra_error") or ""),
                    )
                else:
                    status, reason = STATUS_GOLD, None
        finally:
            # An abandoned (timed-out) grader worker owns the copy and
            # deletes it when its sandbox subprocesses end.
            if not (grade or {}).pop("copy_owned_by_worker", False):
                discard_private_copy(copy_dir)

        return self._apply(
            uuid, hybrid_result, block, hybrid_gradient, status, reason, grade, context
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _warn_mixed_workflow_dir(self) -> int:
        """Warn when ``workflow_dir`` already holds honest (non-oracle) generations.

        Returns:
            The number of honest generations found (0 = clean oracle dir).
        """
        honest = 0
        try:
            for folder in self.workflow_dir.iterdir():
                if (
                    folder.is_dir()
                    and (folder / "state_result.json").exists()
                    and not is_oracle_generation(folder)
                ):
                    honest += 1
        except OSError:
            return 0
        if honest:
            self.logger.warning(
                f"[GOLD FEEDBACK][LEAK GUARD] workflow_dir {self.workflow_dir} already "
                f"holds {honest} non-oracle generation(s). Oracle generations written "
                "here are skipped by honest runs' WorkflowSelector, but use a separate "
                "workflow_dir for gold runs."
            )
        return honest

    def _has_generation(self, uuid: str) -> bool:
        """True when the run left code and a state_result (else hybrid short-circuits)."""
        try:
            info = WorkflowInfo(uuid, self.workflow_dir / uuid)
            return bool(info.state_result) and bool(info.code)
        except Exception:  # noqa: BLE001 — unreadable run = nothing to grade
            return False

    def _grade(self, copy_dir: Path, context: TaskContext) -> dict[str, Any]:
        """Grade the private copy through the shared grader (never raises)."""
        if self.grade_fn is None:
            return grade_private_copy(copy_dir, context, timeout_s=self.timeout_s)
        return grade_private_copy(
            copy_dir, context, grade_fn=self.grade_fn, timeout_s=self.timeout_s
        )

    def _verifier_block(
        self, uuid: str, hybrid_result: dict[str, Any]
    ) -> dict[str, Any]:
        """The hybrid's persisted ``evaluation.verifier`` block (or its result)."""
        path = self.workflow_dir / uuid / "state_result.json"
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            block = (state.get("evaluation") or {}).get("verifier")
            if isinstance(block, dict):
                return dict(block)
        except (OSError, ValueError, AttributeError):
            pass
        return {
            k: v
            for k, v in (hybrid_result or {}).items()
            if k not in ("uuid", "claims")
        }

    def _apply(
        self,
        uuid: str,
        hybrid_result: dict[str, Any],
        block: dict[str, Any],
        hybrid_gradient: str,
        status: str,
        reason: str | None,
        grade: dict[str, Any] | None,
        context: TaskContext | None,
    ) -> dict[str, Any]:
        """Build the updated verifier block, persist it and all sidecars."""
        kind = None
        uninformative = None
        if status == STATUS_GOLD and grade is not None:
            kind = fmt.classify(grade)
            msg = (
                grade.get("VER_message")
                if kind == fmt.EXEC_FAILURE
                else grade.get("SR_message")
            )
            uninformative = fmt.is_uninformative(msg)
            gradient = fmt.build_gold_gradient(grade)
        elif status in (STATUS_CENSORED, STATUS_COPY_FAILED):
            gradient = f"{fmt.censored_note(reason or '')}\n\n{hybrid_gradient}".strip()
        else:
            gradient = hybrid_gradient

        summary: dict[str, Any] = {
            "status": status,
            "reason": reason,
            "kind": kind,
            "uninformative": uninformative,
            "reward_source": self.reward_source,
            "gradient_source": "benchmark_grader"
            if status == STATUS_GOLD
            else "hybrid_verifier",
            "instance_id": (context.row.get("instance_id") if context else None),
        }
        if grade is not None:
            summary.update(
                {
                    "VER": grade.get("VER"),
                    "SR": grade.get("SR"),
                    "CBS": grade.get("CBS"),
                    "grade_status": grade.get("status"),
                    "infra_error": grade.get("infra_error"),
                    "wall_s": grade.get("wall_s"),
                }
            )

        updates: dict[str, Any] = {
            "verifier_kind": "gold",
            "oracle": True,
            "benchmark_leak": True,
            "gold_feedback_warning": LEAK_WARNING,
            "hybrid_textual_gradient": hybrid_gradient,
            "abstracted_textual_gradient": gradient,
            "abstractec_textual_gradient": gradient,
            "gold_feedback": summary,
        }
        if self.full_oracle:
            if status == STATUS_GOLD and grade is not None:
                r = oracle_reward(grade)
                updates.update(
                    {
                        "hybrid_overall_score": block.get("overall_score"),
                        "hybrid_overall_score_uncapped": block.get(
                            "overall_score_uncapped"
                        ),
                        "hybrid_reward_fallback": block.get("reward_fallback"),
                        "overall_score": r,
                        "overall_score_uncapped": r,
                        "reward_fallback": None,
                        "reward_source": "benchmark_grader",
                    }
                )
            else:
                # No grade: the hybrid reward stands, and says so.
                updates["reward_source"] = "hybrid_verifier"
                summary["reward_fallback_reason"] = f"no grader verdict ({status})"
                if status in (STATUS_CENSORED, STATUS_COPY_FAILED):
                    # Censored gen: its hybrid win-rate is on a different
                    # scale from the oracle rewards of graded siblings. Do
                    # not let it reach the QD archive, the early stop or the
                    # capsule argmax: reward = CENSORED_ORACLE_REWARD (0.0,
                    # below admit_threshold and learned_score_threshold),
                    # flagged as a fallback; hybrid values kept in hybrid_*.
                    updates.update(
                        {
                            "hybrid_overall_score": block.get("overall_score"),
                            "hybrid_overall_score_uncapped": block.get(
                                "overall_score_uncapped"
                            ),
                            "hybrid_reward_fallback": block.get("reward_fallback"),
                            "overall_score": CENSORED_ORACLE_REWARD,
                            "overall_score_uncapped": CENSORED_ORACLE_REWARD,
                            "reward_fallback": "oracle_censored",
                            "reward_censored": True,
                            "reward_source": "oracle_censored",
                        }
                    )
        else:
            updates["reward_source"] = "hybrid_verifier"

        block.update(updates)
        self._persist(uuid, block, gradient, hybrid_gradient, summary, grade)

        level = logging.INFO if status == STATUS_GOLD else logging.WARNING
        self.logger.log(
            level,
            f"[GOLD FEEDBACK] {uuid}: status={status} kind={kind} "
            f"reward_source={updates['reward_source']} "
            f"reward={block.get('overall_score')} reason={reason}",
        )
        result = dict(hybrid_result or {})
        result.update(updates)
        return result

    def _persist(
        self,
        uuid: str,
        block: dict[str, Any],
        gradient: str,
        hybrid_gradient: str,
        summary: dict[str, Any],
        grade: dict[str, Any] | None,
    ) -> None:
        """Write state_result blocks, gradient sidecars and the evaluation.txt banner."""
        try:
            self._save_results(block, uuid, "verifier")
            oracle_block = {"warning": LEAK_WARNING, **summary}
            if grade is not None:
                oracle_block["VER_message"] = str(grade.get("VER_message") or "")[
                    :_MSG_KEEP
                ]
                oracle_block["SR_message"] = str(grade.get("SR_message") or "")[
                    :_MSG_KEEP
                ]
            self._save_results(oracle_block, uuid, "gold_oracle")
        except Exception as e:  # noqa: BLE001 — persistence failure logged only
            self.logger.error(
                f"[GOLD FEEDBACK] could not persist scores for {uuid}: {e}"
            )
        out_dir = self.workflow_dir / uuid
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "textual_gradient.txt").write_text(gradient, encoding="utf-8")
            (out_dir / "textual_gradient_hybrid.txt").write_text(
                hybrid_gradient, encoding="utf-8"
            )
            record = {"warning": LEAK_WARNING, **summary}
            if grade is not None:
                record["grade"] = {k: v for k, v in grade.items()}
            (out_dir / "gold_feedback.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            eval_path = out_dir / "evaluation.txt"
            previous = (
                eval_path.read_text(encoding="utf-8") if eval_path.exists() else ""
            )
            if _REPORT_MARKER in previous:
                # Re-evaluation without a fresh hybrid report: replace the
                # old banner instead of stacking a second one.
                previous = previous.split(_REPORT_MARKER, 1)[1]
            header = (
                f"{leak_banner()}\n"
                f"Gold feedback: status={summary['status']} kind={summary['kind']} "
                f"gradient_source={summary['gradient_source']} "
                f"reward_source={block.get('reward_source')} "
                f"reward={block.get('overall_score')}\n"
                + (f"Reason: {summary['reason']}\n" if summary.get("reason") else "")
                + (
                    f"Grader: VER={summary.get('VER')} SR={summary.get('SR')} "
                    f"CBS={summary.get('CBS')}\n"
                    if grade is not None
                    else ""
                )
                + "\n"
                + _REPORT_MARKER
            )
            eval_path.write_text(header + previous, encoding="utf-8")
        except OSError as e:
            self.logger.error(
                f"[GOLD FEEDBACK] could not write artifacts for {uuid}: {e}"
            )

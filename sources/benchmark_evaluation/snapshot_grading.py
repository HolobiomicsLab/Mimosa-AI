"""Shared benchmark grading of one workspace directory.

One helper, two callers: the per-generation snapshot ablations in
``CsvEvaluationMode._evaluate_snapshot_ablations`` and the gold-feedback
oracle verifier (``sources/evaluators/gold_feedback``). Both grade a
directory with the SAME ``CapsuleEvaluator`` (VER re-execution, SR eval
program, CodeBERT CBS) and need the same normalised result, so the
construction of the grading call and the result normalisation live here
instead of being inlined twice.

The helper does not copy the directory. ``ExecutionSandbox.run_generated_code``
copies ``pred_results/`` back into the graded directory, so a caller that
must keep its directory pristine has to pass a private copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sources.benchmark_evaluation.capsule_evaluator import CapsuleEvaluator


def grade_directory(
    directory: str | Path,
    task_row: dict,
    sab_loader: Any,
    api_cost: float = 0.0,
    evaluator_cls: type | None = None,
) -> dict[str, Any]:
    """Grade one directory with ``CapsuleEvaluator`` and normalise the result.

    Does not call ``save_results()``: nothing is written into *directory*
    by this helper (the sandbox itself may still refresh ``pred_results/``,
    see the module docstring). Exceptions from the evaluator propagate, so
    each caller keeps its own harness-error policy.

    Args:
        directory: Directory holding the generated script(s) and outputs.
        task_row: Benchmark CSV row (``instance_id``, ``gold_program_name``,
            ``output_fname``, ``eval_script_name``).
        sab_loader: ``ScienceAgentBenchLoader`` giving the eval program,
            the visual judge and the gold program.
        api_cost: Cost recorded for this directory (forwarded as
            ``CapsuleEvaluator(api_cost=...)``).
        evaluator_cls: Evaluator class to build; ``None`` means
            ``CapsuleEvaluator``. Callers pass their own module-level name so
            existing test patch points keep working.

    Returns:
        For a graded directory, keys in this order: ``VER``, ``VER_message``,
        ``SR``, ``SR_message``, ``CBS``, ``cost`` (= *api_cost*),
        ``status="evaluated"``. For an infra exclusion: ``VER``/``SR``/``CBS``
        = ``None``, ``cost`` (the evaluator's cost, else *api_cost*),
        ``status="excluded"`` and ``infra_error``.
    """
    cls = evaluator_cls if evaluator_cls is not None else CapsuleEvaluator
    evaluator = cls(
        capsule_path=Path(directory),
        task_data=task_row,
        sab_loader=sab_loader,
        api_cost=api_cost,
    )
    eval_results = evaluator.evaluate_all()
    if eval_results.get("status") == "excluded":
        return {
            "VER": None,
            "SR": None,
            "CBS": None,
            "cost": eval_results.get("cost", api_cost),
            "status": "excluded",
            "infra_error": eval_results.get("infra_error"),
        }
    return {
        "VER": eval_results["VER"][0],
        "VER_message": eval_results["VER"][1],
        "SR": eval_results["SR"][0],
        "SR_message": eval_results["SR"][1],
        "CBS": eval_results["CBS"],
        "cost": api_cost,
        "status": "evaluated",
    }

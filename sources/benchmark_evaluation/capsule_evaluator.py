"""
CapsuleEvaluator - Evaluates Mimosa-AI results against ScienceAgentBench metrics.

Implements the four key metrics:
1. VER (Valid Execution Rate) - Binary: code executes without errors
2. SR (Success Rate) - Binary: output meets task-specific criteria
3. CBS (CodeBERTScore) - Float: similarity to gold program (0-1)
4. Cost - Float: API cost in USD
"""

import os
import sys
import logging
from pathlib import Path
import json
from datetime import datetime

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sources.benchmark_evaluation.science_agent_bench import ScienceAgentBenchLoader
from sources.benchmark_evaluation.execution_sandbox import ExecutionSandbox, EvalInfraError
from sources.benchmark_evaluation.codebert_scorer import calculate_codebert_score

class CapsuleEvaluator:
    """Evaluates Mimosa-AI execution results against ScienceAgentBench metrics."""

    def __init__(
        self,
        capsule_path: Path,
        task_data: dict[str, str],
        sab_loader: ScienceAgentBenchLoader,
        api_cost: float = 0.000,
        cpu_only: bool = True,
        base_packages: list[str] | None = None,
    ):
        """
        Initialize CapsuleEvaluator.

        The sandbox is built lazily on first evaluation (not here), so a build
        failure is reported as an infra exclusion rather than crashing setup.

        Args:
            capsule_path: Path to the runs_capsule directory with generated files
            task_data: Dictionary containing task information from CSV row
            sab_loader: ScienceAgentBenchLoader instance for accessing eval scripts
            api_cost: API cost tracked from IndividualRun
            cpu_only: If True (default), the sandbox forces CPU execution by
                hiding any GPU from generated scripts. Avoids CUDA/XLA failures
                (e.g. missing libdevice.10.bc) that would otherwise fail VER.
            base_packages: Optional lighter base-package set for the sandbox venv.
        """
        self.capsule_path = Path(capsule_path)
        self.task_data = task_data
        self.sab_loader = sab_loader
        self.api_cost = api_cost
        self.cpu_only = cpu_only
        self.base_packages = base_packages
        self.logger = logging.getLogger(__name__)
        self.sandbox: ExecutionSandbox | None = None  # built lazily in evaluate_all

        # Extract key task information
        self.instance_id = task_data.get('instance_id', 'unknown')
        self.expected_output = task_data.get('output_fname', '')
        self.eval_script_name = task_data.get('eval_script_name', '')
        self.gold_program_name = task_data.get('gold_program_name', '')

        # Results storage
        self.metrics: dict[str, any] = {}
        # Reason CBS could not be computed (kept distinct from a genuine 0.0).
        self._cbs_error: str | None = None

    def _ensure_sandbox(self) -> None:
        """Build the sandbox on demand; a build failure becomes an infra exclusion."""
        if self.sandbox is not None:
            return
        try:
            self.sandbox = ExecutionSandbox(
                self.capsule_path, cpu_only=self.cpu_only, base_packages=self.base_packages
            )
        except Exception as e:
            raise EvalInfraError(f"Eval sandbox build failed: {e}") from e

    def evaluate_all(self) -> dict[str, any]:
        """
        Run all evaluation metrics.

        Returns:
            Dictionary with all metrics:
            {
                'VER': (bool|None, str),  # (success, message); None if excluded
                'SR': (bool|None, str),   # (success, message); None if excluded
                'CBS': float|None,        # 0.0-1.0; None if excluded
                'cost': float,            # USD
                'status': str,            # 'evaluated' or 'excluded'
                'infra_error': str,       # present only when excluded
                'summary': str
            }

        Infra failures (sandbox build, missing gold_results, missing LLM key)
        raise EvalInfraError internally and yield status='excluded' with
        VER/SR/CBS = None, so the task is dropped from metrics rather than
        counted as a failure.
        """
        self.logger.info(f"[EVAL] Starting evaluation for task {self.instance_id}")
        self.metrics = {}
        try:
            self._ensure_sandbox()  # build failure -> EvalInfraError

            # 1. VER + Success Rate
            sr_success, sr_msg, ver_success, ver_msg = self.evaluate_success_rate()
            self.metrics['VER'] = (ver_success, ver_msg)
            if ver_success:
                self.metrics['SR'] = (sr_success, sr_msg)
            else:
                self.metrics['SR'] = (False, "VER failed - SR is therefore False")

            # 2. CodeBERTScore (official convention: SR=1 -> CBS=1.0)
            if self.metrics['SR'][0]:
                self.metrics['CBS'] = 1.0
                self.logger.info("[EVAL] SR=1, setting CBS=1.0 automatically")
            else:
                self.metrics['CBS'] = self.calculate_codebert_score()
            self.metrics['cost'] = self.api_cost
            self.metrics['status'] = 'evaluated'
            self.logger.info(
                f"[EVAL] Results: VER={ver_success}, SR={self.metrics['SR'][0]}, "
                f"CBS={self.metrics['CBS']:.3f}, Cost=${self.api_cost:.4f}"
            )

        except EvalInfraError as e:
            self.logger.error(
                f"[EVAL] Infra failure — EXCLUDING task {self.instance_id} from metrics: {e}"
            )
            self.metrics = {
                'VER': (None, str(e)),
                'SR': (None, str(e)),
                'CBS': None,
                'cost': self.api_cost,
                'status': 'excluded',
                'infra_error': str(e),
            }
        finally:
            if self.sandbox is not None:
                self.sandbox.cleanup()  # free per-task disk; shared venv persists

        self.metrics['summary'] = self._generate_summary()
        self.logger.info(f"[EVAL] Evaluation complete for task {self.instance_id}")
        return self.metrics

    def evaluate_success_rate(self) -> tuple[bool, str, bool, str]:
        """
        Evaluate Success Rate (SR) and Valid Execution Rate (VER).

        First runs the generated code to check VER (produces expected output),
        then runs the task-specific evaluation script for SR.

        Returns:
            Tuple of (SR_success, SR_message, VER_success, VER_message)
        """
        try:

            # Step 1: VER - Run generated code
            self.logger.info("[EVAL] Running generated code for VER evaluation")
            full_program_path = self.capsule_path / self.gold_program_name
            ver_success, ver_message = self.sandbox.run_generated_code(
                script_path=full_program_path,  # Use generated program (same name as gold program) for execution to check if it runs without error
                script_name=self.gold_program_name,
                expected_output=self.expected_output,
                timeout=300
            )

            if not ver_success:
                self.logger.error(f"[EVAL] VER failed: {ver_message}")
                return False, "VER Failed, therefore SR is false", False, ver_message

            self.logger.info("[EVAL] VER passed - code executed successfully")

            if not self.eval_script_name:
                raise EvalInfraError("No evaluation script specified for this task")
            # Locate the eval script + visual judge; missing files are infra, not SR=0.
            try:
                eval_script_path, judge_path = self.sab_loader.get_eval_script_path(self.task_data)
            except (FileNotFoundError, ValueError) as e:
                raise EvalInfraError(f"Eval script/judge unavailable: {e}") from e
            self.logger.info(f"[EVAL] Running evaluation script: {eval_script_path.name}")

            sr_success, sr_message = self.sandbox.run_eval_script(
                eval_script_path=eval_script_path,
                visual_judge_path=judge_path,
                timeout=180
            )

            return sr_success, sr_message, ver_success, ver_message

        except EvalInfraError:
            raise  # infra problem — evaluate_all will exclude the task
        except Exception as e:
            self.logger.error(f"[EVAL] Error in evaluation: {str(e)}")
            msg = f"Evaluation error: {str(e)}"
            return False, msg, False, msg

    def calculate_codebert_score(self) -> float:
        """
        Calculate CodeBERTScore (CBS).

        Scores the SAME generated file that VER executed (best match to the gold
        program name), not an arbitrary glob pick, then compares it to the gold
        program with CodeBERT embeddings (F1 of matched token embeddings).

        Note: If SR=1, this is skipped and CBS is set to 1.0 (handled in
        evaluate_all), following the official ScienceAgentBench convention.

        On any computation failure the reason is recorded in ``self._cbs_error``
        and 0.0 is returned as a fallback — a failure is logged distinctly so it
        is never mistaken for a genuine zero similarity.

        Returns:
            CodeBERT F1 score (0.0-1.0)
        """
        self._cbs_error = None

        if not self.gold_program_name:
            self._cbs_error = "No gold program name for task"
            self.logger.warning("[EVAL] No gold program specified, CBS=0.0 (fallback)")
            return 0.0

        generated_code_path = self.sandbox.select_generated_script(self.gold_program_name)
        if generated_code_path is None:
            self._cbs_error = "No generated Python file in capsule"
            self.logger.warning("[EVAL] No Python file in capsule, CBS=0.0 (fallback)")
            return 0.0

        try:
            gold_program_path = self.sab_loader.get_gold_program_path(self.task_data)
        except (FileNotFoundError, ValueError) as e:
            self._cbs_error = f"Gold program unavailable: {e}"
            self.logger.warning(f"[EVAL] {self._cbs_error}, CBS=0.0 (fallback)")
            return 0.0

        try:
            self.logger.info(f"[EVAL] Computing CodeBERT score on {generated_code_path.name}")
            score = calculate_codebert_score(
                generated_code_path=generated_code_path,
                gold_code_path=gold_program_path
            )
            self.logger.info(f"[EVAL] CodeBERT score: {score:.3f}")
            return score
        except Exception as e:
            self._cbs_error = f"CBS computation failed: {e}"
            self.logger.error(
                f"[EVAL] {self._cbs_error} — recording 0.0 fallback (NOT a true zero)"
            )
            return 0.0

    def _generate_summary(self) -> str:
        """Generate a human-readable summary; handles excluded (None) metrics."""
        def mark(value) -> str:
            return "—" if value is None else ("✓" if value else "✗")

        ver = self.metrics.get('VER', (None, ''))[0]
        sr = self.metrics.get('SR', (None, ''))[0]
        cbs = self.metrics.get('CBS')
        cbs_str = "—" if cbs is None else f"{cbs:.3f}"
        cost = self.metrics.get('cost', 0.0)

        if self.metrics.get('status') == 'excluded':
            header = (
                f"Task {self.instance_id} EXCLUDED (infra failure): "
                f"{self.metrics.get('infra_error', '')}"
            )
        else:
            header = f"Task {self.instance_id} Evaluation Results:"

        summary = f"""
{header}
  VER (Valid Execution): {mark(ver)}
  SR (Success Rate): {mark(sr)}
  CBS (CodeBERT Score): {cbs_str}
  API Cost: ${cost:.4f}
"""
        return summary.strip()

    def save_results(self, output_path = None) -> Path:
        """
        Save evaluation results to JSON file.

        Args:
            output_path: Optional path for output file

        Returns:
            Path to saved results file
        """

        if output_path is None:
            output_path = self.capsule_path / "evaluation_results.json"

        results = {
            "task_id": self.instance_id,
            "timestamp": datetime.now().isoformat(),
            "status": self.metrics.get('status', 'evaluated'),
            "VER": self.metrics['VER'][0],
            "VER_message": self.metrics['VER'][1],
            "SR": self.metrics['SR'][0],
            "SR_message": self.metrics['SR'][1],
            "CBS": self.metrics['CBS'],
            "cost_usd": self.metrics['cost'],
            "summary": self.metrics['summary']
        }
        # Surface infra exclusion + CBS fallback reason so 0/None aren't misread.
        if self.metrics.get('infra_error'):
            results["infra_error"] = self.metrics['infra_error']
        if self._cbs_error:
            results["CBS_error"] = self._cbs_error

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"[EVAL] Results saved to {output_path}")
        return output_path

if __name__ == "__main__":
    import csv
    from sources.benchmark_evaluation.science_agent_bench import ScienceAgentBenchLoader

    sab_loader = ScienceAgentBenchLoader(base_path="../../datasets/ScienceAgentBench")
    papers_csv_path = "../../datasets/ScienceAgentBench.csv"
    with open(papers_csv_path, encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        total_rows = sum(1 for _ in reader)
        csvfile.seek(0)
        reader = csv.DictReader(csvfile)
        for _, row in enumerate(reader):
            evaluator = CapsuleEvaluator(
                capsule_path=Path("../../runs_capsule") / "clintox",
                task_data=row,
                sab_loader=sab_loader,
                api_cost=0.0
            )
            eval_results = evaluator.evaluate_all()
            print(evaluator._generate_summary())
            break

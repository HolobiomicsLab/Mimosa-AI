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

from sources.evaluation.science_agent_bench import ScienceAgentBenchLoader
from sources.evaluation.execution_sandbox import ExecutionSandbox
from sources.evaluation.codebert_scorer import calculate_codebert_score

class CapsuleEvaluator:
    """Evaluates Mimosa-AI execution results against ScienceAgentBench metrics."""
    
    def __init__(
        self,
        capsule_path: Path,
        task_data: dict[str, str],
        sab_loader: ScienceAgentBenchLoader,
        api_cost: float = 0.0
    ):
        """
        Initialize CapsuleEvaluator.
        
        Args:
            capsule_path: Path to the runs_capsule directory with generated files
            task_data: Dictionary containing task information from CSV row
            sab_loader: ScienceAgentBenchLoader instance for accessing eval scripts
            api_cost: API cost tracked from GodelRun
        """
        self.capsule_path = Path(capsule_path)
        self.task_data = task_data
        self.sab_loader = sab_loader
        self.api_cost = api_cost
        self.logger = logging.getLogger(__name__)
        self.sandbox = ExecutionSandbox(self.capsule_path)
        
        # Extract key task information
        self.instance_id = task_data.get('instance_id', 'unknown')
        self.expected_output = task_data.get('output_fname', '')
        self.eval_script_name = task_data.get('eval_script_name', '')
        self.gold_program_name = task_data.get('gold_program_name', '')
        
        # Results storage
        self.metrics: dict[str, any] = {}
        
    def evaluate_all(self) -> dict[str, any]:
        """
        Run all evaluation metrics.
        
        Returns:
            Dictionary with all metrics:
            {
                'VER': (bool, str),  # (success, message)
                'SR': (bool, str),   # (success, message)
                'CBS': float,        # 0.0-1.0
                'cost': float,       # USD
                'summary': str
            }
        """
        self.logger.info(f"[EVAL] Starting evaluation for task {self.instance_id}")
        # 1. VER + Evaluate Success Rate 
        sr_success, sr_msg, ver_success = self.evaluate_success_rate()
        self.metrics['VER'] = (ver_success, sr_msg)
        if ver_success:
            self.metrics['SR'] = (sr_success, sr_msg)
        else:
            self.metrics['SR'] = (False, "VER failed - SR is therefore False")
        # 2. Calculate CodeBERTScore
        if self.metrics['SR'][0]:
            self.metrics['CBS'] = 1.0
            self.logger.info("[EVAL] SR=1, setting CBS=1.0 automatically")
        else:
            self.metrics['CBS'] = self.calculate_codebert_score()
        self.metrics['cost'] = self.api_cost
        self.metrics['summary'] = self._generate_summary()
        self.logger.info(f"[EVAL] Evaluation complete for task {self.instance_id}")
        self.logger.info(f"[EVAL] Results: VER={ver_success}, SR={self.metrics['SR'][0]}, CBS={self.metrics['CBS']:.3f}, Cost=${self.api_cost:.4f}")
        return self.metrics
    
    def evaluate_success_rate(self) -> tuple[bool, str, bool]:
        """
        Evaluate Success Rate (SR).
        
        Runs the task-specific evaluation script which checks if the
        output meets the task's success criteria (e.g., accuracy threshold).
        
        Returns:
            (success: bool, message: str)
        """
        try:
            if not self.eval_script_name:
                return False, "No evaluation script specified for this task", False
            
            eval_script_path = self.sab_loader.get_eval_script_path(self.task_data)
            
            if not eval_script_path.exists():
                return False, f"Evaluation script not found: {eval_script_path}", False
            
            self.logger.info(f"[EVAL] Running evaluation script: {eval_script_path.name}")
            
            success, message = self.sandbox.run_eval_script(
                eval_script_path=eval_script_path,
                timeout=180
            )
            
            return success, message, True
            
        except Exception as e:
            self.logger.error(f"[EVAL] Error in SR evaluation: {str(e)}")
            return False, f"Evaluation error: {str(e)}", False
    
    def calculate_codebert_score(self) -> float:
        """
        Calculate CodeBERTScore (CBS).
        
        Compares generated code with gold program using CodeBERT embeddings.
        Returns F1 score of matched token embeddings.
        
        Note: If SR=1, this should return 1.0 automatically (handled in evaluate_all).
        
        Returns:
            CodeBERT F1 score (0.0-1.0)
        """
        try:
            if not self.gold_program_name:
                self.logger.warning("[EVAL] No gold program specified, CBS=0.0")
                return 0.0
            
            # Find generated Python file
            py_files = list(self.capsule_path.glob("*.py"))
            if not py_files:
                self.logger.warning("[EVAL] No Python file in capsule, CBS=0.0")
                return 0.0
            generated_code_path = py_files[0]
            
            # Get gold program path
            gold_program_path = self.sab_loader.get_gold_program_path(self.task_data)
            if not gold_program_path.exists():
                self.logger.warning(f"[EVAL] Gold program not found: {gold_program_path}, CBS=0.0")
                return 0.0

            self.logger.info("[EVAL] Computing CodeBERT score")
            # Calculate CodeBERT score
            score = calculate_codebert_score(
                generated_code_path=generated_code_path,
                gold_code_path=gold_program_path
            )
            
            self.logger.info(f"[EVAL] CodeBERT score: {score:.3f}")
            return score
            
        except Exception as e:
            self.logger.error(f"[EVAL] Error calculating CodeBERT score: {str(e)}")
            return 0.0
    
    def _generate_summary(self) -> str:
        """Generate a human-readable summary of evaluation results."""
        ver_status = "✓" if self.metrics['VER'][0] else "✗"
        sr_status = "✓" if self.metrics['SR'][0] else "✗"
        cbs_value = self.metrics['CBS']
        
        summary = f"""
Task {self.instance_id} Evaluation Results:
  VER (Valid Execution): {ver_status} - {self.metrics['VER'][1]}
  SR (Success Rate): {sr_status} - {self.metrics['SR'][1]}
  CBS (CodeBERT Score): {cbs_value:.3f}
  API Cost: ${self.metrics['cost']:.4f}
"""
        return summary.strip()
    
    def save_results(self, output_path: Path | None) -> Path:
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
            "VER": self.metrics['VER'][0],
            "VER_message": self.metrics['VER'][1],
            "SR": self.metrics['SR'][0],
            "SR_message": self.metrics['SR'][1],
            "CBS": self.metrics['CBS'],
            "cost_usd": self.metrics['cost'],
            "summary": self.metrics['summary']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"[EVAL] Results saved to {output_path}")
        return output_path

if __name__ == "__main__":
    import csv
    from sources.evaluation.science_agent_bench import ScienceAgentBenchLoader

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

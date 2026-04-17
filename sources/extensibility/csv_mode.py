"""
CsvEvaluationMode - Autonomous goal generation and execution system
"""

import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from sources.core.dgm import DarwinMachine
from sources.core.llm_provider import LLMConfig, LLMProvider
from sources.core.planner import Planner
from sources.core.schema import Task, IndividualRun
from sources.evaluation.science_agent_bench import ScienceAgentBenchLoader
from sources.evaluation.capsule_evaluator import CapsuleEvaluator
from sources.utils.transfer_toolomics import LocalTransfer

class CsvEvaluationMode:
    """
    Autonomous mode that automatically run Mimosa on various goal's defined in a CSV datasets, such a list of paper to replicate.
    """

    def __init__(self, config, csv_runs_limit: int = 103):
        """
        Initialize CsvEvaluationMode.

        Args:
            config: Mimosa configuration object
            csv_runs_limit: Maximum number of autonomous iterations
        """
        self.config = config
        self.csv_runs_limit = csv_runs_limit
        self.dgm = DarwinMachine(config)
        self.planner = Planner(config)
        self.run_notes_dir = Path("run_notes")
        self.run_notes_dir.mkdir(exist_ok=True)
        self.done_rows = []

        model_name = "anthropic/claude-haiku-4-5-20251001"  # judge
        provider, model = model_name.split("/", 1) if "/" in model_name else ("openai", model_name)

        self.llm_config = LLMConfig(
            model=model,
            provider=provider,
            temperature=0.8,
            max_tokens=8192
        )
        self.result_analyzer = LLMProvider(
            agent_name="result_analyzer",
            memory_path=None,
            system_msg=self._get_result_analyzer_system_prompt(),
            config=self.llm_config
        )
        # Track execution history
        self.execution_history: list[dict] = []
        self.logger = logging.getLogger(__name__)

    def _get_result_analyzer_system_prompt(self) -> str:
        """System prompt for the result analysis LLM."""
        return """You are an autonomous AI scientist result analyzer for Mimosa-AI.

Mimosa-AI is a multi-agent system designed to autonomously conduct scientific goals.

Your role is to analyze workflow execution results and provide insights for the next goal generation.
You must be strict and harsh in your analysis.

ANALYSIS FOCUS:
1. Assess goal completion quality and success level
2. Identify strengths and weaknesses in the execution
3. Note any errors, limitations, or areas for improvement

EVALUATION CRITERIA:
- Task completion: Was the full goal achieved? An incomplete goal should be considered as failed.
- Quality: How well was the goal executed?
- Scalability: Could this approach work for similar goals?

INPUT FORMAT:
You will receive a list of agent name and their corresponding answers from the workflow execution.
The answers will be in the format:
agent 1: <answer from agent 1>
agent 2: <answer from agent 2>
agent 3: <answer from agent 3>
...

OUTPUT FORMAT:
Provide a structured analysis with:
1. SUCCESS_LEVEL: (High/Medium/Low/Incomplete/Failed/Error)
2. COMMENTS: Comments on what the multi-agents workflow tried to do, what worked, what failed and why.
"""

    def _load_previous_run_notes(self) -> dict | None:
        """
        Load the run notes file with the highest total_eval count.
        This allows recovery of previous execution statistics.

        Returns:
            Dictionary with previous run data, or None if no valid notes found
        """
        if not self.run_notes_dir.exists():
            return None

        best_notes = None
        max_total_eval = 0

        for notes_file in self.run_notes_dir.glob("*.json"):
            try:
                with open(notes_file, 'r', encoding='utf-8') as f:
                    notes = json.load(f)
                    model = notes.get('model', '')
                    assert model
                    assert self.config.smolagent_model_id
                    if model != self.config.smolagent_model_id:
                        continue

                    total_eval = notes.get('total_eval', 0)
                    if total_eval > max_total_eval:
                        max_total_eval = total_eval
                        best_notes = notes
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"[CACHE RECOVERY] Could not load {notes_file}: {e}")
                continue

        if best_notes:
            self.logger.info(
                f"[CACHE RECOVERY] Loaded previous run with {max_total_eval} evaluations"
            )
        return best_notes

    def _restore_execution_history_from_cache(self, cached_notes: dict) -> None:
        """
        Restore execution history statistics from cached run notes.
        This reconstructs aggregate metrics for continued evaluation runs.

        Args:
            cached_notes: Dictionary containing previous run statistics
        """
        if not cached_notes or 'total_eval' not in cached_notes:
            return

        # Create synthetic execution_history entries to represent cached runs
        # We create one entry per evaluation from cache to maintain count accuracy
        total_eval = cached_notes.get('total_eval', 0)
        ver_success = cached_notes.get('ver_success', 0)
        sr_success = cached_notes.get('sr_success', 0)
        avg_cbs = cached_notes.get('avg_cbs', 0.0)
        total_cost = cached_notes.get('total_cost', 0.0)

        if total_eval > 0:
            # Calculate per-run averages
            avg_cost_per_run = total_cost / total_eval

            # Create synthetic entries representing cached runs
            # We distribute VER/SR successes across the entries
            for i in range(total_eval):
                synthetic_entry = {
                    "iteration": -(i + 1),  # Negative to distinguish from new runs
                    "goal": "[Cached from previous run]",
                    "execution_time": 0,
                    "success_level": "Cached",
                    "key_insight": "Restored from cache",
                    "VER": i < ver_success,  # Distribute successes
                    "SR": i < sr_success,
                    "CBS": avg_cbs,  # Use average for all
                    "eval_cost": avg_cost_per_run
                }
                self.execution_history.append(synthetic_entry)

            self.logger.info(
                f"[CACHE RECOVERY] Restored {total_eval} evaluations: "
                f"VER={ver_success}, SR={sr_success}, CBS={avg_cbs:.3f}"
            )
            print(f"\033[95mRestored {total_eval} previous evaluations from cache\033[0m")

    def _save_run_notes(self, capsule_name: str, goal: str,
                       analysis: dict, execution_time: float) -> None:
        """Save detailed notes about the run."""
        timestamp = datetime.now().isoformat()
        sab_runs = [exec_data for exec_data in self.execution_history if 'VER' in exec_data]
        notes = {
            "timestamp": timestamp,
            "model": self.config.smolagent_model_id,
            "goal": goal,
            "execution_time_seconds": execution_time,
            "analysis": analysis["full_analysis"],
            "total_eval": len(sab_runs)
        }
        if sab_runs:
            notes = {
                **notes,
                "ver_success": sum(1 for run in sab_runs if run.get('VER', False)),
                "sr_success": sum(1 for run in sab_runs if run.get('SR', False)),
                "avg_cbs": sum(run.get('CBS', 0.0) for run in sab_runs) / len(sab_runs),
                "total_cost": sum(run.get('eval_cost', 0.0) for run in sab_runs),
                "is_success": sab_runs[-1].get('SR', False)
            }

        notes_file = self.run_notes_dir / f"{capsule_name}.json"
        notes_file.parent.mkdir(parents=True, exist_ok=True)
        with open(notes_file, 'w', encoding='utf-8') as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)
        self.logger.info(f"[PAPERS DATASET MODE] Run notes saved to {notes_file}")

    def _generate_task_default(self, row):
        paper_title = row.get('Title', '').strip() if row.get('Title') else ''
        url = row.get('URLS', '').strip() if row.get('URLS') else ''
        prompt = row.get('Prompt', '').strip() if row.get('Prompt') else ''
        if prompt == "":
            prompt = "Reproduce the experiments from the paper and compare the result."
        return f"""
    Paper title: {paper_title}
    Url to paper: {url}
    Goal to achieve: {prompt}
        """.strip()

    def _generate_task_science_agent_bench(self, row):
        task_inst = row.get('task_inst', '').strip()
        domain_knowledge = row.get('domain_knowledge', '').strip()
        dataset_folder_tree = row.get('dataset_folder_tree', '').strip()
        dataset_preview = row.get('dataset_preview', '').strip()
        output_fname = row.get('output_fname', '').strip()

        task_prompt = f"""
    DOMAIN KNOWLEDGE:
    {domain_knowledge}
    INSTRUCTIONS:
    {task_inst}
    DATASET STRUCTURE:
    {dataset_folder_tree}
    DATASET PREVIEW:
    {dataset_preview}
    EXPECTED OUTPUT:
    Save results to a formatted file named exactly: {output_fname}
    Keep only one python script in the workspace (The best one that lead to success).
    """
        return task_prompt

    def _generate_next_task(self, row, dataset_type: str) -> str:
        """Generate the next goal using LLM based on paper from the CSV."""
        if dataset_type == "science_agent_bench":
            return self._generate_task_science_agent_bench(row)
        return self._generate_task_default(row)

    def _format_goal_mode_results(self, tasks_data: Task) -> str:
        """Format task results for analysis."""
        return '\n\n'.join(
            f"Task {task.name}:\n"
            f"  UUID: {task.evolve_runs[-1].current_uuid}\n"
            f"  Description: {task.description}\n"
            f"  Agent Chain: {' -> '.join(task.final_answers)}"
            for task in tasks_data
        )

    def _format_task_mode_results(self, run: IndividualRun) -> str:
        state_result = run.state_result
        if isinstance(state_result.get("answers", None), list) and "step_name" in state_result:
            return "\n".join(
                f"agent {n}: {x}"
                for (n, x) in zip(state_result["step_name"], state_result["answers"], strict=True)
            )
        return "No answers found in workflow execution."

    def _analyze_results(self, goal: str, results_str: str, execution_time: float) -> dict[str, str]:
        """Analyze execution results using LLM."""
        prompt = f"""Analyze the following Mimosa-AI execution:
TASK: {goal}
EXECUTION TIME: {execution_time:.2f} seconds
EXECUTION RESULTS:
{results_str}
Provide your analysis following the specified output format."""

        analysis_text = self.result_analyzer(prompt)
        analysis = {
            "full_analysis": analysis_text,
            "success_level": "Medium",
            "key_insight": "Analysis completed"
        }
        return analysis

    def sab_files_transfer(self, sab_loader, file_transfer, row):
        """Transfer dataset files to workspace with validation."""
        file_transfer.clean_workspace()
        task_dataset_path = sab_loader.get_dataset_path(row)
        self.logger.info(f"[PAPERS DATASET MODE] Transferring dataset from: {task_dataset_path}")
        print(f"\033[95m📁 Transferring dataset: {task_dataset_path.name}\033[0m")
        files_transferred = file_transfer.transfer_files_to_workspace(str(task_dataset_path))
        time.sleep(0.5)  # Give filesystem a moment to sync
        workspace_files_after = file_transfer.count_files_recursive(Path(file_transfer.workspace_path))
        print(f"\033[95m✓ Transferred {files_transferred} files to workspace\033[0m")
        print(f"\033[95m📊 Verification: {workspace_files_after} files present in workspace\033[0m")

        if workspace_files_after == 0:
            raise ValueError(
                f"Files disappeared after transfer! "
                f"Transferred {files_transferred} but workspace now has 0 files."
            )

        self.logger.info(f"[PAPERS DATASET MODE] Successfully transferred {files_transferred} files")

    def _evaluate_with_science_agent_bench(
        self,
        capsule_name: str,
        row: dict,
        runs: list,
        sab_loader,
        execution_data: dict
    ) -> dict:
        """
        Evaluate results using ScienceAgentBench metrics.

        Args:
            capsule_name: Name of the capsule directory containing results
            row: CSV row data with task information
            runs: List of IndividualRun objects from execution
            sab_loader: ScienceAgentBenchLoader instance
            execution_data: Dictionary to update with evaluation results

        Returns:
            Updated execution_data dictionary with evaluation metrics
        """
        try:
            print("\033[95m📊 Evaluating results with ScienceAgentBench metrics...\033[0m")

            api_cost = runs[-1].cost if runs and hasattr(runs[-1], 'cost') else 0.0
            evaluator = CapsuleEvaluator(
                capsule_path=Path(self.config.runs_capsule_dir) / capsule_name,
                task_data=row,
                sab_loader=sab_loader,
                api_cost=api_cost
            )

            eval_results = evaluator.evaluate_all()
            evaluator.save_results()
            execution_data.update({
                'VER': eval_results['VER'][0],
                'VER_message': eval_results['VER'][1],
                'SR': eval_results['SR'][0],
                'SR_message': eval_results['SR'][1],
                'CBS': eval_results['CBS'],
                'eval_cost': eval_results['cost']
            })
            print(f"\033[95m{eval_results['summary']}\033[0m")

            self.logger.info(
                f"[SAB EVAL] Task {row.get('instance_id')}: "
                f"VER={eval_results['VER'][0]}, "
                f"SR={eval_results['SR'][0]}, "
                f"CBS={eval_results['CBS']:.3f}, "
                f"eval_cost={eval_results['cost']:.3f}"
            )

        except Exception as eval_error:
            self.logger.error(f"[SAB EVAL] Evaluation error: {str(eval_error)}")
            print(f"\033[91m⚠️ Evaluation failed: {str(eval_error)}\033[0m")
            execution_data.update({
                'VER': False,
                'SR': False,
                'CBS': 0.0,
                'eval_error': str(eval_error)
            })

        return execution_data

    async def run_autonomous_eval_loop(self, dataset_type: str, dataset_path: str, learning: bool, single_agent_mode: bool = False) -> None:
        """
        Main autonomous execution loop.
        Generates goals from CSV entries, executes them, analyzes results, and learns.
        """
        papers_csv_path = Path(dataset_path)
        user_input = input("Enter starting row ([Enter] 0 by default): ")
        start_row = int(user_input)-1 if user_input.strip() else 0

        # Load and restore from cache if available
        cached_notes = self._load_previous_run_notes()
        if cached_notes:
            restore_input = input("Restore previous run statistics from cache? (y/n) [Enter for yes]: ")
            if restore_input.strip().lower() != 'n':
                self._restore_execution_history_from_cache(cached_notes)

        sab_loader = None
        if dataset_type == "science_agent_bench":
            sab_loader = ScienceAgentBenchLoader()
            self.logger.info("[PAPERS DATASET MODE] ScienceAgentBench mode activated")

        file_transfer = LocalTransfer(
            workspace_path=self.config.workspace_dir,
            runs_capsule_dir=self.config.runs_capsule_dir,
            config=self.config
        )

        with open(papers_csv_path, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            total_rows = sum(1 for _ in reader)
            csvfile.seek(0)
            reader = csv.DictReader(csvfile)
            self.logger.info(f"[PAPERS DATASET MODE] Starting autonomous loop for {total_rows} CSV entry")
            print(f"\n\033[95m{'🤖 RUN ON PAPERS DATASETS':^80}\033[0m")
            print(f"\033[95m{'=' * 80}\033[0m")
            for i, row in enumerate(reader):
                if i < start_row:
                    print("Skipping evaluation (using cache) for :", i+1)
                    continue
                if i >= self.csv_runs_limit:
                    break
                try:
                    iteration_start_time = time.time()
                    goal = self._generate_next_task(row, dataset_type)
                    print(f"\033[95m📋 GOAL: {goal}\033[0m")

                    if dataset_type == "science_agent_bench" and sab_loader:
                        self.sab_files_transfer(sab_loader, file_transfer, row)
                        runs = await self.dgm.start_dgm(goal=goal,
                                                        judge=True,
                                                        learning_mode=learning,
                                                        max_iteration=self.config.max_learning_evolve_iterations,
                                                        single_agent_mode=single_agent_mode
                                                       )
                        results_str = self._format_task_mode_results(runs[-1])
                    else:
                        tasks_data = await self.planner.start_planner(goal=goal,
                                    judge=True,
                                    max_evolve_iteration=self.config.max_learning_evolve_iterations,
                                    max_task_retry=3
                                   )
                        results_str = self._format_goal_mode_results(tasks_data)
                    print("\033[95m📊 Transfering results files...\033[0m")
                    trs = LocalTransfer(workspace_path=self.config.workspace_dir, runs_capsule_dir=self.config.runs_capsule_dir, config=self.config)
                    capsule_name = trs.transfer_workspace_files_to_capsule(goal)
                    print("\033[95m📊 Analyzing results...\033[0m")
                    execution_time = time.time() - iteration_start_time
                    analysis = self._analyze_results(goal, results_str, execution_time)
                    execution_data = {
                        "iteration": i + 1,
                        "goal": goal,
                        "execution_time": execution_time,
                        "success_level": analysis.get("success_level", "Unknown"),
                        "key_insight": analysis.get("full_analysis", "Unknown")
                    }
                    if dataset_type == "science_agent_bench" and sab_loader:
                        execution_data = self._evaluate_with_science_agent_bench(
                            capsule_name=capsule_name,
                            row=row,
                            runs=runs,
                            sab_loader=sab_loader,
                            execution_data=execution_data
                        )

                    self.execution_history.append(execution_data)
                    self._print_final_summary()
                    self._save_run_notes(
                        capsule_name, goal,
                        analysis, execution_time
                    )

                    print(f"\033[95m✅ Iteration {i + 1} completed\033[0m")
                    print(f"\033[95m   Success Level: {analysis.get('success_level', 'Unknown')}\033[0m")
                    print(f"\033[95m   Time: {execution_time:.2f}s\033[0m")
                except Exception as e:
                    self.logger.error(f"[PAPERS DATASET MODE] Error in csv row {i + 1}: {str(e)}")
                    print(f"\033[91m❌ Error in csv row {i + 1}: {str(e)}\033[0m")
                    continue

        self._print_final_summary()

    def _print_final_summary(self) -> None:
        """Print a summary of all autonomous executions."""
        # Filter out cached entries to count only actual runs from this session
        current_runs = [exec_data for exec_data in self.execution_history
                       if exec_data.get("success_level") != "Cached"]

        successful_runs = [exec_data for exec_data in current_runs
                          if exec_data.get("success_level") in ["High", "Medium"]]
        print(f"\n\033[95mSUMMARY step: {len(current_runs)}\033[0m")
        print(f"\033[95m{'=' * 80}\033[0m")
        print(f"\033[95mSuccessful runs: {len(successful_runs)}\033[0m")
        if len(current_runs) > 0:
            print(f"\033[95mSuccess rate: {len(successful_runs)/len(current_runs)*100:.1f}%\033[0m")

        # For SAB metrics, also exclude cached entries
        sab_runs = [exec_data for exec_data in current_runs
                    if 'VER' in exec_data]
        if sab_runs:
            print(f"\033[95m\n{'ScienceAgentBench Metrics':^80}\033[0m")
            print(f"\033[95m{'-' * 80}\033[0m")
            ver_success = sum(1 for run in sab_runs if run.get('VER', False))
            sr_success = sum(1 for run in sab_runs if run.get('SR', False))
            avg_cbs = sum(run.get('CBS', 0.0) for run in sab_runs) / len(sab_runs)
            total_cost = sum(run.get('eval_cost', 0.0) for run in sab_runs)

            print(f"\033[95mVER (Valid Execution Rate): {ver_success}/{len(sab_runs)} ({ver_success/len(sab_runs)*100:.1f}%)\033[0m")
            print(f"\033[95mSR (Success Rate): {sr_success}/{len(sab_runs)} ({sr_success/len(sab_runs)*100:.1f}%)\033[0m")
            print(f"\033[95mCBS (CodeBERT Score) Average: {avg_cbs:.3f}\033[0m")
            print(f"\033[95mTotal API Cost: ${total_cost:.4f}\033[0m")
            print(f"\033[95mAverage API Cost per Task: ${total_cost/len(sab_runs):.4f}\033[0m")
        print(f"\033[95m{'=' * 80}\033[0m\n")

    async def start_evaluation(self, dataset_type: str = "default", dataset_path = "datasets/our_benchmark.csv", learning=False, single_agent_mode=False) -> None:
        """Public method to start the autonomous mode."""
        try:
            await self.run_autonomous_eval_loop(dataset_type, dataset_path, learning, single_agent_mode)
        except KeyboardInterrupt:
            print("\n\033[95m⚠️ Autonomous mode interrupted by user\033[0m")
            self._print_final_summary()
        except Exception as e:
            self.logger.error(f"[PAPERS DATASET MODE] Fatal error: {str(e)}")
            print(f"\033[91m❌ Fatal error in autonomous mode: {str(e)}\033[0m")
            raise

"""
Evaluation of LLM for Workflow generation
"""

import asyncio
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict

from sources.core.orchestrator import WorkflowOrchestrator
from sources.evaluation.csv_mode import CsvEvaluationMode


class WorkflowEval:
    """
    Class for evaluating success of LLMs on generating workflow for a given dataset tasks
    """
    def __init__(self, config, csv_runs_limit: int = 102):
        self.config = config
        self.csv_runs_limit = csv_runs_limit
        self.orchestrator = WorkflowOrchestrator(config)
        self.model_lists = [
            "anthropic/claude-sonnet-4-5",
            "openrouter/z-ai/glm-5"
            #"openrouter/deepseek/deepseek-v3.2",
            #"openrouter/openrouter/inception/mercury-2",
            #"openrouter/minimax/minimax-m2.5",
            #"openrouter/moonshotai/kimi-k2.5",
            #"openrouter/moonshotai/kimi-k2-thinking",
            #"anthropic/claude-opus-4-5"
        ]
        self.model_results: Dict[str, List[Tuple[str, bool]]] = {}
        self.model_timeouts: Dict[str, int] = {}
        self.model_execution_times: Dict[str, List[float]] = {}

    async def run_workflow_eval_loop(self, dataset_type: str, dataset_path: str) -> None:
        """Run evaluation loop for all models and display comparative report."""
        for model in self.model_lists:
            self.config.prompts_llm_model = model
            self.config.workflow_llm_model = model

            print(f"\n{'='*80}")
            print(f"🤖 EVALUATING MODEL: {model}")
            print(f"{'='*80}\n")

            await self.run_workflow_eval_loop_model(dataset_type, dataset_path, model)

        self._display_comparative_report()

    async def run_workflow_eval_loop_model(self, dataset_type: str, dataset_path: str, model_name: str) -> None:
        """
        Main autonomous execution loop for a single model.
        Generates goals from CSV entries, executes them, analyzes results, and learns.
        """
        papers_csv_path = Path(dataset_path)
        start_row = 0

        per_task_success = []
        timeout_count = 0
        execution_times = []

        with open(papers_csv_path, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            total_rows = sum(1 for _ in reader)
            csvfile.seek(0)
            reader = csv.DictReader(csvfile)

            print(f"\n\033[95m{'EVALUATING LLM WORKFLOW GENERATION ON PAPERS DATASETS':^80}\033[0m")
            print(f"\033[95m{'=' * 80}\033[0m")

            for i, row in enumerate(reader):
                if i < start_row:
                    print(f"Skipping evaluation (using cache) for: {i+1}")
                    continue
                if i >= self.csv_runs_limit:
                    break

                try:
                    iteration_start_time = time.time()
                    goal, _, _ = CsvEvaluationMode._generate_next_task(CsvEvaluationMode(self.config), row, dataset_type)
                    print(f"\033[95m📋 GOAL: {goal}\033[0m")

                    # Wrap orchestrate_workflow with timeout
                    try:
                        run_stdout, uuid, workflow_code, executed = await asyncio.wait_for(
                            self.orchestrator.orchestrate_workflow(
                                goal=goal,
                                craft_instructions=goal,
                                original_task=goal,
                                single_agent_mode=False,
                                no_run=True
                            ),
                            timeout=180
                        )
                        success = executed
                        timed_out = False
                    except asyncio.TimeoutError:
                        print(f"\033[91m⏰ Timeout occurred for task {i + 1}\033[0m")
                        success = False
                        timed_out = True
                        timeout_count += 1
                        run_stdout = uuid = workflow_code = None
                        executed = False

                    # debug print
                    if not timed_out:
                        print(f"\033[95mWorkflow Code:\033[0m")
                        print(workflow_code)
                    print(f"\033[95mSuccess: {'Yes' if success else 'No'}\033[0m")
                    execution_time = time.time() - iteration_start_time
                    execution_times.append(execution_time)

                    print(f"\033[95mBenchmark task {i + 1} Generation {"success" if success else "failure"} \033[0m")
                    print(f"\033[95m   Time: {execution_time:.2f}s\033[0m")

                except Exception as e:
                    success = False
                    print(f"\033[91m❌ Error in csv row {i + 1}: {str(e)}\033[0m")
                    execution_times.append(0.0)  # Add 0 for errors

                per_task_success.append((goal[:128], success))

        # Store results for this model
        self.model_results[model_name] = per_task_success
        self.model_timeouts[model_name] = timeout_count
        self.model_execution_times[model_name] = execution_times

        # Display individual model summary
        self._display_model_summary(model_name, per_task_success, timeout_count, execution_times)

    def _display_model_summary(self, model_name: str, results: List[Tuple[str, bool]], timeout_count: int, execution_times: List[float]) -> None:
        """Display summary for a single model."""
        success_count = sum(success for _, success in results)
        total = len(results)
        percent = (success_count / total) * 100 if total > 0 else 0
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0

        print(f"\n{'─'*60}")
        print(f"MODEL : {model_name}")
        print(f"{'─'*60}")
        print(f"   Score: {percent:.2f}%")
        print(f"   Success: {success_count}/{total} tasks")
        print(f"   Timeouts: {timeout_count}")
        print(f"   Avg Time: {avg_time:.2f}s")
        print(f"{'─'*60}\n")

    def _display_comparative_report(self) -> None:
        """Display final comparative report with all models ranked."""
        if not self.model_results:
            print("\n⚠️  No results to display.\n")
            return

        # Calculate scores for all models
        model_scores = []
        for model, results in self.model_results.items():
            success_count = sum(success for _, success in results)
            total = len(results)
            percent = (success_count / total) * 100 if total > 0 else 0
            timeouts = self.model_timeouts.get(model, 0)
            avg_time = sum(self.model_execution_times.get(model, [])) / len(self.model_execution_times.get(model, [])) if self.model_execution_times.get(model, []) else 0
            model_scores.append((model, percent, success_count, total, timeouts, avg_time))

        # Sort by score (descending)
        model_scores.sort(key=lambda x: x[1], reverse=True)

        # Display report
        print("\n" + "="*80)
        print("🏆 FINAL COMPARATIVE REPORT: LLM WORKFLOW GENERATION BENCHMARK")
        print("="*80)

        for rank, (model, percent, success_count, total, timeouts, avg_time) in enumerate(model_scores, 1):
            is_top = (rank == 1)

            # Highlight top performer
            if is_top:
                print(f"\n{'▓'*80}")
                print(f"🥇 #{rank} TOP PERFORMER")
                print(f"{'▓'*80}")
                print(f"   Model:  {model}")
                print(f"   Score:  \033[92m{percent:.2f}%\033[0m")
                print(f"   Tasks:  {success_count}/{total} successful")
                print(f"   Timeouts: {timeouts}")
                print(f"   Avg Time: {avg_time:.2f}s")
                print(f"{'▓'*80}")
            else:
                medal = "🥈" if rank == 2 else "🥉" if rank == 3 else f" #{rank}"
                print(f"\n{'─'*80}")
                print(f"{medal} #{rank}")
                print(f"{'─'*80}")
                print(f"   Model:  {model}")
                print(f"   Score:  {percent:.2f}%")
                print(f"   Tasks:  {success_count}/{total} successful")
                print(f"   Timeouts: {timeouts}")
                print(f"   Avg Time: {avg_time:.2f}s")

        print("\n" + "="*80)
        print(f"📈 Total models evaluated: {len(model_scores)}")
        print("="*80 + "\n")
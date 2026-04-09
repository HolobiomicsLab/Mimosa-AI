"""
CsvEvaluationMode - Autonomous goal generation and execution system with concurrent evaluation support.
"""

import asyncio
import copy
import csv
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sources.core.dgm import DarwinMachine
from sources.core.llm_provider import LLMConfig, LLMProvider
from sources.core.planner import Planner
from sources.core.schema import Task, IndividualRun
from sources.evaluation.science_agent_bench import ScienceAgentBenchLoader
from sources.evaluation.capsule_evaluator import CapsuleEvaluator
from sources.utils.transfer_toolomics import LocalTransfer
from sources.utils.list_files import list_files
from sources.cli.pretty_print import (
    print_ok, print_warn, print_err, print_info,
    print_phase, print_summary,
)


@dataclass
class TaskContext:
    """Context for a single concurrent task evaluation."""
    row_index: int
    row_data: dict
    workspace_dir: str
    task_id: str

class CsvEvaluationMode:
    """
    Autonomous mode that automatically run Mimosa on various goal's defined in a CSV datasets, such a list of paper to replicate.
    Supports concurrent evaluation of multiple tasks.
    """

    def __init__(self, config, csv_runs_limit: int = 103, max_concurrent_tasks: int = 1, task_start_delay: float = 30.0):
        """
        Initialize CsvEvaluationMode.

        Args:
            config: Mimosa configuration object
            csv_runs_limit: Maximum number of autonomous iterations
            max_concurrent_tasks: Maximum number of tasks to run concurrently (default: 1 for sequential)
            task_start_delay: Delay in seconds between launching consecutive tasks (default: 30s).
                              Staggers agent starts to avoid overwhelming shell/API resources.
        """
        self.config = config
        self.csv_runs_limit = csv_runs_limit
        self.max_concurrent_tasks = max_concurrent_tasks
        self.dgm = DarwinMachine(config)
        self.planner = Planner(config)
        self.run_notes_dir = Path("run_notes")
        self.run_notes_dir.mkdir(exist_ok=True)
        self.done_rows = []

        # Concurrency control
        self.task_start_delay = task_start_delay
        self._semaphore: asyncio.Semaphore | None = None
        self._results_lock = asyncio.Lock()
        self._base_workspace_dir = config.workspace_dir

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
            print_info(f"Restored {total_eval} previous evaluations from cache")

    def _save_run_notes(
        self,
        capsule_name: str,
        goal: str,
        analysis: dict,
        execution_time: float,
        current_execution_data: dict | None = None
    ) -> None:
        """
        Save detailed notes about the run.

        Args:
            capsule_name: Name of the capsule directory
            goal: The task goal
            analysis: Analysis results dictionary
            execution_time: Time taken for execution
            current_execution_data: Optional current task execution data (for concurrent mode).
                                   If provided, this task's data is included even if not yet
                                   in self.execution_history.
        """
        timestamp = datetime.now().isoformat()

        # Build the list of SAB runs, including current task if provided
        sab_runs = [exec_data for exec_data in self.execution_history if 'VER' in exec_data]

        # For concurrent mode: include current_execution_data if it has SAB metrics
        if current_execution_data and 'VER' in current_execution_data:
            # Check if this task is not already in execution_history (concurrent mode)
            if current_execution_data not in sab_runs:
                sab_runs = sab_runs + [current_execution_data]

        notes = {
            "timestamp": timestamp,
            "model": self.config.smolagent_model_id,
            "goal": goal,
            "execution_time_seconds": execution_time,
            "analysis": analysis["full_analysis"],
            "total_eval": len(sab_runs)
        }

        if sab_runs:
            # Use current_execution_data if provided, otherwise use last from sab_runs
            current_task_data = current_execution_data if current_execution_data and 'VER' in current_execution_data else sab_runs[-1]
            runs_data = current_task_data.get('runs', [])

            notes = {
                **notes,
                "capsule_name": capsule_name,
                "ver_success": sum(1 for run in sab_runs if run.get('VER', False)),
                "sr_success": sum(1 for run in sab_runs if run.get('SR', False)),
                "avg_cbs": sum(run.get('CBS', 0.0) for run in sab_runs) / len(sab_runs),
                "total_cost": sum(run.get('eval_cost', 0.0) for run in sab_runs),
                "is_success": current_task_data.get('SR', False),
                "task_cost": current_task_data.get('eval_cost', 0.0),
                "max_judge_reward": max((getattr(run, 'reward', 0.0) for run in runs_data), default=0.0),
                "evolution_iterations": len(runs_data),
                "evolved_workflows_uuids": [getattr(run, 'current_uuid', '') for run in runs_data],
                "evolution_rewards": [getattr(run, 'reward', 0.0) for run in runs_data],
                "evolution_costs": [getattr(run, 'cost', 0.0) for run in runs_data],
                "evolution_total_cost": sum(getattr(run, 'cost', 0.0) for run in runs_data),
                "evolution_avg_reward": sum(getattr(run, 'reward', 0.0) for run in runs_data) / len(runs_data) if runs_data else 0,
                "evolution_avg_cost": sum(getattr(run, 'cost', 0.0) for run in runs_data) / len(runs_data) if runs_data else 0
            }

        notes_file = self.run_notes_dir / f"{capsule_name}.json"
        notes_file.parent.mkdir(parents=True, exist_ok=True)
        with open(notes_file, 'w', encoding='utf-8') as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)
        self.logger.info(f"[PAPERS DATASET MODE] Run notes saved to {notes_file}")

    @staticmethod
    def _extract_workspace_name_from_row(row: dict) -> str:
        """
        Extract a clean workspace folder name from the gold_program_name field.

        Args:
            row: CSV row data containing 'gold_program_name' field

        Returns:
            Cleaned folder name (e.g., 'clintox_nn' from 'clintox_nn.py')
        """
        script_name = (row.get('gold_program_name') or '').strip()
        # Remove .py extension if present
        if script_name.endswith('.py'):
            script_name = script_name[:-3]
        # Fallback to instance_id if script_name is empty
        if not script_name:
            script_name = (row.get('instance_id') or f'task_{id(row)}').strip()
        # Sanitize: replace any non-alphanumeric characters with underscore
        return ''.join(c if c.isalnum() or c == '_' else '_' for c in script_name)

    def _generate_task_default(self, row, workspace_subfolder: str | None = None):
        paper_title = (row.get('Title') or '').strip()
        url = (row.get('URLS') or '').strip()
        prompt = (row.get('Prompt') or '').strip()
        if prompt == "":
            prompt = "Reproduce the experiments from the paper and compare the result."

        workspace_instruction = ""
        if workspace_subfolder:
            workspace_instruction = f"""

⚠️ CRITICAL WORKSPACE REQUIREMENT:
You MUST work exclusively in the workspace subfolder: {workspace_subfolder}
All file operations MUST be performed within this subfolder.
"""

        return f"""
    Paper title: {paper_title}
    Url to paper: {url}
    Goal to achieve: {prompt}
    {workspace_instruction}
        """.strip()

    def _generate_task_science_agent_bench(self, row, workspace_subfolder: str | None = None):
        task_inst = (row.get('task_inst') or '').strip()
        domain_knowledge = (row.get('domain_knowledge') or '').strip()
        dataset_folder_tree = (row.get('dataset_folder_tree') or '').strip()
        dataset_preview = (row.get('dataset_preview') or '').strip()
        output_fname = (row.get('output_fname') or '').strip()
        scenario_id = (row.get('instance_id') or '').strip()
        scoring_rubric_file = (row.get('scoring_rubric_file') or '').strip()
        script_name = (row.get('gold_program_name') or '').strip()

        # Build workspace instruction if subfolder is specified (concurrent mode)
        workspace_instruction = ""
        if workspace_subfolder:
            workspace_instruction = f"""
⚠️ CRITICAL WORKSPACE REQUIREMENT:
You MUST work exclusively in the workspace subfolder: {workspace_subfolder}
All file operations (reading, writing, creating files) MUST be performed within this subfolder.
Do NOT access or modify files outside of this designated workspace.
Your working directory is set to this subfolder - use relative paths from there.
"""

        task_prompt = f"""
DOMAIN KNOWLEDGE:
{domain_knowledge}

INSTRUCTIONS:
{task_inst}
{workspace_instruction}
DATASET STRUCTURE:
{dataset_folder_tree}

DATASET PREVIEW:
{dataset_preview}

EXPECTED OUTPUT:
Save results to a formatted file named exactly: {output_fname}
Keep only one final python script at the root of {workspace_subfolder} named exactly: {script_name}.
You need to respect stricly the output format, otherwise the evaluation will fail.
For example if a input data CSV is named FDA_APPROVED then the column in the output file is also named FDA_APPROVED. No modified pattern such as FDA_APPROVED_prob will be tolerated, otherwise the evaluation will fail.
"""
        return task_prompt, scenario_id, scoring_rubric_file

    def _generate_next_task(self, row, dataset_type: str) -> str:
        """Generate the next goal using LLM based on paper from the CSV."""
        try:
            if dataset_type == "science_agent_bench":
                task, scenario_id, scoring_rubric_file = self._generate_task_science_agent_bench(row)
                return task, scenario_id, scoring_rubric_file
            return self._generate_task_default(row), None, None
        except Exception as e:
            self.logger.error(f"Error generating task for row: {row}, error: {e}")
            return "Error generating task", None, None

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
        files = list_files(self.config.workspace_dir, max_depth=3)
        prompt = f"""Analyze the following Mimosa-AI execution:
TASK: {goal}
EXECUTION TIME: {execution_time:.2f} seconds
FILES USED, GENERATED OR MODIFIED (up to 3 levels deep):
{files}
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
        print_info(f"📁 Transferring dataset: {task_dataset_path.name}")
        files_transferred = file_transfer.transfer_files_to_workspace(str(task_dataset_path))
        time.sleep(0.5)  # Give filesystem a moment to sync
        workspace_files_after = file_transfer.count_files_recursive(Path(file_transfer.workspace_path))
        print_ok(f"Transferred {files_transferred} file(s) to workspace")
        print_info(f"Verification: {workspace_files_after} file(s) present in workspace")

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
            print_info("📊 Evaluating results with ScienceAgentBench metrics…")

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
                'eval_cost': eval_results['cost'],
                'runs': runs
            })
            print_ok(eval_results['summary'])

            self.logger.info(
                f"[SAB EVAL] Task {row.get('instance_id')}: "
                f"VER={eval_results['VER'][0]}, "
                f"SR={eval_results['SR'][0]}, "
                f"CBS={eval_results['CBS']:.3f}, "
                f"eval_cost={eval_results['cost']:.3f}"
            )

        except Exception as eval_error:
            self.logger.error(f"[SAB EVAL] Evaluation error: {str(eval_error)}")
            print_err(f"Evaluation failed: {str(eval_error)}")
            execution_data.update({
                'VER': False,
                'SR': False,
                'CBS': 0.0,
                'eval_error': str(eval_error),
                'runs': runs
            })

        return execution_data

    def _create_isolated_config(self, task_id: str) -> Any:
        """
        Create a copy of config with an isolated workspace directory for concurrent execution.

        Args:
            task_id: Unique identifier for the task (used in workspace path)

        Returns:
            A copy of the config with modified workspace_dir
        """
        isolated_config = copy.copy(self.config)
        isolated_workspace = Path(self._base_workspace_dir) / f"worker_{task_id}"
        isolated_workspace.mkdir(parents=True, exist_ok=True)
        isolated_config.workspace_dir = str(isolated_workspace)
        return isolated_config

    def _cleanup_isolated_workspace(self, task_id: str) -> None:
        """
        Clean up an isolated workspace after task completion.

        Args:
            task_id: Unique identifier for the task
        """
        isolated_workspace = Path(self._base_workspace_dir) / f"worker_{task_id}"
        if isolated_workspace.exists():
            try:
                shutil.rmtree(isolated_workspace, ignore_errors=True)
                self.logger.debug(f"[CONCURRENT] Cleaned up workspace for task {task_id}")
            except Exception as e:
                self.logger.warning(f"[CONCURRENT] Failed to cleanup workspace for task {task_id}: {e}")

    def _generate_next_task_concurrent(self, row: dict, dataset_type: str, workspace_subfolder: str) -> tuple[str, str | None, str | None]:
        """
        Generate the next goal for concurrent execution, including workspace subfolder in prompt.

        Args:
            row: CSV row data
            dataset_type: Type of dataset being evaluated
            workspace_subfolder: The workspace subfolder name for this task

        Returns:
            Tuple of (task_prompt, scenario_id, scoring_rubric_file)
        """
        try:
            if dataset_type == "science_agent_bench":
                task, scenario_id, scoring_rubric_file = self._generate_task_science_agent_bench(row, workspace_subfolder)
                return task, scenario_id, scoring_rubric_file
            return self._generate_task_default(row, workspace_subfolder), None, None
        except Exception as e:
            self.logger.error(f"Error generating task for row: {row}, error: {e}")
            return "Error generating task", None, None

    async def _process_single_task(
        self,
        task_context: TaskContext,
        dataset_type: str,
        learning: bool,
        single_agent_mode: bool,
        sab_loader: ScienceAgentBenchLoader | None,
        launch_index: int = 0
    ) -> dict[str, Any]:
        """
        Process a single task evaluation in an isolated environment.

        Args:
            task_context: Context containing row data and workspace info
            dataset_type: Type of dataset being evaluated
            learning: Whether learning mode is enabled
            single_agent_mode: Whether to use single agent mode
            sab_loader: ScienceAgentBench loader instance (if applicable)
            launch_index: Position in the launch queue, used to stagger task starts

        Returns:
            Execution data dictionary with results
        """
        row = task_context.row_data
        i = task_context.row_index

        # Extract workspace name from gold_program_name (e.g., 'clintox_nn' from 'clintox_nn.py')
        workspace_name = self._extract_workspace_name_from_row(row)
        task_id = workspace_name  # Use the clean name as task_id

        # Stagger task launches to avoid overwhelming shell/API resources
        if launch_index > 0 and self.task_start_delay > 0:
            stagger_delay = launch_index * self.task_start_delay
            self.logger.info(f"[CONCURRENT] Task {i + 1} (workspace: {workspace_name}) waiting {stagger_delay:.1f}s before starting (stagger delay)")
            print(f"\033[93m[Worker {workspace_name}] ⏳ Stagger delay: waiting {stagger_delay:.1f}s before starting...\033[0m")
            await asyncio.sleep(stagger_delay)

        # Acquire semaphore to limit concurrency
        async with self._semaphore:
            self.logger.info(f"[CONCURRENT] Starting task {i + 1} (workspace: {workspace_name})")
            print(f"\033[96m[Worker {workspace_name}] Starting task {i + 1}\033[0m")

            # Create isolated config and instances for this task
            isolated_config = self._create_isolated_config(task_id)
            workspace_subfolder = f"worker_{task_id}"

            try:
                iteration_start_time = time.time()
                # Generate task with workspace subfolder information in the prompt
                goal, scenario_id, scenario_rubric_filename = self._generate_next_task_concurrent(
                    row, dataset_type, workspace_subfolder
                )

                print(f"\033[96m[Worker {workspace_name}] 📋 GOAL: {goal[:100]}...\033[0m")
                print(f"\033[96m[Worker {workspace_name}] 📄 Scenario Rubric: {scenario_rubric_filename}\033[0m")

                # Create isolated DGM/Planner instances
                isolated_dgm = DarwinMachine(isolated_config)
                isolated_planner = Planner(isolated_config)

                # Create file transfer with isolated workspace
                file_transfer = LocalTransfer(
                    config=isolated_config,
                    workspace_path=isolated_config.workspace_dir,
                    runs_capsule_dir=self.config.runs_capsule_dir
                )

                runs = None
                if dataset_type == "science_agent_bench" and sab_loader:
                    # Transfer files to isolated workspace
                    self._sab_files_transfer_isolated(sab_loader, file_transfer, row, task_id)

                    runs = await isolated_dgm.start_dgm(
                        goal=goal,
                        judge=True,
                        learning_mode=learning,
                        scenario_rubric=None,
                        max_iteration=self.config.max_learning_evolve_iterations,
                        single_agent_mode=single_agent_mode
                    )
                    results_str = self._format_task_mode_results(runs[-1])
                else:
                    tasks_data = await isolated_planner.start_planner(
                        goal=goal,
                        judge=True,
                        max_evolve_iteration=self.config.max_learning_evolve_iterations,
                        max_task_retry=3
                    )
                    results_str = self._format_goal_mode_results(tasks_data)

                print(f"\033[96m[Worker {task_id}] 📊 Transferring results files...\033[0m")

                # Transfer results to capsule (uses shared capsule dir)
                trs = LocalTransfer(
                    config=isolated_config,
                    workspace_path=isolated_config.workspace_dir,
                    runs_capsule_dir=self.config.runs_capsule_dir
                )
                capsule_name = trs.transfer_workspace_files_to_capsule(goal)

                print(f"\033[96m[Worker {task_id}] 📊 Analyzing results...\033[0m")
                execution_time = time.time() - iteration_start_time

                # Analyze results using isolated workspace path
                analysis = self._analyze_results_isolated(goal, results_str, execution_time, isolated_config.workspace_dir)

                execution_data = {
                    "iteration": i + 1,
                    "goal": goal,
                    "execution_time": execution_time,
                    "success_level": analysis.get("success_level", "Unknown"),
                    "key_insight": analysis.get("full_analysis", "Unknown"),
                    "task_id": task_id
                }

                if dataset_type == "science_agent_bench" and sab_loader and runs:
                    execution_data = self._evaluate_with_science_agent_bench(
                        capsule_name=capsule_name,
                        row=row,
                        runs=runs,
                        sab_loader=sab_loader,
                        execution_data=execution_data
                    )

                print(f"\033[96m[Worker {task_id}] ✅ Task {i + 1} completed in {execution_time:.2f}s\033[0m")
                print(f"\033[96m[Worker {task_id}]    Success Level: {analysis.get('success_level', 'Unknown')}\033[0m")

                # Save run notes (thread-safe via file system)
                # Pass current execution_data for concurrent mode since execution_history isn't updated yet
                self._save_run_notes(
                    capsule_name, goal, analysis, execution_time,
                    current_execution_data=execution_data
                )

                return execution_data

            except Exception as e:
                self.logger.error(f"[CONCURRENT] Error in task {i + 1} (worker_{task_id}): {str(e)}")
                print(f"\033[91m[Worker {task_id}] ❌ Error in task {i + 1}: {str(e)}\033[0m")
                return {
                    "iteration": i + 1,
                    "goal": str(goal) if 'goal' in dir() else "Unknown",
                    "execution_time": time.time() - iteration_start_time if 'iteration_start_time' in dir() else 0,
                    "success_level": "Error",
                    "key_insight": str(e),
                    "task_id": task_id,
                    "error": str(e)
                }

            finally:
                # Cleanup isolated workspace
                self._cleanup_isolated_workspace(task_id)

    def _sab_files_transfer_isolated(self, sab_loader, file_transfer, row, task_id: str):
        """Transfer dataset files to isolated workspace with validation."""
        file_transfer.clean_workspace()
        task_dataset_path = sab_loader.get_dataset_path(row)
        self.logger.info(f"[Worker {task_id}] Transferring dataset from: {task_dataset_path}")
        print(f"\033[96m[Worker {task_id}] 📁 Transferring dataset: {task_dataset_path.name}\033[0m")
        files_transferred = file_transfer.transfer_files_to_workspace(str(task_dataset_path))
        time.sleep(0.3)  # Give filesystem a moment to sync
        workspace_files_after = file_transfer.count_files_recursive(Path(file_transfer.workspace_path))
        print(f"\033[96m[Worker {task_id}] ✓ Transferred {files_transferred} files to workspace\033[0m")

        if workspace_files_after == 0:
            raise ValueError(
                f"Files disappeared after transfer! "
                f"Transferred {files_transferred} but workspace now has 0 files."
            )
        self.logger.info(f"[Worker {task_id}] Successfully transferred {files_transferred} files")

    def _analyze_results_isolated(self, goal: str, results_str: str, execution_time: float, workspace_dir: str) -> dict[str, str]:
        """Analyze execution results using LLM with isolated workspace."""
        files = list_files(workspace_dir, max_depth=3)
        prompt = f"""Analyze the following Mimosa-AI execution:
TASK: {goal}
EXECUTION TIME: {execution_time:.2f} seconds
FILES USED, GENERATED OR MODIFIED (up to 3 levels deep):
{files}
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

    async def run_concurrent_eval_loop(
        self,
        dataset_type: str,
        dataset_path: str,
        learning: bool,
        single_agent_mode: bool = False
    ) -> None:
        """
        Concurrent execution loop that processes multiple tasks in parallel.
        Uses asyncio.gather with semaphore-based concurrency control.

        Args:
            dataset_type: Type of dataset being evaluated
            dataset_path: Path to the CSV dataset file
            learning: Whether learning mode is enabled
            single_agent_mode: Whether to use single agent mode
        """
        papers_csv_path = Path(dataset_path)

        # Get starting row from user
        while True:
            user_input = input("Enter starting row ([Enter] 0 by default): ")
            if not user_input.strip():
                start_row = 0
                break
            try:
                start_row = int(user_input) - 1
                break
            except ValueError:
                print(f"  ⚠️  Invalid value '{user_input}' – please enter a whole number.")

        # Load and restore from cache if available
        cached_notes = self._load_previous_run_notes()
        if cached_notes:
            restore_input = input("Restore previous run statistics from cache? (y/n) [Enter for yes]: ")
            if restore_input.strip().lower() != 'n':
                self._restore_execution_history_from_cache(cached_notes)

        # Initialize semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        # Initialize ScienceAgentBench loader if needed
        sab_loader = None
        if dataset_type == "science_agent_bench":
            sab_loader = ScienceAgentBenchLoader()
            self.logger.info("[CONCURRENT] ScienceAgentBench mode activated")

        # Read CSV and prepare task contexts
        task_contexts: list[TaskContext] = []
        with open(papers_csv_path, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            total_rows = sum(1 for _ in reader)
            csvfile.seek(0)
            reader = csv.DictReader(csvfile)

            self.logger.info(f"[CONCURRENT] Preparing tasks for {total_rows} CSV entries")
            print(f"\n\033[95m{'🤖 CONCURRENT EVALUATION MODE':^80}\033[0m")
            print(f"\033[95m{'=' * 80}\033[0m")
            print(f"\033[95mMax concurrent tasks: {self.max_concurrent_tasks}\033[0m")
            print(f"\033[95mStagger delay between launches: {self.task_start_delay:.1f}s\033[0m")
            print(f"\033[95mTotal rows in CSV: {total_rows}\033[0m")
            print(f"\033[95m{'=' * 80}\033[0m\n")

            for i, row in enumerate(reader):
                if i < start_row:
                    print(f"Skipping evaluation (using cache) for: {i + 1}")
                    continue
                if i >= self.csv_runs_limit:
                    break

                task_id = f"{i + 1}_{int(time.time() * 1000) % 10000}"
                task_contexts.append(TaskContext(
                    row_index=i,
                    row_data=dict(row),  # Make a copy of the row
                    workspace_dir="",  # Will be set in _process_single_task
                    task_id=task_id
                ))

        if not task_contexts:
            print("\033[93m⚠️ No tasks to process\033[0m")
            return

        print(f"\033[95m📋 Processing {len(task_contexts)} tasks with {self.max_concurrent_tasks} concurrent workers\033[0m\n")

        # Create coroutines for all tasks with staggered launch indices
        tasks = [
            self._process_single_task(
                task_context=ctx,
                dataset_type=dataset_type,
                learning=learning,
                single_agent_mode=single_agent_mode,
                sab_loader=sab_loader,
                launch_index=idx
            )
            for idx, ctx in enumerate(task_contexts)
        ]

        # Execute all tasks concurrently with semaphore control
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results thread-safely
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"[CONCURRENT] Task failed with exception: {result}")
                self.execution_history.append({
                    "iteration": -1,
                    "goal": "Unknown",
                    "execution_time": 0,
                    "success_level": "Error",
                    "key_insight": str(result)
                })
            elif isinstance(result, dict):
                self.execution_history.append(result)

        # Sort execution history by iteration for consistent ordering
        self.execution_history.sort(key=lambda x: x.get("iteration", 0))

        self._print_final_summary()

    async def run_single_thread_eval_loop(self, dataset_type: str, dataset_path: str, learning: bool, single_agent_mode: bool = False) -> None:
        """
        Main autonomous execution loop.
        Generates goals from CSV entries, executes them, analyzes results, and learns.
        """
        papers_csv_path = Path(dataset_path)
        while True:
            user_input = input("Enter starting row ([Enter] 0 by default): ")
            if not user_input.strip():
                start_row = 0
                break
            try:
                start_row = int(user_input) - 1
                break
            except ValueError:
                print(f"  ⚠️  Invalid value '{user_input}' – please enter a whole number.")

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
            config=self.config,
            workspace_path=self.config.workspace_dir,
            runs_capsule_dir=self.config.runs_capsule_dir
        )

        with open(papers_csv_path, encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            total_rows = sum(1 for _ in reader)
            csvfile.seek(0)
            reader = csv.DictReader(csvfile)
            self.logger.info(f"[PAPERS DATASET MODE] Starting autonomous loop for {total_rows} CSV entry")
            print_phase("🤖 RUN ON PAPERS DATASETS")
            for i, row in enumerate(reader):
                if i < start_row:
                    print_info(f"Skipping evaluation (using cache) for row {i + 1}")
                    continue
                if i >= self.csv_runs_limit:
                    break
                try:
                    iteration_start_time = time.time()
                    goal, scenario_id, scenario_rubric_filename = self._generate_next_task(row, dataset_type)
                    print_info(f"📋 GOAL: {goal[:120]}…" if len(goal) > 120 else f"📋 GOAL: {goal}")
                    print_info(f"📄 Scenario Rubric: {scenario_rubric_filename}")

                    if dataset_type == "science_agent_bench" and sab_loader:
                        self.sab_files_transfer(sab_loader, file_transfer, row)
                        runs = await self.dgm.start_dgm(goal=goal,
                                                        judge=True,
                                                        learning_mode=learning,
                                                        scenario_rubric=None, # Don't pass scenario rubric to use generic evaluator (file based scenario doesn't give mearningful feedback signal for workflow iterative refinements as for now)
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
                    print_info("📦 Transferring results files…")
                    trs = LocalTransfer(config=self.config, workspace_path=self.config.workspace_dir, runs_capsule_dir=self.config.runs_capsule_dir)
                    capsule_name = trs.transfer_workspace_files_to_capsule(goal)
                    print_info("📊 Analyzing results…")
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

                    print_ok(f"Iteration {i + 1} completed")
                    print_info(f"  Success Level: {analysis.get('success_level', 'Unknown')}")
                    print_info(f"  Time: {execution_time:.2f}s")
                except Exception as e:
                    self.logger.error(f"[PAPERS DATASET MODE] Error in csv row {i + 1}: {str(e)}")
                    print(f"\033[91m❌ Error in csv row {i + 1}: {str(e)}\033[0m")
                    raise e

        self._print_final_summary()

    def _print_final_summary(self) -> None:
        """Print a summary of all autonomous executions."""
        # Filter out cached entries to count only actual runs from this session
        current_runs = [exec_data for exec_data in self.execution_history
                       if exec_data.get("success_level") != "Cached"]

        successful_runs = [exec_data for exec_data in current_runs
                          if exec_data.get("success_level") in ["High", "Medium"]]

        success_rate = (
            f"{len(successful_runs)/len(current_runs)*100:.1f}%"
            if current_runs else "N/A"
        )
        rows = [
            ("Steps evaluated", str(len(current_runs))),
            ("Successful runs", str(len(successful_runs))),
            ("Success rate", success_rate),
        ]

        # For SAB metrics, also exclude cached entries
        sab_runs = [exec_data for exec_data in current_runs if 'VER' in exec_data]
        if sab_runs:
            ver_success = sum(1 for run in sab_runs if run.get('VER', False))
            sr_success = sum(1 for run in sab_runs if run.get('SR', False))
            avg_cbs = sum(run.get('CBS', 0.0) for run in sab_runs) / len(sab_runs)
            total_cost = sum(run.get('eval_cost', 0.0) for run in sab_runs)
            rows += [
                ("── ScienceAgentBench ──", ""),
                ("VER (Valid Exec Rate)", f"{ver_success}/{len(sab_runs)} ({ver_success/len(sab_runs)*100:.1f}%)"),
                ("SR  (Success Rate)", f"{sr_success}/{len(sab_runs)} ({sr_success/len(sab_runs)*100:.1f}%)"),
                ("CBS (CodeBERT avg)", f"{avg_cbs:.3f}"),
                ("Total API Cost", f"${total_cost:.4f}"),
                ("Avg cost/task", f"${total_cost/len(sab_runs):.4f}"),
            ]

        print_summary("📊 EVALUATION SUMMARY", rows)

    async def start_evaluation(
        self,
        dataset_type: str = "default",
        dataset_path: str = "datasets/our_benchmark.csv",
        learning: bool = False,
        single_agent_mode: bool = False,
        concurrent: bool = False
    ) -> None:
        """
        Public method to start the evaluation mode.

        Args:
            dataset_type: Type of dataset ("default" or "science_agent_bench")
            dataset_path: Path to the CSV dataset file
            learning: Whether to enable learning mode
            single_agent_mode: Whether to use single agent mode
            concurrent: Whether to run tasks concurrently (uses max_concurrent_tasks from init)
        """
        try:
            if concurrent and self.max_concurrent_tasks > 1:
                print(f"\033[95mStarting CONCURRENT evaluation with {self.max_concurrent_tasks} workers\033[0m")
                await self.run_concurrent_eval_loop(dataset_type, dataset_path, learning, single_agent_mode)
            else:
                if concurrent and self.max_concurrent_tasks <= 1:
                    print("\033[93m⚠️ Concurrent mode requested but max_concurrent_tasks <= 1, falling back to sequential\033[0m")
                await self.run_single_thread_eval_loop(dataset_type, dataset_path, learning, single_agent_mode)
        except KeyboardInterrupt:
            print_warn("Autonomous mode interrupted by user")
            self._print_final_summary()
        except Exception as e:
            self.logger.error(f"[PAPERS DATASET MODE] Fatal error: {str(e)}")
            print_err(f"Fatal error in autonomous mode: {str(e)}")
            raise

    async def start_concurrent_evaluation(
        self,
        dataset_type: str = "default",
        dataset_path: str = "datasets/our_benchmark.csv",
        learning: bool = False,
        single_agent_mode: bool = False
    ) -> None:
        """
        Convenience method to start concurrent evaluation directly.
        Equivalent to start_evaluation(..., concurrent=True)

        Args:
            dataset_type: Type of dataset ("default" or "science_agent_bench")
            dataset_path: Path to the CSV dataset file
            learning: Whether to enable learning mode
            single_agent_mode: Whether to use single agent mode
        """
        await self.start_evaluation(
            dataset_type=dataset_type,
            dataset_path=dataset_path,
            learning=learning,
            single_agent_mode=single_agent_mode,
            concurrent=True
        )

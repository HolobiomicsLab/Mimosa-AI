"""
CsvEvaluationMode - Autonomous goal generation and execution system with concurrent evaluation support.
"""

import asyncio
import copy
import csv
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sources.core.evolution_engine import EvolutionEngine
from sources.core.planner import Planner
from sources.benchmark_evaluation.science_agent_bench import ScienceAgentBenchLoader
from sources.benchmark_evaluation.capsule_evaluator import CapsuleEvaluator
from sources.utils.transfer_toolomics import LocalTransfer
from sources.utils.email_reporter import send_evaluation_report
from sources.cli.pretty_print import (
    print_ok, print_warn, print_err, print_info,
    print_phase, print_summary,
)


async def _prompt_with_default(prompt: str, default: str = "0") -> str:
    """
    Prompt for input with a default. Headless-safe and recovery-correct.

    Behavior:
      - If stdin is not a TTY (headless / cron / piped run), return *default*
        immediately without reading.
      - Otherwise block on stdin until the user enters a value; return
        *default* on empty input.

    Why: the previous `asyncio.wait_for(loop.run_in_executor(None, input))`
    pattern leaked the blocked reader thread on timeout. Any late keystrokes
    were silently consumed by the orphaned thread, and back-to-back prompts
    raced for stdin — so the recovery prompts (starting row, cache restore)
    could be ignored without the user ever knowing. Block for a TTY, return
    the default for non-TTY: predictable, no thread leak, no race.
    """
    if not sys.stdin.isatty():
        print(f"{prompt} (stdin not a TTY — using default '{default}')")
        return default
    loop = asyncio.get_running_loop()
    print(f"{prompt} (press Enter for '{default}'): ", end="", flush=True)
    raw = await loop.run_in_executor(None, input)
    return raw.strip() if raw.strip() else default


def _is_excluded(run: dict) -> bool:
    """True if a run was dropped as an eval-infra failure (not a real VER/SR result)."""
    return run.get("status") == "excluded" or run.get("success_level") == "Excluded"


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

    def __init__(self, config, csv_runs_limit: int = 103, max_concurrent_tasks: int = 1,
                 task_start_delay: float = 30.0, run_notes_dir: str | Path = "run_notes"):
        """
        Initialize CsvEvaluationMode.

        Args:
            config: Mimosa configuration object
            csv_runs_limit: Maximum number of autonomous iterations
            max_concurrent_tasks: Maximum number of tasks to run concurrently (default: 1 for sequential)
            task_start_delay: Delay in seconds between launching consecutive tasks (default: 30s).
                              Staggers agent starts to avoid overwhelming shell/API resources.
            run_notes_dir: Directory for per-task run notes. Per-run in queued CLI
                              mode so concurrent runs never restore each other's cache.
        """
        self.config = config
        self.csv_runs_limit = csv_runs_limit
        self.max_concurrent_tasks = max_concurrent_tasks
        self.evolve = EvolutionEngine(config)
        self.planner = Planner(config)
        self.run_notes_dir = Path(run_notes_dir)
        self.run_notes_dir.mkdir(parents=True, exist_ok=True)
        self.done_rows = []

        # Concurrency control
        self.task_start_delay = task_start_delay
        self._semaphore: asyncio.Semaphore | None = None
        self._base_workspace_dir = config.workspace_dir

        # Track execution history
        self.execution_history: list[dict] = []
        self.logger = logging.getLogger(__name__)

        # Run-level context captured by start_evaluation for the email report.
        self._dataset_type: str | None = None
        self._dataset_path: str | None = None
        self._learning: bool = False
        self._single_agent_mode: bool = False
        self._concurrent: bool = False
        self._start_row: int = 0

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
                    if not model or not self.config.smolagent_model_id:
                        continue
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
        execution_time: float,
        current_execution_data: dict | None = None
    ) -> None:
        """
        Save detailed notes about the run.

        Args:
            capsule_name: Name of the capsule directory
            goal: The task goal
            execution_time: Time taken for execution
            current_execution_data: Optional current task execution data (for concurrent mode).
                                   If provided, this task's data is included even if not yet
                                   in self.execution_history.
        """
        timestamp = datetime.now().isoformat()

        # Build the list of evaluated SAB runs (VER is None for infra-excluded tasks).
        sab_runs = [exec_data for exec_data in self.execution_history
                    if exec_data.get('VER') is not None]

        # For concurrent mode: include current_execution_data if it was evaluated
        if current_execution_data and current_execution_data.get('VER') is not None:
            # Check if this task is not already in execution_history (concurrent mode)
            if current_execution_data not in sab_runs:
                sab_runs = sab_runs + [current_execution_data]

        notes = {
            "timestamp": timestamp,
            "model": self.config.smolagent_model_id,
            "goal": goal,
            "execution_time_seconds": execution_time,
            "total_eval": len(sab_runs),
            "start_row": self._start_row + 1,
            "git": self._get_git_info()
        }

        if sab_runs:
            # Use current_execution_data if provided, otherwise use last from sab_runs
            current_task_data = current_execution_data if current_execution_data and current_execution_data.get('VER') is not None else sab_runs[-1]
            runs_data = current_task_data.get('runs', [])

            notes = {
                **notes,
                "capsule_name": capsule_name,
                "ver_success": sum(1 for sab in sab_runs if sab.get('VER', False)),
                "sr_success": sum(1 for sab in sab_runs if sab.get('SR', False)),
                "avg_cbs": sum(sab.get('CBS', 0.0) for sab in sab_runs) / len(sab_runs),
                "total_cost": sum(sab.get('eval_cost', 0.0) for sab in sab_runs),
                "is_success": current_task_data.get('SR', False),
                "task_cost": current_task_data.get('eval_cost', 0.0),
                "max_judge_reward": max((getattr(run, 'reward', 0.0) for run in runs_data), default=0.0),
                "evolution_iterations": len(runs_data),
                "evolved_workflows_uuids": [getattr(run, 'current_uuid', '') for run in runs_data],
                "evolution_rewards": [getattr(run, 'reward', 0.0) for run in runs_data],
                "evolution_costs": self._compute_per_iteration_costs(runs_data),
                "evolution_total_cost": getattr(runs_data[-1], 'cost', 0.0) if runs_data else 0.0,
                "evolution_avg_reward": sum(getattr(run, 'reward', 0.0) for run in runs_data) / len(runs_data) if runs_data else 0,
                "evolution_avg_cost": (getattr(runs_data[-1], 'cost', 0.0) / len(runs_data)) if runs_data else 0
            }

        notes_file = self.run_notes_dir / f"{capsule_name}.json"
        notes_file.parent.mkdir(parents=True, exist_ok=True)
        with open(notes_file, 'w', encoding='utf-8') as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)
        self.logger.info(f"[PAPERS DATASET MODE] Run notes saved to {notes_file}")

    @staticmethod
    def _get_git_info() -> dict:
        """
        Capture the current git commit, branch and working-tree state of the repo.

        Recorded in run notes so each run can be tied back to the exact code that
        produced it. Returns None values when git metadata is unavailable.
        """
        repo_dir = Path(__file__).resolve().parent

        def _git(*args: str) -> str | None:
            try:
                return subprocess.run(
                    ["git", *args],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                ).stdout.strip()
            except (subprocess.SubprocessError, OSError):
                return None

        status = _git("status", "--porcelain")
        return {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None,
        }

    @staticmethod
    def _compute_per_iteration_costs(runs_data: list) -> list[float]:
        """
        Compute per-iteration costs from IndividualRun objects.

        IndividualRun.cost is cumulative (each run accumulates cost from prior runs).
        This method extracts the per-iteration cost by computing deltas between
        consecutive runs.

        Args:
            runs_data: List of IndividualRun objects

        Returns:
            List of per-iteration costs (one per run)
        """
        if not runs_data:
            return []
        costs = []
        prev_cost = 0.0
        for run in runs_data:
            run_cost = getattr(run, 'cost', 0.0)
            costs.append(run_cost - prev_cost)
            prev_cost = run_cost
        return costs

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
⚠️ CRITICAL WORKSPACE REQUIREMENT ⚠️
Your ENTIRE working environment is confined to the subfolder: {workspace_subfolder}
• ALL file reads, writes and creations MUST happen inside {workspace_subfolder}/ — never outside.
• Treat {workspace_subfolder}/ as your root directory and use paths relative to it.
• Do NOT access, create or modify anything outside {workspace_subfolder}/.
. Any scripts, files, notes or folders you create during your work MUST also be inside {workspace_subfolder}/
"""

        # Build explicit output paths
        output_path = f"{workspace_subfolder}/{output_fname}" if workspace_subfolder else output_fname
        script_path = f"{workspace_subfolder}/{script_name}" if workspace_subfolder else script_name

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
1. Results file — save to the EXACT path: {output_path}
   (i.e. at the root of your workspace subfolder, not in any sub-directory)
2. Python script — keep exactly ONE final script at: {script_path}
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

    async def sab_files_transfer(self, sab_loader, file_transfer, row):
        """Transfer dataset files to workspace with validation."""
        file_transfer.clean_workspace()
        task_dataset_path = sab_loader.get_dataset_path(row)
        self.logger.info(f"[PAPERS DATASET MODE] Transferring dataset from: {task_dataset_path}")
        print_info(f"📁 Transferring dataset: {task_dataset_path.name}")
        files_transferred = file_transfer.transfer_files_to_workspace(str(task_dataset_path))
        await asyncio.sleep(0.5)  # Give filesystem a moment to sync
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

            if eval_results.get('status') == 'excluded':
                infra_error = eval_results.get('infra_error')
                execution_data.update({
                    'status': 'excluded',
                    'infra_error': infra_error,
                    'VER': None,
                    'SR': None,
                    'CBS': None,
                    'eval_cost': eval_results['cost'],
                    'runs': runs,
                    'success_level': "Excluded",
                })
                print_warn(f"Task {row.get('instance_id')} EXCLUDED (infra): {infra_error}")
                self.logger.warning(
                    f"[SAB EVAL] Task {row.get('instance_id')} EXCLUDED (infra): {infra_error}"
                )
            else:
                execution_data.update({
                    'status': 'evaluated',
                    'VER': eval_results['VER'][0],
                    'VER_message': eval_results['VER'][1],
                    'SR': eval_results['SR'][0],
                    'SR_message': eval_results['SR'][1],
                    'CBS': eval_results['CBS'],
                    'eval_cost': eval_results['cost'],
                    'runs': runs,
                    'success_level': "Success" if eval_results['VER'][0] else "Failed"
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
            # An exception escaping the evaluator is a harness fault, not the
            # agent's — exclude it (loudly) rather than counting it as SR=0.
            self.logger.error(
                f"[SAB EVAL] Unexpected harness error — EXCLUDING task "
                f"{row.get('instance_id')}: {eval_error}",
                exc_info=True,
            )
            print_warn(f"Task {row.get('instance_id')} EXCLUDED (harness error): {eval_error}")
            execution_data.update({
                'status': 'excluded',
                'infra_error': f"Unexpected harness error: {eval_error}",
                'VER': None,
                'SR': None,
                'CBS': None,
                'eval_error': str(eval_error),
                'runs': runs,
                'success_level': "Excluded",
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

                # Create isolated evolution engine/Planner instances
                isolated_dgm = EvolutionEngine(isolated_config)
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
                    await self._sab_files_transfer_isolated(sab_loader, file_transfer, row, task_id)

                    runs = await isolated_dgm.start_workflow_evolution(
                        goal=goal,
                        judge=True,
                        enable_evolution=learning,
                        scenario_rubric=None,
                        single_agent_mode=single_agent_mode
                    )
                else:
                    _ = await isolated_planner.start_planner(
                        goal=goal,
                        judge=True,
                        max_task_retry=3
                    )

                print(f"\033[96m[Worker {task_id}] 📊 Transferring results files...\033[0m")

                # Transfer results to capsule (uses shared capsule dir).
                # Offloaded: the capsule namer is a blocking sync LLM call.
                trs = LocalTransfer(
                    config=isolated_config,
                    workspace_path=isolated_config.workspace_dir,
                    runs_capsule_dir=self.config.runs_capsule_dir
                )
                capsule_name = await asyncio.to_thread(
                    trs.transfer_workspace_files_to_capsule, goal, task_token=task_id
                )

                print(f"\033[96m[Worker {task_id}] 📊 Analyzing results...\033[0m")
                execution_time = time.time() - iteration_start_time

                execution_data = {
                    "iteration": i + 1,
                    "goal": goal,
                    "execution_time": execution_time,
                    "task_id": task_id
                }

                if dataset_type == "science_agent_bench" and sab_loader and runs:
                    # Offloaded: sandbox build, VER/SR subprocesses and CBS are blocking.
                    execution_data = await asyncio.to_thread(
                        self._evaluate_with_science_agent_bench,
                        capsule_name=capsule_name,
                        row=row,
                        runs=runs,
                        sab_loader=sab_loader,
                        execution_data=execution_data
                    )

                print(f"\033[96m[Worker {task_id}] ✅ Task {i + 1} completed in {execution_time:.2f}s\033[0m")

                # Save run notes (thread-safe via file system)
                # Pass current execution_data for concurrent mode since execution_history isn't updated yet
                self._save_run_notes(
                    capsule_name, goal, execution_time,
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

    async def _sab_files_transfer_isolated(self, sab_loader, file_transfer, row, task_id: str):
        """Transfer dataset files to isolated workspace with validation."""
        file_transfer.clean_workspace()
        task_dataset_path = sab_loader.get_dataset_path(row)
        self.logger.info(f"[Worker {task_id}] Transferring dataset from: {task_dataset_path}")
        print(f"\033[96m[Worker {task_id}] 📁 Transferring dataset: {task_dataset_path.name}\033[0m")
        files_transferred = file_transfer.transfer_files_to_workspace(str(task_dataset_path))
        await asyncio.sleep(0.3)  # Give filesystem a moment to sync
        workspace_files_after = file_transfer.count_files_recursive(Path(file_transfer.workspace_path))
        print(f"\033[96m[Worker {task_id}] ✓ Transferred {files_transferred} files to workspace\033[0m")

        if workspace_files_after == 0:
            raise ValueError(
                f"Files disappeared after transfer! "
                f"Transferred {files_transferred} but workspace now has 0 files."
            )
        self.logger.info(f"[Worker {task_id}] Successfully transferred {files_transferred} files")


    async def run_concurrent_eval_loop(
        self,
        dataset_type: str,
        dataset_path: str,
        learning: bool,
        single_agent_mode: bool = False,
        start_row: int | None = None,
        restore_cache: bool | None = None
    ) -> None:
        """
        Concurrent execution loop that processes multiple tasks in parallel.
        Uses asyncio.gather with semaphore-based concurrency control.

        Args:
            dataset_type: Type of dataset being evaluated
            dataset_path: Path to the CSV dataset file
            learning: Whether learning mode is enabled
            single_agent_mode: Whether to use single agent mode
            start_row: 0-based first CSV row to process. None = prompt the user
                (interactive mode only; queued CLI runs must pass a value so no
                stdin prompt happens after the queue launches).
            restore_cache: Whether to restore stats from previous run notes.
                None = prompt the user when a cache is found.
        """
        papers_csv_path = Path(dataset_path)

        # Get starting row (pre-resolved by the caller, or prompt interactively)
        if start_row is None:
            while True:
                user_input = await _prompt_with_default("Enter starting row", default="0")
                try:
                    start_row = max(0, int(user_input) - 1)
                    break
                except ValueError:
                    print(f"  ⚠️  Invalid value '{user_input}' – please enter a whole number.")
        self._start_row = start_row
        print(f"  → starting at row {start_row + 1}")

        # Load and restore from cache if available
        cached_notes = self._load_previous_run_notes()
        if cached_notes:
            if restore_cache is None:
                restore_input = await _prompt_with_default(
                    "Restore previous run statistics from cache? (y/n)", default="y"
                )
                restore_cache = restore_input.lower() != 'n'
            if restore_cache:
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
        self._send_email_report(status="completed")

    async def run_single_thread_eval_loop(self, dataset_type: str, dataset_path: str, learning: bool,
                                          single_agent_mode: bool = False,
                                          start_row: int | None = None,
                                          restore_cache: bool | None = None) -> None:
        """
        Main autonomous execution loop.
        Generates goals from CSV entries, executes them, analyzes results, and learns.

        Args:
            start_row: 0-based first CSV row; None = prompt interactively.
            restore_cache: Whether to restore previous run stats; None = prompt.
        """
        papers_csv_path = Path(dataset_path)

        # Get starting row (pre-resolved by the caller, or prompt interactively)
        if start_row is None:
            while True:
                user_input = await _prompt_with_default("Enter starting row", default="0")
                try:
                    start_row = max(0, int(user_input) - 1)
                    break
                except ValueError:
                    print(f"  ⚠️  Invalid value '{user_input}' – please enter a whole number.")
        self._start_row = start_row
        print_info(f"→ starting at row {start_row + 1}")

        # Load and restore from cache if available
        cached_notes = self._load_previous_run_notes()
        if cached_notes:
            if restore_cache is None:
                restore_input = await _prompt_with_default(
                    "Restore previous run statistics from cache? (y/n)", default="y"
                )
                restore_cache = restore_input.lower() != 'n'
            if restore_cache:
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
            print_phase("Evaluating on paper datasets...")
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
                        await self.sab_files_transfer(sab_loader, file_transfer, row)
                        runs = await self.evolve.start_workflow_evolution(goal=goal,
                                                        judge=True,
                                                        enable_evolution=learning,
                                                        scenario_rubric=None,
                                                        single_agent_mode=single_agent_mode
                                                       )
                    else:
                        _ = await self.planner.start_planner(goal=goal,
                                    judge=True,
                                    max_task_retry=3
                                   )
                    print_info("📦 Transferring results files…")
                    # Offloaded: the capsule namer is a blocking sync LLM call.
                    trs = LocalTransfer(config=self.config, workspace_path=self.config.workspace_dir, runs_capsule_dir=self.config.runs_capsule_dir)
                    task_id = self._extract_workspace_name_from_row(row)
                    capsule_name = await asyncio.to_thread(
                        trs.transfer_workspace_files_to_capsule, goal, task_token=task_id
                    )
                    print_info("📊 Analyzing results…")
                    execution_time = time.time() - iteration_start_time
                    execution_data = {
                        "iteration": i + 1,
                        "goal": goal,
                        "execution_time": execution_time,
                        "task_id": task_id,
                    }
                    if dataset_type == "science_agent_bench" and sab_loader:
                        # Offloaded: sandbox build, VER/SR subprocesses and CBS are blocking.
                        execution_data = await asyncio.to_thread(
                            self._evaluate_with_science_agent_bench,
                            capsule_name=capsule_name,
                            row=row,
                            runs=runs,
                            sab_loader=sab_loader,
                            execution_data=execution_data
                        )

                    self.execution_history.append(execution_data)
                    self._print_final_summary()
                    self._save_run_notes(
                        capsule_name, goal, execution_time
                    )

                    print_ok(f"Iteration {i + 1} completed")
                    print_info(f"  Time: {execution_time:.2f}s")
                except Exception as e:
                    self.logger.error(f"[PAPERS DATASET MODE] Error in csv row {i + 1}: {str(e)}")
                    print(f"\033[91m❌ Error in csv row {i + 1}: {str(e)}\033[0m")
                    self.execution_history.append({
                        "iteration": i + 1,
                        "goal": "Unknown",
                        "execution_time": 0,
                        "success_level": "Error",
                        "key_insight": str(e),
                    })
                    continue

        self._print_final_summary()
        self._send_email_report(status="completed")

    def _build_summary_rows(self) -> tuple[list[tuple[str, str]], list[dict], list[dict]]:
        """Build the rows used for both the printed summary and the email report."""
        # Filter out cached entries to count only actual runs from this session
        current_runs = [exec_data for exec_data in self.execution_history
                       if exec_data.get("success_level") != "Cached"]

        # Infra-excluded runs are neither pass nor fail — drop from denominators.
        excluded_runs = [d for d in current_runs if _is_excluded(d)]
        evaluable = [d for d in current_runs if not _is_excluded(d)]
        successful_runs = [d for d in evaluable
                          if d.get("success_level") == "Success"]

        success_rate = (
            f"{len(successful_runs)/len(evaluable)*100:.1f}%"
            if evaluable else "N/A"
        )
        rows: list[tuple[str, str]] = [
            ("Steps evaluated", str(len(evaluable))),
            ("Excluded (infra)", str(len(excluded_runs))),
            ("Successful runs", str(len(successful_runs))),
            ("Success rate", success_rate),
        ]

        sab_runs = [d for d in current_runs if d.get('VER') is not None]
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
        return rows, current_runs, sab_runs

    def _print_final_summary(self) -> None:
        """Print a summary of all autonomous executions."""
        rows, current_runs, sab_runs = self._build_summary_rows()

        # Recompute the values needed for the cli-notes side-effect below.
        evaluable = [d for d in current_runs if not _is_excluded(d)]
        successful_runs = [d for d in evaluable
                          if d.get("success_level") == "Success"]
        success_rate = (
            f"{len(successful_runs)/len(evaluable)*100:.1f}%"
            if evaluable else "N/A"
        )
        if sab_runs:
            ver_success = sum(1 for run in sab_runs if run.get('VER', False))
            sr_success = sum(1 for run in sab_runs if run.get('SR', False))
            avg_cbs = sum(run.get('CBS', 0.0) for run in sab_runs) / len(sab_runs)
            total_cost = sum(run.get('eval_cost', 0.0) for run in sab_runs)

        print_summary("📊 EVALUATION SUMMARY", rows)

        # If launched via EvaluationCLI, append final metrics to the run notes file.
        notes_path = getattr(self, "_evaluation_cli_notes_path", None)
        if notes_path and Path(notes_path).exists():
            try:
                with open(notes_path, "r", encoding="utf-8") as fh:
                    notes_data = json.load(fh)
                final = {
                    "steps_evaluated": len(current_runs),
                    "successful_runs": len(successful_runs),
                    "success_rate": success_rate,
                }
                if sab_runs:
                    final.update({
                        "ver_success": ver_success,
                        "ver_total": len(sab_runs),
                        "sr_success": sr_success,
                        "sr_total": len(sab_runs),
                        "avg_cbs": avg_cbs,
                        "total_cost": total_cost,
                    })
                notes_data["final_results"] = final
                notes_data["finished_at"] = datetime.now().isoformat()
                with open(notes_path, "w", encoding="utf-8") as fh:
                    json.dump(notes_data, fh, indent=2, ensure_ascii=False)
                    fh.write("\n")
            except Exception:
                pass  # best-effort

    def _build_config_rows(self) -> list[tuple[str, str]]:
        """Build the run-configuration rows shown at the top of the email."""
        mode = "single-agent" if self._single_agent_mode else "multi-agent"
        concurrency = (
            f"{self.max_concurrent_tasks} workers (stagger {self.task_start_delay:.1f}s)"
            if self._concurrent else "sequential"
        )
        return [
            ("Execution mode", mode),
            ("Learning", "enabled" if self._learning else "disabled"),
            ("Concurrency", concurrency),
            ("Dataset type", self._dataset_type or "unknown"),
            ("Dataset path", self._dataset_path or "unknown"),
            ("CSV runs limit", str(self.csv_runs_limit)),
            ("Start row", str(self._start_row + 1)),
            ("smolagent_model_id", getattr(self.config, "smolagent_model_id", "unknown")),
            ("orchestrator_choose_model", getattr(self.config, "orchestrator_choose_model", "unknown")),
            ("literrature_grounding", getattr(self.config, "literrature_grounding", "unknown")),
            ("selection_strategy", getattr(self.config, "selection_strategy", "unknown")),
            ("learned_score_threshold", getattr(self.config, "learned_score_threshold", "unknown")),
            ("max_learning_evolve_iterations", getattr(self.config, "max_learning_evolve_iterations", "unknown")),
            ("parent_threshold_similarity", getattr(self.config, "parent_threshold_similarity", "unknown")),
            ("crossover_rate", getattr(self.config, "crossover_rate", "unknown"))
        ]

    def _build_task_table(self) -> dict | None:
        """Build the per-task results table for the email. Returns None if empty."""
        current_runs = [
            exec_data for exec_data in self.execution_history
            if exec_data.get("success_level") != "Cached"
        ]
        if not current_runs:
            return None

        has_sab = any(d.get("VER") is not None for d in current_runs)
        headers = ["#", "Task", "Time (s)", "Success"]
        if has_sab:
            headers += ["VER", "SR", "CBS", "Cost ($)"]

        def mark(value) -> str:
            return "—" if value is None else ("✓" if value else "✗")

        rows: list[list[str]] = []
        for d in current_runs:
            task_label = d.get("task_id") or (d.get("goal", "") or "")[:40]
            row = [
                str(d.get("iteration", "?")),
                task_label,
                f"{d.get('execution_time', 0):.1f}",
                str(d.get("success_level", "?")),
            ]
            if has_sab:
                cbs = d.get("CBS")
                row += [
                    mark(d.get("VER")),
                    mark(d.get("SR")),
                    "—" if cbs is None else f"{cbs:.3f}",
                    f"{d.get('eval_cost', 0.0):.4f}",
                ]
            rows.append(row)
        return {"headers": headers, "rows": rows}

    def _send_email_report(self, status: str = "completed") -> None:
        """Send the final summary by email (no-op if email env vars are unset)."""
        try:
            rows, current_runs, _ = self._build_summary_rows()
            config_rows = self._build_config_rows()
            task_table = self._build_task_table()
            model = getattr(self.config, "smolagent_model_id", "unknown")
            subject = f"[Mimosa] Evaluation {status} — {len(current_runs)} runs ({model})"
            body_prefix = (
                f"Mimosa-AI evaluation {status} at "
                f"{datetime.now().isoformat(timespec='seconds')}."
            )
            send_evaluation_report(
                subject=subject,
                rows=rows,
                body_prefix=body_prefix,
                config_rows=config_rows,
                task_table=task_table,
            )
        except Exception as e:
            self.logger.warning(f"[EMAIL] Skipped email report due to error: {e}")

    async def start_evaluation(
        self,
        dataset_type: str = "default",
        dataset_path: str = "datasets/our_benchmark.csv",
        learning: bool = False,
        single_agent_mode: bool = False,
        concurrent: bool = False,
        start_row: int | None = None,
        restore_cache: bool | None = None
    ) -> None:
        """
        Public method to start the evaluation mode.

        Args:
            dataset_type: Type of dataset ("default" or "science_agent_bench")
            dataset_path: Path to the CSV dataset file
            learning: Whether to enable learning mode
            single_agent_mode: Whether to use single agent mode
            concurrent: Whether to run tasks concurrently (uses max_concurrent_tasks from init)
            start_row: 0-based first CSV row to process; None = prompt interactively
            restore_cache: Whether to restore previous run stats; None = prompt when a cache is found
        """
        # Snapshot the run-level args so the email report can describe what ran.
        self._dataset_type = dataset_type
        self._dataset_path = dataset_path
        self._learning = learning
        self._single_agent_mode = single_agent_mode
        self._concurrent = concurrent and self.max_concurrent_tasks > 1

        try:
            if concurrent and self.max_concurrent_tasks > 1:
                print(f"\033[95mStarting CONCURRENT evaluation with {self.max_concurrent_tasks} workers\033[0m")
                await self.run_concurrent_eval_loop(dataset_type, dataset_path, learning, single_agent_mode,
                                                    start_row=start_row, restore_cache=restore_cache)
            else:
                if concurrent and self.max_concurrent_tasks <= 1:
                    print("\033[93m⚠️ Concurrent mode requested but max_concurrent_tasks <= 1, falling back to sequential\033[0m")
                await self.run_single_thread_eval_loop(dataset_type, dataset_path, learning, single_agent_mode,
                                                       start_row=start_row, restore_cache=restore_cache)
        except KeyboardInterrupt:
            print_warn("Autonomous mode interrupted by user")
            self._print_final_summary()
            self._send_email_report(status="interrupted")
        except Exception as e:
            self.logger.error(f"[PAPERS DATASET MODE] Fatal error: {str(e)}")
            print_err(f"Fatal error in autonomous mode: {str(e)}")
            self._send_email_report(status="failed")
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

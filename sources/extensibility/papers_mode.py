"""
PaperEvaluationMode - Autonomous goal generation and execution system
"""

import asyncio
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from sources.core.dgm import GodelMachine
from sources.core.llm_provider import LLMConfig, LLMProvider
from sources.core.workflow_info import WorkflowInfo
from sources.core.planner import Planner
from sources.core.schema import Task, GodelRun
from sources.post_processing.bs_detection import BullshitDetectorNumerical
from sources.utils.science_agent_bench import ScienceAgentBenchLoader
from sources.utils.transfer_toolomics import LocalTransfer
from sources.utils.mock_data import MockDataGenerator

class PaperEvaluationMode:
    """
    Autonomous mode that automatically run Mimosa on various goal's defined in a CSV datasets, such a list of paper to replicate.
    """

    def __init__(self, config, csv_runs_limit: int = 100):
        """
        Initialize PaperEvaluationMode.

        Args:
            config: Mimosa configuration object
            csv_runs_limit: Maximum number of autonomous iterations
        """
        self.config = config
        self.csv_runs_limit = csv_runs_limit
        self.dgm = GodelMachine(config)
        self.planner = Planner(config)
        self.run_notes_dir = Path("run_notes")
        self.run_notes_dir.mkdir(exist_ok=True)
        self.done_rows = []

        model_name = "deepseek/deepseek-chat"  # judge 
        provider, model = model_name.split("/", 1) if "/" in model_name else ("openai", model_name)

        self.llm_config = LLMConfig(
            model=model,
            provider=provider,
            temperature=0.8  # Some creativity for goal generation
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

    def _save_run_notes(self, capsule_name: str, goal: str,
                       analysis: dict, execution_time: float) -> None:
        """Save detailed notes about the run."""
        timestamp = datetime.now().isoformat()
        notes = {
            "timestamp": timestamp,
            "goal": goal,
            "execution_time_seconds": execution_time,
            "analysis": analysis["full_analysis"],
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
            f"  UUID: {task.dgm_runs[-1].current_uuid}\n"
            f"  Description: {task.description}\n"
            f"  Agent Chain: {' -> '.join(task.final_answers)}"
            for task in tasks_data
        )

    def _format_task_mode_results(self, run: GodelRun) -> str:
        state_result = run.state_result
        if isinstance(state_result["answers"], list) and "step_name" in state_result:
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
        
        # Transfer files and validate
        files_transferred = file_transfer.transfer_files_to_workspace(str(task_dataset_path))
        
        print(f"\033[95m✓ Transferred {files_transferred} files to workspace\033[0m")
        self.logger.info(f"[PAPERS DATASET MODE] Successfully transferred {files_transferred} files")

    async def run_autonomous_eval_loop(self, dataset_type: str, dataset_path: str) -> None:
        """
        Main autonomous execution loop.
        Generates goals, executes them, analyzes results, and learns.
        """
        papers_csv_path = Path(dataset_path)
        user_input = input("Enter starting row ([Enter] 0 by default): ")
        start_row = int(user_input)-1 if user_input.strip() else -1
        
        # Initialize ScienceAgentBench loader if needed
        sab_loader = None
        if dataset_type == "science_agent_bench":
            sab_loader = ScienceAgentBenchLoader()
            self.logger.info("[PAPERS DATASET MODE] ScienceAgentBench mode activated")
        # Initialize file transfer utility
        file_transfer = LocalTransfer(
            workspace_path=self.config.workspace_dir,
            runs_capsule_dir=self.config.runs_capsule_dir
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
                    continue
                if i > self.csv_runs_limit:
                    break
                try:
                    iteration_start_time = time.time()
                    goal = self._generate_next_task(row, dataset_type)
                    print(f"\033[95m📋 GOAL: {goal}\033[0m")

                    if dataset_type == "science_agent_bench" and sab_loader:
                        self.sab_files_transfer(sab_loader, file_transfer, row)
                        runs = await self.dgm.start_dgm(goal=goal, judge=True)
                        results_str = self._format_task_mode_results(runs[-1])
                    else:
                        tasks_data = await self.planner.start_planner(goal=goal,
                                    judge=False,
                                    max_dgm_iteration=1,
                                    max_task_retry=2
                                   )
                        results_str = self._format_goal_mode_results(tasks_data)
                    print("\033[95m📊 Transfering results files...\033[0m")
                    trs = LocalTransfer(workspace_path=self.config.workspace_dir, runs_capsule_dir=self.config.runs_capsule_dir)
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
                    self.execution_history.append(execution_data)
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
                    raise e

        self._print_final_summary()

    def _print_final_summary(self) -> None:
        """Print a summary of all autonomous executions."""
        print(f"\n\033[95m{'🏁 AUTONOMOUS MODE COMPLETED':^80}\033[0m")
        print(f"\033[95m{'=' * 80}\033[0m")

        successful_runs = [exec_data for exec_data in self.execution_history
                          if exec_data.get("success_level") in ["High", "Medium"]]
        print(f"\033[95mSuccessful runs: {len(successful_runs)}\033[0m")
        print(f"\033[95mSuccess rate: {len(successful_runs)/len(self.execution_history)*100:.1f}%\033[0m")
        print(f"\033[95mNotes saved in: {self.run_notes_dir}\033[0m")
        print(f"\033[95m{'=' * 80}\033[0m\n")

    async def start_paper_eval_mode(self, dataset_type: str = "default", dataset_path = "datasets/our_benchmark.csv") -> None:
        """Public method to start the autonomous mode."""
        try:
            await self.run_autonomous_eval_loop(dataset_type, dataset_path)
        except KeyboardInterrupt:
            print("\n\033[95m⚠️ Autonomous mode interrupted by user\033[0m")
            self._print_final_summary()
        except Exception as e:
            self.logger.error(f"[PAPERS DATASET MODE] Fatal error: {str(e)}")
            print(f"\033[91m❌ Fatal error in autonomous mode: {str(e)}\033[0m")
            raise

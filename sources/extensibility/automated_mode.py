"""
AutomatedMode - Autonomous goal generation and execution system
"""

import asyncio
import csv
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

from sources.core.dgm import GodelMachine
from sources.core.llm_provider import LLMConfig, LLMProvider
from sources.core.workflow_info import WorkflowInfo
from sources.core.planner import Planner
from sources.core.schema import Task

class AutomatedMode:
    """
    Autonomous mode that uses LLM to generate goals, execute them via DGM,
    and learn from results to generate progressively more challenging goals.
    """

    def __init__(self, config, max_iterations: int = 10):
        """
        Initialize AutomatedMode.
        
        Args:
            config: Mimosa configuration object
            max_iterations: Maximum number of autonomous iterations
        """
        self.config = config
        self.max_iterations = max_iterations
        self.dgm = GodelMachine(config)
        self.planner = Planner(config)
        self.run_notes_dir = Path("run_notes")
        self.run_notes_dir.mkdir(exist_ok=True)
        
        model_name = "deepseek/deepseek-chat"  # Use OpenRouter format: provider/model
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
2. STRENGTHS: Key positive aspects
3. WEAKNESSES: Areas that need improvement
"""

    def _save_run_notes(self, iteration: int, goal: str, 
                       analysis: str, execution_time: float) -> None:
        """Save detailed notes about the run."""
        timestamp = datetime.now().isoformat()
        notes = {
            "iteration": iteration,
            "timestamp": timestamp,
            "goal": goal,
            "execution_time_seconds": execution_time,
            "analysis": analysis
        }
        
        notes_file = self.run_notes_dir / f"run_{iteration:03d}.json"
        notes_file.parent.mkdir(parents=True, exist_ok=True)
        with open(notes_file, 'w', encoding='utf-8') as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"[AUTOMATED MODE] Run notes saved to {notes_file}")

    def _extract_wf_answers(self, workflow_info: WorkflowInfo) -> str:
        """Extract and format flow answers from workflow state result."""
        state_result = workflow_info.load_state_result()
        
        if not state_result or "answers" not in state_result:
            return "No answers found in workflow execution."
        
        # Format answers as specified in the requirements
        if isinstance(state_result["answers"], list) and "step_name" in state_result:
            wf_answers = "\n".join(
                f"agent {n}: {x}" 
                for (n, x) in zip(state_result["step_name"], state_result["answers"], strict=True)
            )
        else:
            wf_answers = str(state_result["answers"])
        
        return wf_answers
    
    def _get_random_paper_title(self, start_row=0) -> tuple[str, str, str]:
        """Get a random paper title from the papers.csv file."""
        papers_csv_path = Path("datasets/our_benchmark.csv")

        try:
            with open(papers_csv_path, encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                total_rows = sum(1 for _ in reader)
                csvfile.seek(0)
                reader = csv.DictReader(csvfile)
                random_row = random.randint(0, total_rows - 1) if start_row == -1 else start_row
                for i, row in enumerate(reader):
                    if i == random_row:
                        return row['Title'].strip(), row['URLS'].strip(), row['Prompt'].strip()

        except Exception as e:
            self.logger.warning(f"[AUTOMATED MODE] Could not read random paper from CSV: {str(e)}")
            raise e
        raise Exception("Run out of papers or failed to open CSV.")

    def _generate_next_task(self, iteration: int, start_row: int = -1) -> str:
        """Generate the next goal using LLM based on a random paper from the CSV."""
        paper_title, url, prompt = self._get_random_paper_title(start_row)
        if prompt == "":
            prompt = "Reproduce the experiments from the paper and compare the result."

        goal = f"""
    Paper title: {paper_title}
    Url to paper: {url}
    Goal to achieve: {prompt}
        """
        return goal.strip()

    def _format_task_results(self, tasks_data: Task) -> str:
        """Format task results for analysis."""
        return '\n\n'.join(
            f"Task {task.name}:\n"
            f"  UUID: {task.dgm_runs[-1].current_uuid}\n"
            f"  Description: {task.description}\n"
            f"  Agent Chain: {' -> '.join(task.final_answers)}"
            for task in tasks_data
        )

    def _analyze_results(self, goal: str, tasks_data: list[Task], execution_time: float) -> dict[str, str]:
        """Analyze execution results using LLM."""
        results_str = self._format_task_results(tasks_data)
        prompt = f"""Analyze the following Mimosa-AI execution:
TASK: {goal}
EXECUTION TIME: {execution_time:.2f} seconds
EXECUTION RESULTS:
{results_str}
Provide your analysis following the specified output format."""

        analysis_text = self.result_analyzer(prompt)
        analysis = {
            "full_analysis": analysis_text,
            "success_level": "Medium",  # Default
            "key_insight": "Analysis completed"
        }
        lines = analysis_text.split('\n')
        for line in lines:
            if line.startswith("SUCCESS_LEVEL:"):
                analysis["success_level"] = line.split(":", 1)[1].strip()
            elif line.startswith("NEXT_CHALLENGE:"):
                analysis["key_insight"] = line.split(":", 1)[1].strip()
        return analysis

    async def run_autonomous_loop(self) -> None:
        """
        Main autonomous execution loop.
        Generates goals, executes them, analyzes results, and learns.
        """
        self.logger.info(f"[AUTOMATED MODE] Starting autonomous loop for {self.max_iterations} iterations")
        
        print(f"\n\033[95m{'🤖 AUTONOMOUS MODE':^80}\033[0m")
        print(f"\033[95m{'=' * 80}\033[0m")
        print(f"\033[95mRunning {self.max_iterations} autonomous iterations\033[0m")
        print(f"\033[95mNotes will be saved to: {self.run_notes_dir}\033[0m")
        print(f"\033[95m{'=' * 80}\033[0m\n")

        user_input = input("Enter starting row ([Enter] for random): ")
        start_row = int(user_input)-1 if user_input.strip() else -1

        for iteration in range(self.max_iterations):
            try:
                iteration_start_time = time.time()
                print(f"\n\033[95m{'─' * 60}\033[0m")
                print(f"\033[95mAUTONOMOUS ITERATION {iteration + 1}/{self.max_iterations}\033[0m")
                print(f"\033[95m{'─' * 60}\033[0m")
                # Generate next goal
                goal = self._generate_next_task(iteration, start_row)
                print(f"\033[95m📋 Task: {goal}\033[0m")
                # Execute goal via planner
                tasks_data = await self.planner.start_planner(goal=goal, 
                            judge=False,
                            max_dgm_iteration=1,
                            max_task_retry=3
                           )
                print("automated_mode: catch dgm_runs:")
                # Load and analyze results
                print("\033[95m📊 Analyzing results...\033[0m")

                execution_time = time.time() - iteration_start_time
                analysis = self._analyze_results(goal, tasks_data, execution_time)
                # Save execution data
                execution_data = {
                    "iteration": iteration + 1,
                    "goal": goal,
                    "execution_time": execution_time,
                    "success_level": analysis.get("success_level", "Unknown"),
                    "key_insight": analysis.get("full_analysis", "Unknown")
                }
                self.execution_history.append(execution_data)
                
                # Save detailed notes
                self._save_run_notes(
                    iteration + 1, goal, 
                    analysis["full_analysis"], execution_time
                )
                print(f"\033[95m✅ Iteration {iteration + 1} completed\033[0m")
                print(f"\033[95m   Success Level: {analysis.get('success_level', 'Unknown')}\033[0m")
                print(f"\033[95m   Time: {execution_time:.2f}s\033[0m")
                # Brief pause between iterations
                if iteration < self.max_iterations - 1:
                    print("\033[95m⏸️  Pausing before next iteration...\033[0m")
                    await asyncio.sleep(2)
            except Exception as e:
                self.logger.error(f"[AUTOMATED MODE] Error in iteration {iteration + 1}: {str(e)}")
                print(f"\033[91m❌ Error in iteration {iteration + 1}: {str(e)}\033[0m")
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

    async def start_autonomous_mode(self) -> None:
        """Public method to start the autonomous mode."""
        try:
            await self.run_autonomous_loop()
        except KeyboardInterrupt:
            print("\n\033[95m⚠️ Autonomous mode interrupted by user\033[0m")
            self._print_final_summary()
        except Exception as e:
            self.logger.error(f"[AUTOMATED MODE] Fatal error: {str(e)}")
            print(f"\033[91m❌ Fatal error in autonomous mode: {str(e)}\033[0m")
            raise

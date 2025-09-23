"""
AutomatedMode - Autonomous task generation and execution system
"""

import asyncio
import csv
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from sources.core.dgm import GodelMachine
from sources.core.llm_provider import LLMConfig, LLMProvider
from sources.core.workflow_info import WorkflowInfo
from sources.post_processing.bs_detection import BullshitDetector

class AutomatedMode:
    """
    Autonomous mode that uses LLM to generate tasks, execute them via DGM,
    and learn from results to generate progressively more challenging tasks.
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
        self.bs_detector = BullshitDetector()
        self.run_notes_dir = Path("run_notes")
        self.run_notes_dir.mkdir(exist_ok=True)
        
        model_name = "openai/gpt-4o"  # Use OpenRouter format: provider/model
        provider, model = model_name.split("/", 1) if "/" in model_name else ("openai", model_name)
        
        self.llm_config = LLMConfig(
            model=model,
            provider=provider,
            temperature=0.8  # Some creativity for task generation
        )
        self.task_generator = LLMProvider(
            agent_name="task_generator",
            memory_path=None,
            system_msg=self._get_task_generator_system_prompt(),
            config=self.llm_config
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

    def _get_task_generator_system_prompt(self) -> str:
        """System prompt for the task generation LLM."""
        return """You are an autonomous AI scientist task generator for Mimosa-AI, a framework for scientific research automation.

Your role is to generate scientific tasks that test Mimosa-AI capabilities.

GUIDELINES:
1. Generate tasks that are scientific, research-oriented, and realistic
2. Start with simpler tasks and gradually increase complexity based on previous results
3. Focus on areas like: literature review, data analysis, software installation, paper reproduction
4. Always assume empty workfolder at start of each task, no pre-existing files or data, you must specify any data download or code installation required in the task description

RULES:

- Avoid any task that would require a GUI based software. GUI is not supported.
- Avoid task that would require excessive computational resources (e.g. training large models)
- Avoid tasks that are too vague or broad. Be specific and focused.
- Avoid tasks that require installation of heavy software with complex dependencies.
- Avoid tasks that would require sudo or admin access to install software.

INPUT FORMAT:
You will be given information about a research papers to challenge Mimosa on.

OUTPUT FORMAT:
Provide only the task description as a clear, specific goal statement. Do not include explanations or metadata.
"""

    def _get_result_analyzer_system_prompt(self) -> str:
        """System prompt for the result analysis LLM."""
        return """You are an autonomous AI scientist result analyzer for Mimosa-AI.

Mimosa-AI is a multi-agent system designed to autonomously conduct scientific tasks.

Your role is to analyze workflow execution results and provide insights for the next task generation.
You must be strict and harsh in your analysis.

ANALYSIS FOCUS:
1. Assess task completion quality and success level
2. Identify strengths and weaknesses in the execution
3. Note any errors, limitations, or areas for improvement
4. Suggest areas where Mimosa-AI could be challenged further
5. Recommend complexity adjustments for future tasks

EVALUATION CRITERIA:
- Task completion: Was the full goal achieved? An incomplete task should be considered as failed.
- Quality: How well was the task executed?
- Scalability: Could this approach work for similar tasks?

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
4. NEXT_CHALLENGE: Suggested direction for next task
5. COMPLEXITY_ADJUSTMENT: (Increase/Maintain/Decrease)"""

    def _save_run_notes(self, iteration: int, task: str, uuid: str, 
                       flow_answers: str, analysis: str, fraud: str, execution_time: float) -> None:
        """Save detailed notes about the run."""
        timestamp = datetime.now().isoformat()
        notes = {
            "iteration": iteration,
            "timestamp": timestamp,
            "task": task,
            "workflow_uuid": uuid,
            "execution_time_seconds": execution_time,
            "flow_answers": flow_answers,
            "analysis": analysis,
            "potential_fraud": fraud,
            "workflow_dir": str(Path(self.config.workflow_dir) / uuid)
        }
        
        notes_file = self.run_notes_dir / f"run_{iteration:03d}_{uuid}.json"
        with open(notes_file, 'w') as f:
            json.dump(notes, f, indent=2)
        
        self.logger.info(f"[AUTOMATED MODE] Run notes saved to {notes_file}")

    def _extract_flow_answers(self, workflow_info: WorkflowInfo) -> str:
        """Extract and format flow answers from workflow state result."""
        state_result = workflow_info.load_state_result()
        
        if not state_result or "answers" not in state_result:
            return "No answers found in workflow execution."
        
        # Format answers as specified in the requirements
        if isinstance(state_result["answers"], list) and "step_name" in state_result:
            flow_answers = "\n".join(
                f"agent {n}: {x}" 
                for (n, x) in zip(state_result["step_name"], state_result["answers"], strict=True)
            )
        else:
            flow_answers = str(state_result["answers"])
        
        return flow_answers
    
    def _get_random_paper_title(self) -> str:
        """Get a random paper title from the papers.csv file."""
        papers_csv_path = Path("sources/extensibility/data/paper_bench.csv")
        
        try:
            with open(papers_csv_path, encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                total_rows = sum(1 for _ in reader)
                csvfile.seek(0)
                reader = csv.DictReader(csvfile)
                random_row = random.randint(0, total_rows - 1)
                for i, row in enumerate(reader):
                    if i == random_row:
                        return row['Title'].strip()
                        
        except Exception as e:
            self.logger.warning(f"[AUTOMATED MODE] Could not read random paper from CSV: {str(e)}")
            raise e
        raise Exception("Run out of papers or failed to open CSV.")
    
    def _generate_next_task(self, iteration: int) -> str:
        """Generate the next task using LLM based on a random paper from the CSV."""
        # Get a random paper title from the CSV
        paper_title = self._get_random_paper_title()
        
        prompt = f"""
Given this paper: "{paper_title}"

Write an efficient goal description for an AI that will attempt reproduction of the paper experiments.

For the paper, write a efficient, detailled goal to attempt a full reproduction of the papers experiments.

Assume the data is not available locally, you must specify any data download or code installation required unless previous execution summary confirms data or files is already downloaded

Example: Reproduce the computational analysis from 'Decoding the constraints acting on a coastal fish using landscape transcriptomics'. Focus on the RNA extraction, library preparation, sequencing, and data processing methodology. Find the available resources online (dataset and code). Primary objective: Execute the computational pipeline and validate reported results against paper claims. Flag any methodological gaps or result discrepancies.

Always specify the FULL paper title ({paper_title}) in the task.
Task:"""

        task = self.task_generator(prompt)
        self.logger.info(f"[AUTOMATED MODE] Generated task {iteration + 1} based on paper '{paper_title}': {task[:256]}...")
        return task.strip()

    def _analyze_results(self, task: str, flow_answers: str, execution_time: float) -> dict[str, str]:
        """Analyze execution results using LLM."""
        prompt = f"""Analyze the following Mimosa-AI workflow execution:
TASK: {task}
EXECUTION TIME: {execution_time:.2f} seconds
WORKFLOW EXECUTION RESULTS:
{flow_answers}
Provide your analysis following the specified output format."""

        analysis_text = self.result_analyzer(prompt)
        # Parse the analysis into structured format
        analysis = {
            "full_analysis": analysis_text,
            "success_level": "Medium",  # Default
            "key_insight": "Analysis completed"
        }
        # Extract key metrics from analysis text
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
        Generates tasks, executes them, analyzes results, and learns.
        """
        self.logger.info(f"[AUTOMATED MODE] Starting autonomous loop for {self.max_iterations} iterations")
        
        print(f"\n\033[95m{'🤖 AUTONOMOUS MODE ACTIVATED':^80}\033[0m")
        print(f"\033[95m{'=' * 80}\033[0m")
        print(f"\033[95mRunning {self.max_iterations} autonomous iterations\033[0m")
        print(f"\033[95mNotes will be saved to: {self.run_notes_dir}\033[0m")
        print(f"\033[95m{'=' * 80}\033[0m\n")

        for iteration in range(self.max_iterations):
            try:
                iteration_start_time = time.time()
                
                print(f"\n\033[95m{'─' * 60}\033[0m")
                print(f"\033[95mAUTONOMOUS ITERATION {iteration + 1}/{self.max_iterations}\033[0m")
                print(f"\033[95m{'─' * 60}\033[0m")
                
                # Generate next task
                print("\033[95m🎯 Generating next task...\033[0m")
                task = self._generate_next_task(iteration)
                print(f"\033[95m📋 Task: {task}\033[0m")
                
                # Execute task via DGM
                print("\033[95m🚀 Executing task via DGM...\033[0m")
                dgm_runs = await self.dgm.start_dgm(
                    goal=task,
                    judge=True,  # Enable evaluation
                    human_validation=False,
                    max_iteration=1  # Allow some self-improvement
                )
                print("automated_mode: catch dgm_runs:")
                for r in dgm_runs:
                    print(r)
                uuid = dgm_runs[-1].current_uuid
                if not uuid:
                    print("Failed to get uuid of DGM run. Press [Enter] to continue\n")
                    input()
                
                # Load and analyze results
                print("\033[95m📊 Analyzing results...\033[0m")
                workflow_info = WorkflowInfo(uuid, Path(self.config.workflow_dir) / uuid)
                flow_answers = self._extract_flow_answers(workflow_info)
                
                execution_time = time.time() - iteration_start_time
                analysis = self._analyze_results(task, flow_answers, execution_time)
                fraud_detection = self.bs_detector.analyze_all_agents(uuid)
                
                # Save execution data
                execution_data = {
                    "iteration": iteration + 1,
                    "task": task,
                    "uuid": uuid,
                    "execution_time": execution_time,
                    "success_level": analysis.get("success_level", "Unknown"),
                    "key_insight": analysis.get("full_analysis", "Unknown")
                }
                self.execution_history.append(execution_data)
                
                # Save detailed notes
                self._save_run_notes(
                    iteration + 1, task, uuid, flow_answers, 
                    analysis["full_analysis"], fraud_detection, execution_time
                )
                
                print(f"\033[95m✅ Iteration {iteration + 1} completed\033[0m")
                print(f"\033[95m   UUID: {uuid}\033[0m")
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
                
                # Save error information
                error_data = {
                    "iteration": iteration + 1,
                    "task": task if 'task' in locals() else "Task generation failed",
                    "error": str(e),
                    "execution_time": time.time() - iteration_start_time if 'iteration_start_time' in locals() else 0
                }
                self.execution_history.append(error_data)
                
                # Continue with next iteration
                continue
        
        # Final summary
        self._print_final_summary()

    def _print_final_summary(self) -> None:
        """Print a summary of all autonomous executions."""
        print(f"\n\033[95m{'🏁 AUTONOMOUS MODE COMPLETED':^80}\033[0m")
        print(f"\033[95m{'=' * 80}\033[0m")
        
        successful_runs = [exec_data for exec_data in self.execution_history 
                          if exec_data.get("success_level") in ["High", "Medium"]]
        
        print(f"\033[95mTotal iterations: {len(self.execution_history)}\033[0m")
        print(f"\033[95mSuccessful runs: {len(successful_runs)}\033[0m")
        print(f"\033[95mSuccess rate: {len(successful_runs)/len(self.execution_history)*100:.1f}%\033[0m")
        
        total_time = sum(exec_data.get("execution_time", 0) for exec_data in self.execution_history)
        print(f"\033[95mTotal execution time: {total_time:.2f}s\033[0m")
        print(f"\033[95mNotes saved in: {self.run_notes_dir}\033[0m")
        
        print(f"\n\033[95mTop performing tasks:\033[0m")
        high_success = [exec_data for exec_data in self.execution_history 
                       if exec_data.get("success_level") == "High"]
        
        for i, exec_data in enumerate(high_success[:3], 1):
            print(f"\033[95m{i}. {exec_data['task'][:60]}...\033[0m")
            print(f"\033[95m   UUID: {exec_data.get('uuid', 'N/A')}\033[0m")
        
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

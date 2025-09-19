"""
AutomatedMode - Autonomous task generation and execution system
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from sources.core.dgm import GodelMachine
from sources.core.llm_provider import LLMConfig, LLMProvider
from sources.core.workflow_info import WorkflowInfo


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
        self.run_notes_dir = Path("run_notes")
        self.run_notes_dir.mkdir(exist_ok=True)
        
        # Initialize LLM for task generation and analysis
        # Extract provider and model from OpenRouter format
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
        self.execution_history: List[Dict] = []
        self.logger = logging.getLogger(__name__)

    def _get_task_generator_system_prompt(self) -> str:
        """System prompt for the task generation LLM."""
        return """You are an autonomous AI scientist task generator for Mimosa-AI, a framework for scientific research automation.

Your role is to generate scientific tasks that test Mimosa-AI capabilities.

GUIDELINES:
1. Generate tasks that are scientific, research-oriented, and realistic
2. Start with simpler tasks and gradually increase complexity based on previous results
3. Focus on areas like: literature review, data analysis, software installation, paper reproduction
4. Consider the execution history to avoid repetition and build upon previous successes/failures
5. Tasks should be specific, actionable, and measurable
6. Always assume empty workfolder at start of each task, no pre-existing files or data, you must specify any data download or code installation required in the task description

TASK CATEGORIES TO EXPLORE:
- Literature search and analysis
- Software installation and setup
- Data processing and visualization  
- Statistical analysis
- Paper reproduction attempts
- Multi-modal data analysis

GOOD EXAMPLE:

- https://github.com/tanjeffreyz/deep-residual-learning is a github for a deep learning project, you must install it and review the code, suggest improvement to the algorithm and keep editing the code until you get a test error below 9%
- find the github for the research paper titled MOGONET integrates multi-omics data using graph convolutional networks allowing patient classification and biomarker identification, download the source code, set it up by installing any requirements, then proceed to identify the main code entry point and run a full training run
- find information about the CNRS holobiomics lab and try to run and install their most stared project
- download the data needed to redo the analysis of this scientific paper https://www.nature.com/articles/s41467-021-23774-w. Use the source code provided to find the data (github link).
- Reproduce key results from XOmiVAE (https://academic.oup.com/bib/article/22/6/bbab315/6353242): open the paper in the browser, download the PDF, extract hyperparameters, retrieve and run the official code, prepare an omics dataset, train/evaluate the model, compute gene- and latent-dimension attributions (Deep SHAP), explain VAE-derived clusters (Welch’s t-test on latent dimensions), and produce a reproducibility report mapping our outputs to the paper’s claims.
- Search for the paper "Simulating Metabolic Pathways to Enhance Interpretations of MGWAS Results," identify all required software and tools needed for the reproduction of the experiments described in the paper, download and install them in the work environment, and document the setup process in detail for future reference.

RULES:

- Avoid any task that would require a GUI based software. GUI is not supported.
- Avoid task that would require excessive computational resources (e.g. training large models)
- Avoid tasks that are too vague or broad. Be specific and focused.
- Avoid tasks that require installation of heavy software with complex dependencies.
- Avoid tasks that would require sudo or admin access to install software.

OUTPUT FORMAT:
Provide only the task description as a clear, specific goal statement. Do not include explanations or metadata.

Please prioritize tasks that use one of these papers :
- XOmiVAE paper: https://academic.oup.com/bib/article/22/6/bbab315/6353242
- Simulating Metabolic Pathways to Enhance Interpretations of MGWAS Results
- "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" by Sergey Ioffe, Christian Szegedy

Always provide the full paper name or github link in your task description if you are using one of the above papers or github repositories as reference.
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
                       flow_answers: str, analysis: str, execution_time: float) -> None:
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
            "workflow_dir": str(Path(self.config.workflow_dir) / uuid)
        }
        
        notes_file = self.run_notes_dir / f"run_{iteration:03d}_{uuid}.json"
        with open(notes_file, 'w') as f:
            json.dump(notes, f, indent=2)
        
        self.logger.info(f"[AUTOMATED MODE] Run notes saved to {notes_file}")

    def _get_execution_history_summary(self) -> str:
        """Create a summary of previous executions for context."""
        if not self.execution_history:
            return "No previous executions."
        
        summary_parts = ["EXECUTION HISTORY SUMMARY:"]
        for i, exec_data in enumerate(self.execution_history[-5:], 1):  # Last 5 executions
            summary_parts.append(f"\n{i}. Task: {exec_data['task'][:256]}...")
            summary_parts.append(f"   Result: {exec_data.get('success_level', 'Unknown')}")
            summary_parts.append(f"   Key insight: {exec_data.get('key_insight', 'N/A')}")
        
        return "\n".join(summary_parts)

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

    def _generate_next_task(self, iteration: int) -> str:
        """Generate the next task using LLM."""
        history_summary = self._get_execution_history_summary()
        
        prompt = f"""Generate the next scientific research task for Mimosa-AI.

ITERATION: {iteration + 1}/{self.max_iterations}

{history_summary}

Based on the execution history above, generate a new task that:
1. Builds upon previous learnings
2. Explores new capabilities or challenges existing ones
3. Is appropriate for the current difficulty level
4. Focuses on scientific research automation
5. Assume the data is not available locally, you must specify any data download or code installation required unless previous execution summary confirms data or files is already downloaded

If a task appear to be impossible or too difficult for Mimosa, give up and suggest a totally different task.

Task:"""

        task = self.task_generator(prompt)
        self.logger.info(f"[AUTOMATED MODE] Generated task {iteration + 1}: {task[:256]}...")
        return task.strip()

    def _analyze_results(self, task: str, flow_answers: str, execution_time: float) -> Dict[str, str]:
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
                final_dgm_info = dgm_runs[-1]
                uuid = final_dgm_info.current_uuid
                
                # Load and analyze results
                print("\033[95m📊 Analyzing results...\033[0m")
                workflow_info = WorkflowInfo(uuid, Path(self.config.workflow_dir) / uuid)
                flow_answers = self._extract_flow_answers(workflow_info)
                
                execution_time = time.time() - iteration_start_time
                analysis = self._analyze_results(task, flow_answers, execution_time)
                
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
                    analysis["full_analysis"], execution_time
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

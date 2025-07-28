#!/usr/bin/env python3
"""
Mimosa - A AI Agent Framework for advancing scientific research
============================================================================
"""

import argparse
import asyncio
import csv
import datetime
import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import dotenv
import requests
from fastmcp import Client

from config import Config
from sources.core.dgm import GodelMachine
from sources.core.planner import Planner
from sources.core.parallel_testing import ParallelTesting
import shutil

dotenv.load_dotenv()


def validate_environment() -> None:
    """Validate required environment configuration."""
    if not os.getenv("HF_TOKEN"):
        raise ValueError(
            "⚠️ HF_TOKEN environment variable is not set. Please set it to your Hugging Face token."
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "⚠️ OPENAI_API_KEY environment variable is not set. Please set it to your OpenAI API key."
        )

def add_config_arguments(parser: argparse.ArgumentParser, config: Config) -> None:
    """Add CLI arguments for config parameters that can be overridden."""
    parser.add_argument("--workflow_dir", type=str, help="Override workflow directory path")
    parser.add_argument("--schema_code_path", type=str, help="Override state schema file path")
    parser.add_argument("--smolagent_factory_code_path", type=str, help="Override SmolAgent factory file path")
    parser.add_argument("--prompt_workflow_creator", type=str, help="Override system prompt file path")
    parser.add_argument("--workflow_llm_provider", type=str, help="Override LLM provider for workflows")
    parser.add_argument("--mcp_health_endpoint", type=str, help="Override MCP health endpoint URL")
    parser.add_argument("--runner_default_python_version", type=str, help="Override default Python version for runners")
    parser.add_argument("--runner_default_timeout", type=int, help="Override default timeout for runners (seconds)")
    parser.add_argument("--runner_default_max_memory_mb", type=int, help="Override default max memory for runners (MB)")
    parser.add_argument("--runner_default_max_cpu_percent", type=int, help="Override default max CPU percent for runners")
    parser.add_argument("--runner_temp_dir", type=str, help="Override temp directory path for runners")
    parser.add_argument("--pushover_token", type=str, help="Override Pushover API token")
    parser.add_argument("--pushover_user", type=str, help="Override Pushover user key")

def apply_config_overrides(args: argparse.Namespace, config: Config) -> None:
    """Apply CLI argument overrides to config."""
    if args.workflow_dir:
        config.workflow_dir = args.workflow_dir
    if args.schema_code_path:
        config.schema_code_path = args.schema_code_path
    if args.smolagent_factory_code_path:
        config.smolagent_factory_code_path = args.smolagent_factory_code_path
    if args.prompt_workflow_creator:
        config.prompt_workflow_creator = args.prompt_workflow_creator
    if args.workflow_llm_provider:
        config.workflow_llm_provider = args.workflow_llm_provider
    if args.mcp_health_endpoint:
        config.mcp_health_endpoint = args.mcp_health_endpoint
    if args.runner_default_python_version:
        config.runner_default_python_version = args.runner_default_python_version
    if args.runner_default_timeout:
        config.runner_default_timeout = args.runner_default_timeout
    if args.runner_default_max_memory_mb:
        config.runner_default_max_memory_mb = args.runner_default_max_memory_mb
    if args.runner_default_max_cpu_percent:
        config.runner_default_max_cpu_percent = args.runner_default_max_cpu_percent
    if args.runner_temp_dir:
        config.runner_temp_dir = args.runner_temp_dir
    if args.pushover_token:
        config.pushover_token = args.pushover_token
    if args.pushover_user:
        config.pushover_user = args.pushover_user

def collect_goals_from_user() -> List[str]:
    """Collect goals from user input for mass testing.
    
    Returns:
        List of goal strings entered by the user
    """
    goals = []
    print("\n🎯 Mass Testing Mode - Enter your goals")
    print("=" * 50)
    print("Enter goals one at a time. Press Enter with empty input to finish.")
    print("Type 'quit' or 'exit' to cancel.\n")
    
    goal_count = 1
    while True:
        try:
            goal = input(f"Goal {goal_count}: ").strip()
            
            if not goal:
                if goals:
                    break
                else:
                    print("⚠️ Please enter at least one goal or type 'quit' to cancel.")
                    continue
                    
            if goal.lower() in ['quit', 'exit']:
                print("❌ Mass testing cancelled by user.")
                return []
                
            goals.append(goal)
            goal_count += 1
            
        except KeyboardInterrupt:
            print("\n❌ Mass testing cancelled by user.")
            return []
    
    print(f"\n✅ Collected {len(goals)} goals for mass testing:")
    for i, goal in enumerate(goals, 1):
        print(f"  {i}. {goal[:60]}{'...' if len(goal) > 60 else ''}")
    
    return goals

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print(f"\n⚠️ Received signal {signum}. Shutting down gracefully...")
        # Cancel all running tasks
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def parallel_execution_mode(args, config):
    if getattr(args, 'mass_testing', False):
        goals = collect_goals_from_user()
        if not goals:
            print("❌ No goals provided for mass testing. Exiting.")
            return
        parallel_testing = ParallelTesting(config)
        results = parallel_testing.start_parallel_testing(
            goals=goals,
            template_uuid=args.load_template,
            judge=args.judge,
            human_validation=False,
            max_workers=getattr(args, 'max_workers', None)
        )
        print("\n📊 Mass Testing Results:")
        print("=" * 50)
        print(json.dumps(results, indent=2))
        print("=" * 50)

async def normal_execution_mode(args, config):
    dgm = GodelMachine(config)
    planner = Planner(config)
    if args.task:
        await dgm.start_dgm(goal_prompt=args.task, judge=args.judge, human_validation=False)
    elif args.goal:
        await planner.start_planner(goal_prompt=args.goal, template_uuid=args.load_template)
    else:
        raise ValueError("No goal provided. Use --task, --goal, or --mass-testing to start.")
            
async def main():
    """Main execution function"""
    config = Config()
    setup_signal_handlers()
    
    parser = argparse.ArgumentParser(
        description="Mimosa - A AI Agent Framework for advancing scientific research"
    )
    parser.add_argument(
        "--goal", type=str, help="Goal prompt for the workflow"
    )
    parser.add_argument(
        "--task",  type=str, help="Goal prompt for the workflow"
    )
    parser.add_argument(
        "--load_template", type=str, help="Optional workflow UUID to load"
    )
    parser.add_argument(
        "--judge", action="store_true", default=False, help="Enable judge for workflow evaluation"
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset to use (in csv format)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=16, help="Number of samples to use from dataset"
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=16, help="Maximum number of concurrent tasks"
    )

    add_config_arguments(parser, config)
    args = parser.parse_args()
    apply_config_overrides(args, config)

    validate_environment()
    config.validate_paths()

    dgm = GodelMachine(config)
    planner = Planner(config)

    try:
        if (args.dataset):
            print(f"Using {args.dataset} dataset")
            # Default to 3 concurrent tasks, can be overridden with --max_concurrent
            dataset_questions = read_dataset(args.dataset, args.num_samples)
            
            if dataset_questions:
                print(f"Running {len(dataset_questions)} questions with max {args.max_concurrent} concurrent tasks")
                
                # Create a semaphore to limit concurrent tasks
                semaphore = asyncio.Semaphore(args.max_concurrent)
                
                async def run_with_semaphore(question, answer):
                    """Run a single question with semaphore to limit concurrency"""
                    async with semaphore:
                        return question, answer, await planner.start_planner(
                            goal=question,
                            template_uuid=args.load_template,
                            judge=True,
                            answer=answer
                        )
                
                # Create tasks for all questions
                tasks = []
                for question, answer in dataset_questions:
                    task = asyncio.create_task(run_with_semaphore(question, answer))
                    tasks.append(task)
                
                # Run all tasks with concurrency limited by semaphore
                all_run = await asyncio.gather(*tasks)
                
                # Calculate average of good_answer values
                calculate_good_answer_average(all_run, dataset_name=args.dataset, workflow_prompt=config.prompt_workflow_creator)
            else:
                print("❌ No questions found in dataset or no goal provided.")
        else:    
            if args.single_task:
                await dgm.start_dgm(goal=args.single_task, judge=args.judge)
            elif args.goal:
                await planner.start_planner(goal=args.goal, template_uuid=args.load_template, judge=args.judge)
            else:
                raise ValueError("No goal provided. Use --single_task or --goal to start a task.")
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise

def read_dataset(dataset_file: str, num_samples: int = 10) -> List[Tuple[str, str]]:
    """
    Read dataset files from the specified path and return a subset of questions.
    
    Args:
        dataset_path: Path to the dataset directory or file
        num_samples: Number of samples to return (default: 10)
        
    Returns:
        List of tuples containing (question, answer) pairs
    """
    dataset_path = Path('datasets') / f"{dataset_file}.jsonl" 
    results = []
    
    try:  
        if dataset_path.exists():
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if "question" in data and "answer" in data:
                                match = re.search(r'#### (-?\d+)', data["answer"])
                                if match:
                                    answer = match.group(1)
                                    results.append((data["question"], data["answer"]))
                                else:
                                    print(f"No answer found for question: {data['question']}")
                        except json.JSONDecodeError:
                            print(f"⚠️ Error parsing JSON in {dataset_path}")
        else:
            print(f"❌ Dataset path {dataset_path} is neither a file nor a directory")
            return []
            
        # Return a random subset of the results
        if results:
            if len(results) > num_samples:
                return random.sample(results, num_samples)
            return results
        else:
            print(f"⚠️ No valid questions found in {dataset_path}")
            return []
            
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return []

def calculate_good_answer_average(runs: List[str], dataset_name: str, workflow_prompt: str) -> float:
    """
    Calculate the average of good_answer values across all workflow runs
    and save results to CSV and JSON files in the datasets folder.
    
    Args:
        uuids: List of workflow UUIDs to analyze
        dataset_name: Name of the dataset used for the workflows
        template_uuid: UUID of the workflow template used
        
    Returns:
        Average of good_answer values (0.0 to 1.0)
    """
    if not runs:
        print("No workflow UUIDs to analyze")
        return 0.0
    
    good_answer_count = 0
    total_workflows = len(runs)
    
    print(f"\nAnalyzing results for {total_workflows} workflows...")
    
    # Create a timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"datasets/runs/run_{dataset_name}_{timestamp}.json"

    os.makedirs('datasets/runs',exist_ok=True)
    
    # Prepare data for CSV and JSON
    csv_data = []
    threshold = 7
    
    for question, answer, uuid in runs:
        state_result_path = Path(f"sources/workflows/{uuid}/state_result.json")
        is_good_answer = False
        
        try:
            if state_result_path.exists():
                with open(state_result_path, 'r', encoding='utf-8') as f:
                    state_result = json.load(f)
                    
                    if "answer_correctness" in state_result["evaluation_scores"]:
                        answer_correctness = state_result["evaluation_scores"]["answer_correctness"]
                        is_good_answer = answer_correctness >= threshold
                        if is_good_answer:
                            good_answer_count += 1
                        
                        # Add data for CSV
                        csv_data.append({
                            "uuid": uuid,
                            "answer_correctness": answer_correctness,
                            "is_good_answer": is_good_answer,
                            "question":question,
                            "answer":answer
                        })
                    else:
                        print(f"⚠️ No 'answer_correctness' key found in state_result for UUID: {uuid}")
            else:
                print(f"⚠️ State result file not found for UUID: {uuid}")
        except Exception as e:
            print(f"❌ Error processing state result for UUID {uuid}: {e}")
    
    average = good_answer_count / total_workflows if total_workflows > 0 else 0
    
    # Create and save JSON file with analysis results
    try:
        json_data = {
            "dataset_name": dataset_name,
            "workflow_prompt": workflow_prompt,
            "average_good_answer": average,
            "thresold in range 1-10": threshold,
            "details": csv_data
        }
        
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, indent=2)
            
        print(f"✅ Analysis results saved to {json_filename}")
    except Exception as e:
        print(f"❌ Error writing to JSON file: {e}")
    
    print(f"\n=== Results Summary ===")
    print(f"Total workflows analyzed: {total_workflows}")
    print(f"Workflows with good answer: {good_answer_count}")
    print(f"Average good_answer rate: {average:.2f} ({good_answer_count}/{total_workflows})")
    
    return average

if __name__ == "__main__":
    asyncio.run(main())

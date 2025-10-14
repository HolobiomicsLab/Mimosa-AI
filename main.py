#!/usr/bin/env python3
"""
Mimosa - A AI Agent Framework for advancing scientific research
============================================================================
"""

import argparse
import asyncio
import os
import signal
import sys

import dotenv

# Prevent tokenizers parallelism warnings when forking processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import Config
from sources.core.dgm import GodelMachine
from sources.core.planner import Planner
from sources.extensibility.human_mode import HumanMode
from sources.extensibility.automated_mode import AutomatedMode
from sources.utils.dataset import calculate_good_answer_average, read_dataset
from sources.utils.scenario_loader import ScenarioLoader
from sources.utils.user_entry import collect_goals_from_user
from sources.utils.logging import setup_logging

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
    """Add additional CLI arguments for config parameters that can be overridden."""
    parser.add_argument("--workflow_dir", type=str, help="Override workflow directory path")
    parser.add_argument("--schema_code_path", type=str, help="Override state schema file path")
    parser.add_argument("--smolagent_factory_code_path", type=str, help="Override SmolAgent factory file path")
    parser.add_argument("--prompt_workflow_creator", type=str, help="Override system prompt file path")
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

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print(f"\n⚠️ Received signal {signum}. Shutting down gracefully...")
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def manual_mode(args, config):
    hm = HumanMode(config)
    await hm.shellLoop()

async def dataset_execution_mode(args, config):
    dgm = GodelMachine(config)
    print(f"Using {args.dataset} dataset")
    dataset_questions = read_dataset(args.dataset, args.num_samples)
    if dataset_questions:
        print(f"Running {len(dataset_questions)} questions with max {args.max_concurrent} concurrent tasks")
        semaphore = asyncio.Semaphore(args.max_concurrent)
        async def run_with_semaphore(question, answer):
            """Run a single question with semaphore to limit concurrency"""
            async with semaphore:
                return question, answer, await dgm.start_dgm(
                    goal=question,
                    template_uuid=args.load_template,
                    judge=True,
                    answer=answer,
                    max_iteration=1
                )
        tasks = []
        for question, answer in dataset_questions:
            task = asyncio.create_task(run_with_semaphore(question, answer))
            tasks.append(task)
        all_run = await asyncio.gather(*tasks)
        calculate_good_answer_average(all_run, dataset_name=args.dataset, workflow_prompt=config.prompt_workflow_creator)
    else:
        print("❌ No questions found in dataset or no goal provided.")

async def automated_mode(args, config):
    """Run autonomous mode where LLM generates and executes tasks automatically."""
    automated = AutomatedMode(config, csv_runs_limit=args.csv_runs_limit)
    await automated.start_autonomous_mode()

async def normal_execution_mode(args, config):
    dgm = GodelMachine(config)
    planner = Planner(config)
    if args.scenario:
        scenario_file = ScenarioLoader().load_scenario(args.scenario)
        args.task = scenario_file["goal"]
        args.judge = True
    if args.task:
        await dgm.start_dgm(goal=args.task,
                            judge=args.judge,
                            scenario_id=args.scenario,
                            human_validation=False,
                            max_iteration=args.max_dgm_iterations
                           )
    elif args.goal:
        await planner.start_planner(goal=args.goal,
                                    judge=args.judge,
                                    max_dgm_iteration=args.max_dgm_iterations
                                   )
    else:
        raise ValueError("No goal provided. Use --task, --goal to start.")

async def main():
    """Main execution function"""
    config = Config()
    setup_signal_handlers()

    parser = argparse.ArgumentParser(
        description="Mimosa - A AI Agent Framework for advancing scientific research"
    )
    parser.add_argument(
        "--goal", type=str, help="Goal for Mimosa to achieve (for planner mode)"
    )
    parser.add_argument(
        "--task",  type=str, help="Single task mode (no planner)"
    )
    parser.add_argument(
        "--manual", action="store_true", help="Full manual mode (No LLM, human choose all actions)."
    )
    parser.add_argument(
        "--automated", action="store_true", help="Autonomous mode (Run Mimosa on multiple papers from a CSV, automatically monitor run, evaluate, save capsules)"
    )
    parser.add_argument(
        "--csv_runs_limit", type=int, default=200, help="Maximum number of autonomous iterations (for --automated mode)"
    )
    parser.add_argument(
        "--dataset", type=str, help="evalaluation mode for single task mode, specify dataset folder to use such as GSMK8 (csv)"
    )
    parser.add_argument(
        "--load_template", type=str, help="Optional workflow UUID to load", default=None
    )
    parser.add_argument(
        "--judge", action="store_true", default=False, help="Enable judge for workflow evaluation"
    )
    parser.add_argument(
        "--scenario", type=str, help="Scenario for workflow evaluation"
    )
    parser.add_argument(
        "--num_samples", type=int, default=16, help="Number of samples to use from dataset"
    )
    parser.add_argument(
        "--max_concurrent", type=int, default=16, help="Maximum number of concurrent tasks"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging to console"
    )
    parser.add_argument(
        "--max_dgm_iterations", type=int, default=1, help="Maximum number of DGM retry iterations"
    )

    add_config_arguments(parser, config)
    args = parser.parse_args()

    # Setup logging with debug flag
    setup_logging(debug=args.debug)

    apply_config_overrides(args, config)

    validate_environment()

    config.create_paths()
    config.validate_paths()

    try:
        if (args.dataset):
            await dataset_execution_mode(args, config)
        elif (args.manual):
            await manual_mode(args, config)
        elif (args.automated):
            await automated_mode(args, config)
        elif args.task or args.goal or args.scenario:
            await normal_execution_mode(args, config)
        else:
            raise ValueError("No goal provided. Use --task, --goal, --automated  to start.")
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

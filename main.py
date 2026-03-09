#!nne/usr/bin/env python3
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
from sources.core.dgm import DarwinMachine
from sources.core.planner import Planner
from sources.extensibility.human_mode import HumanMode
from sources.evaluation.csv_mode import CsvEvaluationMode
from sources.evaluation.scenario_loader import ScenarioLoader
from sources.utils.logging import setup_logging
from sources.utils.transfer_toolomics import LocalTransfer
from sources.utils.precheck import PreCheck

dotenv.load_dotenv()

def validate_environment() -> None:
    """Validate required environment configuration."""
    if not os.getenv("HF_TOKEN"):
        raise ValueError(
            "⚠️ HF_TOKEN environment variable is not set. Please set it to your Hugging Face token. "
        )
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError(
            "⚠️ ANTHROPIC_API_KEY environment variable is not set. Please set it to your OpenAI API key."
        )
    if not os.getenv("PUSHOVER_USER") or not os.getenv("PUSHOVER_TOKEN"):
        print(
            "⚠️ PUSHOVER_USER/PUSHOVER_TOKEN not set. We advice using pushover for getting notifications upon task completion. "
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
    if args.max_evolve_iterations:
        config.max_learning_evolve_iterations = args.max_evolve_iterations

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

async def papers_mode(args, config):
    papers = CsvEvaluationMode(config, csv_runs_limit=args.csv_runs_limit)
    await papers.start_evaluation(dataset_type="default",
                                  dataset_path=args.papers,
                                  learning=args.learn,
                                  single_agent_mode=args.single_agent
                                 )

async def science_bench_papers_mode(args, config):
    papers = CsvEvaluationMode(config, csv_runs_limit=args.csv_runs_limit)
    if args.single_agent:
        print(f"⚠️ Starting in single agent mode")
    await papers.start_evaluation(dataset_type="science_agent_bench",
                                  dataset_path="datasets/ScienceAgentBench.csv",
                                  learning=args.learn,
                                  single_agent_mode=args.single_agent
                                 )

def load_goal_from_file_or_string(goal_input: str) -> str:
    """
    Load goal from file if the input is a file path, otherwise return the string as-is.

    Args:
        goal_input: Either a file path or a goal string

    Returns:
        The goal content (either loaded from file or the original string)
    """
    if goal_input and os.path.isfile(goal_input):
        try:
            with open(goal_input, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                print(f"✅ Loaded goal from file: {goal_input}")
                return content
        except Exception as e:
            print(f"⚠️ Failed to read file '{goal_input}': {e}")
            print(f"Using input as a literal string instead.")
            return goal_input
    return goal_input

async def normal_execution_mode(args, config):
    dgm = DarwinMachine(config)
    planner = Planner(config)
    if args.scenario:
        scenario_file = ScenarioLoader().load_scenario(args.scenario)
        args.task = scenario_file["goal"]
    if args.task:
        # Load goal from file if args.task is a file path
        goal_content = load_goal_from_file_or_string(args.task)
        if args.single_agent:
            print(f"⚠️ Starting in single agent mode")
        await dgm.start_dgm(goal=goal_content,
                            judge=not args.disable_judge,
                            scenario_rubric=args.scenario,
                            max_iteration=args.max_evolve_iterations,
                            learning_mode=args.learn,
                            single_agent_mode=args.single_agent
                           )
    elif args.goal:
        # Load goal from file if args.goal is a file path
        goal_content = load_goal_from_file_or_string(args.goal)
        await planner.start_planner(goal=goal_content,
                                    judge=not args.disable_judge,
                                    max_evolve_iteration=args.max_evolve_iterations
                                   )
        trs = LocalTransfer(workspace_path=config.workspace_dir, runs_capsule_dir=config.runs_capsule_dir)
        trs.transfer_workspace_files_to_capsule(args.goal or args.task)
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
        "--config", type=str, help="Path to configuration JSON file to load"
    )
    parser.add_argument(
        "--goal", type=str, help="Goal for Mimosa to achieve (for planner mode)"
    )
    parser.add_argument(
        "--task",  type=str, help="Single task mode (no planner)"
    )
    parser.add_argument(
        "--learn", action="store_true", help="Learning mode. Retry task with Iterative-Learning until threshold score it met.", default=False
    )
    parser.add_argument(
        "--single_agent", action="store_true", help="Single-agent mode for benchmark comparaison, not recommended for real-tasks use.", default=False
    )
    parser.add_argument(
        "--manual", action="store_true", help="Full manual mode (No LLM, human choose all actions)."
    )
    parser.add_argument(
        "--papers", type=str, help="Papers evaluation mode (Run Mimosa on multiple papers from a CSV, automatically monitor run, evaluate, save capsules)"
    )
    parser.add_argument(
        "--science_agent_bench", action="store_true", help="Papers mode on ScienceAgentBench (Run Mimosa on multiple science agent bench task from a CSV, automatically monitor run, evaluate, save capsules)"
    )
    parser.add_argument(
        "--csv_runs_limit", type=int, default=200, help="Maximum number of autonomous iterations (for --papers mode)"
    )
    parser.add_argument(
        "--disable_judge", action="store_true", default=False, help="Disable judge for workflow evaluation"
    )
    parser.add_argument(
        "--scenario", type=str, help="Use scenario benchmark (eg: datasets/scenarios/X.json) with criterions for workflow evaluation and auto-improvement"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging to console"
    )
    parser.add_argument(
        "--max_evolve_iterations", type=int, default=1, help="Maximum number of learning iterations. Used for retrying/learning a task."
    )

    add_config_arguments(parser, config)
    args = parser.parse_args()

    # Load config from file if provided
    if args.config:
        config.load(args.config)
        print(f"Configuration loaded from: {args.config}")

    # Setup logging with debug flag
    setup_logging(debug=args.debug)

    # Apply CLI argument overrides (these override config file values)
    apply_config_overrides(args, config)

    validate_environment()

    config.create_paths()
    config.validate_paths()

    PreCheck(config).run()

    try:
        if (args.manual):
            await manual_mode(args, config)
        elif (args.papers):
            await papers_mode(args, config)
        elif (args.science_agent_bench):
            await science_bench_papers_mode(args, config)
        elif args.task or args.goal or args.scenario:
            await normal_execution_mode(args, config)
        else:
            raise ValueError("No goal provided. Use --task, --goal, --papers  to start.")
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

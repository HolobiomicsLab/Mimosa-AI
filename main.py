#!/usr/bin/env python3
"""
Mimosa - A AI Agent Framework for advancing scientific research
============================================================================
"""

import argparse
import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional

import dotenv
import requests
from fastmcp import Client

from config import Config
from sources.core.dgm import GodelMachine
from sources.core.planner import Planner
from sources.core.parallel_testing import ParallelTesting

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

async def mass_testing_mode(args, config):
    if getattr(args, 'mass_testing', False):
        # Mass testing mode
        goals = collect_goals_from_user()
        if not goals:
            print("❌ No goals provided for mass testing. Exiting.")
            return
        parallel_testing = ParallelTesting(config)
        results = parallel_testing.start_parallel_testing(
            goals=goals,
            template_uuid=args.load_template,
            judge=args.judge,
            human_validation=False,  # Disable human validation for mass testing
            max_workers=getattr(args, 'max_workers', None)
        )
        print("\n📊 Mass Testing Results:")
        print("=" * 50)
        print(json.dumps(results, indent=2))
        print("=" * 50)
            
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
        "--mass-testing", action="store_true", default=False, help="Enable mass testing mode with multiple goals"
    )
    parser.add_argument(
        "--max-workers", type=int, help="Maximum number of parallel processes for mass testing"
    )

    add_config_arguments(parser, config)
    args = parser.parse_args()
    apply_config_overrides(args, config)

    validate_environment()
    config.validate_paths()

    try:
        if args.mass_testing:
            # Mass testing mode
            await mass_testing_mode(args, config)
        else:
            # Single execution mode
            dgm = GodelMachine(config)
            planner = Planner(config)
            if args.task:
                await dgm.start_dgm(goal_prompt=args.task, judge=args.judge, human_validation=False)
            elif args.goal:
                await planner.start_planner(goal_prompt=args.goal, template_uuid=args.load_template)
            else:
                raise ValueError("No goal provided. Use --task, --goal, or --mass-testing to start.")
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Cleaning up...")
        raise
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

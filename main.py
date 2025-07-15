#!/usr/bin/env python3
"""
Mimosa - A AI Agent Framework for advancing scientific research
============================================================================
"""

import os
import asyncio
import argparse
import requests
import dotenv

from config import Config
from fastmcp import Client
import asyncio
from typing import Optional, List

from sources.core.planner import Planner
from sources.core.dgm import GodelMachine

dotenv.load_dotenv()

def validate_environment() -> None:
    """Validate required environment configuration.
    """
    if not os.getenv('HF_TOKEN') or os.getenv('HF_TOKEN') == "":
        raise ValueError("⚠️ HF_TOKEN environment variable is not set. Please set it to your Hugging Face token.")
    if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == "":
        raise ValueError("⚠️ OPENAI_API_KEY environment variable is not set. Please set it to your OpenAI API key.")

async def main():
    """Main execution function"""
    config = Config()
    parser = argparse.ArgumentParser(description="Mimosa - A AI Agent Framework for advancing scientific research")
    parser.add_argument("--single-task-goal", type=str, help="Start a single task with a goal prompt.")
    parser.add_argument("--goal", type=str, help="Goal prompt for the Mimosa using task planning.")
    parser.add_argument("--load_template", type=str, help="Optional workflow UUID to load")
    args = parser.parse_args()

    validate_environment()
    config.validate_paths()

    dgm = GodelMachine(config)
    if args.single_task_goal:
        await dgm.start_dgm(goal_prompt=args.single_task_goal)
    elif args.goal:
        planner = Planner(config)
        await planner.start_planner(goal_prompt=args.goal, template_uuid=args.load_template)
    else:
        raise ValueError("No goal provided. Use --single-task-goal or --goal to start a task.")

if __name__ == "__main__":
    asyncio.run(main())

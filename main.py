#!/usr/bin/env python3
"""
Mimosa - A AI Agent Framework for advancing scientific research
============================================================================
"""

import argparse
import asyncio
import os
from typing import List, Optional

import dotenv
import requests
from fastmcp import Client

from config import Config
from sources.core.dgm import GodelMachine

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


async def main():
    """Main execution function"""
    config = Config()
    parser = argparse.ArgumentParser(
        description="Mimosa - A AI Agent Framework for advancing scientific research"
    )
    parser.add_argument(
        "--goal", required=True, type=str, help="Goal prompt for the workflow"
    )
    parser.add_argument(
        "--load_template", type=str, help="Optional workflow UUID to load"
    )
    args = parser.parse_args()

    validate_environment()
    config.validate_paths()

    dgm = GodelMachine(config)
    await dgm.start_dgm(goal_prompt=args.goal, template_uuid=args.load_template)


if __name__ == "__main__":
    asyncio.run(main())

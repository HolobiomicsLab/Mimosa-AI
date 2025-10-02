#!/usr/bin/env python3
"""
Test for the combined WorkflowEvaluator.
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from sources.core.dgm import GodelMachine
from sources.post_processing.evaluator import WorkflowEvaluator

load_dotenv()


class TestConfig:
    """Simple config class for testing."""

    def __init__(self, memory_dir="sources/memory", workflow_dir="sources/workflows"):
        self.memory_dir = memory_dir
        self.workflow_dir = workflow_dir
        self.model_pricing = {}
        self.pushover_token = None
        self.pushover_user = None

config = Config()


def test_judge_evaluation(workflow_id, answer):
    """Test the judge-based evaluation."""
    
    evaluator = WorkflowEvaluator(config)

    print(f"Testing judge evaluation for workflow {workflow_id}")
    return evaluator.evaluate(workflow_id, answer=answer)


def test_scenario_evaluation(workflow_id, scenario_id):
    """Test the scenario-based evaluation."""
    evaluator = WorkflowEvaluator(config)

    print(
        f"Testing scenario evaluation for workflow {workflow_id} with scenario {scenario_id}"
    )
    return evaluator.evaluate(workflow_id, scenario_id=scenario_id)


def main():
    """Main function for testing the evaluator."""
    parser = argparse.ArgumentParser(description="Test the WorkflowEvaluator")
    parser.add_argument(
        "--workflow_id", required=True, help="UUID of workflow to evaluate"
    )
    parser.add_argument(
        "--scenario_id", help="Optional scenario ID for scenario-based evaluation"
    )
    parser.add_argument(
        "--answer", help="Optional expected answer for judge evaluation"
    )

    args = parser.parse_args()

    if args.scenario_id:
        eval_type = test_scenario_evaluation(args.workflow_id, args.scenario_id)
    else:
        eval_type = test_judge_evaluation(args.workflow_id, args.answer)

    dgm = GodelMachine(config)

    wf_state = dgm.load_wf_state_result(args.workflow_id)
    wf_rewards = dgm.get_total_rewards(wf_state, eval_type)

    print(f"Evaluation type: {eval_type}")
    print(f"Flow state: {json.dumps(wf_state, indent=2)}")
    print(f"Total rewards: {wf_rewards}")


if __name__ == "__main__":
    exit(main())

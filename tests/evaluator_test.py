#!/usr/bin/env python3
"""
Test for the combined WorkflowEvaluator.
"""

import argparse
import json
import os
from pathlib import Path

from sources.core.evaluator import WorkflowEvaluator


class TestConfig:
    """Simple config class for testing."""
    
    def __init__(self, memory_dir="sources/memory", workflow_dir="sources/workflows"):
        self.memory_dir = memory_dir
        self.workflow_dir = workflow_dir
        self.model_pricing = {}
        self.pushover_token = None
        self.pushover_user = None


def test_judge_evaluation(workflow_id):
    """Test the judge-based evaluation."""
    config = TestConfig()
    evaluator = WorkflowEvaluator(config)
    
    print(f"Testing judge evaluation for workflow {workflow_id}")
    scores = evaluator.evaluate(workflow_id, short=True)
    
    print("Evaluation scores:")
    print(json.dumps(scores, indent=2))
    
    return scores


def test_scenario_evaluation(workflow_id, scenario_id):
    """Test the scenario-based evaluation."""
    config = TestConfig()
    evaluator = WorkflowEvaluator(config)
    
    print(f"Testing scenario evaluation for workflow {workflow_id} with scenario {scenario_id}")
    results = evaluator.evaluate(workflow_id, scenario_id=scenario_id)
    
    print("Evaluation results:")
    print(json.dumps(results, indent=2))
    
    return results


def main():
    """Main function for testing the evaluator."""
    parser = argparse.ArgumentParser(description="Test the WorkflowEvaluator")
    parser.add_argument("--workflow_id", required=True, help="UUID of workflow to evaluate")
    parser.add_argument("--scenario_id", help="Optional scenario ID for scenario-based evaluation")
    parser.add_argument("--answer", help="Optional expected answer for judge evaluation")
    
    args = parser.parse_args()
    
    if args.scenario_id:
        test_scenario_evaluation(args.workflow_id, args.scenario_id)
    else:
        test_judge_evaluation(args.workflow_id)
    
    return 0


if __name__ == "__main__":
    exit(main())
#!/usr/bin/env python3
"""
Simple test script for the WorkflowEvaluator class.
Tests both generic evaluation (with and without answer) and scenario-based evaluation.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add the parent directory to the path so we can import from sources
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sources.core.evaluator import WorkflowEvaluator

# Load environment variables
load_dotenv()


class TestConfig:
    """Simple config class for testing."""

    def __init__(self, memory_dir="sources/memory", workflow_dir="sources/workflows"):
        self.memory_dir = memory_dir
        self.workflow_dir = workflow_dir
        self.model_pricing = {}


def test_generic_evaluation(workflow_id, answer=None):
    """
    Test the generic evaluation (judge) functionality.
    
    Args:
        workflow_id: UUID of the workflow to evaluate
        answer: Optional expected answer for evaluation
    """
    print(f"\n=== Testing Generic Evaluation for Workflow {workflow_id} ===")
    if answer:
        print(f"With expected answer: {answer}")
    
    config = TestConfig()
    evaluator = WorkflowEvaluator(config)
    
    try:
        result_type = evaluator.evaluate(workflow_id, answer=answer)
        print(f"Evaluation completed successfully. Type: {result_type}")
        
        # Load and display the evaluation results
        display_evaluation_results(config.workflow_dir, workflow_id, "generic")
        
        return True
    except Exception as e:
        print(f"Error during generic evaluation: {str(e)}")
        return False


def test_scenario_evaluation(workflow_id, scenario_id):
    """
    Test the scenario-based evaluation functionality.
    
    Args:
        workflow_id: UUID of the workflow to evaluate
        scenario_id: ID of the scenario to evaluate against
    """
    print(f"\n=== Testing Scenario Evaluation for Workflow {workflow_id} ===")
    print(f"With scenario: {scenario_id}")
    
    config = TestConfig()
    evaluator = WorkflowEvaluator(config)
    
    try:
        result_type = evaluator.evaluate(workflow_id, scenario_id=scenario_id)
        print(f"Evaluation completed successfully. Type: {result_type}")
        
        # Load and display the evaluation results
        display_evaluation_results(config.workflow_dir, workflow_id, "scenario")
        
        return True
    except Exception as e:
        print(f"Error during scenario evaluation: {str(e)}")
        return False


def display_evaluation_results(workflow_dir, workflow_id, eval_type):
    """
    Display the evaluation results from the state_result.json file.
    
    Args:
        workflow_dir: Directory containing workflow data
        workflow_id: UUID of the workflow
        eval_type: Type of evaluation ('generic' or 'scenario')
    """
    workflow_path = Path(workflow_dir) / workflow_id
    state_result_path = workflow_path / "state_result.json"
    
    try:
        with open(state_result_path) as f:
            state_result = json.load(f)
        
        if "evaluation" in state_result and eval_type in state_result["evaluation"]:
            print("\nEvaluation Results:")
            print(json.dumps(state_result["evaluation"][eval_type], indent=2))
        else:
            print(f"\nNo {eval_type} evaluation results found in state_result.json")
    except Exception as e:
        print(f"\nError loading evaluation results: {str(e)}")


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
        "--answer", help="Optional expected answer for generic evaluation"
    )
    parser.add_argument(
        "--test_all", action="store_true", 
        help="Test both generic (with and without answer) and scenario evaluation"
    )

    args = parser.parse_args()
    
    success = True
    
    if args.test_all:
        # Test generic evaluation without answer
        success = test_generic_evaluation(args.workflow_id) and success
        
        # Test generic evaluation with answer (if provided)
        if args.answer:
            success = test_generic_evaluation(args.workflow_id, args.answer) and success
        
        # Test scenario evaluation (if scenario_id provided)
        if args.scenario_id:
            success = test_scenario_evaluation(args.workflow_id, args.scenario_id) and success
    else:
        # Test based on provided arguments
        if args.scenario_id:
            success = test_scenario_evaluation(args.workflow_id, args.scenario_id)
        else:
            success = test_generic_evaluation(args.workflow_id, args.answer)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
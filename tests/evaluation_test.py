import os
import sys

from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sources.evaluation.simple_evaluator import Evaluator

load_dotenv()


def test_evaluation(workflow_id: str, scenario_id: str):
    evaluator = Evaluator()

    # Run evaluation
    result = evaluator.evaluate_workflow(workflow_id, scenario_id)

    print(f"Evaluation completed for workflow {workflow_id}")
    print(f"Scenario: {scenario_id}")
    print(f"Score: {result.get('score', 'N/A')}")
    print(
        f"Passed assertions: {result.get('passed_assertions', 0)}/"
        f"{result.get('total_assertions', 0)}"
    )


if __name__ == "__main__":
    workflow_id = "90e94aa823e0427b88059b8ed480c42c"
    scenario_id = "llama_cpp_installation"
    test_evaluation(workflow_id, scenario_id)
    print("Evaluation test completed.")

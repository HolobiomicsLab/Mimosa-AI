import json

import pytest
from dotenv import load_dotenv

from config import Config

# ``sources.core.judge`` / ``WorkflowJudge`` were removed during the
# refactor that moved the judging logic into ``sources.evaluators``.
# This file was left behind; skip collection cleanly until the user decides
# whether to rewrite it against the current evaluator API or delete it.
pytest.importorskip(
    "sources.core.judge",
    reason="sources.core.judge no longer exists; legacy test file kept for review",
)
from sources.core.judge import WorkflowJudge  # noqa: E402  pragma: no cover

load_dotenv()


def test_judge(uuid: str):
    config = Config()
    judge = WorkflowJudge(config)

    # Generate evaluation text
    judge.evaluate(uuid)

    # Calculate costs
    # judge.calculate_cost(uuid)


if __name__ == "__main__":
    path = "datasets/runs/run_GSMK8_20250728_113102.json"

    with open(path) as f:
        json_runs = json.load(f)
        for run in json_runs["details"]:
            test_judge(run["uuid"])

    print("Benchmark format test completed.")

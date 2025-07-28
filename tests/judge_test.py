import json

from dotenv import load_dotenv

from config import Config
from sources.core.judge import WorkflowJudge

load_dotenv()


def test_judge(uuid: str):
    config = Config()
    judge = WorkflowJudge(config)

    # Generate evaluation text
    judge.evaluate(uuid)

    # Calculate costs
    #judge.calculate_cost(uuid)


if __name__ == "__main__":

    path = 'datasets/runs/run_GSMK8_20250728_113102.json'

    with open(path) as f:
        json_runs = json.load(f)
        for run in json_runs['details']:
            test_judge(run['uuid'])

    print("Benchmark format test completed.")

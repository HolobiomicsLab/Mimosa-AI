from dotenv import load_dotenv

from config import Config
from sources.core.judge import WorkflowJudge

load_dotenv()

def test_judge(uuid:str):
    config = Config()
    judge = WorkflowJudge(config)

    # Generate evaluation text
    judge.evaluate(uuid)

    # Calculate costs
    judge.calculate_cost(uuid)


if __name__ == "__main__":
    uuid = "630f8e9a2a26430aad4296363e790464"
    test_judge(uuid)
    print("Benchmark format test completed.")

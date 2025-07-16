from dotenv import load_dotenv

from config import Config
from sources.core.benchmarker import Benchmarker

load_dotenv()

def test_benchmark_format(uuid:str):
    config = Config()
    benchmarker = Benchmarker(config, uuid)
    benchmarker.generate_text()
    benchmarker.evaluate()


if __name__ == "__main__":
    uuid = "42b6e35b59a7474eaf9711653740795a"
    test_benchmark_format(uuid)
    print("Benchmark format test completed.")
    # You can add more tests or assertions here to validate the output
    # For example, check if the generated text file exists and contains expected content
    # Or validate the results returned by benchmarker.get_results()

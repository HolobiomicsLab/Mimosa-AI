from sources.core.benchmarker import Benchmarker
from config import Config

def test_benchmark_format(uuid:str):
    config = Config()
    benchmarker = Benchmarker(config, uuid)
    benchmarker.generate_text()
    print(benchmarker)


if __name__ == "__main__":
    uuid = "794d54e2d7ae49f9a426e66431d63232"
    test_benchmark_format(uuid)
    print("Benchmark format test completed.")
    # You can add more tests or assertions here to validate the output
    # For example, check if the generated text file exists and contains expected content
    # Or validate the results returned by benchmarker.get_results()
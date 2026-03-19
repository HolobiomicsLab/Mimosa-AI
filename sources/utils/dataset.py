import datetime
import json
import os
import random
from pathlib import Path

eval_path = Path('sources/evaluation')

def resolve_dataset_path(dataset_file: str) -> Path:
    path_1 = eval_path /"datasets" / f"{dataset_file}.jsonl"
    path_2 = Path("evaluation/datasets") / dataset_file
    if os.path.exists(path_1):
        return path_1
    if os.path.exists(path_2):
        return path_2
    return Path(dataset_file)  # Fallback to raw file path if neither exists


def read_dataset(dataset_file: str, num_samples: int = 10) -> list[tuple[str, str]]:
    """
    Read dataset files from the specified path and return a subset of questions.

    Args:
        dataset_path: Path to the dataset directory or file
        num_samples: Number of samples to return (default: 10)

    Returns:
        List of tuples containing (question, answer) pairs
    """
    dataset_path = resolve_dataset_path(dataset_file)
    results = []

    try:
        if dataset_path.exists():
            with open(dataset_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if "question" in data and "answer" in data:
                                results.append((data["question"], data["answer"]))
                        except json.JSONDecodeError:
                            print(f"⚠️ Error parsing JSON in {dataset_path}")
        else:
            print(f"❌ Dataset path {dataset_path} is neither a file nor a directory")
            return []

        # Return a random subset of the results
        if results:
            if len(results) > num_samples:
                return random.sample(results, num_samples)
            return results
        else:
            print(f"⚠️ No valid questions found in {dataset_path}")
            return []

    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return []


def calculate_good_answer_average(
    runs: list[str], dataset_name: str, workflow_prompt: str
) -> float:
    """
    Calculate the average of good_answer values across all workflow runs
    and save results to CSV and JSON files in the datasets folder.

    Args:
        uuids: List of workflow UUIDs to analyze
        dataset_name: Name of the dataset used for the workflows
        template_uuid: UUID of the workflow template used

    Returns:
        Average of good_answer values (0.0 to 1.0)
    """
    if not runs:
        print("No workflow UUIDs to analyze")
        return 0.0

    good_answer_count = 0
    total_workflows = len(runs)

    print(f"\nAnalyzing results for {total_workflows} workflows...")

    # Create a timestamp for filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = eval_path / "runs" / f"run_{dataset_name}_{timestamp}.json"

    os.makedirs("datasets/runs", exist_ok=True)

    # Prepare data for CSV and JSON
    csv_data = []
    threshold = 7

    for question, answer, uuid in runs:
        state_result_path = Path(f"sources/workflows/{uuid}/state_result.json")
        is_good_answer = False

        try:
            if state_result_path.exists():
                with open(state_result_path, encoding="utf-8") as f:
                    state_result = json.load(f)

                    evaluation_scores = state_result.get("evaluation_scores", {})
                    answer_plausibility = evaluation_scores.get(
                        "answer_plausibility",
                        evaluation_scores.get("answer_correctness"),
                    )

                    if answer_plausibility is not None:
                        is_good_answer = answer_plausibility >= threshold
                        if is_good_answer:
                            good_answer_count += 1

                        # Add data for CSV
                        csv_data.append(
                            {
                                "uuid": uuid,
                                "answer_plausibility": answer_plausibility,
                                "is_good_answer": is_good_answer,
                                "question": question,
                                "answer": answer,
                            }
                        )
                    else:
                        print(
                            f"⚠️ No 'answer_plausibility' or legacy 'answer_correctness' key found in state_result for UUID: {uuid}"
                        )
            else:
                print(f"⚠️ State result file not found for UUID: {uuid}")
        except Exception as e:
            print(f"❌ Error processing state result for UUID {uuid}: {e}")

    average = good_answer_count / total_workflows if total_workflows > 0 else 0

    # Create and save JSON file with analysis results
    try:
        json_data = {
            "dataset_name": dataset_name,
            "workflow_prompt": workflow_prompt,
            "average_good_answer": average,
            "thresold": threshold,
            "details": csv_data,
        }

        with open(json_filename, "w", encoding="utf-8") as jsonfile:
            json.dump(json_data, jsonfile, indent=2)

        print(f"✅ Analysis results saved to {json_filename}")
    except Exception as e:
        print(f"❌ Error writing to JSON file: {e}")

    print("\n=== Results Summary ===")
    print(f"Total workflows analyzed: {total_workflows}")
    print(f"Workflows with good answer: {good_answer_count}")
    print(
        f"Average good_answer rate: {average:.2f} ({good_answer_count}/{total_workflows})"
    )

    return average

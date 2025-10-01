#!/usr/bin/env python3
"""
Error Analysis Script for LLM-based Multi-Agent System

This script analyzes the memory data for a list of UUIDs and uses an LLM to generate
explanations of errors that occurred during the workflow execution.
"""

import csv
import json
import os
from pathlib import Path
from typing import Any

import dotenv

from sources.core.llm_provider import LLMConfig, LLMProvider

dotenv.load_dotenv()

# Constants
MEMORY_DIR = Path("sources/memory")
RESULTS_DIR = Path("evaluation/runs")
config_llm = LLMConfig.from_dict({"model": "gpt-4o-mini"})

assert os.path.exists(MEMORY_DIR), f"Memory folder does not exists at {MEMORY_DIR}."
assert os.path.exists(RESULTS_DIR), f"Results folder does not exists at {RESULTS_DIR}"


def load_memory_data(uuid: str) -> dict[str, Any]:
    """
    Load memory data for a given UUID, excluding judge.json.

    Args:
        uuid: The UUID of the workflow execution

    Returns:
        A dictionary containing the memory data
    """
    memory_path = MEMORY_DIR / uuid

    if not memory_path.exists():
        raise FileNotFoundError(f"Memory directory for UUID {uuid} not found")

    # Load formatted.txt
    formatted_path = memory_path / "formated.txt"
    formatted_content = ""
    if formatted_path.exists():
        with open(formatted_path) as f:
            formatted_content = f.read()

    # Return all data
    return {"formatted": formatted_content}


def analyze_error(uuid: str, memory_data: dict[str, Any]) -> str:
    """
    Analyze the error in the workflow execution using LLMProvider.

    Args:
        uuid: The UUID of the workflow execution
        memory_data: The memory data for the workflow execution

    Returns:
        An explanation of the error causes
    """

    # Create LLMProvider
    llm_provider = LLMProvider(
        agent_name="error_analyzer",
        system_msg="""You are an expert at analyzing errors in LLM-based multi-agent systems.
Your task is to analyze the memory data from a workflow execution and identify what went wrong.
Focus on:
1. Where the error occurred in the workflow
2. What caused the error
3. How the error could be fixed

Provide a concise explanation of the error causes in the following format:
ERROR_LOCATION: [Where the error occurred]
ERROR_CAUSE: [What caused the error]
ERROR_FIX: [How the error could be fixed]
""",
        config=config_llm,
    )

    # Prepare the prompt
    prompt = f"""
# Workflow Memory Analysis for UUID: {uuid}

## Workflow Trace
```
{memory_data.get("formatted", "No formatted data available")}
```
"""
    try:
        explanation = llm_provider(prompt)
        return explanation
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return f"Error analyzing workflow: {str(e)}"


def create_error_summary(error_analyses: dict[str, str], run_filename: str) -> str:
    """
    Create a summary of all error causes.

    Args:
        error_analyses: Dictionary mapping UUIDs to error analyses

    Returns:
        A summary of all error causes
    """
    # Create a directory for the summary
    summary_dir = RESULTS_DIR
    summary_dir.mkdir(exist_ok=True)

    # Create LLMProvider for summary
    llm_provider = LLMProvider(
        agent_name="error_summarizer",
        system_msg="""You are an expert at analyzing patterns in errors from LLM-based multi-agent systems.
Your task is to create a summary of error causes from multiple workflow executions.
Focus on:
1. Common patterns in errors
2. Root causes of failures
3. Recommendations for system improvements

Provide a comprehensive summary of the error patterns and recommendations for improvement.
""",
        config=config_llm,
    )

    # Prepare the prompt
    prompt = "# Error Analysis Summary\n\nBelow are the error analyses for multiple workflow executions:\n\n"

    for uuid, analysis in error_analyses.items():
        prompt += f"## UUID: {uuid}\n```\n{analysis}\n```\n\n"

    prompt += "\n# Summary Request\nBased on the above error analyses, create a comprehensive summary of error patterns and recommendations for system improvements."

    # Call the LLM
    try:
        summary = llm_provider(prompt)

        # Save summary to file in results directory
        summary_path = RESULTS_DIR / f"summary_{run_filename.replace('run_', '')}.txt"
        with open(summary_path, "w") as f:
            f.write(summary)

        print(f"Error summary saved to {summary_path}")

        return summary
    except Exception as e:
        print(f"Error creating summary: {e}")
        return f"Error creating summary: {str(e)}"


def extract_error_components(analysis: str) -> dict[str, str]:
    """
    Extract error components from the analysis.

    Args:
        analysis: The error analysis text

    Returns:
        A dictionary with error location, cause, and fix
    """
    components = {"ERROR_LOCATION": "", "ERROR_CAUSE": "", "ERROR_FIX": ""}

    for line in analysis.split("\n"):
        for component in components:
            if line.startswith(f"{component}:"):
                components[component] = line.replace(f"{component}:", "").strip()

    return components


def save_to_csv(error_analyses: dict[str, str], run_filename: Path) -> None:
    """
    Save error causes to a CSV file.

    Args:
        error_analyses: Dictionary mapping UUIDs to error analyses
        csv_path: Path to save the CSV file
    """
    csv_path = RESULTS_DIR / f"csv_{run_filename.replace('run_', '')}.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["UUID", "ERROR_LOCATION", "ERROR_CAUSE", "ERROR_FIX"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for uuid, analysis in error_analyses.items():
            components = extract_error_components(analysis)
            writer.writerow(
                {
                    "UUID": uuid,
                    "ERROR_LOCATION": components["ERROR_LOCATION"],
                    "ERROR_CAUSE": components["ERROR_CAUSE"],
                    "ERROR_FIX": components["ERROR_FIX"],
                }
            )

    print(f"Error causes saved to {csv_path}")


def main():
    """
    Main function to process a list of UUIDs and analyze errors.
    """

    run_filename = "run_evaluation"

    try:
        with open(RESULTS_DIR / f"{run_filename}.json") as f:
            json_run = json.load(f)
            uuids = [run["uuid"] for run in json_run["details"]]
    except Exception as e:
        print(f"Error for {run_filename}: {str(e)}")
        raise

    if not uuids:
        print("No UUIDs provided for analysis")
        return

    error_analyses = {}

    for uuid in uuids:
        try:
            print(f"Analyzing UUID: {uuid}")

            # Load memory data
            memory_data = load_memory_data(uuid)

            # Analyze error
            analysis = analyze_error(uuid, memory_data)

            # Store result
            error_analyses[uuid] = analysis

            # Save result to file
            output_path = MEMORY_DIR / f"{uuid}/error_analysis.txt"
            with open(output_path, "w") as f:
                f.write(analysis)

            print(f"Analysis for UUID {uuid} saved to {output_path}")

        except Exception as e:
            print(f"Error processing UUID {uuid}: {e}")

    # Create summary of all error causes
    if error_analyses:
        create_error_summary(error_analyses, run_filename)
        save_to_csv(error_analyses, run_filename)
    # Print completion message
    print(f"Analysis completed for {len(error_analyses)} UUIDs in {run_filename}.json")


if __name__ == "__main__":
    main()

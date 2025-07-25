# Mimosa-AI Evaluation Framework

Simplified evaluation system for scientific workflows that assesses goal achievement and workflow execution quality using LLM-based judging.

## Overview

This evaluation framework provides a streamlined approach to evaluate Mimosa-AI workflows by analyzing the raw execution data against predefined test scenarios. It uses an LLM judge to assess whether scientific goals were achieved and whether the workflow executed successfully.

## Architecture

```
sources/evaluation/
├── simple_evaluator.py     # Main evaluation engine (Evaluator class)
├── scenario_loader.py      # Test scenario management (ScenarioLoader class)
├── scenarios/              # Test scenario definitions
│   ├── protein_structure_analysis.json
│   ├── gene_expression_analysis.json
│   ├── literature_review.json
│   └── llama_cpp_installation.json
└── __init__.py            # Module exports
```

## Usage

### Command Line

**Note**: Make sure you have installed the required dependencies (see Requirements section below).

```bash
# Evaluate a workflow against a scenario (run from project root)
python3 -m sources.evaluation.simple_evaluator --workflow_id <UUID> --scenario protein_structure_analysis

# Use custom judge model
python3 -m sources.evaluation.simple_evaluator --workflow_id <UUID> --scenario gene_expression_analysis --judge_model gpt-4o

# Verbose output with full results
python3 -m sources.evaluation.simple_evaluator --workflow_id <UUID> --scenario literature_review --verbose

# Example with actual workflow ID
python3 -m sources.evaluation.simple_evaluator --workflow_id 615905648cef4896a202e1b6fe8cac85 --scenario llama_cpp_installation
```

### Programmatic Usage

```python
from sources.evaluation import Evaluator

evaluator = Evaluator(judge_model="gpt-4o")
results = evaluator.evaluate_workflow("workflow-uuid", "protein_structure_analysis")

print(f"Score: {results['score']:.2f}")
print(f"Assertions: {results['passed_assertions']}/{results['total_assertions']}")
```

## Evaluation Process

1. **Load Scenario**: Read test scenario with assertions from JSON file
2. **Load Workflow Data**: Load `state_result.json` and workflow code from UUID directory  
3. **LLM Judge Evaluation**: For each assertion, use LLM judge to analyze raw workflow data
4. **Score Calculation**: Compute percentage score (passed assertions / total assertions)
5. **Save Results**: Store evaluation results in workflow UUID directory with timestamp

## Scenario Structure

Each evaluation scenario defines:

- **Goal**: Scientific objective to achieve
- **Assertions**: Requirements to evaluate (goal achievement, result quality, tool usage, coordination)
- **Test Data**: Expected files, MCP calls, success indicators
- **Configuration**: Timeouts, required tools, judge model settings

Example assertion:
```json
{
  "id": "u1",
  "description": "User received correlation coefficient results",
  "evaluation_criteria": "Output contains specific correlation values (r) for gene-phenotype pairs"
}
```

## Available Scenarios

- **protein_structure_analysis**: PDB protein analysis with 3D visualization
- **gene_expression_analysis**: Statistical correlation analysis of genomic data
- **literature_review**: Web search and PDF analysis for research summaries

## Scoring

The evaluation produces a single **score** representing the percentage of assertions that passed (0.0 to 1.0).

Each assertion is evaluated independently by an LLM judge that:
- Analyzes the complete workflow execution data (JSON state result + workflow code)
- Returns VERDICT (TRUE/FALSE), EVIDENCE, and CONFIDENCE (0.0-1.0)
- Contributes 1 point if passed, 0 if failed

## Output Format

Evaluation results are saved as JSON files with the following structure:

```json
{
  "workflow_id": "workflow-uuid",
  "scenario_id": "scenario_name", 
  "timestamp": "2024-01-01T12:00:00",
  "goal": "workflow goal description",
  "score": 0.75,
  "passed_assertions": 6,
  "total_assertions": 8,
  "assertion_results": [
    {
      "id": "1",
      "description": "assertion description",
      "passed": true,
      "evidence": "evidence from LLM judge",
      "confidence": 0.9
    }
  ],
  "judge_model": "gpt-4o"
}
```

Results are saved in the workflow's UUID directory (`sources/workflows/{workflow_id}/`) with filename format: `evaluation_{scenario_id}_{timestamp}.json`

## Requirements

### Environment Setup

```bash
# Install required dependencies
pip install openai

# Or using uv
uv pip install openai
```

### API Configuration

- **OpenAI API key**: Set `OPENAI_API_KEY` environment variable for LLM judge evaluation
- **Workflow data access**: Requires workflow execution data in `sources/workflows/{workflow_id}/`
  - `state_result.json` - Contains workflow execution results
  - `workflow_code_{workflow_id}.py` - Contains generated workflow code

### Python Dependencies

- `openai` - For LLM judge API calls
- Standard library modules: `pathlib`, `json`, `datetime`, `argparse`, `typing`

## Extension

To add new evaluation scenarios:

1. **Create scenario file**: Add JSON definition to `sources/evaluation/scenarios/`
2. **Define assertions**: Each assertion needs `id`, `description`, and `evaluation_criteria`
3. **Test scenario**: Run evaluation against actual workflow executions
4. **Refine assertions**: Iterate based on LLM judge performance and results

### Implementation Details

The `Evaluator` class provides the core functionality:
- `evaluate_workflow(workflow_id, scenario_id)` - Main evaluation method
- `_load_workflow_data()` - Loads state_result.json and workflow code
- `_build_judge_prompt()` - Creates LLM judge prompt with raw workflow data
- `_save_results()` - Saves evaluation results to workflow directory

The system is designed to be simple and maintainable, with minimal data transformation between workflow execution and LLM judge evaluation.
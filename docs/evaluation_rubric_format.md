# Evaluation System - Rubric Format Migration

## Overview

The evaluation system has been updated to support the **ScienceAgentBench rubric format** in addition to the legacy assertion format. This allows for more structured, point-based evaluation of scientific workflows.

## Format Comparison

### Legacy Format (Assertions)
```json
{
  "id": "scenario_name",
  "goal": "Scenario description",
  "assertions": [
    {
      "id": "1",
      "description": "What should be checked",
      "evaluation_criteria": "How to evaluate it"
    }
  ]
}
```

### New Rubric Format (ScienceAgentBench)
```json
{
  "data_loading": [
    {
      "name": "Item Name",
      "description": "What should be accomplished",
      "points": 10
    }
  ],
  "data_processing": [...],
  "modeling_or_analysis_or_visualization": [...],
  "output_formatting": [...],
  "output_saving": [...],
  "total_points": 50
}
```

## Categories

The rubric format uses five standard categories from ScienceAgentBench:

1. **data_loading** - Loading and validating input data
2. **data_processing** - Data transformation and preprocessing
3. **modeling_or_analysis_or_visualization** - Core analytical tasks
4. **output_formatting** - Formatting results appropriately
5. **output_saving** - Saving outputs to correct locations

## Implementation Details

### ScenarioLoader (`sources/evaluation/scenario_loader.py`)

- **Automatic Format Detection**: Checks for `total_points` field to identify rubric format
- **Dual Format Support**: Validates both legacy and rubric formats
- **Validation**: Ensures all rubric items have required fields (name, description, points)
- **Point Validation**: Verifies that sum of item points matches declared `total_points`

### ScenarioEvaluator (`sources/evaluation/evaluator.py`)

- **Format Routing**: Automatically selects evaluation method based on format
- **Rubric Evaluation**: `_evaluate_rubric_format()` handles new format
- **Legacy Support**: `_evaluate_legacy_format()` maintains backward compatibility
- **LLM Judging**: Each rubric item is evaluated by LLM judge independently
- **Partial Credit**: Non-passing items can earn partial points based on confidence score

### Evaluation Process (Rubric Format)

1. **Load Scenario**: ScenarioLoader loads and validates the rubric
2. **Collect Items**: All items from all categories are gathered
3. **Evaluate Each Item**: LLM judge evaluates each item independently
   - Receives workflow execution state and code
   - Returns verdict (true/false), evidence, and confidence (0.0-1.0)
4. **Calculate Points**:
   - Passing items: earn full points
   - Failing items: earn `points × confidence` (partial credit)
5. **Generate Results**:
   - `earned_points`: Total points earned
   - `total_points`: Maximum possible points
   - `score`: Percentage (earned/total)
   - `item_results`: Detailed results for each item

### Return Format

For rubric-based evaluation, the return value includes:

```python
{
    'earned_points': 42.5,      # Total points earned
    'total_points': 50,          # Maximum possible points
    'score': 0.85,               # Percentage (42.5/50)
    'scenario_id': 'feature_detection'
}
```

## Migration Guide

### Converting Scenarios to Rubric Format

1. **Identify Categories**: Classify each assertion into one of the five categories
2. **Assign Points**: Determine point value for each item based on importance
3. **Rewrite Descriptions**: Make descriptions specific and actionable
4. **Calculate Total**: Sum all item points and set `total_points`
5. **Validate**: Run the test suite to ensure format is valid

### Example: feature_detection.json

**Before (Legacy):**
```json
{
  "id": "mzml_feature_detection",
  "goal": "Perform feature detection...",
  "assertions": [
    {
      "id": "1",
      "description": "mzML file was successfully loaded",
      "evaluation_criteria": "Workflow confirms file exists..."
    }
  ]
}
```

**After (Rubric):**
```json
{
  "data_loading": [
    {
      "name": "mzML File Loading",
      "description": "The program successfully loads and validates the QC_0.mzML file...",
      "points": 10
    }
  ],
  "total_points": 50
}
```

## Benefits

1. **Structured Evaluation**: Clear categorization of workflow stages
2. **Weighted Scoring**: Different tasks can have different point values
3. **Partial Credit**: Partial implementation receives proportional points
4. **Standardization**: Compatible with ScienceAgentBench benchmark
5. **Backward Compatible**: Legacy format still supported

## Testing

Run the test suite to verify rubric format functionality:

```bash
python tests/scenario_rubric_test.py
```

Tests verify:
- Scenario loading for rubric format
- Format detection (rubric vs legacy)
- Validation of rubric structure
- Point calculation accuracy

## Usage Example

```python
from sources.evaluation.evaluator import WorkflowEvaluator
from config import Config

config = Config()
evaluator = WorkflowEvaluator(config)

# Evaluate with rubric format
result = evaluator.evaluate(
    uuid="workflow-12345",
    scenario_id="feature_detection"
)

print(f"Score: {result['score']:.2%}")
print(f"Points: {result['earned_points']}/{result['total_points']}")
```

## Future Enhancements

- Support for custom category weights
- Multi-level rubric items (sub-items)
- Rubric template generation
- Automatic rubric conversion tool

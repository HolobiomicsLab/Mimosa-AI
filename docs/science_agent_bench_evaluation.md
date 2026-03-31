# ScienceAgentBench Evaluation System

## Overview

The ScienceAgentBench evaluation system provides comprehensive evaluation of Mimosa-AI's performance on scientific computing tasks using standardized metrics. This system implements the complete evaluation pipeline from the ScienceAgentBench benchmark.

## Architecture

### Components

```
sources/evaluation/
├── capsule_evaluator.py      # Main evaluation orchestrator
├── execution_sandbox.py       # Safe code execution utilities
└── codebert_scorer.py         # Code similarity scoring

sources/extensibility/
└── papers_mode.py             # Integration with autonomous mode
```

### Evaluation Flow

```
1. Task Execution (Mimosa-AI)
   ↓
2. File Transfer to Capsule
   ↓
3. CapsuleEvaluator Initialization
   ↓
4. VER Evaluation (Code Execution)
   ↓
5. SR Evaluation (Task-Specific Metrics)
   ↓
6. CBS Calculation (Code Similarity)
   ↓
7. Results Aggregation & Storage
```

## Metrics

### 1. VER (Valid Execution Rate)

**Type:** Binary (True/False)

**Definition:** Checks if the generated code executes without errors and produces expected output.

**Evaluation Process:**
- Locates Python file in capsule
- Executes with 5-minute timeout
- Validates expected output file creation
- Returns success status and error message

**Implementation:**
```python
def evaluate_exec_rate(self) -> Tuple[bool, str]:
    """
    Checks:
    1. Python file exists
    2. Executes without errors
    3. Expected output file created
    """
```

### 2. SR (Success Rate)

**Type:** Binary (True/False)

**Definition:** Evaluates if the output meets task-specific success criteria (e.g., accuracy thresholds, metric requirements).

**Evaluation Process:**
- Runs task-specific evaluation script
- Compares predictions with gold results
- Returns success based on threshold criteria

**Example Evaluation Script:**
```python
# From BBBC002_cell_count_eval.py
def eval():
    preds = pd.read_csv('pred_results/cell-count_pred.csv').values
    labels = pd.read_csv('benchmark/eval_programs/gold_results/cell-count_gold.csv')['count'].to_numpy()
    
    metric = mean_absolute_error(labels, preds)
    threshold = 30.0
    
    return int(metric <= threshold), f"MAE: {metric}"
```

### 3. CBS (CodeBERTScore)

**Type:** Float (0.0-1.0)

**Definition:** Measures semantic similarity between generated code and reference (gold) implementation using CodeBERT embeddings.

**Special Rule:** If SR=1 (task successful), CBS is automatically set to 1.0

**Calculation Method:**
- Tokenize both codes using CodeBERT tokenizer
- Generate contextual embeddings
- Compute cosine similarity matrix
- Calculate F1 score using greedy matching

**Fallback:** If transformers library unavailable, uses token-based Jaccard similarity.

### 4. API Cost

**Type:** Float (USD)

**Definition:** Tracks total API costs from LLM calls during task execution.

## Usage

### Reported Results Snapshot

The manuscript evaluates Mimosa on all `102` ScienceAgentBench tasks in `task` mode and reports the following DeepSeek-V3.2 results:

- Single-agent: `SR 38.2%`, `CBS 0.898`, `$0.05/task`
- One-shot multi-agent: `SR 32.4%`, `CBS 0.794`, `$0.38/task`
- Iterative-learning: `SR 43.1%`, `CBS 0.921`, `$1.7/task`

These figures are manuscript results, not a guaranteed console output for every local run. Actual summary metrics will vary with the selected model, run subset, and configuration.

**Evaluation on ScienceAgentBench limited to 102 tasks with learning limited to 10 iterations**

```sh
uv run main.py --science_agent_bench --csv_runs_limit 102 --max_evolve_iterations 10
```

### Output Structure

#### Per-Task Results

Saved to `runs_capsule/<capsule_name>/evaluation_results.json`:

```json
{
  "task_id": "1",
  "timestamp": "2025-10-29T10:20:00",
  "VER": true,
  "VER_message": "Execution successful, output file created",
  "SR": true,
  "SR_message": "Test accuracy: 0.89",
  "CBS": 1.0,
  "cost_usd": 0.023,
  "summary": "Task 1 Evaluation Results:..."
}
```

#### Aggregate Summary

At completion, Mimosa prints an aggregate ScienceAgentBench summary for the selected run. The exact values depend on the evaluated subset, execution mode, and model configuration.


## Configuration

### Required Dependencies

Add to `requirements.txt`:
```
transformers>=4.30.0
torch>=2.0.0
scikit-learn>=1.0.0
pandas>=1.5.0
```

### Environment Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download CodeBERT model (automatic on first use):
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
```

## Dataset Structure

ScienceAgentBench CSV requires these columns:

- `instance_id`: Unique task identifier
- `task_inst`: Task instructions
- `domain_knowledge`: Domain-specific context
- `dataset_folder_tree`: Dataset structure info
- `dataset_preview`: Sample data
- `output_fname`: Expected output file name
- `eval_script_name`: Evaluation script filename
- `gold_program_name`: Reference implementation filename

## References

- [ScienceAgentBench Paper](https://arxiv.org/abs/2410.05080)
- [CodeBERT Model](https://huggingface.co/microsoft/codebert-base)
- [Mimosa-AI Documentation](../README.md)

## License

This evaluation system is part of Mimosa-AI and follows the project's license terms.

# ScienceAgentBench Evaluation

!!! info "Two different things called 'evaluation'"
    Mimosa has two distinct evaluation systems and this page covers one
    of them.

    - **The judge / verifier** ([Evaluation pipeline](concepts/evaluation-pipeline.md))
      runs after every workflow execution and produces the **pressure
      signal** that drives workflow evolution. It writes deterministic
      Python programs that recompute claims from the workspace.
    - **ScienceAgentBench evaluation** (this page) is an **external
      benchmark grader** that compares a workflow's output file against
      a **ground-truth file shipped by the ScienceAgentBench authors**.
      It is not part of the evolutionary loop and is not written by us
      — VER / SR / CBS are the benchmark's metrics, not Mimosa's
      internal scoring.

## Overview

ScienceAgentBench is a 102-task benchmark of scientific computing tasks
released by the ScienceAgentBench authors
([paper](https://arxiv.org/abs/2410.05080)). Each task ships with:

- A task description and dataset preview.
- A **gold reference Python program** written by the benchmark authors.
- A **gold output file** computed by running that gold program.
- A **task-specific evaluation script** that decides whether a candidate
  output meets the task's acceptance criterion.

The Mimosa-side ScienceAgentBench code in
[`sources/benchmark_evaluation/`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/benchmark_evaluation/)
runs the candidate workflow, hands the produced file to the benchmark's
own evaluation machinery, and aggregates the resulting metrics. It does
not re-implement the benchmark; it bridges Mimosa to it.

## Architecture

### Components

```
sources/benchmark_evaluation/
├── csv_mode.py               # Drives evaluation over the task CSV, aggregates metrics
├── capsule_evaluator.py      # Per-task VER/SR/CBS orchestrator
├── execution_sandbox.py      # Sandboxed code execution + shared venv
├── codebert_scorer.py        # Code similarity scoring
└── science_agent_bench.py    # Dataset/eval-script/gold-program loader
```

### Evaluation Flow

```
1. Task Execution (Mimosa-AI)
   ↓
2. File Transfer to Capsule
   ↓
3. CapsuleEvaluator (sandbox built lazily; base venv reused across tasks)
   ↓
4. VER Evaluation (execute generated code; SR is conditioned on VER)
   ↓
5. SR Evaluation (task-specific eval script)
   ↓
6. CBS Calculation (CBS=1.0 when SR passes)
   ↓
7. Results Aggregation & Storage (infra failures excluded, not counted)
```

### Infra exclusion vs. genuine failure

A task is **excluded** from the metrics (VER/SR/CBS reported as `null`, shown
as `—`, counted under *Excluded (infra)*) when the eval **harness** — not the
agent's code — fails. This keeps setup problems from being mis-counted as
agent failures. Excluded conditions:

- the sandbox/venv fails to build,
- the task's `gold_results` are missing,
- a **figure-judged** task has no `OPENAI_API_KEY` / `AZURE_OPENAI_KEY` set,
- the eval script itself is missing.

Genuine agent failures still count: a generated program that crashes or does
not save its output is `VER = False`; an eval script that returns `(0, …)` is
`SR = False`.

### Sandbox environment

The sandbox mirrors the ScienceAgentBench authors' pinned eval environment
(`config_conda_env.py`) so gold/agent programs reproduce instead of failing on
version drift:

- **Python 3.10** — a hard requirement (`SANDBOX_PYTHON_VERSION`); the sandbox
  errors clearly if `python3.10` is absent. Old rdkit/deepchem-era wheels do not
  exist for 3.12, so matching 3.10 is required to reproduce their results.
- **Pinned base stack:** `numpy<2.0`, `scipy<1.14.0`, `pandas<=1.5.3`,
  `matplotlib<3.8.0`, `scikit-learn`, `torch<=2.3.0`, `tensorflow<=2.17.0`,
  `tf_keras<=2.17.0`, `rdkit<=2023.09.5`, `openai==1.54.4`.
- **Constraints file:** those version caps are written once and passed as `-c`
  to every install (base and per-task), so a program's `pipreqs`-discovered deps
  cannot pull an incompatible numpy/rdkit.
- **Handcrafted rules** (matching the authors): import-name remaps
  (`scvi`→`scvi-tools`, `skimage`→`scikit-image`, `iris`→`scitools-iris`, drop
  `benchmark`), extra deps (`biopsykit`→`mne`, `scanpy`→`scikit-misc`+`leidenalg`,
  `oggm`→`salem`+`tables`+`geopandas`), and special-case installs
  (`deepchem`→`dgl`, `DeepPurpose`→`descriptastorus`, `qsprpred`→
  `papyrus-scaffold-visualizer`+`kaleido`).

The venv is built **once per process** and reused across tasks (per-program deps
installed on top), so the first task pays the heavy install and the rest reuse it.
Without the rdkit/numpy pins, RDKit-based tasks (e.g. `deepchem`/clintox) crash
with a numpy ragged-array error; with them, the gold reproduces (VER=SR=1).

## Metrics

### 1. VER (Valid Execution Rate)

**Type:** Binary (True/False)

**Definition:** Checks if the generated code executes without errors and produces expected output.

**Evaluation Process:**
- Locates Python file in capsule
- Executes with 5-minute timeout
- Validates expected output file creation
- Returns success status and error message

The sandbox pre-creates the `pred_results/` output directory, so a program that
writes there without `mkdir` does not fail VER on its final line.

**Implementation:** `CapsuleEvaluator.evaluate_success_rate` runs the generated
program (VER) and then the task-specific eval script (SR).

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

**On failure:** if CBS cannot be computed (e.g. `transformers`/`torch` missing,
or the gold program is unavailable), the scorer raises; the evaluator records the
reason in `CBS_error` and falls back to `0.0`. That fallback is logged distinctly
so it is never mistaken for a genuine zero similarity.

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

**Run the full benchmark (all 102 tasks).** Run from the repository root where
the input datasets live under `datasets/ScienceAgentBench/datasets/` (these are
untracked and are **not** present in a git worktree — use the main checkout):

```sh
# one-shot multi-agent over all 102 tasks
uv run main.py --science_agent_bench --csv_runs_limit 102

# iterative-learning mode (adds workflow evolution)
uv run main.py --science_agent_bench --csv_runs_limit 102 --learn

# single-agent mode
uv run main.py --science_agent_bench --csv_runs_limit 102 --single_agent
```

Set `OPENAI_API_KEY` (or `AZURE_OPENAI_KEY`) first, or figure-judged tasks are
excluded rather than scored. Concurrency is controlled by `max_concurrent_eval_tasks`
in the config.

### Validating the eval pipeline against the gold solutions

`tests/brute_gold_eval.py` is a sanity check: it runs each task's **gold**
program (VER) and feeds the result through the eval script (SR). A correct
pipeline scores VER = SR = 100% on the gold. With no arguments it runs the
**full benchmark** (all 102 tasks from the CSV) using the same base packages as
Mimosa (`BASIC_PACKAGES`), so the first task pays a one-time heavy install into
the shared venv. Run from the main checkout (the input datasets are not in a
worktree):

```sh
# full benchmark — all 102 tasks
python3.12 tests/brute_gold_eval.py

# list the tasks, run nothing
python3.12 tests/brute_gold_eval.py --list

# a fast subset with a lighter base env
python3.12 tests/brute_gold_eval.py --light CogSci_pattern_high_sim_eval mountainLion3_eval

# SR-only: feed the gold output through the eval, skip executing the gold
python3.12 tests/brute_gold_eval.py --seed-only clintox_nn_eval
```

Set `OPENAI_API_KEY` (or `AZURE_OPENAI_KEY`) for figure tasks, or they are excluded.

Unit tests for the error-handling behaviour:

```sh
python -m pytest tests/test_benchmark_eval_error_handling.py
```

### Output Structure

#### Per-Task Results

Saved to `runs_capsule/<capsule_name>/evaluation_results.json`:

```json
{
  "task_id": "1",
  "timestamp": "2025-10-29T10:20:00",
  "status": "evaluated",
  "VER": true,
  "VER_message": "Execution successful, output file created",
  "SR": true,
  "SR_message": "Test accuracy: 0.89",
  "CBS": 1.0,
  "cost_usd": 0.023,
  "summary": "Task 1 Evaluation Results:..."
}
```

`status` is `"evaluated"` or `"excluded"`. An excluded task has `VER`/`SR`/`CBS`
= `null` and an `infra_error` field. A CBS that fell back to `0.0` carries a
`CBS_error` field.

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

## What this is *not*

This page does **not** describe the multi-source per-claim verifier that
drives workflow evolution. The verifier runs on every workflow execution
(benchmark or not), writes deterministic Python programs to check the
agents' claims against the workspace, and emits a coarse prompt gradient
that the mutator can act on. For that, see
[Evaluation pipeline](concepts/evaluation-pipeline.md).

When running ScienceAgentBench in `--learn` mode, both systems run: the
verifier provides the per-generation pressure signal, and the
ScienceAgentBench grader scores the final per-task output for the
benchmark report.

## References

- [ScienceAgentBench Paper](https://arxiv.org/abs/2410.05080)
- [CodeBERT Model](https://huggingface.co/microsoft/codebert-base)
- [Evaluation pipeline](concepts/evaluation-pipeline.md) — the *other*
  evaluation system in Mimosa.
- [Mimosa-AI Documentation](index.md)

## License

This evaluation system is part of Mimosa-AI and follows the project's license terms.

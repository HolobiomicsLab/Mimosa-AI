# Mimosa-AI: Quick Iterative Research Guide

**For AI Researchers Working on Self-Evolving Multi-Agent Systems**

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start for Research](#quick-start-for-research)
3. [Understanding the Architecture](#understanding-the-architecture)
4. [Evaluation Protocol](#evaluation-protocol)
5. [Key Improvement Areas](#key-improvement-areas)
6. [Research Workflow](#research-workflow)
7. [Debugging and Analysis](#debugging-and-analysis)
8. [File Structure Reference](#file-structure-reference)
9. [Common Pitfalls](#common-pitfalls)

---

## Overview

Mimosa is a **self-evolving multi-agent framework** that implements a Darwin-Gödel Machine (DGM) approach to automatically improve workflows through empirical feedback. The system operates in **task mode** for quick research iterations.

**Research Goal:** Maximize performance on ScienceAgentBench by improving the self-evolution mechanism to beat single-agent baselines.

---

## Quick Start for Research

### Prerequisites

```bash
# 1. Set up environment
cp config_default.json my_config.json
# 2. Edit my_config.json with your settings
```

### Running Quick Experiments

**Single-Agent Baseline (7 tasks):**
```bash
uv run main.py --science_agent_bench --csv_runs_limit 7 --config my_config.json --single_agent
```

**Self-Evolving Multi-Agent (7 tasks):**
```bash
uv run main.py --science_agent_bench --csv_runs_limit 7 --config my_config.json
```

For now it is advised to run for only 7 iterations and try to get maximize results.

At the end of an evaluation you should see:

                           ScienceAgentBench Metrics
--------------------------------------------------------------------------------
VER (Valid Execution Rate): 21/102 (20.6%)
SR (Success Rate): 21/102 (20.6%)
CBS (CodeBERT Score) Average: 0.781
Total API Cost: $0.0000
Average API Cost per Task: $0.0000
================================================================================


### Clean Slate Evaluation

```bash
# Remove cached workflows for unbiased evaluation
./cleanup.sh

# Run fresh evaluation
uv run main.py --science_agent_bench --csv_runs_limit 7 --config my_config.json
```

---

## Understanding the Architecture

### Four-Layer Structure

```
┌─────────────────────────────────────────────────────┐
│  Layer 0 (Optional): Planning                       │
│  - Goal → Task decomposition                        │
│  - Not used in quick research (task mode only)      │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Layer 1: Tool Discovery (Toolomics + MCP)          │
│  - Scans network for MCP servers                    │
│  - Dynamically loads available tools                │
│  - Source: sources/core/tools_manager.py            │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Layer 2: Meta-Orchestration (DGM Core)             │
│  ┌───────────────────────────────────────────────┐  │
│  │ Workflow Selection                            │  │
│  │ - Embedding similarity search                 │  │
│  │ - Source: sources/core/workflow_selection.py  │  │
│  └───────────────────────────────────────────────┘  │
│                      ↓                               │
│  ┌───────────────────────────────────────────────┐  │
│  │ Workflow Generation                           │  │
│  │ - LLM creates LangGraph workflows             │  │
│  │ - Source: sources/core/workflow_factory.py    │  │
│  │ - Prompt: sources/prompts/workflow_v7.md      │  │
│  └───────────────────────────────────────────────┘  │
│                      ↓                               │
│  ┌───────────────────────────────────────────────┐  │
│  │ Self-Improvement Loop                         │  │
│  │ - Execution → Evaluation → Mutation           │  │
│  │ - Source: sources/core/dgm.py                 │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Layer 3: Agent Execution (SmolAgent)               │
│  - Code-generating agents                           │
│  - Python-as-action paradigm                        │
│  - Source: sources/core/workflow_runner.py          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Layer 4: Evaluation (LLM Judge)                    │
│  - Generic evaluation (4 criteria)                  │
│  - Scenario evaluation (rubric-based)               │
│  - Source: sources/evaluation/evaluator.py          │
└─────────────────────────────────────────────────────┘
```

### Key Files for Research

| File | Purpose | Research Impact |
|------|---------|-----------------|
| `sources/prompts/workflow_v7.md` | **DGM workflow generation prompt** | ⭐⭐⭐⭐⭐ Critical for workflow quality |
| `sources/core/dgm.py` | **Self-improvement loop logic** | ⭐⭐⭐⭐⭐ Core DGM implementation |
| `sources/core/workflow_selection.py` | **Similarity matching for workflows** | ⭐⭐⭐⭐ Affects learning transfer |
| `sources/evaluation/evaluator.py` | **Judge system & feedback generation** | ⭐⭐⭐⭐ Quality of improvement signal |
| `sources/core/workflow_factory.py` | **Workflow code generation** | ⭐⭐⭐ Implementation details |
| `config.py` or `my_config.json` | **System configuration** | ⭐⭐⭐ Model selection, thresholds |

**Only modify these files for your research.**
**You are not allowed to change any of the LLM in the config.**
**You are not allowed to change discovery_addresses in the config they been setup already.**
**You are not allowed to change any of the runner settings in the config.**

---

## Evaluation Protocol

### ScienceAgentBench Details

- **Dataset:** 102 scientific tasks from 44 papers (4 domains)
- **Location:** `datasets/ScienceAgentBench.csv`
- **Metrics:**
  - Valid Execution Rate (VER)
  - Success Rate (SR)
  - Overall Score (0.0-1.0 from rubric evaluation)

### Scoring System

**Rubric-Based Evaluation (ScienceAgentBench):**
```python
# Categories (see sources/evaluation/evaluator.py)
1. data_loading (points vary by task)
2. data_processing (points vary by task)
3. modeling_or_analysis_or_visualization (points vary by task)
4. output_formatting (points vary by task)
5. output_saving (points vary by task)

# Score = earned_points / total_points
```

This rubric judge is not currently used but could to replace the LLM judge.

**Generic LLM Judge (4 criteria, each 0.0-1.0):**
```python
1. Goal Alignment: Did execution achieve objectives?
2. Agent Collaboration: Did agents pass data correctly?
3. Output Quality: Is output complete and well-formatted?
4. Answer Correctness: Is the result factually accurate? (optional)

# Overall Score = average of criteria scores
```

### Results Location

```
runs_capsule/
└── <timestamp>_<task_name>/
    ├── state_result.json          # Execution state + evaluation scores
    ├── workflow_<uuid>.png         # Workflow graph visualization
    ├── reward_progress.png         # DGM improvement curve
    └── <workspace files>           # All generated outputs

sources/workflows/
└── <uuid>/
    ├── workflow_code_<uuid>.py    # Generated workflow code
    ├── goal_<uuid>.txt            # Task description
    ├── original_task_<uuid>.txt   # For similarity matching
    ├── state_result.json          # Final state with scores
    └── evaluation.txt             # Judge feedback
```

---

## Key Improvement Areas

### 1. **Workflow Generation Prompt Engineering** ⭐⭐⭐⭐⭐

**File:** `sources/prompts/workflow_v7.md`

**Current Issues:**
- Agent decomposition may be too fine-grained or too coarse
- Routing logic sometimes causes infinite loops
- Tool selection not always optimal

**Research Approach:**
```bash
# 1. Create new prompt variant
cp sources/prompts/workflow_v7.md sources/prompts/workflow_v8_experiment.md

# 2. Edit the new prompt with your improvements
tool_for_edit_file sources/prompts/workflow_v8_experiment.md

# 3. Update config to use new prompt
# Edit my_config.json:
{
  "prompt_workflow_creator": "sources/prompts/workflow_v8_experiment.md"
}

# 4. Run quick evaluation
uv run main.py --science_agent_bench --csv_runs_limit 7 --config my_config.json

# 5. Compare results
python memory_explorer.py  # Analyze performance differences
```

**Key Sections to Modify:**

```markdown
## 1. Core Principles
### A. Task Decomposition
- Modify agent granularity guidelines
- Adjust "One Agent, One Job" thresholds

### B. State-Driven Routing
- Improve retry/fallback logic
- Add backtracking strategies

### C. Agent Design
- Refine completion keyword protocol
- Improve error handling patterns

## 2. Technical Specification
### Master router
- Modify routing logic
- Add new routing states beyond [SUCCESS, RETRY, FALLBACK, FAILURE]
```

### 2. **Judge System Improvement** ⭐⭐⭐⭐

**File:** `sources/evaluation/evaluator.py`

**Current Issues:**
- Judge feedback may be too generic
- Not enough actionable improvement suggestions
- Rubric items may miss critical task aspects

**Research Approach:**

```python
# Modify judge system prompt in evaluator.py

# Current location: Line ~180-200
def _get_judge_system_prompt(self) -> str:
    """Get system prompt for LLM judge."""
    return """You are an expert scientific researcher..."""

# Improvement ideas:
# 1. Add domain-specific evaluation criteria
# 2. Request more specific failure diagnostics
# 3. Include code-level analysis in feedback
# 4. Add confidence calibration
```

**Testing Judge Changes:**
```bash
# Test on single task with scenario
uv run main.py --task "Train model on Clintox dataset" \
               --scenario clintox_nn_rubric \
               --config my_config.json
```

### 3. **Workflow Selection Similarity** ⭐⭐⭐⭐

**File:** `sources/core/workflow_selection.py`

**Current Behavior:**
```python
# Default threshold: 0.8 (very strict)
candidates = self.workflow_selector.select_best_workflows(
    goal=goal,
    threshold_similary=0.8,  # Only near-identical tasks selected
    threshod_score=0.0,
)
```

**Research Question:** Does loosening similarity allow better learning transfer?

**Experiment:**
```python
# In sources/core/dgm.py, Line ~195
def select_workflow_template(self, goal, template_uuid: str = None):
    # ...
    candidates = self.workflow_selector.select_best_workflows(
        goal=goal,
        threshold_similary=0.6,  # Try: 0.5, 0.6, 0.7, 0.8
        threshod_score=0.0,
    )
```

**Ablation Study:**
```bash
# Test different thresholds
for thresh in 0.5 0.6 0.7 0.8; do
  # Modify code to use threshold
  uv run main.py --science_agent_bench --csv_runs_limit 7 \
                 --config my_config.json > results_thresh_${thresh}.log
done

Note: this example would require changing main.py to handle such parameter.
```

### 4. **Open-Ended Evolution (Branching)** ⭐⭐⭐⭐⭐

**Current Limitation:** DGM only mutates the best workflow → local optimum trap

**Proposed Solution:** Generate multiple divergent workflow variants

**Implementation Location:** `sources/core/dgm.py`, around Line 450-500

```python
# Current: Single mutation
async def recursive_self_improvement(self, runs, ...):
    # ... existing code ...
    
    # Currently:
    # 1. Select BEST workflow
    # 2. Mutate it once
    # 3. Evaluate
    
    # Proposed:
    # 1. Select TOP-K workflows (e.g., K=3)
    # 2. Generate 2 divergent mutations per workflow
    # 3. Evaluate all in parallel
    # 4. Keep best branch for next iteration
```

**Pseudo-code Addition:**
```python
# Add to dgm.py after line ~460
def generate_divergent_mutations(
    self, 
    workflow_info: WorkflowInfo, 
    num_variants: int = 2
) -> list[str]:
    """Generate multiple divergent workflow improvements.
    
    Args:
        workflow_info: Base workflow to mutate
        num_variants: Number of divergent variants to generate
        
    Returns:
        list[str]: Different improvement prompts for diversity
    """
    # Strategy 1: Different agent decompositions
    # Strategy 2: Different tool selections
    # Strategy 3: Different routing strategies
    pass
```

### 5. **DGM Memory System** ⭐⭐⭐⭐

**Current Gap:** No persistent learning of workflow patterns

**Proposed:** Cross-task pattern memory

```python
# New file: sources/core/pattern_memory.py

class PatternMemory:
    """Store and retrieve successful workflow patterns."""
    
    def __init__(self, memory_dir: str):
        self.patterns = {}
        # Pattern examples:
        # - "document_analysis" → always use multi-agent + judge
        # - "data_preprocessing" → always validate before modeling
        # - "error_handling" → backtrack 2 nodes, not 1
    
    def learn_pattern(self, task_type: str, workflow: WorkflowInfo):
        """Extract and store successful patterns."""
        pass
    
    def suggest_pattern(self, task: str) -> dict:
        """Retrieve relevant patterns for new task."""
        pass
```

**Integration Point:** `sources/core/workflow_factory.py`, Line ~200

### 6. **Failure Analysis** ⭐⭐⭐

**Research Question:** What are the systematic failure modes?

**Analysis Script:**
```python
# Create: analysis/failure_analysis.py

import json
from pathlib import Path

def analyze_failures():
    """Analyze common failure patterns in workflows."""
    workflows = Path("sources/workflows").iterdir()
    
    failures = {
        "tool_not_found": 0,
        "infinite_loop": 0,
        "syntax_error": 0,
        "timeout": 0,
        "agent_coordination_error": 0,
    }
    
    for wf in workflows:
        state = json.load(open(wf / "state_result.json"))
        # Analyze failure reasons
        # ...
    
    print(failures)

if __name__ == "__main__":
    analyze_failures()
```

---

## Research Workflow

### Standard Research Iteration

```bash
# 1. HYPOTHESIS
# Example: "Increasing agent decomposition improves success rate"

# 2. IMPLEMENTATION
# Edit: sources/prompts/workflow_v7.md
# Modify: Section "A. Task Decomposition"

# 3. BASELINE (clean slate)
./cleanup.sh
uv run main.py --science_agent_bench --csv_runs_limit 7 \
               --single_agent --config my_config.json

# 4. EXPERIMENTAL CONDITION
uv run main.py --science_agent_bench --csv_runs_limit 7 \
               --config my_config.json


# 5. ITERATE
# If improvement: expand to more tasks
# If no improvement: try different hypothesis
```

### Full Evaluation Protocol

```bash
# After promising quick results (csv_runs_limit 7)

# 1. Clean cached workflows
./cleanup.sh

# 2. Run full benchmark
uv run main.py --science_agent_bench --csv_runs_limit 103 \
               --config my_config.json \
               > full_evaluation.log 2>&1

# 3. Generate report
# Results saved in: runs_capsule/science_agent_bench_<timestamp>/
```

### A/B Testing Framework

```bash
# Compare two configurations

# Condition A: Current system
uv run main.py --science_agent_bench --csv_runs_limit 7 \
               --config config_baseline.json \
               > results_A.log

Warning: log might be extremely long.

# Condition B: Modified system
uv run main.py --science_agent_bench --csv_runs_limit 7 \
               --config config_experiment.json \
               > results_B.log

Warning: log might be extremely long.

```

---

## Debugging and Analysis

### Examining Individual Workflow

```python
# Use memory_explorer.py (already in repo)
python memory_explorer.py

# Or manually inspect:
import json
from pathlib import Path

uuid = "20260115_143415_7cb7a33b"  # Example UUID
wf_path = Path(f"sources/workflows/{uuid}")

# Load workflow state
state = json.load(open(wf_path / "state_result.json"))
print("Score:", state["evaluation"]["scenario"]["score"])
print("Answers:", state["answers"])

# Load generated code
code = open(wf_path / f"workflow_code_{uuid}.py").read()
print("Workflow code length:", len(code))

# Load judge feedback
feedback = open(wf_path / "evaluation.txt").read()
print("Judge feedback:", feedback[:500])
```

### Visualizing DGM Progress

```python
# Reward curves saved automatically
# Location: sources/workflows/<uuid>/reward_progress.png

# For custom analysis:
import matplotlib.pyplot as plt
import json

def plot_dgm_improvement(run_folder):
    """Plot improvement over DGM iterations."""
    # Load state_result.json from each iteration
    # Extract scores
    # Plot curve
    pass
```

### Common Debug Patterns

**Workflow Generation Failed:**
```bash
# Check logs
tail -f logs/mimosa.log | grep "WORKFLOW_GENERATION_ERROR"

# Common causes:
# 1. No MCP servers running → check Toolomics
# 2. Invalid workflow prompt → syntax check workflow_v7.md
# 3. LLM timeout → increase timeout in config
```

**Workflow Execution Timeout:**
```python
# In config.py or my_config.json
"runner_default_timeout": 1800  # Increase from 600 to 1800 seconds
```

**Similarity Search Not Finding Workflows:**
```python
# Debug similarity search
from sources.core.workflow_selection import WorkflowSelector

config = Config()
selector = WorkflowSelector(config)

task = "Your task here"
candidates = selector.sort_similar_workflows(
    task, 
    threshold=0.7,
    debug=True  # Prints all similarities
)
```

---

## File Structure Reference

```
Mimosa-AI/
├── main.py                          # Entry point
├── config.py                        # Config class definition
├── config_default.json              # Default configuration
├── my_config.json                   # Your custom config
│
├── sources/
│   ├── core/                        # Core system logic
│   │   ├── dgm.py                   # ⭐ DGM self-improvement loop
│   │   ├── workflow_factory.py      # Workflow code generation
│   │   ├── workflow_selection.py    # ⭐ Similarity search
│   │   ├── orchestrator.py          # Workflow execution
│   │   ├── workflow_runner.py       # Sandbox runner
│   │   ├── llm_provider.py          # LLM interface
│   │   └── planner.py               # Goal→Task decomposition
│   │
│   ├── prompts/                     # System prompts
│   │   ├── workflow_v7.md           # ⭐⭐⭐ Main workflow prompt
│   │   ├── workflow_v8.md           # Variant
│   │   └── planner_*.md             # Planner prompts
│   │
│   ├── evaluation/                  # Evaluation system
│   │   ├── evaluator.py             # ⭐ Judge system
│   │   ├── scenario_loader.py       # Rubric loader
│   │   ├── science_agent_bench.py   # Benchmark handler
│   │   └── execution_sandbox.py     # Safe execution
│   │
│   ├── workflows/                   # Generated workflows (cached)
│   │   └── <uuid>/
│   │       ├── workflow_code_<uuid>.py
│   │       ├── state_result.json
│   │       ├── goal_<uuid>.txt
│   │       └── evaluation.txt
│   │
│   ├── memory/                      # Agent execution traces
│   │   └── <uuid>/
│   │       └── task_<agent>.json
│   │
│   └── utils/                       # Utilities
│       ├── pricing.py               # Cost tracking
│       ├── visualization.py         # Plot generation
│       └── notify.py                # Pushover notifications
│
├── datasets/
│   ├── ScienceAgentBench.csv        # ⭐ Main benchmark
│   ├── scenarios/                   # Rubric definitions
│   │   └── <task>_rubric.json
│   └── scienceagentbench/           # Full dataset
│       └── datasets/
│
├── runs_capsule/                    # Evaluation results
│   └── <timestamp>_<description>/
│       ├── state_result.json
│       └── <all workspace files>
│
└── logs/
    └── mimosa.log                   # System logs
```

---

## Advanced Topics

### Custom Evaluation Metrics

```python
# Add to sources/evaluation/evaluator.py

def custom_metric(self, workflow_info: WorkflowInfo) -> float:
    """Custom domain-specific metric."""
    # Example: Measure code quality
    code_length = len(workflow_info.code)
    agent_count = workflow_info.code.count("SmolAgentFactory")
    
    # Prefer concise workflows with appropriate decomposition
    score = 1.0 / (1.0 + code_length / 1000) * (agent_count / 5)
    return min(1.0, score)
```

### Hyperparameter Tuning

```python
# Key hyperparameters in config.py

# DGM parameters
"learned_score_threshold": 0.85,      # When to stop improving
"max_learning_dgm_iterations": 5,     # Max retries per task

# Workflow selection
threshold_similary = 0.8              # In workflow_selection.py
threshod_score = 0.0                  # Minimum score for template
```

---

## Success Criteria

### Short-term Goals (1-2 weeks)

- [ ] Achieve >60% success rate on 7-task quick evaluation
- [ ] Demonstrate improvement over single-agent baseline
- [ ] Identify and fix 3 major failure modes
- [ ] Implement 1 significant improvement (prompt, judge, or selection)

### Medium-term Goals (1 month)

- [ ] Achieve >70% success rate on full ScienceAgentBench
- [ ] Implement open-ended evolution (workflow branching)
- [ ] Add pattern memory system
- [ ] Publish ablation study results

### Long-term Goals (3 months)

- [ ] Achieve >80% success rate on ScienceAgentBench
- [ ] Demonstrate generalization to new domains
- [ ] Contribute improvements back to framework
- [ ] Write research paper on self-evolution mechanisms

---

## Quick Reference Card

```bash
# Essential Commands
./cleanup.sh                                     # Clean cached workflows
uv run main.py --task "<task>" --config X       # Single task
uv run main.py --science_agent_bench --csv_runs_limit 7  # Quick eval
uv run main.py --single_agent ...               # Baseline comparison

# Key Files to Modify
sources/prompts/workflow_v7.md                  # Workflow generation
sources/core/dgm.py                             # Self-improvement loop
sources/evaluation/evaluator.py                 # Judge system
sources/core/workflow_selection.py              # Similarity threshold

# Results Location
sources/workflows/<uuid>/                       # Generated workflows
runs_capsule/<timestamp>/                       # Evaluation results
logs/mimosa.log                                 # System logs
```
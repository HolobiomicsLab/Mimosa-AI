# Mimosa-AI: Quick Iterative Research Guide

**For AI Researchers Working on Self-Evolving Multi-Agent Systems**
**Used as well for Mimosa-AI or other agentic system to conduct research on Mimosa**

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

Mimosa is a **self-evolving multi-agent framework** that implements a Iterative-Learning (IL) approach to automatically improve workflows through empirical feedback.

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
uv run main.py --science_agent_bench --csv_runs_limit 7 --config my_config.json --learn --max_evolve_iterations 10
```

It is advised to run for only 7 iterations and try to get maximize results.

At the end of an evaluation you should see:

```sh
                           ScienceAgentBench Metrics
--------------------------------------------------------------------------------
VER (Valid Execution Rate): 21/102 (20.6%)
SR (Success Rate): 21/102 (20.6%)
CBS (CodeBERT Score) Average: 0.781
Total API Cost: $100.0
Average API Cost per Task: $1
================================================================================
```


### Clean Slate Evaluation

```bash
# Remove cached workflows for unbiased evaluation
./cleanup.sh

# Run fresh evaluation
uv run main.py --science_agent_bench --csv_runs_limit 7 --config my_config.json
```

---

## Understanding the Architecture

### Five-Layer Architecture

The manuscript describes Mimosa as a five-layer architecture numbered `0` through `4`: planning, tool discovery, meta-orchestration, agent execution, and judge/evaluation. Quick research on ScienceAgentBench runs in `task` mode, so Layer `0` is bypassed during benchmarking, but it remains part of the full system design.

```
┌─────────────────────────────────────────────────────┐
│  Layer 0 (Optional): Planning                       │
│  - Goal → Task decomposition                        │
│  - Bypassed in quick research (task mode)           │
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
│  Layer 2: Meta-Orchestration                        │
│  ┌───────────────────────────────────────────────┐  │
│  │ Workflow Selection                            │  │
│  │ - Embedding similarity search                 │  │
│  │ - Source: sources/core/workflow_selection.py  │  │
│  └───────────────────────────────────────────────┘  │
│                      ↓                              │
│  ┌───────────────────────────────────────────────┐  │
│  │ Workflow Generation                           │  │
│  │ - LLM creates LangGraph workflows             │  │
│  │ - Source: sources/core/workflow_factory.py    │  │
│  │ - Prompt: sources/prompts/workflow_v8.md      │  │
│  └───────────────────────────────────────────────┘  │
│                      ↓                              │
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
│  Layer 4: Judge / Evaluation                        │
│  - LLM judge with structured feedback               │
│  - Scenario evaluation (rubric-based)               │
│  - Source: sources/evaluation/evaluator.py          │
└─────────────────────────────────────────────────────┘
```

### Results Location

```
runs_capsule/
└── <timestamp>_<task_name>/
    ├── state_result.json          # Execution state + evaluation scores
    ├── workflow_<uuid>.png         # Workflow graph visualization
    ├── reward_progress.png         # improvement curve
    └── <workspace files>           # All generated outputs

sources/workflows/
└── <uuid>/
    ├── workflow_code_<uuid>.py    # Generated workflow code
    ├── goal_<uuid>.txt            # Task description
    ├── original_task_<uuid>.txt   # For similarity matching
    ├── state_result.json          # Final state with scores
    └── evaluation.txt             # Judge feedback
```

These folders form the main audit trail for a run: `sources/workflows/<uuid>/` captures the generated workflow, evaluation artifacts, and iteration history, while `runs_capsule/` preserves the final workspace snapshot copied from Toolomics. Use `memory_explorer.py <uuid>` when you want to replay an execution trace interactively.

---

## Research Workflow

### Standard Research Iteration

```bash
# 1. HYPOTHESIS
# Example: "Increasing agent decomposition improves success rate"

# 2. IMPLEMENTATION
# Edit: sources/prompts/workflow_v8.md
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
uv run main.py --science_agent_bench --csv_runs_limit 102 \
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


### Common Debug Patterns

**Workflow Generation Failed:**
```bash
# Check logs
tail -f logs/mimosa.log | grep "WORKFLOW_GENERATION_ERROR"

# Common causes:
# 1. No MCP servers running → check Toolomics
# 2. Invalid workflow prompt instructions → syntax check workflow_xxx.md
# 3. LLM timeout → increase timeout in config
# 4. Sometime LLM fail to generate valid workflow -> Try again. Use Claude Opus to exclude model specific issues.
```

**Workflow Execution Timeout:**
```python
# In config.py or my_config.json
"runner_default_timeout": 1800  # Increase from 600 to 1800 seconds
```


---

### Parameter Tuning

```python
# Key hyperparameters in config.py

# Learning parameters
"learned_score_threshold": 0.9,      # When to stop improving
"max_learning_evolve_iterations": 5,     # Max retries per task

# Workflow selection
threshold_similary = 0.8              # In workflow_selection.py
threshod_score = 0.0                  # Minimum score for template
```

---

## Quick Reference Card

```bash
# Essential Commands
./cleanup.sh                                     # Clean cached workflows
uv run main.py --task "<task>" --config X       # Single task
uv run main.py --science_agent_bench --csv_runs_limit 7  # Quick eval
uv run main.py --single_agent ...               # Baseline comparison

# Key Files to Modify
sources/prompts/workflow_v8.md                  # Workflow generation
sources/core/dgm.py                             # Self-improvement loop
sources/evaluation/evaluator.py                 # Judge system
sources/core/workflow_selection.py              # Similarity threshold

# Results Location
sources/workflows/<uuid>/                       # Generated workflows
runs_capsule/<timestamp>/                       # Evaluation results
logs/mimosa.log                                 # System logs
```

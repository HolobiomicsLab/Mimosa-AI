# Contributing to Mimosa-AI

> Note: This file is the technical contribution guide. For licensing/CLA policy (including the required `CLA Check` status before merge), use the repository-root `CONTRIBUTING.md`.  
> CLA contact: dr20.spv@listes.cnrs.fr

## Table of Contents
1. [Project Philosophy](#project-philosophy)
2. [Architecture Overview](#architecture-overview)
3. [Directory Structure](#directory-structure)
4. [Core Components](#core-components)
5. [Execution Flow](#execution-flow)
6. [Development Guide](#development-guide)
7. [Adding New Features](#adding-new-features)
8. [Testing & Evaluation](#testing--evaluation)

---

## Project Philosophy

### Vision
Mimosa-AI is an **autonomous AI-scientist framework** designed to reproduce published scientific findings and execute end-to-end research pipelines with minimal human intervention. It provides academics with an open, transparent alternative to closed corporate systems for scientific discovery.

### Core Principles

#### 1. **Polymorphic Multi-Agent Architecture**
Rather than forcing all tasks through fixed pipelines, Mimosa synthesizes specialized multi-agent workflows **on-demand** for each unique task. This polymorphic approach allows:
- Dynamic adaptation to domain-specific requirements
- Custom tool/agent combinations per task
- Evolutionary optimization over time

#### 2. **Goal-Task Decomposition**
The system operates on a hierarchical decomposition pattern:
```
HIGH-LEVEL GOAL (e.g., "Reproduce paper X")
    ↓
PLANNER decomposes into multiple TASKS with dependencies
    ↓
Each TASK triggers workflow synthesis
    ↓
Specialized WORKFLOW created and executed for each task
    ↓
Results aggregated into CAPSULE with full artifacts
```

#### 3. **Iterative-Learning (IL) Self-Improvement**
Inspired by evolutionary algorithms and Gödel machines, the system:
- Maintains a library of successful workflow patterns
- Learns from past executions via similarity matching
- Validates improvements before integration (safety principle)
- Optimizes its own workflow generation over time
- Visualizes reward progress through iterations

#### 4. **Tool Discovery & Integration**
Uses MCP (Model Context Protocol) for:
- Automatic discovery of available tools on local network
- Dynamic tool invocation in generated workflows
- Seamless integration with lab instruments, web services, data analysis tools
- Toolhive integration for centralized tool repositories

---

---

## Directory Structure

```
mimosa-ai/
├── config.py                           # Configuration management
├── main.py                             # Entry point & mode selection
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Project metadata
│
├── sources/
│   ├── core/                           # Core orchestration logic
│   │   ├── orchestrator.py             # Workflow execution manager
│   │   ├── planner.py                  # Goal → Task decomposition
│   │   ├── dgm.py                      # Iterative-Learning (learning loop)
│   │   ├── workflow_factory.py         # Workflow synthesis from LLM
│   │   ├── workflow_runner.py          # Code execution runtime
│   │   ├── workflow_selection.py       # Similarity-based template matching
│   │   ├── workflow_info.py            # Workflow metadata & state
│   │   ├── llm_provider.py             # Multi-provider LLM abstraction
│   │   ├── tools_manager.py            # MCP tool discovery & management
│   │   ├── improvement_validator.py    # Improvement validation
│   │   └── schema.py                   # Data classes (IndividualRun, Task, Plan, etc.)
│   │
│   ├── evaluation/                     # Performance assessment
│   │   ├── evaluator.py                # Generic/Scenario evaluation
│   │   ├── capsule_evaluator.py        # ScienceAgentBench metrics (VER/SR/CBS)
│   │   ├── codebert_scorer.py          # Semantic code similarity
│   │   ├── execution_sandbox.py        # Safe code execution environment
│   │   ├── scenario_loader.py          # Load benchmark scenarios
│   │   └── science_agent_bench.py      # ScienceAgentBench dataset integration
│   │
│   ├── extensibility/                  # Alternative execution modes
│   │   ├── csv_mode.py                 # Batch evaluation on CSV datasets
│   │   ├── human_mode.py               # Interactive manual CLI
│   │   └── text_to_speech.py           # Text to speech 
│   │
│   ├── modules/                        # Reusable code snippets as pre-fabricated part of workflow
│   │   ├── state_schema.py             # Workflow state template
│   │   └── smolagent_factory.py        # SmolAgent factory template
│   │
│   ├── utils/                          # Utility functions
│   │   ├── pricing.py                  # LLM pricing calculation
│   │   ├── logging.py                  # logging
│   │   ├── notify.py                   # Push notifications (Pushover)
│   │   ├── transfer_toolomics.py       # Workspace file management for transfer from/to toolomics
│   │   ├── planner_visualization.py    # Real-time plan visualization
│   │   └── precheck.py                 # Environment validation
│   │
│   ├── memory/                         # Execution memory & history
│   │   └── (llm call cache, memory traces)
│   │
│   └── prompts/                        # LLM system prompts
│       ├── planner_reproduction.md     # Planner system prompt
│       ├── workflow_v7.md              # Workflow generation prompt
│       └── (other specialized prompts)
│
├── sources/workflows/                  # Generated workflow storage
│   └── <uuid>/                         # Per-execution folders
│       ├── workflow.py                 # Generated agent code
│       ├── state_result.json           # Execution state & results
│       ├── memory/                     # Agent memory traces
│       ├── reward_progress.png         # learning progress visualization
│       └── ...
│
├── runs_capsule/                       # Results storage
│   └── <capsule_name>/                 # Per-execution capsule
│       ├── workflow.py                 # Final workflow code
│       ├── results/                    # Output artifacts
│       ├── logs/                       # Execution logs
│       ├── evaluation_results.json     # Metrics & evaluation
│       └── ...
│
├── datasets/                           # Test & benchmark data
│   ├── our_benchmark.csv               # Custom benchmark
│   ├── paper_bench.csv                 # OpenAI Paper Bench
│   ├── ScienceAgentBench.csv           # ScienceAgentBench dataset
│   └── scenarios/                      # Scenario rubrics for evaluation
│       └── <scenario_name>.json        # Individual scenario definitions
│
├── docs/                               # Documentation
│   ├── DEVELOPER_GUIDE.md              # This file
│   ├── science_agent_bench_evaluation.md
│   ├── diagrams/                       # Architecture diagrams
│   └── images/                         # Documentation images
│
└── tests/                              # Test suite
    ├── evaluator_test.py
    ├── scenario_rubric_test.py
    └── ...
```

---

## Core Components

### 1. **Planner** (`sources/core/planner.py`)
**Purpose**: Decompose high-level goals into executable task plans

**Key Responsibilities**:
- Parse goals and generate multi-step execution plans
- Track dependencies between tasks
- Validate task I/O requirements
- Maintain task history and knowledge context
- Support human-in-the-loop plan validation

---

### 2. **Iterative-Learning (IL)** (`sources/core/dgm.py`)
**Purpose**: Execute and iteratively improve task solutions through self-optimization

**Key Responsibilities**:
- Synthesize workflows for new tasks
- Look up cached successful workflows for similar tasks
- Propose improvements to workflows
- Validate improvements before integration
- Maintain reward progress metrics
- Generate learning visualizations

**Learning Loop**:
```
1. TASK RECOGNITION
   ├─ Calculate similarity to past tasks
   ├─ Look up historical workflows
   └─ Check success threshold (0.85)
       ├─ If found → Return cached result
       └─ If not → Create new workflow

2. WORKFLOW SYNTHESIS
   ├─ Call WorkflowFactory to generate code
   ├─ Execute workflow via Orchestrator
   └─ Evaluate results

3. IMPROVEMENT VALIDATION (if learning_mode)
   ├─ Propose modifications to workflow
   ├─ Re-execute modified workflow
   ├─ Validate improvement (reward delta)
   └─ Accept/reject based on threshold

4. ITERATE (up to max_depth)
   └─ Repeat steps 2-3 until target score reached
```

---

### 3. **WorkflowFactory** (`sources/core/workflow_factory.py`)
**Purpose**: Synthesize specialized multi-agent workflows for tasks

**Key Responsibilities**:
- Generate LLM prompts for workflow creation
- Orchestrate LLM calls to synthesize Python code
- Extract and validate generated workflow code
- Assemble workflows with necessary modules/dependencies
- Provide system prompt with available tools
- Save in sources/workflows folder.

**Workflow Folder Structure**:
```python
# Auto-generated workflow contains:
# - State schema (from state_schema.py)
# - Tool client initialization (from tools_manager)
# - Multi-agent orchestration (SmolagentFactory)
# - Main execution loop
# - State serialization (to state_result.json)
```

---

### 4. **Orchestrator** (`sources/core/orchestrator.py`)
**Purpose**: Manage workflow execution and tool integration

**Key Responsibilities**:
- Generate multi-agent workflow tailored or mutate existing one.
- Install workflow dependencies
- Execute workflows in sandboxed environment
- Monitor execution and stream output
---

### 5. **LLMProvider** (`sources/core/llm_provider.py`)
**Purpose**: Abstraction layer for multi-provider LLM access

**Supported Providers**:
- Anthropic Claude (via LiteLLM)
- OpenAI (via LiteLLM)
- DeepSeek (via LiteLLM)
- Hugging Face
- Local models via OpenRouter

**Key Features**:
- Unified interface across providers
- Request caching for cost reduction
- Retry logic with exponential backoff
- Token counting and cost tracking
- Reasoning token support (Claude)

---

### 6. **ToolsManager** (`sources/core/tools_manager.py`)
**Purpose**: Discover and manage MCP-based tools

**Key Responsibilities**:
- Auto-discover MCP servers on network
- Extract tool definitions and schemas
- Generate tool integration code for workflows
- Provide tool client instantiation code
- Handle tool registration

---

### 7. **Evaluation System** (`sources/evaluation/`)

#### BaseEvaluator & GenericEvaluator
- Generic evaluation using LLM judge
- Extracts scores from judge output
- Saves results to workflow state

#### ScenarioEvaluator
- Task-specific rubric-based evaluation
- Supports both legacy and rubric formats
- Evaluates assertions with detailed prompts

#### CapsuleEvaluator (ScienceAgentBench)
- **VER** (Valid Execution Rate): Code executes without errors
- **SR** (Success Rate): Output meets task-specific criteria
- **CBS** (CodeBERT Score): Semantic similarity to reference implementation
- Cost tracking and aggregation

---

## Execution Flow

### Mode 1: Goal Mode (Multi-step Planning)
```
main.py --goal "Reproduce paper X"
    ↓
Planner.start_planner(goal)
    ├─ Generate multi-step plan
    ├─ Request human approval
    └─ For each step:
        ├─ start_dgm(step_task)
        │   ├─ Check workflow cache
        │   ├─ Synthesize workflow (if new)
        │   ├─ Execute workflow
        │   ├─ Evaluate results
        │   └─ Optionally iterate for improvement
        ├─ Store results
        └─ Continue to next step
    ↓
Generate Capsule with results
```

### Mode 2: Task Mode
```
main.py --task "Task description" --learn
    ↓
start_dgm(task, learning_mode=True, max_iterations=10)
    ├─ Check workflow cache (similarity > 0.8)
    ├─ If found & successful: use cached workflow as inspiration
    ├─ Else: Synthesize workflow
    ├─ Execute and evaluate
    ├─ If learning_mode:
    │   ├─ Propose improvements
    │   ├─ Validate improvements
    │   └─ Iterate until threshold reached
    └─ Return best result
    ↓
Generate Capsule with results
```
---

## Testing & Evaluation

### Running Tests

```bash
# All tests
python -m pytest tests/

# Specific test
python -m pytest tests/evaluator_test.py

# With verbose output
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=sources
```

### Evaluation Modes

```bash
# papers dataset evaluation
python main.py --papers datasets/our_benchmark.csv --csv_runs_limit 10 --config my_config.json
# ScienceAgentBench
uv run main.py --science_agent_bench --csv_runs_limit 42 --config my_config.json
```

### Monitoring & Debugging

**Inspect Workflow State**:
```bash
# View generated workflow code
cat sources/workflows/<uuid>/workflow.py

# View execution results
cat sources/workflows/<uuid>/state_result.json

# View memory traces
ls sources/workflows/<uuid>/memory/
```

**Pushover Notifications** (optional):
1. Create Pushover account at pushover.net
2. Set PUSHOVER_USER and PUSHOVER_TOKEN
3. Receive notifications on task completion/failure

---

## Contributing Guidelines

### Before Submitting PR
1. ✅ Run tests: `pytest tests/`
2. ✅ Format code: Follow existing style
3. ✅ Add docstrings
4. ✅ Update relevant documentation

### PR Description Should Include
- **Problem**: What issue does this solve?
- **Solution**: How does it solve it?
- **Testing**: How was it tested?
- **Backwards Compatibility**: Any breaking changes?

---

## Questions & Support

Raise an **Issue** for any question or support request.


# Mimosa-AI 🔬🤖

**An open framework for autonomous AI-driven science**

Mimosa is an automated AI scientist framework designed to reproduce scientific findings and enable autonomous research.
Its mission is to provide a modular and open alternative to big tech initiatives, empowering academic researchers with next-generation AI tools for scientific discovery.

**Objectives**

- **Reproduce scientific research** with reliability and transparency

- **Enable autonomous research** by generating new hypotheses and insights

---


## Installation

## Prerequisites
- Python 3.10 or higher
- pip3 package manager

## ▶ Install & Run

### Step 1: Environment Setup

**Option A: Using pip**
```sh
# Create and activate virtual environment
python3 -m venv .venv
source mimosa-env/bin/activate  # On Windows: mimosa-env\Scripts\activate
```

**Option B: Using uv (faster alternative)**
```sh
# Install uv if not already installed
pip install uv

# Create and activate virtual environment
uv venv .venv
```

Configure your `.env` file with the following API keys:
- `HF_TOKEN`: Your Hugging Face token **OR** `DEEPSEEK_API_KEY` Your deepseek api key, depending on your provider.
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `LANGFUSE_PUBLIC_KEY` & `LANGFUSE_PRIVATE_KEY`: Optional, for [telemetry monitoring](https://huggingface.co/docs/smolagents/tutorials/inspect_runs)

### Step 2: Install Dependencies

First go into the `mimosa` folder
```sh
cd mimosa
```

**Using pip:**
```sh
pip3 install -r requirements.txt
```

**Using uv (better):**
```sh
uv pip install -r requirements.txt
```

### Step 3: Start MCP Server

Launch the toolomics MCP server following the instructions at [HolobiomicsLab/toolomics](https://github.com/HolobiomicsLab/toolomics) on a port range of your choice: (eg: 5000-5100)

### Step 4: Edit Mimosa-AI config

Edit the `config.py` file of Mimosa with your configuration

most importantly you probably need to change these:

```sh
# Toolomics workspace/ for Mimosa
self.workspace_dir = "/Users/holobiomicslab/Documents/repository/toolomics/workspace"
# LLMs choices
self.planner_llm_model: str = "anthropic/claude-opus-4-1-20250805"
self.prompts_llm_model: str = "anthropic/claude-opus-4-1-20250805"
self.workflow_llm_model: str = "anthropic/claude-opus-4-1-20250805"
self.smolagent_model_id: str = "anthropic/claude-haiku-4-5-20251001" # haiku is cheaper for agent
# address for MCPs discovery
self.discovery_addresses: list[AddressMCP] = [
    AddressMCP(ip="0.0.0.0", port_min=5000, port_max=5200)
    #AddressMCP(ip="xxx.xx.xx.xx", port_min=5000, port_max=5200) # add another computer with MCP tools
]
```

### Step 5: Run Mimosa-AI

```sh
python3 main.py --goal "Your objective here"
# or with uv
uv run main.py --goal "Your objective here"
```

> **Note**: Remember to activate your virtual environment (`source mimosa-env/bin/activate`) before running Mimosa-AI in future sessions.

## ▶ Command Line Arguments

Mimosa-AI supports various command line arguments to customize execution:

### Execution Modes
- `--goal GOAL`: Specify your research objective, research reproduction or scientific question (planner mode)
- `--task TASK`: Run & learn how to do a single task (Litterature review, code installation, dataset download)
- `--manual`: Interact & use Mimosa tools (from Toolomics MCPs) using a CLI interface. Allow to debug MCPs and put yourself in Mimosa shoes.
- `--papers`: Automated run on CSV of research papers. Will load a CSV dataset from `datasets/` containing a list of paper and prompt for Mimosa (such as reproduce the paper X ...), will automatically run Mimosa in goal mode for every papers in the csv dataset.
- `--scenario`: Specify scenario to evaluate Mimosa on.

### Evaluation & Performance
- `--learn`: enable learning mode, use DGM to try to get the best score on a task.
- `--max_dgm_iterations N`: Maximum number of DGM retry to learn a task (default: 3 in task mode, 1 in goal mode).
- `--csv_runs_limit`: Limit number of CSV to evaluate on.

**Example:**

**Normal usage, try to accomplish a goal.**
```sh
uv run main.py --goal "You are assigned the paper Dual Aggregation Transformer for Image Super-Resolution (https://arxiv.org/pdf/2306.00306) to replicate. Attempt to reproduce the experiments and compare the results."
```

**Single task mode, no long-term planning.**
```sh
 uv run main.py --task " Train a multitask model on the Clintox dataset to predict a drug's toxicity and FDA approval status..." --judge 
```

**Evaluation on openai paper bench (https://arxiv.org/abs/2504.01848)**
```sh
 uv run main.py --papers datasets/paper_bench.csv --csv_runs_limit 20  --learn 
```

**Evaluation on subset of scienceAgentBench (https://arxiv.org/pdf/2410.05080) with learning mode enabled.**
```sh
uv run main.py --science_agent_bench --csv_runs_limit 10 --max_dgm_iterations 10 --learn
```

> **Note**: Requires Toolomics to be installed and MCP servers to be running.

## ▶ Core Architecture

Mimosa-AI uses a **polymorphic meta-agent system** that dynamically synthesizes specialized workflows for scientific tasks. Rather than forcing tasks through fixed pipelines, the system composes custom multi-agent architectures on-demand and learns from execution patterns to optimize future performance.

The system operates on an "agent-within-agent" pattern:
- Goals decompose into learnable tasks
- Each task triggers the synthesis of a specialized multi-agent workflow
- Successful workflow patterns are retained and refined over time
- The system continuously optimizes its own architecture through execution feedback

## Task vs Goal philosophy

**Goal**: A high-level scientific objective requiring multiple distinct complex tasks to reach.
- Example: "Develop a machine learning model to predict protein-ligand binding affinity" or "Try to reproduce the research paper X and compare the experimental results".

**Task**: A granular, repeatable operation frequently encountered across different goals.
- Examples: "literature review on topic X", "download dataset from source Y", "implement algorithm Z"

## ▶ System Architecture

### Layer 0: Strategic Planning
- Decomposes goals into executable task sequences
- Maintains adaptive execution roadmap
- Adjusts plans based on Layer 2 workflow performance

### Layer 1: Meta-Orchestration
- **Dynamic Workflow Synthesis**: Advanced LLMs (e.g., claude-3.7) generate task-specific multi-agent architectures
- **Architecture Search**: Designs custom agent topologies rather than applying generic pipelines
- **Pattern Recognition**: Identifies structural similarities across tasks to better workflow generation over time

### Layer 2: Workflow Execution (LangGraph)
- Implements workflows as directed graphs with heterogeneous nodes:
  - **SmolAgent Nodes**: Autonomous code-generating agents for complex reasoning
  - **Deterministic Nodes**: Validation, transformation, and control logic
- **Graph Flexibility**: Supports arbitrary agent topologies (sequential, conditional, cyclical)
- **State Management**: Maintains workflow context across distributed execution

### Layer 3: Agent Runtime (SmolAgent)
- Code-generating agents operating in action-observation loops
- **Tool-as-Code Paradigm**: Agents generate Python to interact with tools
- **Iterative Refinement**: Continues execution until success criteria met or failure threshold reached

### Layer 4: Tool Ecosystem (MCP)
- Extensible tool primitives built on Model Context Protocol
- **Distributed Execution**: Tools run on HPC clusters, lab instruments, cloud infrastructure
- **Protocol Standardization**: MCP enables seamless client-server tool interaction
- **Horizontal Scalability**: Add new tools without modifying core system

### ▶ (Experimental) Self-Improvement on task

The system implements a **Darwinian inspired evolution** inspired by Gödel machine principles:

1. **Goal Decomposition** (Layer 0): High-level scientific goal → ordered list of tasks
   - "Develop binding affinity model" → [literature review, dataset acquisition, feature engineering, model implementation, validation]

2. **Task Recognition** (Layer 1): For each task, the system:
   - Searches its workflow library for similar historical tasks
   - If found: Uses best-performing workflow as **template**, adapting for current context
   - If novel: Synthesizes new workflow from scratch orchestrator

3. **Evolutionary Optimization**: Over time, the system:
   - Maintains multiple workflow variants per task type
   - Selects high-performing workflows based on success metrics (speed, accuracy, cost)
   - Mutates/recombines successful patterns to explore architecture space

4. **Self-Improvement Mechanism**:
   - The system can propose modifications to its own workflow generation logic
   - Performance improvements are validated before integration (Gödel machine principle)
   - Meta-learning: Learns how to generate better workflows from execution history

## ▶ Phone notification setup

Get real-time updates about Mimosa's status directly on your phone using Pushover.

## Setup Steps

1. **Register for Pushover**
   - Go to [pushover.net](https://pushover.net/) and create an account
   - After registration, you'll receive your **User Key**

2. **Create an Application**
   - In your Pushover dashboard, click "Create an Application/API Token"
   - Name it (e.g., "Mimosa") and create it
   - Copy your **API Token/Key**

3. **Configure Environment Variables**
```bash
   export PUSHOVER_USER="your_user_key_here"
   export PUSHOVER_TOKEN="your_api_token_here"
```

4. **Install Pushover on your phone***

Download pushover from play/app store and login.

## 📈 Telemetry Setup

Monitor and debug your AI agents with real-time observability dashboards.

### Quick Start

1. **Deploy Langfuse locally**:
    ```sh
    git clone https://github.com/langfuse/langfuse.git
    cd langfuse
    docker compose up -d
    ```

2. **Configure environment variables** in your `.env` file:
    ```env
    LANGFUSE_PUBLIC_KEY=your_public_key
    LANGFUSE_PRIVATE_KEY=your_private_key
    ```

3. **Access the dashboard** at `http://localhost:3000` while Mimosa-AI is running

### What You'll See

The telemetry dashboard provides:
- **Agent Execution Traces**: Step-by-step workflow visualization
- **Performance Metrics**: Response times and success rates
- **Error Debugging**: Detailed failure analysis
- **Resource Usage**: Token consumption and API calls

![Telemetry Dashboard](https://langfuse.com/images/cookbook/integration-smolagents/smolagent_example_trace.png)

> **Note**: Telemetry is optional but highly recommended for debugging.
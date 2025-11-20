# Mimosa-AI

**An open framework for autonomous AI-driven science**

Mimosa is an automated AI scientist framework designed to reproduce scientific findings and enable autonomous research. Its mission is to provide a modular and open alternative to big tech initiatives, empowering academic researchers with next-generation AI tools for scientific discovery.

## Objectives

- Reproduce scientific research with reliability and transparency
- Enable autonomous research by generating new hypotheses and insights

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip3 package manager

### Step 1: Environment Setup

Choose one of the following options:

**Option A: Using pip**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Option B: Using uv (faster alternative)**
```bash
pip install uv
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 2: Configure Environment Variables

Create a `.env` file in the project root with your API keys:

```env
HF_TOKEN=your_hugging_face_token          # OR use DEEPSEEK_API_KEY
ANTHROPIC_API_KEY=your_anthropic_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key    # Optional
LANGFUSE_PRIVATE_KEY=your_langfuse_private_key  # Optional
```

**Provider Options:**
- `HF_TOKEN`: Hugging Face token for LLM access
- `DEEPSEEK_API_KEY`: DeepSeek API key (alternative to HF_TOKEN)
- `ANTHROPIC_API_KEY`: Anthropic API for Claude models
- `LANGFUSE_*`: Optional telemetry keys for monitoring (see [Telemetry Setup](#telemetry-setup))

### Step 3: Install Dependencies

Navigate to the project directory and install dependencies:

```bash
cd mimosa
pip3 install -r requirements.txt
# OR with uv:
uv pip install -r requirements.txt
```

### Step 4: Launch MCP Server

Start the Toolomics MCP server following the instructions at [HolobiomicsLab/toolomics](https://github.com/HolobiomicsLab/toolomics).

Configure the server to run on a port range (e.g., 5000-5100).

### Step 5: Configure Mimosa-AI

Edit `config.py` with your settings. Key configuration parameters:

```python
# Toolomics workspace directory
self.workspace_dir = "/path/to/toolomics/workspace"

# LLM Model Selection
self.planner_llm_model = "anthropic/claude-opus-4-1-20250805"
self.prompts_llm_model = "anthropic/claude-opus-4-1-20250805"
self.workflow_llm_model = "anthropic/claude-opus-4-1-20250805"
self.smolagent_model_id = "anthropic/claude-haiku-4-5-20251001"

# MCP Server Discovery
self.discovery_addresses = [
    AddressMCP(ip="0.0.0.0", port_min=5000, port_max=5200)
    # Add additional MCP servers from other machines as needed
]
```

### Step 6: Run Mimosa-AI

```bash
python3 main.py --goal "Your objective here"
# OR with uv:
uv run main.py --goal "Your objective here"
```

> **Note:** Remember to activate your virtual environment before running Mimosa-AI in future sessions.

---

## Command Line Arguments

### Execution Modes

| Argument | Description |
|----------|-------------|
| `--goal GOAL` | Specify a high-level research objective, paper reproduction, or scientific question (planner mode) |
| `--task TASK` | Execute a single task: literature review, code installation, dataset download |
| `--manual` | Interactive CLI mode to debug MCPs and test Mimosa tools directly |
| `--papers CSV` | Batch evaluation on a CSV dataset containing research papers and prompts |
| `--scenario SCENARIO` | Run evaluation on a specific scenario |

### Learning & Optimization

| Argument | Description |
|----------|-------------|
| `--learn` | Enable learning mode using DGM to optimize task performance |
| `--max_dgm_iterations N` | Maximum DGM iterations for learning (default: 3 for tasks, 1 for goals) |
| `--csv_runs_limit N` | Limit number of CSV entries to evaluate |

### Examples

**Standard usage - accomplish a goal:**
```bash
uv run main.py --goal "Reproduce the experiments from 'Dual Aggregation Transformer for Image Super-Resolution' (https://arxiv.org/pdf/2306.00306) and compare results."
```

**Single task mode - no long-term planning:**
```bash
uv run main.py --task "Train a multitask model on the Clintox dataset to predict drug toxicity and FDA approval status" --judge
```

**Batch evaluation - OpenAI Paper Bench:**
```bash
uv run main.py --papers datasets/paper_bench.csv --csv_runs_limit 20 --learn
```

**ScienceAgentBench evaluation with learning:**
```bash
uv run main.py --science_agent_bench --csv_runs_limit 10 --max_dgm_iterations 10 --learn
```

> **Note:** Requires Toolomics to be installed and MCP servers to be running.

---

## Architecture

### System Overview

Mimosa-AI uses a **polymorphic meta-agent system** that dynamically synthesizes specialized workflows for scientific tasks. Rather than forcing tasks through fixed pipelines, the system composes custom multi-agent architectures on-demand and learns from execution patterns to optimize future performance.

The system operates on an **agent-within-agent** pattern:
- Goals decompose into learnable tasks
- Each task triggers synthesis of a specialized multi-agent workflow
- Successful workflow patterns are retained and refined over time
- The system continuously optimizes its own architecture through execution feedback

### Goal vs Task Philosophy

**Goal:** High-level scientific objective requiring multiple distinct complex tasks
- *Example:* "Develop a machine learning model to predict protein-ligand binding affinity"
- *Example:* "Reproduce research paper X and compare experimental results"

**Task:** Granular, repeatable operation frequently encountered across different goals
- *Example:* "Conduct literature review on topic X"
- *Example:* "Download dataset from source Y"
- *Example:* "Implement algorithm Z"

### Layered Architecture

**Layer 0: Strategic Planning**
- Decomposes goals into executable task sequences
- Maintains adaptive execution roadmap
- Adjusts plans based on workflow performance feedback

**Layer 1: Meta-Orchestration**
- Dynamic Workflow Synthesis: Advanced LLMs generate task-specific multi-agent architectures
- Architecture Search: Designs custom agent topologies rather than applying generic pipelines
- Pattern Recognition: Identifies structural similarities across tasks for improved generation

**Layer 2: Workflow Execution (LangGraph)**
- Implements workflows as directed graphs with heterogeneous nodes
  - SmolAgent Nodes: Autonomous code-generating agents for complex reasoning
  - Deterministic Nodes: Validation, transformation, and control logic
- Graph Flexibility: Supports arbitrary agent topologies (sequential, conditional, cyclical)
- State Management: Maintains workflow context across distributed execution

**Layer 3: Agent Runtime (SmolAgent)**
- Code-generating agents operating in action-observation loops
- Tool-as-Code Paradigm: Agents generate Python to interact with tools
- Iterative Refinement: Continues execution until success criteria met or failure threshold reached

**Layer 4: Tool Ecosystem (MCP)**
- Extensible tool primitives built on Model Context Protocol
- Distributed Execution: Tools run on HPC clusters, lab instruments, cloud infrastructure
- Protocol Standardization: MCP enables seamless client-server tool interaction
- Horizontal Scalability: Add new tools without modifying core system

### Self-Improvement Mechanism (Experimental)

The system implements a Darwinian-inspired evolution approach based on Gödel machine principles:

1. **Goal Decomposition** (Layer 0): High-level scientific goal → ordered list of tasks
   - *Example:* "Develop binding affinity model" → [literature review, dataset acquisition, feature engineering, model implementation, validation]

2. **Task Recognition** (Layer 1): For each task, the system:
   - Searches its workflow library for similar historical tasks
   - If found: Uses best-performing workflow as template, adapting for current context
   - If novel: Synthesizes new workflow from scratch

3. **Evolutionary Optimization**: Over time, the system:
   - Maintains multiple workflow variants per task type
   - Selects high-performing workflows based on success metrics (speed, accuracy, cost)
   - Mutates/recombines successful patterns to explore architecture space

4. **Self-Improvement**: 
   - The system can propose modifications to its own workflow generation logic
   - Performance improvements are validated before integration (Gödel machine principle)
   - Meta-learning: Learns how to generate better workflows from execution history

---

## Phone Notifications

Receive real-time updates about Mimosa's status via Pushover notifications.

### Setup Instructions

1. **Create Pushover Account**
   - Visit [pushover.net](https://pushover.net/)
   - Register and note your **User Key**

2. **Create Application**
   - In Pushover dashboard, click "Create an Application/API Token"
   - Name it "Mimosa" and copy the generated **API Token**

3. **Configure Environment**
   ```bash
   export PUSHOVER_USER="your_user_key"
   export PUSHOVER_TOKEN="your_api_token"
   ```

4. **Install Mobile App**
   - Download Pushover from your device's app store
   - Log in with your Pushover account

---

## Telemetry Setup

Monitor and debug AI agents with real-time observability dashboards using Langfuse.

### Quick Start

1. **Deploy Langfuse Locally**
   ```bash
   git clone https://github.com/langfuse/langfuse.git
   cd langfuse
   docker compose up -d
   ```

2. **Configure Environment Variables**
   
   Add to your `.env` file:
   ```env
   LANGFUSE_PUBLIC_KEY=your_public_key
   LANGFUSE_PRIVATE_KEY=your_private_key
   ```

3. **Access Dashboard**
   
   While Mimosa-AI is running, visit `http://localhost:3000`

### Available Metrics

The telemetry dashboard provides:
- **Agent Execution Traces**: Step-by-step workflow visualization
- **Performance Metrics**: Response times and success rates
- **Error Debugging**: Detailed failure analysis
- **Resource Usage**: Token consumption and API calls

**Example Dashboard:**
![Langfuse Dashboard](https://langfuse.com/images/cookbook/integration-smolagents/smolagent_example_trace.png)

> **Note:** Telemetry is optional but recommended for debugging and performance optimization.

---

## Support & Contributing

For issues, questions, or contributions, please visit the project repository at [github.com/HolobiomicsLab/Mimosa-AI](https://github.com/HolobiomicsLab/Mimosa-AI).

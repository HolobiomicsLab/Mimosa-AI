
# Mimosa-AI 🔬🤖

**An automated AI scientist framework for reproducing scientific findings & making discoveries.**

## Installation

### Prerequisites
- Python 3.10 or higher
- pip3 package manager

### Install & Run

0. Setup your env file

```sh
mv .env.example .env
```

Fill with your API key for `HF_TOKEN` and `OPENAI_API_KEY`.

`LANGFUSE_PUBLIC/PRIVATE_KEY` are only needed if you wish to setup agent [telemetry](https://huggingface.co/docs/smolagents/tutorials/inspect_runs).

1. **Install dependencies**
    ```bash
    pip3 install -r requirements.txt
    ```

2. **Run the toolomics MCP server**

See instructions at [HolobiomicsLab/toolomics](https://github.com/HolobiomicsLab/toolomics)

2. **Run the main script with a goal**:
    ```bash
    python3 main.py --goal "Your research objective here"
    ```

### Usage Options

- `--goal GOAL`: Required. Specify your research objective or scientific question
- `--load_template LOAD_TEMPLATE`: Optional. Load a predefined workflow template

**Example:**
```bash
python3 main.py --goal "Search for the latest research on transformer architectures, identify a promising technique, and implement a novel variant to evaluate its performance" --load_template 64a644cf1ef449a684a8a369e47387e2
```

## 📈 Telemetry Setup

Monitor and debug your AI agents with real-time observability dashboards.

### Quick Start

1. **Deploy Langfuse locally**:
    ```bash
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

## 🏗️ Architecture Overview

Mimosa-AI employs a **polymorphic meta-agent system** that dynamically composes specialized workflows for each scientific task, moving beyond rigid pipeline architectures.

### 🧠 Core Philosophy

Our system uses an "agent-within-agent" pattern where a meta-orchestrator intelligently designs custom multi-agent systems tailored to specific research objectives.

---

## 📊 System Layers

### 🎯 Layer 1: Meta-Orchestration
- **Dynamic Workflow Generation**: Advanced LLMs (openai o3) create bespoke multi-agent workflows for every goal
- **Task-Specific Architecture**: Custom agent topologies designed per task rather than one-size-fits-all pipelines

### 🔗 Layer 2: Workflow Composition (LangGraph)
Multi-agent workflows as directed graphs with two node types:
- **SmolAgent Instances**: Autonomous code-generating agents for complex reasoning
- **Deterministic Functions**: Validation, data transformation, and control logic

### ⚡ Layer 3: Agent Execution (SmolAgent)
Code-generating agents operating in action/observation loops:
- **Tool-as-Code**: Agents generate Python code to interact with scientific tools
- **Iterative Refinement**: Continuous execution until success or failure threshold
- **Domain Flexibility**: Unified framework for web scraping, analysis, visualization, and more

### 🛠️ Layer 4: Tool Primitives
Extensible ecosystem built on Model Context Protocol (MCP):
- **Distributed Execution**: Tools run across HPC clusters, instruments, or cloud services
- **Protocol Standardization**: Seamless client-server tool interaction via MCP
- **Domain Coverage**: From web browsers to specialized scientific software


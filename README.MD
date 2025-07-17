
# Mimosa-AI 🔬🤖

**An automated AI scientist framework for reproducing scientific findings & making discoveries.**

## Installation

### Prerequisites
- Python 3.10 or higher
- pip3 package manager

### 🚀 Install & Run

#### Step 1: Environment Setup

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
- `HF_TOKEN`: Your Hugging Face token
- `OPENAI_API_KEY`: Your OpenAI API key
- `LANGFUSE_PUBLIC_KEY` & `LANGFUSE_PRIVATE_KEY`: Optional, for [telemetry monitoring](https://huggingface.co/docs/smolagents/tutorials/inspect_runs)

#### Step 2: Install Dependencies

First go into the `mimosa` folder
```sh
cd mimosa
```

**Using pip:**
```sh
pip3 install -r requirements.txt
```

**Using uv (faster):**
```sh
uv pip install -r requirements.txt
```

#### Step 3: Start MCP Server
Launch the toolomics MCP server following the instructions at [HolobiomicsLab/toolomics](https://github.com/HolobiomicsLab/toolomics)

#### Step 4: Run Mimosa-AI

```sh
python3 main.py --goal "Your objective here" --load_template <optional existing workflow uuid>
# or
uv run main.py --goal "Your objective here" --load_template <optional existing workflow uuid>
```

> **Note**: Remember to activate your virtual environment (`source mimosa-env/bin/activate`) before running Mimosa-AI in future sessions.

### Usage Options

- `--goal GOAL`: Required. Specify your research objective or scientific question
- `--load_template TEMPLATE_UUID`: Optional. Load a predefined workflow template from workflows folder

**Example:**
```sh
python3 main.py --goal "search and install llama.cpp for this OS architecture" --load_template 64a644cf1ef449a684a8a369e47387e2
```

**`load_template` allow you to use an existing workflow to avoid waiting generating a functional workflow again for an identical task.**

## Use Caching system

You'll need to open 2 terminal.

### First terminal 

1. **Create a config.yaml file in cached_server**

```sh
server:
  port: 6767
  storage_path: "./cache"
  cache_enabled: true

providers:
  - name: "deepseek"
    base_url: "https://api.deepseek.com"
    api_key: "xxxxxxxx"
    weight: 1
```

2. **Start the cached server**

```sh
go run main.go
```

### Second terminal 

First export this:

```sh
export USE_CACHED_ENGINE="true"
```

Then run *Mimosa* as usual.

```sh
python3 main.py --goal "<goal>" --load_template <optional existing workflow uuid>
```



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


# Mimosa-AI 🌾

<img src="./docs/images/mimosa_illustration.png" alt="Mimosa" height="650"/>

***Mimosa-AI*** 🌾 is an AI-scientist framework built to carry out end-to-end research and reproduce published finding autonomously. Built to give academics a powerful open and modular alternative to closed black-box systems.

**Use case:**
- Reproduce scientific studies with rigorous, auditable workflows
- Automate full pipelines: bioinformatics, molecular docking, metabolomics, and beyond

## How does it work ?

The user gives ***Mimosa-AI*** a research goal.

- ***Mimosa*** automatically discovers available MCP-based tools on the local network or via Toolhive (anything from data analysis utilities to web browsers or lab instruments like mass spectrometers).
- Using the user’s objective and the discovered tools, ***Mimosa*** decomposes the problem, builds a tailored multi-agent workflow for each tasks.
- Each task runs autonomously. Failures are used for self-improvement via a Iterative-learning loop.
- ***Mimosa*** generates a final capsule containing results, visualizations, reports, logs, and all relevant artifacts.

![schema](./docs/images/mimosa_overall.jpg)


## Installation & Run

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
OPENROUTER_API_KEY=your_openrouter_key # if using openrouter
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key    # Optional
LANGFUSE_PRIVATE_KEY=your_langfuse_private_key  # Optional
```

**Explanations:**
- `HF_TOKEN`: Hugging Face token for LLM access
- `OPENROUTER_API_KEY`: openrouter API key, if using openrouter to use any model (**see:** [Openrouter](https://openrouter.ai/))
- `MISTRAL_API_KEY`: Mistral API key, if using Mistral.
- `DEEPSEEK_API_KEY`: Deepseek API key, if using a Deepseek model.
- `OPENAI_API_KEY`: OpenAI API key, if using GPT model such as GPT-5.2.
- `ANTHROPIC_API_KEY`: Anthropic API key, if using Claude.
- `LANGFUSE_*`: Totally telemetry keys for monitoring (see [Telemetry Setup](#telemetry-setup))

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

You could add your custom MCPs to toolomics, see [toolomics documentation](https://github.com/HolobiomicsLab/toolomics/README.md).

Configure the server to run on a port range (e.g., 5000-5100).

### Step 5: Configure Mimosa-AI

Create your custom **config.json** file.

1. copy default config:
```sh
cp config_default.json my_config.json
```

2. edit the config:
```sh
vim my_config.json
# or:
code my_config.json
# or: just open it in VS code the normal way
```

**Alternative**: directly modify the config value in `config.py`

**Mimosa Configuration Overview**

| Parameter | Description |
|---------|-------------|
| `workspace_dir` | Path to the Toolomics workspace. All files created or modified by ***Mimosa*** appear here. Must point to the Toolomics project directory. |
| `discovery_addresses` | Network ranges (IP + port range) used to discover MCP tool servers. |
| `planner_llm_model` | LLM used to decompose tasks and build execution plans. |
| `prompts_llm_model` | LLM used for workflow prompts generation.  |
| `workflow_llm_model` | LLM used to generate and orchestrate multi-agent workflows.  (**Recommand:** anthropic/claude-opus-4-5-20251101)  |
| `smolagent_model_id` | Model used for HuggingFace SmolAgents handling execution subtasks. |
| `judge_model` | LLM used to evaluate ***Mimosa’s*** own outputs and assign performance scores. |
| `engine_name` | Inference engine used to route and manage model calls (LiteLLM). |
| `prompt_planner` | Prompt file used by the planner to decompose tasks. |
| `prompt_workflow_creator` | Prompt used to create multi-agent workflows. **Do not modify.** |
| `reasoning_effort` | Controls the depth of reasoning for **gpt5** only. |
| `learned_score_threshold` | Score at which self-improvement stops and the result is accepted. |
| `max_learning_evolve_iterations` | Maximum number of self-improvement iterations allowed. |
| `schema_code_path` | Internal state schema definition. **Do not modify.** |
| `smolagent_factory_code_path` | SmolAgent factory implementation. **Do not modify.** |
| `runs_capsule_dir` | Directory where each run saves a full workspace snapshot in an auto-named capsule. |
| `workflow_dir` | Directory containing predefined multi-agent workflows. |
| `memory_dir` | Persistent storage for ***Mimosa’s*** long-term memory. |
| `runner_*` | Execution, timeout, and resource settings. **Do not touch.** |


### Step 6: Run Mimosa-AI

```bash
python3 main.py --goal "Your objective here" --config my_config.json
# OR with uv:
uv run main.py --goal "Your objective here" --config my_config.json
```

**Standard usage - accomplish a goal:**
```bash
uv run main.py --goal "Reproduce the experiments from 'Dual Aggregation Transformer for Image Super-Resolution' (https://arxiv.org/pdf/2306.00306) and compare results." --config my_config.json
```

**Single task mode - no long-term planning:**
```bash
uv run main.py --task "Train a multitask model on the Clintox dataset to predict drug toxicity and FDA approval status" --config my_config.json
```
> **Note:** Requires Toolomics to be installed and MCP servers to be running.

**Goal:** High-level scientific objective requiring multiple distinct complex tasks
- *Example:* "Develop a machine learning model to predict protein-ligand binding affinity"
- *Example:* "Reproduce research paper X and compare experimental results"

**Task:** Granular, repeatable operation frequently encountered across different goals
- *Example:* "Conduct literature review on topic X"
- *Example:* "Download dataset from source Y"
- *Example:* "Implement algorithm Z"

### Step 7: Access output files

Output files will appear during execution in **toolomics** `workspace` folder, when the execution its content will be transfered inside a new folder in `Mimosa-AI/runs_capsule/`

## Learning

***Mimosa-AI*** learns from failure. For any new task, start with learn mode to let it build competence before full autonomy.

**Start in Learning mode**

```bash
uv run main.py --task "Train a multitask model on the Clintox dataset to predict drug toxicity and FDA approval status" --learn --config my_config.json
```

**Progress visualization:**

Once ***Mimosa-AI*** completes its learning phase on a task, you can visualize exactly how it improved over time. The reward progress plot showing performance gains across attempts is automatically saved.

The reward progress plot is saved under the `sources/workflows/<uuid>` folder under the filename `reward_progress.png`.

***Example:***

![dgm](./docs/images/evolve_example.png)

## Transparency

We ship an interactive debugger, `memory_explorer.py`, that lets you step through any agent execution in granular detail.

Start it with a workflow <uuid> (eg: `memory_explorer.py 20260115_113303_9bb63437`)

This replays the full execution trace—thoughts, tool calls and outputs so you can inspect exactly how decisions unfolded.

---

## Command Line Arguments

### Execution Modes

| Argument | Description |
|----------|-------------|
| `--goal GOAL` | Specify a high-level research objective, paper reproduction, or scientific question (planner mode) |
| `--task TASK` | Execute a single task: literature review, datasets download, implement a machine learning model... |
| `--manual` | Interactive CLI mode to debug MCPs and test ***Mimosa*** tools directly |
| `--papers <CSV path>` | Evaluation on a CSV dataset containing research papers and prompts |
| `--science_agent_bench` | Evaluation on ScienceAgentBench |

### Other parameters
| Argument | Description |
|----------|-------------|
| `--learn` | Enable iterative-learning to optimize task performance |
| `--max_evolve_iterations N` | Maximum learning iterations |
| `--csv_runs_limit N` | Limit number of CSV entries to evaluate |
| `--scenario <scenario file name>` | Use specific scenario based assertions instead of LLM-as-a-judge for scoring execution  |
| `--single_agent` | Single agent mode. fast, but can't improve throught learning |
| `--debug` | Enable debug mode for more verbose logging |

---

### System Overview

***Mimosa-AI*** core innovation is at it's **self-evolution** of multi-agent system: It dynamically synthesizes specialized workflows for scientific tasks. Rather than forcing tasks through fixed pipelines, the system composes custom multi-agent architectures on-demand and learns from execution patterns to optimize future performance.

- Goals decompose into learnable tasks
- Each task triggers synthesis of a specialized multi-agent workflow
- Successful workflow patterns are retained and refined over time
- The system continuously optimizes task-specific multi-agent architectures through execution feedback

**Self-Improvement Mechanism**

The system implements a Darwinian-inspired evolution approach to workflow evolution:

1. **Task Recognition**: For each task, the system:
   - Searches workflow library for similar historical tasks
   - If found: Uses best-performing workflow as template, adapting for current context
   - If novel: Synthesizes new workflow from scratch

2. **Evolutionary Optimization**: Over time, the system:
   - Maintains multiple workflow variants per task type
   - Selects high-performing workflows based on success metrics
   - Mutates/recombines successful patterns to explore architecture space

3. **Self-Improvement**: 
   - The system can propose modifications to its own workflow generation logic
   - Performance improvements are validated before integration (Gödel machine principle)
   - Meta-learning: Learns how to generate better workflows from execution history

![dgm](./docs/images/workflow_mutation.png)

---

## Evaluation

***Mimosa-AI*** can be evaluated either on [ScienceAgentBench](https://arxiv.org/abs/2410.05080) or [PaperBench](https://arxiv.org/pdf/2504.01848).

⚠️ For unbiased evaluation it is advised to run `./cleanup.sh` first, this will prevent ***Mimosa*** from using existing or cached workflows.

### ScienceAgentBench

To evaluate on ScienceAgentBench you must:

1. download the ScienceAgentBench full dataset:
[dataset link](https://buckeyemailosu-my.sharepoint.com/personal/chen_8336_buckeyemail_osu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch%2Fbenchmark%2Ezip&parent=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch&ga=1)
2. unzip it with password: `scienceagentbench`
3. copy content of `benchmark/benchmark/datasets` folder to  `Mimosa-AI/datasets/scienceagentbench/datasets`

**Evaluation on ScienceAgentBench with learning**
```sh
uv run main.py --science_agent_bench --learn
```

**Evaluation on ScienceAgentBench limited to 10 tasks with learning limited to 4 learning iterations**

```sh
uv run main.py --science_agent_bench --csv_runs_limit 10 --max_evolve_iterations 4
```

### PaperBench

**Evaluation on OpenAI PaperBench with learning mode**

OpenAI PaperBench is a benchmark for evaluating the ability of AI agents to replicate AI research, from the paper `PaperBench: Evaluating AI’s Ability to Replicate AI Research`.

```sh
uv run main.py --papers datasets/paper_bench.csv --csv_runs_limit 20  --learn
```

⚠️ This will save in runs_capsule/ folder the result of all paper's reproduction attempt, refer to [Paper Bench documentation](https://github.com/openai/frontier-evals/tree/main/project/paperbench) for complete evaluation.

**Evaluation on custom benchmark of research paper**

1. Place your benchmark CSV with the same format as `paper_bench.csv` in `datasets/<your_benchmark_name>.csv`.

2. Run on your benchmark:

```sh
uv run main.py --papers datasets/<your_benchmark_name>.csv --csv_runs_limit 20  --learn
```

---

## Phone Notifications

Receive real-time updates about ***Mimosa's*** status via Pushover notifications.

### Setup Instructions

1. **Create Pushover Account**
   - Visit [pushover.net](https://pushover.net/)
   - Register and note your **User Key**

2. **Create Application**
   - In Pushover dashboard, click "Create an Application/API Token"
   - Name it "***Mimosa***" and copy the generated **API Token**

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
   
   While ***Mimosa-AI*** is running, visit `http://localhost:3000`

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

## License

This repository is publicly distributed under the GNU Affero General Public License v3.0 (AGPLv3). A separate commercial licensing path may be available from the designated rights-holder.

For contribution and licensing details, see:
- `docs/licensing-notes.md`
- `CLA/INDIVIDUAL_CLA.md`
- `CLA/CORPORATE_CLA.md`

---
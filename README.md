<div align="center">
<br>

<img src="./docs/images/logo_mimosa.png" width="22%" style="border-radius: 8px;" alt="Mimosa-AI logo — self-evolving multi-agent AI framework for autonomous scientific research (Holobiomics Lab, CNRS)">

</div>

<h1 align="center">Mimosa-AI 🌼🔬</h1>

<p align="center">
  <a href="./README.md">English</a> &nbsp;|&nbsp;
  <a href="./README.CHS.md">简体中文</a> &nbsp;|&nbsp;
  <a href="./README.CHT.md">繁體中文</a> &nbsp;|&nbsp;
  <a href="./README.JPN.md">日本語</a> &nbsp;|&nbsp;
  <a href="./README.KOR.md">한국어</a>
</p>

<p align="center">
    <em>Self-evolving multi-agent framework for autonomous scientific research — LLM-driven workflow synthesis, Quality-Diversity evolutionary search, MCP tool discovery.</em>
</p>

<p align="center">
  🧬 Quality-Diversity workflow evolution &nbsp;·&nbsp;
  🔍 MCP-based tool auto-discovery &nbsp;·&nbsp;
  🧪 Multi-source per-claim verification &nbsp;·&nbsp;
  📦 Full audit trail & reproducibility
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2603.28986"><img src="https://img.shields.io/badge/arXiv-2603.28986-b31b1b.svg?logo=arxiv&style=flat-square&logoColor=white" alt="arXiv Preprint"></a>
    <a href="https://doi.org/10.48550/arXiv.2603.28986"><img src="https://img.shields.io/badge/DOI-10.48550%2FarXiv.2603.28986-blue?style=flat-square" alt="DOI"></a>
    <a href="https://holobiomicslab.cnrs.fr/"><img src="https://img.shields.io/badge/website-holobiomicslab.cnrs.fr-4caf82?style=flat-square&logo=globe&logoColor=white" alt="website"></a>
</p>

<p align="center">
    <a href="https://github.com/HolobiomicsLab/Mimosa-AI/stargazers"><img src="https://img.shields.io/github/stars/HolobiomicsLab/Mimosa-AI?style=social" alt="GitHub Stars"></a>&nbsp;
    <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=flat-square" alt="License: Apache 2.0"></a>
</p>

---

## TL;DR

Mimosa-AI is an **open-source Python framework for autonomous scientific research**: it writes a **custom multi-agent workflow per task**, runs it in a sandbox, checks what the agents actually did against independent vantage points, and evolves the workflow across generations with a **Quality-Diversity** inspired search to find the optimal workflow for the task.

The workflow is emitted as plain Python — no DSL, no YAML — so any generation can be inspected, diffed, or re-run standalone. The verifier score workflows by running deterministic Python checks that verify litterature grounding, non-triviality, and quality metrics against artifacts that the agents produced. Every generation is on disk with its lineage and the exact LLM prompt that produced it.

```bash
uv sync && uv run main.py        # interactive onboarding
```

---

## Demo

<p align="center">
    <em>Mimosa-AI autonomously regenerated the LC-MS/MS molecular networking pipeline of <a href="https://www.researchgate.net/publication/323525305_Bioactivity-Based_Molecular_Networking_for_the_Discovery_of_Drug_Leads_in_Natural_Product_Bioassay-Guided_Fractionation">Nothias et al. (2018)</a> — feature detection on the <code>.mzML</code> files (MZmine / OpenMS / matchms-class tooling — the agents pick the stack), alignment, and classical molecular networking (GNPS-style cosine clustering).</em>
</p>

https://github.com/user-attachments/assets/dcd04ade-9c43-44a8-b3e3-a999d3dc895d

The reproduced network matches the topology reported in the paper at the cluster level, output as a Cytoscape-loadable `.graphml` plus the underlying feature quantification table. Scope note: this reproduces the **molecular networking** stage only — the bioactivity-guided fractionation, manual annotation review, and library matching (GNPS / SIRIUS / CSI:FingerID) from the original study are out of scope for the autonomous run.

<p align="center">
  <img src="./docs/images/network.png" alt="LC-MS/MS molecular network autonomously reproduced by Mimosa-AI — GNPS-style cosine clustering exported as Cytoscape GraphML" width="80%">
</p>

---

## Benchmark (V1)

Evaluated on **ScienceAgentBench** (102 tasks, `task` mode — planning layer bypassed so workflow synthesis and refinement are evaluated in isolation):

| Mode                                    | Success Rate | Code-BLEU | Cost / task |
| --------------------------------------- | ------------ | --------- | ----------- |
| DeepSeek-V3.2 single-agent              | 38.2 %       | 0.898     | $0.05       |
| DeepSeek-V3.2 one-shot multi-agent      | 32.4 %       | 0.794     | $0.38       |
| **DeepSeek-V3.2 iterative-learning**    | **43.1 %**   | **0.921** | **$1.70**   |

> **43.1 % success rate on ScienceAgentBench with DeepSeek-V3.2 iterative-learning — +4.9 pp over the single-agent baseline at $1.70 per task.**

> On ScienceAgentBench with DeepSeek-V3.2, iterative learning improves GPT-4o but yields marginal degradation on Claude Haiku 4.5 — model-dependent behaviour is analysed in the [manuscript](https://arxiv.org/abs/2603.28986). For PaperBench results, see [`docs/papers_bench_evaluation.md`](./docs/papers_bench_evaluation.md).

---

## How it works

Five layers, wired through small dataclass schemas — full details in [`docs/concepts/architecture.md`](./docs/concepts/architecture.md).

<p align="center">
  <img src="./docs/images/mimosa_overall.jpg" alt="Mimosa-AI architecture: planner, MCP tool manager, evolution engine, sandboxed SmolAgents workflow runner, multi-source per-claim verifier" width="90%">
</p>

| Layer | Component | What it does |
|-------|-----------|--------------|
| 0 | **Planner** *(optional, `--goal` only)* | Decomposes a high-level objective into discrete tasks. |
| 1 | **ToolManager + Perspicacité** | Discovers MCP tools on the configured address/port range; optionally pulls literature snippets. |
| 2 | **EvolutionEngine** | Synthesizes the workflow and evolves it across generations (see below). |
| 3 | **WorkflowRunner** | Runs the synthesized Python workflow in a sandbox using Hugging Face [SmolAgents](https://github.com/huggingface/smolagents) (`LocalPythonExecutor` with AST allow-list) and shared LangGraph state. |
| 4 | **VerifierEvaluator** | Multi-source per-claim verifier. Drives the next mutation. |

### The evolution loop — what's actually evolving

Workflows are **full Python programs**, mutated as source code. The **code-as-genotype** is the workflow file; the phenotype is whatever it produces in the workspace.

- **Selection: Quality-Diversity archive** (**MAP-Elites**-style) — max population of 50, `qd_score = (1−w)·quality + w·novelty` (`w=0.4`). **Novelty search** uses k-NN distance (`k=25`) over a **behaviour descriptor** `[n_agents, n_edges, n_branches, prompt_chars]`. Parents drawn by inverse-child-count roulette so the archive spreads.
- **Variation: stagnation-driven scope** — mutation boldness is a continuous function of how much the last 4 prompt gradients repeat themselves. Near-winners stay protected. Scope bands run from "prompt-only tweak" to "complete topology rethink."
- **Crossover** — ~30 % of generations combine two parents, strongest-first.
- **Cold start** — when the archive is empty, a similarity-filtered scan of past runs on disk (MiniLM cosine ≥ 0.5) seeds the search. Useful workflows transfer across tasks.

Full mechanics: [`docs/concepts/evolution-engine.md`](./docs/concepts/evolution-engine.md).

### The verifier — what scores actually mean

After each run, six independent claim sources look at the workspace and emit success-polarity claims:

| Source | Vantage |
|--------|---------|
| **A** | Peer-reviewed practice (via Perspicacité literature grounding) |
| **B** | The literal goal text — did the agents deliver what was asked? |
| **C** | Agent narration — can claimed numbers / artefacts be reproduced from disk? |
| **D** | Math invariants — probabilities in [0,1], shape consistency, no NaN, conservation |
| **E** | Computational reproducibility — declared deps cover used imports, no absolute paths, seeds on stochastic ops |
| **F** | Statistical fingerprint — beats a baseline, no degenerate predictions, no leakage signatures |

Each claim is verified by a **python program** the judge writes against the workspace — not by re-asking an LLM whether it believes the agent.

**Rubric-blind mutation — the mutator never sees the rubric.** The only signal that flows back is an `abstracted_prompt_gradient` — a code-named diagnosis of failure modes that does not name claims, scores, or sources. By construction the search cannot over-fit to a rubric vocabulary it never sees.

Full pipeline: [`docs/concepts/evaluation-pipeline.md`](./docs/concepts/evaluation-pipeline.md).

---

## Quickstart

### 1. Install

```bash
pip install uv
git clone https://github.com/HolobiomicsLab/Mimosa-AI.git
cd Mimosa-AI
uv sync
```

### 2. Add at least one LLM key

Create `.env` at the project root. Only the providers you actually use are required.

```env
ANTHROPIC_API_KEY=...       # Claude — recommended for workflow synthesis
OPENAI_API_KEY=...
MISTRAL_API_KEY=...
DEEPSEEK_API_KEY=...
HF_TOKEN=...
OPENROUTER_API_KEY=...      # Any model via OpenRouter

# Optional: Langfuse observability
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_PRIVATE_KEY=...
```

### 3. Expose MCP tools

Mimosa discovers any MCP server reachable on the address/port range in your config (default `0.0.0.0:5000–5100`).

- **Easiest path:** install our companion platform **[Toolomics](https://github.com/HolobiomicsLab/toolomics)** — a per-workspace MCP shell sandbox where agents install scientific packages on demand, plus pre-built MCP servers exposing common scientific stacks, shared workspace management, and an easy registration flow for new MCPs.
- **Bring-your-own:** point `discovery_addresses` at any reachable MCP server — `fastmcp` scripts, ToolHive, third-party MCP containers. Toolomics is not required; see [`docs/concepts/tools-and-mcp.md`](./docs/concepts/tools-and-mcp.md#operating-without-toolomics).

### 4. Run

```bash
uv run main.py                   # interactive onboarding (recommended first time)
```

Or skip the wizard:

```bash
uv run main.py --task "Train a multitask model on Clintox to predict toxicity and FDA approval"
uv run main.py --goal "Reproduce experiments from https://arxiv.org/pdf/2306.00306 and compare results"
```

Add `--learn` to evolve across generations instead of one-shotting:

```bash
uv run main.py --task "..." --learn --config my_config.json
```

Full quickstart: [`docs/getting-started/quickstart.md`](./docs/getting-started/quickstart.md).

### 5. (Optional) Scientific grounding via Perspicacité

[Perspicacité](https://github.com/HolobiomicsLab/Perspicacite-AI) grounds workflow synthesis and Source A claims in the literature. When it's running, Mimosa picks it up automatically.

```bash
git clone https://github.com/HolobiomicsLab/Perspicacite-AI.git && cd Perspicacite-AI
uv sync && uv run web_app_full.py
```

---

## Execution modes

| Mode | Use when | Command |
|------|----------|---------|
| `--task` | Single focused operation | `uv run main.py --task "..."` |
| `--goal` | Multi-step objective requiring planning | `uv run main.py --goal "..."` |
| `--learn` | Add to either mode — evolve across generations | `... --learn` |
| `--single_agent` | Skip multi-agent synthesis (fast, no learning) | `... --single_agent` |
| `--manual` | Interactive CLI to test individual MCP tools | `uv run main.py --manual` |
| Batch | Evaluate a CSV of tasks | `... --papers <csv>` |
| Benchmark | ScienceAgentBench | `... --science_agent_bench` |

Details: [`docs/usage/modes.md`](./docs/usage/modes.md), [`docs/usage/learning.md`](./docs/usage/learning.md), [`docs/reference/cli.md`](./docs/reference/cli.md).

---

## Audit trail and replay

Mimosa is built for scientific use — every decision is inspectable after the fact.

| Tool | What it does |
|------|--------------|
| `uv run memory_explorer.py <uuid>` | Step through one generation's full trace — thoughts, tool calls, outputs, state deltas. |
| `uv run main.py --memory_cli` | RAG-backed Q&A over a finished run's memory. Ask "*what classifier did task_builder use?*" instead of scrolling. |
| `uv run memory_timelapse.py <uuid>` | Animated frame-by-frame view of memory growth across iterations. |
| `sources/workflows/<uuid>/workflow_genotype_<uuid>.py` | The exact Python the agents executed. No DSL. |
| `sources/workflows/<uuid>/lineage_<uuid>.json` | Parents and operator (`seed | mutation | crossover`) for this generation. |
| `sources/workflows/<uuid>/evolution_prompt_<uuid>.md` | The exact LLM prompt that produced this code. Same prompt + seed = same code. |
| `sources/workflows/<uuid>/evolution_tree.png` | Rendered lineage tree of the whole `--learn` run. |
| `sources/workflows/<uuid>/reward_progress.png` | Score-over-iteration curve. |
| `runs_capsule/<capsule_name>/` | Archived snapshot of the final workspace for sharing or re-running. |

Full layout: [`docs/usage/transparency.md`](./docs/usage/transparency.md), [`docs/usage/workspace.md`](./docs/usage/workspace.md).

---

## Configuration

Copy `config_default.json` to `my_config.json` and edit. The fields you'll touch most often:

| Field | What it controls |
|-------|------------------|
| `workspace_dir` | Shared workspace — all generated files appear here |
| `discovery_addresses` | IP + port ranges for MCP discovery |
| `workflow_llm_model` | Synthesizes the multi-agent workflow (e.g. `anthropic/claude-opus-4-5`) |
| `smolagent_model_id` | Model used by execution agents |
| `judge_model` | LLM that writes verifier programs and renders soft verdicts |
| `learned_score_threshold` | Early-stop threshold in `--learn` mode (default `0.97`) |
| `max_learning_evolve_iterations` | Cap on generations (default `35`) |
| `population_size` / `novelty_weight` / `min_improvement_threshold` | QD archive tuning |

Full reference: [`docs/reference/configuration.md`](./docs/reference/configuration.md).

---

## Evaluation

```bash
# ScienceAgentBench (download dataset first — see docs)
uv run main.py --science_agent_bench --learn

# Quick smoke (10 tasks)
uv run main.py --science_agent_bench --csv_runs_limit 10

# PaperBench
uv run main.py --papers datasets/paper_bench.csv --csv_runs_limit 20 --learn

# Custom CSV
uv run main.py --papers datasets/<your_benchmark>.csv --learn
```

> ⚠️ For unbiased evaluation, run `./cleanup.sh` first to prevent Mimosa from reusing cached workflows.

Setup details for each benchmark: [`docs/science_agent_bench_evaluation.md`](./docs/science_agent_bench_evaluation.md), [`docs/papers_bench_evaluation.md`](./docs/papers_bench_evaluation.md).

---

## Notifications and telemetry

- **Pushover** — real-time progress on your phone. Set `PUSHOVER_USER` and `PUSHOVER_TOKEN`. Details: [`docs/usage/notifications.md`](./docs/usage/notifications.md).
- **Langfuse** — span-level traces of every LLM call. `docker compose up -d` from the Langfuse repo, then add `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_PRIVATE_KEY` to `.env`. Dashboard at `http://localhost:3000`. Details: [`docs/usage/telemetry.md`](./docs/usage/telemetry.md).

---

## Related work

Mimosa-AI sits in a small but active lineage of LLM-driven program-search and autonomous-research systems. We don't claim to subsume any of them — they answer different questions:

| Project | What it does | How Mimosa differs |
|---------|--------------|--------------------|
| [Sakana AI Scientist](https://github.com/SakanaAI/AI-Scientist) | End-to-end paper generation in ML | Mimosa optimises **per-task workflow synthesis** with QD+verifier, not full-paper generation |
| [DiscoPOP](https://github.com/SakanaAI/DiscoPOP) (Lange et al. 2024) | LLM-driven discovery of preference-optimisation algorithms | Same "LLM as variation operator over code" paradigm; Mimosa applies it to multi-agent workflow code rather than loss functions |
| [FunSearch](https://github.com/google-deepmind/funsearch) (Romera-Paredes et al. 2024) | Evolutionary search over Python functions guided by an LLM | Mimosa evolves whole multi-agent programs and adds a multi-source per-claim verifier instead of a single fitness function |
| [ELM](https://github.com/CarperAI/OpenELM) (Lehman et al. 2022) | LLM-mediated quality-diversity over code | Closest QD ancestor; Mimosa's behaviour descriptor is workflow-structural rather than domain-specific |
| AIDE | Automated ML pipelines on Kaggle-like tasks | Mimosa targets broader scientific reproduction (ScienceAgentBench, PaperBench, lab data) and ships an auditable per-claim verifier |

If you're publishing comparative work, the [manuscript](https://arxiv.org/abs/2603.28986) has the detailed positioning.

---

## Full documentation

```bash
uvx --with mkdocs-material mkdocs serve   # live preview at http://localhost:8000
uvx --with mkdocs-material mkdocs build   # static HTML to ./site
```

Site config: [`mkdocs.yml`](./mkdocs.yml). Index: [`docs/index.md`](./docs/index.md).

---

## Contributing

Patches, MCP tools, evaluators, and new claim sources welcome. Start with [`CONTRIBUTING.md`](./CONTRIBUTING.md), the [Developer guide](./docs/DEVELOPER_GUIDE.md), and the contribution terms in [`CLA/`](./CLA/).

---

## License

Apache 2.0. See [`NOTICE`](./NOTICE), [`docs/licensing-notes.md`](./docs/licensing-notes.md), and the [`CLA/`](./CLA/) folder for contribution terms.

---

## Cite this work

<p align="center">
<em><a href="https://arxiv.org/abs/2603.28986">Mimosa Framework: Toward Evolving Multi-Agent Systems for Scientific Research</a></em><br>
M. Legrand, T. Jiang, M. Feraud, B. Navet, Y. Taghzouti, F. Gandon, E. Dumont, L.-F. Nothias — <em>arXiv:2603.28986, 2026</em> — <a href="https://doi.org/10.48550/arXiv.2603.28986">DOI</a>
</p>

```bibtex
@article{legrand2026mimosa,
  title         = {Mimosa Framework: Toward Evolving Multi-Agent Systems for Scientific Research},
  author        = {Legrand, Martin and Jiang, Tao and Feraud, Matthieu and Navet, Benjamin
                   and Taghzouti, Yousouf and Gandon, Fabien and Dumont, Elise and Nothias, Louis-F{\'e}lix},
  journal       = {arXiv preprint arXiv:2603.28986},
  year          = {2026},
  eprint        = {2603.28986},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```

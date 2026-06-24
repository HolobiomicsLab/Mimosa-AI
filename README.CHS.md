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
    <em>面向自主科学研究的自进化多智能体框架 —— LLM 驱动的工作流合成、质量-多样性进化搜索、MCP 工具自动发现。</em>
</p>

<p align="center">
  🧬 质量-多样性工作流进化 &nbsp;·&nbsp;
  🔍 基于 MCP 的工具自动发现 &nbsp;·&nbsp;
  🧪 多源逐声明验证 &nbsp;·&nbsp;
  📦 完整审计追踪与可复现性
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

Mimosa-AI 是一个**面向自主科学研究的开源 Python 框架**:它为**每个任务编写定制的多智能体工作流**,在沙箱中运行它,并对照六个独立视角(文献、你的目标、智能体叙述、数学不变量、计算可复现性、统计指纹)核查智能体的实际行为;并且当你要求它学习时,它会以**质量-多样性 (Quality-Diversity)** 搜索(MAP-Elites 谱系)跨代进化工作流,同时保持性能与结构多样性。

工作流以纯 Python 形式输出 —— 无 DSL、无 YAML —— 因此任何一代都可以被检视、对比或独立重新执行。验证器运行确定性 Python 检查,重新计算智能体所声称的内容。每一代都连同其谱系和生成它的精确 LLM 提示一并落盘。

```bash
uv sync && uv run main.py        # interactive onboarding
```

---

## 演示

<p align="center">
    <em>Mimosa-AI 自主重建了 <a href="https://www.researchgate.net/publication/323525305_Bioactivity-Based_Molecular_Networking_for_the_Discovery_of_Drug_Leads_in_Natural_Product_Bioassay-Guided_Fractionation">Nothias et al. (2018)</a> 的 LC-MS/MS 分子网络流水线 —— 对 <code>.mzML</code> 文件进行特征检测(MZmine / OpenMS / matchms 类工具栈 —— 由智能体自行选择)、对齐与经典分子网络构建(GNPS 风格的余弦聚类) —— 仅需一条命令,无需任何固定流水线。</em>
</p>

https://github.com/user-attachments/assets/dcd04ade-9c43-44a8-b3e3-a999d3dc895d

复现得到的网络在聚类层面与论文所报告的拓扑相吻合,输出为可在 Cytoscape 中加载的 `.graphml` 文件,以及对应的特征定量表。范围说明:此处仅复现**分子网络**阶段 —— 原研究中的生物活性导向分离、人工注释审核以及库匹配(GNPS / SIRIUS / CSI:FingerID)不在本次自主运行的范围内。

<p align="center">
  <img src="./docs/images/network.png" alt="LC-MS/MS molecular network autonomously reproduced by Mimosa-AI — GNPS-style cosine clustering exported as Cytoscape GraphML" width="80%">
</p>

---

## 基准测试

在 **ScienceAgentBench**(102 个任务,`task` 模式 —— 规划层被绕过,因此工作流合成与精化以隔离方式进行评估)上进行评估:

| 模式                                    | 成功率       | Code-BLEU | 单任务成本   |
| --------------------------------------- | ------------ | --------- | ----------- |
| DeepSeek-V3.2 single-agent              | 38.2 %       | 0.898     | $0.05       |
| DeepSeek-V3.2 one-shot multi-agent      | 32.4 %       | 0.794     | $0.38       |
| **DeepSeek-V3.2 iterative-learning**    | **43.1 %**   | **0.921** | **$1.70**   |

> **在 ScienceAgentBench 上,DeepSeek-V3.2 迭代学习取得 43.1% 成功率 —— 相较单智能体基线提升 +4.9 个百分点,单任务成本 $1.70。**

> 在 ScienceAgentBench 上使用 DeepSeek-V3.2 时,迭代学习提升了 GPT-4o 的表现,但在 Claude Haiku 4.5 上出现了边际退化 —— 与模型相关的行为在[手稿](https://arxiv.org/abs/2603.28986)中有详细分析。PaperBench 结果见 [`docs/papers_bench_evaluation.md`](./docs/papers_bench_evaluation.md)。

> **成本与运行时间。** `$1.70/task` 的数字是按默认 `--learn` 预算(最多 35 代,在 `overall_score > 0.97` 时早停)摊销得到的。在 DeepSeek-V3.2 下,一次典型的进化运行每个任务的挂钟时间为 30–90 分钟,并大致随模型价格线性扩展。单次运行(无 `--learn`)约为 5–15 分钟,成本要低一个数量级。

---

## 工作原理

五层架构,通过小型 dataclass 模式串联 —— 完整细节见 [`docs/concepts/architecture.md`](./docs/concepts/architecture.md)。

<p align="center">
  <img src="./docs/images/mimosa_overall.jpg" alt="Mimosa-AI architecture: planner, MCP tool manager, evolution engine, sandboxed SmolAgents workflow runner, multi-source per-claim verifier" width="90%">
</p>

| 层级 | 组件 | 作用 |
|-------|-----------|--------------|
| 0 | **Planner** *(可选,仅 `--goal` 使用)* | 将高层目标分解为离散任务。 |
| 1 | **ToolManager + Perspicacité** | 在配置的地址/端口范围内发现 MCP 工具;可选地拉取文献片段。 |
| 2 | **EvolutionEngine** | 合成工作流并跨代进化它(见下文)。 |
| 3 | **WorkflowRunner** | 在沙箱中运行所合成的 Python 工作流,使用 Hugging Face [SmolAgents](https://github.com/huggingface/smolagents)(`LocalPythonExecutor` 配合 AST 白名单),并共享 LangGraph 状态。 |
| 4 | **VerifierEvaluator** | 多源逐声明验证器。驱动下一次变异。 |

### 进化循环 —— 真正在进化的是什么

工作流是**完整的 Python 程序**,作为源代码被变异。**代码即基因型 (code-as-genotype)** 是工作流文件;表现型是它在工作空间中产生的任何东西。

- **选择:质量-多样性归档**(**MAP-Elites** 风格) —— 种群规模为 50,`qd_score = (1−w)·quality + w·novelty`(`w=0.25`)。**新颖性搜索 (novelty search)** 使用**基因型嵌入 (genotype embedding)** 行为描述符——对工作流生成源代码做 L2 归一化的嵌入(默认使用本地 `all-MiniLM-L6-v2`,可选 OpenAI `text-embedding-3-small`)——上的余弦距离 k-NN(`k=15`)。父代通过子代数量倒数轮盘抽取,使归档分布开来(`MAX_CHILDREN_PER_PARENT = 2`)。
- **变异:由停滞驱动的尺度** —— 变异的大胆程度是过去 4 个提示梯度自身重复程度的连续函数。接近优胜者的个体会被保护。变异尺度的范围从"仅提示微调"一直到"完整拓扑重思"。
- **交叉** —— 约 30% 的代次会以强者优先的方式组合两个父代。
- **冷启动** —— 当归档为空时,会基于相似度过滤,从磁盘上过往运行进行扫描(MiniLM 余弦相似度 ≥ 0.5)以种子化搜索。有用的工作流可在任务之间迁移。

完整机制:[`docs/concepts/evolution-engine.md`](./docs/concepts/evolution-engine.md)。

### 验证器 —— 分数究竟意味着什么

每次运行后,六个独立的声明源会查看工作空间并发出成功极性的声明:

| 源 | 视角 |
|--------|---------|
| **A** | 同行评审的实践(通过 Perspicacité 文献接地) |
| **B** | 字面意义上的目标文本 —— 智能体是否交付了所要求的内容? |
| **C** | 智能体叙述 —— 所声称的数字 / 产物能否从磁盘上重新复算出来? |
| **D** | 数学不变量 —— 概率位于 [0,1] 内、形状一致、无 NaN、守恒律 |
| **E** | 计算可复现性 —— 声明的依赖覆盖了实际使用的导入、无绝对路径、随机操作上有种子 |
| **F** | 统计指纹 —— 超过基线、无退化预测、无泄漏特征 |

每个声明都由裁判针对工作空间编写的**确定性 Python 程序**进行验证 —— 而不是再去问 LLM 它是否相信智能体。**自我验证 (self-verification)** 使用反同义反复的绊线,拒绝那些把智能体输出与自身比较的程序。

**评分细则盲变异 (rubric-blind mutation) —— 变异器从不接触评分细则。**唯一反馈回流的信号是 `abstracted_prompt_gradient` —— 一种用代号表述的失效模式诊断,不会指明声明、分数或源。从结构上讲,搜索无法对从未见过的评分词汇过拟合。

完整流水线:[`docs/concepts/evaluation-pipeline.md`](./docs/concepts/evaluation-pipeline.md)。

---

## 快速开始

### 1. 安装

```bash
pip install uv
git clone https://github.com/HolobiomicsLab/Mimosa-AI.git
cd Mimosa-AI
uv sync
```

### 2. 至少添加一个 LLM 密钥

在项目根目录创建 `.env`。只有你实际使用的提供商才是必需的。

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

### 3. 暴露 MCP 工具

Mimosa 会发现配置中地址/端口范围内可达的任意 MCP 服务器(默认 `0.0.0.0:5000–5100`)。

- **最简路径:**安装我们的配套平台 **[Toolomics](https://github.com/HolobiomicsLab/toolomics)** —— 每个工作空间一个 MCP shell 沙箱,智能体按需安装科研依赖(同一工作空间的后续运行可复用已安装的工具);此外提供预构建的 MCP 服务以暴露常见科研工具栈,并附带共享工作空间管理与规范化的注册流程。
- **自带方案:**将 `discovery_addresses` 指向任意可达的 MCP 服务器 —— `fastmcp` 脚本、ToolHive、第三方 MCP 容器。Toolomics 并非必需;参见 [`docs/concepts/tools-and-mcp.md`](./docs/concepts/tools-and-mcp.md#operating-without-toolomics)。

### 4. 运行

```bash
uv run main.py                   # interactive onboarding (recommended first time)
```

或跳过引导向导:

```bash
uv run main.py --task "Train a multitask model on Clintox to predict toxicity and FDA approval"
uv run main.py --goal "Reproduce experiments from https://arxiv.org/pdf/2306.00306 and compare results"
```

添加 `--learn` 以跨代进化,而非单次运行:

```bash
uv run main.py --task "..." --learn --config my_config.json
```

完整快速开始:[`docs/getting-started/quickstart.md`](./docs/getting-started/quickstart.md)。

### 5.(可选)通过 Perspicacité 进行科学接地

[Perspicacité](https://github.com/HolobiomicsLab/Perspicacite-AI) 将工作流合成和 Source A 声明接地到文献中。当它在运行时,Mimosa 会自动接入。

```bash
git clone https://github.com/HolobiomicsLab/Perspicacite-AI.git && cd Perspicacite-AI
uv sync && uv run web_app_full.py
```

---

## 执行模式

| 模式 | 适用场景 | 命令 |
|------|----------|---------|
| `--task` | 单一聚焦的操作 | `uv run main.py --task "..."` |
| `--goal` | 需要规划的多步目标 | `uv run main.py --goal "..."` |
| `--learn` | 添加到任一模式 —— 跨代进化 | `... --learn` |
| `--single_agent` | 跳过多智能体合成(快速、无学习) | `... --single_agent` |
| `--manual` | 用于测试单个 MCP 工具的交互式 CLI | `uv run main.py --manual` |
| Batch | 评估一个 CSV 的任务集 | `... --papers <csv>` |
| Benchmark | ScienceAgentBench | `... --science_agent_bench` |

详情:[`docs/usage/modes.md`](./docs/usage/modes.md)、[`docs/usage/learning.md`](./docs/usage/learning.md)、[`docs/reference/cli.md`](./docs/reference/cli.md)。

---

## 审计追踪与回放

Mimosa 为科学用途而构建 —— 所有决策事后均可检视。

| 工具 | 作用 |
|------|--------------|
| `uv run memory_explorer.py <uuid>` | 逐步浏览某一代的完整轨迹 —— 思考、工具调用、输出、状态增量。 |
| `uv run main.py --memory_cli` | 基于 RAG 对已完成运行的记忆进行问答。可以问"*task_builder 用了哪个分类器?*"而无需滚动浏览。 |
| `uv run memory_timelapse.py <uuid>` | 跨迭代逐帧动画式查看记忆增长。 |
| `sources/workflows/<uuid>/workflow_genotype_<uuid>.py` | 智能体所执行的精确 Python 代码。无 DSL。 |
| `sources/workflows/<uuid>/lineage_<uuid>.json` | 本代的父代与算子(`seed | mutation | crossover`)。 |
| `sources/workflows/<uuid>/evolution_prompt_<uuid>.md` | 生成该代码的精确 LLM 提示。相同提示 + 相同种子 = 相同代码。 |
| `sources/workflows/<uuid>/evolution_tree.png` | 渲染出的整次 `--learn` 运行的谱系树。 |
| `sources/workflows/<uuid>/reward_progress.png` | 分数随迭代变化的曲线。 |
| `runs_capsule/<capsule_name>/` | 用于分享或重新运行的最终工作空间归档快照。 |

完整布局:[`docs/usage/transparency.md`](./docs/usage/transparency.md)、[`docs/usage/workspace.md`](./docs/usage/workspace.md)。

---

## 配置

将 `config_default.json` 复制为 `my_config.json` 并编辑。你最常需要调整的字段:

| 字段 | 控制内容 |
|-------|------------------|
| `workspace_dir` | 共享工作空间 —— 所有生成的文件均出现在此 |
| `discovery_addresses` | MCP 发现的 IP 与端口范围 |
| `workflow_llm_model` | 合成多智能体工作流(例如 `anthropic/claude-opus-4-5`) |
| `smolagent_model_id` | 执行智能体所使用的模型 |
| `judge_model` | 编写验证器程序并给出软性裁定的 LLM |
| `learned_score_threshold` | `--learn` 模式下的早停阈值(默认 `0.97`) |
| `max_learning_evolve_iterations` | 代数上限(默认 `35`) |
| `population_size` / `novelty_weight` / `min_improvement_threshold` | QD 归档调参 |

完整参考:[`docs/reference/configuration.md`](./docs/reference/configuration.md)。

---

## 评估

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

> ⚠️ 为了进行无偏评估,请先运行 `./cleanup.sh`,防止 Mimosa 复用已缓存的工作流。

每个基准的配置细节:[`docs/science_agent_bench_evaluation.md`](./docs/science_agent_bench_evaluation.md)、[`docs/papers_bench_evaluation.md`](./docs/papers_bench_evaluation.md)。

---

## 通知与遥测

- **Pushover** —— 将实时进度推送到你的手机。设置 `PUSHOVER_USER` 与 `PUSHOVER_TOKEN`。详情:[`docs/usage/notifications.md`](./docs/usage/notifications.md)。
- **Langfuse** —— 对每次 LLM 调用进行 span 级别的追踪。在 Langfuse 仓库中执行 `docker compose up -d`,然后将 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_PRIVATE_KEY` 添加到 `.env`。仪表盘位于 `http://localhost:3000`。详情:[`docs/usage/telemetry.md`](./docs/usage/telemetry.md)。

---

## 相关工作

Mimosa-AI 处于一条规模虽小但活跃的 LLM 驱动程序搜索与自主研究系统的脉络中。我们并不声称取代其中任何一个 —— 它们回答的是不同的问题:

| 项目 | 作用 | Mimosa 的差异 |
|---------|--------------|--------------------|
| [Sakana AI Scientist](https://github.com/SakanaAI/AI-Scientist) | 机器学习领域的端到端论文生成 | Mimosa 以 QD + 验证器对**每任务的工作流合成**进行优化,而非完整论文生成 |
| [DiscoPOP](https://github.com/SakanaAI/DiscoPOP) (Lange et al. 2024) | LLM 驱动的偏好优化算法发现 | 同样的"LLM 作为代码变异算子"范式;Mimosa 将其应用于多智能体工作流代码,而非损失函数 |
| [FunSearch](https://github.com/google-deepmind/funsearch) (Romera-Paredes et al. 2024) | 在 LLM 指导下对 Python 函数进行的进化搜索 | Mimosa 进化的是完整的多智能体程序,并以多源逐声明验证器替代单一适应度函数 |
| [ELM](https://github.com/CarperAI/OpenELM) (Lehman et al. 2022) | 由 LLM 中介的代码质量-多样性 | 最接近的 QD 祖先;Mimosa 的行为描述符是工作流结构性的,而非领域特定的 |
| AIDE | 在类 Kaggle 任务上的自动化 ML 流水线 | Mimosa 面向更广泛的科学复现(ScienceAgentBench、PaperBench、实验室数据),并提供可审计的逐声明验证器 |

若你在发表对比性工作,[手稿](https://arxiv.org/abs/2603.28986)中包含详细的定位说明。

---

## 完整文档

```bash
uvx --with mkdocs-material mkdocs serve   # live preview at http://localhost:8000
uvx --with mkdocs-material mkdocs build   # static HTML to ./site
```

站点配置:[`mkdocs.yml`](./mkdocs.yml)。索引:[`docs/index.md`](./docs/index.md)。

---

## 贡献

欢迎补丁、MCP 工具、评估器以及新的声明源。请从 [`CONTRIBUTING.md`](./CONTRIBUTING.md)、[Developer guide](./docs/DEVELOPER_GUIDE.md) 以及 [`CLA/`](./CLA/) 中的贡献条款开始。

---

## 许可证

Apache 2.0。请参见 [`NOTICE`](./NOTICE)、[`docs/licensing-notes.md`](./docs/licensing-notes.md) 以及 [`CLA/`](./CLA/) 文件夹中的贡献条款。

---

## 引用本工作

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
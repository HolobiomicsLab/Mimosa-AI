<div align="center">
<br>

<img src="./docs/images/logo_mimosa.png" width="22%" style="border-radius: 8px;" alt="Mimosa-AI Logo">

</div>

<h1 align="center">Mimosa-AI 🌼🔬</h1>

<p align="center">
  <a href="./README.md"> English</a> &nbsp;|&nbsp;
  <a href="./README.CHS.md"> 简体中文</a> &nbsp;|&nbsp;
  <a href="./README.CHT.md"> 繁體中文</a> &nbsp;|&nbsp;
  <a href="./README.JPN.md"> 日本語</a> &nbsp;|&nbsp;
  <a href="./README.KOR.md"> 한국어</a>
</p>

<p align="center">
    <em>一个用于演化自主 AI 科学家工作流的开源框架。</em>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2603.28986"><img src="https://img.shields.io/badge/arXiv-2603.28986-b31b1b.svg?logo=arxiv&style=flat-square&logoColor=white" alt="arXiv"></a>
    <a href="https://holobiomicslab.cnrs.fr/"><img src="https://img.shields.io/badge/网站-holobiomicslab.cnrs.fr-4caf82?style=flat-square&logo=globe&logoColor=white" alt="网站"></a>
</p>

<p align="center">
    <a href="https://github.com/HolobiomicsLab/Mimosa-AI/stargazers">
        <img src="https://img.shields.io/github/stars/HolobiomicsLab/Mimosa-AI?style=social" alt="GitHub Stars">
    </a>
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img src="https://img.shields.io/badge/许可证-Apache%202.0-blue.svg?style=flat-square" alt="许可证：Apache 2.0">
    </a>
</p>

---

> ***Mimosa-AI 🌼*** — 如同能感知、学习与适应的含羞草植物，Mimosa 是一个 AI 科学家框架，旨在自主完成端到端的科学研究并复现已发表的研究成果。***Mimosa-AI*** 能够自动发现可用工具，将研究目标分解为结构化工作流，并通过多智能体工作流的进化式自我迭代驱动多智能体执行。

**应用场景：**
- 通过严格、可审计的工作流复现科学研究
- 自动化完整流水线：生物信息学、分子对接、代谢组学等

## 目录

- [工作原理](#工作原理)
- [示例：复现生物活性分子网络论文](#示例复现生物活性分子网络论文)
- [前置条件](#前置条件)
- [安装](#安装)
- [配置](#配置)
- [运行 Mimosa](#运行-mimosa)
- [输出](#输出)
- [通过多智能体工作流的进化进行学习](#通过多智能体工作流的进化进行学习)
- [透明度](#透明度)
- [命令行参数](#命令行参数)
- [评估](#评估)
- [手机通知](#手机通知)
- [遥测配置](#遥测配置)
- [许可证](#许可证)

## 工作原理

用户向 ***Mimosa-AI*** 提供一个研究目标。

- ***Mimosa*** 自动发现本地网络上或通过 Toolhive 提供的基于 MCP 的工具（涵盖从数据分析工具到网页浏览器，乃至质谱仪等实验室仪器的一切工具）。
- 基于用户目标和已发现的工具，***Mimosa*** 对问题进行分解，为每个任务构建量身定制的多智能体工作流。
- 每个任务自主运行，失败案例通过迭代学习循环用于自我改进。
- ***Mimosa*** 最终生成包含结果、可视化内容、报告、日志及所有相关产出物的完整胶囊。

![schema](./docs/images/mimosa_overall.jpg)

## 示例：复现生物活性分子网络论文

> **目标** — 端到端复现 [Nothias et al. (2018)](https://www.researchgate.net/publication/323525305_Bioactivity-Based_Molecular_Networking_for_the_Discovery_of_Drug_Leads_in_Natural_Product_Bioassay-Guided_Fractionation)：从特征检测到网络可视化，以 `.mzML` 文件作为起点（假设原始数据转换已完成）。

Mimosa-AI 获得了该论文、原始数据，以及一套涵盖论文中未完整说明的配置细节的领域专属技能。

**执行延时演示**

https://github.com/user-attachments/assets/dcd04ade-9c43-44a8-b3e3-a999d3dc895d

**输出结果**

![molecular_network](./docs/images/network.png)

所得分子网络与原论文报告的拓扑结构相符，包括聚类分离和边权重均由系统自主复现。

---

## 前置条件

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)（推荐）或 pip
- 正在运行的 [Toolomics MCP 服务器](https://github.com/HolobiomicsLab/toolomics)

---

## 安装

### 1. 克隆仓库并创建虚拟环境

```bash
# 使用 uv（推荐）
pip install uv
uv venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 或使用 pip
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
cd mimosa
uv pip install -r requirements.txt
```

### 3. 设置 API 密钥

在项目根目录创建 `.env` 文件，仅包含您计划使用的 LLM 提供商的密钥：

```env
ANTHROPIC_API_KEY=...       # Claude — 推荐用于工作流编排
OPENAI_API_KEY=...          # OpenAI 模型 - 可选
MISTRAL_API_KEY=...         # Mistral 模型 - 可选
DEEPSEEK_API_KEY=...        # Deepseek - 可选
HF_TOKEN=...                # HuggingFace 提供商，可选
OPENROUTER_API_KEY=...      # 通过 OpenRouter 访问任意模型

# 可选 — 通过 Langfuse 进行可观测性监控
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_PRIVATE_KEY=...
```

### 4. 启动 MCP 服务器

请参考 [HolobiomicsLab/toolomics](https://github.com/HolobiomicsLab/toolomics) 的配置说明，将其配置为在某个端口范围内运行（例如 `5000–5100`）。自定义 MCP 工具可通过 [Toolomics 文档](https://github.com/HolobiomicsLab/toolomics/README.md) 添加。

---

## 配置

```bash
cp config_default.json my_config.json
```

编辑 `my_config.json`，关键参数说明：

| 参数 | 描述 |
|------|------|
| `workspace_dir` | Toolomics 工作区路径 — 所有生成文件均保存于此 |
| `discovery_addresses` | MCP 服务器发现的 IP 及端口范围 |
| `planner_llm_model` | 用于任务分解与规划的 LLM |
| `prompts_llm_model` | 用于工作流提示生成的 LLM |
| `workflow_llm_model` | 用于多智能体编排的 LLM（推荐：anthropic/claude-opus-4-5 或 z-ai/glm-5） |
| `smolagent_model_id` | 用于 SmolAgents 执行子任务的模型 |
| `judge_model` | 用于输出自我评估和评分的 LLM |
| `learned_score_threshold` | 接受结果并停止迭代的最低分数 |
| `max_learning_evolve_iterations` | 接受结果前的最大自我改进迭代次数 |

---

## 运行 Mimosa

Mimosa 支持两种执行模式：**目标（Goal）** 和 **任务（Task）**。

### 目标模式 — 多步骤科学目标

当您的目标需要跨多个不同操作进行规划时使用（例如复现论文、构建机器学习流水线）。

```bash
uv run main.py --goal "您的科学目标" --config my_config.json
```

**示例：**
```bash
uv run main.py \
  --goal "复现《Dual Aggregation Transformer for Image Super-Resolution》(https://arxiv.org/pdf/2306.00306) 中的实验并比较结果。" \
  --config my_config.json

uv run main.py \
  --goal "开发一个用于预测蛋白质-配体结合亲和力的机器学习模型。" \
  --config my_config.json
```

### 任务模式 — 单一粒度操作

适用于不需要长期规划的专注型、自包含操作。

```bash
uv run main.py --task "您的任务描述" --config my_config.json
```

**示例：**
```bash
uv run main.py \
  --task "在 Clintox 数据集上训练一个多任务模型，以预测药物毒性和 FDA 批准状态。" \
  --config my_config.json

uv run main.py --task "对图神经网络在药物发现中的应用进行文献综述。" --config my_config.json
```

> **注意：** 执行任何模式前，必须先安装 Toolomics 并确保 MCP 服务器正在运行。

## 输出

执行过程中，文件将写入 Toolomics 的 `workspace/` 目录。执行完成后，完整快照将保存至 `runs_capsule/` 中带时间戳的胶囊目录。

---

## 通过多智能体工作流的进化进行学习

***Mimosa-AI*** 能够为科学任务动态合成专用工作流，并通过**达尔文式多智能体工作流进化**从失败中学习：
系统不再将任务强行套入固定流水线，而是为每个任务组合定制的多智能体图，并通过单候选局部搜索对其进行优化。在每次迭代中，仅由表现最优的工作流生成继承者，且只保留改进结果。随着时间推移，系统积累了一个经过验证的工作流库，使类似的未来任务能够从强基线出发，而非从零开始。

对于任何新任务，建议先以学习模式运行，让系统在获得完全自主权之前积累能力。

**启动学习模式**

```bash
uv run main.py --task "在 Clintox 数据集上训练一个多任务模型，以预测药物毒性和 FDA 批准状态" --learn --config my_config.json
```

![dgm](./docs/images/workflow_mutation.png)

**进度可视化：**

***Mimosa-AI*** 完成某任务的学习阶段后，您可以可视化其随时间的改进过程。展示各次尝试中性能提升的奖励进度图将自动保存。

奖励进度图保存于 `sources/workflows/<uuid>` 文件夹下，文件名为 `reward_progress.png`。

***示例：***

![dgm](./docs/images/evolve_example.png)

## 透明度

我们内置了交互式调试器 `memory_explorer.py`，允许您以精细粒度逐步回溯任意智能体执行过程。

使用工作流 `<uuid>` 启动（例如：`memory_explorer.py 20260115_113303_9bb63437`）

该工具将重放完整的执行轨迹——包括思维过程、工具调用及输出结果——以便您精确检查决策是如何展开的。

## 命令行参数

### 执行模式

| 参数 | 描述 |
|------|------|
| `--goal GOAL` | 指定高层次研究目标、论文复现任务或科学问题（规划器模式） |
| `--task TASK` | 执行单一任务：文献综述、数据集下载、实现机器学习模型等 |
| `--manual` | 交互式 CLI 模式，用于调试 MCP 并直接测试 ***Mimosa*** 工具 |
| `--papers <CSV 路径>` | 在包含研究论文和提示的 CSV 数据集上进行评估 |
| `--science_agent_bench` | 在 ScienceAgentBench 上进行评估 |

### 其他参数

| 参数 | 描述 |
|------|------|
| `--learn` | 启用迭代学习以优化任务性能 |
| `--max_evolve_iterations N` | 最大学习迭代次数 |
| `--csv_runs_limit N` | 限制评估的 CSV 条目数量 |
| `--scenario <场景文件名>` | 使用基于特定场景断言的评分方式，而非 LLM 作为评判者 |
| `--single_agent` | 单智能体模式，速度快，但无法通过学习自我改进 |
| `--debug` | 启用调试模式以获取更详细的日志 |

---

## 评估

***Mimosa-AI*** 可在 [ScienceAgentBench](https://arxiv.org/abs/2410.05080) 或 [PaperBench](https://arxiv.org/pdf/2504.01848) 上进行评估。

⚠️ 为确保评估公正，建议先运行 `./cleanup.sh`，以防止 ***Mimosa*** 使用已有或缓存的工作流。

### ScienceAgentBench

在 ScienceAgentBench 上进行评估，您需要：

1. 下载 ScienceAgentBench 完整数据集：
[数据集链接](https://buckeyemailosu-my.sharepoint.com/personal/chen_8336_buckeyemail_osu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch%2Fbenchmark%2Ezip&parent=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch&ga=1)
2. 使用密码 `scienceagentbench` 解压
3. 将 `benchmark/benchmark/datasets` 文件夹的内容复制到 `Mimosa-AI/datasets/scienceagentbench/datasets`

**启用学习模式在 ScienceAgentBench 上评估**
```sh
uv run main.py --science_agent_bench --learn
```

**限制 10 个任务且学习迭代次数不超过 4 次的 ScienceAgentBench 评估**

```sh
uv run main.py --science_agent_bench --csv_runs_limit 10 --max_evolve_iterations 4
```

### PaperBench

**启用学习模式在 OpenAI PaperBench 上评估**

OpenAI PaperBench 是一个用于评估 AI 智能体复现 AI 研究能力的基准，来源于论文 `PaperBench: Evaluating AI's Ability to Replicate AI Research`。

```sh
uv run main.py --papers datasets/paper_bench.csv --csv_runs_limit 20  --learn
```

⚠️ 此操作将在 `runs_capsule/` 文件夹中保存所有论文复现尝试的结果，完整评估请参考 [Paper Bench 文档](https://github.com/openai/frontier-evals/tree/main/project/paperbench)。

**在自定义研究论文基准上评估**

1. 将与 `paper_bench.csv` 格式相同的基准 CSV 文件放置于 `datasets/<您的基准名称>.csv`。

2. 在您的基准上运行：

```sh
uv run main.py --papers datasets/<您的基准名称>.csv --csv_runs_limit 20  --learn
```

---

## 手机通知

通过 Pushover 通知接收 ***Mimosa*** 状态的实时更新。

### 配置步骤

1. **创建 Pushover 账户**
   - 访问 [pushover.net](https://pushover.net/)
   - 注册并记录您的**用户密钥（User Key）**

2. **创建应用程序**
   - 在 Pushover 控制台中，点击"创建应用程序/API Token"
   - 命名为"***Mimosa***"并复制生成的 **API Token**

3. **配置环境变量**
   ```bash
   export PUSHOVER_USER="您的用户密钥"
   export PUSHOVER_TOKEN="您的 API Token"
   ```

4. **安装移动应用**
   - 从设备应用商店下载 Pushover
   - 使用您的 Pushover 账户登录

---

## 遥测配置

使用 Langfuse 通过实时可观测性仪表板监控和调试 AI 智能体。

### 快速开始

1. **本地部署 Langfuse**
   ```bash
   git clone https://github.com/langfuse/langfuse.git
   cd langfuse
   docker compose up -d
   ```

2. **配置环境变量**

   在您的 `.env` 文件中添加：
   ```env
   LANGFUSE_PUBLIC_KEY=您的公钥
   LANGFUSE_PRIVATE_KEY=您的私钥
   ```

3. **访问仪表板**

   ***Mimosa-AI*** 运行时，访问 `http://localhost:3000`

### 可用指标

遥测仪表板提供：
- **智能体执行追踪**：逐步工作流可视化
- **性能指标**：响应时间和成功率
- **错误调试**：详细故障分析
- **资源使用情况**：Token 消耗和 API 调用统计

**仪表板示例：**
![Langfuse Dashboard](https://langfuse.com/images/cookbook/integration-smolagents/smolagent_example_trace.png)

> **注意：** 遥测为可选功能，但推荐用于调试和性能优化。

---

## 许可证

本仓库在 Apache License 2.0 下公开发布。Apache 2.0 是一个宽松的开源许可证，允许商业使用、修改和再分发，但要求再分发时保留适用的许可证、版权、专利、商标、署名及 NOTICE 声明，且修改后的文件须附带显著说明，注明已做出的更改。

有关贡献与许可细节，请参阅：
- `NOTICE`
- `docs/licensing-notes.md`
- `CLA/INDIVIDUAL_CLA.md`
- `CLA/EMPLOYER_AUTHORIZATION.md`

---

<div align="center">
<br>

<img src="./docs/images/logo_mimosa.png" width="22%" style="border-radius: 8px;" alt="Mimosa-AI Logo">

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
    <em>自主科學研究的自演化AI框架</em>
</p>
<p align="center">
  🧬 自演化多智能體工作流 &nbsp;·&nbsp;
  🔍 基於MCP的工具自動發現 &nbsp;·&nbsp;
  🔁 達爾文式工作流優化 &nbsp;·&nbsp;
  📦 完整審計追蹤與可重現性 &nbsp;·&nbsp;
</p>

<p align="center">
    <em>Mimosa-AI 自主完成了 Nothias et al. (2018) 的端到端重現——從原始 .mzML 檔案到分子網路——僅需一條命令。</em>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2603.28986"><img src="https://img.shields.io/badge/arXiv-2603.28986-b31b1b.svg?logo=arxiv&style=flat-square&logoColor=white" alt="arXiv 預印本"></a>
    <a href="https://doi.org/10.48550/arXiv.2603.28986"><img src="https://img.shields.io/badge/DOI-10.48550%2FarXiv.2603.28986-blue?style=flat-square" alt="DOI"></a>
    <a href="https://holobiomicslab.cnrs.fr/"><img src="https://img.shields.io/badge/網站-holobiomicslab.cnrs.fr-4caf82?style=flat-square&logo=globe&logoColor=white" alt="網站"></a>
</p>

<p align="center">
    <a href="https://github.com/HolobiomicsLab/Mimosa-AI/stargazers">
        <img src="https://img.shields.io/github/stars/HolobiomicsLab/Mimosa-AI?style=social" alt="GitHub Stars">
    </a>
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img src="https://img.shields.io/badge/授權條款-Apache%202.0-blue.svg?style=flat-square" alt="授權條款：Apache 2.0">
    </a>
</p>

---

## 即時示範：自主重現科學論文

https://github.com/user-attachments/assets/dcd04ade-9c43-44a8-b3e3-a999d3dc895d

**結果：** 下方的分子網路從原始 `.mzML` 檔案出發自主重現，與 [Nothias et al. (2018)](https://www.researchgate.net/publication/323525305_Bioactivity-Based_Molecular_Networking_for_the_Discovery_of_Drug_Leads_in_Natural_Product_Bioassay-Guided_Fractionation) 報告的拓撲結構完全吻合——包括叢集分離和邊權重。

<p align="center">
  <img src="./docs/images/network.png" alt="重現的分子網路" width="80%">
</p>

---

## 基準測試結果

在 **ScienceAgentBench** 上的評估結果（102 個任務，`task` 模式）：

| 模式 | 成功率 | Code-BLEU 分數 | 每任務成本 |
|------|--------|----------------|-----------|
| DeepSeek-V3.2 單智能體 | 38.2% | 0.898 | $0.05 |
| DeepSeek-V3.2 一次性多智能體 | 32.4% | 0.794 | $0.38 |
| **DeepSeek-V3.2 迭代學習** | **43.1%** | **0.921** | **$1.7** |

> 迭代學習可提升 GPT-4o 的效能，但對 Claude Haiku 4.5 影響輕微——詳見[論文](https://arxiv.org/abs/2603.28986)中的模型相關行為分析。

---

## Mimosa-AI 是什麼？

> ***Mimosa-AI 🌼*** — 如同能感知、學習與適應的含羞草植物，Mimosa 是一個用於自主科學研究的開源框架，能自動合成任務專屬的多智能體工作流程，並透過執行回饋持續優化。基於 MCP 工具發現、程式碼生成智能體和 LLM 評估，為學術研究提供模組化、可稽核的替代方案。

**核心能力：**
- **重現科學研究**，具備可追蹤性和嚴謹性——從原始資料到可發表的圖表
- **自動化計算流水線**，涵蓋生物資訊學、分子對接、代謝體學、機器學習等領域
- **自我進化**，透過達爾文式工作流程變異——每次失敗都為下一次嘗試提供資訊

### 架構概覽

框架分為五層：

1. **規劃層**（可選）——將高層次科學目標分解為離散任務
2. **工具發現層**——透過 Toolomics 自動發現本地網路上的 MCP 工具
3. **元編排層**——合成任務專屬的多智能體工作流程，將工具分配給專屬智能體
4. **智能體執行層**——程式碼生成智能體使用發現的工具和科學函式庫執行子任務
5. **評判/評估層**——LLM 評判對輸出評分；在學習模式下驅動迭代工作流程優化

<p align="center">
  <img src="./docs/images/mimosa_overall.jpg" alt="Mimosa 架構概覽" width="90%">
</p>

在基準測試 `task` 模式下，規劃層（1）被跳過，以便單獨評估工作流程合成和優化。

---

## 目錄

- [Toolomics 是什麼？我需要它嗎？](#toolomics-是什麼我需要它嗎)
- [前置條件](#前置條件)
- [安裝](#安裝)
- [設定](#設定)
- [執行 Mimosa](#執行-mimosa)
- [工作區與稽核追蹤](#工作區與稽核追蹤)
- [透過多智能體工作流程演化進行學習](#透過多智能體工作流程演化進行學習)
- [透明度](#透明度)
- [命令列參數](#命令列參數)
- [評估](#評估)
- [手機通知](#手機通知)
- [遙測設定](#遙測設定)
- [授權條款](#授權條款)

---

## Toolomics 是什麼？我需要它嗎？

**[Toolomics](https://github.com/HolobiomicsLab/toolomics)** 是 Mimosa 的配套平台，用於 MCP 伺服器管理。它將科學工具（資料分析工具、網路服務、實驗室儀器）作為可發現的 MCP 服務公開，提供 Mimosa 讀寫任務產出物的共享工作區，並允許在不修改 Mimosa 核心的情況下註冊自訂工具。

**是否必須安裝？** 是——在執行任何 Mimosa 模式前，Toolomics 必須處於執行狀態。好消息是：安裝設定只需幾分鐘。

- Mimosa 和 Toolomics 均採用 Apache 2.0 授權條款，完全免費。
- Toolomics 在本地可設定的連接埠範圍內執行（預設 `5000–5100`）。
- 可透過 [Toolomics 文件](https://github.com/HolobiomicsLab/toolomics)新增自訂 MCP 工具。

> **快速啟動路徑：** 複製 Toolomics → 在預設連接埠範圍啟動 → 執行 Mimosa。除 LLM API 金鑰外，無需任何雲端帳戶或付費服務。

---

## 前置條件

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)（建議）或 pip
- 正在執行的 [Toolomics MCP 伺服器](https://github.com/HolobiomicsLab/toolomics)

---

## 安裝

### 1. 安裝相依套件

```bash
# 使用 uv（建議——一步完成虛擬環境建立和相依套件安裝）
pip install uv
uv sync

# 或使用 pip
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install .
```

### 2. 設定 API 金鑰

在專案根目錄建立 `.env` 檔案，僅包含您計畫使用的 LLM 提供商的金鑰：

```env
ANTHROPIC_API_KEY=...       # Claude — 建議用於工作流程編排
OPENAI_API_KEY=...          # OpenAI 模型 - 可選
MISTRAL_API_KEY=...         # Mistral 模型 - 可選
DEEPSEEK_API_KEY=...        # Deepseek - 可選
HF_TOKEN=...                # HuggingFace 提供商，可選
OPENROUTER_API_KEY=...      # 透過 OpenRouter 存取任意模型

# 可選 — 透過 Langfuse 進行可觀測性監控
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_PRIVATE_KEY=...
```

### 3. 啟動 MCP 伺服器

請參考 [HolobiomicsLab/toolomics](https://github.com/HolobiomicsLab/toolomics) 的設定說明，將其設定為在某個連接埠範圍內執行（例如 `5000–5100`）。

自訂 MCP 工具可透過 [Toolomics 文件](https://github.com/HolobiomicsLab/toolomics/README.md) 新增。

---

## 設定

```bash
cp config_default.json my_config.json
```

編輯 `my_config.json`，關鍵參數說明：

| 參數 | 描述 |
|------|------|
| `workspace_dir` | Toolomics 工作區路徑 — 所有產生的檔案均儲存於此 |
| `discovery_addresses` | MCP 伺服器探索的 IP 及連接埠範圍 |
| `planner_llm_model` | 用於任務分解與規劃的 LLM |
| `prompts_llm_model` | 用於工作流程提示產生的 LLM |
| `workflow_llm_model` | 用於多智能體編排的 LLM（建議：`anthropic/claude-opus-4-5` 或 `z-ai/glm-5`） |
| `smolagent_model_id` | 用於 SmolAgents 執行子任務的模型 |
| `judge_model` | 用於輸出自我評估和評分的 LLM |
| `learned_score_threshold` | 接受結果並停止迭代的最低分數 |
| `max_learning_evolve_iterations` | 接受結果前的最大自我改進迭代次數 |

---

## 執行 Mimosa

Mimosa 支援兩種執行模式：**目標（Goal）** 和 **任務（Task）**。

### 目標模式 — 多步驟科學目標

當您的目標需要跨多個不同操作進行規劃時使用（例如重現論文、構建機器學習流水線）。

```bash
uv run main.py --goal "您的科學目標" --config my_config.json
```

**範例：**
```bash
uv run main.py \
  --goal "重現《Dual Aggregation Transformer for Image Super-Resolution》(https://arxiv.org/pdf/2306.00306) 中的實驗並比較結果。" \
  --config my_config.json

uv run main.py \
  --goal "開發一個用於預測蛋白質-配體結合親和力的機器學習模型。" \
  --config my_config.json
```

### 任務模式 — 單一粒度操作

適用於不需要長期規劃的專注型、自包含操作。

```bash
uv run main.py --task "您的任務描述" --config my_config.json
```

**範例：**
```bash
uv run main.py \
  --task "在 Clintox 資料集上訓練一個多任務模型，以預測藥物毒性和 FDA 核准狀態。" \
  --config my_config.json

uv run main.py --task "對圖神經網路在藥物探索中的應用進行文獻綜述。" --config my_config.json
```

> **基準測試說明：** 論文中報告的結果在 `task` 模式下測量，規劃層被停用，以單獨評估工作流程合成和迭代優化。
>
> **注意：** 執行任何模式前，必須先安裝 Toolomics 並確保 MCP 伺服器正在執行。

---

## 工作區與稽核追蹤

執行過程中，檔案將寫入 `workspace_dir` 設定的 Toolomics 工作區。執行完成後，工作區內容被複製到 `runs_capsule/` 下帶時間戳記的資料夾，作為存檔永久保存。

- **Toolomics `workspace/`** — 即時工作目錄：中間檔案、腳本、下載內容、產生的輸出
- **`sources/workflows/<uuid>/`** — 產生的工作流程和執行元資料：`state_result.json`、`evaluation.txt`、`reward_progress.png`、`memory/` 追蹤
- **`runs_capsule/<capsule_name>/`** — 執行快照存檔，可供後續檢查、比較或分享
- **`memory_explorer.py <uuid>`** — 逐步回放工作流程執行，檢查智能體追蹤、工具呼叫和輸出

這些位置共同構成 Mimosa 的完整稽核追蹤：規劃、執行、評估和產出的全過程。

---

## 透過多智能體工作流程演化進行學習

***Mimosa-AI*** 是一個**自我進化的多智能體系統**，能為科學任務動態合成專用工作流程。系統透過**達爾文式單候選局部搜尋**進化工作流程：在每次迭代中，僅由表現最優的工作流程產生繼承者，且只保留改進結果。隨著時間推移，系統積累了一個經過驗證的工作流程庫，使類似的未來任務能夠從強基準出發，而非從零開始。

對於任何新任務，建議**先以學習模式執行**，讓系統在獲得完全自主權之前積累能力。

**啟動學習模式**

```bash
uv run main.py --task "在 Clintox 資料集上訓練一個多任務模型，以預測藥物毒性和 FDA 核准狀態" --learn --config my_config.json
```

<p align="center">
  <img src="./docs/images/workflow_mutation.png" alt="工作流程變異示意圖" width="80%">
</p>

**進度視覺化：**

***Mimosa-AI*** 完成學習階段後，獎勵進度圖（各次嘗試的效能提升）將自動儲存至 `sources/workflows/<uuid>/reward_progress.png`。

<p align="center">
  <img src="./docs/images/evolve_example.png" alt="獎勵進度範例" width="80%">
</p>

---

## 透明度

我們內建了互動式除錯器 `memory_explorer.py`，允許以精細粒度逐步回溯任意智能體執行過程。

```bash
python memory_explorer.py 20260115_113303_9bb63437
```

該工具將重播完整的執行軌跡——包括思考過程、工具呼叫及輸出結果——以便精確檢查每個決策是如何展開的。

---

## 命令列參數

### 執行模式

| 參數 | 描述 |
|------|------|
| `--goal GOAL` | 指定高層次研究目標、論文重現任務或科學問題（規劃器模式） |
| `--task TASK` | 執行單一任務：文獻綜述、資料集下載、實作機器學習模型等 |
| `--manual` | 互動式 CLI 模式，用於除錯 MCP 並直接測試 ***Mimosa*** 工具 |
| `--papers <CSV 路徑>` | 在包含研究論文和提示的 CSV 資料集上進行評估 |
| `--science_agent_bench` | 在 ScienceAgentBench 上進行評估 |

### 其他參數

| 參數 | 描述 |
|------|------|
| `--learn` | 啟用迭代學習以優化任務效能 |
| `--max_evolve_iterations N` | 最大學習迭代次數 |
| `--csv_runs_limit N` | 限制評估的 CSV 條目數量 |
| `--scenario <情境檔案名稱>` | 使用基於特定情境斷言的評分方式，而非 LLM 作為評判者 |
| `--single_agent` | 單智能體模式，速度快，但無法透過學習自我改進 |
| `--debug` | 啟用除錯模式以獲取更詳細的日誌 |

---

## 評估

***Mimosa-AI*** 可在 [ScienceAgentBench](https://arxiv.org/abs/2410.05080) 或 [PaperBench](https://arxiv.org/pdf/2504.01848) 上進行評估。

⚠️ 為確保評估公正，請先執行 `./cleanup.sh`，以防止 ***Mimosa*** 使用快取的工作流程。

### ScienceAgentBench

1. 下載 ScienceAgentBench 完整資料集：
   [資料集連結](https://buckeyemailosu-my.sharepoint.com/personal/chen_8336_buckeyemail_osu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch%2Fbenchmark%2Ezip&parent=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch&ga=1)
2. 使用密碼 `scienceagentbench` 解壓縮
3. 將 `benchmark/benchmark/datasets/` 複製到 `Mimosa-AI/datasets/scienceagentbench/datasets/`

**啟用學習模式進行完整評估：**
```sh
uv run main.py --science_agent_bench --learn
```

**快速評估（10 個任務，4 次學習迭代）：**
```sh
uv run main.py --science_agent_bench --csv_runs_limit 10 --max_evolve_iterations 4
```

### PaperBench

OpenAI PaperBench 用於評估 AI 智能體重現 AI 研究的能力（*PaperBench: Evaluating AI's Ability to Replicate AI Research*）。

```sh
uv run main.py --papers datasets/paper_bench.csv --csv_runs_limit 20 --learn
```

⚠️ 結果儲存於 `runs_capsule/`。完整評估請參考 [PaperBench 文件](https://github.com/openai/frontier-evals/tree/main/project/paperbench)。

**自訂基準測試：**

```sh
uv run main.py --papers datasets/<您的基準名稱>.csv --csv_runs_limit 20 --learn
```

---

## 手機通知

透過 Pushover 通知接收 ***Mimosa*** 狀態的即時更新。

### 設定步驟

1. 建立 [Pushover](https://pushover.net/) 帳戶並記錄**使用者金鑰**
2. 建立名為「Mimosa」的應用程式——複製 **API Token**
3. 匯出環境變數：
   ```bash
   export PUSHOVER_USER="您的使用者金鑰"
   export PUSHOVER_TOKEN="您的 API Token"
   ```
4. 安裝 Pushover 行動應用程式並登入

---

## 遙測設定

使用 Langfuse 透過即時可觀測性儀表板監控和除錯 AI 智能體。

### 快速開始

1. **本地部署 Langfuse：**
   ```bash
   git clone https://github.com/langfuse/langfuse.git
   cd langfuse
   docker compose up -d
   ```

2. **新增到 `.env`：**
   ```env
   LANGFUSE_PUBLIC_KEY=您的公開金鑰
   LANGFUSE_PRIVATE_KEY=您的私密金鑰
   ```

3. **存取儀表板**：***Mimosa-AI*** 執行時，造訪 `http://localhost:3000`

儀表板提供智能體執行追蹤、效能指標、錯誤除錯和 Token/API 使用統計。

> **注意：** 遙測為可選功能，但建議用於除錯和效能優化。

---

## 授權條款

本儲存庫在 Apache License 2.0 下公開發布。有關貢獻與授權細節，請參閱：
- `NOTICE`
- `docs/licensing-notes.md`
- `CLA/INDIVIDUAL_CLA.md`
- `CLA/EMPLOYER_AUTHORIZATION.md`

---

## 引用

<p align="center">
<b>引用：</b> <em><a href="https://arxiv.org/abs/2603.28986">Mimosa Framework: Toward Evolving Multi-Agent Systems for Scientific Research</a></em><br>
M. Legrand, T. Jiang, M. Feraud, B. Navet, Y. Taghzouti, F. Gandon, E. Dumont, L.-F. Nothias — <em>arXiv:2603.28986, 2026</em> — <a href="https://doi.org/10.48550/arXiv.2603.28986">DOI</a>
</p>

```bibtex
@article{legrand2026mimosa,
  title={Mimosa Framework: Toward Evolving Multi-Agent Systems for Scientific Research},
  author={Legrand, Martin and Jiang, Tao and Feraud, Matthieu and Navet, Benjamin and Taghzouti, Yousouf and Gandon, Fabien and Dumont, Elise and Nothias, Louis-F{\'e}lix},
  journal={arXiv preprint arXiv:2603.28986},
  year={2026}
}
```

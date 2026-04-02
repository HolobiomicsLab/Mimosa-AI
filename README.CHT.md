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
    <em>一個用於演化自主 AI 科學家工作流程的開源框架。</em>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2603.28986"><img src="https://img.shields.io/badge/arXiv-2603.28986-b31b1b.svg?logo=arxiv&style=flat-square&logoColor=white" alt="arXiv"></a>
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

> ***Mimosa-AI 🌼*** — 如同能感知、學習與適應的含羞草植物，Mimosa 是一個 AI 科學家框架，旨在自主完成端到端的科學研究並重現已發表的研究成果。***Mimosa-AI*** 能夠自動發現可用工具，將研究目標分解為結構化工作流程，並透過多智能體工作流程的演化式自我迭代驅動多智能體執行。

**應用場景：**
- 透過嚴謹、可稽核的工作流程重現科學研究
- 自動化完整流水線：生物資訊學、分子對接、代謝體學等

## 目錄

- [運作原理](#運作原理)
- [範例：重現生物活性分子網路論文](#範例重現生物活性分子網路論文)
- [前置條件](#前置條件)
- [安裝](#安裝)
- [設定](#設定)
- [執行 Mimosa](#執行-mimosa)
- [輸出](#輸出)
- [透過多智能體工作流程的演化進行學習](#透過多智能體工作流程的演化進行學習)
- [透明度](#透明度)
- [命令列參數](#命令列參數)
- [評估](#評估)
- [手機通知](#手機通知)
- [遙測設定](#遙測設定)
- [授權條款](#授權條款)

## 運作原理

使用者向 ***Mimosa-AI*** 提供一個研究目標。

- ***Mimosa*** 自動發現本地網路上或透過 Toolhive 提供的 MCP 工具（涵蓋從資料分析工具到網頁瀏覽器，乃至質譜儀等實驗室儀器的一切工具）。
- 基於使用者目標和已發現的工具，***Mimosa*** 對問題進行分解，為每個任務構建量身定製的多智能體工作流程。
- 每個任務自主執行，失敗案例透過迭代學習迴圈用於自我改進。
- ***Mimosa*** 最終產生包含結果、視覺化內容、報告、日誌及所有相關產出物的完整膠囊。

![schema](./docs/images/mimosa_overall.jpg)

## 範例：重現生物活性分子網路論文

> **目標** — 端到端重現 [Nothias et al. (2018)](https://www.researchgate.net/publication/323525305_Bioactivity-Based_Molecular_Networking_for_the_Discovery_of_Drug_Leads_in_Natural_Product_Bioassay-Guided_Fractionation)：從特徵偵測到網路視覺化，以 `.mzML` 檔案作為起點（假設原始資料轉換已完成）。

Mimosa-AI 獲得了該論文、原始資料，以及一套涵蓋論文中未完整說明的設定細節的領域專屬技能。

**執行延時展示**

https://github.com/user-attachments/assets/dcd04ade-9c43-44a8-b3e3-a999d3dc895d

**輸出結果**

![molecular_network](./docs/images/network.png)

所得分子網路與原論文報告的拓撲結構相符，包括叢集分離和邊權重均由系統自主重現。

---

## 前置條件

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)（建議）或 pip
- 正在執行的 [Toolomics MCP 伺服器](https://github.com/HolobiomicsLab/toolomics)

---

## 安裝

### 1. 複製儲存庫並建立虛擬環境

```bash
# 使用 uv（建議）
pip install uv
uv venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 或使用 pip
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安裝相依套件

```bash
cd mimosa
uv pip install -r requirements.txt
```

### 3. 設定 API 金鑰

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

### 4. 啟動 MCP 伺服器

請參考 [HolobiomicsLab/toolomics](https://github.com/HolobiomicsLab/toolomics) 的設定說明，將其設定為在某個連接埠範圍內執行（例如 `5000–5100`）。自訂 MCP 工具可透過 [Toolomics 文件](https://github.com/HolobiomicsLab/toolomics/README.md) 新增。

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
| `workflow_llm_model` | 用於多智能體編排的 LLM（建議：anthropic/claude-opus-4-5 或 z-ai/glm-5） |
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

> **注意：** 執行任何模式前，必須先安裝 Toolomics 並確保 MCP 伺服器正在執行。

## 輸出

執行過程中，檔案將寫入 Toolomics 的 `workspace/` 目錄。執行完成後，完整快照將儲存至 `runs_capsule/` 中帶時間戳記的膠囊目錄。

---

## 透過多智能體工作流程的演化進行學習

***Mimosa-AI*** 能夠為科學任務動態合成專用工作流程，並透過**達爾文式多智能體工作流程演化**從失敗中學習：
系統不再將任務強行套入固定流水線，而是為每個任務組合定製的多智能體圖，並透過單候選局部搜尋對其進行優化。在每次迭代中，僅由表現最優的工作流程產生繼承者，且只保留改進結果。隨著時間推移，系統累積了一個經過驗證的工作流程庫，使類似的未來任務能夠從強基準出發，而非從零開始。

對於任何新任務，建議先以學習模式執行，讓系統在獲得完全自主權之前積累能力。

**啟動學習模式**

```bash
uv run main.py --task "在 Clintox 資料集上訓練一個多任務模型，以預測藥物毒性和 FDA 核准狀態" --learn --config my_config.json
```

![dgm](./docs/images/workflow_mutation.png)

**進度視覺化：**

***Mimosa-AI*** 完成某任務的學習階段後，您可以視覺化其隨時間的改進過程。展示各次嘗試中效能提升的獎勵進度圖將自動儲存。

獎勵進度圖儲存於 `sources/workflows/<uuid>` 資料夾下，檔案名稱為 `reward_progress.png`。

***範例：***

![dgm](./docs/images/evolve_example.png)

## 透明度

我們內建了互動式除錯器 `memory_explorer.py`，允許您以精細粒度逐步回溯任意智能體執行過程。

使用工作流程 `<uuid>` 啟動（例如：`memory_explorer.py 20260115_113303_9bb63437`）

該工具將重播完整的執行軌跡——包括思考過程、工具呼叫及輸出結果——以便您精確檢查決策是如何展開的。

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

⚠️ 為確保評估公正，建議先執行 `./cleanup.sh`，以防止 ***Mimosa*** 使用已有或快取的工作流程。

### ScienceAgentBench

在 ScienceAgentBench 上進行評估，您需要：

1. 下載 ScienceAgentBench 完整資料集：
[資料集連結](https://buckeyemailosu-my.sharepoint.com/personal/chen_8336_buckeyemail_osu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch%2Fbenchmark%2Ezip&parent=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch&ga=1)
2. 使用密碼 `scienceagentbench` 解壓縮
3. 將 `benchmark/benchmark/datasets` 資料夾的內容複製到 `Mimosa-AI/datasets/scienceagentbench/datasets`

**啟用學習模式在 ScienceAgentBench 上評估**
```sh
uv run main.py --science_agent_bench --learn
```

**限制 10 個任務且學習迭代次數不超過 4 次的 ScienceAgentBench 評估**

```sh
uv run main.py --science_agent_bench --csv_runs_limit 10 --max_evolve_iterations 4
```

### PaperBench

**啟用學習模式在 OpenAI PaperBench 上評估**

OpenAI PaperBench 是一個用於評估 AI 智能體重現 AI 研究能力的基準，來源於論文 `PaperBench: Evaluating AI's Ability to Replicate AI Research`。

```sh
uv run main.py --papers datasets/paper_bench.csv --csv_runs_limit 20  --learn
```

⚠️ 此操作將在 `runs_capsule/` 資料夾中儲存所有論文重現嘗試的結果，完整評估請參考 [Paper Bench 文件](https://github.com/openai/frontier-evals/tree/main/project/paperbench)。

**在自訂研究論文基準上評估**

1. 將與 `paper_bench.csv` 格式相同的基準 CSV 檔案放置於 `datasets/<您的基準名稱>.csv`。

2. 在您的基準上執行：

```sh
uv run main.py --papers datasets/<您的基準名稱>.csv --csv_runs_limit 20  --learn
```

---

## 手機通知

透過 Pushover 通知接收 ***Mimosa*** 狀態的即時更新。

### 設定步驟

1. **建立 Pushover 帳戶**
   - 造訪 [pushover.net](https://pushover.net/)
   - 註冊並記錄您的**使用者金鑰（User Key）**

2. **建立應用程式**
   - 在 Pushover 控制台中，點選「建立應用程式/API Token」
   - 命名為「***Mimosa***」並複製產生的 **API Token**

3. **設定環境變數**
   ```bash
   export PUSHOVER_USER="您的使用者金鑰"
   export PUSHOVER_TOKEN="您的 API Token"
   ```

4. **安裝行動應用程式**
   - 從裝置應用程式商店下載 Pushover
   - 使用您的 Pushover 帳戶登入

---

## 遙測設定

使用 Langfuse 透過即時可觀測性儀表板監控和除錯 AI 智能體。

### 快速開始

1. **本地部署 Langfuse**
   ```bash
   git clone https://github.com/langfuse/langfuse.git
   cd langfuse
   docker compose up -d
   ```

2. **設定環境變數**

   在您的 `.env` 檔案中新增：
   ```env
   LANGFUSE_PUBLIC_KEY=您的公開金鑰
   LANGFUSE_PRIVATE_KEY=您的私密金鑰
   ```

3. **存取儀表板**

   ***Mimosa-AI*** 執行時，造訪 `http://localhost:3000`

### 可用指標

遙測儀表板提供：
- **智能體執行追蹤**：逐步工作流程視覺化
- **效能指標**：回應時間和成功率
- **錯誤除錯**：詳細故障分析
- **資源使用情況**：Token 消耗和 API 呼叫統計

**儀表板範例：**
![Langfuse Dashboard](https://langfuse.com/images/cookbook/integration-smolagents/smolagent_example_trace.png)

> **注意：** 遙測為可選功能，但建議用於除錯和效能優化。

---

## 授權條款

本儲存庫在 Apache License 2.0 下公開發布。Apache 2.0 是一個寬鬆的開源授權條款，允許商業使用、修改和再發布，但要求再發布時保留適用的授權條款、版權、專利、商標、署名及 NOTICE 聲明，且修改後的檔案須附帶顯著說明，注明已做出的變更。

有關貢獻與授權細節，請參閱：
- `NOTICE`
- `docs/licensing-notes.md`
- `CLA/INDIVIDUAL_CLA.md`
- `CLA/EMPLOYER_AUTHORIZATION.md`

---

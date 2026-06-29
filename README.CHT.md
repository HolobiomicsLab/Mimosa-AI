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
    <em>面向自主科學研究的自我演化多智能體框架 —— LLM 驅動的工作流程合成、品質-多樣性演化搜尋、MCP 工具自動探索。</em>
</p>

<p align="center">
  🧬 品質-多樣性 (Quality-Diversity) 工作流程演化 &nbsp;·&nbsp;
  🔍 基於 MCP 的工具自動探索 &nbsp;·&nbsp;
  🧪 多來源逐項聲明驗證 &nbsp;·&nbsp;
  📦 完整稽核追蹤與可重現性
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

Mimosa-AI 是一個**面向自主科學研究的開源 Python 框架**:它會**為每一項任務撰寫一套客製化的多智能體工作流程**,於沙箱中執行,並以六個獨立觀察視角(文獻、您的目標、智能體敘述、數學不變量、計算可重現性、統計指紋)檢核智能體實際完成的內容;當您要求其學習時,則透過**品質-多樣性 (Quality-Diversity)** 搜尋於世代之間演化該工作流程,同時保留效能與結構多樣性。

工作流程以純 Python 程式碼產出 —— 無 DSL、無 YAML —— 因此每一個世代皆可被檢視、比對或獨立重新執行。驗證器執行確定性的 Python 檢核程式,重新計算智能體所宣稱的結果。每一個世代都連同其譜系與產生該世代的確切 LLM 提示一併保存於磁碟。

```bash
uv sync && uv run main.py        # interactive onboarding
```

---

## 示範

<p align="center">
    <em>Mimosa-AI 僅透過單一指令、且無預設管線的情況下,自主重現了 <a href="https://www.researchgate.net/publication/323525305_Bioactivity-Based_Molecular_Networking_for_the_Discovery_of_Drug_Leads_in_Natural_Product_Bioassay-Guided_Fractionation">Nothias 等人 (2018)</a> 的 LC-MS/MS 分子網路分析管線——對 <code>.mzML</code> 檔案進行特徵偵測(MZmine / OpenMS / matchms 類型工具——由智能體自行挑選)、對齊,以及經典分子網路分析 (GNPS 風格的餘弦聚類)。</em>
</p>

https://github.com/user-attachments/assets/dcd04ade-9c43-44a8-b3e3-a999d3dc895d

所重現的網路在叢集層級上與該論文所報告的拓樸相符,輸出為可於 Cytoscape 中載入的 `.graphml` 檔案,以及對應的特徵定量表。範圍說明:此處僅重現**分子網路分析**階段——原始研究中的生物活性導向分餾、人工註解審閱及資料庫比對 (GNPS / SIRIUS / CSI:FingerID) 皆不在自主執行的範圍內。

<p align="center">
  <img src="./docs/images/network.png" alt="LC-MS/MS molecular network autonomously reproduced by Mimosa-AI — GNPS-style cosine clustering exported as Cytoscape GraphML" width="80%">
</p>

---

## 基準測試

於 **ScienceAgentBench** 上進行評估(102 項任務,`task` 模式——略過規劃層,使工作流程合成與精煉得以獨立評估):

| 模式                                    | 成功率       | Code-BLEU | 每任務成本   |
| --------------------------------------- | ------------ | --------- | ----------- |
| DeepSeek-V3.2 單一智能體                | 38.2 %       | 0.898     | $0.05       |
| DeepSeek-V3.2 單發多智能體              | 32.4 %       | 0.794     | $0.38       |
| **DeepSeek-V3.2 迭代學習**              | **43.1 %**   | **0.921** | **$1.70**   |

> **於 ScienceAgentBench 上,DeepSeek-V3.2 迭代學習取得 43.1% 成功率 —— 較單一智能體基線提升 +4.9 個百分點,單任務成本 $1.70。**

> 於 ScienceAgentBench 上使用 DeepSeek-V3.2 時,迭代學習能改善 GPT-4o 的表現,但對 Claude Haiku 4.5 卻造成輕微的退化——與模型相關的行為差異已於[論文](https://arxiv.org/abs/2603.28986)中分析。PaperBench 結果請見 [`docs/papers_bench_evaluation.md`](./docs/papers_bench_evaluation.md)。

---

## 運作原理

五個層級透過小型的 dataclass 結構互相串接——完整細節請見 [`docs/concepts/architecture.md`](./docs/concepts/architecture.md)。

<p align="center">
  <img src="./docs/images/mimosa_overall.jpg" alt="Mimosa-AI architecture: planner, MCP tool manager, evolution engine, sandboxed SmolAgents workflow runner, multi-source per-claim verifier" width="90%">
</p>

| 層級 | 元件 | 功能 |
|-------|-----------|--------------|
| 0 | **Planner** *(選用,僅 `--goal`)* | 將高階目標拆解為離散任務。 |
| 1 | **ToolManager + Perspicacité** | 於設定的位址/連接埠範圍上探索 MCP 工具;選擇性地擷取文獻片段。 |
| 2 | **EvolutionEngine** | 合成工作流程並於世代之間進行演化(詳見下文)。 |
| 3 | **WorkflowRunner** | 於沙箱中執行所合成的 Python 工作流程,使用 Hugging Face [SmolAgents](https://github.com/huggingface/smolagents)(`LocalPythonExecutor` 搭配 AST 白名單),並透過共享的 LangGraph 狀態運作。 |
| 4 | **VerifierEvaluator** | 多來源逐項聲明驗證器,並驅動下一次的突變。 |

### 演化迴圈——實際在演化的是什麼

工作流程是**完整的 Python 程式**,以原始碼形式進行突變。**程式碼即基因型 (code-as-genotype)** 即為工作流程檔案;表現型則為其在工作區內所產生的一切。

- **選擇:品質-多樣性 (QD) 檔案庫** —— **非結構化檔案庫**(單一名單,而非離散網格),族群大小上限為 50,以單一標量化目標 `qd_score = (1−w)·quality + w·novelty` (`w=0.25`) 評分;額滿時淘汰 `qd_score` 最低的成員。**新穎性搜尋 (novelty search)** 以**基因型嵌入 (genotype embedding)** 行為描述子——對工作流程生成原始碼做 L2 正規化的嵌入(預設使用本機 `all-MiniLM-L6-v2`,可選 OpenAI `text-embedding-3-small`)——上的餘弦距離 k-NN (`k=15`) 衡量。親代以反向子代數量輪盤選取,以促使檔案庫均勻擴散 (`MAX_CHILDREN_PER_PARENT = 8`)。
- **變異:由停滯驅動的範圍** —— 突變的大膽程度是過去 4 次提示梯度自我重複程度的連續函數。接近獲勝者者受到保護。範圍從「僅微調提示」至「完整重新思考拓樸」分為多個級距。
- **交配** —— 約 30 % 的世代會組合兩個親代,依強者優先。
- **冷啟動** —— 當檔案庫為空時,以磁碟上過往執行的相似度過濾掃描 (MiniLM 餘弦 ≥ 0.5) 為搜尋播種。可用的工作流程能跨任務遷移。

完整機制:[`docs/concepts/evolution-engine.md`](./docs/concepts/evolution-engine.md)。

### 驗證器——分數實際代表的意義

每一次執行後,六個獨立的聲明來源會檢視工作區並產生成功極性的聲明:

| 來源 | 觀察視角 |
|--------|---------|
| **A** | 經同儕審閱的實作實務(透過 Perspicacité 文獻接地) |
| **B** | 字面上的目標文本——智能體是否交付了所要求的內容? |
| **C** | 智能體敘述——所聲明的數字/產物能否從磁碟重現? |
| **D** | 數學不變量——機率落於 [0,1]、形狀一致、無 NaN、守恆性 |
| **E** | 計算可重現性——已宣告的相依套件涵蓋實際使用的 import、無絕對路徑、隨機操作皆有種子 |
| **F** | 統計指紋——優於基線、無退化預測、無洩漏特徵 |

每一項聲明皆由評審撰寫、針對工作區執行的**確定性 Python 程式**來驗證——而非再次詢問 LLM 是否相信該智能體。**自我驗證 (self-verification)** 透過反同語反複的觸發機制,拒絕將智能體輸出與自身比對的程式。

**評分準則盲變異 (rubric-blind mutation) —— 突變器永遠看不到評分準則。** 唯一回流的訊號是 `abstracted_prompt_gradient`——一份以代號描述失效模式的診斷,當中不會點名任何聲明、分數或來源。就架構而言,搜尋無法對其從未看到的評分詞彙過度擬合。

完整管線:[`docs/concepts/evaluation-pipeline.md`](./docs/concepts/evaluation-pipeline.md)。

---

## 快速上手

### 1. 安裝

```bash
pip install uv
git clone https://github.com/HolobiomicsLab/Mimosa-AI.git
cd Mimosa-AI
uv sync
```

### 2. 至少新增一組 LLM 金鑰

於專案根目錄建立 `.env`。只需填入您實際會用到的服務供應商即可。

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

### 3. 公開 MCP 工具

Mimosa 會在您組態中的位址/連接埠範圍內探索任何可連線的 MCP 伺服器(預設為 `0.0.0.0:5000–5100`)。

- **最簡途徑:** 安裝我們的配套平台 **[Toolomics](https://github.com/HolobiomicsLab/toolomics)** —— 每個工作區一個 MCP shell 沙箱,智能體可依需求安裝科學套件(同一工作區的後續執行可重複使用已安裝的工具);另提供預先建置的 MCP 服務以暴露常見科學工具堆疊,並搭配共享工作區管理與一套既定的註冊流程。
- **自備工具:** 將 `discovery_addresses` 指向任何可連線的 MCP 伺服器 —— `fastmcp` 指令稿、ToolHive、第三方 MCP 容器皆可。Toolomics 並非必要;請見 [`docs/concepts/tools-and-mcp.md`](./docs/concepts/tools-and-mcp.md#operating-without-toolomics)。

### 4. 執行

```bash
uv run main.py                   # interactive onboarding (recommended first time)
```

或略過精靈:

```bash
uv run main.py --task "Train a multitask model on Clintox to predict toxicity and FDA approval"
uv run main.py --goal "Reproduce experiments from https://arxiv.org/pdf/2306.00306 and compare results"
```

加上 `--learn` 以跨世代演化,而非僅執行單發:

```bash
uv run main.py --task "..." --learn --config my_config.json
```

完整快速上手:[`docs/getting-started/quickstart.md`](./docs/getting-started/quickstart.md)。

### 5. (選用) 透過 Perspicacité 進行科學接地

[Perspicacité](https://github.com/HolobiomicsLab/Perspicacite-AI) 會將工作流程合成與 Source A 聲明接地於文獻。當其運行時,Mimosa 會自動偵測。

```bash
git clone https://github.com/HolobiomicsLab/Perspicacite-AI.git && cd Perspicacite-AI
uv sync && uv run web_app_full.py
```

---

## 執行模式

| 模式 | 使用時機 | 指令 |
|------|----------|---------|
| `--task` | 單一聚焦的操作 | `uv run main.py --task "..."` |
| `--goal` | 需要規劃的多步驟目標 | `uv run main.py --goal "..."` |
| `--learn` | 可加在任一模式上 —— 跨世代演化 | `... --learn` |
| `--single_agent` | 略過多智能體合成(快速,不學習) | `... --single_agent` |
| `--manual` | 互動式 CLI,用於測試個別 MCP 工具 | `uv run main.py --manual` |
| 批次 | 評估 CSV 中的多項任務 | `... --papers <csv>` |
| 基準測試 | ScienceAgentBench | `... --science_agent_bench` |

詳情:[`docs/usage/modes.md`](./docs/usage/modes.md)、[`docs/usage/learning.md`](./docs/usage/learning.md)、[`docs/reference/cli.md`](./docs/reference/cli.md)。

---

## 稽核追蹤與重播

Mimosa 專為科學用途而設計——每一項決策皆可事後檢視。

| 工具 | 功能 |
|------|--------------|
| `uv run workflow_evolution_anim.py` | 互動式檢視器,走訪演化樹並重播每一代的軌跡 —— 思路、工具呼叫、評分通過/未通過。 |
| `uv run main.py --memory_cli` | 對已完成的執行記憶進行 RAG 支援的問答。可直接詢問「*task_builder 用了哪個分類器?*」,而不必滾動翻找。 |
| `uv run memory_timelapse.py <uuid>` | 以動畫方式逐影格檢視記憶在各次迭代間的成長。 |
| `sources/workflows/<uuid>/workflow_genotype_<uuid>.py` | 智能體實際執行的精確 Python 程式碼。無 DSL。 |
| `sources/workflows/<uuid>/lineage_<uuid>.json` | 此世代的親代與運算子 (`seed | mutation | crossover`)。 |
| `sources/workflows/<uuid>/evolution_prompt_<uuid>.md` | 產生此份程式碼的確切 LLM 提示。相同提示 + 種子 = 相同程式碼。 |
| `sources/workflows/<uuid>/evolution_tree.png` | 整個 `--learn` 執行的譜系樹渲染圖。 |
| `sources/workflows/<uuid>/reward_progress.png` | 分數隨迭代變化的曲線。 |
| `runs_capsule/<capsule_name>/` | 最終工作區的封存快照,供分享或重新執行。 |

完整版面配置:[`docs/usage/transparency.md`](./docs/usage/transparency.md)、[`docs/usage/workspace.md`](./docs/usage/workspace.md)。

---

## 組態設定

將 `config_default.json` 複製為 `my_config.json` 後再進行編輯。最常需調整的欄位:

| 欄位 | 控制內容 |
|-------|------------------|
| `workspace_dir` | 共享工作區 —— 所有生成的檔案皆置於此 |
| `discovery_addresses` | MCP 探索的 IP 與連接埠範圍 |
| `workflow_llm_model` | 合成多智能體工作流程的模型(例如 `anthropic/claude-opus-4-5`) |
| `smolagent_model_id` | 執行智能體所使用的模型 |
| `judge_model` | 撰寫驗證器程式並產出軟性裁決的 LLM |
| `learned_score_threshold` | `--learn` 模式下的提前終止門檻(預設 `0.9`) |
| `max_learning_evolve_iterations` | 世代上限(預設 `20`) |

完整參考:[`docs/reference/configuration.md`](./docs/reference/configuration.md)。

---

## 評估

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

> ⚠️ 為了得到不偏頗的評估結果,請先執行 `./cleanup.sh`,以避免 Mimosa 重複使用已快取的工作流程。

各基準測試的設定細節:[`docs/science_agent_bench_evaluation.md`](./docs/science_agent_bench_evaluation.md)、[`docs/papers_bench_evaluation.md`](./docs/papers_bench_evaluation.md)。

---

## 通知與遙測

- **Pushover** —— 即時將進度推播至手機。設定 `PUSHOVER_USER` 與 `PUSHOVER_TOKEN`。詳情:[`docs/usage/notifications.md`](./docs/usage/notifications.md)。
- **Langfuse** —— 每一次 LLM 呼叫的 span 層級追蹤。於 Langfuse 儲存庫執行 `docker compose up -d`,再將 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_PRIVATE_KEY` 加入 `.env`。儀表板位於 `http://localhost:3000`。詳情:[`docs/usage/telemetry.md`](./docs/usage/telemetry.md)。

---

## 相關研究

Mimosa-AI 屬於 LLM 驅動之程式碼搜尋與自主研究系統中一條規模雖小但活躍的脈絡。我們並不主張取代其中任何一者——各系統各自回答不同的問題:

| 專案 | 內容 | Mimosa 的差異 |
|---------|--------------|--------------------|
| [Sakana AI Scientist](https://github.com/SakanaAI/AI-Scientist) | 端到端的機器學習論文生成 | Mimosa 以 QD + 驗證器最佳化**逐任務的工作流程合成**,而非整篇論文的生成 |
| [DiscoPOP](https://github.com/SakanaAI/DiscoPOP) (Lange 等人,2024) | LLM 驅動的偏好最佳化演算法探索 | 同屬「LLM 作為對程式碼的變異運算子」典範;Mimosa 將其應用於多智能體工作流程程式碼,而非損失函數 |
| [FunSearch](https://github.com/google-deepmind/funsearch) (Romera-Paredes 等人,2024) | 由 LLM 引導的 Python 函數演化搜尋 | Mimosa 演化的是完整的多智能體程式,並以多來源逐項聲明驗證器取代單一適應度函數 |
| [ELM](https://github.com/CarperAI/OpenELM) (Lehman 等人,2022) | 由 LLM 媒介的程式碼品質-多樣性搜尋 | 最接近的 QD 前身;Mimosa 的行為描述子屬於工作流程結構面,而非特定領域 |
| AIDE | 針對 Kaggle 風格任務的自動化 ML 管線 | Mimosa 鎖定更廣泛的科學重現工作 (ScienceAgentBench、PaperBench、實驗室資料),並附帶可稽核的逐項聲明驗證器 |

若您正在發表比較性研究,[論文](https://arxiv.org/abs/2603.28986)中有更詳盡的定位說明。

---

## 完整文件

```bash
uvx --with mkdocs-material mkdocs serve   # live preview at http://localhost:8000
uvx --with mkdocs-material mkdocs build   # static HTML to ./site
```

站點組態:[`mkdocs.yml`](./mkdocs.yml)。索引:[`docs/index.md`](./docs/index.md)。

---

## 貢獻

歡迎提交修補、MCP 工具、評估器與新的聲明來源。請從 [`CONTRIBUTING.md`](./CONTRIBUTING.md)、[Developer guide](./docs/DEVELOPER_GUIDE.md) 以及 [`CLA/`](./CLA/) 中的貢獻條款著手。

---

## 授權

Apache 2.0。請見 [`NOTICE`](./NOTICE)、[`docs/licensing-notes.md`](./docs/licensing-notes.md),以及 [`CLA/`](./CLA/) 資料夾中的貢獻條款。

---

## 引用本作品

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
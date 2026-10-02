<div align="center">
<br>

<img src="./docs/images/logo_mimosa.png" width="22%" style="border-radius: 8px;" alt="Mimosa-AI logo — self-evolving multi-agent AI framework for autonomous scientific research (Holobiomics Lab, CNRS)">

</div>

<h1 align="center">Mimosa-AI — 自主科學研究的演化式多智能體框架</h1>


<p align="center">
  <a href="./README.md">English</a> &nbsp;|&nbsp;
  <a href="./README.CHS.md">简体中文</a> &nbsp;|&nbsp;
  <a href="./README.CHT.md">繁體中文</a> &nbsp;|&nbsp;
  <a href="./README.JPN.md">日本語</a> &nbsp;|&nbsp;
  <a href="./README.KOR.md">한국어</a>
</p>

<p align="center">
    <em>自我演化的多智能體框架，用於自主科學研究 —— LLM 驅動的工作流程合成、品質-多樣性演化搜尋、MCP 工具探索。</em>
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


https://github.com/user-attachments/assets/744d2c34-4ac3-415c-bd8c-3454cd502271


---

## TL;DR

Mimosa-AI 是一個**用於自主科學研究的開源 Python 框架**：它會**為每一項任務撰寫一套客製化的多智能體工作流程**，於沙箱中執行，並以獨立視角檢核智能體實際完成的內容，再透過**品質-多樣性 (Quality-Diversity)** 啟發式搜尋，跨世代演化工作流程，以找出該任務的最佳工作流程。

工作流程以純 Python 程式碼產出 —— 無 DSL、無 YAML —— 因此任何世代都可以被檢視、比對或獨立重新執行。驗證器透過執行確定性的 Python 檢核程式來評分工作流程，驗證文獻基礎、非平凡性與品質指標是否與智能體產出的成品一致。每個世代都連同其譜系及產生該世代的確切 LLM 提示一併保存在磁碟上。

## 自動安裝

在終端機中執行：

```bash
curl https://raw.githubusercontent.com/HolobiomicsLab/Mimosa-AI/refs/heads/mimosa_v2/auto-install.sh | bash 
```

注意：此操作將自動安裝並啟動我們的配套專案 `Toolomics` 與 `Perspicacité`。

手動安裝請參閱：[手動安裝](##手動安裝)

---

## Web 介面

在瀏覽器中開啟 `http://localhost:5173/` 即可存取 Web 介面。

<p align="center">
  <img src="./docs/images/interface.png" alt="Mimosa web interface" width="80%">
</p>


---

## 在代謝體學上的示範 (V1)

此示範使用 V1 版本完成，即將更新。

<p align="center">
    <em>Mimosa-AI 自主重現了 <a href="https://www.researchgate.net/publication/323525305_Bioactivity-Based_Molecular_Networking_for_the_Discovery_of_Drug_Leads_in_Natural_Product_Bioassay-Guided_Fractionation">Nothias 等人 (2018)</a> 的 LC-MS/MS 分子網路分析管線 —— 對 <code>.mzML</code> 檔案進行特徵偵測（MZmine / OpenMS / matchms 類型工具——由智能體自行挑選）、對齊，以及經典分子網路分析（GNPS 風格的餘弦聚類）。</em>
</p>

https://github.com/user-attachments/assets/dcd04ade-9c43-44a8-b3e3-a999d3dc895d

所重現的網路在叢集層級上與該論文所報告的拓樸相符，輸出為可於 Cytoscape 中載入的 `.graphml` 檔案，以及對應的特徵定量表。範圍說明：此處僅重現**分子網路分析**階段——原始研究中的生物活性導向分餾、人工註解審閱及資料庫比對 (GNPS / SIRIUS / CSI:FingerID) 皆不在自主執行的範圍內。

<p align="center">
  <img src="./docs/images/network.png" alt="LC-MS/MS molecular network autonomously reproduced by Mimosa-AI — GNPS-style cosine clustering exported as Cytoscape GraphML" width="80%">
</p>

---


## 運作原理

五個層級透過小型的 dataclass 結構互相串接——完整細節請見 [`docs/concepts/architecture.md`](./docs/concepts/architecture.md)。

<p align="center">
  <img src="./docs/images/mimosa_overall.jpg" alt="Mimosa-AI architecture: planner, MCP tool manager, evolution engine, sandboxed SmolAgents workflow runner, multi-source per-claim verifier" width="90%">
</p>

| 層級 | 元件 | 功能 |
|-------|-----------|--------------|
| 0 | **Planner** *(選用，僅 `--goal`)* | 將高階目標拆解為離散任務。 |
| 1 | **ToolManager + Perspicacité** | 於設定的位址/連接埠範圍上探索 MCP 工具；選擇性地擷取文獻片段。 |
| 2 | **EvolutionEngine** | 合成工作流程並於世代之間進行演化（詳見下文）。 |
| 3 | **WorkflowRunner** | 於沙箱中執行所合成的 Python 工作流程，使用 Hugging Face [SmolAgents](https://github.com/huggingface/smolagents)（`LocalPythonExecutor` 搭配 AST 白名單），並透過共享的 LangGraph 狀態運作。 |
| 4 | **VerifierEvaluator** | 多來源逐項聲明驗證器，驅動下一次的突變。 |

### 演化迴圈——實際在演化的是什麼

工作流程是**完整的 Python 程式**，以原始碼形式進行突變。**程式碼即基因型 (code-as-genotype)** 即為工作流程檔案；表現型則為其於工作區內所產生的一切。

- **選擇：品質-多樣性 (QD) 檔案庫** —— **非結構化檔案庫**（單一名單，而非離散網格），族群大小上限為 20，以單一標量化目標 `qd_score = (1−w)·quality + w·novelty`（`w = novelty_weight = 0.25`）評分；額滿時淘汰 `qd_score` 最低的成員。**新穎性搜尋 (novelty search)** 以**基因型嵌入 (genotype embedding)** 行為描述子——對工作流程生成原始碼做 L2 正規化的嵌入（預設使用本機 `all-MiniLM-L6-v2`，可選 OpenAI `text-embedding-3-small`）——上的餘弦距離 k-NN（`k = 15`）衡量。親代以反向子代數量輪盤選取，以促使檔案庫均勻擴散（`MAX_CHILDREN_PER_PARENT = 8`）。
- **變異：Rechenberg-1/5 + 高原驅動的範圍** —— 突變的大膽程度結合了最近 5 個已評分子代的成功率（Rechenberg 1/5 規則，閾值 `0.20`）與 `iters_since_improvement` 高原計數器（耐心值 `6`）。接近獲勝者（親代分數 > 0.95）會受到阻尼保護。範圍從 `EXPLOITATION`（點突變）到 `RE-SPECIATION`（全新重新設計），由有效大膽程度閾值（`< 0.35 / 0.50 / 0.65 / 0.90 / 1.01`）決定。
- **交配** —— 預設約 40% 的世代會組合兩個親代，依強者優先，子代智能體數量以上限較高的親代為硬上限。
- **冷啟動** —— 當檔案庫為空時，以磁碟上過往執行的相似度過濾掃描（MiniLM 餘弦 ≥ 0.8）為搜尋播種。可用的工作流程能跨任務遷移。

完整機制：[`docs/concepts/evolution-engine.md`](./docs/concepts/evolution-engine.md)。

### 驗證器——分數實際代表的意義

每一次執行後，六個獨立的聲明來源會檢視工作區並產生成功極性的聲明：

| 來源 | 觀察視角 |
|--------|---------|
| **A** | 經同儕審閱的實作實務（透過 Perspicacité 文獻接地） |
| **B** | 字面上的目標文本——智能體是否交付了所要求的內容？ |
| **C** | 智能體敘述——所聲明的數字/產物能否從磁碟重現？ |
| **D** | 數學不變量——機率落於 [0,1]、形狀一致、無 NaN、守恆性 |
| **E** | 計算可重現性——已宣告的相依套件涵蓋實際使用的 import、無絕對路徑、隨機操作皆有種子 |
| **F** | 統計指紋——優於基線、無退化預測、無洩漏特徵 |

每一項聲明皆由評審針對工作區所撰寫的 **Python 程式**來驗證——而非再次詢問 LLM 是否相信該智能體。

**評分準則盲突變 (rubric-blind mutation) —— 突變器永遠看不到評分準則。** 唯一回流的訊號是 `abstracted_prompt_gradient`——一份以代號描述失效模式的診斷，當中不會點名任何聲明、分數或來源。就架構而言，搜尋無法對其從未看到的評分詞彙過度擬合。

完整管線：[`docs/concepts/evaluation-pipeline.md`](./docs/concepts/evaluation-pipeline.md)。

## 基準測試 (V1)

於 **ScienceAgentBench** 上進行評估（102 項任務，`task` 模式——略過規劃層，使工作流程合成與精煉得以獨立評估）：

| 模式                                    | 成功率       | Code-BLEU | 每任務成本   |
| --------------------------------------- | ------------ | --------- | ----------- |
| DeepSeek-V3.2 單一智能體                | 38.2 %       | 0.898     | $0.05       |
| DeepSeek-V3.2 單發多智能體              | 32.4 %       | 0.794     | $0.38       |
| **DeepSeek-V3.2 迭代學習**              | **43.1 %**   | **0.921** | **$1.70**   |

> **於 ScienceAgentBench 上，DeepSeek-V3.2 迭代學習取得 43.1% 成功率 —— 較單一智能體基線提升 +4.9 個百分點，單任務成本 $1.70。**

> 於 ScienceAgentBench 上使用 DeepSeek-V3.2 時，迭代學習能改善 GPT-4o 的表現，但對 Claude Haiku 4.5 卻造成輕微的退化——與模型相關的行為差異已於[論文](https://arxiv.org/abs/2603.28986)中分析。PaperBench 結果請見 [`docs/papers_bench_evaluation.md`](./docs/papers_bench_evaluation.md)。

## 基準測試 (V2)

**目前正在評估中**

---

## 手動安裝

### 1. 公開 MCP 工具

Mimosa 會在您組態中的位址/連接埠範圍內探索任何可連線的 MCP 伺服器（預設為 `0.0.0.0:5000–5100`）。

- **最簡途徑：** 安裝我們的配套平台 **[Toolomics](https://github.com/HolobiomicsLab/toolomics)** —— 每個工作區一個 MCP shell 沙箱，智能體可依需求安裝科學套件；另提供預先建置的 MCP 服務以暴露常見科學工具堆疊，並搭配共享工作區管理與一套簡便的新 MCP 註冊流程。
- **自備工具：** 將 `discovery_addresses` 指向任何可連線的 MCP 伺服器 —— `fastmcp` 指令稿、ToolHive、第三方 MCP 容器皆可。Toolomics 並非必要；請見 [`docs/concepts/tools-and-mcp.md`](./docs/concepts/tools-and-mcp.md#operating-without-toolomics)。

### 2. 安裝 Mimosa

```bash
pip install uv
git clone https://github.com/HolobiomicsLab/Mimosa-AI.git
cd Mimosa-AI
uv sync
# 然後執行：
uv run main.py
```

**或安裝為獨立的 `mimosa` 指令**，可從任何目錄使用：

```bash
uv tool install git+https://github.com/HolobiomicsLab/Mimosa-AI.git   # 或：uv tool install /path/to/Mimosa-AI
mimosa
```

以此方式安裝時，設定會保存至 `~/.config/mimosa/config.json`（由入門精靈寫入，每次執行時自動載入），API 金鑰可存放於 `~/.config/mimosa/.env`，執行時期狀態（記憶體、工作流程、執行膠囊）則存放於 `~/.local/share/mimosa/`。儲存庫檢出保留歷史佈局：`config_default.json` 與狀態目錄位於檢出目錄內。

完整快速入門：[`docs/getting-started/quickstart.md`](./docs/getting-started/quickstart.md)。

### 3. （選用）透過 Perspicacité 進行科學接地

[Perspicacité](https://github.com/HolobiomicsLab/Perspicacite-AI) 會將工作流程合成與 Source A 聲明接地於文獻。當其運行時，Mimosa 會自動偵測。

```bash
git clone https://github.com/HolobiomicsLab/Perspicacite-AI.git && cd Perspicacite-AI
export DEEPSEEK_API_KEY="xxxxx" # 匯出您的 API 金鑰；亦支援 anthropic 與 openrouter
uv run perspicacite -c config.yml serve
```

---

## Web 介面（Observatory）

Mimosa 本身僅提供 CLI；**Observatory** 是一個選用的本機 Web 介面
（FastAPI + React），用於呈現一次執行所產生的內容——演化系譜樹、重播、
工作區，以及一套設定/啟動流程——讓你可以直接觀察與檢視演化過程，
而不必翻閱記錄檔。它是一個單人使用、僅限本機（localhost）的工具，
不含身分驗證；請勿將其公開於共享或公開網路上。

```bash
cd webui && ./deploy.sh    # 安裝相依套件，執行後端 + 前端；開啟 http://localhost:5173
```

詳細說明、環境變數與完整 API 一覽請見：
[`webui/README.md`](./webui/README.md)。

---

## 稽核追蹤與重播

請參閱：[`docs/usage/transparency.md`](./docs/usage/transparency.md)、[`docs/usage/workspace.md`](./docs/usage/workspace.md)。

---

## 組態設定

請參閱：[`docs/reference/configuration.md`](./docs/reference/configuration.md)。

---

## 評估

各基準測試的設定細節：[`docs/science_agent_bench_evaluation.md`](./docs/science_agent_bench_evaluation.md)、[`docs/papers_bench_evaluation.md`](./docs/papers_bench_evaluation.md)。

---

## 通知與遙測

- **Pushover** —— 即時將進度推播至手機。設定 `PUSHOVER_USER` 與 `PUSHOVER_TOKEN`。詳情：[`docs/usage/notifications.md`](./docs/usage/notifications.md)。
- **Langfuse** —— 每一次 LLM 呼叫的 span 層級追蹤。於 Langfuse 儲存庫執行 `docker compose up -d`，再將 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_PRIVATE_KEY` 加入 `.env`。儀表板位於 `http://localhost:3000`。詳情：[`docs/usage/telemetry.md`](./docs/usage/telemetry.md)。

---

## 相關研究

Mimosa-AI 屬於 LLM 驅動之程式碼搜尋與自主研究系統中一條規模雖小但活躍的脈絡。我們並不主張取代其中任何一者——各系統各自回答不同的問題：

| 專案 | 內容 | Mimosa 的差異 |
|---------|--------------|--------------------|
| [Sakana AI Scientist](https://github.com/SakanaAI/AI-Scientist) | 端到端的機器學習論文生成 | Mimosa 以 QD + 驗證器最佳化**逐任務的工作流程合成**，而非整篇論文的生成 |
| [DiscoPOP](https://github.com/SakanaAI/DiscoPOP) (Lange 等人，2024) | LLM 驅動的偏好最佳化演算法探索 | 同屬「LLM 作為對程式碼的變異運算子」典範；Mimosa 將其應用於多智能體工作流程程式碼，而非損失函數 |
| [FunSearch](https://github.com/google-deepmind/funsearch) (Romera-Paredes 等人，2024) | 由 LLM 引導的 Python 函數演化搜尋 | Mimosa 演化的是完整的多智能體程式，並以多來源逐項聲明驗證器取代單一適應度函數 |
| [ELM](https://github.com/CarperAI/OpenELM) (Lehman 等人，2022) | 由 LLM 媒介的程式碼品質-多樣性搜尋 | 最接近的 QD 前身；Mimosa 的行為描述子採用與領域無關的工作流程程式碼嵌入，而非針對特定領域手工設計的描述子 |
| AIDE | 針對 Kaggle 風格任務的自動化 ML 管線 | Mimosa 鎖定更廣泛的科學重現工作（ScienceAgentBench、PaperBench、實驗室資料），並附帶可稽核的逐項聲明驗證器 |

若您正在發表比較性研究，[論文](https://arxiv.org/abs/2603.28986)中有更詳盡的定位說明。

---

## 完整文件

```bash
uvx --with mkdocs-material mkdocs serve   # 即時預覽，位於 http://localhost:8000
uvx --with mkdocs-material mkdocs build   # 靜態 HTML 輸出至 ./site
```

站台組態：[`mkdocs.yml`](./mkdocs.yml)。索引：[`docs/index.md`](./docs/index.md)。

---

## 貢獻

歡迎提交修補、MCP 工具、評估器與新的聲明來源。請從 [`CONTRIBUTING.md`](./CONTRIBUTING.md)、[開發者指南](./docs/DEVELOPER_GUIDE.md) 以及 [`CLA/`](./CLA/) 中的貢獻條款著手。

---

## 授權

Apache 2.0。請見 [`NOTICE`](./NOTICE)、[`docs/licensing-notes.md`](./docs/licensing-notes.md)，以及 [`CLA/`](./CLA/) 資料夾中的貢獻條款。

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

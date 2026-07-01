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
    <em>自律的科学研究のための自己進化型マルチエージェントフレームワーク —— LLM 駆動のワークフロー合成、Quality-Diversity 進化探索、MCP ツール自動検出。</em>
</p>

<p align="center">
  🧬 Quality-Diversity ワークフロー進化 &nbsp;·&nbsp;
  🔍 MCP ベースのツール自動検出 &nbsp;·&nbsp;
  🧪 マルチソース・クレーム単位検証 &nbsp;·&nbsp;
  📦 完全な監査証跡と再現性
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

Mimosa-AI は **自律的科学研究のためのオープンソース Python フレームワーク** です。**タスクごとにカスタムのマルチエージェントワークフローを記述**し、サンドボックスで実行し、エージェントが実際に行った内容を 6 つの独立した視点（文献、ユーザーの目標、エージェントのナラレーション、数学的不変量、計算再現性、統計的フィンガープリント）に照らして検証します。学習を指示された場合は、性能と構造的多様性の両方を維持する **Quality-Diversity**（品質多様性）探索によって、世代を超えてワークフローを進化させます。

ワークフローはプレーンな Python として出力されます — DSL なし、YAML なし — そのため、どの世代も検査・差分比較・単独での再実行が可能です。verifier はエージェントが主張する内容を再計算する決定論的な Python チェックを実行します。各世代は系譜とそのコードを生成した正確な LLM プロンプトと共にディスクに保存されます。

```bash
uv sync && uv run main.py        # interactive onboarding
```

---

## デモ

<p align="center">
    <em>Mimosa-AI は <a href="https://www.researchgate.net/publication/323525305_Bioactivity-Based_Molecular_Networking_for_the_Discovery_of_Drug_Leads_in_Natural_Product_Bioassay-Guided_Fractionation">Nothias et al. (2018)</a> の LC-MS/MS 分子ネットワーキングパイプライン（<code>.mzML</code> ファイルでの特徴検出（MZmine / OpenMS / matchms 系のツールスタック — エージェントが自ら選択）、アラインメント、古典的な分子ネットワーキング（GNPS 形式のコサインクラスタリング））を、固定パイプラインなしの単一コマンドから自律的に再生成しました。</em>
</p>

https://github.com/user-attachments/assets/dcd04ade-9c43-44a8-b3e3-a999d3dc895d

再現されたネットワークは、論文で報告されたクラスタレベルのトポロジーと一致し、Cytoscape で読み込み可能な `.graphml` ファイルと、それに対応する特徴量定量テーブルとして出力されます。スコープに関する注記: ここで再現されるのは **分子ネットワーキング** のステージのみです。元の研究における生物活性ガイド分画、手動アノテーションレビュー、ライブラリマッチング（GNPS / SIRIUS / CSI:FingerID）は自律実行の対象外です。

<p align="center">
  <img src="./docs/images/network.png" alt="LC-MS/MS molecular network autonomously reproduced by Mimosa-AI — GNPS-style cosine clustering exported as Cytoscape GraphML" width="80%">
</p>

---

## ベンチマーク

**ScienceAgentBench**（102 タスク、`task` モード — 計画レイヤをバイパスしてワークフロー合成と改良を単独で評価）で評価しました:

| モード                                  | 成功率       | Code-BLEU | タスクあたりコスト |
| --------------------------------------- | ------------ | --------- | ----------- |
| DeepSeek-V3.2 single-agent              | 38.2 %       | 0.898     | $0.05       |
| DeepSeek-V3.2 one-shot multi-agent      | 32.4 %       | 0.794     | $0.38       |
| **DeepSeek-V3.2 iterative-learning**    | **43.1 %**   | **0.921** | **$1.70**   |

> **ScienceAgentBench における DeepSeek-V3.2 反復学習で 43.1% の成功率 — シングルエージェントベースラインに対して +4.9 ポイント、タスクあたりコスト $1.70。**

> ScienceAgentBench 上で DeepSeek-V3.2 を用いた場合、反復学習は GPT-4o を改善しますが、Claude Haiku 4.5 では僅かな劣化をもたらします。モデル依存の挙動については[論文](https://arxiv.org/abs/2603.28986)で分析しています。PaperBench の結果は [`docs/papers_bench_evaluation.md`](./docs/papers_bench_evaluation.md) を参照してください。

---

## 仕組み

5 つのレイヤーが小さな dataclass スキーマで配線されています。詳細は [`docs/concepts/architecture.md`](./docs/concepts/architecture.md) を参照してください。

<p align="center">
  <img src="./docs/images/mimosa_overall.jpg" alt="Mimosa-AI architecture: planner, MCP tool manager, evolution engine, sandboxed SmolAgents workflow runner, multi-source per-claim verifier" width="90%">
</p>

| レイヤー | コンポーネント | 役割 |
|-------|-----------|--------------|
| 0 | **Planner** *(任意、`--goal` のみ)* | 高レベルの目標を離散的なタスクに分解します。 |
| 1 | **ToolManager + Perspicacité** | 設定されたアドレス/ポート範囲で MCP ツールを検出し、必要に応じて文献スニペットを取得します。 |
| 2 | **EvolutionEngine** | ワークフローを合成し、世代を超えて進化させます（後述）。 |
| 3 | **WorkflowRunner** | 合成された Python ワークフローを Hugging Face [SmolAgents](https://github.com/huggingface/smolagents)（`LocalPythonExecutor` と AST 許可リスト方式）のサンドボックスで実行し、LangGraph の状態を共有します。 |
| 4 | **VerifierEvaluator** | マルチソース・クレーム単位の verifier。次の変異を駆動します。 |

### 進化ループ — 実際に進化しているもの

ワークフローは **完全な Python プログラム** であり、ソースコードとして変異されます。**コードを遺伝子型として扱う方式 (code-as-genotype)** で、遺伝子型はワークフローファイル、表現型はそれがワークスペース上に生成するものです。

- **選択: Quality-Diversity アーカイブ** — **非構造アーカイブ**（離散グリッドではなく単一の名簿）。最大集団サイズ 20、単一のスカラ化スコア `qd_score = (1−w)·quality + w·novelty`（`w=0.25`）で評価し、満杯時は最低 `qd_score` のメンバーを退去させます。**新規性探索 (novelty search)** は **ゲノタイプ埋め込み (genotype embedding)** の行動記述子 — ワークフローの生成ソースコードを L2 正規化した埋め込み（既定はローカルの `all-MiniLM-L6-v2`、オプションで OpenAI `text-embedding-3-small`）— 上のコサイン距離 k-NN（`k=15`）で測られます。親は子の数の逆数によるルーレット選択で抽出され、アーカイブを拡散させます（`MAX_CHILDREN_PER_PARENT = 8`）。
- **変異: 停滞駆動のスコープ** — 変異の大胆さは、直近 4 回のプロンプト勾配がどの程度反復しているかの連続関数です。勝者に近い個体は保護されます。スコープ帯域は「プロンプトのみの微調整」から「トポロジー全面再考」まで及びます。
- **交叉** — 約 30 % の世代で 2 つの親（強いもの優先）を組み合わせます。
- **コールドスタート** — アーカイブが空の場合、ディスク上の過去実行を類似度フィルタ（MiniLM コサイン ≥ 0.5）でスキャンして探索を播種します。有用なワークフローはタスク間で転移します。

詳細な仕組み: [`docs/concepts/evolution-engine.md`](./docs/concepts/evolution-engine.md)。

### verifier — スコアが実際に意味するもの

各実行後、6 つの独立したクレームソースがワークスペースを参照し、成功極性を持つクレームを発行します:

| ソース | 視点 |
|--------|---------|
| **A** | 査読された実践（Perspicacité による文献グラウンディング経由） |
| **B** | 目標テキストそのもの — エージェントは依頼通りに成果を出したか? |
| **C** | エージェントのナラレーション — 主張された数値・成果物はディスクから再現できるか? |
| **D** | 数学的不変量 — 確率は [0,1] の範囲、形状の一貫性、NaN なし、保存則 |
| **E** | 計算再現性 — 宣言された依存関係が使用された import を網羅、絶対パスなし、確率的操作のシード |
| **F** | 統計的フィンガープリント — ベースラインを上回り、退化した予測がなく、漏洩シグネチャがない |

各クレームは、judge がワークスペースに対して記述する **決定論的 Python プログラム** によって検証されます — LLM にエージェントを信じるかどうかを再質問することはしません。**自己検証 (self-verification)** では、反トートロジーのトリップワイヤがエージェントの出力をそれ自身と比較するプログラムを拒絶します。

**ルーブリック盲変異 (rubric-blind mutation) — mutator はルーブリックを決して見ません。** フィードバックされる唯一のシグナルは `abstracted_prompt_gradient` です。これはクレーム、スコア、ソースの名前を含まない、コードネーム化された失敗モードの診断です。構造上、探索は決して見ないルーブリック語彙に過適合できません。

完全なパイプライン: [`docs/concepts/evaluation-pipeline.md`](./docs/concepts/evaluation-pipeline.md)。

---

## クイックスタート

### 1. インストール

```bash
pip install uv
git clone https://github.com/HolobiomicsLab/Mimosa-AI.git
cd Mimosa-AI
uv sync
```

### 2. 少なくとも 1 つの LLM キーを追加

プロジェクトのルートに `.env` を作成します。実際に使用するプロバイダのみが必要です。

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

### 3. MCP ツールの公開

Mimosa は設定のアドレス/ポート範囲（デフォルト `0.0.0.0:5000–5100`）で到達可能な任意の MCP サーバを検出します。

- **最も簡単な方法:** 関連プラットフォーム **[Toolomics](https://github.com/HolobiomicsLab/toolomics)** をインストールしてください — ワークスペースごとに 1 つの MCP shell サンドボックスを提供し、エージェントが必要に応じて科学パッケージをインストール（同一ワークスペース内の後続実行ではインストール済みのツールを再利用）できます。さらに、一般的な科学ツールスタックを公開するビルド済み MCP サーバ、共有ワークスペース管理、定型的な登録フローも提供します。
- **持ち込み方式:** `discovery_addresses` を任意の到達可能な MCP サーバに向けてください — `fastmcp` スクリプト、ToolHive、サードパーティ MCP コンテナなど。Toolomics は必須ではありません。詳細は [`docs/concepts/tools-and-mcp.md`](./docs/concepts/tools-and-mcp.md#operating-without-toolomics) を参照してください。

### 4. 実行

```bash
uv run main.py                   # interactive onboarding (recommended first time)
```

ウィザードをスキップする場合:

```bash
uv run main.py --task "Train a multitask model on Clintox to predict toxicity and FDA approval"
uv run main.py --goal "Reproduce experiments from https://arxiv.org/pdf/2306.00306 and compare results"
```

ワンショットではなく世代を超えて進化させるには `--learn` を追加します:

```bash
uv run main.py --task "..." --learn --config my_config.json
```

完全なクイックスタート: [`docs/getting-started/quickstart.md`](./docs/getting-started/quickstart.md)。

### 5. （任意）Perspicacité による科学的グラウンディング

[Perspicacité](https://github.com/HolobiomicsLab/Perspicacite-AI) はワークフロー合成とソース A のクレームを文献にグラウンディングします。実行中であれば、Mimosa は自動的に認識します。

```bash
git clone https://github.com/HolobiomicsLab/Perspicacite-AI.git && cd Perspicacite-AI
uv sync && uv run web_app_full.py
```

---

## 実行モード

| モード | 使用場面 | コマンド |
|------|----------|---------|
| `--task` | 単一の焦点を絞った操作 | `uv run main.py --task "..."` |
| `--goal` | 計画を要する複数ステップの目標 | `uv run main.py --goal "..."` |
| `--learn` | いずれかのモードに追加 — 世代を超えて進化 | `... --learn` |
| `--single_agent` | マルチエージェント合成をスキップ（高速、学習なし） | `... --single_agent` |
| `--manual` | 個別の MCP ツールをテストする対話的 CLI | `uv run main.py --manual` |
| バッチ | タスクの CSV を評価 | `... --papers <csv>` |
| ベンチマーク | ScienceAgentBench | `... --science_agent_bench` |

詳細: [`docs/usage/modes.md`](./docs/usage/modes.md)、[`docs/usage/learning.md`](./docs/usage/learning.md)、[`docs/reference/cli.md`](./docs/reference/cli.md)。

---

## 監査証跡と再生

Mimosa は科学的用途のために構築されており、すべての決定は事後に検査可能です。

| ツール | 役割 |
|------|--------------|
| `uv run workflow_evolution_anim.py` | 進化ツリーを辿り、各世代のトレース（思考、ツール呼び出し、ルーブリックの合否）を再生するインタラクティブビューア。 |
| `uv run main.py --memory_cli` | 完了した実行のメモリに対する RAG ベースの Q&A。「*task_builder が使用した分類器は何か?*」のように、スクロールせずに質問できます。 |
| `uv run memory_timelapse.py <uuid>` | 反復にわたるメモリ成長のフレーム単位アニメーション表示。 |
| `sources/workflows/<uuid>/workflow_genotype_<uuid>.py` | エージェントが実行した正確な Python。DSL なし。 |
| `sources/workflows/<uuid>/lineage_<uuid>.json` | この世代の親と演算子（`seed | mutation | crossover`）。 |
| `sources/workflows/<uuid>/evolution_prompt_<uuid>.md` | このコードを生成した正確な LLM プロンプト。同じプロンプト + シード = 同じコード。 |
| `sources/workflows/<uuid>/evolution_tree.png` | `--learn` 実行全体のレンダリングされた系譜ツリー。 |
| `sources/workflows/<uuid>/reward_progress.png` | 反復に対するスコア曲線。 |
| `runs_capsule/<capsule_name>/` | 共有または再実行のためにアーカイブされた最終ワークスペースのスナップショット。 |

完全なレイアウト: [`docs/usage/transparency.md`](./docs/usage/transparency.md)、[`docs/usage/workspace.md`](./docs/usage/workspace.md)。

---

## 設定

`config_default.json` を `my_config.json` にコピーして編集します。最も頻繁に触れるフィールド:

| フィールド | 制御する内容 |
|-------|------------------|
| `workspace_dir` | 共有ワークスペース — 生成されたファイルがすべてここに現れます |
| `discovery_addresses` | MCP 検出のための IP + ポート範囲 |
| `workflow_llm_model` | マルチエージェントワークフローを合成（例: `anthropic/claude-opus-4-5`） |
| `smolagent_model_id` | 実行エージェントが使用するモデル |
| `judge_model` | verifier プログラムを記述し、ソフト判定を下す LLM |
| `learned_score_threshold` | `--learn` モードでの早期停止しきい値（デフォルト `0.92`） |
| `max_learning_evolve_iterations` | 世代数の上限（デフォルト `25`） |

完全なリファレンス: [`docs/reference/configuration.md`](./docs/reference/configuration.md)。

---

## 評価

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

> ⚠️ 偏りのない評価のためには、まず `./cleanup.sh` を実行して Mimosa がキャッシュされたワークフローを再利用しないようにしてください。

各ベンチマークのセットアップ詳細: [`docs/science_agent_bench_evaluation.md`](./docs/science_agent_bench_evaluation.md)、[`docs/papers_bench_evaluation.md`](./docs/papers_bench_evaluation.md)。

---

## 通知とテレメトリ

- **Pushover** — スマートフォンへのリアルタイム進捗通知。`PUSHOVER_USER` と `PUSHOVER_TOKEN` を設定します。詳細: [`docs/usage/notifications.md`](./docs/usage/notifications.md)。
- **Langfuse** — すべての LLM 呼び出しのスパンレベルトレース。Langfuse リポジトリで `docker compose up -d` を実行し、`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_PRIVATE_KEY` を `.env` に追加します。ダッシュボードは `http://localhost:3000`。詳細: [`docs/usage/telemetry.md`](./docs/usage/telemetry.md)。

---

## 関連研究

Mimosa-AI は LLM 駆動のプログラム探索と自律研究システムの、小規模ながら活発な系譜に位置しています。これらのいずれかを包含すると主張するわけではありません。それぞれが異なる問いに答えています:

| プロジェクト | 内容 | Mimosa との相違点 |
|---------|--------------|--------------------|
| [Sakana AI Scientist](https://github.com/SakanaAI/AI-Scientist) | ML におけるエンドツーエンドの論文生成 | Mimosa は **タスクごとのワークフロー合成** を QD + verifier で最適化するもので、論文全体の生成ではありません |
| [DiscoPOP](https://github.com/SakanaAI/DiscoPOP)（Lange et al. 2024） | LLM 駆動による選好最適化アルゴリズムの発見 | 同じ「コードに対する変異オペレータとしての LLM」パラダイム。Mimosa はこれを損失関数ではなくマルチエージェントワークフローコードに適用します |
| [FunSearch](https://github.com/google-deepmind/funsearch)（Romera-Paredes et al. 2024） | LLM 誘導による Python 関数の進化的探索 | Mimosa はマルチエージェントプログラム全体を進化させ、単一の適応度関数の代わりにマルチソース・クレーム単位 verifier を追加します |
| [ELM](https://github.com/CarperAI/OpenELM)（Lehman et al. 2022） | LLM を介したコード上の Quality-Diversity | 最も近い QD 祖先。Mimosa の行動記述子はドメイン固有ではなくワークフロー構造的です |
| AIDE | Kaggle ライクなタスクでの自動 ML パイプライン | Mimosa はより広範な科学的再現（ScienceAgentBench、PaperBench、実験室データ）を対象とし、監査可能なクレーム単位 verifier を提供します |

比較研究を出版する場合、詳細な位置付けは[論文](https://arxiv.org/abs/2603.28986)に記載されています。

---

## 完全なドキュメント

```bash
uvx --with mkdocs-material mkdocs serve   # live preview at http://localhost:8000
uvx --with mkdocs-material mkdocs build   # static HTML to ./site
```

サイト設定: [`mkdocs.yml`](./mkdocs.yml)。インデックス: [`docs/index.md`](./docs/index.md)。

---

## コントリビューション

パッチ、MCP ツール、評価器、新しいクレームソースを歓迎します。まずは [`CONTRIBUTING.md`](./CONTRIBUTING.md)、[Developer guide](./docs/DEVELOPER_GUIDE.md)、および [`CLA/`](./CLA/) のコントリビューション規約から始めてください。

---

## ライセンス

Apache 2.0。コントリビューション規約については [`NOTICE`](./NOTICE)、[`docs/licensing-notes.md`](./docs/licensing-notes.md)、および [`CLA/`](./CLA/) フォルダを参照してください。

---

## 本研究を引用する

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
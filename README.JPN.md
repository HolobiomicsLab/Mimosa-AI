<div align="center">
<br>

<img src="./docs/images/logo_mimosa.png" width="22%" style="border-radius: 8px;" alt="Mimosa-AI Logo">

</div>

<h1 align="center">Mimosa-AI 🌼🔬</h1>

<p align="center">
  <a href="./README.md">🇬🇧 English</a> &nbsp;|&nbsp;
  <a href="./README.CHS.md">🇨🇳 简体中文</a> &nbsp;|&nbsp;
  <a href="./README.CHT.md">🇹🇼 繁體中文</a> &nbsp;|&nbsp;
  <a href="./README.JPN.md">🇯🇵 日本語</a> &nbsp;|&nbsp;
  <a href="./README.KOR.md">🇰🇷 한국어</a>
</p>

<p align="center">
    <em>自律型 AI 科学者ワークフローを進化させるオープンソースフレームワーク。</em>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2603.28986"><img src="https://img.shields.io/badge/arXiv-2603.28986-b31b1b.svg?logo=arxiv&style=flat-square&logoColor=white" alt="arXiv"></a>
    <a href="https://holobiomicslab.cnrs.fr/"><img src="https://img.shields.io/badge/ウェブサイト-holobiomicslab.cnrs.fr-4caf82?style=flat-square&logo=globe&logoColor=white" alt="ウェブサイト"></a>
</p>

<p align="center">
    <a href="https://github.com/HolobiomicsLab/Mimosa-AI/stargazers">
        <img src="https://img.shields.io/github/stars/HolobiomicsLab/Mimosa-AI?style=social" alt="GitHub Stars">
    </a>
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img src="https://img.shields.io/badge/ライセンス-Apache%202.0-blue.svg?style=flat-square" alt="ライセンス：Apache 2.0">
    </a>
</p>

---

> ***Mimosa-AI 🌼*** — 感知し、学習し、適応するオジギソウのように、Mimosa はエンドツーエンドの研究を自律的に遂行し、発表された研究成果を再現するために構築された AI 科学者フレームワークです。***Mimosa-AI*** は利用可能なツールを自動で発見し、研究目標を構造化されたワークフローに分解し、マルチエージェントワークフローの進化を通じた反復的な自己改善によってマルチエージェント実行を推進します。

**ユースケース：**
- 厳密で監査可能なワークフローによる科学的研究の再現
- バイオインフォマティクス、分子ドッキング、メタボロミクスなど、完全なパイプラインの自動化

## 目次

- [仕組み](#仕組み)
- [例：生物活性分子ネットワーク論文の再現](#例生物活性分子ネットワーク論文の再現)
- [前提条件](#前提条件)
- [インストール](#インストール)
- [設定](#設定)
- [Mimosa の実行](#mimosa-の実行)
- [出力](#出力)
- [マルチエージェントワークフローの進化による学習](#マルチエージェントワークフローの進化による学習)
- [透明性](#透明性)
- [コマンドライン引数](#コマンドライン引数)
- [評価](#評価)
- [スマートフォン通知](#スマートフォン通知)
- [テレメトリ設定](#テレメトリ設定)
- [ライセンス](#ライセンス)

## 仕組み

ユーザーが ***Mimosa-AI*** に研究目標を与えます。

- ***Mimosa*** はローカルネットワーク上または Toolhive 経由で利用可能な MCP ベースのツール（データ分析ユーティリティからウェブブラウザ、質量分析計などの実験室機器まで）を自動的に発見します。
- ユーザーの目標と発見されたツールを使用して、***Mimosa*** は問題を分解し、各タスクに合わせたマルチエージェントワークフローを構築します。
- 各タスクは自律的に実行されます。失敗は反復学習ループによる自己改善に活用されます。
- ***Mimosa*** は結果、可視化、レポート、ログ、および関連するすべての成果物を含む最終カプセルを生成します。

![schema](./docs/images/mimosa_overall.jpg)

## 例：生物活性分子ネットワーク論文の再現

> **目標** — [Nothias et al. (2018)](https://www.researchgate.net/publication/323525305_Bioactivity-Based_Molecular_Networking_for_the_Discovery_of_Drug_Leads_in_Natural_Product_Bioassay-Guided_Fractionation) をエンドツーエンドで再現：`.mzML` ファイルを起点として（生データ変換は完了済みと仮定）、特徴検出からネットワーク可視化まで。

Mimosa-AI には論文、生データ、および論文に完全には記載されていない設定の詳細をカバーするドメイン固有のスキルが提供されました。

**実行タイムラプス**

https://github.com/user-attachments/assets/dcd04ade-9c43-44a8-b3e3-a999d3dc895d

**出力結果**

![molecular_network](./docs/images/network.png)

得られた分子ネットワークは、クラスター分離とエッジ重みを含め、元の論文で報告されたトポロジーと一致し、自律的に再現されました。

---

## 前提条件

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)（推奨）または pip
- 実行中の [Toolomics MCP サーバー](https://github.com/HolobiomicsLab/toolomics)

---

## インストール

### 1. リポジトリのクローンと仮想環境の作成

```bash
# uv を使用（推奨）
pip install uv
uv venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# または pip を使用
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 依存関係のインストール

```bash
cd mimosa
uv pip install -r requirements.txt
```

### 3. API キーの設定

プロジェクトルートに `.env` ファイルを作成します。使用する予定の LLM プロバイダーのキーのみを含めてください：

```env
ANTHROPIC_API_KEY=...       # Claude — ワークフローオーケストレーションに推奨
OPENAI_API_KEY=...          # OpenAI モデル - オプション
MISTRAL_API_KEY=...         # Mistral モデル - オプション
DEEPSEEK_API_KEY=...        # Deepseek - オプション
HF_TOKEN=...                # HuggingFace プロバイダー、オプション
OPENROUTER_API_KEY=...      # OpenRouter 経由で任意のモデルにアクセス

# オプション — Langfuse によるオブザーバビリティ
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_PRIVATE_KEY=...
```

### 4. MCP サーバーの起動

[HolobiomicsLab/toolomics](https://github.com/HolobiomicsLab/toolomics) のセットアップ手順に従ってください。ポート範囲（例：`5000–5100`）で実行するように設定します。カスタム MCP ツールは [Toolomics ドキュメント](https://github.com/HolobiomicsLab/toolomics/README.md) から追加できます。

---

## 設定

```bash
cp config_default.json my_config.json
```

`my_config.json` を編集します。主要パラメーター：

| パラメーター | 説明 |
|------------|------|
| `workspace_dir` | Toolomics ワークスペースのパス — 生成されたすべてのファイルがここに保存されます |
| `discovery_addresses` | MCP サーバー探索用の IP + ポート範囲 |
| `planner_llm_model` | タスク分解と計画のための LLM |
| `prompts_llm_model` | ワークフロープロンプト生成のための LLM |
| `workflow_llm_model` | マルチエージェントオーケストレーションのための LLM（推奨：anthropic/claude-opus-4-5 または z-ai/glm-5） |
| `smolagent_model_id` | SmolAgents 実行サブタスクのモデル |
| `judge_model` | 出力の自己評価とスコアリングのための LLM |
| `learned_score_threshold` | 結果を受け入れて反復を停止する最低スコア |
| `max_learning_evolve_iterations` | 結果を受け入れる前の最大自己改善反復回数 |

---

## Mimosa の実行

Mimosa は **ゴール（Goal）** と **タスク（Task）** の 2 つの実行モードをサポートします。

### ゴールモード — 複数ステップの科学的目標

複数の異なる操作にわたる計画が必要な場合に使用します（例：論文の再現、ML パイプラインの構築）。

```bash
uv run main.py --goal "あなたの科学的目標" --config my_config.json
```

**例：**
```bash
uv run main.py \
  --goal "「Dual Aggregation Transformer for Image Super-Resolution」(https://arxiv.org/pdf/2306.00306) の実験を再現して結果を比較する。" \
  --config my_config.json

uv run main.py \
  --goal "タンパク質-リガンド結合親和性を予測する機械学習モデルを開発する。" \
  --config my_config.json
```

### タスクモード — 単一の粒度の高い操作

長期的な計画を必要としない、集中した自己完結型の操作に使用します。

```bash
uv run main.py --task "あなたのタスクの説明" --config my_config.json
```

**例：**
```bash
uv run main.py \
  --task "Clintox データセットでマルチタスクモデルをトレーニングして、薬物毒性と FDA 承認状況を予測する。" \
  --config my_config.json

uv run main.py --task "創薬のためのグラフニューラルネットワークに関する文献レビューを実施する。" --config my_config.json
```

> **注意：** いずれのモードを実行する前に、Toolomics をインストールし、MCP サーバーが実行されていることを確認してください。

## 出力

実行中にファイルが Toolomics の `workspace/` ディレクトリに書き込まれます。完了時には、タイムスタンプ付きのカプセルとして完全なスナップショットが `runs_capsule/` に保存されます。

---

## マルチエージェントワークフローの進化による学習

***Mimosa-AI*** は科学的タスクのための特化したワークフローを動的に合成し、**ダーウィン的に着想を得たマルチエージェントワークフローの進化**を通じて失敗から学習します：
タスクを固定パイプラインに強制的に当てはめるのではなく、システムは各タスクにカスタムのマルチエージェントグラフを構成し、単一候補局所探索によって洗練させます。各反復では最もパフォーマンスの高いワークフローのみが後継者を生成し、改善のみが保持されます。時間の経過とともに、システムは実証済みのワークフライブラリを構築し、類似した将来のタスクがゼロからではなく、強いベースラインから開始できるようにします。

新しいタスクには、完全な自律性の前にシステムが能力を構築できるよう、学習モードから始めることをお勧めします。

**学習モードで開始**

```bash
uv run main.py --task "Clintox データセットでマルチタスクモデルをトレーニングして、薬物毒性と FDA 承認状況を予測する" --learn --config my_config.json
```

![dgm](./docs/images/workflow_mutation.png)

**進捗の可視化：**

***Mimosa-AI*** があるタスクの学習フェーズを完了すると、時間の経過とともにどのように改善したかを正確に可視化できます。各試行にわたるパフォーマンス向上を示す報酬進捗プロットが自動的に保存されます。

報酬進捗プロットは `sources/workflows/<uuid>` フォルダーの `reward_progress.png` というファイル名で保存されます。

***例：***

![dgm](./docs/images/evolve_example.png)

## 透明性

インタラクティブデバッガー `memory_explorer.py` が付属しており、エージェントの実行を細粒度でステップスルーできます。

ワークフロー `<uuid>` で起動します（例：`memory_explorer.py 20260115_113303_9bb63437`）

これにより完全な実行トレース（思考、ツール呼び出し、出力）が再生され、意思決定がどのように展開されたかを正確に検査できます。

## コマンドライン引数

### 実行モード

| 引数 | 説明 |
|------|------|
| `--goal GOAL` | 高レベルの研究目標、論文再現、または科学的質問を指定（プランナーモード） |
| `--task TASK` | 単一タスクを実行：文献レビュー、データセットのダウンロード、機械学習モデルの実装など |
| `--manual` | MCP をデバッグし、***Mimosa*** ツールを直接テストするためのインタラクティブ CLI モード |
| `--papers <CSV パス>` | 研究論文とプロンプトを含む CSV データセットで評価 |
| `--science_agent_bench` | ScienceAgentBench で評価 |

### その他のパラメーター

| 引数 | 説明 |
|------|------|
| `--learn` | タスクパフォーマンスを最適化するための反復学習を有効化 |
| `--max_evolve_iterations N` | 最大学習反復回数 |
| `--csv_runs_limit N` | 評価する CSV エントリの数を制限 |
| `--scenario <シナリオファイル名>` | スコアリング実行に LLM ジャッジの代わりに特定のシナリオベースのアサーションを使用 |
| `--single_agent` | シングルエージェントモード。高速だが、学習を通じた改善ができない |
| `--debug` | より詳細なログのためのデバッグモードを有効化 |

---

## 評価

***Mimosa-AI*** は [ScienceAgentBench](https://arxiv.org/abs/2410.05080) または [PaperBench](https://arxiv.org/pdf/2504.01848) で評価できます。

⚠️ 偏りのない評価のため、まず `./cleanup.sh` を実行することをお勧めします。これにより ***Mimosa*** が既存またはキャッシュされたワークフローを使用することを防ぎます。

### ScienceAgentBench

ScienceAgentBench で評価するには：

1. ScienceAgentBench の完全なデータセットをダウンロードします：
[データセットリンク](https://buckeyemailosu-my.sharepoint.com/personal/chen_8336_buckeyemail_osu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch%2Fbenchmark%2Ezip&parent=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch&ga=1)
2. パスワード `scienceagentbench` で解凍します
3. `benchmark/benchmark/datasets` フォルダーの内容を `Mimosa-AI/datasets/scienceagentbench/datasets` にコピーします

**学習モードで ScienceAgentBench を評価**
```sh
uv run main.py --science_agent_bench --learn
```

**10 タスクに制限し、学習反復を 4 回に制限した ScienceAgentBench 評価**

```sh
uv run main.py --science_agent_bench --csv_runs_limit 10 --max_evolve_iterations 4
```

### PaperBench

**学習モードで OpenAI PaperBench を評価**

OpenAI PaperBench は、論文 `PaperBench: Evaluating AI's Ability to Replicate AI Research` からの AI 研究を複製する AI エージェントの能力を評価するベンチマークです。

```sh
uv run main.py --papers datasets/paper_bench.csv --csv_runs_limit 20  --learn
```

⚠️ これにより、すべての論文再現試行の結果が `runs_capsule/` フォルダーに保存されます。完全な評価については [Paper Bench ドキュメント](https://github.com/openai/frontier-evals/tree/main/project/paperbench) を参照してください。

**カスタム研究論文ベンチマークでの評価**

1. `paper_bench.csv` と同じ形式のベンチマーク CSV を `datasets/<あなたのベンチマーク名>.csv` に置きます。

2. ベンチマークで実行：

```sh
uv run main.py --papers datasets/<あなたのベンチマーク名>.csv --csv_runs_limit 20  --learn
```

---

## スマートフォン通知

Pushover 通知で ***Mimosa*** のステータスについてリアルタイムの更新を受信します。

### セットアップ手順

1. **Pushover アカウントの作成**
   - [pushover.net](https://pushover.net/) にアクセス
   - 登録して **ユーザーキー** をメモします

2. **アプリケーションの作成**
   - Pushover ダッシュボードで「アプリケーション/API トークンを作成」をクリック
   - 「***Mimosa***」と名前を付けて生成された **API トークン** をコピーします

3. **環境変数の設定**
   ```bash
   export PUSHOVER_USER="あなたのユーザーキー"
   export PUSHOVER_TOKEN="あなたの API トークン"
   ```

4. **モバイルアプリのインストール**
   - デバイスのアプリストアから Pushover をダウンロード
   - Pushover アカウントでログイン

---

## テレメトリ設定

Langfuse を使用してリアルタイムのオブザーバビリティダッシュボードで AI エージェントを監視およびデバッグします。

### クイックスタート

1. **Langfuse をローカルにデプロイ**
   ```bash
   git clone https://github.com/langfuse/langfuse.git
   cd langfuse
   docker compose up -d
   ```

2. **環境変数の設定**

   `.env` ファイルに追加：
   ```env
   LANGFUSE_PUBLIC_KEY=あなたの公開キー
   LANGFUSE_PRIVATE_KEY=あなたの秘密キー
   ```

3. **ダッシュボードへのアクセス**

   ***Mimosa-AI*** の実行中に `http://localhost:3000` にアクセス

### 利用可能なメトリクス

テレメトリダッシュボードが提供するもの：
- **エージェント実行トレース**：ステップバイステップのワークフロー可視化
- **パフォーマンスメトリクス**：レスポンスタイムと成功率
- **エラーデバッグ**：詳細な障害分析
- **リソース使用量**：トークン消費と API 呼び出し

**ダッシュボード例：**
![Langfuse Dashboard](https://langfuse.com/images/cookbook/integration-smolagents/smolagent_example_trace.png)

> **注意：** テレメトリはオプションですが、デバッグとパフォーマンス最適化に推奨されます。

---

## ライセンス

このリポジトリは Apache License 2.0 の下で公開配布されています。Apache 2.0 は、商業利用、修正、再配布を許可する寛容なオープンソースライセンスであり、再配布時には適用されるライセンス、著作権、特許、商標、帰属、および NOTICE の通知を保持し、修正されたファイルには変更が行われたことを示す目立つ通知を付けることが条件です。

貢献とライセンスの詳細については、以下を参照してください：
- `NOTICE`
- `docs/licensing-notes.md`
- `CLA/INDIVIDUAL_CLA.md`
- `CLA/EMPLOYER_AUTHORIZATION.md`

---

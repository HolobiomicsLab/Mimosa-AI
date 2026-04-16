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
    <em>自律的な科学研究のための自己進化AIフレームワーク</em>
</p>
<p align="center">
  🧬 自己進化型マルチエージェントワークフロー &nbsp;·&nbsp;
  🔍 MCPベースのツール自動検出 &nbsp;·&nbsp;
  🔁 ダーウィン式ワークフロー最適化 &nbsp;·&nbsp;
  📦 完全な監査証跡と再現性 &nbsp;·&nbsp;
</p>

<p align="center">
    <em>Mimosa-AI は Nothias et al. (2018) をエンドツーエンドで自律的に再現しました——生の .mzML ファイルから分子ネットワークまで——たった1つのコマンドで。</em>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2603.28986"><img src="https://img.shields.io/badge/arXiv-2603.28986-b31b1b.svg?logo=arxiv&style=flat-square&logoColor=white" alt="arXiv プレプリント"></a>
    <a href="https://doi.org/10.48550/arXiv.2603.28986"><img src="https://img.shields.io/badge/DOI-10.48550%2FarXiv.2603.28986-blue?style=flat-square" alt="DOI"></a>
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

## ライブデモ：自律的な論文再現

https://github.com/user-attachments/assets/dcd04ade-9c43-44a8-b3e3-a999d3dc895d

**結果：** 以下の分子ネットワークは、生の `.mzML` ファイルから自律的に再現されたものです。[Nothias et al. (2018)](https://www.researchgate.net/publication/323525305_Bioactivity-Based_Molecular_Networking_for_the_Discovery_of_Drug_Leads_in_Natural_Product_Bioassay-Guided_Fractionation) で報告されたトポロジー——クラスター分離とエッジ重みを含む——と完全に一致しています。

<p align="center">
  <img src="./docs/images/network.png" alt="再現された分子ネットワーク" width="80%">
</p>

---

## ベンチマーク結果

**ScienceAgentBench** での評価（102タスク、`task` モード）：

| モード | 成功率 | Code-BLEU スコア | タスク単価 |
|--------|--------|------------------|-----------|
| DeepSeek-V3.2 シングルエージェント | 38.2% | 0.898 | $0.05 |
| DeepSeek-V3.2 ワンショット マルチエージェント | 32.4% | 0.794 | $0.38 |
| **DeepSeek-V3.2 反復学習** | **43.1%** | **0.921** | **$1.7** |

> 反復学習は GPT-4o を改善しますが、Claude Haiku 4.5 には微妙な性能低下をもたらします——詳細はモデル依存動作分析について[論文](https://arxiv.org/abs/2603.28986)を参照してください。

---

## Mimosa-AI とは？

> ***Mimosa-AI 🌼*** — 感知し、学習し、適応するオジギソウのように、Mimosa はタスク固有のマルチエージェントワークフローを自動合成し、実行フィードバックを通じて洗練させるオープンソースの自律科学研究フレームワークです。MCPベースのツール発見、コード生成エージェント、LLMベースの評価を中心に構築されており、学術研究者にとってクローズドな「ブラックボックス」システムの代わりとなるモジュール化された監査可能な選択肢を提供します。

**実現できること：**
- **科学的研究の再現**——トレーサビリティと厳密性を持って、生データから出版可能な図表まで
- **計算パイプラインの自動化**——バイオインフォマティクス、分子ドッキング、メタボロミクス、MLなど
- **自己進化**——ダーウィン的ワークフロー変異による——各失敗が次の試みに活かされる

### アーキテクチャ概要

フレームワークは5つの層で構成されています：

1. **計画層**（オプション）——高レベルの科学目標を個別タスクに分解
2. **ツール発見層**——Toolomics を通じてローカルネットワーク上の MCP ツールを自動発見
3. **メタオーケストレーション層**——タスク固有のマルチエージェントワークフローを合成し、専門エージェントにツールを割り当て
4. **エージェント実行層**——コード生成エージェントが発見されたツールと科学ライブラリを使用してサブタスクを実行
5. **評判/評価層**——LLMベースの評判が出力をスコアリング；学習モードでは反復的なワークフロー改善を推進

<p align="center">
  <img src="./docs/images/mimosa_overall.jpg" alt="Mimosa アーキテクチャ概要" width="90%">
</p>

ベンチマーク `task` モードでは、計画層（1）をバイパスすることで、ワークフロー合成と改善を独立して評価できます。

---

## 目次

- [Toolomics とは何か？必要ですか？](#toolomics-とは何か必要ですか)
- [前提条件](#前提条件)
- [インストール](#インストール)
- [オプション：Perspicacité による科学的根拠付け](#オプションperspicacité-による科学的根拠付け)
- [設定](#設定)
- [Mimosa の実行](#mimosa-の実行)
- [ワークスペースと監査証跡](#ワークスペースと監査証跡)
- [マルチエージェントワークフローの進化による学習](#マルチエージェントワークフローの進化による学習)
- [透明性](#透明性)
- [コマンドライン引数](#コマンドライン引数)
- [評価](#評価)
- [スマートフォン通知](#スマートフォン通知)
- [テレメトリ設定](#テレメトリ設定)
- [ライセンス](#ライセンス)

---

## Toolomics とは何か？必要ですか？

**[Toolomics](https://github.com/HolobiomicsLab/toolomics)** は Mimosa のコンパニオンプラットフォームであり、MCP サーバー管理に使用されます。科学ツール（データ分析ユーティリティ、ウェブサービス、実験室機器）を発見可能な MCP サービスとして公開し、Mimosa がタスクアーティファクトを読み書きする共有ワークスペースを提供し、Mimosa のコアを変更せずにカスタムツールを登録できます。

**必要ですか？** はい——Mimosa のどのモードを実行する前も、Toolomics が起動している必要があります。良いニュースは、セットアップには数分しかかかりません。

- Mimosa と Toolomics はどちらも Apache 2.0 ライセンスで、無料で使用できます。
- Toolomics は設定可能なポート範囲（デフォルト `5000–5100`）でローカルに実行されます。
- [Toolomics ドキュメント](https://github.com/HolobiomicsLab/toolomics)を通じてカスタム MCP ツールを追加できます。

> **クイックスタートパス：** Toolomics をクローン → デフォルトのポート範囲で起動 → Mimosa を実行。LLM API キー以外、クラウドアカウントや有料サービスは不要です。

---

## 前提条件

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)（推奨）または pip
- 実行中の [Toolomics MCP サーバー](https://github.com/HolobiomicsLab/toolomics)

---

## インストール

### 1. 依存関係のインストール

```bash
# uv を使用（推奨——仮想環境の作成と依存関係のインストールを一括で実行）
pip install uv
uv sync

# または pip を使用
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install .
```

### 2. API キーの設定

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

### 3. MCP サーバーの起動

[HolobiomicsLab/toolomics](https://github.com/HolobiomicsLab/toolomics) のセットアップ手順に従ってください。ポート範囲（例：`5000–5100`）で実行するように設定します。

カスタム MCP ツールは [Toolomics ドキュメント](https://github.com/HolobiomicsLab/toolomics/README.md) から追加できます。

### 4. （オプション）科学的根拠付けのため Perspicacité を起動

**[Perspicacité](https://github.com/HolobiomicsLab/Perspicacite-AI)** は、Mimosa のワークフロー作成と評価に科学的根拠を提供するオプションのコンパニオン AI です。起動中、Mimosa は自動的に Perspicacité と連携し、出力の科学的厳密性を向上させます。

**セットアップ：**

```bash
git clone https://github.com/HolobiomicsLab/Perspicacite-AI.git
cd Perspicacite-AI
uv sync
uv run web_app_full.py
```

以上です —— Perspicacité が起動し、連携準備が整います。別のターミナルで通常通り Mimosa を起動すれば、自動的に相互作用できるようになります。

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
| `workflow_llm_model` | マルチエージェントオーケストレーションのための LLM（推奨：`anthropic/claude-opus-4-5` または `z-ai/glm-5`） |
| `smolagent_model_id` | SmolAgents 実行サブタスクのモデル |
| `judge_model` | 出力の自己評価とスコアリングのための LLM |
| `learned_score_threshold` | 結果を受け入れて反復を停止する最低スコア |
| `max_learning_evolve_iterations` | 結果を受け入れる前の最大自己改善反復回数 |

---

## Mimosa の実行

Mimosa は **ゴール（Goal）** と **タスク（Task）** の2つの実行モードをサポートします。

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

> **ベンチマーク注：** 論文で報告された結果は `task` モードで計測されており、ワークフロー合成と反復改善を独立して評価するために計画層が無効化されています。
>
> **注意：** いずれのモードを実行する前に、Toolomics をインストールし、MCP サーバーが実行されていることを確認してください。

---

## ワークスペースと監査証跡

実行中、Mimosa は `workspace_dir` で設定された Toolomics ワークスペース内のファイルを読み書きします。実行完了時、ワークスペースの内容はタイムスタンプ付きのフォルダとして `runs_capsule/` にコピーされ、アーカイブとして保存されます。

- **Toolomics `workspace/`** — ライブ作業ディレクトリ：中間ファイル、スクリプト、ダウンロード、生成された出力
- **`sources/workflows/<uuid>/`** — 生成されたワークフローと実行メタデータ：`state_result.json`、`evaluation.txt`、`reward_progress.png`、`memory/` トレース
- **`runs_capsule/<capsule_name>/`** — 実行のアーカイブスナップショット、後での検査・比較・共有のため
- **`memory_explorer.py <uuid>`** — ワークフロー実行をステップバイステップで再生し、エージェントトレース・ツール呼び出し・出力を検査

これらの場所が合わさって Mimosa の完全な監査証跡を形成します：計画・実行・評価・産出の全過程。

---

## マルチエージェントワークフローの進化による学習

***Mimosa-AI*** は科学的タスクのための特化したワークフローを動的に合成する**自己進化マルチエージェントシステム**です。**ダーウィン的単一候補局所探索**によりワークフローを進化させます：各反復では最もパフォーマンスの高いワークフローのみが後継者を生成し、改善のみが保持されます。時間の経過とともに、実証済みのワークフローライブラリが構築され、類似した将来のタスクがゼロからではなく強いベースラインから開始できるようになります。

新しいタスクには、**まず学習モードで開始する**ことをお勧めします——完全な自律性の前にシステムが能力を構築できるように。

**学習モードで開始**

```bash
uv run main.py --task "Clintox データセットでマルチタスクモデルをトレーニングして、薬物毒性と FDA 承認状況を予測する" --learn --config my_config.json
```

<p align="center">
  <img src="./docs/images/workflow_mutation.png" alt="ワークフロー変異ダイアグラム" width="80%">
</p>

**進捗の可視化：**

***Mimosa-AI*** が学習フェーズを完了すると、報酬進捗プロット（各試行にわたるパフォーマンス向上）が `sources/workflows/<uuid>/reward_progress.png` に自動保存されます。

<p align="center">
  <img src="./docs/images/evolve_example.png" alt="報酬進捗の例" width="80%">
</p>

---

## 透明性

インタラクティブデバッガー `memory_explorer.py` が付属しており、エージェントの実行を細粒度でステップスルーできます。

```bash
python memory_explorer.py 20260115_113303_9bb63437
```

これにより完全な実行トレース（思考、ツール呼び出し、出力）が再生され、各意思決定がどのように展開されたかを正確に検査できます。

---

## コマンドライン引数

### 実行モード

| 引数 | 説明 |
|------|------|
| `--goal GOAL` | 高レベルの研究目標、論文再現、または科学的質問を指定（プランナーモード） |
| `--task TASK` | 単一タスクを実行：文献レビュー、データセットのダウンロード、ML モデルの実装など |
| `--manual` | MCP をデバッグし、***Mimosa*** ツールを直接テストするためのインタラクティブ CLI モード |
| `--papers <CSV パス>` | 研究論文とプロンプトを含む CSV データセットで評価 |
| `--science_agent_bench` | ScienceAgentBench で評価 |

### その他のパラメーター

| 引数 | 説明 |
|------|------|
| `--learn` | タスクパフォーマンスを最適化するための反復学習を有効化 |
| `--max_evolve_iterations N` | 最大学習反復回数 |
| `--csv_runs_limit N` | 評価する CSV エントリの数を制限 |
| `--scenario <シナリオファイル名>` | スコアリングに LLM ジャッジの代わりに特定シナリオベースのアサーションを使用 |
| `--single_agent` | シングルエージェントモード——高速だが学習による改善不可 |
| `--debug` | より詳細なログのためのデバッグモードを有効化 |

---

## 評価

***Mimosa-AI*** は [ScienceAgentBench](https://arxiv.org/abs/2410.05080) または [PaperBench](https://arxiv.org/pdf/2504.01848) で評価できます。

⚠️ 偏りのない評価のため、まず `./cleanup.sh` を実行してキャッシュされたワークフローの使用を防いでください。

### ScienceAgentBench

1. ScienceAgentBench の完全なデータセットをダウンロード：
   [データセットリンク](https://buckeyemailosu-my.sharepoint.com/personal/chen_8336_buckeyemail_osu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch%2Fbenchmark%2Ezip&parent=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch&ga=1)
2. パスワード `scienceagentbench` で解凍
3. `benchmark/benchmark/datasets/` → `Mimosa-AI/datasets/scienceagentbench/datasets/` にコピー

**学習モードでの完全評価：**
```sh
uv run main.py --science_agent_bench --learn
```

**クイック評価（10タスク、4回の学習反復）：**
```sh
uv run main.py --science_agent_bench --csv_runs_limit 10 --max_evolve_iterations 4
```

### PaperBench

OpenAI PaperBench は AI 研究の複製における AI エージェントの能力を評価します（*PaperBench: Evaluating AI's Ability to Replicate AI Research*）。

```sh
uv run main.py --papers datasets/paper_bench.csv --csv_runs_limit 20 --learn
```

⚠️ 結果は `runs_capsule/` に保存されます。完全な評価については [PaperBench ドキュメント](https://github.com/openai/frontier-evals/tree/main/project/paperbench) を参照してください。

**カスタムベンチマーク：**

```sh
uv run main.py --papers datasets/<あなたのベンチマーク名>.csv --csv_runs_limit 20 --learn
```

---

## スマートフォン通知

Pushover 通知で ***Mimosa*** のステータスについてリアルタイムの更新を受信します。

### セットアップ

1. [Pushover](https://pushover.net/) アカウントを作成し、**ユーザーキー**をメモ
2. 「Mimosa」という名前のアプリケーションを作成し、**API トークン**をコピー
3. 環境変数をエクスポート：
   ```bash
   export PUSHOVER_USER="あなたのユーザーキー"
   export PUSHOVER_TOKEN="あなたの API トークン"
   ```
4. Pushover モバイルアプリをインストールしてログイン

---

## テレメトリ設定

Langfuse を使用してリアルタイムのオブザーバビリティダッシュボードで AI エージェントを監視およびデバッグします。

### クイックスタート

1. **Langfuse をローカルにデプロイ：**
   ```bash
   git clone https://github.com/langfuse/langfuse.git
   cd langfuse
   docker compose up -d
   ```

2. **`.env` に追加：**
   ```env
   LANGFUSE_PUBLIC_KEY=あなたの公開キー
   LANGFUSE_PRIVATE_KEY=あなたの秘密キー
   ```

3. **ダッシュボードにアクセス**：***Mimosa-AI*** の実行中に `http://localhost:3000` にアクセス

ダッシュボードはエージェント実行トレース、パフォーマンスメトリクス、エラーデバッグ、トークン/API 使用量を提供します。

> **注意：** テレメトリはオプションですが、デバッグとパフォーマンス最適化に推奨されます。

---

## ライセンス

このリポジトリは Apache License 2.0 の下で公開配布されています。貢献とライセンスの詳細については：
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

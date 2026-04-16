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
    <em>자율 과학 연구를 위한 자기진화 AI 프레임워크</em>
</p>
<p align="center">
  🧬 자기진화 멀티에이전트 워크플로우 &nbsp;·&nbsp;
  🔍 MCP 기반 도구 자동 탐색 &nbsp;·&nbsp;
  🔁 다윈식 워크플로우 최적화 &nbsp;·&nbsp;
  📦 완전한 감사 추적 및 재현성 &nbsp;·&nbsp;
</p>

<p align="center">
    <em>Mimosa-AI는 Nothias et al. (2018)을 엔드투엔드로 자율적으로 재현했습니다——원시 .mzML 파일부터 분자 네트워크까지——단 하나의 명령으로.</em>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2603.28986"><img src="https://img.shields.io/badge/arXiv-2603.28986-b31b1b.svg?logo=arxiv&style=flat-square&logoColor=white" alt="arXiv 프리프린트"></a>
    <a href="https://doi.org/10.48550/arXiv.2603.28986"><img src="https://img.shields.io/badge/DOI-10.48550%2FarXiv.2603.28986-blue?style=flat-square" alt="DOI"></a>
    <a href="https://holobiomicslab.cnrs.fr/"><img src="https://img.shields.io/badge/웹사이트-holobiomicslab.cnrs.fr-4caf82?style=flat-square&logo=globe&logoColor=white" alt="웹사이트"></a>
</p>

<p align="center">
    <a href="https://github.com/HolobiomicsLab/Mimosa-AI/stargazers">
        <img src="https://img.shields.io/github/stars/HolobiomicsLab/Mimosa-AI?style=social" alt="GitHub Stars">
    </a>
    <a href="https://opensource.org/licenses/Apache-2.0">
        <img src="https://img.shields.io/badge/라이선스-Apache%202.0-blue.svg?style=flat-square" alt="라이선스: Apache 2.0">
    </a>
</p>

---

## 라이브 데모: 자율적 논문 재현

https://github.com/user-attachments/assets/dcd04ade-9c43-44a8-b3e3-a999d3dc895d

**결과:** 아래 분자 네트워크는 원시 `.mzML` 파일에서 자율적으로 재현되었으며, [Nothias et al. (2018)](https://www.researchgate.net/publication/323525305_Bioactivity-Based_Molecular_Networking_for_the_Discovery_of_Drug_Leads_in_Natural_Product_Bioassay-Guided_Fractionation)에서 보고된 토폴로지——클러스터 분리 및 엣지 가중치 포함——와 완벽히 일치합니다.

<p align="center">
  <img src="./docs/images/network.png" alt="재현된 분자 네트워크" width="80%">
</p>

---

## 벤치마크 결과

**ScienceAgentBench** 평가 결과（102개 작업, `task` 모드）：

| 모드 | 성공률 | Code-BLEU 점수 | 작업당 비용 |
|------|--------|----------------|------------|
| DeepSeek-V3.2 단일 에이전트 | 38.2% | 0.898 | $0.05 |
| DeepSeek-V3.2 원샷 멀티에이전트 | 32.4% | 0.794 | $0.38 |
| **DeepSeek-V3.2 반복 학습** | **43.1%** | **0.921** | **$1.7** |

> 반복 학습은 GPT-4o를 개선하지만 Claude Haiku 4.5에는 미미한 성능 저하를 가져옵니다——모델 의존 행동 분석은 [논문](https://arxiv.org/abs/2603.28986)을 참조하세요.

---

## Mimosa-AI란 무엇인가요?

> ***Mimosa-AI 🌼*** — 감지하고, 학습하고, 적응하는 미모사 식물처럼, Mimosa는 작업별 맞춤 멀티에이전트 워크플로우를 자동으로 합성하고 실행 피드백을 통해 지속적으로 개선하는 오픈소스 자율 과학 연구 프레임워크입니다. MCP 기반 도구 발견, 코드 생성 에이전트, LLM 기반 평가를 중심으로 구축되어, 학술 연구자들에게 폐쇄형 블랙박스 시스템의 모듈화되고 감사 가능한 대안을 제공합니다.

**핵심 기능:**
- **과학적 연구 재현**——추적 가능성과 엄밀성으로——원시 데이터에서 출판 가능한 그림까지
- **계산 파이프라인 자동화**——바이오인포매틱스, 분자 도킹, 대사체학, ML 등
- **자기 진화**——다윈적 워크플로우 변이를 통해——각 실패가 다음 시도를 개선

### 아키텍처 개요

프레임워크는 5개 계층으로 구성됩니다:

1. **계획 계층** (선택사항) — 고수준 과학 목표를 개별 작업으로 분해
2. **도구 발견 계층** — Toolomics를 통해 로컬 네트워크의 MCP 도구 자동 발견
3. **메타 오케스트레이션 계층** — 작업별 멀티에이전트 워크플로우 합성; 전문 에이전트에 도구 할당
4. **에이전트 실행 계층** — 코드 생성 에이전트가 발견된 도구와 과학 라이브러리를 사용해 서브태스크 실행
5. **판단/평가 계층** — LLM 기반 판사가 출력 채점; 학습 모드에서 반복적 워크플로우 개선 추진

<p align="center">
  <img src="./docs/images/mimosa_overall.jpg" alt="Mimosa 아키텍처 개요" width="90%">
</p>

벤치마크 `task` 모드에서는 계획 계층(1)을 우회하여 워크플로우 합성과 개선을 독립적으로 평가합니다.

---

## 목차

- [Toolomics란 무엇이며, 필요한가요?](#toolomics란-무엇이며-필요한가요)
- [사전 요구사항](#사전-요구사항)
- [설치](#설치)
- [설정](#설정)
- [Mimosa 실행](#mimosa-실행)
- [워크스페이스와 감사 추적](#워크스페이스와-감사-추적)
- [멀티에이전트 워크플로우 진화를 통한 학습](#멀티에이전트-워크플로우-진화를-통한-학습)
- [투명성](#투명성)
- [명령줄 인수](#명령줄-인수)
- [평가](#평가)
- [스마트폰 알림](#스마트폰-알림)
- [텔레메트리 설정](#텔레메트리-설정)
- [라이선스](#라이선스)

---

## Toolomics란 무엇이며, 필요한가요?

**[Toolomics](https://github.com/HolobiomicsLab/toolomics)**는 Mimosa의 동반 플랫폼으로, MCP 서버 관리에 사용됩니다. 과학 도구(데이터 분석 유틸리티, 웹 서비스, 실험실 기기)를 발견 가능한 MCP 서비스로 제공하고, Mimosa가 작업 아티팩트를 읽고 쓰는 공유 워크스페이스를 제공하며, Mimosa 핵심 코드를 수정하지 않고 사용자 정의 도구를 등록할 수 있습니다.

**필요한가요?** 예——Mimosa의 어떤 모드를 실행하기 전에도 Toolomics가 실행 중이어야 합니다. 좋은 소식은: 설정에 몇 분밖에 걸리지 않습니다.

- Mimosa와 Toolomics 모두 Apache 2.0 라이선스로, 무료로 사용할 수 있습니다.
- Toolomics는 설정 가능한 포트 범위(기본값 `5000–5100`)에서 로컬로 실행됩니다.
- [Toolomics 문서](https://github.com/HolobiomicsLab/toolomics)를 통해 사용자 정의 MCP 도구를 추가할 수 있습니다.

> **빠른 시작 경로:** Toolomics 클론 → 기본 포트 범위에서 시작 → Mimosa 실행. LLM API 키 외에 클라우드 계정이나 유료 서비스가 필요 없습니다.

---

## 사전 요구사항

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (권장) 또는 pip
- 실행 중인 [Toolomics MCP 서버](https://github.com/HolobiomicsLab/toolomics)

---

## 설치

### 1. 의존성 설치

```bash
# uv 사용 (권장 — 가상 환경 생성과 의존성 설치를 한 번에 수행)
pip install uv
uv sync

# 또는 pip 사용
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install .
```

### 2. API 키 설정

프로젝트 루트에 `.env` 파일을 생성합니다. 사용할 LLM 프로바이더의 키만 포함하세요:

```env
ANTHROPIC_API_KEY=...       # Claude — 워크플로우 오케스트레이션에 권장
OPENAI_API_KEY=...          # OpenAI 모델 - 선택사항
MISTRAL_API_KEY=...         # Mistral 모델 - 선택사항
DEEPSEEK_API_KEY=...        # Deepseek - 선택사항
HF_TOKEN=...                # HuggingFace 프로바이더, 선택사항
OPENROUTER_API_KEY=...      # OpenRouter를 통해 모든 모델에 접근

# 선택사항 — Langfuse를 통한 관찰 가능성
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_PRIVATE_KEY=...
```

### 3. MCP 서버 시작

[HolobiomicsLab/toolomics](https://github.com/HolobiomicsLab/toolomics)의 설정 지침을 따르세요. 포트 범위(예: `5000–5100`)에서 실행되도록 설정합니다.

사용자 정의 MCP 도구는 [Toolomics 문서](https://github.com/HolobiomicsLab/toolomics/README.md)를 통해 추가할 수 있습니다.

---

## 설정

```bash
cp config_default.json my_config.json
```

`my_config.json`을 편집합니다. 주요 파라미터:

| 파라미터 | 설명 |
|---------|------|
| `workspace_dir` | Toolomics 워크스페이스 경로 — 생성된 모든 파일이 여기에 저장됩니다 |
| `discovery_addresses` | MCP 서버 탐색을 위한 IP + 포트 범위 |
| `planner_llm_model` | 작업 분해 및 계획을 위한 LLM |
| `prompts_llm_model` | 워크플로우 프롬프트 생성을 위한 LLM |
| `workflow_llm_model` | 멀티에이전트 오케스트레이션을 위한 LLM (권장: `anthropic/claude-opus-4-5` 또는 `z-ai/glm-5`) |
| `smolagent_model_id` | SmolAgents 실행 서브태스크를 위한 모델 |
| `judge_model` | 출력 자기평가 및 채점을 위한 LLM |
| `learned_score_threshold` | 결과를 수용하고 반복을 중단하는 최소 점수 |
| `max_learning_evolve_iterations` | 결과를 수용하기 전 최대 자기개선 반복 횟수 |

---

## Mimosa 실행

Mimosa는 **목표(Goal)**와 **작업(Task)** 두 가지 실행 모드를 지원합니다.

### 목표 모드 — 다단계 과학적 목표

여러 개별 작업에 걸친 계획이 필요한 경우 사용합니다(예: 논문 재현, ML 파이프라인 구축).

```bash
uv run main.py --goal "귀하의 과학적 목표" --config my_config.json
```

**예시:**
```bash
uv run main.py \
  --goal "「Dual Aggregation Transformer for Image Super-Resolution」(https://arxiv.org/pdf/2306.00306)의 실험을 재현하고 결과를 비교한다." \
  --config my_config.json

uv run main.py \
  --goal "단백질-리간드 결합 친화도를 예측하는 머신러닝 모델을 개발한다." \
  --config my_config.json
```

### 작업 모드 — 단일 세밀 작업

장기 계획 없이 집중적이고 자기완결적인 작업에 사용합니다.

```bash
uv run main.py --task "귀하의 작업 설명" --config my_config.json
```

**예시:**
```bash
uv run main.py \
  --task "Clintox 데이터셋에서 멀티태스크 모델을 훈련하여 약물 독성과 FDA 승인 상태를 예측한다." \
  --config my_config.json

uv run main.py --task "신약 발견을 위한 그래프 신경망에 관한 문헌 검토를 수행한다." --config my_config.json
```

> **벤치마크 참고:** 논문에서 보고된 결과는 `task` 모드에서 측정되며, 워크플로우 합성과 반복 개선을 독립적으로 평가하기 위해 계획 계층이 비활성화됩니다.
>
> **참고:** 어떤 모드를 실행하기 전에 Toolomics를 설치하고 MCP 서버가 실행 중인지 확인해야 합니다.

---

## 워크스페이스와 감사 추적

실행 중에 Mimosa는 `workspace_dir`로 설정된 Toolomics 워크스페이스 내 파일을 읽고 씁니다. 실행 완료 시 워크스페이스 내용이 타임스탬프가 있는 폴더로 `runs_capsule/`에 복사되어 아카이브로 보존됩니다.

- **Toolomics `workspace/`** — 실시간 작업 디렉토리: 중간 파일, 스크립트, 다운로드, 생성된 출력
- **`sources/workflows/<uuid>/`** — 생성된 워크플로우와 실행 메타데이터: `state_result.json`, `evaluation.txt`, `reward_progress.png`, `memory/` 추적
- **`runs_capsule/<capsule_name>/`** — 실행의 아카이브 스냅샷, 나중에 검사·비교·공유를 위한
- **`memory_explorer.py <uuid>`** — 워크플로우 실행을 단계별로 재생하여 에이전트 추적·도구 호출·출력 검사

이러한 위치들이 합쳐져 Mimosa의 완전한 감사 추적을 형성합니다: 계획·실행·평가·산출의 전 과정.

---

## 멀티에이전트 워크플로우 진화를 통한 학습

***Mimosa-AI***는 과학적 작업을 위한 특화된 워크플로우를 동적으로 합성하는 **자기 진화 멀티에이전트 시스템**입니다. **다윈적 단일 후보 지역 탐색**을 통해 워크플로우를 진화시킵니다: 각 반복에서 성능이 가장 우수한 워크플로우만이 후계자를 생성하며 개선된 것만 유지됩니다. 시간이 지남에 따라 검증된 워크플로우 라이브러리가 구축되어 유사한 미래 작업이 처음부터가 아닌 강력한 기준점에서 시작할 수 있습니다.

새로운 작업의 경우, 완전한 자율성 전에 시스템이 역량을 쌓을 수 있도록 **먼저 학습 모드로 시작하는 것**을 권장합니다.

**학습 모드로 시작**

```bash
uv run main.py --task "Clintox 데이터셋에서 멀티태스크 모델을 훈련하여 약물 독성과 FDA 승인 상태를 예측한다" --learn --config my_config.json
```

<p align="center">
  <img src="./docs/images/workflow_mutation.png" alt="워크플로우 변이 다이어그램" width="80%">
</p>

**진행 상황 시각화:**

***Mimosa-AI***가 학습 단계를 완료하면, 보상 진행 그래프(시도별 성능 향상)가 `sources/workflows/<uuid>/reward_progress.png`에 자동으로 저장됩니다.

<p align="center">
  <img src="./docs/images/evolve_example.png" alt="보상 진행 예시" width="80%">
</p>

---

## 투명성

에이전트 실행을 세밀한 단위로 단계별로 살펴볼 수 있는 대화형 디버거 `memory_explorer.py`가 제공됩니다.

```bash
python memory_explorer.py 20260115_113303_9bb63437
```

이를 통해 전체 실행 추적(생각, 도구 호출, 출력)이 재생되어 각 의사결정이 어떻게 전개되었는지 정확히 검사할 수 있습니다.

---

## 명령줄 인수

### 실행 모드

| 인수 | 설명 |
|-----|------|
| `--goal GOAL` | 고수준 연구 목표, 논문 재현 또는 과학적 질문 지정 (플래너 모드) |
| `--task TASK` | 단일 작업 실행: 문헌 검토, 데이터셋 다운로드, ML 모델 구현 등 |
| `--manual` | MCP를 디버깅하고 ***Mimosa*** 도구를 직접 테스트하는 대화형 CLI 모드 |
| `--papers <CSV 경로>` | 연구 논문과 프롬프트가 포함된 CSV 데이터셋으로 평가 |
| `--science_agent_bench` | ScienceAgentBench로 평가 |

### 기타 파라미터

| 인수 | 설명 |
|-----|------|
| `--learn` | 작업 성능 최적화를 위한 반복 학습 활성화 |
| `--max_evolve_iterations N` | 최대 학습 반복 횟수 |
| `--csv_runs_limit N` | 평가할 CSV 항목 수 제한 |
| `--scenario <시나리오 파일명>` | 채점에 LLM 판정자 대신 특정 시나리오 기반 어설션 사용 |
| `--single_agent` | 단일 에이전트 모드——빠르지만 학습을 통한 개선 불가 |
| `--debug` | 더 자세한 로깅을 위한 디버그 모드 활성화 |

---

## 평가

***Mimosa-AI***는 [ScienceAgentBench](https://arxiv.org/abs/2410.05080) 또는 [PaperBench](https://arxiv.org/pdf/2504.01848)에서 평가할 수 있습니다.

⚠️ 편향 없는 평가를 위해 먼저 `./cleanup.sh`를 실행하여 캐시된 워크플로우 사용을 방지하세요.

### ScienceAgentBench

1. ScienceAgentBench 전체 데이터셋 다운로드:
   [데이터셋 링크](https://buckeyemailosu-my.sharepoint.com/personal/chen_8336_buckeyemail_osu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch%2Fbenchmark%2Ezip&parent=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch&ga=1)
2. 비밀번호 `scienceagentbench`로 압축 해제
3. `benchmark/benchmark/datasets/` → `Mimosa-AI/datasets/scienceagentbench/datasets/`로 복사

**학습 모드로 전체 평가:**
```sh
uv run main.py --science_agent_bench --learn
```

**빠른 평가 (10개 작업, 4회 학습 반복):**
```sh
uv run main.py --science_agent_bench --csv_runs_limit 10 --max_evolve_iterations 4
```

### PaperBench

OpenAI PaperBench는 AI 연구 복제에서 AI 에이전트의 능력을 평가합니다 (*PaperBench: Evaluating AI's Ability to Replicate AI Research*).

```sh
uv run main.py --papers datasets/paper_bench.csv --csv_runs_limit 20 --learn
```

⚠️ 결과는 `runs_capsule/`에 저장됩니다. 완전한 평가는 [PaperBench 문서](https://github.com/openai/frontier-evals/tree/main/project/paperbench)를 참조하세요.

**사용자 정의 벤치마크:**

```sh
uv run main.py --papers datasets/<귀하의 벤치마크 이름>.csv --csv_runs_limit 20 --learn
```

---

## 스마트폰 알림

Pushover 알림을 통해 ***Mimosa*** 상태에 대한 실시간 업데이트를 받습니다.

### 설정

1. [Pushover](https://pushover.net/) 계정 생성 및 **사용자 키** 메모
2. "Mimosa"라는 이름의 애플리케이션 생성——**API 토큰** 복사
3. 환경 변수 내보내기:
   ```bash
   export PUSHOVER_USER="귀하의 사용자 키"
   export PUSHOVER_TOKEN="귀하의 API 토큰"
   ```
4. Pushover 모바일 앱 설치 및 로그인

---

## 텔레메트리 설정

Langfuse를 사용하여 실시간 관찰 가능성 대시보드로 AI 에이전트를 모니터링하고 디버그합니다.

### 빠른 시작

1. **Langfuse를 로컬에 배포:**
   ```bash
   git clone https://github.com/langfuse/langfuse.git
   cd langfuse
   docker compose up -d
   ```

2. **`.env`에 추가:**
   ```env
   LANGFUSE_PUBLIC_KEY=귀하의 공개 키
   LANGFUSE_PRIVATE_KEY=귀하의 비밀 키
   ```

3. **대시보드 접속**: ***Mimosa-AI*** 실행 중에 `http://localhost:3000` 방문

대시보드는 에이전트 실행 추적, 성능 메트릭, 오류 디버깅, 토큰/API 사용량을 제공합니다.

> **참고:** 텔레메트리는 선택사항이지만 디버깅 및 성능 최적화에 권장됩니다.

---

## 라이선스

이 저장소는 Apache License 2.0 하에 공개 배포됩니다. 기여 및 라이선스 세부사항은 다음을 참조하세요:
- `NOTICE`
- `docs/licensing-notes.md`
- `CLA/INDIVIDUAL_CLA.md`
- `CLA/EMPLOYER_AUTHORIZATION.md`

---

## 인용

<p align="center">
<b>인용:</b> <em><a href="https://arxiv.org/abs/2603.28986">Mimosa Framework: Toward Evolving Multi-Agent Systems for Scientific Research</a></em><br>
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

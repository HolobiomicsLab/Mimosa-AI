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
    <em>자율적인 AI 과학자 워크플로우를 진화시키기 위한 오픈소스 프레임워크.</em>
</p>

<p align="center">
    <a href="https://arxiv.org/abs/2603.28986"><img src="https://img.shields.io/badge/arXiv-2603.28986-b31b1b.svg?logo=arxiv&style=flat-square&logoColor=white" alt="arXiv"></a>
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

> ***Mimosa-AI 🌼*** — 감지하고, 학습하고, 적응하는 미모사 식물처럼, Mimosa는 엔드투엔드 연구를 자율적으로 수행하고 발표된 연구 결과를 재현하기 위해 구축된 AI 과학자 프레임워크입니다. ***Mimosa-AI***는 사용 가능한 도구를 자동으로 발견하고, 연구 목표를 구조화된 워크플로우로 분해하며, 멀티에이전트 워크플로우의 진화를 통한 반복적 자기개선으로 멀티에이전트 실행을 구동합니다.

**활용 사례:**
- 엄격하고 감사 가능한 워크플로우로 과학적 연구 재현
- 바이오인포매틱스, 분자 도킹, 대사체학 등 전체 파이프라인 자동화

## 목차

- [작동 원리](#작동-원리)
- [예시: 생물활성 분자 네트워크 논문 재현](#예시-생물활성-분자-네트워크-논문-재현)
- [사전 요구사항](#사전-요구사항)
- [설치](#설치)
- [설정](#설정)
- [Mimosa 실행](#mimosa-실행)
- [출력](#출력)
- [멀티에이전트 워크플로우 진화를 통한 학습](#멀티에이전트-워크플로우-진화를-통한-학습)
- [투명성](#투명성)
- [명령줄 인수](#명령줄-인수)
- [평가](#평가)
- [스마트폰 알림](#스마트폰-알림)
- [텔레메트리 설정](#텔레메트리-설정)
- [라이선스](#라이선스)

## 작동 원리

사용자가 ***Mimosa-AI***에게 연구 목표를 제공합니다.

- ***Mimosa***는 로컬 네트워크 또는 Toolhive를 통해 이용 가능한 MCP 기반 도구(데이터 분석 유틸리티부터 웹 브라우저, 질량 분석계 등 실험실 기기까지)를 자동으로 발견합니다.
- 사용자의 목표와 발견된 도구를 활용하여, ***Mimosa***는 문제를 분해하고 각 작업에 맞는 멀티에이전트 워크플로우를 구축합니다.
- 각 작업은 자율적으로 실행됩니다. 실패는 반복 학습 루프를 통한 자기개선에 활용됩니다.
- ***Mimosa***는 결과, 시각화, 보고서, 로그 및 모든 관련 산출물을 담은 최종 캡슐을 생성합니다.

![schema](./docs/images/mimosa_overall.jpg)

## 예시: 생물활성 분자 네트워크 논문 재현

> **목표** — [Nothias et al. (2018)](https://www.researchgate.net/publication/323525305_Bioactivity-Based_Molecular_Networking_for_the_Discovery_of_Drug_Leads_in_Natural_Product_Bioassay-Guided_Fractionation)을 엔드투엔드로 재현: `.mzML` 파일을 시작점으로（원시 데이터 변환 완료 가정）특징 감지부터 네트워크 시각화까지.

Mimosa-AI에는 논문, 원시 데이터, 그리고 논문에 완전히 명시되지 않은 설정 세부사항을 다루는 도메인 특화 스킬이 제공되었습니다.

**실행 타임랩스**

https://github.com/user-attachments/assets/dcd04ade-9c43-44a8-b3e3-a999d3dc895d

**출력 결과**

![molecular_network](./docs/images/network.png)

결과 분자 네트워크는 클러스터 분리와 엣지 가중치를 포함하여 원본 논문에서 보고된 토폴로지와 일치하며, 자율적으로 재현되었습니다.

---

## 사전 요구사항

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)（권장）또는 pip
- 실행 중인 [Toolomics MCP 서버](https://github.com/HolobiomicsLab/toolomics)

---

## 설치

### 1. 저장소 클론 및 가상 환경 생성

```bash
# uv 사용（권장）
pip install uv
uv venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 또는 pip 사용
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 의존성 설치

```bash
cd mimosa
uv pip install -r requirements.txt
```

### 3. API 키 설정

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

### 4. MCP 서버 시작

[HolobiomicsLab/toolomics](https://github.com/HolobiomicsLab/toolomics)의 설정 지침을 따르세요. 포트 범위(예: `5000–5100`)에서 실행되도록 설정합니다. 커스텀 MCP 도구는 [Toolomics 문서](https://github.com/HolobiomicsLab/toolomics/README.md)를 통해 추가할 수 있습니다.

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
| `workflow_llm_model` | 멀티에이전트 오케스트레이션을 위한 LLM（권장: anthropic/claude-opus-4-5 또는 z-ai/glm-5） |
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

> **참고:** 어떤 모드를 실행하기 전에 Toolomics를 설치하고 MCP 서버가 실행 중인지 확인해야 합니다.

## 출력

실행 중에 파일이 Toolomics의 `workspace/` 디렉토리에 기록됩니다. 완료 시 타임스탬프가 있는 캡슐로 전체 스냅샷이 `runs_capsule/`에 저장됩니다.

---

## 멀티에이전트 워크플로우 진화를 통한 학습

***Mimosa-AI***는 과학적 작업을 위한 특화된 워크플로우를 동적으로 합성하며, **다윈적 영감을 받은 멀티에이전트 워크플로우 진화**를 통해 실패로부터 학습합니다:
작업을 고정된 파이프라인에 강제로 맞추는 대신, 시스템은 각 작업에 맞춤형 멀티에이전트 그래프를 구성하고 단일 후보 지역 탐색으로 개선합니다. 각 반복에서 성능이 가장 우수한 워크플로우만이 후계자를 생성하며 개선된 것만 유지됩니다. 시간이 지남에 따라 시스템은 검증된 워크플로우 라이브러리를 구축하여 유사한 미래 작업이 처음부터가 아닌 강력한 기준점에서 시작할 수 있도록 합니다.

새로운 작업의 경우 완전한 자율성 전에 시스템이 역량을 쌓을 수 있도록 학습 모드로 시작하는 것을 권장합니다.

**학습 모드로 시작**

```bash
uv run main.py --task "Clintox 데이터셋에서 멀티태스크 모델을 훈련하여 약물 독성과 FDA 승인 상태를 예측한다" --learn --config my_config.json
```

![dgm](./docs/images/workflow_mutation.png)

**진행 상황 시각화:**

***Mimosa-AI***가 작업의 학습 단계를 완료하면 시간이 지남에 따라 어떻게 개선되었는지 정확히 시각화할 수 있습니다. 시도별 성능 향상을 보여주는 보상 진행 그래프가 자동으로 저장됩니다.

보상 진행 그래프는 `sources/workflows/<uuid>` 폴더 아래 `reward_progress.png` 파일명으로 저장됩니다.

***예시:***

![dgm](./docs/images/evolve_example.png)

## 투명성

에이전트 실행을 세밀한 단위로 단계별로 살펴볼 수 있는 대화형 디버거 `memory_explorer.py`가 제공됩니다.

워크플로우 `<uuid>`로 시작합니다（예: `memory_explorer.py 20260115_113303_9bb63437`）

이를 통해 전체 실행 추적(생각, 도구 호출 및 출력)이 재생되어 의사결정이 어떻게 전개되었는지 정확히 검사할 수 있습니다.

## 명령줄 인수

### 실행 모드

| 인수 | 설명 |
|-----|------|
| `--goal GOAL` | 고수준 연구 목표, 논문 재현 또는 과학적 질문 지정（플래너 모드） |
| `--task TASK` | 단일 작업 실행: 문헌 검토, 데이터셋 다운로드, 머신러닝 모델 구현 등 |
| `--manual` | MCP를 디버깅하고 ***Mimosa*** 도구를 직접 테스트하는 대화형 CLI 모드 |
| `--papers <CSV 경로>` | 연구 논문과 프롬프트가 포함된 CSV 데이터셋으로 평가 |
| `--science_agent_bench` | ScienceAgentBench로 평가 |

### 기타 파라미터

| 인수 | 설명 |
|-----|------|
| `--learn` | 작업 성능 최적화를 위한 반복 학습 활성화 |
| `--max_evolve_iterations N` | 최대 학습 반복 횟수 |
| `--csv_runs_limit N` | 평가할 CSV 항목 수 제한 |
| `--scenario <시나리오 파일명>` | 채점 실행에 LLM 판정자 대신 특정 시나리오 기반 어설션 사용 |
| `--single_agent` | 단일 에이전트 모드. 빠르지만 학습을 통한 개선 불가 |
| `--debug` | 더 자세한 로깅을 위한 디버그 모드 활성화 |

---

## 평가

***Mimosa-AI***는 [ScienceAgentBench](https://arxiv.org/abs/2410.05080) 또는 [PaperBench](https://arxiv.org/pdf/2504.01848)에서 평가할 수 있습니다.

⚠️ 편향 없는 평가를 위해 먼저 `./cleanup.sh`를 실행하는 것이 좋습니다. 이를 통해 ***Mimosa***가 기존 또는 캐시된 워크플로우를 사용하는 것을 방지합니다.

### ScienceAgentBench

ScienceAgentBench에서 평가하려면:

1. ScienceAgentBench 전체 데이터셋을 다운로드합니다:
[데이터셋 링크](https://buckeyemailosu-my.sharepoint.com/personal/chen_8336_buckeyemail_osu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch%2Fbenchmark%2Ezip&parent=%2Fpersonal%2Fchen%5F8336%5Fbuckeyemail%5Fosu%5Fedu%2FDocuments%2FResearch&ga=1)
2. 비밀번호 `scienceagentbench`로 압축을 해제합니다
3. `benchmark/benchmark/datasets` 폴더의 내용을 `Mimosa-AI/datasets/scienceagentbench/datasets`에 복사합니다

**학습 모드로 ScienceAgentBench 평가**
```sh
uv run main.py --science_agent_bench --learn
```

**10개 작업으로 제한하고 학습 반복을 4회로 제한한 ScienceAgentBench 평가**

```sh
uv run main.py --science_agent_bench --csv_runs_limit 10 --max_evolve_iterations 4
```

### PaperBench

**학습 모드로 OpenAI PaperBench 평가**

OpenAI PaperBench는 논문 `PaperBench: Evaluating AI's Ability to Replicate AI Research`에서 나온 AI 연구를 복제하는 AI 에이전트의 능력을 평가하는 벤치마크입니다.

```sh
uv run main.py --papers datasets/paper_bench.csv --csv_runs_limit 20  --learn
```

⚠️ 이렇게 하면 모든 논문 재현 시도 결과가 `runs_capsule/` 폴더에 저장됩니다. 완전한 평가는 [Paper Bench 문서](https://github.com/openai/frontier-evals/tree/main/project/paperbench)를 참조하세요.

**커스텀 연구 논문 벤치마크로 평가**

1. `paper_bench.csv`와 동일한 형식의 벤치마크 CSV를 `datasets/<귀하의 벤치마크 이름>.csv`에 배치합니다.

2. 벤치마크에서 실행:

```sh
uv run main.py --papers datasets/<귀하의 벤치마크 이름>.csv --csv_runs_limit 20  --learn
```

---

## 스마트폰 알림

Pushover 알림을 통해 ***Mimosa*** 상태에 대한 실시간 업데이트를 받습니다.

### 설정 단계

1. **Pushover 계정 생성**
   - [pushover.net](https://pushover.net/) 방문
   - 등록하고 **사용자 키**를 메모합니다

2. **애플리케이션 생성**
   - Pushover 대시보드에서 "애플리케이션/API 토큰 생성" 클릭
   - "***Mimosa***"로 이름을 지정하고 생성된 **API 토큰**을 복사합니다

3. **환경 변수 설정**
   ```bash
   export PUSHOVER_USER="귀하의 사용자 키"
   export PUSHOVER_TOKEN="귀하의 API 토큰"
   ```

4. **모바일 앱 설치**
   - 기기의 앱 스토어에서 Pushover를 다운로드
   - Pushover 계정으로 로그인

---

## 텔레메트리 설정

Langfuse를 사용하여 실시간 관찰 가능성 대시보드로 AI 에이전트를 모니터링하고 디버그합니다.

### 빠른 시작

1. **Langfuse를 로컬에 배포**
   ```bash
   git clone https://github.com/langfuse/langfuse.git
   cd langfuse
   docker compose up -d
   ```

2. **환경 변수 설정**

   `.env` 파일에 추가:
   ```env
   LANGFUSE_PUBLIC_KEY=귀하의 공개 키
   LANGFUSE_PRIVATE_KEY=귀하의 비밀 키
   ```

3. **대시보드 접속**

   ***Mimosa-AI*** 실행 중에 `http://localhost:3000` 방문

### 사용 가능한 메트릭

텔레메트리 대시보드가 제공하는 것:
- **에이전트 실행 추적**: 단계별 워크플로우 시각화
- **성능 메트릭**: 응답 시간 및 성공률
- **오류 디버깅**: 상세한 장애 분석
- **리소스 사용량**: 토큰 소비 및 API 호출

**대시보드 예시:**
![Langfuse Dashboard](https://langfuse.com/images/cookbook/integration-smolagents/smolagent_example_trace.png)

> **참고:** 텔레메트리는 선택사항이지만 디버깅 및 성능 최적화에 권장됩니다.

---

## 라이선스

이 저장소는 Apache License 2.0 하에 공개 배포됩니다. Apache 2.0은 상업적 사용, 수정 및 재배포를 허용하는 관대한 오픈소스 라이선스이며, 재배포 시 적용 가능한 라이선스, 저작권, 특허, 상표, 귀속 및 NOTICE 고지를 보존하고, 수정된 파일에는 변경이 이루어졌음을 나타내는 눈에 띄는 고지를 첨부해야 합니다.

기여 및 라이선스 세부사항은 다음을 참조하세요:
- `NOTICE`
- `docs/licensing-notes.md`
- `CLA/INDIVIDUAL_CLA.md`
- `CLA/EMPLOYER_AUTHORIZATION.md`

---

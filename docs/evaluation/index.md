# Evaluation

!!! info "Two systems, both called 'evaluation'"
    Mimosa has **two distinct evaluation systems** and this section
    covers only one of them — external benchmark grading.

    | System | Where it runs | What it scores against | What it produces |
    | ------ | ------------- | ---------------------- | ---------------- |
    | **Judge / verifier** ([Evaluation pipeline](../concepts/evaluation-pipeline.md)) | Every workflow execution, inside the evolutionary loop | Deterministic Python programs that recompute claims from the workspace, plus literature grounding for soft claims | A reward and a *prompt gradient* — the **pressure signal** that drives workflow evolution |
    | **External benchmark graders** (this section) | After Mimosa produces a candidate output, against a fixed dataset | **Ground-truth files provided by the benchmark authors** | Benchmark metrics: SR / VER / CBS for ScienceAgentBench, judge grades for PaperBench |

    The two systems are independent. The verifier runs whether or not
    you are on a benchmark; the benchmark grader runs whether or not
    `--learn` was used to refine the workflow.

Mimosa-AI can be evaluated on two external benchmarks or on your own CSV
of tasks.

| Page | Benchmark | Author of the grader |
| ---- | --------- | -------------------- |
| [ScienceAgentBench](../science_agent_bench_evaluation.md) | 102-task scientific computing benchmark — VER / SR / CBS metrics. | ScienceAgentBench authors |
| [PaperBench](../papers_bench_evaluation.md) | OpenAI's AI-research-replication benchmark. | OpenAI frontier-evals |
| [Custom benchmarks](custom-benchmarks.md) | Your own CSV of (task, expected output) rows. | You |

!!! warning "Reset before benchmarking"
    For unbiased evaluation, run `./cleanup.sh` before kickoff. Otherwise
    cached workflows from previous runs can leak through the disk-similarity
    fallback in `WorkflowSelector` and contaminate the result.

## Benchmark snapshot

Headline result on ScienceAgentBench (102 tasks, `task` mode):

| Mode | Success Rate | Code-BLEU | Cost / task |
| ---- | ------------ | --------- | ----------- |
| DeepSeek-V3.2 single-agent | 38.2 % | 0.898 | $0.05 |
| DeepSeek-V3.2 one-shot multi-agent | 32.4 % | 0.794 | $0.38 |
| **DeepSeek-V3.2 iterative-learning** | **43.1 %** | **0.921** | **$1.70** |

Iterative learning improves GPT-4o but yields marginal degradation for
Claude Haiku 4.5 — see the [manuscript](https://arxiv.org/abs/2603.28986)
for the full model-dependent behaviour analysis.

These numbers are produced by the **ScienceAgentBench grader**, not by
the in-loop verifier.

## How batch evaluation works

All three modes share the same engine:

```
CsvEvaluationMode
  ├─ per-row: spawn start_workflow_evolution() as an asyncio task
  ├─ throttle: max_concurrent_eval_tasks (default 1)
  ├─ stagger: task_start_delay seconds between launches
  └─ on completion: capsule_evaluator → metrics
```

`capsule_evaluator` is what calls into the external grader. Each
`start_workflow_evolution()` call still runs the in-loop verifier on
every generation it produces.

Concurrency knobs:

| Field | Default | What it does |
| ----- | ------- | ------------ |
| `max_concurrent_eval_tasks` | `1` | Tasks running in parallel. |
| `task_start_delay` | `30.0` s | Stagger between task launches. |

Bump `max_concurrent_eval_tasks` if you have spare API budget and want to
finish faster; keep it at `1` if you're hitting rate limits.

## See also

- [Evaluation pipeline](../concepts/evaluation-pipeline.md) — the **judge
  / verifier** that drives evolution. Different system, different
  purpose.
- [Iterative learning](../usage/learning.md) — how the verifier and
  evolution loop work together.

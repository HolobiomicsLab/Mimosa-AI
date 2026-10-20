# Quickstart

From a fresh clone to a finished workflow in five minutes.

## 0. Prerequisites

Make sure you've completed [Installation](installation.md): dependencies
installed, `.env` populated, Toolomics MCP server running.

## 1. Interactive onboarding (easiest path)

Run Mimosa with **no arguments** and it launches an interactive wizard that
guides you through workspace selection, model choice, and the first run:

```bash
uv run main.py
```

Your choices are saved to `config_default.json` in a repo checkout, or to
`~/.config/mimosa/config.json` for installed runs. Pass that file with
`--config` on later runs; repo checkouts do not load `config_default.json` implicitly.

## 2. Manual path — one task

If you prefer skipping the wizard, copy the default config and tweak it:

```bash
cp config_default.json my_config.json
```

Open `my_config.json` and set at minimum:

- `workspace_dir` — path to your Toolomics workspace.
- `workflow_llm_model` — LLM that synthesizes multi-agent workflows
  (`anthropic/claude-opus-4-5` is a suggested choice; the shipped default is `openrouter/z-ai/glm-5.3`).
- `smolagent_model_id` — LLM used by execution agents.

Then run a single task:

```bash
uv run main.py \
  --task "Train a multitask model on the Clintox dataset to predict drug toxicity and FDA approval status." \
  --config my_config.json
```

Mimosa will:

1. Discover Toolomics MCP tools on your network.
2. Synthesize a multi-agent workflow as Python code.
3. Execute it in a sandbox against your workspace.
4. Score the result with the default `policy_bt` verifier (Bradley-Terry
   tournament over deterministic policy scorers).
5. Archive the run under `runs_capsule/`.

## 3. Add learning

To let Mimosa retry and improve on its own attempts:

```bash
uv run main.py \
  --task "Train a multitask model on the Clintox dataset…" \
  --learn \
  --config my_config.json
```

In learning mode, Mimosa evolves up to `max_learning_evolve_iterations`
generations or stops as soon as `overall_score >= learned_score_threshold`
(default `0.9`). See [Iterative learning](../usage/learning.md).

## 4. Try goal mode

For multi-step scientific objectives (e.g. reproducing a paper), use
`--goal` — Mimosa runs the planner first, then evolves a workflow per task:

```bash
uv run main.py \
  --goal "Reproduce experiments from 'Dual Aggregation Transformer for Image Super-Resolution' (https://arxiv.org/pdf/2306.00306) and compare results." \
  --config my_config.json
```

## 5. Inspect the result

After the run finishes, the archive is written under `runs_capsule/`
(`runs_capsule_dir`). Benchmark capsules contain `evaluation_results.json`.

You can also open the memory chat over a run's trace:

```bash
uv run main.py --memory_cli                       # latest run
uv run main.py --memory_cli --memory_uuid <uuid>  # specific run
```

## 6. Where to go next

- [Execution modes](../usage/modes.md) — `--task`, `--goal`, `--manual`, batch.
- [Iterative learning](../usage/learning.md) — what's actually happening between generations.
- [Workspace & audit trail](../usage/workspace.md) — where every artefact lives.
- [Run benchmarks](../evaluation/index.md) — ScienceAgentBench, PaperBench, custom CSV.

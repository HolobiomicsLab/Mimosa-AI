# Extending Mimosa

Mimosa-AI is built to be extended. This page lists the common extension
points with the file you'd touch and the interface to respect.

## Add a new MCP tool

**Don't touch Mimosa.** Tools live in
[Toolomics](https://github.com/HolobiomicsLab/toolomics):

1. Write the tool as a Python function with type hints + docstring.
2. Register it as an MCP service in Toolomics.
3. Restart Toolomics — Mimosa picks it up on the next discovery cycle.

See [Tool discovery & MCP](../concepts/tools-and-mcp.md).

## Add an LLM provider

[`LLMProvider`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/llm_provider.py)
wraps LiteLLM, so any provider LiteLLM supports works out of the box —
just use the right `model_id` (e.g. `huggingface/.../...`, `mlx/...`).

For providers LiteLLM doesn't support, extend `LLMProvider` directly. The
public surface is small:

- `complete(prompt, **kwargs) → str` (or async equivalent).
- Cost reporting via the OpenRouter pricing client.

## Add an evaluator backend

The verifier facade is
[`evaluators/evaluator.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/evaluators/evaluator.py).
To add a new scoring strategy:

1. Subclass `BaseEvaluator` in `evaluators/base.py`.
2. Implement `evaluate(state, workflow_code, task) → EvaluationResult`.
3. Register it in `WorkflowEvaluator`'s `evaluator_type` switch.

The returned `EvaluationResult` must include at minimum:

- `overall_score: float ∈ [0, 1]`
- `reward_uncapped: float ∈ [0, 1]` (or `None` for backends without a cap)
- `abstracted_diagnosis: str` — the **only** signal fed back to the mutator.

Anything else you put on the result is fine — it's just persisted to
`state_result.json` for auditing.

## Add a selection strategy

[`SelectionPressure`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/selection.py)
ships four strategies (`greedy`, `tournament`, `novelty`, `qd`). To add a
fifth, add a branch in `select_parents()` and `validate_survivor()`. The
interface is small; copy `qd` as a template.

## Customize the workflow prompt

The system prompt that drives synthesis lives at
`sources/prompts/workflow_v10.md`. To iterate without breaking existing
runs:

1. Copy `workflow_v10.md` to `workflow_v11.md` (or whatever name).
2. Point `Config.prompt_workflow_creator` at the new file (CLI:
   `--prompt_workflow_creator sources/prompts/workflow_v11.md`).
3. Run `--workflow_eval_mode` to measure generation-quality impact.

## Add a planner template

The planner prompts live under `sources/prompts/planner_*.md`. Add a new
template alongside, then override
`Config.prompt_planner` per run.

## Add a scenario rubric

For rubric-based scoring, add a JSON file under `datasets/scenarios/` and
load it with `--scenario datasets/scenarios/<name>.json`. The loader is at
[`sources/evaluation/scenario_loader.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/evaluation/scenario_loader.py).

## Wire a different sandbox

The sandbox interface is
[`workflow_runner.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/workflow_runner.py).
The contract:

- Install pinned deps (`Config.runner_requirements`).
- Run the workflow Python file with the configured limits.
- Return an `ExecutionResult` containing `stdout`, `stderr`, `state`, and
  status flags.

For a Docker-based sandbox you'd swap the subprocess machinery for a
container runner.

## Add a notification channel

Notifications are sent by
[`sources/utils/notify.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/utils/notify.py).
Pushover is wired in; to add Slack or email, model the new client after
the existing `notify_pushover()` call and gate it on env-var presence.

## Where to put tests

Every new module should have tests under `tests/`. Naming convention:
`tests/<module>_test.py`. Run with `uv run pytest tests/`.

## See also

- [Developer guide](../DEVELOPER_GUIDE.md) — full architecture deep dive.
- [Contributing](contributing.md) — PR process and CLA.

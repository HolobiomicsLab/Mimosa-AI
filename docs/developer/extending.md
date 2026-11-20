# Extending Mimosa

Mimosa-AI is built to be extended. This page lists the common extension
points with the file you'd touch and the interface to respect.

## Add a new MCP tool

**Don't touch Mimosa.** Tools live in
[Toolomics](https://github.com/HolobiomicsLab/toolomics):

1. Write the tool as a Python function with type hints + docstring.
2. Register it as an MCP service in Toolomics.
3. Restart Toolomics. Mimosa calls `ToolsManager.discover_mcp_servers()` while it builds a workflow (`sources/core/factory.py:79`), so the server is found on the next run.

See [Tool discovery & MCP](../concepts/tools-and-mcp.md).

## Add an LLM provider

[`LLMProvider`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/llm_provider.py)
wraps LiteLLM (`litellm.completion`), so any provider LiteLLM supports works out of the box. Use the `provider/model` form from `config_default.json`, for example `openrouter/deepseek/deepseek-v4-flash-0731`.

For providers LiteLLM doesn't support, subclass or change `LLMProvider`. The public surface is small:

- `__call__(prompt, timeout=180, use_cache=True) -> str` is the completion entry point (`llm_provider.py:618`).
- Cost is computed from recorded usage in `sources/utils/pricing.py` (`PricingCalculator`, `OpenRouterPricingClient`).

## Add an evaluator backend

The verifier facade is
[`evaluators/evaluator.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/evaluators/evaluator.py).
To add a new scoring strategy:

1. Subclass `BaseEvaluator` in `evaluators/base.py`.
2. Implement `evaluate(uuid)` for verifier backends and persist scores via `_save_results` (`base.py:227`).
3. Register it in `WorkflowEvaluator` (`evaluator.py`): add a `verifier_kind` branch in `__init__` (`evaluator.py:89-118`) for a new verifier, or extend the `evaluator_type` dispatch (`evaluator.py:158-190`) for a new evaluator type.

The returned scores should include:

- `overall_score: float ∈ [0, 1]` — the reward QD ranks on. Under the default `policy_bt` verifier this is the Bradley-Terry rating (`sources/evaluators/policy_bt_verifier/results.py:240`).
- `overall_score_uncapped` — optional. When absent, `WorkflowInfo.overall_score_uncapped` falls back to `overall_score` (`sources/core/workflow_info.py:252-266`). The `policy_bt` output does not set it (`results.py:238-246`).
- `abstracted_textual_gradient: str` — the **only** signal fed back to
  the mutator.

To add an evidence channel to the hybrid verifier (`verifier_kind="hybrid"`, not the default) instead of replacing it, implement the `EvidenceLayer` protocol from
[`hybrid_verifier/layers.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/evaluators/hybrid_verifier/layers.py)
(`collect` / optional `revise` / optional `gate`) and pass
`extra_layers=[...]` to `HybridVerifierEvaluator` — no aggregation,
registry, reward, or gradient code needs to change.

Anything else you put on the result is fine — it's just persisted to
`state_result.json` for auditing.

## Add a selection strategy

[`SelectionPressure`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/selection.py)
ships four strategies (`greedy`, `tournament`, `novelty`, `qd`). To add a fifth, add an enum member to `SelectionStrategy` (`selection.py:38-43`), a branch in `select_parent()` and `validate_survivor()`, and the name in the CLI strategy list (`sources/cli/evaluation_cli.py:697`). Copy `qd` as a template.

## Customize the workflow prompt

The system prompt that drives synthesis is `sources/prompts/workflow_v15.md`, the default for `Config.prompt_workflow_creator` (`config.py:262-263`). To iterate without breaking existing
runs:

1. Copy `workflow_v15.md` to `workflow_v16.md` (or whatever name).
2. Point `Config.prompt_workflow_creator` at the new file (CLI:
   `--prompt_workflow_creator sources/prompts/workflow_v16.md`).
3. Run `--workflow_eval_mode` to measure generation-quality impact.

## Add a planner template

The planner prompts live under `sources/prompts/planner_*.md`. Add a new
template alongside, then set `prompt_planner` in the config (`config_default.json:72`) or in code. No CLI flag sets it.

## Add a scenario rubric

For rubric-based scoring, add `<name>.json` to both `datasets/scenarios/` and `datasets/ScienceAgentBench/scoring_rubrics/` (the two directories hold identical copies today), then run with `--scenario <name>.json`. `ScenarioLoader` joins the value onto its directory, so a path such as `datasets/scenarios/<name>.json` does not resolve. The loader is at
[`sources/benchmark_evaluation/scenario_loader.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/benchmark_evaluation/scenario_loader.py). `main.py` reads from `datasets/ScienceAgentBench/scoring_rubrics/` (`main.py:352-353`); `ScenarioEvaluator` defaults to `datasets/scenarios/` (`evaluator.py:57`).

## Wire a different sandbox

The sandbox interface is
[`workflow_runner.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/core/workflow_runner.py).
The contract:

- Install pinned deps (`Config.runner_requirements`, `WorkflowRunner.install_dependencies`).
- Run the workflow Python file with the configured limits.
- Return an `ExecutionResult` (`workflow_runner.py:72`) with `status`, `return_code`, `stdout`, `stderr`, `execution_time`, and `resource_usage`.

For a Docker-based sandbox you'd swap the subprocess machinery for a
container runner.

## Add a notification channel

Notifications are sent by
[`sources/utils/notify.py`](https://github.com/HolobiomicsLab/Mimosa-AI/blob/main/sources/utils/notify.py).
Pushover is wired in through `PushNotifier` (`send_message(message, title=None, priority=0)`), gated on the `PUSHOVER_TOKEN` and `PUSHOVER_USER` environment variables. To add Slack or email, model a new class on `PushNotifier` and gate it on env-var presence the same way.

## Where to put tests

Every new module should have tests under `tests/`. Naming: both `tests/<module>_test.py` and `tests/test_<name>.py` occur, and pytest collects both. Run with `uv run pytest tests/`.

## See also

- [Developer guide](../DEVELOPER_GUIDE.md) — full architecture deep dive.
- [Contributing](contributing.md) — PR process and CLA.

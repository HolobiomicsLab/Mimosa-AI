"""Subscription agent telemetry must not be billed at fallback API rates."""

import json
from types import SimpleNamespace

from sources.utils.pricing import PricingCalculator


def test_native_task_memory_is_not_double_charged(tmp_path):
    memory = tmp_path / "memory" / "run"
    workflow = tmp_path / "workflows" / "run"
    memory.mkdir(parents=True); workflow.mkdir(parents=True)
    (memory / "task_reader.json").write_text(json.dumps([
        {"model":"codex-cli/gpt-5.6-luna", "token_usage":{"input_tokens":10000,"output_tokens":1000,"total_tokens":11000}}
    ]))
    (memory / "native_completion_1.json").write_text(json.dumps({"completion_metadata":{
        "backend":"codex_cli", "requested_model":"gpt-5.6-luna", "actual_model":None,
        "usage":{"input_tokens":10000,"output_tokens":1000,"total_tokens":11000},
        "usage_kind":"subscription", "cost_usd":None, "cost_kind":"unavailable",
    }}))
    calculator = PricingCalculator(SimpleNamespace(memory_dir=memory.parent, workflow_dir=workflow.parent, model_pricing={}))
    assert calculator.calculate_cost("run") == 0
    assert calculator.last_cost_metadata["has_unknown_unbilled_cli_cost"] is True
    assert len(calculator.last_cost_metadata["cli_completions"]) == 1

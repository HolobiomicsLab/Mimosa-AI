"""The native smolagents adapter keeps Codex transport explicit and auditable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from smolagents import CodeAgent
from smolagents.models import ChatMessage, MessageRole

from sources.core.harness_budget import HarnessBudgetError
from sources.core.harness_model import HarnessCompletionModel, HarnessModelError


def _settings(tmp_path, **changes):
    bridge = tmp_path / "harness_completion.py"
    bridge.write_text("def complete(request): return {}\n")
    values = {
        "bridge_path": str(bridge),
        "bridge_sha256": hashlib.sha256(bridge.read_bytes()).hexdigest(),
        "ledger_path": str(tmp_path / "calls.jsonl"),
        "max_calls": 3,
        "total_timeout_seconds": 60.0,
        "call_timeout_seconds": 20,
        "reasoning_effort": "max",
        "max_observed_tokens": None,
    }
    values.update(changes)
    return values


def _result(text="answer", *, status="completed", usage=True):
    model = "gpt-5.6-sol"
    token_usage = None
    if usage is True:
        token_usage = {
            "input_tokens": 11,
            "output_tokens": 4,
            "total_tokens": 15,
            "cached_input_tokens": 3,
            "cache_creation_input_tokens": None,
        }
    elif isinstance(usage, dict):
        token_usage = usage
    return {
        "protocol_version": 1,
        "status": status,
        "text": text if status == "completed" else "",
        "requested_model": model,
        "actual_model": None,
        "observed_models": [],
        "backend": "codex_cli",
        "auth_mode": "subscription",
        "usage_kind": "chatgpt_subscription",
        "usage": token_usage,
        "cost_usd": None,
        "cost_kind": "unavailable",
        "cli_version": "codex-cli fixture",
        "diagnostic_count": 0,
        "error": None if status == "completed" else "neutral failure",
        "model_identity": {
            "requested": {"model": model, "source": "request"},
            "configured": {"model": model, "source": "explicit_cli_argument"},
            "reported": None,
        },
    }


def _messages():
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="Keep the analysis quantitative."),
        {"role": "user", "content": [{"type": "text", "text": "Estimate the orbit. "},
                                      {"type": "text", "text": "Show units."}]},
        ChatMessage(role=MessageRole.TOOL_CALL, content="Calling tools:\npython_orbit"),
        ChatMessage(role=MessageRole.TOOL_RESPONSE, content="Observation:\nperiod = 12.4 d"),
    ]


def test_generate_uses_explicit_codex_route_and_ordered_text_messages(tmp_path):
    captured = []
    model = HarnessCompletionModel("gpt-5.6-sol", _settings(tmp_path), "orbital-agent")

    def complete(request, bridge_path=None):
        captured.append((request, bridge_path))
        return _result("```py\nvalue = 12.4\n```")

    with patch("sources.core.harness_model.call_completion_bridge", side_effect=complete):
        message = model.generate(_messages())

    request, bridge_path = captured[0]
    assert bridge_path == _settings(tmp_path)["bridge_path"]
    assert request == {
        "protocol_version": 1,
        "backend": "codex_cli",
        "model": "gpt-5.6-sol",
        "messages": [
            {"role": "system", "content": "Keep the analysis quantitative."},
            {"role": "user", "content": "Estimate the orbit. Show units."},
            {"role": "assistant", "content": "Calling tools:\npython_orbit"},
            {"role": "user", "content": "Observation:\nperiod = 12.4 d"},
        ],
        "response_format": "text",
        "auth_mode": "subscription",
        "effort": "max",
        "timeout_seconds": 20,
    }
    assert message.role == MessageRole.ASSISTANT
    assert message.content == "```py\nvalue = 12.4\n```"
    assert message.token_usage.dict() == {
        "input_tokens": 11, "output_tokens": 4, "total_tokens": 15
    }
    assert message.raw["reservation_id"]
    assert message.raw["usage"]["cached_input_tokens"] == 3
    assert model.receipts[0]["model_identity"]["configured"]["model"] == "gpt-5.6-sol"
    assert "text" not in model.receipts[0]
    model.assert_healthy()


@pytest.mark.parametrize(
    "generate_kwargs",
    [
        {"tools_to_call_from": [object()]},
        {"response_format": {"type": "json_schema"}},
        {"temperature": 0},
        {"max_tokens": 10},
        {"logprobs": True},
    ],
)
def test_unsupported_generation_features_fail_before_reservation(tmp_path, generate_kwargs):
    model = HarnessCompletionModel("gpt-5.6-sol", _settings(tmp_path))
    with patch("sources.core.harness_model.call_completion_bridge") as complete:
        with pytest.raises(HarnessModelError, match="unsupported"):
            model.generate(_messages(), **generate_kwargs)

    complete.assert_not_called()
    assert not Path(_settings(tmp_path)["ledger_path"]).exists()


def test_images_and_malformed_roles_fail_before_reservation(tmp_path):
    model = HarnessCompletionModel("gpt-5.6-sol", _settings(tmp_path))
    with pytest.raises(HarnessModelError, match="image"):
        model.generate([{"role": "user", "content": [{"type": "image", "image": b"x"}]}])
    model = HarnessCompletionModel("gpt-5.6-sol", _settings(tmp_path))
    with pytest.raises(HarnessModelError, match="role"):
        model.generate([{"role": "developer", "content": "hidden"}])
    assert not Path(_settings(tmp_path)["ledger_path"]).exists()


def test_bridge_hash_is_rechecked_before_every_reservation(tmp_path):
    configured = _settings(tmp_path)
    model = HarnessCompletionModel("gpt-5.6-sol", configured)
    Path(configured["bridge_path"]).write_text("tampered = True\n")

    with patch("sources.core.harness_model.call_completion_bridge") as complete:
        with pytest.raises(HarnessModelError, match="digest"):
            model.generate(_messages())
    complete.assert_not_called()
    assert not Path(configured["ledger_path"]).exists()


def test_configuration_and_model_identity_are_sealed_at_construction(tmp_path):
    model = HarnessCompletionModel("gpt-5.6-sol", _settings(tmp_path))
    with pytest.raises(TypeError):
        model.settings["reasoning_effort"] = "low"
    with pytest.raises(AttributeError):
        model.settings = {}
    with pytest.raises(AttributeError):
        model.model_id = "gpt-5.6-luna"


def test_a_latched_preflight_failure_blocks_later_dispatch(tmp_path):
    model = HarnessCompletionModel("gpt-5.6-sol", _settings(tmp_path))
    with pytest.raises(HarnessModelError, match="unsupported"):
        model.generate(_messages(), temperature=0)

    with patch("sources.core.harness_model.call_completion_bridge") as complete:
        with pytest.raises(HarnessModelError, match="already failed"):
            model.generate(_messages())
    complete.assert_not_called()
    with pytest.raises(HarnessModelError, match="generation failed"):
        model.assert_healthy()


def test_local_stop_uses_earliest_match_and_discloses_local_trimming(tmp_path):
    model = HarnessCompletionModel("gpt-5.6-sol", _settings(tmp_path))
    with patch(
        "sources.core.harness_model.call_completion_bridge",
        return_value=_result("code Calling tools: later Observation: never"),
    ):
        message = model.generate(
            _messages(), stop_sequences=["Observation:", "Calling tools:", "```<end_code>"]
        )

    assert message.content == "code "
    assert message.raw["stop_applied"] is True
    assert message.raw["stop_sequence"] == "Calling tools:"
    assert message.raw["provider_stop_enforced"] is False


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"input_tokens": 8, "output_tokens": None, "total_tokens": None,
         "cached_input_tokens": None, "cache_creation_input_tokens": None},
    ],
)
def test_token_usage_object_requires_both_counts_but_receipt_keeps_usage(tmp_path, usage):
    model = HarnessCompletionModel("gpt-5.6-sol", _settings(tmp_path))
    with patch("sources.core.harness_model.call_completion_bridge", return_value=_result(usage=usage)):
        message = model.generate(_messages())

    assert message.token_usage is None
    assert message.raw["usage"] == usage


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": 8, "output_tokens": 3, "total_tokens": 99,
         "cached_input_tokens": None, "cache_creation_input_tokens": None},
        {"input_tokens": -1, "output_tokens": 3, "total_tokens": 2,
         "cached_input_tokens": None, "cache_creation_input_tokens": None},
    ],
)
def test_inconsistent_token_telemetry_is_rejected_and_poisoned(tmp_path, usage):
    model = HarnessCompletionModel("gpt-5.6-sol", _settings(tmp_path))
    with patch("sources.core.harness_model.call_completion_bridge", return_value=_result(usage=usage)):
        with pytest.raises(HarnessModelError, match="token usage"):
            model.generate(_messages())
    with pytest.raises(HarnessBudgetError):
        model.assert_healthy()


def test_bridge_failure_is_receipted_and_poisoned_without_retry(tmp_path):
    receipt_dir = tmp_path / "memory"
    model = HarnessCompletionModel(
        "gpt-5.6-sol", _settings(tmp_path), receipt_dir=str(receipt_dir)
    )
    with patch(
        "sources.core.harness_model.call_completion_bridge", return_value=_result(status="timeout")
    ) as complete:
        with pytest.raises(HarnessModelError, match="timeout"):
            model.generate(_messages())

    assert complete.call_count == 1
    assert model.receipts[0]["status"] == "timeout"
    saved = json.loads(next(receipt_dir.glob("native_completion_*.json")).read_text())
    assert saved["completion_metadata"] == model.receipts[0]
    with pytest.raises(HarnessBudgetError, match="timeout"):
        model.assert_healthy()
    with pytest.raises(HarnessModelError, match="already failed"):
        model.generate(_messages())
    assert complete.call_count == 1


def test_unknown_exception_is_sanitized_in_receipts_and_ledger(tmp_path):
    secret = "sk-secret raw prompt or error"
    model = HarnessCompletionModel(
        "gpt-5.6-sol", _settings(tmp_path), receipt_dir=str(tmp_path / "memory")
    )
    messages = [{"role": "user", "content": secret}]
    with patch(
        "sources.core.harness_model.call_completion_bridge",
        side_effect=RuntimeError(secret),
    ):
        with pytest.raises(HarnessModelError, match="unknown"):
            model.generate(messages)

    assert model.receipts[0]["status"] == "unknown"
    serialized = Path(_settings(tmp_path)["ledger_path"]).read_text()
    exported = json.dumps(model.receipts)
    assert secret not in serialized
    assert secret not in exported


def test_assert_healthy_catches_codeagent_final_fallback_after_call_exhaustion(tmp_path):
    model = HarnessCompletionModel(
        "gpt-5.6-sol", _settings(tmp_path, max_calls=1), "bounded-code-agent"
    )
    agent = CodeAgent(tools=[], model=model, max_steps=1, verbosity_level=0)
    with patch(
        "sources.core.harness_model.call_completion_bridge",
        return_value=_result("```py\nintermediate_value = 2\n```"),
    ) as complete:
        agent.run("Compute a bounded intermediate value.")

    assert complete.call_count == 1
    with pytest.raises(HarnessModelError, match="generation failed"):
        model.assert_healthy()


@pytest.mark.parametrize("model_id", ["claude-opus-5", "claude-cli/opus", "openai/gpt-5"])
def test_non_codex_or_provider_routed_model_ids_are_rejected(tmp_path, model_id):
    with pytest.raises(ValueError, match="Codex"):
        HarnessCompletionModel(model_id, _settings(tmp_path))

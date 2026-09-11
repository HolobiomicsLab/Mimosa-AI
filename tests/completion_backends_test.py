"""Offline tests for portable API and harness completion backends."""

from __future__ import annotations

import asyncio
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import Config
from sources.core.completion_backends import (
    CompletionBackendError,
    call_completion_bridge,
)
from sources.core.evolution_engine import EvolutionEngine
from sources.core.llm_provider import LLMConfig, LLMProvider
from sources.core.orchestrator import WorkflowOrchestrator
from sources.core.schema import IndividualRun
from sources.core.single_agent_factory import SingleAgentFactory
from sources.core.workflow_factory import WorkflowFactory
from sources.utils.precheck import PreCheck
from sources.utils.pricing import PricingCalculator


SCIENCE_PROMPTS = [
    "Summarize this proteomics observation.",
    "Explain this orbital residual.",
    "Describe this climate anomaly.",
    "Interpret this electrophysiology trace.",
]


class FakeResponse:
    """Minimal LiteLLM response used by the API-route test."""

    def __init__(self, text: str = "ok") -> None:
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")
        ]
        self.usage = None
        self.model = "glm-5.3"

    def json(self) -> dict:
        return {}


def write_fake_bridge(path) -> None:
    """Write an importable bridge that echoes its request in a valid result."""
    path.write_text(
        "import json\n"
        "def complete(request):\n"
        "    return {\n"
        "        'protocol_version': 1, 'status': 'completed',\n"
        "        'text': json.dumps(request, sort_keys=True),\n"
        "        'requested_model': request['model'], 'actual_model': request['model'],\n"
        "        'backend': request['backend'], 'auth_mode': request['auth_mode'],\n"
        "        'usage_kind': 'subscription', 'usage': None, 'cost_usd': None,\n"
        "        'cost_kind': 'unknown', 'cli_version': 'fake-1',\n"
        "        'diagnostic_count': 0, 'error': None,\n"
        "    }\n"
    )


def valid_bridge_result(**changes) -> dict:
    """Return one valid completed Claude bridge envelope."""
    result = {
        "protocol_version": 1,
        "status": "completed",
        "text": "ok",
        "requested_model": "claude-opus-5",
        "actual_model": "claude-opus-5",
        "backend": "claude_cli",
        "auth_mode": "subscription",
        "usage_kind": "claude_subscription",
        "usage": None,
        "cost_usd": None,
        "cost_kind": "unavailable",
        "cli_version": "fake-1",
        "diagnostic_count": 0,
        "error": None,
    }
    result.update(changes)
    return result


@pytest.mark.parametrize("prompt", SCIENCE_PROMPTS)
def test_fake_bridge_integration_is_text_only_and_cacheless(tmp_path, monkeypatch, prompt):
    """Four unrelated sciences cross the same exact in-process bridge seam."""
    bridge = tmp_path / "bridge.py"
    write_fake_bridge(bridge)
    monkeypatch.setenv("HARNESS_COMPLETION_BRIDGE", str(bridge))
    provider = LLMProvider(
        agent_name="science",
        system_msg="Answer as plain text.",
        config=LLMConfig(
            provider="codex-cli",
            model="gpt-6-astra",
            reasoning_effort="max",
            temperature=0.4,
            max_tokens=1234,
        ),
    )
    with patch.object(provider, "_find_cache_match", side_effect=AssertionError("cache used")), patch(
        "sources.core.llm_provider.litellm.completion",
        side_effect=AssertionError("API fallback used"),
    ):
        echoed = json.loads(provider(prompt))
    assert echoed == {
        "protocol_version": 1,
        "backend": "codex_cli",
        "model": "gpt-6-astra",
        "messages": [
            {"role": "system", "content": "Answer as plain text."},
            {"role": "user", "content": prompt},
        ],
        "response_format": "text",
        "auth_mode": "subscription",
        "effort": "max",
        "timeout_seconds": 180,
    }
    assert provider.last_completion_metadata["unsupported_controls"] == {
        "temperature": 0.4,
        "max_tokens": 1234,
    }


@pytest.mark.parametrize("status", ["failed", "timeout", "malformed", "unsupported"])
def test_bridge_failure_is_not_retried_or_sent_to_litellm(status):
    """A failed bridge result raises after exactly one completion attempt."""
    result = {
        "protocol_version": 1,
        "status": status,
        "text": "",
        "requested_model": "claude-opus-5",
        "actual_model": None,
        "backend": "claude_cli",
        "auth_mode": "subscription",
        "usage_kind": "unknown",
        "usage": None,
        "cost_usd": None,
        "cost_kind": "unknown",
        "cli_version": "fake-1",
        "diagnostic_count": 0,
        "error": "neutral failure",
    }
    provider = LLMProvider(
        config=LLMConfig(provider="claude-cli", model="claude-opus-5")
    )
    with patch(
        "sources.core.llm_provider.call_completion_bridge", return_value=result
    ) as bridge, patch(
        "sources.core.llm_provider.litellm.completion",
        side_effect=AssertionError("API fallback used"),
    ):
        with pytest.raises(CompletionBackendError, match="neutral failure"):
            provider("hello")
    bridge.assert_called_once()


@pytest.mark.parametrize(
    "change",
    [
        {"backend": "codex_cli"},
        {"requested_model": "fallback-model"},
        {"auth_mode": "api_key"},
        {"text": "   "},
    ],
)
def test_completed_bridge_result_must_match_requested_route(change):
    """A silent route change or empty completion cannot pass as success."""
    result = {
        "protocol_version": 1,
        "status": "completed",
        "text": "ok",
        "requested_model": "claude-opus-5",
        "actual_model": "claude-opus-5",
        "backend": "claude_cli",
        "auth_mode": "subscription",
        "usage_kind": "subscription",
        "usage": None,
        "cost_usd": None,
        "cost_kind": "unknown",
        "cli_version": "fake-1",
        "diagnostic_count": 0,
        "error": None,
    }
    result.update(change)
    provider = LLMProvider(
        config=LLMConfig(provider="claude-cli", model="claude-opus-5")
    )
    with patch("sources.core.llm_provider.call_completion_bridge", return_value=result):
        with pytest.raises(CompletionBackendError):
            provider("hello")
    assert provider.last_completion_metadata["status"] == "malformed"


def test_bridge_envelope_accepts_additive_provenance():
    """Required protocol fields coexist with future provenance additions."""
    request = {
        "backend": "claude_cli",
        "model": "claude-opus-5",
        "auth_mode": "subscription",
    }
    result = valid_bridge_result(trace_id="local-trace")
    bridge = SimpleNamespace(complete=lambda supplied: result)
    with patch(
        "sources.core.completion_backends.load_completion_bridge",
        return_value=bridge,
    ):
        assert call_completion_bridge(request) is result


@pytest.mark.parametrize(
    "change",
    [
        {"backend": "codex_cli"},
        {"text": " "},
        {"actual_model": ""},
        {"cli_version": None},
        {"diagnostic_count": -1},
        {"usage": {"input_tokens": -1}},
        {"cost_usd": float("nan")},
    ],
)
def test_bridge_rejects_malformed_completed_provenance(change):
    """Malformed success provenance fails before the client records success."""
    result = valid_bridge_result(**change)
    bridge = SimpleNamespace(complete=lambda supplied: result)
    provider = LLMProvider(
        config=LLMConfig(provider="claude-cli", model="claude-opus-5")
    )
    with patch(
        "sources.core.completion_backends.load_completion_bridge",
        return_value=bridge,
    ):
        with pytest.raises(CompletionBackendError):
            provider("hello")
    assert provider.last_completion_metadata["status"] == "malformed"


def test_openai_compatible_endpoint_uses_only_named_key(monkeypatch):
    """The direct API route forwards its HTTPS base and exact named key."""
    monkeypatch.setenv("ZAI_API_KEY_TEST", "named-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    config = LLMConfig(
        provider="openai",
        model="glm-5.3",
        api_base="https://api.z.ai/api/paas/v4/",
        api_key_env="ZAI_API_KEY_TEST",
    )
    provider = LLMProvider(config=config)
    with patch(
        "sources.core.llm_provider.litellm.completion", return_value=FakeResponse()
    ) as completion:
        assert provider("hello", use_cache=False) == "ok"
    assert completion.call_args.kwargs["api_base"] == config.api_base
    assert completion.call_args.kwargs["api_key"] == "named-secret"


def test_existing_api_defaults_do_not_add_custom_endpoint():
    """Absent endpoint settings preserve the existing LiteLLM request shape."""
    config = LLMConfig(provider="openai", model="gpt-5", key="explicit")
    assert config.api_base is None
    assert config.api_key_env is None
    assert config.harness_auth_mode == "subscription"
    with patch(
        "sources.core.llm_provider.litellm.completion", return_value=FakeResponse()
    ) as completion:
        assert LLMProvider(config=config)("hello", use_cache=False) == "ok"
    assert "api_base" not in completion.call_args.kwargs


def test_claude_api_bridge_receives_names_not_secret_values(tmp_path, monkeypatch):
    """Claude API mode sends the endpoint and key variable name only."""
    bridge = tmp_path / "bridge.py"
    write_fake_bridge(bridge)
    monkeypatch.setenv("HARNESS_COMPLETION_BRIDGE", str(bridge))
    monkeypatch.setenv("CLAUDE_ZAI_TEST_KEY", "never-in-request")
    provider = LLMProvider(
        config=LLMConfig(
            provider="claude-cli",
            model="glm-5.3",
            harness_auth_mode="api_key",
            api_base="https://api.z.ai/api/anthropic",
            api_key_env="CLAUDE_ZAI_TEST_KEY",
        )
    )
    request = json.loads(provider("hello"))
    assert request["api_base"] == "https://api.z.ai/api/anthropic"
    assert request["api_key_env"] == "CLAUDE_ZAI_TEST_KEY"
    assert "never-in-request" not in json.dumps(request)


def test_missing_named_key_refuses_ambient_fallback(monkeypatch):
    """Naming an absent key environment never falls back to OPENAI_API_KEY."""
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    monkeypatch.delenv("MISSING_ZAI_KEY", raising=False)
    with pytest.raises(ValueError, match="MISSING_ZAI_KEY"):
        LLMConfig(
            provider="openai",
            model="glm-5.3",
            api_base="https://api.z.ai/api/paas/v4/",
            api_key_env="MISSING_ZAI_KEY",
        )


def test_custom_endpoint_cannot_receive_an_ambient_provider_key(monkeypatch):
    """A custom service requires explicit credential binding before dispatch."""
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-secret")
    with patch(
        "sources.core.llm_provider.litellm.completion",
        side_effect=AssertionError("provider dispatched"),
    ) as completion:
        with pytest.raises(ValueError, match="api_key_env|explicitly supplied"):
            LLMConfig(
                provider="openai",
                model="glm-5.3",
                api_base="https://api.z.ai/api/paas/v4/",
            )
    completion.assert_not_called()

    config = Config()
    config.api_base = "https://api.z.ai/api/paas/v4/"
    with pytest.raises(ValueError, match="api_key_env"):
        config.validate_completion_backend_config()


@pytest.mark.parametrize(
    "api_base",
    [
        "http://api.example.test/v1",
        "https://user:pass@api.example.test/v1",
        "https://api.example.test/v1?secret=value",
        "https://api.example.test/v1#fragment",
    ],
)
def test_endpoint_requires_plain_https_without_userinfo(api_base):
    """Unsafe endpoint forms fail before any completion call."""
    with pytest.raises(ValueError, match="api_base"):
        LLMConfig(provider="openai", model="model", api_base=api_base)
    config = Config()
    config.api_base = api_base
    with pytest.raises(ValueError, match="api_base"):
        config.validate_completion_backend_config()


@pytest.mark.parametrize("use_flat_cache", [False, True])
def test_cli_record_cannot_be_replayed_by_api_cache(
    tmp_path, monkeypatch, use_flat_cache
):
    """CLI provenance makes matching records ineligible in both cache layouts."""
    bridge = tmp_path / "bridge.py"
    write_fake_bridge(bridge)
    monkeypatch.setenv("HARNESS_COMPLETION_BRIDGE", str(bridge))
    memory_path = tmp_path if use_flat_cache else tmp_path / "cli-run"
    memory_path.mkdir(exist_ok=True)
    cli = LLMProvider(
        agent_name="shared-agent",
        memory_path=str(memory_path),
        system_msg="same system",
        use_flat_cache=use_flat_cache,
        config=LLMConfig(provider="codex-cli", model="gpt-6-astra"),
    )
    cli("same prompt")
    saved = json.loads((memory_path / "shared-agent.json").read_text())
    assert saved["cache_eligible"] is False

    api = LLMProvider(
        agent_name="shared-agent",
        memory_path=str(memory_path),
        system_msg="same system",
        use_flat_cache=use_flat_cache,
        config=LLMConfig(provider="openai", model="gpt-5", key="explicit"),
    )
    assert api._find_cache_match("same prompt") is None


def test_custom_api_endpoint_record_is_cache_ineligible(tmp_path, monkeypatch):
    """Opt-in endpoint calls bypass a cache whose identity omits routing."""
    monkeypatch.setenv("ZAI_API_KEY_TEST", "named-secret")
    provider = LLMProvider(
        agent_name="direct-route",
        memory_path=str(tmp_path),
        use_flat_cache=True,
        config=LLMConfig(
            provider="openai",
            model="glm-5.3",
            api_base="https://api.z.ai/api/paas/v4/",
            api_key_env="ZAI_API_KEY_TEST",
        ),
    )
    with patch.object(
        provider,
        "_find_cache_match",
        side_effect=AssertionError("custom route read the cache"),
    ), patch(
        "sources.core.llm_provider.litellm.completion",
        return_value=FakeResponse(),
    ):
        assert provider("hello") == "ok"
    saved = json.loads((tmp_path / "direct-route.json").read_text())
    assert saved["cache_eligible"] is False


def test_cli_only_effort_levels_do_not_change_api_defaults():
    """xhigh/max are accepted only for explicit CLI providers."""
    assert LLMConfig(provider="codex-cli", model="gpt-6-astra", reasoning_effort="xhigh")
    assert LLMConfig(provider="claude-cli", model="claude-opus-5", reasoning_effort="max")
    with pytest.raises(ValueError, match="reasoning_effort"):
        LLMConfig(provider="openai", model="gpt-5", reasoning_effort="max")
    with pytest.raises(ValueError, match="reasoning_effort"):
        LLMConfig(
            provider="codex-cli",
            model="gpt-6-astra",
            reasoning_effort="minimal",
        )


def test_pricing_marks_unknown_cli_cost_without_api_pricing(tmp_path, capsys):
    """Portable call records never enter model-price lookup or appear free."""
    run_id = "portable-run"
    memory_path = tmp_path / "memory" / run_id
    workflow_path = tmp_path / "workflow" / run_id
    memory_path.mkdir(parents=True)
    workflow_path.mkdir(parents=True)
    result = valid_bridge_result(
        backend="codex_cli",
        requested_model="gpt-6-astra",
        actual_model="gpt-6-astra",
        usage_kind="chatgpt_subscription",
        usage={
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
        cost_usd=None,
        cost_kind="unavailable",
    )
    (memory_path / "workflow_creator.json").write_text(
        json.dumps(
            {
                "response": "text",
                "message": [{"role": "user", "content": "prompt"}],
                "completion_metadata": result,
                "cache_eligible": False,
            }
        )
    )
    estimated = valid_bridge_result(
        cost_usd=0.012,
        cost_kind="client_estimate",
        usage_kind="claude_subscription",
    )
    (memory_path / "judge.json").write_text(
        json.dumps(
            {
                "response": "text",
                "message": [{"role": "user", "content": "judge"}],
                "completion_metadata": estimated,
                "cache_eligible": False,
            }
        )
    )
    config = SimpleNamespace(
        memory_dir=str(tmp_path / "memory"),
        workflow_dir=str(tmp_path / "workflow"),
        model_pricing={},
    )
    calculator = PricingCalculator(config)
    assert calculator.calculate_cost(run_id) == pytest.approx(0.012)
    assert calculator.last_cost_metadata["has_unknown_unbilled_cli_cost"] is True
    assert {
        record["cost_kind"]
        for record in calculator.last_cost_metadata["cli_completions"]
    } == {"unavailable", "client_estimate"}
    output = capsys.readouterr().out
    assert "unknown/unbilled" in output
    assert "total is incomplete" in output


def test_pricing_retains_unknown_cli_flag_when_workflow_is_missing(
    tmp_path, capsys
):
    """A failed run still reports an earlier CLI call as unknown/unbilled."""
    run_id = "failed-before-workflow"
    memory_path = tmp_path / "memory" / run_id
    memory_path.mkdir(parents=True)
    (memory_path / "workflow_creator.json").write_text(
        json.dumps(
            {
                "completion_metadata": valid_bridge_result(
                    cost_usd=None,
                    cost_kind="unavailable",
                )
            }
        )
    )
    calculator = PricingCalculator(
        SimpleNamespace(
            memory_dir=str(tmp_path / "memory"),
            workflow_dir=str(tmp_path / "workflow"),
            model_pricing={},
        )
    )
    assert calculator.calculate_cost(run_id) == 0.0
    assert calculator.last_cost_metadata["has_unknown_unbilled_cli_cost"] is True
    assert "unknown/unbilled" in capsys.readouterr().out


def test_malformed_workflow_keeps_uuid_for_cli_cost_accounting(tmp_path):
    """A paid generation record stays reachable when output has no valid code."""
    workflow_root = tmp_path / "workflow"
    memory_root = tmp_path / "memory"
    schema_path = tmp_path / "schema.py"
    factory_path = tmp_path / "factory.py"
    schema_path.write_text("class WorkflowState: pass\n")
    factory_path.write_text("class SmolAgentFactory: pass\n")
    config = SimpleNamespace(
        workflow_dir=str(workflow_root),
        memory_dir=str(memory_root),
        schema_code_path=str(schema_path),
        smolagent_factory_code_path=str(factory_path),
        prompt_workflow_creator=str(tmp_path / "prompt.md"),
    )
    factory = WorkflowFactory(config)
    factory.load_tools_code = AsyncMock(return_value=("", ""))

    def malformed_output(*args):
        memory_path = Path(args[2])
        (memory_path / "workflow_creator.json").write_text(
            json.dumps(
                {
                    "completion_metadata": valid_bridge_result(
                        backend="codex_cli",
                        requested_model="gpt-6-astra",
                        actual_model="gpt-6-astra",
                        usage_kind="chatgpt_subscription",
                    )
                }
            )
        )
        raise ValueError("LLM did not return valid workflow code")

    factory.create_workflow_genotype_code = malformed_output
    with pytest.raises(ValueError, match=r"^UUID:") as raised:
        asyncio.run(factory.craft_workflow("goal", "instructions"))
    failed_uuid, error = WorkflowOrchestrator._parse_generation_error(raised.value)
    assert failed_uuid != "generation_failed"
    assert "valid workflow code" in error

    calculator = PricingCalculator(
        SimpleNamespace(
            memory_dir=str(memory_root),
            workflow_dir=str(workflow_root),
            model_pricing={},
        )
    )
    assert calculator.calculate_cost(failed_uuid) == 0.0
    assert calculator.last_cost_metadata["has_unknown_unbilled_cli_cost"] is True


def test_run_metrics_and_summary_label_incomplete_cli_cost():
    """The main run artifacts and notification qualify the known subtotal."""
    engine = object.__new__(EvolutionEngine)
    engine.variation = SimpleNamespace(last_variation_state={})
    engine.notifier = SimpleNamespace(send_message=MagicMock())
    engine.extract_agents_behavior = lambda state: "answer"
    run = IndividualRun(
        goal="goal",
        prompt="prompt",
        cost=0.012,
        cost_incomplete=True,
        unknown_cli_completion_count=2,
    )
    wf_info = SimpleNamespace(overall_score=0.5, overall_score_uncapped=0.5)
    context = {
        "verdict": None,
        "iteration_cost": 0.012,
        "iteration_cost_incomplete": True,
        "iteration_unknown_cli_completion_count": 1,
        "iteration_start_time": 0.0,
        "on_error": False,
    }
    with patch(
        "sources.core.evolution_engine.grounding_stats", return_value={}
    ):
        metrics = engine._build_run_metrics_snapshot(run, wf_info, context)
    assert metrics["iteration_cost_incomplete"] is True
    assert metrics["cumulative_cost_incomplete"] is True
    assert metrics["iteration_unknown_cli_completion_count"] == 1
    assert metrics["cumulative_unknown_cli_completion_count"] == 2

    with patch("sources.core.evolution_engine.print_summary") as summary:
        engine._log_iteration_completion(
            0,
            1,
            0.0,
            0.5,
            0.012,
            True,
            1,
            "goal",
            "uuid",
            {},
            [0.5],
        )
    rows = summary.call_args.args[1]
    assert ("Known Cost", "$0.012000") in rows
    assert any(label == "CLI Cost" and "unknown/unbilled" in value for label, value in rows)
    notification = engine.notifier.send_message.call_args.args[0]
    assert "Known Cost" in notification
    assert "unknown/unbilled" in notification


def test_completion_fields_round_trip_and_bind_only_openai_service():
    """Nonsecret endpoint settings persist without leaking onto other providers."""
    original = Config()
    original.api_base = "https://api.z.ai/api/paas/v4/"
    original.api_key_env = "ZAI_API_KEY"
    original.harness_auth_mode = "api_key"
    restored = Config()
    restored.from_json(original.jsonify())
    assert restored.api_base == original.api_base
    assert restored.api_key_env == "ZAI_API_KEY"
    assert restored.harness_auth_mode == "api_key"
    assert restored.completion_endpoint_for("openai/glm-5.3") == (
        original.api_base,
        "ZAI_API_KEY",
    )
    assert restored.completion_endpoint_for("openrouter/z-ai/glm-5.3") == (
        None,
        None,
    )


def test_subscription_cli_plus_local_tool_model_needs_no_api_key(
    tmp_path, monkeypatch
):
    """The global environment gate accepts the API-free supported topology."""
    from main import validate_environment

    for name in (
        "ANTHROPIC_API_KEY",
        "MISTRAL_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "HF_TOKEN",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    bridge = tmp_path / "bridge.py"
    bridge.write_text("def complete(request): return {}\n")
    monkeypatch.setenv("HARNESS_COMPLETION_BRIDGE", str(bridge))
    config = Config()
    config.planner_llm_model = "codex-cli/gpt-6-astra"
    config.workflow_llm_model = "claude-cli/claude-opus-5"
    config.judge_model = "codex-cli/gpt-6-astra"
    config.judge_extraction_model = "claude-cli/claude-sonnet-5"
    config.capsule_namer_model = "claude-cli/claude-opus-5"
    config.smolagent_model_id = "mlx-community/local-model"
    validate_environment(config)


def test_environment_rejects_cli_tool_before_bridge_or_key_checks(monkeypatch):
    """Unsupported tool routing remains the earliest actionable error."""
    from main import validate_environment

    monkeypatch.delenv("HARNESS_COMPLETION_BRIDGE", raising=False)
    config = Config()
    config.smolagent_model_id = "codex-cli/gpt-6-astra"
    with pytest.raises(ValueError, match="cannot power ToolSmolAgent"):
        validate_environment(config)


def test_precheck_marks_cli_text_roles_metadata_only_without_api_probe():
    """Stock precheck records CLI text roles without sending paid probes."""
    config = Config()
    config.planner_llm_model = "codex-cli/gpt-6-astra"
    config.workflow_llm_model = "claude-cli/claude-opus-5"
    config.judge_model = "codex-cli/gpt-6-astra"
    config.judge_extraction_model = "claude-cli/claude-sonnet-5"
    config.capsule_namer_model = "claude-cli/claude-opus-5"
    config.smolagent_model_id = "mlx-community/local-model"
    precheck = PreCheck(config)
    with patch.object(precheck, "_basic_check", side_effect=AssertionError("API probed")):
        precheck.run(check_provider=False)
    assert precheck.model_check_metadata["planner"]["status"] == "not_tested"
    assert precheck.model_check_metadata["workflow"]["method"] == "metadata_only"
    assert precheck.model_check_metadata["judge_extraction"]["status"] == "not_tested"


def test_precheck_rejects_any_cli_tool_candidate_before_probes():
    """A CLI model anywhere in the tool-model list fails before API checks."""
    config = Config()
    config.smolagent_model_id = [
        "openrouter/z-ai/glm-5.3",
        "codex-cli/gpt-6-astra",
    ]
    precheck = PreCheck(config)
    with patch.object(precheck, "_basic_check") as basic_check:
        with pytest.raises(ValueError, match="cannot power ToolSmolAgent"):
            precheck.run(check_provider=False)
    basic_check.assert_not_called()


def test_single_agent_rejects_cli_before_loading_tools():
    """Single-agent construction refuses CLI tool engines at its first seam."""
    config = Config()
    config.smolagent_model_id = "claude-cli/claude-opus-5"
    factory = SingleAgentFactory(config)
    factory.load_tools_code = AsyncMock(side_effect=AssertionError("tools loaded"))
    with pytest.raises(ValueError, match="cannot power ToolSmolAgent"):
        asyncio.run(factory.craft_single_agent("test"))
    factory.load_tools_code.assert_not_awaited()


def test_smolagent_engine_rejects_cli_before_engine_dispatch():
    """The generated SmolAgent factory repeats the fail-fast tool guard."""
    source = (Path(__file__).parents[1] / "sources/modules/smolagent_factory.py").read_text()
    tree = ast.parse(source)
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SmolAgentFactory"
    )
    get_engine = next(
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_engine"
    )
    test_class = ast.ClassDef(
        name="IsolatedFactory",
        bases=[],
        keywords=[],
        body=[get_engine],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))
    namespace = {}
    exec(compile(module, "<isolated-smolagent-get-engine>", "exec"), namespace)
    factory = object.__new__(namespace["IsolatedFactory"])
    factory.model_id = "codex-cli/gpt-6-astra"
    with pytest.raises(ValueError, match="cannot power ToolSmolAgent"):
        factory.get_engine()

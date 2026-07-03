"""Tests for token-logprobs capture in the single-agent factory.

``SingleAgentFactory`` emits an inline agent script instead of loading
``smolagent_factory``, so its logprobs wiring lives in generated code.
These tests craft that script with the I/O-heavy factory methods stubbed,
exec its setup portion (everything before ``# Run agent``), and exercise
the resulting engine, ``extract_logprobs`` and ``save_agent_memories``
the same way ``smolagent_factory_logprobs_test.py`` does for the
workflow path.
"""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from smolagents.memory import ActionStep, Timing
from smolagents.models import ChatMessage

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sources.core.single_agent_factory import SingleAgentFactory  # noqa: E402


class FakeLogprobs:
    """Mimics litellm's ChoiceLogprobs pydantic object."""

    def model_dump(self):
        return {
            "content": [
                {
                    "token": "hi",
                    "bytes": [104, 105],
                    "logprob": -0.1,
                    "top_logprobs": [{"token": "hi", "bytes": [104, 105], "logprob": -0.1}],
                }
            ]
        }


EXPECTED_LOGPROBS = {
    "content": [
        {
            "token": "hi",
            "logprob": -0.1,
            "top_logprobs": [{"token": "hi", "logprob": -0.1}],
        }
    ]
}


def craft_setup_namespace(tmp_path, save_logprobs):
    """Generate the single-agent script and exec everything before the run.

    Stubs keep crafting offline: no MCP tools, no prompt file, and
    workflow/memory folders under ``tmp_path``. Returns the exec'd module
    namespace holding ``engine``, ``extract_logprobs`` and
    ``save_agent_memories``.
    """
    config = SimpleNamespace(
        workflow_dir=str(tmp_path / "workflows"),
        memory_dir=str(tmp_path / "memory"),
        smolagent_model_id="openrouter/test-model",
        max_tokens=64,
        engine_name="litellm",
        save_logprobs=save_logprobs,
        openrouter_provider_for=lambda model_id: None,
    )
    factory = SingleAgentFactory(config)

    async def no_tools():
        return "", ""

    async def stub_prompt():
        return "You are a test agent."

    factory.load_tools_code = no_tools
    factory.load_single_agent_system_prompt = stub_prompt
    factory.create_folder_structure = lambda uuid_str: (
        str(tmp_path / "workflows"),
        str(tmp_path / "memory"),
    )
    factory.save_workflow_files = lambda *args, **kwargs: None

    code, _, _ = asyncio.run(factory.craft_single_agent("test goal"))
    setup_code = code.split("# Run agent")[0]
    namespace = {}
    exec(compile(setup_code, "<single_agent_generated>", "exec"), namespace)
    return namespace


@pytest.fixture(scope="module")
def enabled_namespace(tmp_path_factory):
    return craft_setup_namespace(tmp_path_factory.mktemp("enabled"), save_logprobs=True)


@pytest.fixture(scope="module")
def disabled_namespace(tmp_path_factory):
    return craft_setup_namespace(tmp_path_factory.mktemp("disabled"), save_logprobs=False)


def make_action_step(raw) -> ActionStep:
    step = ActionStep(step_number=1, timing=Timing(start_time=0.0, end_time=1.0))
    step.model_output_message = ChatMessage(role="assistant", content="hi", raw=raw)
    return step


def test_generated_engine_requests_logprobs_when_enabled(enabled_namespace):
    assert enabled_namespace["SAVE_LOGPROBS"] is True
    engine = enabled_namespace["engine"]
    assert engine.kwargs["logprobs"] is True
    assert engine.kwargs["top_logprobs"] == enabled_namespace["TOP_LOGPROBS"]


def test_generated_engine_skips_logprobs_when_disabled(disabled_namespace):
    assert disabled_namespace["SAVE_LOGPROBS"] is False
    engine = disabled_namespace["engine"]
    assert "logprobs" not in engine.kwargs
    assert "top_logprobs" not in engine.kwargs


def test_generated_extract_logprobs_strips_byte_arrays(enabled_namespace):
    extract_logprobs = enabled_namespace["extract_logprobs"]
    raw = SimpleNamespace(choices=[SimpleNamespace(logprobs=FakeLogprobs())])
    assert extract_logprobs(make_action_step(raw)) == EXPECTED_LOGPROBS


def test_generated_extract_logprobs_none_without_raw_response(enabled_namespace):
    extract_logprobs = enabled_namespace["extract_logprobs"]
    assert extract_logprobs(make_action_step(raw=None)) is None

    step_without_message = ActionStep(
        step_number=1, timing=Timing(start_time=0.0, end_time=1.0)
    )
    assert extract_logprobs(step_without_message) is None

    replayed_step = make_action_step(raw=None)
    replayed_step.model_output_message = {"role": "assistant", "content": "hi"}
    assert extract_logprobs(replayed_step) is None


def test_generated_extract_logprobs_none_when_provider_omits_them(enabled_namespace):
    extract_logprobs = enabled_namespace["extract_logprobs"]
    raw = SimpleNamespace(choices=[SimpleNamespace(logprobs=None)])
    assert extract_logprobs(make_action_step(raw)) is None


def test_generated_save_memories_writes_logprobs_and_strips_raw(
    enabled_namespace, tmp_path
):
    raw = SimpleNamespace(choices=[SimpleNamespace(logprobs=FakeLogprobs())])
    fake_agent = SimpleNamespace(memory=SimpleNamespace(steps=[make_action_step(raw)]))
    enabled_namespace["save_agent_memories"](fake_agent, str(tmp_path), "single_agent")

    with open(tmp_path / "task_single_agent.json") as f:
        memories = json.load(f)
    assert memories[0]["logprobs"] == EXPECTED_LOGPROBS
    assert "raw" not in memories[0]["model_output_message"]


def test_generated_save_memories_warns_only_when_logprobs_all_missing(
    enabled_namespace, tmp_path, capsys
):
    def save_with_step(step):
        fake_agent = SimpleNamespace(memory=SimpleNamespace(steps=[step]))
        enabled_namespace["save_agent_memories"](
            fake_agent, str(tmp_path), "single_agent"
        )
        return capsys.readouterr().out

    assert "WARNING: logprobs requested but none returned" in save_with_step(
        make_action_step(raw=None)
    )

    with_logprobs = SimpleNamespace(choices=[SimpleNamespace(logprobs=FakeLogprobs())])
    assert "WARNING" not in save_with_step(make_action_step(with_logprobs))


def test_generated_save_memories_no_warning_when_disabled(
    disabled_namespace, tmp_path, capsys
):
    fake_agent = SimpleNamespace(
        memory=SimpleNamespace(steps=[make_action_step(raw=None)])
    )
    disabled_namespace["save_agent_memories"](fake_agent, str(tmp_path), "single_agent")
    assert "WARNING" not in capsys.readouterr().out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

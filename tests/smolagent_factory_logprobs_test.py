"""Tests for token-logprobs capture in the SmolAgent factory.

``smolagent_factory.py`` is not importable directly: it expects globals
(``MODEL_ID``, ``WorkflowState``, ...) injected by the workflow factory.
It is exec'd here with those globals stubbed, mirroring the sandbox.
"""

import json
import os
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from smolagents.memory import ActionStep, Timing
from smolagents.models import ChatMessage

REPO_ROOT = Path(__file__).resolve().parent.parent
FACTORY_PATH = REPO_ROOT / "sources/modules/smolagent_factory.py"
LANGFUSE_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


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


def load_factory_module(**extra_globals) -> types.ModuleType:
    """Exec the factory file with workflow-factory globals stubbed.

    Blanks LANGFUSE_* during exec so the factory's top level never
    activates the synchronous OTel exporter on machines that set them
    (load_dotenv inside the factory does not override existing vars).
    """
    saved_env = {key: os.environ.get(key) for key in LANGFUSE_KEYS}
    for key in LANGFUSE_KEYS:
        os.environ[key] = ""
    try:
        module = types.ModuleType("smolagent_factory_under_test")
        injected_globals = {
            "WorkflowState": dict,
            "MODEL_ID": "openrouter/test-model",
            "MEMORY_PATH": "sources/memory",
            "ENGINE_NAME": "litellm",
            "SYSTEM_PROMPT": "",
            **extra_globals,
        }
        module.__dict__.update(injected_globals)
        exec(compile(FACTORY_PATH.read_text(), str(FACTORY_PATH), "exec"), module.__dict__)
        return module
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(scope="module")
def factory_module() -> types.ModuleType:
    return load_factory_module()


def make_bare_factory(factory_module, **attrs):
    """Build a SmolAgentFactory without running __init__ (no engine, no agent)."""
    factory = object.__new__(factory_module.SmolAgentFactory)
    for key, value in attrs.items():
        setattr(factory, key, value)
    return factory


def make_action_step(raw) -> ActionStep:
    step = ActionStep(step_number=1, timing=Timing(start_time=0.0, end_time=1.0))
    step.model_output_message = ChatMessage(role="assistant", content="hi", raw=raw)
    return step


def test_extract_logprobs_returns_dict_without_byte_arrays(factory_module):
    factory = make_bare_factory(factory_module)
    raw = SimpleNamespace(choices=[SimpleNamespace(logprobs=FakeLogprobs())])
    assert factory.extract_logprobs(make_action_step(raw)) == EXPECTED_LOGPROBS


def test_extract_logprobs_none_without_raw_response(factory_module):
    factory = make_bare_factory(factory_module)
    assert factory.extract_logprobs(make_action_step(raw=None)) is None

    step_without_message = ActionStep(
        step_number=1, timing=Timing(start_time=0.0, end_time=1.0)
    )
    assert factory.extract_logprobs(step_without_message) is None

    replayed_step = make_action_step(raw=None)
    replayed_step.model_output_message = {"role": "assistant", "content": "hi"}
    assert factory.extract_logprobs(replayed_step) is None


def test_extract_logprobs_none_when_provider_omits_them(factory_module):
    factory = make_bare_factory(factory_module)
    raw = SimpleNamespace(choices=[SimpleNamespace(logprobs=None)])
    assert factory.extract_logprobs(make_action_step(raw)) is None


@pytest.mark.parametrize("save_logprobs", [True, False])
def test_get_engine_requests_logprobs_only_when_enabled(factory_module, save_logprobs):
    logprobs_kwargs = (
        {"logprobs": True, "top_logprobs": factory_module.TOP_LOGPROBS}
        if save_logprobs
        else {}
    )
    factory = make_bare_factory(
        factory_module,
        engine_name="litellm",
        save_logprobs=save_logprobs,
        logprobs_kwargs=logprobs_kwargs,
        openrouter_provider=None,
        model_id="openrouter/test-model",
        temperature=0.7,
        max_tokens=64,
        timeout=60,
    )
    engine = factory.get_engine()
    if save_logprobs:
        assert engine.kwargs["logprobs"] is True
        assert engine.kwargs["top_logprobs"] == factory_module.TOP_LOGPROBS
    else:
        assert "logprobs" not in engine.kwargs
        assert "top_logprobs" not in engine.kwargs


def test_logprobs_kwargs_full_for_supporting_provider(factory_module):
    assert factory_module.logprobs_kwargs_for("openrouter/test-model") == {
        "logprobs": True,
        "top_logprobs": factory_module.TOP_LOGPROBS,
    }


def test_logprobs_kwargs_empty_when_provider_lacks_logprobs(factory_module, capsys):
    """Mistral rejects both params with UnsupportedParamsError; drop them."""
    assert factory_module.logprobs_kwargs_for("mistral/mistral-small-2603") == {}
    assert "logprobs disabled" in capsys.readouterr().out


def test_logprobs_kwargs_drops_top_logprobs_when_unsupported(
    factory_module, monkeypatch
):
    import litellm

    monkeypatch.setattr(
        litellm, "get_supported_openai_params", lambda model: ["logprobs"]
    )
    assert factory_module.logprobs_kwargs_for("partial/model") == {"logprobs": True}


def test_logprobs_kwargs_keeps_request_without_param_map(factory_module, monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "get_supported_openai_params", lambda model: None)
    assert factory_module.logprobs_kwargs_for("custom/unmapped-model") == {
        "logprobs": True,
        "top_logprobs": factory_module.TOP_LOGPROBS,
    }


def test_save_logprobs_global_wires_through_real_init(tmp_path):
    """Full __init__ with the injected global, catching wiring-name drift."""
    module = load_factory_module(
        SAVE_LOGPROBS=True, MEMORY_PATH=str(tmp_path / "memory")
    )
    factory = module.SmolAgentFactory(name="wired", instruct_prompt="noop")
    assert factory.save_logprobs is True
    assert factory.engine.kwargs["logprobs"] is True
    assert factory.engine.kwargs["top_logprobs"] == module.TOP_LOGPROBS


def test_real_init_disables_logprobs_for_unsupported_provider(tmp_path):
    """Regression: mistral rejects logprobs params; __init__ must drop the
    request (and the missing-logprobs warning) instead of letting litellm
    raise UnsupportedParamsError on the first model call."""
    module = load_factory_module(
        SAVE_LOGPROBS=True,
        MODEL_ID="mistral/mistral-small-2603",
        MEMORY_PATH=str(tmp_path / "memory"),
    )
    factory = module.SmolAgentFactory(name="wired", instruct_prompt="noop")
    assert factory.save_logprobs is False
    assert "logprobs" not in factory.engine.kwargs
    assert "top_logprobs" not in factory.engine.kwargs


def test_workflow_factory_injects_save_logprobs_global():
    source = (REPO_ROOT / "sources/core/workflow_factory.py").read_text()
    assert "SAVE_LOGPROBS = {self.config.save_logprobs!r}" in source


def test_save_memories_writes_logprobs_and_strips_raw(factory_module, tmp_path):
    raw = SimpleNamespace(choices=[SimpleNamespace(logprobs=FakeLogprobs())])
    factory = make_bare_factory(
        factory_module,
        name="test",
        memory_folder=str(tmp_path),
        save_logprobs=True,
        agent=SimpleNamespace(memory=SimpleNamespace(steps=[make_action_step(raw)])),
    )
    factory.save_memories(workflow_uuid="test-uuid")

    with open(tmp_path / "task_test.json") as f:
        memories = json.load(f)
    assert memories[0]["logprobs"] == EXPECTED_LOGPROBS
    assert "raw" not in memories[0]["model_output_message"]


def test_save_memories_warns_only_when_logprobs_all_missing(
    factory_module, tmp_path, capsys
):
    def save_with_step(step):
        factory = make_bare_factory(
            factory_module,
            name="test",
            memory_folder=str(tmp_path),
            save_logprobs=True,
            agent=SimpleNamespace(memory=SimpleNamespace(steps=[step])),
        )
        factory.save_memories(workflow_uuid="test-uuid")
        return capsys.readouterr().out

    assert "WARNING: logprobs requested but none returned" in save_with_step(
        make_action_step(raw=None)
    )

    with_logprobs = SimpleNamespace(choices=[SimpleNamespace(logprobs=FakeLogprobs())])
    assert "WARNING" not in save_with_step(make_action_step(with_logprobs))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

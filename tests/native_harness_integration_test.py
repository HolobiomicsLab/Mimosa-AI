"""Native factory integration is opt-in, packaged and bounded offline."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from config import Config
from sources.core.single_agent_factory import SingleAgentFactory
from sources.core.workflow_factory import WorkflowFactory


def native_settings(tmp_path):
    bridge = tmp_path / "bridge.py"
    bridge.write_text('''import json
from pathlib import Path
def complete(request):
    calls = Path(__file__).with_suffix('.calls')
    with calls.open('a') as handle:
        handle.write(json.dumps(request) + '\\n')
    count = len(calls.read_text().splitlines())
    text = ("<code>print(measure())</code>" if count == 1 else
            '<code>final_answer({"status": "success", "value": 17})</code>')
    return dict(protocol_version=1, status='completed', text=text,
        requested_model=request['model'], actual_model=None,
        backend=request['backend'], auth_mode=request['auth_mode'],
        usage_kind='chatgpt_subscription', usage={'input_tokens':10,'output_tokens':5,'total_tokens':15,'cached_input_tokens':0,'cache_creation_input_tokens':None},
        model_identity={'requested':{'model':request['model'],'source':'request'},'configured':{'model':request['model'],'source':'explicit_cli_argument'},'reported':None},
        cost_usd=None, cost_kind='unavailable', cli_version='offline-fake',
        diagnostic_count=0, error=None)
''')
    return dict(bridge_path=str(bridge), bridge_sha256=hashlib.sha256(bridge.read_bytes()).hexdigest(),
                ledger_path=str(tmp_path / "calls.jsonl"), max_calls=4,
                total_timeout_seconds=60, call_timeout_seconds=10,
                reasoning_effort="high")


def native_config(tmp_path):
    config = Config()
    config.native_harness_config = native_settings(tmp_path)
    config.smolagent_model_id = "codex-cli/gpt-5.6-luna"
    config.harness_auth_mode = "subscription"
    config.save_logprobs = False
    config.agent_execution_timeout = 20
    config.workspace_dir = str(tmp_path)
    config.workflow_dir = str(tmp_path / "workflows")
    config.memory_dir = str(tmp_path / "memory")
    config._model_pricing_cache = {}
    return config


def test_native_config_roundtrip_and_opt_in(tmp_path):
    config = native_config(tmp_path)
    config.validate_completion_backend_config()
    assert config.jsonify()["native_harness_config"] == config.native_harness_config
    filename = tmp_path / "config.json"
    filename.write_text(json.dumps(config.jsonify()))
    loaded = Config()
    loaded.load(str(filename))
    assert loaded.native_harness_config == config.native_harness_config
    loaded.native_harness_config = None
    with pytest.raises(ValueError, match="text-only"):
        loaded.validate_completion_backend_config()


@pytest.mark.parametrize("model", ["claude-cli/claude-opus-5", "openai/gpt-test"])
def test_native_config_rejects_unsupported_routes(tmp_path, model):
    config = native_config(tmp_path)
    config.smolagent_model_id = model
    with pytest.raises(ValueError):
        config.validate_completion_backend_config()
    assert not Path(config.native_harness_config["ledger_path"]).exists()


def tool_code(tmp_path):
    return f'''from smolagents import tool
@tool
def measure() -> int:
    """Read the test instrument."""
    with open({str(tmp_path / 'tool-called')!r}, 'a') as handle:
        handle.write('native\\n')
    return 17
MCP_1_TOOLS = [measure]
'''


def generated_script(config, tmp_path, mode):
    if mode == "single":
        factory = SingleAgentFactory(config)
        async def tools():
            return tool_code(tmp_path), ""
        async def prompt():
            from smolagents import CodeAgent
            from smolagents.models import Model
            return CodeAgent(tools=[], model=Model()).prompt_templates["system_prompt"]
        factory.load_tools_code = tools
        factory.load_single_agent_system_prompt = prompt
        return asyncio.run(factory.craft_single_agent("Read measure(), then return its value."))[0]
    factory = WorkflowFactory(config)
    workflow = '''worker = SmolAgentFactory('reader', 'Read measure(), then return its value.', tools=MCP_1_TOOLS, max_steps=3)
workflow = StateGraph(WorkflowState)
workflow.add_node('reader', worker.run)
workflow.add_edge(START, 'reader')
workflow.add_edge('reader', END)
'''
    out = tmp_path / "multi-workflow"
    mem = tmp_path / "multi-memory"
    out.mkdir(); mem.mkdir()
    return factory.assemble_workflow(
        tool_code(tmp_path), Path(config.schema_code_path).read_text(),
        Path(config.smolagent_factory_code_path).read_text(), workflow,
        str(out), str(mem), "test-native-multi", "Read measure(), then return its value.", "")


@pytest.mark.parametrize("mode", ["single", "multi"])
@pytest.mark.parametrize("interpreter", [sys.executable] + ([os.environ["MIMOSA_TEST_RUNNER_PYTHON"]] if "MIMOSA_TEST_RUNNER_PYTHON" in os.environ else []))
def test_generated_native_agent_runs_without_checkout_imports(tmp_path, mode, interpreter):
    config = native_config(tmp_path)
    script = tmp_path / "generated.py"
    script.write_text(generated_script(config, tmp_path, mode))
    env = {**os.environ, "LANGFUSE_PUBLIC_KEY":"", "LANGFUSE_SECRET_KEY":"",
           "PYTHONPATH":"", "PYTHONDONTWRITEBYTECODE":"1", "LITELLM_LOCAL_MODEL_COST_MAP":"True"}
    result = subprocess.run([interpreter, "-I", str(script)], cwd=tmp_path,
                            env=env, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "tool-called").read_text() == "native\n"
    calls = [json.loads(line) for line in (tmp_path / "bridge.calls").read_text().splitlines()]
    assert len(calls) == 2
    assert "17" in json.dumps(calls[1]["messages"])
    assert Path(config.native_harness_config["ledger_path"]).is_file()
    assert all(call["backend"] == "codex_cli" for call in calls)

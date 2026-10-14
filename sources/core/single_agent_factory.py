"""Craft the executable script for one single-agent baseline run.

This module owns Mimosa-AI's single-agent code emitter: it turns a goal
into a ready-to-run SmolAgents script plus cost-tracking folders. The
workflow orchestrator constructs :class:`SingleAgentFactory` from here
when a task runs as one agent instead of a full workflow graph. Crafting
happens once per run, before the workflow runner (Layer 3) executes
anything.
"""

import logging
import os
import re
import time
import uuid

from .llm_provider import LLMConfig, LLMProvider, extract_model_pattern
from .factory import Factory

class SingleAgentFactory(Factory):
    """Craft the runnable script for a single-agent baseline run.

    The workflow orchestrator constructs it with the global ``Config``
    and calls it once per single-agent run. It mirrors
    :class:`WorkflowFactory` but emits a one-agent script, and it keeps
    the same workflow/memory folder pair so cost tracking stays uniform.
    """

    def __init__(self, config: "Config") -> None:
        """Store the config, the output folders, and a module logger.

        Called by the orchestrator at startup with the global config.

        Args:
            config: Global ``Config`` object from caller state; supplies
                ``workflow_dir``, ``memory_dir``, model ids, and engine
                settings.
        """
        self.workflow_dir = config.workflow_dir
        self.memory_dir = config.memory_dir
        self.config = config
        self.logger = logging.getLogger(__name__)


    async def craft_single_agent(self, goal: str, original_task: str | None = None) -> tuple[str, str, str]:
        """Craft a single-agent script and its cost-tracking folders.

        The orchestrator calls this once per single-agent run, passing
        the knowledge-wrapped goal and the raw task. It resolves the
        primary model, loads MCP tool code and the system prompt, renders
        the executable script, and writes the metadata files to disk.

        Args:
            goal: Goal text from the planner; may be knowledge-wrapped.
            original_task: Original unwrapped task from caller state,
                used for similarity matching. Optional.

        Returns:
            Tuple ``(complete_code, workflow_genotype_code, uuid_str)``.
            Both code entries hold the same executable script;
            ``uuid_str`` is the run id of the form
            ``single_agent_<timestamp>_<8-hex>``.

        Raises:
            RuntimeError: If MCP tool code or the system prompt cannot
                be loaded.
        """
        INSTRUCTIONS = goal
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        uuid_str = f"single_agent_{timestamp}_{short_uuid}"
        # `smolagent_model_id` may be a list of candidate models; a single agent
        # runs one model, so resolve to the primary (first) for the engine,
        # provider lookup and cost tracking below.
        model_id = self.config.smolagent_model_id[0] if isinstance(self.config.smolagent_model_id, list) else self.config.smolagent_model_id
        max_tokens = getattr(self.config, 'max_tokens', 8192)
        provider, _ = extract_model_pattern(model_id)
        token = os.getenv("HF_TOKEN") if provider == "huggingface" else None

        try:
            tools_code, existing_tool_prompt = await self.load_tools_code()
        except Exception as e:
            self.logger.error(f"craft_single_agent: Failed to load tools code: {str(e)}")
            raise RuntimeError(f"Failed to load tools code: {str(e)}") from e
        try:
            SYSTEM_PROMPT = await self.load_single_agent_system_prompt()
        except Exception as e:
            self.logger.error(f"craft_single_agent: Failed to load system prompt: {str(e)}")
            raise RuntimeError(f"Failed to load system prompt: {str(e)}") from e

        # Create folder structure for cost tracking (like multi-agent mode)
        workflow_path, memory_path = self.create_folder_structure(uuid_str)

        # Resolve absolute paths (like craft_workflow does)
        from pathlib import Path
        script_dir = Path(__file__).resolve().parent.parent.parent
        memory_path_abs = str((script_dir / memory_path).resolve())
        workflow_path_abs = str((script_dir / workflow_path).resolve())
        # create MCPs
        mcp_vars = sorted(set(
            re.findall(r"\bMCP_\d+_TOOLS\b", tools_code)
        ))
        mcps_string = "MCPS = [\n" + ",\n".join(f"    {name}" for name in mcp_vars) + "\n]"
        engine = "mlx" if "mlx-community/" in model_id else self.config.engine_name

        code = f"""
import os
import json
from dataclasses import asdict
from typing import List

import smolagents
from smolagents import CodeAgent, LiteLLMModel, ActionStep, InferenceClientModel, MLXModel
from smolagents.models import get_dict_from_nested_dataclasses
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = {model_id!r}
SYSTEM_PROMPT = {SYSTEM_PROMPT!r}
INSTRUCTIONS = {INSTRUCTIONS!r}
MEMORY_PATH = {memory_path_abs!r}
WORKFLOW_PATH = {workflow_path_abs!r}

model_id = {model_id!r}
max_tokens = {max_tokens}
provider = {provider!r}
token = {token!r}
engine_name = {engine!r}
openrouter_provider = {self.config.openrouter_provider_for(model_id)!r}
SAVE_LOGPROBS = {self.config.save_logprobs!r}
TOP_LOGPROBS = 5  # keep in sync with smolagent_factory.TOP_LOGPROBS

def logprobs_kwargs_for(model_id):
    \"\"\"Return the logprobs request params the model's provider accepts.

    Mirrors smolagent_factory.logprobs_kwargs_for: litellm raises
    UnsupportedParamsError for request params a provider lacks (mistral and
    anthropic take neither param, gemini takes logprobs but not
    top_logprobs), so build the request from its param map instead of
    crashing at call time. Providers litellm has no param map for keep the
    full request.
    \"\"\"
    kwargs = {{"logprobs": True, "top_logprobs": TOP_LOGPROBS}}
    try:
        import litellm
        supported = litellm.get_supported_openai_params(model=model_id)
    except Exception:
        return kwargs
    if supported is None:
        return kwargs
    if "logprobs" not in supported:
        print(f"WARNING: logprobs disabled for '{{model_id}}': provider does not support them.")
        return {{}}
    if "top_logprobs" not in supported:
        del kwargs["top_logprobs"]
    return kwargs

# Only the litellm engine forwards the logprobs request, and only when the
# provider accepts the params — gating both keeps the missing-logprobs
# warning honest.
logprobs_kwargs = logprobs_kwargs_for(model_id) if SAVE_LOGPROBS and engine_name == "litellm" else {{}}
save_logprobs = bool(logprobs_kwargs)
engine = None

if engine_name == "mlx":
    print("Using MLXModel for local execution.")
    engine = MLXModel(
        model_id=model_id,
        max_tokens=max_tokens,
    )
elif engine_name == "inference_client":
    print("Using InferenceClientModel for inference client execution.")
    if not token:
        raise ValueError("Hugging Face token is required. Please set the HF_TOKEN environment variable or pass a token.")
    engine = InferenceClientModel(
        model_id=model_id,
        provider=provider,
        token=token,
        max_tokens=max_tokens,
    )
elif engine_name == "litellm":
    _litellm_extra = dict(logprobs_kwargs)
    if openrouter_provider and str(model_id).startswith("openrouter/"):
        _order = [openrouter_provider] if isinstance(openrouter_provider, str) else list(openrouter_provider)
        _litellm_extra["extra_body"] = {{
            "provider": {{
                "order": _order,
                "allow_fallbacks": False,
                "require_parameters": True,
            }}
        }}
    # OpenAI deprecated `max_tokens` for `max_completion_tokens`;
    # newer OpenAI models hard-reject `max_tokens` with a 400.
    # LiteLLMModel forwards extra kwargs verbatim to litellm.
    _budget_kwarg = (
        "max_completion_tokens"
        if str(model_id).startswith("openai/")
        else "max_tokens"
    )
    engine = LiteLLMModel(
        model_id=model_id,
        temperature=1.0,
        **{{_budget_kwarg: max_tokens}},
        **_litellm_extra,
    )
elif engine_name == "openai":
    engine = InferenceClientModel(
        model_id=model_id,
        provider=provider,
        api_key=os.getenv("OPENAI_API_KEY")
    )
else:
    raise ValueError(f"Unknown engine name.. Supported engines are: mlx, hf_api, inference_client and litellm.")
{tools_code}
{mcps_string}

all_tools = []
for mcp_tools in MCPS:
    all_tools.extend(mcp_tools)

agent = CodeAgent(
    tools=all_tools,
    model=engine,
    name="single_agent",
    max_steps=64,
    additional_authorized_imports = [
        'requests', 'json', 'requests.exceptions',
        'os', 'sys', 'pathlib', 'shutil', 'glob', 'tempfile', 'argparse',
        'configparser', 'logging',
        'collections', 'itertools', 'functools', 'heapq', 'bisect', 'queue',
        'dataclasses', 'enum', 'types',
        're', 'string', 'textwrap', 'difflib', 'unicodedata',
        'csv', 'xml', 'xml.etree', 'xml.etree.ElementTree', 'pickle', 'base64',
        'html', 'html.parser', 'pandas', 'numpy', 'json', 'yaml',
        'datetime', 'time', 'calendar',
        'urllib', 'urllib.parse', 'urllib.request', 'urllib.error', 'http',
        'http.client', 'socket', 'email', 'mimetypes',
        'hashlib', 'hmac', 'secrets', 'uuid',
        'math', 'random', 'statistics', 'decimal', 'fractions',
        'traceback', 'inspect', 'gc', 'warnings', 'io',
        'gzip', 'zipfile', 'tarfile', 'zlib',
    ]
)

agent.prompt_templates["system_prompt"] = SYSTEM_PROMPT

def extract_logprobs(step):
    \"\"\"Return token logprobs from a step's raw model response, or None.

    Mirrors smolagent_factory.extract_logprobs: logprobs only exist on the
    raw provider response kept in model_output_message.raw, which
    save_agent_memories strips. Drops the per-token "bytes" arrays
    (redundant with "token") to keep memory files small.
    \"\"\"
    raw = getattr(getattr(step, "model_output_message", None), "raw", None)
    if raw is None:
        return None
    try:
        logprobs = raw.choices[0].logprobs
        if logprobs is None:
            return None
        dumped = logprobs.model_dump() if hasattr(logprobs, "model_dump") else dict(logprobs)
        for token_entry in dumped.get("content") or []:
            token_entry.pop("bytes", None)
            for alternative in token_entry.get("top_logprobs") or []:
                alternative.pop("bytes", None)
        return dumped
    except (AttributeError, IndexError, TypeError, KeyError):
        return None

def save_agent_memories(agent, memory_path: str, agent_name: str):
    print(f"Saving agent memory to: {{{{memory_path}}}}")
    try:
        memories = []
        for idx, step in enumerate(agent.memory.steps):
            if isinstance(step, ActionStep):
                action_step = step.dict()
                action_step["model_input_messages"] = (
                    get_dict_from_nested_dataclasses(
                        [asdict(msg) if hasattr(msg, '__dataclass_fields__') else msg for msg in step.model_input_messages],
                        ignore_key="raw"
                    )
                    if step.model_input_messages
                    else None
                )
                action_step["model_output_message"] = (
                    get_dict_from_nested_dataclasses(
                        step.model_output_message, ignore_key="raw"
                    )
                    if step.model_output_message
                    else None
                )
                action_step["logprobs"] = extract_logprobs(step)
                action_step["model"] = self.model_id
                memories.append(action_step)

        if save_logprobs and memories and all(m["logprobs"] is None for m in memories):
            print(f"WARNING: logprobs requested but none returned for agent '{{agent_name}}'; check provider support.")
        os.makedirs(memory_path, exist_ok=True)
        agent_task_path = os.path.join(memory_path, f"task_{{agent_name}}.json")
        with open(agent_task_path, "w") as f:
            json.dump(memories, f, indent=2)
        print(f"✅ Agent memories saved successfully to {{{{agent_task_path}}}}")
    except Exception as e:
        print(f"⚠️  Failed to save memory: {{{{str(e)}}}}")

# Run agent
result = agent.run(INSTRUCTIONS)

# Save agent memories for cost tracking
save_agent_memories(agent, MEMORY_PATH, "single_agent")

# Save state_result.json for cost tracking and evaluation
state_result = {{
    "model_id": MODEL_ID,
    "goal": INSTRUCTIONS,
    "workflow_uuid": "{uuid_str}",
    "single_agent_mode": True,
    "step_name": ["single_agent"],
    "answers": [str(result)],
    "success": [True]  # Assume success if no exception
}}

try:
    with open(os.path.join(WORKFLOW_PATH, "state_result.json"), "w") as f:
        json.dump(state_result, f, indent=2)
    print(f"✅ Saved state_result.json to {{WORKFLOW_PATH}}")
except Exception as e:
    print(f"❌ Failed to save state_result.json: {{e}}")
        """

        # Save metadata files (like multi-agent mode)
        self.save_workflow_files(
            workflow_path,
            uuid_str,
            code,  # Save the single agent code
            goal,
            original_task
        )

        return code, code, uuid_str

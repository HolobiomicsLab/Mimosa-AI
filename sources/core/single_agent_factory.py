"""
This class handles the creation and assembly of Single agent code.
"""

import logging
import os
import re
import time
import uuid

from .llm_provider import LLMConfig, LLMProvider, extract_model_pattern
from .factory import Factory

class SingleAgentFactory(Factory):
    """Handles the creation and management of Langraph-SmolAgent workflow generation"""

    def __init__(self, config) -> None:
        """Initialize the workflow crafting system.
        Args:
            config: Configuration object containing paths and settings
        """
        self.workflow_dir = config.workflow_dir
        self.memory_dir = config.memory_dir
        self.config = config
        self.logger = logging.getLogger(__name__)


    async def craft_single_agent(self, goal: str, original_task: str = None):
        """
        For crafting single agent with cost tracking support.

        Args:
            goal: The goal description (may be knowledge-wrapped)
            original_task: The original unwrapped task for similarity matching

        Returns:
            tuple[str, str, str]: (complete_code, workflow_genotype_code, uuid)
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        uuid_str = f"single_agent_{timestamp}_{short_uuid}"
        model_id = self.config.smolagent_model_id
        max_tokens = getattr(self.config, 'max_tokens', 8192)
        provider, _ = extract_model_pattern(self.config.smolagent_model_id)
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

        INSTRUCTIONS = ". ".join([
            "TASK:",
            goal,
            "",
            "CONSTRAINTS:",
            "- Never plot anything to the user. Plotting causes: 'terminating due to uncaught exception of type NSException'.",
            "- Save outputs instead of plotting.",
            "- Only use execute_command to install packages.",
            "- Wrap any command that may take significant time (>5 minutes) in a timeout.",
            "",
            "INITIAL STEP:",
            "- Assess the workspace by running: ls -la"
        ])
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

MODEL_ID = {self.config.smolagent_model_id!r}
GOAL = {goal!r}
SYSTEM_PROMPT = {SYSTEM_PROMPT!r}
INSTRUCTIONS = {INSTRUCTIONS!r}
MEMORY_PATH = {memory_path_abs!r}
WORKFLOW_PATH = {workflow_path_abs!r}

model_id = {model_id!r}
max_tokens = {max_tokens}
provider = {provider!r}
token = {token!r}
engine_name = {self.config.engine_name!r}
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
    engine = LiteLLMModel(
        model_id=model_id,
        temperature=1.0,
        max_tokens=max_tokens,
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
    max_steps=256,
    additional_authorized_imports = [
        'requests', 'bs4', 'json', 'requests.exceptions',
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
                memories.append(action_step)

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
    "goal": GOAL,
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

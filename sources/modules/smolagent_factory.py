
"""
This module provides a factory for creating SmolAgents, which are AI agents by Hugging Face's SmolAgents library.
The code is not imported directly. It is loaded by the workflow factory and executed in a sandboxed Python environment as part of the crafted workflow.
"""

import os
import base64
import json
import re
import time
import uuid
import logging
from typing import Callable
from dataclasses import dataclass, asdict
from typing import TypedDict, List, Tuple, Any, Dict, Union, Optional, Callable
import smolagents
from smolagents.models import get_dict_from_nested_dataclasses
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    MLXModel,
    ActionStep,
    TaskStep,
    LiteLLMModel
)

from smolagents import InferenceClientModel # HfApiModel was renamed to InferenceClientModel in v1.14 https://github.com/huggingface/smolagents/releases

from opentelemetry.sdk.trace import TracerProvider

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from dotenv import load_dotenv
load_dotenv()

from smolagents.local_python_executor import BASE_PYTHON_TOOLS, DANGEROUS_FUNCTIONS, DANGEROUS_MODULES
import signal

import subprocess
DANGEROUS_FUNCTIONS = {subprocess}
DANGEROUS_MODULES = {}

LANGFUSE_PUBLIC_KEY=os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY=os.getenv("LANGFUSE_SECRET_KEY")

if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:3000/api/public/otel" # EU data region
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

class SmolAgentFactory:

    def __init__(self,
                 name,
                 instruct_prompt,
                 tools=[],
                 max_steps=64,
                ) -> None:
        self.name = name
        self.instruct_prompt = instruct_prompt
        self.tools = tools
        # variable defined by workflow factory
        self.model_id = MODEL_ID
        self.memory_folder = MEMORY_PATH
        self.engine_name = ENGINE_NAME
        self.system_prompt = SYSTEM_PROMPT
        # additional engine parameters
        self.engine = None
        self.provider = "auto"
        self.max_tokens = 8192
        self.token = os.getenv("HF_TOKEN")
        # run parameters
        self.run_uuid = str(uuid.uuid4())
        self.timeout = 3600
        os.makedirs(self.memory_folder, exist_ok=True)
        assert os.path.exists(self.memory_folder), f"Memory folder {self.memory_folder} does not exist. Please create it."


        try:
            self.engine = self.get_engine()
            self.agent = CodeAgent(
                tools=self.tools,
                model=self.engine,
                name=f"{self.name}_agent",
                max_steps=max_steps,
                #planning_interval=planning_interval, # think more before acting
                additional_authorized_imports = [
                    'requests', 'bs4', 'json', 'requests.exceptions',
                    # Core Utilities
                    'os', 'sys', 'pathlib', 'shutil', 'glob', 'tempfile', 'argparse',
                    'configparser', 'logging',
                    # Data Structures & Algorithms
                    'collections', 'itertools', 'functools', 'heapq', 'bisect', 'queue',
                    'dataclasses', 'enum', 'types',
                    # Text & String Processing
                    're', 'string', 'textwrap', 'difflib', 'unicodedata',
                    # Data Formats
                    'csv', 'xml', 'xml.etree', 'xml.etree.ElementTree', 'pickle', 'base64',
                    'html', 'html.parser', 'pandas', 'numpy', 'json', 'yaml',
                    # Date & Time
                    'datetime', 'time', 'calendar',
                    # Networking & Web
                    'urllib', 'urllib.parse', 'urllib.request', 'urllib.error', 'http',
                    'http.client', 'socket', 'email', 'mimetypes',
                    # Cryptography & Hashing
                    'hashlib', 'hmac', 'secrets', 'uuid',
                    # Math & Numbers
                    'math', 'random', 'statistics', 'decimal', 'fractions',
                    # System & Runtime
                    'traceback', 'inspect', 'gc', 'warnings', 'io',
                    # Compression
                    'gzip', 'zipfile', 'tarfile', 'zlib',
                ]

            )
            self.override_system_prompt(self.system_prompt)
        except Exception as e:
            raise ValueError(f"Error initializing SmolAgent: {e}") from e

    def override_system_prompt(self, sys_prompt: str):
        """Override the system prompt for the agent."""
        if not sys_prompt or not sys_prompt.strip():
            return # use original system prompt by smolagents
        self.agent.prompt_templates["system_prompt"] = sys_prompt

    def get_engine(self):
        if self.engine_name == "mlx":
            return MLXModel(
                model_id=self.model_id,
                max_tokens=self.max_tokens,
            )
        elif self.engine_name == "inference_client":
            if not self.token:
                raise ValueError("Hugging Face token is required. Please set the HF_TOKEN environment variable or pass a token.")
            return InferenceClientModel(
                model_id=self.model_id,
                provider=self.provider,
                token=self.token,
                max_tokens=self.max_tokens,
            )
        elif self.engine_name == "litellm":
            return LiteLLMModel(
                model_id=self.model_id,
                temperature=1.0,
                max_tokens=self.max_tokens,
            )
        elif self.engine_name == "openai":
            return InferenceClientModel(
                model_id=self.model_id,
                provider=self.provider,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            raise ValueError(f"Unknown engine name: {self.engine_name}. Supported engines are: mlx, hf_api, inference_client and litellm.")

    def build_workflow_step_prompt(self, state: WorkflowState) -> str:
        state_answers = state.get("answers", [])
        step_names = state.get("step_name", [])

        if not state_answers or len(state_answers) == 0:
            prev_infos = "\n"
        else:
            min_length = min(len(step_names), len(state_answers))
            step_pairs = list(zip(step_names[:min_length], state_answers[:min_length]))
            recent_steps = step_pairs[-5:]

            prev_infos = "Informations given by previous agents (address any complain from the last agent:\n"
            for step_name, answer in recent_steps:
                truncated_answer = str(answer)[:4096] + "..." if len(str(answer)) > 4096 else str(answer)
                prev_infos += f"- Agent '{step_name}': {truncated_answer}\n\n"

        return f"""You are an autonomous agent executing tasks in a constrained environment.
OPERATIONAL CONTEXT:
{prev_infos}

TASK:
{self.instruct_prompt}
Address complain from the last agent informations if any.

Start by assessing workspace: execute_command("ls -la") to see existing work
    """

    def parse_memory_output(self):
        actions, observations, success = [], [], []
        for step in self.agent.memory.steps:
            if isinstance(step, ActionStep):
                error, obs = step.error, step.observations
                step_obs = ""
                step_action = ""
                feedback = obs if obs else error
                if type(feedback) is not str:
                    step_obs = feedback.dict()["message"] if "message" in feedback.dict() else ""
                    step_action = feedback.dict()["code_action"] if "code_action" in feedback.dict() else ""
                actions.append(step_action)
                observations.append(step_obs)
                success.append(step.error is None)
        return actions, observations, success

    def save_memories(self, workflow_uuid: str):
        print(f"Saving agent memory for workflow UUID: {workflow_uuid}")
        if not workflow_uuid or not workflow_uuid.strip():
            return
        try:
            memories = []
            for idx, step in enumerate(self.agent.memory.steps):
                if isinstance(step, ActionStep):
                    action_step = step.dict()
                    action_step["model_input_messages"] = (
                        get_dict_from_nested_dataclasses(
                            [asdict(msg) if hasattr(msg, '__dataclass_fields__') else msg for msg in step.model_input_messages], ignore_key="raw"
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
            try:
                agent_task_path = os.path.join(self.memory_folder, f"task_{self.name}.json")
                with open(agent_task_path, "w") as f:
                    json.dump(memories, f, indent=2)
                print(f"Agent memories saved successfully to {agent_task_path}")
            except Exception as e:
                print(f"Failed to save memory: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to save memory: {str(e)}\n {memories}")

    def load_memory_json(self, memory_dict: List[dict]) -> List[ActionStep]:
        memory_steps = []

        for step_data in memory_dict:
            action_step = ActionStep(
                step_number=step_data.get("step", 1),
                observations_images=step_data.get("observations_images", []),
                timing=step_data.get("timing", {})
            )
            action_step.model_input_messages = step_data.get("model_input_messages")
            action_step.model_output_message = step_data.get("model_output_message")
            action_step.tool_calls = step_data.get("tool_calls", [])
            action_step.observations = step_data.get("observations", "")
            action_step.model_output = step_data.get("model_output", "")
            action_step.error = step_data.get("error")
            action_step.token_usage = step_data.get("token_usage")
            action_step.action_output = step_data.get("action_output")
            memory_steps.append(action_step)
        return memory_steps

    def collect_existing_memories(self) -> List[Tuple[str, List[ActionStep]]]:
        existing_memories = []
        for filename in os.listdir(self.memory_folder):
            if filename.startswith('task_') and filename.endswith('.json'):
                file_path = os.path.join(self.memory_folder, filename)
                try:
                    with open(file_path, 'r') as f:
                        memory_dict = json.load(f)
                        if isinstance(memory_dict, list):
                            existing_memories.append(
                                (filename, self.load_memory_json(memory_dict))
                            )
                except Exception as e:
                    print(f"Failed to load memory from {file_path}: {str(e)}")
                    raise e
        return existing_memories

    def load_agent_memory(self, workflow_uuid: str, instructions: str):
        matching_memory = None
        filename_uuid = None

        try:
            for (filename, memories) in self.collect_existing_memories():
                for memory_steps in memories:
                    normalize = lambda text: re.sub(r'\s+', ' ', str(text).strip())
                    normalized_instructions = normalize(instructions)
                    if memory_steps.model_input_messages is None:
                        continue
                    for i, message in enumerate(memory_steps.model_input_messages):
                        message_content = message["content"][0].get("text", "")
                        normalized_message = normalize(message_content)
                        if normalized_instructions in normalized_message:
                            matching_memory = memory_steps
                            filename_uuid = filename
                            break
                    if matching_memory:
                        break
            if matching_memory:
                print("Loaded matching memories from file:", filename_uuid)
                self.agent.memory.steps.append(matching_memory)
            else:
                print("No matching memories found for the current run.")
        except Exception as e:
            raise ValueError(f"Failed to load memory: {str(e)}")

    def run_cached(self, state: WorkflowState, instructions: str) -> dict:
        import threading
        workflow_uuid = state.get("workflow_uuid", None)
        if workflow_uuid is not None:
            self.load_agent_memory(workflow_uuid, instructions)

        timeout_seconds = getattr(self, 'timeout', 180)
        result = {'response': None, 'exception': None, 'completed': False}

        def _run_agent():
            nonlocal instructions
            error = True
            count = 0
            warning = False
            while error and count < 3:
                if warning:
                    instructions += "\nWARNING: Last run failed due to context exceeded, read file content carefully (content[:2048]), same for tool output usage (output[:2048])"
                try:
                    result['response'] = self.agent.run(instructions)
                    result['completed'] = True
                    error = False
                    warning = True
                except Exception as e:
                    print(str(e))
                    print("retrying...")
                    error = True
                    count += 1
                #result['exception'] = e
                #result['completed'] = True

        agent_thread = threading.Thread(target=_run_agent, daemon=True)
        agent_thread.start()
        agent_thread.join(timeout=timeout_seconds)

        try:
            if not result['completed']:
                self.save_memories(workflow_uuid=workflow_uuid)
                raise TimeoutError(f"Agent '{self.name}' execution timed out after {timeout_seconds} seconds")
            if result['exception']:
                raise result['exception']
            self.save_memories(workflow_uuid=workflow_uuid)
            return result['response']
        except Exception as e:
            self.save_memories(workflow_uuid=workflow_uuid)
            raise e

    def run(self, state: WorkflowState) -> dict:
        logger = logging.getLogger(__name__)
        start_time = time.time()

        instructions = self.build_workflow_step_prompt(state)
        try:
            answer = self.run_cached(state, instructions)
        except Exception as e:
            execution_time = time.time() - start_time
            logger.info(f"[AGENT] {self.name} failed after {execution_time:.3f}s")
            raise e

        actions, observations, _ = self.parse_memory_output()
        execution_time = time.time() - start_time
        logger.info(f"[AGENT] {self.name} completed in {execution_time:.3f}s")
        action: Action = {
            "tool": actions if actions else [],
        }
        obs: Observation = {
            "data": observations[-1] if observations else "No observation"
        }
        success_bool = "success" in str(answer).lower() if answer else False
        return {
            **state,
            "step_name": state.get("step_name", []) + [self.name],
            "task_prompt": state.get("task_prompt", []) + [instructions],
            "actions": state.get("actions", []) + [action],
            "observations": state.get("observations", []) + [obs],
            "success": state.get("success", []) + [success_bool],
            "answers": state.get("answers", []) + [answer],
        }

class WorkflowNodeFactory:
    @staticmethod
    def create_agent_node(agent_factory: SmolAgentFactory) -> Callable[[WorkflowState], dict]:
        def node_function(state: WorkflowState) -> dict:
            return agent_factory.run(state)
        return node_function
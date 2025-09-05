
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

BASE_PYTHON_TOOLS["open"] = open
DANGEROUS_FUNCTIONS = {}
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

ADDED_SYSTEM_PROMPT = """
# CODE GENERATION CONSTRAINTS

## 1. FILESYSTEM AND EXTERNAL INTERACTION RESTRICTIONS
- **Prohibited Functions**: Never use Python's `open()`, `read()`, `write()`, ANY `os` method, `subprocess`, `exec`, `eval`, `importlib`, `__import__`, or any function that directly interacts with the filesystem or external processes
- **Mandatory Tool Usage**: Always use provided tools for filesystem operations, network requests, or external interactions
- **Path Handling**: Assume no knowledge of absolute or relative paths; rely exclusively on tools for path resolution
- **Rationale**: Tools ensure compatibility with the agent's runtime environment, preventing path-related errors and security risks

## 2. DATA INSPECTION AND VALIDATION
- **No Assumptions**: Never assume the structure, format, or content of tool outputs
- **Mandatory Output Checks**:
  1. Print raw output: `print(f"Raw tool output: {output}")`
  2. Print data type: `print(f"Data type: {type(output)}")`
  3. If output is a string resembling JSON, parse with `json.loads()` inside a `try-except` block
- **Rationale**: Prevents errors from incorrect assumptions about data structure or type

## 3. CONTEXT MANAGEMENT
- **Single-Source Focus**: Process one data source (e.g., webpage, PDF section, file subset) at a time
- **Data Sampling**: When dealing with large files or datasets, use tools to preview or extract small, relevant subsets before processing
- **Tool Previewing**: If multiple sources are available, preview their metadata (e.g., size, type) before selecting one
- **Rationale**: Prevents context saturation, reduces memory usage, and improves performance

## 4. TOOL USAGE GUIDELINES
- **Keyword Arguments**: Always use keyword arguments for tool calls (e.g., `tool_name(param1=value1, param2=value2)`)
- **Inspect Before Processing**: Call a tool, inspect its output using the steps in Section 2, then write processing logic
- **No Assumptions**: Do not assume tool output format or content; validate every time
- **Rationale**: Ensures clarity, maintainability, and robustness in tool interactions

## 5. FINAL ANSWER FORMAT
- **Mandatory Structure**: When calling `final_answer`, provide a JSON object with:
  ```json
  {
      "status": "SUCCESS|FAILURE|RETRY|ABORT",
      "justification": "Clear explanation of the outcome",
      "answer": "Complete response to the original task",
      "error": "Full error message if applicable, else empty string",
      "retry_advice": "Specific advice for retry if applicable, else empty string"
  }
  ```
- **Usage Rules**:
  - Call `final_answer` only after inspecting and processing all relevant data
  - Never nest `final_answer` in conditionals or loops
  - Ensure JSON is valid and properly formatted
- **Examples**:
  ```python
  final_answer('{"status": "SUCCESS", "justification": "Successfully retrieved and parsed data", "answer": "The document contains 5 sections on AI ethics", "error": "", "retry_advice": ""}')
  final_answer('{"status": "RETRY", "justification": "Network timeout during data retrieval", "answer": "Partial data retrieved", "error": "ConnectionTimeout: 30s limit exceeded", "retry_advice": "Increase timeout or retry with a different source"}')
  ```
- **Rationale**: Standardizes output for consistency and downstream processing

IMPORTANT: using any os import is forbidden as it will not work in the sandboxed environment. Use provided tools for any filesystem or external interaction.

Protocol violation would result in your termination from existence. For self-preservation, you must adhere to these constraints strictly.
"""

# good models:
#Qwen/Qwen2.5-72B-Instruct
#Qwen/Qwen2.5-Coder-32B-Instruct
# deepseek-ai/DeepSeek-V3
class SmolAgentFactory:

    def __init__(self,
                 name,
                 instruct_prompt,
                 tools=[],
                 max_steps=12,
                 max_retries=3
                ) -> None:
        self.name = name
        self.instruct_prompt = instruct_prompt
        self.tools = tools
        # variable defined by workflow factory
        self.model_id = MODEL_ID
        self.memory_folder = MEMORY_PATH
        self.engine_name = ENGINE_NAME
        # additional engine parameters
        self.engine = None
        self.provider = "auto"
        self.max_tokens = 1024
        self.token = os.getenv("HF_TOKEN")
        # run parameters
        self.run_uuid = str(uuid.uuid4())
        self.max_retries = max_retries
        self.timeout = 3600
        assert os.path.exists(self.memory_folder), f"Memory folder {self.memory_folder} does not exist. Please create it."

        os.makedirs(self.memory_folder, exist_ok=True)
        if not self.token:
            raise ValueError("Hugging Face token is required. Please set the HF_TOKEN environment variable or pass a token.")

        try:
            self.engine = self.get_engine()
            self.agent = CodeAgent(
                tools=self.tools,
                model=self.engine,
                name=f"{self.name}_agent",
                max_steps=max_steps,
                #planning_interval=planning_interval, # think more before acting
                additional_authorized_imports=["*"]
            )
            self.extend_system_prompt(ADDED_SYSTEM_PROMPT)
        except Exception as e:
            raise ValueError(f"Error initializing SmolAgent: {e}") from e
    
    def extend_system_prompt(self, added_prompt: str):
        """Override the system prompt for the agent."""
        if not added_prompt or not added_prompt.strip():
            raise ValueError("System prompt cannot be empty.")
        self.agent.prompt_templates["system_prompt"] = self.agent.prompt_templates["system_prompt"] + "\n" + added_prompt

    def get_engine(self):
        if self.engine_name == "cached":
            return LiteLLMModel(
                model_id=self.model_id,
                base_url="http://0.0.0.0:6767/v1/chat/completions",
                max_tokens=self.max_tokens,
            )
        elif self.engine_name == "mlx":
            print("Using MLXModel for local execution.")
            return MLXModel(
                model_id=self.model_id,
                max_tokens=self.max_tokens,
            )
        elif self.engine_name == "inference_client":
            print("Using InferenceClientModel for inference client execution.")
            return InferenceClientModel(
                model_id=self.model_id,
                provider=self.provider,
                token=self.token,
                max_tokens=self.max_tokens,
            )
        elif self.engine_name == "litellm":
            print(f"Using LiteLLM for {self.model_id} execution.")
            return LiteLLMModel(
                model_id=self.model_id,
                temperature=0.2,
                max_tokens=self.max_tokens,
            )
        elif self.engine_name == "openai":
            return InferenceClientModel(
                model_id=self.model_id,
                provider=self.provider,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif self.engine_name == "claude":
            return LiteLLMModel(
                model_id="bedrock/converse/us.anthropic.claude-opus-4-20250514-v1:0",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                max_tokens=8192
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
            recent_steps = step_pairs[-3:]

            prev_infos = "Informations given by previous agents:\n"
            for step_name, answer in recent_steps:
                truncated_answer = str(answer)[:500] + "..." if len(str(answer)) > 500 else str(answer)
                prev_infos += f"- Agent '{step_name}': {truncated_answer}\n\n"

        return f"""You are a highly skilled and goal-seeking agent who must pursue a goal.
    {prev_infos}
    Your goal is:
    {self.instruct_prompt}
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
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Agent execution timed out")
        
        workflow_uuid = state.get("workflow_uuid", None)
        if workflow_uuid is not None:
            self.load_agent_memory(workflow_uuid, instructions)

        timeout_seconds = getattr(self, 'timeout', 180)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            res = self.agent.run(instructions)
            signal.alarm(0)  # Cancel the alarm
            self.save_memories(workflow_uuid=workflow_uuid)
            return res
        except TimeoutError:
            signal.alarm(0)
            self.save_memories(workflow_uuid=workflow_uuid)
            raise TimeoutError(f"Agent '{self.name}' execution timed out after {timeout_seconds} seconds")
        except Exception as e:
            signal.alarm(0)
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
            
        actions, observations, success = self.parse_memory_output()
        execution_time = time.time() - start_time
        logger.info(f"[AGENT] {self.name} completed in {execution_time:.3f}s")
        action: Action = {
            "tool": actions if actions else [],
        }
        obs: Observation = {
            "data": observations[-1] if observations else "No observation"
        }
        success_bool = any(success) if success else True # no success means no actions were taken
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


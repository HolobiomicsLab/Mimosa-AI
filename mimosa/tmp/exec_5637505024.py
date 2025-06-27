
import os
import sys
import re
import json
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

# Load tools
'''
This module provides a set of tools for interacting with a CSV management API.
It is loaded in a sandbox python environment as part of the crafted workflow.
'''

from typing import List, Any, Dict
import asyncio
import json
from fastmcp import Client

from smolagents import Tool

API_SHELL_TOOLS_URL = 'http://localhost:5003'

def build_formatted_output(action: str, observation: str, reward: float) -> str:
    output = {
        "action": action[:256].strip().replace('\n', ' - '),
        "observation": observation[:4096],
        "reward": reward
    }
    return f"\n```json\n{json.dumps(output, indent=2)}\n```\n"

async def _async_shell_tool_call(tool_name: str, params: dict) -> dict:
    async with Client(f"{API_SHELL_TOOLS_URL}/mcp") as client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]
        assert tool_name in tool_names, "Fatal Error: " + tool_name + " not in tools list for mcp at " + API_SHELL_TOOLS_URL
        buffer = await client.call_tool(tool_name, params)
        return json.loads(buffer[0].text)

class ExecuteBashCommand(Tool):
    name = "execute_bash_command"
    description = "Execute a bash command."
    inputs = {"command": {"type": "string", "description": "The bash command to execute. Do not specify interpreter or shell, just the command itself."}}
    output_type = "string"

    def forward(self, command: str) -> str:
        action = f"execute_bash_command({command})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(_async_shell_tool_call("execute_command", {"command": command}))
            
            if result and result.get('status') == 'success' and 'stdout' in result:
                obs = result.get('stdout', 'No output') 
                reward = 1.0
            else:
                obs = result.get('stderr', 'Command execution failed')
                reward = 0.0
        except Exception as e:
            obs = str(e)
        return build_formatted_output(action, obs, reward)

# Tool instances
create_csv_tool = ExecuteBashCommand()

tools = [
    create_csv_tool
]

tools_name = [tool.name for tool in tools]

SHELL_TOOLS = tools
'''
This module provides a set of tools for interacting with a web browser instance.
It is loaded in a sandbox python environment as part of the crafted workflow.
'''

from typing import List, Any
import requests
import asyncio

from fastmcp import Client
from smolagents import (
    Tool,
    DuckDuckGoSearchTool
)
import json

API_BROWSER_TOOLS_URL = 'http://localhost:5002'

def build_formatted_output(action: str, observation: str, reward: float) -> str:
    output = {
        "action": action[:256].strip().replace('\n', ' - '),
        "observation": observation[:4096],
        "reward": reward
    }
    return f"\n```json\n{json.dumps(output, indent=2)}\n```\n"

async def _async_browser_tool_call(tool_name: str, params: dict) -> dict:
    async with Client(f"{API_BROWSER_TOOLS_URL}/mcp") as client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]
        assert tool_name in tool_names, "Fatal Error: " + tool_name + " not in tools list for mcp at " + API_BROWSER_TOOLS_URL
        buffer = await client.call_tool(tool_name, params)
        return json.loads(buffer[0].text)

class RestartBrowserTool(Tool):
    name = "restart_browser_tool"
    description = "Restart the browser instance. Use in case of browser errors or crashes."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'restart_browser_tool()'
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(_async_browser_tool_call("restart", {}))
            if result.get('status') == 'success':
                obs = 'Browser restarted successfully.'
                reward = 1.0
            else:
                obs = 'Failed to restart browser: ' + result.get('message', 'Unknown error')
        except Exception as e:
            print(str(e))
            obs = 'Error restarting browser: ' + str(e)
        return build_formatted_output(action, obs, reward)

class SearchTool(Tool):
    name = "search_tool"
    description = "Perform a search using DuckDuckGo and return the results."
    inputs = {"query": {"type": "string", "description": "The search query."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        obs = ''
        action = f"search_tool(query='{query}')"
        try:
            result = asyncio.run(_async_browser_tool_call("search", {"query": query}))
            obs = result.get('result', 'No results found')
            reward = 1.0 if obs else 0.0
        except Exception as e:
            print(str(e))
            obs = "Search failed for query: " + query + " due to error: " + str(e)
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class GoToUrlTool(Tool):
    name = "go_to_url_tool"
    description = "Navigate to a specified URL and return the page content as Markdown."
    inputs = {"url": {"type": "string", "description": "The URL to navigate to."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        action = f"go_to_url_tool(url='{url}')"
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(_async_browser_tool_call("navigate", {"url": url}))
        except Exception as e:
            print(str(e))
            obs = f'failed to navigate to {url} due to error: {str(e)}'
            return build_formatted_output(action, obs, reward)
        
        if not result or not 'success' in result.get('status', {}):
            obs = f'Error navigating to {url}: ' + result.get('message', 'Unknown error')
            return build_formatted_output(action, obs, reward)
        
        title = result.get('title', 'No title found')
        content = result.get('content', 'No content found')
        obs = f'''Tile: {title}
            {content}
        '''
        reward = 1.0
        return build_formatted_output(action, obs, reward)

class GetNavigableLinksTool(Tool):
    name = "get_navigable_links_tool"
    description = "Retrieves a list of navigable links on the current page."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'get_navigable_links_tool()'
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(_async_browser_tool_call("get_links", {}))
        except Exception as e:
            print(str(e))
            obs = 'Error getting navigable links due to error ' + str(e)
            return build_formatted_output(action, obs, reward)
        
        if 'error' in result.get('status', {}):
            obs = 'Error getting navigable links: ' + result['status']['error']
            return build_formatted_output(action, obs, reward)
        
        obs = result.get('links', [])
        reward = 1.0 if obs else 0.0
        return build_formatted_output(action, obs, reward)

class ScreenshotTool(Tool):
    name = "screenshot_tool"
    description = "Take a screenshot of the current page."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'screenshot()'
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(_async_browser_tool_call("screenshot", {}))
        except Exception as e:
            print(str(e))
            obs = 'Error taking screenshot due to error ' + str(e)
            reward = 0.0
            return build_formatted_output(action, obs, reward)
        
        if 'error' in result.get('status', {}):
            return build_formatted_output(action, obs, reward)
        
        filename = result.get('filename', 'No screenshot available')
        obs = f'Screenshot saved as {filename}'
        reward = 1.0
        return build_formatted_output(action, obs, reward)

search_tool = SearchTool()
go_to_url_tool = GoToUrlTool()
get_navigable_links_tool = GetNavigableLinksTool()
screenshot_tool = ScreenshotTool()
restart_tool = RestartBrowserTool()

tools = [
    search_tool,
    go_to_url_tool,
    get_navigable_links_tool,
    screenshot_tool,
    restart_tool
]

tools_name = [tool.name for tool in tools]

BROWSER_TOOLS = tools
'''
This module provides a set of tools for interacting with a CSV management API.
It is loaded in a sandbox python environment as part of the crafted workflow.
'''

from typing import List, Any, Dict
import asyncio
import json
from fastmcp import Client

from smolagents import Tool

API_CSV_TOOLS_URL = 'http://localhost:5001'

def build_formatted_output(action: str, observation: str, reward: float) -> str:
    output = {
        "action": action[:256].strip().replace('\n', ' - '),
        "observation": observation[:4096],
        "reward": reward
    }
    return f"\n```json\n{json.dumps(output, indent=2)}\n```\n"

async def _async_csv_tool_call(tool_name: str, params: dict) -> dict:
    async with Client(f"{API_CSV_TOOLS_URL}/mcp") as client:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]
        assert tool_name in tool_names, "Fatal Error: " + tool_name + " not in tools list for mcp at " + API_CSV_TOOLS_URL
        buffer = await client.call_tool(tool_name, params)
        return json.loads(buffer[0].text)

class CreateCSVTool(Tool):
    name = "create_csv_tool"
    description = "Create a new CSV dataset with optional columns and initial data."
    inputs = {
        "name": {"type": "string", "description": "Name for the dataset."},
        "columns": {"type": "array", "description": "List of column names (optional).", "nullable": True},
        "rows": {"type": "array", "description": "List of row data (optional).", "nullable": True}
    }
    output_type = "string"

    def forward(self, name: str, columns: List[str] = None, rows: List[List] = None) -> str:
        action = f"create_csv_tool({name})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(_async_csv_tool_call("create_csv", {"name": name, "columns": columns, "rows": rows}))
            
            if result and result.get('status') == 'success':
                obs = f"Created dataset '{name}' with shape {result.get('shape')}"
                reward = 1.0
            else:
                obs = f"Failed to create dataset: {result.get('status', 'Unknown error') if result else 'No response'}"
        except Exception as e:
            obs = f"Error creating dataset: {str(e)}"
            
        return build_formatted_output(action, obs, reward)

class LoadCSVTool(Tool):
    name = "load_csv_tool" 
    description = "Load CSV data from a file path into a named dataset."
    inputs = {
        "file_path": {"type": "string", "description": "Path to the CSV file."},
        "name": {"type": "string", "description": "Name for the dataset (optional, uses filename if not provided).", "nullable": True}
    }
    output_type = "string"

    def forward(self, file_path: str, name: str = None) -> str:
        action = f"load_csv_tool({file_path})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(_async_csv_tool_call("load_csv_from_path", {"source_path": file_path, "name": name})) if not result else result
            
            if result and result.get('status') == 'success':
                obs = f"Loaded dataset '{result.get('name')}' with shape {result.get('shape')}"
                reward = 1.0
            else:
                error_msg = result.get('status', 'Unknown error') if result else 'No response'
                obs = f"Failed to load CSV from '{file_path}': {error_msg}"
                if name:
                    obs += f" (as dataset: '{name}')"
                    
        except Exception as e:
            obs = f"Error loading CSV from '{file_path}': {str(e)}"
            
        return build_formatted_output(action, obs, reward)

class GetCSVDataTool(Tool):
    name = "get_csv_data_tool"
    description = "Get data from a dataset with optional filtering and pagination."
    inputs = {
        "name": {"type": "string", "description": "Name of the dataset."},
        "limit": {"type": "integer", "description": "Maximum number of rows to return (optional).", "nullable": True},
        "columns": {"type": "array", "description": "List of column names to return (optional).", "nullable": True}
    }
    output_type = "string"

    def forward(self, name: str, limit: int = None, columns: List[str] = None) -> str:
        action = f"get_csv_data_tool({name})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(_async_csv_tool_call("get_csv_data", {"name": name, "limit": limit, "columns": columns})) if not result else result
            
            if result and result.get('status') == 'success':
                rows = result.get('data', [])
                obs = f"Retrieved {len(rows)} rows from '{name}'"
                if rows:
                    obs += f" - Sample: {str(rows[0])[:200]}"
                reward = 1.0
            else:
                error_msg = result.get('status', 'Unknown error') if result else 'No response'
                obs = f"Failed to get data from '{name}': {error_msg}"
        except Exception as e:
            obs = f"Error getting data from '{name}': {str(e)}"
        return build_formatted_output(action, obs, reward)

class AddCSVRowTool(Tool):
    name = "add_csv_row_tool"
    description = "Add a new row to a dataset."
    inputs = {
        "name": {"type": "string", "description": "Name of the dataset."},
        "row": {"type": "object", "description": "Dictionary of column:value pairs for the new row."}
    }
    output_type = "string"

    def forward(self, name: str, row: Dict[str, Any]) -> str:
        action = f"add_csv_row_tool({name})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(_async_csv_tool_call("add_csv_row", {"name": name, "row": row})) if not result else result
            
            if result and result.get('status') == 'success':
                obs = f"Added row to '{name}', new shape: {result.get('shape')}"
                reward = 1.0
            else:
                obs = f"Failed to add row: {result.get('status', 'Unknown error') if result else 'No response'}"
                
        except Exception as e:
            obs = f"Error adding row: {str(e)}"
            
        return build_formatted_output(action, obs, reward)

class UpdateCSVRowTool(Tool):
    name = "update_csv_row_tool"
    description = "Update a specific row in a dataset by index."
    inputs = {
        "name": {"type": "string", "description": "Name of the dataset."},
        "index": {"type": "integer", "description": "Row index to update."},
        "row": {"type": "object", "description": "Dictionary of column:value pairs to update."}
    }
    output_type = "string"

    def forward(self, name: str, index: int, row: Dict[str, Any]) -> str:
        action = f"update_csv_row_tool({name})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(_async_csv_tool_call("update_csv_row", {"name": name, "index": index, "row": row})) if not result else result
            
            if result and result.get('status') == 'success':
                obs = f"Updated row {index} in '{name}'"
                reward = 1.0
            else:
                obs = f"Failed to update row: {result.get('status', 'Unknown error') if result else 'No response'}"

        except Exception as e:
            obs = f"Error updating row: {str(e)}"
            
        return build_formatted_output(action, obs, reward)

class AddCSVColumnTool(Tool):
    name = "add_csv_column_tool"
    description = "Add a new column to a dataset with optional default value."
    inputs = {
        "name": {"type": "string", "description": "Name of the dataset."},
        "column_name": {"type": "string", "description": "Name of the new column."},
        "default_value": {"type": "string", "description": "Default value for the new column (optional).", "nullable": True}
    }
    output_type = "string"

    def forward(self, name: str, column_name: str, default_value: Any = None) -> str:
        action = f"add_csv_column_tool({name}, {column_name})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(_async_csv_tool_call("add_csv_column", {"name": name, "column_name": column_name, "default_value": default_value})) if not result else result
            
            if result and result.get('status') == 'success':
                obs = f"Added column '{column_name}' to '{name}', new shape: {result.get('shape')}"
                reward = 1.0
            else:
                obs = f"Failed to add column: {result.get('status', 'Unknown error') if result else 'No response'}"
                
        except Exception as e:
            obs = f"Error adding column: {str(e)}"
            
        return build_formatted_output(action, obs, reward)

class QueryCSVTool(Tool):
    name = "query_csv_tool"
    description = "Perform analytical operations on a dataset (describe, value_counts, groupby, filter)."
    inputs = {
        "name": {"type": "string", "description": "Name of the dataset."},
        "operation": {"type": "string", "description": "Type of operation: 'describe', 'value_counts', 'groupby', 'filter'."},
        "column": {"type": "string", "description": "Column name for operations that require it (optional).", "nullable": True},
        "conditions": {"type": "object", "description": "Filter conditions as column:value pairs (for filter operation).", "nullable": True}
    }
    output_type = "string"

    def forward(self, name: str, operation: str, column: str = None, conditions: Dict[str, Any] = None) -> str:
        action = f"query_csv_tool({name}, {operation})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(_async_csv_tool_call("query_csv", {"name": name, "operation": operation, "column": column, "conditions": conditions}))
            
            if result and result.get('status') == 'success':
                query_result = result.get('result')
                obs = f"Query '{operation}' on '{name}' completed - Result: {str(query_result)[:500]}"
                reward = 1.0
            else:
                obs = f"Query failed: {result.get('status', 'Unknown error') if result else 'No response'}"
                
        except Exception as e:
            obs = f"Error querying dataset: {str(e)}"
            
        return build_formatted_output(action, obs, reward)

class ListCSVDatasetsTool(Tool):
    name = "list_csv_datasets_tool"
    description = "List all available datasets with their basic information."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = "list_csv_datasets_tool()"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(_async_csv_tool_call("list_csv_datasets", {})) if not result else result
            
            if result and result.get('status') == 'success':
                datasets = result.get('columns', [])
                obs = f"Found {len(datasets)} datasets: " + ", ".join([f"{d['name']}({d['shape']})" for d in datasets])
                reward = 1.0
            else:
                obs = "Failed to list datasets:" + f" {result.get('status', 'Unknown error') if result else 'No response'}"
        except Exception as e:
            obs = f"Error listing datasets: {str(e)}"
            
        return build_formatted_output(action, obs, reward)

# Tool instances
create_csv_tool = CreateCSVTool()
load_csv_tool = LoadCSVTool()
get_csv_data_tool = GetCSVDataTool()
add_csv_row_tool = AddCSVRowTool()
update_csv_row_tool = UpdateCSVRowTool()
add_csv_column_tool = AddCSVColumnTool()
query_csv_tool = QueryCSVTool()
list_csv_datasets_tool = ListCSVDatasetsTool()

tools = [
    create_csv_tool,
    load_csv_tool,
    get_csv_data_tool,
    add_csv_row_tool,
    update_csv_row_tool,
    add_csv_column_tool,
    query_csv_tool,
    list_csv_datasets_tool,
]

tools_name = [tool.name for tool in tools]

CSV_TOOLS = tools


# Load state schema
from typing import TypedDict, List

class Action(TypedDict):
    tool: str

class Observation(TypedDict):
    data: str

class WorkflowState(TypedDict):
    workflow_uuid: str
    step_name: List[str]
    step_uuid: List[str]
    task_prompt: List[str]
    actions: List[Action]
    observations: List[Observation]
    rewards: List[float]
    answers: List[str]
    success: List[bool]

# Load smolagent factory

import os
import base64
import json
import re
import time
import uuid
from typing import Callable
from typing import TypedDict, List, Tuple, Any, Dict, Union, Optional, Callable
import smolagents
from smolagents import (
    CodeAgent,
    MLXModel,
    ActionStep,
    TaskStep
)

try:
    from smolagents import HfApiModel
except ImportError:
    from smolagents import InferenceClientModel as HfApiModel # HfApiModel was renamed to InferenceClientModel in v1.14 https://github.com/huggingface/smolagents/releases

from opentelemetry.sdk.trace import TracerProvider

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from dotenv import load_dotenv
load_dotenv()

from smolagents.local_python_executor import BASE_PYTHON_TOOLS, DANGEROUS_FUNCTIONS, DANGEROUS_MODULES

BASE_PYTHON_TOOLS["open"] = open
DANGEROUS_FUNCTIONS = {}
DANGEROUS_MODULES = {}

LANGFUSE_PUBLIC_KEY=os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY=os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()

os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:3000/api/public/otel" # EU data region
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"

trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

# good models:
#Qwen/Qwen2.5-72B-Instruct
#Qwen/Qwen2.5-Coder-32B-Instruct
# deepseek-ai/DeepSeek-V3
class SmolAgentFactory:

    def __init__(self,
                 instruct_prompt,
                 tools,
                 model_id="deepseek-ai/DeepSeek-V3",
                 engine_name="hf_api",
                 max_steps=10
                ):
        self.model_id = model_id
        self.max_tokens = 1024
        self.provider = "fireworks-ai"
        self.token = os.getenv("HF_TOKEN")
        self.tools = tools or []
        self.instruct_prompt = instruct_prompt
        self.local = False
        self.engine_name = engine_name
        self.engine = None
        self.run_uuid = str(uuid.uuid4())
        self.memory_folder = './memory' 
        os.makedirs(self.memory_folder, exist_ok=True)

        if not self.token:
            raise ValueError("Hugging Face token is required. Please set the HF_TOKEN environment variable or pass a token.")
        try:
            self.engine = self.get_engine()
            self.agent = CodeAgent(
                tools=self.tools,
                model=self.engine,
                name="agent",
                max_steps=max_steps,
                additional_authorized_imports=["*"]
        )
        except Exception as e:
            raise ValueError(f"Error initializing SmolAgent: {e}") from e
        
    def get_engine(self):
        if self.engine_name == "mlx":
            print("Using MLXModel for local execution.")
            self.local = True
            return MLXModel(
                model_id=self.model_id,
                max_tokens=self.max_tokens,
            )
        elif self.engine_name == "hf_api":
            print("Using HfApiModel for Hugging Face API execution.")
            return HfApiModel(
                model_id=self.model_id,
                provider=self.provider,
                token=self.token,
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
        elif self.engine_name == "openai":
            return InferenceClientModel(
                model_id="gpt-4o",
                provider="openai",
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            raise ValueError(f"Unknown engine name: {self.engine_name}. Supported engines are: mlx, hf_api, inference_client.")

    def build_workflow_step_prompt(self, state: WorkflowState) -> str:
        state_steps = state.get("step_name", [])
        state_actions = state.get("actions", [])
        state_observations = state.get("observations", [])
        state_success = state.get("success", [])
        state_rewards = state.get("rewards", [])
        state_answers = state.get("answers", [])
        trajectories = zip(
            state_actions, 
            state_observations, 
            state_success
        )
        trajectory_str = ""
        #for idx, (action, observation, success) in enumerate(trajectories):
        #    if not action or action == {}:
        #        continue
        #    trajectory_str += f"""
        #    ### Step {idx + 1}:
        #    Action: {action.get("tool", "No action specified")}
        #    Observation: {observation.get('data', 'No observation data')}... (truncated for brevity)
        #    Success: {success}
        #    ---
        #    """
        prev_infos = state_observations[-1]['data'][:2048] if state_observations else "No observations yet"
        print("Debug state_answers:", state_answers)
        prev_infos += '\n' + state_answers[-1] if state_answers else ""
        return f"""
        You are an AI agent designed to assist with a specific task.
        Previous agents have provided the following information:
        {prev_infos}
        Your need to follow instructions:
        {self.instruct_prompt}
        Avoid making overly complex code for simple tasks. Be patient and thorough.
        Do not make assumptions about the data returned by the tools. Try a tool, see its output, then you might write code to process it.
        If encountering rate limits, timeout, or processing time issues, you might use a while loop with state checks, retries, or exponential backoff strategies.
        """

    def parse_tool_output(self, output: str):
        actions = []
        observations = []
        rewards = []
        success = []
        # Look for ```json blocks in the output
        json_blocks = re.findall(r"```json\n(.*?)\n```", output, re.DOTALL)
        if not json_blocks:
            return (output, "Completed", 0.0, True)  # No valid JSON blocks found
        
        for block in json_blocks:
            try:
                data = json.loads(block)
                if "action" in data:
                    actions.append(data["action"])
                if "observation" in data:
                    observations.append(data["observation"])
                if "reward" in data:
                    reward = float(data["reward"])
                    rewards.append(reward)
                    success.append(reward > 0)
            except json.JSONDecodeError:
                continue
        
        return (
            "\n".join(actions),
            "\n".join(observations),
            (sum(rewards) / len(rewards)) if len(rewards) > 0 else 0,
            any(success) or len(success) == 0
        )
    
    
    def parse_memory_output(self):
        text_memory_length = 0 
        actions, observations, rewards, success = [], [], [], []
        for idx, step in enumerate(self.agent.memory.steps):
            if isinstance(step, ActionStep):
                error, feedback = step.error, step.observations
                step_output = error if error else feedback
                if not isinstance(step_output, str):
                    continue
                text_memory_length += len(step_output)
                action_step, obs_step, reward_step, success_step = self.parse_tool_output(step_output)
                if reward_step <= 0.0:
                    continue
                actions.append(action_step)
                observations.append(obs_step)
                rewards.append(reward_step)
                success.append(success_step)
        print(f"Parsed {len(actions)} actions, {len(observations)} observations, {len(rewards)} rewards, and {len(success)} success flags from memory.")
        print(f"Total text memory length: {text_memory_length} characters.")
        return actions, observations, rewards, success

    def save_memories(self, workflow_uuid: str):
        memories = []
        if not workflow_uuid or not workflow_uuid.strip():
            return
        memory_folder_path = os.path.join(self.memory_folder, workflow_uuid)
        #self.agent.save(f"./{memory_folder_path}/save_{workflow_uuid}_{self.run_uuid}.json")
        for memory_json in self.agent.memory.get_full_steps():
            memories.append(memory_json)
        try:
            os.makedirs(memory_folder_path, exist_ok=True)
            with open(os.path.join(memory_folder_path, f"memory_{self.run_uuid}.json"), "w") as f:
                json.dump(str(memories), f, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to save memory: {str(e)}")

    def get_memory_file_paths(self, workflow_uuid: Optional[str] = None) -> str:
        files = []
        if workflow_uuid is None or not workflow_uuid.strip():
            return []
        memory_folder_path = os.path.join(self.memory_folder, workflow_uuid)
        if not os.path.exists(memory_folder_path):
            print(f"Memory folder {memory_folder_path} does not exist. No cached memories available.")
            return []
        for file in os.listdir(memory_folder_path):
            if file.startswith("memory_") and file.endswith(".json"):
                files.append(file)
        return files
    
    def load_memories(self, file_path):
        memories = []
        try:
            with open(file_path, "r") as f:
                memories = json.load(f)
        except FileNotFoundError:
            print(f"No cached memory found for run {self.run_uuid}. Starting fresh.")
            return []
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to load memory: {str(e)}")
        print(f"Loaded {len(memories)} steps from memory for run {self.run_uuid}.")
        return memories
    
    def run_cached(self, state: WorkflowState, instructions: str) -> dict:
        memories = []
        workflow_uuid = state.get("workflow_uuid", None)
        memories_files = self.get_memory_file_paths(workflow_uuid=workflow_uuid)
        for memory_file in memories_files:
            memory = self.load_memories(memory_file)
            memories.extend(memory)
        if not memories or len(memories) == 0:
            output = self.agent.run(instructions)
            self.save_memories(workflow_uuid=workflow_uuid)
            return output
        for memory in memories:
            print("loading memory:\n", memory)
            # TODO how to make a ActionStep from a memory?
            #if memory.get("task") == state.get("task_prompt"):
            #    self.agent.memory.steps.append(memory)
            exit()
        return self.agent.run(instructions)


    def run(self, state: WorkflowState) -> dict:
        instructions = self.build_workflow_step_prompt(state)
        try:
            answer = self.run_cached(state, instructions)
        except Exception as e:
            print(f"Error running agent: {e}")
            raise e
            return {
                **state,
                "step_uuid": state.get("step_uuid", []) + [self.run_uuid],
                "actions": state.get("actions", []) + [{"tool": "LLM request"}],
                "observations": state.get("observations", []) + [{"data": str(e)}],
                "rewards": state.get("rewards", []) + [0.0],
                "success": state.get("success", []) + [False],
                "answers": state.get("answers", []) + ["Error in step execution."],
            }
        actions, observations, rewards, success = self.parse_memory_output()
        action: Action = { # Only the last action matters for the state
            "tool": actions[-1] if actions else "No action",
        }
        obs: Observation = { # Only the last observation matters for the state
            "data": observations[-1] if observations else "No observation"
        }
        reward = sum(rewards) / len(rewards) if rewards else 0.0
        # return True if final answer was called (no tool called, so array is empty).
        success_bool = success[-1] if len(success) > 0 else True
        return {
            **state,
            "step_uuid": state.get("step_uuid", []) + [self.run_uuid],
            "actions": state.get("actions", []) + [action],
            "observations": state.get("observations", []) + [obs],
            "rewards": state.get("rewards", []) + [reward],
            "success": state.get("success", []) + [success_bool],
            "answers": state.get("answers", []) + [answer],
        }

class WorkflowNodeFactory:
    @staticmethod
    def create_agent_node(agent_factory: SmolAgentFactory) -> Callable[[WorkflowState], dict]:
        def node_function(state: WorkflowState) -> dict:
            return agent_factory.run(state)
        return node_function



# Generated workflow
# =====================================================================
# LANGGRAPH ‑ SMOLAGENT WORKFLOW : “mzmind batch-mode deep-dive”
# =====================================================================
#
#  Goal :
#     1. Perform an exhaustive web investigation on “mzmind” batch /
#        command-line usage
#     2. Extract concrete CLI + config-file examples
#     3. (Optionally) validate syntax with a shell probe
#     4. Aggregate results in CSV
#     5. Deliver a final human-readable report
#
#  Architectural highlights
#     • 5 atomic agents  (+2 dedicated fall-backs)  = 7 total nodes
#     • Multiple retry & alternative paths (≥2 fall-backs)
#     • Robust routing with attempt counters & emergency END gate
# =====================================================================

from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------
# 1.  AGENT INSTRUCTIONS
# ---------------------------------------------------------------------

# --- Web searcher ----------------------------------------------------
instruct_web_search = """
You are WEB_SEARCHER, an internet research agent.

TASK
- Search for authoritative information about using "mzmind" in batch / command-line mode
- Collect URLs, page titles and short excerpts that mention:
  * command syntax (mzmind …)
  * batch configuration files (yaml / json / txt etc.)
  * usage examples, flags, options
- Store findings in your final answer as a numbered list

UPON COMPLETION
  If you found ≥3 distinct credible sources say exactly: SEARCH_COMPLETE
  If you failed or found <3 useful sources say: SEARCH_FAILURE
  On unexpected errors say: GIVE_UP
"""

# --- Deep/fallback searcher -----------------------------------------
instruct_alt_search = """
You are ALT_WEB_SEARCHER, a deep-dive research agent specialising in forums,
GitHub issues and niche blogs.

TASK
- Perform secondary searches for "mzmind batch", "mzmind --batch", "mzmind cli",
  "mzmind config file", "mzmind.exe /batch"
- Look into sources such as GitHub, StackOverflow, Gists, archived docs, etc.
- Output additional links and snippets NOT returned by the primary search.

UPON COMPLETION
  If you added ≥2 NEW sources say: ALT_SEARCH_COMPLETE
  If still insufficient info say: ALT_SEARCH_FAILURE
  On technical problems say: GIVE_UP
"""

# --- Content extractor ----------------------------------------------
instruct_extractor = """
You are CONTENT_EXTRACTOR, an agent that visits provided URLs and extracts
specific artefacts.

INPUT
- The previous agent's answer contains URLs and snippets.

TASK
- For each URL: visit the page, extract any CLI commands (mzmind …),
  flags, and configuration file blocks.
- Consolidate extracted artefacts into a clean markdown list.

UPON COMPLETION
  If at least one command and one config snippet extracted say: EXTRACTION_COMPLETE
  Else say: EXTRACTION_FAILURE
  For technical issues say: GIVE_UP
"""

# --- Shell validator -------------------------------------------------
instruct_validator = """
You are SHELL_VALIDATOR, an agent with terminal access (mock / limited).

TASK
- For each extracted command create a dry-run check:
     * If 'mzmind' exists, run 'mzmind --help' to verify availability.
     * Otherwise simulate with 'echo "<command>"'.
- Report whether syntax appears valid (exit code 0 etc.)
- Do NOT modify commands.

UPON COMPLETION
  If all commands checked produce no critical errors say: VALIDATION_COMPLETE
  If some commands problematic say: VALIDATION_FAILURE
  On tool errors say: GIVE_UP
"""

# --- CSV compiler ----------------------------------------------------
instruct_csv = """
You are CSV_COMPILER.

TASK
- Convert validated (or partially validated) examples into CSV with columns:
    use_case, cli_command, config_file_name, notes
- Use CSV tools to generate and store the file.

UPON COMPLETION
  If CSV created successfully say: CSV_COMPLETE
  Otherwise say: CSV_FAILURE
  On errors say: GIVE_UP
"""

# --- Final report formatter -----------------------------------------
instruct_report = """
You are REPORT_FORMATTER.

TASK
- Produce a final human-readable brief that includes:
   * Overview of what mzmind batch-mode is
   * The CLI examples (formatted)
   * The corresponding config files (formatted)
   * Validation results
   * Short CSV preview with link / path
- Be thorough but concise.

UPON COMPLETION
  If satisfied say: REPORT_COMPLETE
  If critical data missing say: REPORT_FAILURE
  On technical problems say: GIVE_UP
"""

# --- Optional “give-up” emitter -------------------------------------
instruct_give_up = """
You are GIVE_UP_AGENT.

TASK
- Explain clearly what went wrong in the workflow and suggest human
  interventions.

Always end with: WORKFLOW_ABORTED
"""

# ---------------------------------------------------------------------
# 2.  AGENT DECLARATION
# ---------------------------------------------------------------------

search_agent          = SmolAgentFactory(instruct_web_search,  BROWSER_TOOLS)
alt_search_agent      = SmolAgentFactory(instruct_alt_search, BROWSER_TOOLS)
extractor_agent       = SmolAgentFactory(instruct_extractor,  BROWSER_TOOLS)
validator_agent       = SmolAgentFactory(instruct_validator,  SHELL_TOOLS)
csv_agent             = SmolAgentFactory(instruct_csv,        CSV_TOOLS)
report_agent          = SmolAgentFactory(instruct_report,     [])              # no tools needed
giveup_agent          = SmolAgentFactory(instruct_give_up,    [])              # explanatory only

# ---------------------------------------------------------------------
# 3.  WORKFLOW GRAPH
# ---------------------------------------------------------------------

workflow = StateGraph(WorkflowState)

# ---- Add nodes ------------------------------------------------------
workflow.add_node("web_searcher",         WorkflowNodeFactory.create_agent_node(search_agent))
workflow.add_node("alt_web_searcher",     WorkflowNodeFactory.create_agent_node(alt_search_agent))
workflow.add_node("content_extractor",    WorkflowNodeFactory.create_agent_node(extractor_agent))
workflow.add_node("shell_validator",      WorkflowNodeFactory.create_agent_node(validator_agent))
workflow.add_node("csv_compiler",         WorkflowNodeFactory.create_agent_node(csv_agent))
workflow.add_node("report_formatter",     WorkflowNodeFactory.create_agent_node(report_agent))
workflow.add_node("give_up",              WorkflowNodeFactory.create_agent_node(giveup_agent))

# ---------------------------------------------------------------------
# 4.  ROUTING / EDGE LOGIC
# ---------------------------------------------------------------------

def safe_get(lst, idx, default=None):
    try:
        return lst[idx]
    except Exception:
        return default


# ---------- Router : generic with step-specific triggers -------------
def router(state: WorkflowState,
           success_phrases: list[str],
           failure_phrases: list[str],
           current_node: str,
           success_target: str,
           retry_target: str,
           alt_target: str):
    """
    Generic decision routine used by individual wrappers below.
    """
    try:
        ans_list     = state.get("answers", [])
        steps        = state.get("step_name", [])
        last_answer  = safe_get(ans_list, -1, "")
        retry_count  = steps.count(current_node)

        # --- success check ------------------------------------------
        if any(phrase in last_answer for phrase in success_phrases):
            return success_target

        # --- failure handling ---------------------------------------
        if any(phrase in last_answer for phrase in failure_phrases):
            if retry_count < 3:
                # simple retry (same node)
                return retry_target
            else:
                # after 3 attempts, escalate
                return alt_target

        # --- explicit GIVE_UP word ----------------------------------
        if "GIVE_UP" in last_answer:
            return "give_up"

        # --- defensive default --------------------------------------
        # Treat unknown response as failure and escalate
        if retry_count < 3:
            return retry_target
        else:
            return alt_target

    except Exception as e:
        print(f"💥 Router error at {current_node}: {e}")
        return "give_up"


# ---- Individual tiny wrappers (for readability) ---------------------

def search_router(state: WorkflowState) -> str:
    return router(state,
                  success_phrases=["SEARCH_COMPLETE"],
                  failure_phrases=["SEARCH_FAILURE"],
                  current_node="web_searcher",
                  success_target="content_extractor",
                  retry_target="web_searcher",
                  alt_target="alt_web_searcher")

def alt_search_router(state: WorkflowState) -> str:
    return router(state,
                  success_phrases=["ALT_SEARCH_COMPLETE"],
                  failure_phrases=["ALT_SEARCH_FAILURE"],
                  current_node="alt_web_searcher",
                  success_target="content_extractor",
                  retry_target="alt_web_searcher",
                  alt_target="give_up")

def extract_router(state: WorkflowState) -> str:
    return router(state,
                  success_phrases=["EXTRACTION_COMPLETE"],
                  failure_phrases=["EXTRACTION_FAILURE"],
                  current_node="content_extractor",
                  success_target="shell_validator",
                  retry_target="content_extractor",
                  alt_target="give_up")

def validate_router(state: WorkflowState) -> str:
    # If validation fails we will still allow progress to CSV after 3 tries
    route = router(state,
                   success_phrases=["VALIDATION_COMPLETE"],
                   failure_phrases=["VALIDATION_FAILURE"],
                   current_node="shell_validator",
                   success_target="csv_compiler",
                   retry_target="shell_validator",
                   alt_target="csv_compiler")
    return route

def csv_router(state: WorkflowState) -> str:
    return router(state,
                  success_phrases=["CSV_COMPLETE"],
                  failure_phrases=["CSV_FAILURE"],
                  current_node="csv_compiler",
                  success_target="report_formatter",
                  retry_target="csv_compiler",
                  alt_target="give_up")

def report_router(state: WorkflowState) -> str:
    route = router(state,
                   success_phrases=["REPORT_COMPLETE"],
                   failure_phrases=["REPORT_FAILURE"],
                   current_node="report_formatter",
                   success_target=END,
                   retry_target="report_formatter",
                   alt_target="give_up")
    # If report passes END else continues / escalates
    return route

# ---------------------------------------------------------------------
# 5.  EDGE DEFINITIONS
# ---------------------------------------------------------------------

workflow.add_edge(START, "web_searcher")

workflow.add_conditional_edges("web_searcher",
                               search_router,
                               {
                                   "content_extractor":  "content_extractor",
                                   "web_searcher":       "web_searcher",       # retry
                                   "alt_web_searcher":   "alt_web_searcher",
                                   "give_up":            "give_up"
                               })

workflow.add_conditional_edges("alt_web_searcher",
                               alt_search_router,
                               {
                                   "content_extractor":  "content_extractor",
                                   "alt_web_searcher":   "alt_web_searcher",   # retry
                                   "give_up":            "give_up"
                               })

workflow.add_conditional_edges("content_extractor",
                               extract_router,
                               {
                                   "shell_validator":    "shell_validator",
                                   "content_extractor":  "content_extractor", # retry
                                   "give_up":            "give_up"
                               })

workflow.add_conditional_edges("shell_validator",
                               validate_router,
                               {
                                   "csv_compiler":       "csv_compiler",
                                   "shell_validator":    "shell_validator",   # retry
                                   "give_up":            "give_up"
                               })

workflow.add_conditional_edges("csv_compiler",
                               csv_router,
                               {
                                   "report_formatter":   "report_formatter",
                                   "csv_compiler":       "csv_compiler",      # retry
                                   "give_up":            "give_up"
                               })

workflow.add_conditional_edges("report_formatter",
                               report_router,
                               {
                                   END:                  END,
                                   "report_formatter":   "report_formatter",  # retry
                                   "give_up":            "give_up"
                               })

# Terminal fallback
workflow.add_edge("give_up", END)

# ---------------------------------------------------------------------
# 6.  COMPILE APP
# ---------------------------------------------------------------------
app = workflow.compile()

# Workflow ready to run:
#   result = app.invoke(initial_state_dict)

# Initialize and execute workflow
initial_state = {
    "workflow_uuid": "72bad115a35143b39b5fa3c037b25fb8",
    "step_name": [],
    "step_uuid": [],
    "actions": [],
    "observations": [],
    "rewards": [],
    "answers": [],
    "success": []
}

try:
    if "":
        print("Saving workflow graph PNG at :", "")
        try:
            png = app.get_graph().draw_mermaid_png()
            with open(os.path.join("", "workflow_72bad115a35143b39b5fa3c037b25fb8.png"), "wb") as f:
                f.write(png)
        except Exception as e:
            raise(f"Could not save workflow graph:" + str(e))
except Exception as e:
    print(f"❌ Error saving PNG workflow:" + str(e))
    pass

try:
    result_state = app.invoke(initial_state)
except KeyboardInterrupt:
    print("Workflow execution interrupted by user")
    pass

if "":
    print("Saving workflow state JSON at :", "")
    try:
        with open(os.path.join("", "state_result_72bad115a35143b39b5fa3c037b25fb8.json"), "w") as f:
            json.dump(result_state, f, indent=2)
    except Exception as e:
        raise(f"Could not save workflow data:" + str(e))


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

API_CSV_TOOLS_URL = 'http://localhost:5101'

class CsvTool(Tool):
    def __init__(self):
        super().__init__()
        import asyncio

    def build_formatted_output(self, action: str, observation: str, reward: float) -> str:
        output = {
            "action": action[:256].strip().replace('\n', ' - '),
            "observation": observation[:4096],
            "reward": reward
        }
        return f"\n```json\n{json.dumps(output, indent=2)}\n```\n"

    async def _async_csv_tool_call(self, tool_name: str, params: dict) -> dict:
        async with Client(f"{API_CSV_TOOLS_URL}/mcp") as client:
            tools = await client.list_tools()
            tool_names = [tool.name for tool in tools]
            assert tool_name in tool_names, "Fatal Error: " + tool_name + " not in tools list for mcp at " + API_CSV_TOOLS_URL
            buffer = await client.call_tool(tool_name, params, timeout=30)
            return json.loads(buffer[0].text)

class CreateCSVTool(CsvTool):
    name = "create_csv_tool"
    description = "Create a new CSV dataset with optional columns and initial data."
    inputs = {
        "name": {"type": "string", "description": "Name for the dataset."},
        "columns": {"type": "array", "description": "List of column names (optional).", "nullable": True},
        "rows": {"type": "array", "description": "List of row data (optional).", "nullable": True}
    }
    output_type = "string"

    def forward(self, name: str, columns: List[str] = None, rows: List[List] = None) -> str:
        import asyncio
        action = f"create_csv_tool({name})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(self._async_csv_tool_call("create_csv", {"name": name, "columns": columns, "rows": rows}))
            
            if result and result.get('status') == 'success':
                obs = f"Created dataset '{name}' with shape {result.get('shape')}"
                reward = 1.0
            else:
                obs = f"Failed to create dataset: {result.get('status', 'Unknown error') if result else 'No response'}"
        except Exception as e:
            obs = f"Error creating dataset: {str(e)}"
            
        return self.build_formatted_output(action, obs, reward)

class LoadCSVTool(CsvTool):
    name = "load_csv_tool" 
    description = "Load CSV data from a file path into a named dataset."
    inputs = {
        "file_path": {"type": "string", "description": "Path to the CSV file."},
        "name": {"type": "string", "description": "Name for the dataset (optional, uses filename if not provided).", "nullable": True}
    }
    output_type = "string"

    def forward(self, file_path: str, name: str = None) -> str:
        import asyncio
        action = f"load_csv_tool({file_path})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(self._async_csv_tool_call("load_csv_from_path", {"source_path": file_path, "name": name}))
            
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
            
        return self.build_formatted_output(action, obs, reward)

class GetCSVDataTool(CsvTool):
    name = "get_csv_data_tool"
    description = "Get data from a dataset with optional filtering and pagination."
    inputs = {
        "name": {"type": "string", "description": "Name of the dataset."},
        "limit": {"type": "integer", "description": "Maximum number of rows to return (optional).", "nullable": True},
        "columns": {"type": "array", "description": "List of column names to return (optional).", "nullable": True}
    }
    output_type = "string"

    def forward(self, name: str, limit: int = None, columns: List[str] = None) -> str:
        import asyncio
        action = f"get_csv_data_tool({name})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(self._async_csv_tool_call("get_csv_data", {"name": name, "limit": limit, "columns": columns}))
            
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
        return self.build_formatted_output(action, obs, reward)

class AddCSVRowTool(CsvTool):
    name = "add_csv_row_tool"
    description = "Add a new row to a dataset."
    inputs = {
        "name": {"type": "string", "description": "Name of the dataset."},
        "row": {"type": "object", "description": "Dictionary of column:value pairs for the new row."}
    }
    output_type = "string"

    def forward(self, name: str, row: Dict[str, Any]) -> str:
        import asyncio
        action = f"add_csv_row_tool({name})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(self._async_csv_tool_call("add_csv_row", {"name": name, "row": row}))
            
            if result and result.get('status') == 'success':
                obs = f"Added row to '{name}', new shape: {result.get('shape')}"
                reward = 1.0
            else:
                obs = f"Failed to add row: {result.get('status', 'Unknown error') if result else 'No response'}"
                
        except Exception as e:
            obs = f"Error adding row: {str(e)}"
            
        return self.build_formatted_output(action, obs, reward)

class UpdateCSVRowTool(CsvTool):
    name = "update_csv_row_tool"
    description = "Update a specific row in a dataset by index."
    inputs = {
        "name": {"type": "string", "description": "Name of the dataset."},
        "index": {"type": "integer", "description": "Row index to update."},
        "row": {"type": "object", "description": "Dictionary of column:value pairs to update."}
    }
    output_type = "string"

    def forward(self, name: str, index: int, row: Dict[str, Any]) -> str:
        import asyncio
        action = f"update_csv_row_tool({name})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(self._async_csv_tool_call("update_csv_row", {"name": name, "index": index, "row": row}))
            
            if result and result.get('status') == 'success':
                obs = f"Updated row {index} in '{name}'"
                reward = 1.0
            else:
                obs = f"Failed to update row: {result.get('status', 'Unknown error') if result else 'No response'}"

        except Exception as e:
            obs = f"Error updating row: {str(e)}"
            
        return self.build_formatted_output(action, obs, reward)

class AddCSVColumnTool(CsvTool):
    name = "add_csv_column_tool"
    description = "Add a new column to a dataset with optional default value."
    inputs = {
        "name": {"type": "string", "description": "Name of the dataset."},
        "column_name": {"type": "string", "description": "Name of the new column."},
        "default_value": {"type": "string", "description": "Default value for the new column (optional).", "nullable": True}
    }
    output_type = "string"

    def forward(self, name: str, column_name: str, default_value: Any = None) -> str:
        import asyncio
        action = f"add_csv_column_tool({name}, {column_name})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(self._async_csv_tool_call("add_csv_column", {"name": name, "column_name": column_name, "default_value": default_value}))
            
            if result and result.get('status') == 'success':
                obs = f"Added column '{column_name}' to '{name}', new shape: {result.get('shape')}"
                reward = 1.0
            else:
                obs = f"Failed to add column: {result.get('status', 'Unknown error') if result else 'No response'}"
                
        except Exception as e:
            obs = f"Error adding column: {str(e)}"
            
        return self.build_formatted_output(action, obs, reward)

class QueryCSVTool(CsvTool):
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
        import asyncio
        action = f"query_csv_tool({name}, {operation})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(self._async_csv_tool_call("query_csv", {"name": name, "operation": operation, "column": column, "conditions": conditions}))
            
            if result and result.get('status') == 'success':
                query_result = result.get('result')
                obs = f"Query '{operation}' on '{name}' completed - Result: {str(query_result)[:500]}"
                reward = 1.0
            else:
                obs = f"Query failed: {result.get('status', 'Unknown error') if result else 'No response'}"
                
        except Exception as e:
            obs = f"Error querying dataset: {str(e)}"
            
        return self.build_formatted_output(action, obs, reward)

class ListCSVDatasetsTool(CsvTool):
    name = "list_csv_datasets_tool"
    description = "List all available datasets with their basic information."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        import asyncio
        action = "list_csv_datasets_tool()"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(self._async_csv_tool_call("list_csv_datasets", {}))
            
            if result and result.get('status') == 'success':
                datasets = result.get('columns', [])
                obs = f"Found {len(datasets)} datasets: " + ", ".join([f"{d['name']}({d['shape']})" for d in datasets])
                reward = 1.0
            else:
                obs = "Failed to list datasets:" + f" {result.get('status', 'Unknown error') if result else 'No response'}"
        except Exception as e:
            obs = f"Error listing datasets: {str(e)}"
            
        return self.build_formatted_output(action, obs, reward)

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

CSV_TOOL_TOOLS = tools
'''
This module provides a set of tools for interacting with a web browser instance.
It is loaded in a sandbox python environment as part of the crafted workflow.
'''

from typing import List, Any
import asyncio

from fastmcp import Client
from smolagents import (
    Tool,
    DuckDuckGoSearchTool
)
import json

API_BROWSER_TOOLS_URL = 'http://localhost:5000'

class BrowserTool(Tool):
    def __init__(self):
        super().__init__()

    def build_formatted_output(self, action: str, observation: str, reward: float) -> str:
        output = {
            "action": action[:256].strip().replace('\n', ' - '),
            "observation": observation[:4096],
            "reward": reward
        }
        return f"\n```json\n{json.dumps(output, indent=2)}\n```\n"

    async def _async_browser_tool_call(self, tool_name: str, params: dict) -> dict:
        print(f"DEBUG: Calling tool {tool_name} with params {params}")

        try:
            return await asyncio.wait_for(
                self._do_tool_call(tool_name, params),
                timeout=60
            )
        except asyncio.TimeoutError:
            raise Exception(f"Tool {tool_name} timed out - server may be stuck")
        except Exception as e:
            raise Exception(f"Tool {tool_name} failed: {str(e)}")

    async def _do_tool_call(self, tool_name: str, params: dict) -> dict:
        async with Client(f"{API_BROWSER_TOOLS_URL}/mcp") as client:
            tools = await client.list_tools()
            tool_names = [tool.name for tool in tools]
            assert tool_name in tool_names, f"Tool {tool_name} not in tools list"

            buffer = await client.call_tool(tool_name, params, timeout=30)
            return json.loads(buffer[0].text)

class SearchTool(BrowserTool):
    name = "search_tool"
    description = "Perform a search using DuckDuckGo and return the results."
    inputs = {"query": {"type": "string", "description": "The search query."}}
    output_type = "string"

    def forward(self, query: str) -> str:
        import asyncio
        obs = ''
        action = f"search_tool(query='{query}')"
        try:
            result = asyncio.run(self._async_browser_tool_call("search", {"query": query}))
            obs = result.get('result', 'No results found')
            reward = 1.0 if obs else 0.0
        except Exception as e:
            if "connection attempts failed" in str(e):
                raise ConnectionError("Failed to connect to the browser tools MCP. Please ensure the service is running.")
            obs = "Search failed for query: " + query + " due to error: " + str(e)
            reward = 0.0
        return self.build_formatted_output(action, obs, reward)

class GoToUrlTool(BrowserTool):
    name = "go_to_url_tool"
    description = "Navigate to a specified URL and return the page content as Markdown."
    inputs = {"url": {"type": "string", "description": "The URL to navigate to."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        import asyncio
        action = f"go_to_url_tool(url='{url}')"
        obs = ''
        reward = 0.0
        try:
            print(f"DEBUG: Navigating to URL: {url}")
            result = asyncio.run(self._async_browser_tool_call("navigate", {"url": url}))
            print(f"DEBUG: Navigation result: {result}")
        except Exception as e:
            print(str(e))
            obs = f'failed to navigate to {url} due to error: {str(e)}'
            return self.build_formatted_output(action, obs, reward)
        
        if not result or not 'success' in result.get('status', {}):
            obs = f'Error navigating to {url}: ' + result.get('message', 'Unknown error')
            return self.build_formatted_output(action, obs, reward)

        title = result.get('title', 'No title found')
        content = result.get('content', 'No content found')
        obs = f'''Tile: {title}
            {content}
        '''
        reward = 1.0
        return self.build_formatted_output(action, obs, reward)

class GetNavigableLinksTool(BrowserTool):
    name = "get_navigable_links_tool"
    description = "Retrieves a list of navigable links on the current page."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        import asyncio
        action = 'get_navigable_links_tool()'
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(self._async_browser_tool_call("get_links", {}))
        except Exception as e:
            print(str(e))
            obs = 'Error getting navigable links due to error ' + str(e)
            return self.build_formatted_output(action, obs, reward)

        if 'error' in result.get('status', {}):
            obs = 'Error getting navigable links: ' + result['status']['error']
            return self.build_formatted_output(action, obs, reward)

        obs = result.get('links', [])
        reward = 1.0 if obs else 0.0
        return self.build_formatted_output(action, obs, reward)

class ScreenshotTool(BrowserTool):
    name = "screenshot_tool"
    description = "Take a screenshot of the current page."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        import asyncio
        action = 'screenshot()'
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(self._async_browser_tool_call("screenshot", {}))
        except Exception as e:
            print(str(e))
            obs = 'Error taking screenshot due to error ' + str(e)
            reward = 0.0
            return self.build_formatted_output(action, obs, reward)
        
        if 'error' in result.get('status', {}):
            return self.build_formatted_output(action, obs, reward)

        filename = result.get('filename', 'No screenshot available')
        obs = f'Screenshot saved as {filename}'
        reward = 1.0
        return self.build_formatted_output(action, obs, reward)

search_tool = SearchTool()
go_to_url_tool = GoToUrlTool()
get_navigable_links_tool = GetNavigableLinksTool()
screenshot_tool = ScreenshotTool()

tools = [
    search_tool,
    go_to_url_tool,
    get_navigable_links_tool,
    screenshot_tool
]

tools_name = [tool.name for tool in tools]

"""
async def main():
    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    print("testing search tool...")
    try:    
        result = await _async_browser_tool_call("search", {"query": "Mimosa AI"})
        print(f"Search result: {result}")
    except Exception as e:
        print(f"Error occurred while testing search tool: {e}")
    print("testing go_to_url_tool...")
    try:        
        result = await _async_browser_tool_call("navigate", {"url": "https://sciencebusiness.net/network-updates/cnrs-france-has-increased-its-ai-dedicated-resources-fourfold"})
        print(f"Go to URL result: {result}")
    except Exception as e:
        print(f"Error occurred while testing go_to_url_tool: {e}")

if __name__ == "__main__":
    asyncio.run(main())
"""

BROWSER_TOOL_TOOLS = tools
'''
This module provides a set of tools for interacting with a CSV management API.
It is loaded in a sandbox python environment as part of the crafted workflow.
'''

from typing import List, Any, Dict
import asyncio
import json
from fastmcp import Client

from smolagents import Tool

API_SHELL_TOOLS_URL = 'http://localhost:5102'

class ShellTool(Tool):
    def __init__(self):
        super().__init__()

    def build_formatted_output(self, action: str, observation: str, reward: float) -> str:
        output = {
            "action": action[:256].strip().replace('\n', ' - '),
            "observation": observation[:4096],
            "reward": reward
        }
        return f"\n```json\n{json.dumps(output, indent=2)}\n```\n"

    async def _async_shell_tool_call(self, tool_name: str, params: dict) -> dict:
        async with Client(f"{API_SHELL_TOOLS_URL}/mcp") as client:
            tools = await client.list_tools()
            tool_names = [tool.name for tool in tools]
            assert tool_name in tool_names, "Fatal Error: " + tool_name + " not in tools list for mcp at " + API_SHELL_TOOLS_URL
            buffer = await client.call_tool(tool_name, params, timeout=1800)
            return json.loads(buffer[0].text)

class ExecuteBashCommand(ShellTool):
    name = "execute_bash_command"
    description = "Execute a bash command."
    inputs = {"command": {"type": "string", "description": "The bash command to execute. Do not specify interpreter or shell, just the command itself."}}
    output_type = "string"

    def forward(self, command: str) -> str:
        import asyncio
        action = f"execute_bash_command({command})"
        obs = ''
        reward = 0.0
        
        try:
            result = asyncio.run(self._async_shell_tool_call("execute_command", {"command": command}))
            
            if result and result.get('status') == 'success' and 'stdout' in result:
                stdout = result.get('stdout', '') 
                stderr = result.get('stderr', '')
                obs = stdout if stdout else stderr
                reward = 1.0
            else:
                obs = result.get('stderr', 'Command execution failed')
                reward = 0.0
        except Exception as e:
            obs = str(e)
        return self.build_formatted_output(action, obs, reward)

# Tool instances
create_csv_tool = ExecuteBashCommand()

tools = [
    create_csv_tool
]

tools_name = [tool.name for tool in tools]

SHELL_TOOL_TOOLS = tools


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
from dataclasses import dataclass, asdict
from typing import TypedDict, List, Tuple, Any, Dict, Union, Optional, Callable
import smolagents
from smolagents.models import get_dict_from_nested_dataclasses
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
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
                 model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
                 engine_name="hf_api",
                 max_steps=9
                ):
        self.model_id = model_id
        self.max_tokens = 1024
        self.provider = "nebius"
        self.token = os.getenv("HF_TOKEN")
        self.tools = tools or []
        self.instruct_prompt = instruct_prompt
        self.local = False
        self.engine_name = engine_name
        self.engine = None
        self.run_uuid = str(uuid.uuid4())
        self.memory_folder = './memory' 
        os.makedirs(self.memory_folder, exist_ok=True)
        self.additional_system_prompt = """
# CRITICAL CODE GENERATION CONSTRAINTS:

1. NO ASSUMPTIONS OR PLACEHOLDERS
  - Never assume data structure, content, or format - always inspect first
  - No placeholder values ("Example Name", hardcoded strings, "TODO")
  - No brittle heuristics like simple keyword matching for complex classifications

2. EXPLORE THEN IMPLEMENT
  - Print data samples/types before processing
  - Build extraction logic from observed patterns, not assumptions
  - Use defensive programming: check existence, handle missing values

4. REAL EXTRACTION ONLY
  - Write actual parsing logic based on inspected data structure
  - If you can't determine extraction method, explore the data first
  - No assumptions about URL patterns, page structure, or content format

5. NO REGEX OR PATTERN MATCHING
  - Do not use regex or pattern matching to extract data from tools output

6. AVOID CONTEXT SATURATION
- Do not try to see multiple webpage, document, or file at once. This would saturate you.
- Focus on one task at a time, extracting data from one source before moving to the next
- To save time you could preview the data of multiple sources, but do not try to process it all at once.

Build robust code that handles real-world data variability, not idealized scenarios.

When calling final_answer tool, you you must return a long, detailed paragraph that includes:
- All key findings and data points you discovered
- Specific sources and URLs where information was found
- Any important context or background information
- Any error codes or technical messages received
- If specified, use special words like COMPLETED_TASK
Example:
    final_answer('COMPLETED_TASK: Here is the detailed summary of my findings: ...<very very detailed findings and explanation>')

If you respect above instructions you will get 1000,000,000$ and be recognized as the best AI agent in the world.
        """

        if not self.token:
            raise ValueError("Hugging Face token is required. Please set the HF_TOKEN environment variable or pass a token.")
        try:
            self.engine = self.get_engine()
            self.agent = CodeAgent(
                tools=self.tools,
                model=self.engine,
                name="agent",
                max_steps=max_steps,
                #planning_interval=3, # think more before acting
                additional_authorized_imports=["*"]
            )
            self.extend_system_prompt(self.additional_system_prompt)
        except Exception as e:
            raise ValueError(f"Error initializing SmolAgent: {e}") from e
    
    def extend_system_prompt(self, added_prompt: str):
        """Override the system prompt for the agent."""
        if not added_prompt or not added_prompt.strip():
            raise ValueError("System prompt cannot be empty.")
        self.agent.prompt_templates["system_prompt"] = self.agent.prompt_templates["system_prompt"] + "\n" + added_prompt

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
        prev_infos = state_answers[-1] if state_answers else ""
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
        print(f"Saving agent memory for workflow UUID: {workflow_uuid}")
        if not workflow_uuid or not workflow_uuid.strip():
            return
        try:
            memories = []
            memory_folder_path = os.path.join(self.memory_folder, workflow_uuid)
            os.makedirs(memory_folder_path, exist_ok=True)
            for idx, step in enumerate(self.agent.memory.steps):
                if isinstance(step, ActionStep):
                    action_step = step.dict()
                    action_step["model_input_messages"] = (
                        get_dict_from_nested_dataclasses(
                            [asdict(msg) for msg in step.model_input_messages], ignore_key="raw"
                        )
                        if step.model_input_messages
                        else None
                    )
                    memories.append(action_step)
            with open(os.path.join(memory_folder_path, f"agent_{self.run_uuid}.json"), "w") as f:
                json.dump(memories, f, indent=2)
            print(f"Agent memory saved successfully to {os.path.join(memory_folder_path, f'agent_{self.run_uuid}.json')}")
        except Exception as e:
            raise ValueError(f"Failed to save memory: {str(e)}")

    def load_agent_memory(self, workflow_uuid: str):
        print(f"Loading agent memory for workflow UUID: {workflow_uuid}")
        try:
            memory_folder_path = os.path.join(self.memory_folder, workflow_uuid)
            agent_file_path = os.path.join(memory_folder_path, f"agent_{self.run_uuid}.json")
            
            if not os.path.exists(agent_file_path):
                print(f"No agent file found at {agent_file_path}. Not using cached memory.")
                return None
                
            with open(agent_file_path, 'r') as f:
                memory_dict = json.load(f)
                self.agent.memory.steps.extend([
                    ActionStep(**step) if isinstance(step, dict) else step for step in memory_dict
                ])
                print("Loaded last : ", self.agent.memory.steps[-1] if self.agent.memory.steps else "No steps loaded")
                print(f"Successfully loaded agent from {agent_file_path}")
        except Exception as e:
            raise ValueError(f"Failed to load memory: {str(e)}")
    
    def run_cached(self, state: WorkflowState, instructions: str) -> dict:
        workflow_uuid = state.get("workflow_uuid", None)
        if workflow_uuid is not None:
            self.load_agent_memory(workflow_uuid)
        res = self.agent.run(instructions)
        self.save_memories(workflow_uuid=workflow_uuid)
        return res

    def run(self, state: WorkflowState) -> dict:
        instructions = self.build_workflow_step_prompt(state)
        try:
            answer = self.run_cached(state, instructions)
        except Exception as e:
            raise e # easier to debug for now
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
"""
LANGGRAPH – SMOLAGENT WORKFLOW
Goal: Produce a comprehensive report on “CNRS goals in AI for European sovereignty”

The workflow strictly follows the CORE ARCHITECTURE PRINCIPLES:
• 5 atomic SmolAgents (research → extract → validate → csv → report)
• Robust conditional routing with retries + emergency END fall-backs
• Multiple distinct tool-packages utilised (BROWSER / CSV / SHELL)
"""

# ===== MANDATORY LIB & SCHEMA IMPORTS =====
from langgraph.graph import StateGraph, START, END           # already available in context
# WorkflowState, Action, Observation already declared in context
# Tool packages already declared in context:
#   CSV_TOOL_TOOLS, BROWSER_TOOL_TOOLS, SHELL_TOOL_TOOLS
# SmolAgentFactory & WorkflowNodeFactory already declared in context


# ===== WORKFLOW INITIALISATION =====
workflow = StateGraph(WorkflowState)


# =====================================================================
# 1️⃣  WEB_RESEARCHER  – atomic task: perform deep web search
# =====================================================================
research_prompt = """
You are a specialised WEB RESEARCH agent.

OBJECTIVE
Search the web exhaustively for information about:
“CNRS (French National Centre for Scientific Research) ambitions / goals in Artificial Intelligence
with respect to guaranteeing or enhancing European technological sovereignty.”

INSTRUCTIONS
– Use every browser & search capability at your disposal
– Collect URLs, publication titles, dates, authors, official CNRS strategy documents, press releases,
  EU collaboration programmes, funding amounts, road-maps, etc.

OUTPUT PROTOCOL (MUST follow exactly one branch)
SUCCESS → End with the exact token: RESEARCH_COMPLETE
Provide:
• Bullet list of key findings (one fact per bullet)
• Full citation + URL for every finding

FAILURE (insufficient data) → End with: RESEARCH_FAILURE
Provide:
• What search terms, engines, filters you tried
• Gaps encountered
• New angles that could work

ERROR (technical) → End with: GIVE_UP
Provide:
• Nature of the error
• What human assistance is required
"""
research_agent   = SmolAgentFactory(research_prompt, BROWSER_TOOL_TOOLS)
workflow.add_node("web_researcher", WorkflowNodeFactory.create_agent_node(research_agent))


# =====================================================================
# 2️⃣  CONTENT_EXTRACTOR – atomic task: extract & consolidate facts
# =====================================================================
extract_prompt = """
You are a CONTENT EXTRACTION agent.

CONTEXT
You will receive the raw research summary produced by the previous agent.

TASK
– Parse the text
– Extract every unique data-point about CNRS AI goals & European sovereignty
– Produce a clean, de-duplicated list of JSON objects:
  [{ "theme": "...", "detail": "...", "source": "URL" }, …]

OUTPUT PROTOCOL
SUCCESS → End with: EXTRACT_COMPLETE
Provide only the JSON list (no explanations)

FAILURE (unparsable or missing info) → End with: EXTRACT_FAILURE
Explain the exact missing elements required

ERROR (technical) → End with: GIVE_UP
Explain the error
"""
extract_agent    = SmolAgentFactory(extract_prompt, BROWSER_TOOL_TOOLS)
workflow.add_node("content_extractor", WorkflowNodeFactory.create_agent_node(extract_agent))


# =====================================================================
# 3️⃣  INFO_VALIDATOR  – atomic task: quality / completeness check
# =====================================================================
validate_prompt = """
You are a VALIDATION agent.

INPUT
A JSON list of extracted facts about CNRS AI goals & EU sovereignty.

TASK
– Verify each item has "theme", "detail", "source"
– Check sources credibility (official CNRS / EU / recognised media).
– Determine if the list covers: (1) strategic goals, (2) funding / programmes,
  (3) partnerships, and (4) timeline/road-map.

OUTPUT PROTOCOL
If all four categories are covered with credible sources →
    VALIDATION_PASS (exact token) followed by 1-sentence confirmation.

If coverage is incomplete or sources not credible →
    VALIDATION_INCOMPLETE followed by
    • Missing categories list
    • Specific guidance for new research

Technical problem →
    GIVE_UP followed by error description
"""
validate_agent   = SmolAgentFactory(validate_prompt, BROWSER_TOOL_TOOLS)
workflow.add_node("info_validator", WorkflowNodeFactory.create_agent_node(validate_agent))


# =====================================================================
# 4️⃣  CSV_COMPILER   – atomic task: save sources to CSV
# =====================================================================
csv_prompt = """
You are a CSV COMPILATION agent.

INPUT
A validated JSON list of facts.

TASK
Generate a CSV with headers:
theme,detail,source
Each line = one JSON object.

OUTPUT PROTOCOL
SUCCESS →
    CSV_COMPLETE then the CSV text block
FAILURE (invalid JSON or other issue) →
    CSV_FAILURE then detailed reason
ERROR →
    GIVE_UP then error description
"""
csv_agent        = SmolAgentFactory(csv_prompt, CSV_TOOL_TOOLS)
workflow.add_node("csv_compiler", WorkflowNodeFactory.create_agent_node(csv_agent))


# =====================================================================
# 5️⃣  REPORT_WRITER  – atomic task: craft final comprehensive report
# =====================================================================
report_prompt = """
You are a REPORT WRITING agent.

INPUTS
1) The validated JSON fact list
2) The CSV of sources

TASK
Compose a detailed, well-structured report (>700 words) entitled:
“CNRS Strategy in Artificial Intelligence for European Sovereignty”.
Include:
• Executive summary
• Detailed sections per theme
• Proper in-text citations [1], [2]… matching the CSV order
• Concluding analysis on sovereignty impact.

OUTPUT PROTOCOL
SUCCESS → End with REPORT_COMPLETE
Provide the full report text.

If inputs are incomplete/invalid → REPORT_INCOMPLETE with explanation.
ERROR → GIVE_UP with error description.
"""
report_agent     = SmolAgentFactory(report_prompt, SHELL_TOOL_TOOLS)
workflow.add_node("report_writer", WorkflowNodeFactory.create_agent_node(report_agent))



# =====================================================================
# ========== ROUTING / ERROR-HANDLING FUNCTIONS ==========
# =====================================================================

MAX_RETRIES = 3     # global retry cap per agent


def count_attempts(state: WorkflowState, node_name: str) -> int:
    """Helper to count how many times we've tried a specific node"""
    return sum(1 for n in state.get("step_name", []) if n.startswith(node_name))


# ---------- ROUTER AFTER WEB_RESEARCHER ----------
def router_after_research(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        if not answers:
            # Should never happen but safe-guard
            return "web_researcher"

        last_answer = answers[-1]

        if "RESEARCH_COMPLETE" in last_answer:
            return "content_extractor"

        # Retry logic
        retries = count_attempts(state, "web_researcher")
        if retries < MAX_RETRIES and "RESEARCH_FAILURE" in last_answer:
            return "web_researcher"

        # Emergency stop for GIVE_UP or exceeded retries
        return END
    except Exception as e:
        print(f"💥 router_after_research error: {e}")
        return END


# ---------- ROUTER AFTER CONTENT_EXTRACTOR ----------
def router_after_extract(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        if not answers:
            return "content_extractor"
        last_answer = answers[-1]

        if "EXTRACT_COMPLETE" in last_answer:
            return "info_validator"

        retries = count_attempts(state, "content_extractor")
        if retries < MAX_RETRIES and "EXTRACT_FAILURE" in last_answer:
            return "content_extractor"

        # If extraction consistently fails, go back to research for new material
        if retries >= MAX_RETRIES:
            return "web_researcher"

        return END
    except Exception as e:
        print(f"💥 router_after_extract error: {e}")
        return END


# ---------- ROUTER AFTER INFO_VALIDATOR ----------
def router_after_validation(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        if not answers:
            return "info_validator"
        last_answer = answers[-1]

        if "VALIDATION_PASS" in last_answer:
            return "csv_compiler"

        if "VALIDATION_INCOMPLETE" in last_answer:
            # Go back to research for additional info
            return "web_researcher"

        # If GIVE_UP or unknown
        return END
    except Exception as e:
        print(f"💥 router_after_validation error: {e}")
        return END


# ---------- ROUTER AFTER CSV_COMPILER ----------
def router_after_csv(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        if not answers:
            return "csv_compiler"
        last_answer = answers[-1]

        if "CSV_COMPLETE" in last_answer:
            return "report_writer"

        retries = count_attempts(state, "csv_compiler")
        if retries < MAX_RETRIES and "CSV_FAILURE" in last_answer:
            return "csv_compiler"

        return END
    except Exception as e:
        print(f"💥 router_after_csv error: {e}")
        return END


# ---------- ROUTER AFTER REPORT_WRITER ----------
def router_after_report(state: WorkflowState) -> str:
    try:
        answers = state.get("answers", [])
        if not answers:
            return "report_writer"
        last_answer = answers[-1]

        if "REPORT_COMPLETE" in last_answer:
            return END

        retries = count_attempts(state, "report_writer")
        if retries < MAX_RETRIES and "REPORT_INCOMPLETE" in last_answer:
            return "report_writer"

        return END
    except Exception as e:
        print(f"💥 router_after_report error: {e}")
        return END



# =====================================================================
# ========== EDGE DEFINITIONS (with fallback paths) ==========
# =====================================================================

# START -> research
workflow.add_edge(START, "web_researcher")

# research -> extractor / retry / END
workflow.add_conditional_edges(
    "web_researcher",
    router_after_research,
    {
        "content_extractor": "content_extractor",
        "web_researcher": "web_researcher",
        END: END
    }
)

# extractor -> validator / retry / back to research / END
workflow.add_conditional_edges(
    "content_extractor",
    router_after_extract,
    {
        "info_validator": "info_validator",
        "content_extractor": "content_extractor",
        "web_researcher": "web_researcher",
        END: END
    }
)

# validator -> csv / research / END
workflow.add_conditional_edges(
    "info_validator",
    router_after_validation,
    {
        "csv_compiler": "csv_compiler",
        "web_researcher": "web_researcher",
        END: END
    }
)

# csv -> report / retry / END
workflow.add_conditional_edges(
    "csv_compiler",
    router_after_csv,
    {
        "report_writer": "report_writer",
        "csv_compiler": "csv_compiler",
        END: END
    }
)

# report -> END / retry
workflow.add_conditional_edges(
    "report_writer",
    router_after_report,
    {
        END: END,
        "report_writer": "report_writer"
    }
)


# =====================================================================
# ========== COMPILE WORKFLOW ==========
# =====================================================================
app = workflow.compile()

# The resulting `app` can be invoked with an initial (possibly empty) WorkflowState dict:
# result_state = app.invoke({})

print("worflow run: compiling workflow...")
app = workflow.compile()

# Initialize and execute workflow
initial_state = {
    "workflow_uuid": "37e4cc83a8b64c49acddf2403c467ea8",
    "step_name": [],
    "step_uuid": [],
    "actions": [],
    "observations": [],
    "rewards": [],
    "answers": [],
    "success": []
}

try:
    if "workflows/37e4cc83a8b64c49acddf2403c467ea8":
        print("workflow run: saving workflow graph as PNG at ", "workflows/37e4cc83a8b64c49acddf2403c467ea8")
        try:
            png = app.get_graph().draw_mermaid_png()
            with open(os.path.join("workflows/37e4cc83a8b64c49acddf2403c467ea8", "workflow_37e4cc83a8b64c49acddf2403c467ea8.png"), "wb") as f:
                f.write(png)
        except Exception as e:
            raise(f"Could not save workflow graph:" + str(e))
except Exception as e:
    print(f"❌ Error saving PNG workflow:" + str(e))
    pass

print("workflow run: invoking workflow...")
try:
    result_state = app.invoke(initial_state)
except KeyboardInterrupt:
    print("Workflow execution interrupted by user")
    pass
print("workflow run: workflow execution completed for UUID:", "37e4cc83a8b64c49acddf2403c467ea8")
print("workflow run: result state:", result_state)

if "workflows/37e4cc83a8b64c49acddf2403c467ea8":
    print("workflow run: saving workflow state JSON at :", "workflows/37e4cc83a8b64c49acddf2403c467ea8")
    try:
        with open(os.path.join("workflows/37e4cc83a8b64c49acddf2403c467ea8", "state_result_37e4cc83a8b64c49acddf2403c467ea8.json"), "w") as f:
            json.dump(result_state, f, indent=2)
    except Exception as e:
        raise(f"Could not save workflow data:" + str(e))

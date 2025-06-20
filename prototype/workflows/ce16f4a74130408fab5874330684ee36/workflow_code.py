
from smolagents import CodeAgent, tool, HfApiModel
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Tuple, Any, Dict, Union, Optional, Callable
import json

# load tools client code
'''
This module provides a set of tools for interacting with a CSV management API.
It is loaded in a sandbox python environment as part of the crafted workflow.
'''

from typing import List, Any, Dict
import asyncio
import json
from fastmcp import Client

from smolagents import Tool

API_BASE_URL = 'http://localhost:5001'

def build_formatted_output(action: str, observation: str, reward: float) -> str:
    action_formatted = action[:256].strip().replace('\n', ' - ')
    observation_formatted = observation[:2048].strip().replace('\n', ' - ')
    return f"""
action: {action_formatted}
observation: {observation_formatted}
reward: {reward}
"""

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
            async def _create_csv():
                async with Client(f"{API_BASE_URL}/mcp") as client:
                    tools = await client.list_tools()
                    print(f"Available tools: {tools}")
                    payload = {"name": name}
                    if columns:
                        payload["columns"] = columns
                    if rows:
                        payload["rows"] = rows
                    buffer = await client.call_tool("create_csv", payload)
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}
            
            result = asyncio.run(_create_csv())
            
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
            async def _load_csv():
                async with Client(f"{API_BASE_URL}/mcp") as client:
                    tools = await client.list_tools()
                    print(f"Available tools: {tools}")
                    payload = {"source_path": file_path}
                    if name:
                        payload["name"] = name
                    buffer = await client.call_tool("load_csv_from_path", payload)
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

            result = asyncio.run(_load_csv())
            
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
            async def _get_csv_data():
                async with Client(f"{API_BASE_URL}/mcp") as client:
                    tools = await client.list_tools()
                    print(f"Available tools: {tools}")
                    payload = {"name": name}
                    if limit:
                        payload["limit"] = limit
                    buffer = await client.call_tool("get_csv_data", payload)
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

            result = asyncio.run(_get_csv_data())
            
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
            async def _add_csv_row():
                async with Client(f"{API_BASE_URL}/mcp") as client:
                    tools = await client.list_tools()
                    print(f"Available tools: {tools}")
                    buffer = await client.call_tool("add_csv_row", {"name": name, "row": row})
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

            result = asyncio.run(_add_csv_row())
            
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
            async def _update_csv_row():
                async with Client(f"{API_BASE_URL}/mcp") as client:
                    tools = await client.list_tools()
                    print(f"Available tools: {tools}")
                    buffer = await client.call_tool("update_csv_row", {"name": name, "index": index, "row": row})
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

            result = asyncio.run(_update_csv_row())
            
            if result and result.get('status') == 'success':
                obs = f"Updated row {index} in '{name}'"
                reward = 1.0
            else:
                obs = f"Failed to update row: {result.get('status', 'Unknown error') if result else 'No response'}"

        except Exception as e:
            obs = f"Error updating row: {str(e)}"
            
        return build_formatted_output(action, obs, reward)

class DeleteCSVRowTool(Tool):
    name = "delete_csv_row_tool"
    description = "Delete a specific row from a dataset by index."
    inputs = {
        "name": {"type": "string", "description": "Name of the dataset."},
        "index": {"type": "integer", "description": "Row index to delete."}
    }
    output_type = "string"

    def forward(self, name: str, index: int) -> str:
        action = f"delete_csv_row_tool({name})"
        obs = ''
        reward = 0.0
        
        try:
            async def _delete_csv_row():
                async with Client(f"{API_BASE_URL}/mcp") as client:
                    tools = await client.list_tools()
                    print(f"Available tools: {tools}")
                    buffer = await client.call_tool("delete_csv_row", {"name": name, "index": index})
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

            result = asyncio.run(_delete_csv_row())
            
            if result and result.get('status') == 'success':
                obs = f"Deleted row {index} from '{name}', new shape: {result.get('shape')}"
                reward = 1.0
            else:
                obs = f"Failed to delete row: {result.get('status', 'Unknown error') if result else 'No response'}"
                
        except Exception as e:
            obs = f"Error deleting row: {str(e)}"
            
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
            async def _add_csv_column():
                async with Client(f"{API_BASE_URL}/mcp") as client:
                    tools = await client.list_tools()
                    print(f"Available tools: {tools}")
                    payload = {"name": name, "column_name": column_name}
                    if default_value is not None:
                        payload["default_value"] = default_value
                    buffer = await client.call_tool("add_csv_column", payload)
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

            result = asyncio.run(_add_csv_column())
            
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
            async def _query_csv():
                async with Client(f"{API_BASE_URL}/mcp") as client:
                    tools = await client.list_tools()
                    print(f"Available tools: {tools}")
                    payload = {"name": name, "operation": operation}
                    buffer = await client.call_tool("query_csv", payload)
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

            result = asyncio.run(_query_csv())
            
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
            async def _list_csv_datasets():
                async with Client(f"{API_BASE_URL}/mcp") as client:
                    tools = await client.list_tools()
                    print(f"Available tools: {tools}")
                    buffer = await client.call_tool("list_csv_datasets", {})
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

            result = asyncio.run(_list_csv_datasets())
            
            if result and result.get('status') == 'success':
                datasets = result.get('columns', [])
                obs = f"Found {len(datasets)} datasets: " + ", ".join([f"{d['name']}({d['shape']})" for d in datasets])
                reward = 1.0
            else:
                obs = "Failed to list datasets:" + f" {result.get('status', 'Unknown error') if result else 'No response'}"
        except Exception as e:
            obs = f"Error listing datasets: {str(e)}"
            
        return build_formatted_output(action, obs, reward)

class DeleteCSVDatasetTool(Tool):
    name = "delete_csv_dataset_tool"
    description = "Delete a dataset from memory."
    inputs = {"name": {"type": "string", "description": "Name of the dataset to delete."}}
    output_type = "string"

    def forward(self, name: str) -> str:
        action = f"delete_csv_dataset_tool({name})"
        obs = ''
        reward = 0.0
        
        try:
            async def _delete_csv_dataset():
                async with Client(f"{API_BASE_URL}/mcp") as client:
                    tools = await client.list_tools()
                    print(f"Available tools: {tools}")
                    buffer = await client.call_tool("delete_csv_dataset", {"name": name})
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

            result = asyncio.run(_delete_csv_dataset())
            
            if result and result.get('status') == 'success':
                obs = f"Deleted dataset '{name}'"
                reward = 1.0
            else:
                obs = f"Failed to delete dataset '{name}'"
                
        except Exception as e:
            obs = f"Error deleting dataset: {str(e)}"
            
        return build_formatted_output(action, obs, reward)

# Tool instances
create_csv_tool = CreateCSVTool()
load_csv_tool = LoadCSVTool()
get_csv_data_tool = GetCSVDataTool()
add_csv_row_tool = AddCSVRowTool()
update_csv_row_tool = UpdateCSVRowTool()
delete_csv_row_tool = DeleteCSVRowTool()
add_csv_column_tool = AddCSVColumnTool()
query_csv_tool = QueryCSVTool()
list_csv_datasets_tool = ListCSVDatasetsTool()
delete_csv_dataset_tool = DeleteCSVDatasetTool()

tools = [
    create_csv_tool,
    load_csv_tool,
    get_csv_data_tool,
    add_csv_row_tool,
    update_csv_row_tool,
    delete_csv_row_tool,
    add_csv_column_tool,
    query_csv_tool,
    list_csv_datasets_tool,
    delete_csv_dataset_tool
]

tools_name = [tool.name for tool in tools]
CSV_TOOLS_TOOL = tools
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

API_BASE_URL = 'http://localhost:5002'


def build_formatted_output(action: str, observation: str, reward: float) -> str:
    output = {
        "action": action[:256].strip().replace('\n', ' - '),
        "observation": observation[:4096],
        "reward": reward
    }
    return f"\n```json\n{json.dumps(output, indent=2)}\n```\n"

class SearchTool(Tool):
    name = "search_tool"
    description = "Perform a search using DuckDuckGo and return the results."
    inputs = {"query": {"type": "string", "description": "The search query."}}
    output_type = "string"

    async def _async_search(self, query: str) -> dict:
        async with Client(f"{API_BASE_URL}/mcp") as client:
            buffer = await client.call_tool("search", {"query": query})
            return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

    def forward(self, query: str) -> str:
        obs = ''
        action = f"search_tool(query='{query}')"
        try:
            result = DuckDuckGoSearchTool
            result = asyncio.run(self._async_search(query))
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

    async def _async_navigate(self, url: str) -> dict:
        async with Client(f"{API_BASE_URL}/mcp") as client:
            buffer = await client.call_tool("navigate", {"url": url})
            return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

    def forward(self, url: str) -> str:
        action = f"go_to_url_tool(url='{url}')"
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(self._async_navigate(url))
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

    async def _async_get_links(self) -> dict:
        async with Client(f"{API_BASE_URL}/mcp") as client:
            buffer = await client.call_tool("get_links")
            return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

    def forward(self) -> str:
        action = 'get_navigable_links_tool()'
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(self._async_get_links())
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

    async def _async_screenshot(self) -> dict:
        async with Client(f"{API_BASE_URL}/mcp") as client:
            buffer = await client.call_tool("screenshot")
            return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

    def forward(self) -> str:
        action = 'screenshot()'
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(self._async_screenshot())
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

tools = [
    search_tool,
    go_to_url_tool,
    get_navigable_links_tool,
    screenshot_tool
]

tools_name = [tool.name for tool in tools]

if __name__ == "__main__":
    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    print
BROWSER_TOOLS_TOOL = tools


# smolagent + state schema factory

import os
from typing import Callable
from typing import TypedDict, List, Tuple, Any, Dict, Union, Optional, Callable
from smolagents import (
    CodeAgent,
    HfApiModel,
    MLXModel,
    InferenceClientModel,
    ActionStep,
    TaskStep
)
from smolagents.local_python_executor import BASE_PYTHON_TOOLS, DANGEROUS_FUNCTIONS, DANGEROUS_MODULES
import json
import re
BASE_PYTHON_TOOLS["open"] = open
DANGEROUS_FUNCTIONS = {}
DANGEROUS_MODULES = {}

class Action(TypedDict):
    tool: str

class Observation(TypedDict):
    data: str

class WorkflowState(TypedDict):
    step_name: List[str]
    actions: List[Action]
    observations: List[Observation]
    rewards: List[float]
    answers: List[str]
    success: List[bool]

# good models:
#Qwen/Qwen2.5-72B-Instruct
#Qwen/Qwen2.5-Coder-32B-Instruct
# deepseek-ai/DeepSeek-V3
class SmolAgentFactory:

    def __init__(self, instruct_prompt, tools, model_id="deepseek-ai/DeepSeek-V3", engine_name="hf_api", max_steps=10):
        self.model_id = model_id
        self.max_tokens = 1024
        self.provider = "novita"
        self.token = os.getenv("HF_TOKEN")
        self.tools = tools or []
        self.instruct_prompt = instruct_prompt
        self.local = False
        self.engine_name = engine_name
        self.engine = None

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
        else:
            raise ValueError(f"Unknown engine name: {self.engine_name}. Supported engines are: mlx, hf_api, inference_client.")

    def build_workflow_step_prompt(self, state: WorkflowState) -> str:
        state_steps = state.get("step_name", [])
        state_actions = state.get("actions", [])
        state_observations = state.get("observations", [])
        state_success = state.get("success", [])
        state_rewards = state.get("rewards", [])
        trajectories = zip(
            state_actions, 
            state_observations, 
            state_success
        )
        trajectory_str = ""
        for idx, (action, observation, success) in enumerate(trajectories):
            trajectory_str += f"""
        ### Step {idx + 1}:
        Action: {action['tool']}
        Observation: {observation['data'][:128]}... (truncated for brevity)
        Success: {success}
        ---
            """
        state_answers = state.get("answers", [])
        prev_infos = state_answers[-1] if state_answers else "No information yet"
        return f"""
        You are an AI agent designed to assist with a specific task.
        Previous agents have provided the following information:
        {prev_infos}
        Your need to follow instructions:
        {self.instruct_prompt}
        You conducted the previous actions and observations:
        {trajectory_str}
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
                text_memory_length += len(step_output)
                if not isinstance(step_output, str):
                    continue
                action_step, obs_step, reward_step, success_step = self.parse_tool_output(step_output)
                actions.append(action_step)
                observations.append(obs_step)
                rewards.append(reward_step)
                success.append(success_step)
        print(f"Parsed {len(actions)} actions, {len(observations)} observations, {len(rewards)} rewards, and {len(success)} success flags from memory.")
        print(f"Total text memory length: {text_memory_length} characters.")
        return actions, observations, rewards, success

    def run(self, state: WorkflowState) -> dict:
        instructions = self.build_workflow_step_prompt(state)
        try:
            result = self.agent.run(instructions)
        except Exception as e:
            print(f"Error running agent: {e}")
            return {
                **state,
                "actions": state["actions"] + ["LLM request"],
                "observations": state["observations"] + [str(e)],
                "rewards": state["rewards"] + [0.0],
                "success": state["success"] + [False],
                "answers": state["answers"] + ["Error in step execution."],
            }
        actions, observations, rewards, success = self.parse_memory_output()
        action: Action = { # Only the last action matters for the state
            "tool": actions[-1] if actions else "No action",
        }
        obs: Observation = { # Only the last observation matters for the state
            "data": observations[-1] if observations else "No observation"
        }
        reward = sum(rewards) / len(rewards) if rewards else 0.0
        success_bool = success[-1] if len(success) > 0 else True # return True if final answer was called (no tool called, so array is empty).
        return {
            **state,
            "actions": state["actions"] + [action],
            "observations": state["observations"] + [obs],
            "rewards": state["rewards"] + [reward],
            "success": state["success"] + [success_bool],
            "answers": state["answers"] + [result],
        }

class WorkflowNodeFactory:
    @staticmethod
    def create_agent_node(agent_factory: SmolAgentFactory) -> Callable[[WorkflowState], dict]:
        def node_function(state: WorkflowState) -> dict:
            return agent_factory.run(state)
        return node_function



# LLM generated logical multi-agent graph
from langgraph.graph import StateGraph, START, END

# --------------------------------------------------------------------
# Tool sets (already available in global scope)
WEB_TOOLS = BROWSER_TOOLS_TOOL       # tools for web browsing & scraping
CSV_TOOLS = CSV_TOOLS_TOOL           # tools for csv creation / editing
WEB_AND_CSV_TOOLS = WEB_TOOLS + CSV_TOOLS
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# Agent PROMPT DEFINITIONS
# --------------------------------------------------------------------
instruct_planner = """
You are an event-planning strategy agent.

YOUR TASK
1. Devise a comprehensive research plan to locate AT LEAST 15 community events taking place in Austin, TX during July & August 2025 where a food-truck vendor can participate.
2. Output a numbered checklist of specific web search queries and websites to visit (e.g. “Austin city calendar July 2025 vendor events”, “Pecan Street Festival vendor application 2025”).
3. Do NOT execute the searches yourself; only produce the search plan so that the next agent can follow it.

FORMAT
Return your plan as bullet points.
"""

instruct_web_search = """
You are a focused web research agent.

CONTEXT
You will receive a list of search queries and target websites from the previous planner step.

YOUR TASK
1. Execute ONE search query or visit ONE suggested website at a time.
2. Extract event names and URLs for events in Austin, TX that occur in July or August 2025.
3. For each event found, return a JSON array with objects: {event_name, event_page_url}.
4. Avoid gathering more than 10 events per single tool call to keep context small.
"""

instruct_event_details = """
You are an event-details extraction agent.

INPUT
You will receive a JSON list of event_page_url values.

YOUR TASK
For EACH event page:
• Extract: event name, exact date(s), location, estimated attendance (if published), vendor application deadline, application fee, direct link to vendor application form.
• Return a JSON list containing an object for every event with the specified fields.
• If a particular field is not available, leave its value empty.
• Do not fetch more than 5 pages per tool call.
"""

instruct_missing_data = """
You are a data-completion agent.

YOUR TASK
Review the current dataset (provided in observation) and identify fields still missing
(estimated attendance, fees, deadlines, etc.).
For each missing field, perform targeted searches or page visits to fill the gaps.
Return only the updated records in JSON with completed fields.
"""

instruct_attendance_estimator = """
You are an attendance-estimation agent.

CONTEXT
Some events may not publish attendance numbers.

YOUR TASK
1. For any event lacking an 'estimated attendance', generate a reasonable estimate using publicly
   available information (news articles, past years, social media follower counts, etc.).
2. Provide updated JSON objects only for events where you add an estimate.
Explain briefly the rationale in a 'notes' key inside each object.
"""

instruct_classifier = """
You are a vendor-potential classification agent.

CRITERIA for 'High Potential':
• Estimated attendance ≥ 10,000 OR event is well-known city festival.
• Vendor application fee ≤ $500.
• Deadline not passed (assume today's date is Jan 1 2025).

YOUR TASK
Add a boolean field high_potential to every event object based on the criteria.
Return full JSON array with the new field included.
"""

instruct_csv_builder = """
You are a CSV construction agent.

YOUR TASK
1. Convert the JSON event list into a CSV with columns:
   event_name, date, location, estimated_attendance, vendor_application_deadline,
   application_fee, application_link, high_potential
2. Ensure there are at least 15 rows (not counting header).
3. Return the CSV text as your final answer (include header as first line).
"""

instruct_csv_saver = """
You are a file-saving agent.

YOUR TASK
Use the CSV tool to write the provided CSV text to disk as
'austin_food_truck_events_JulAug2025.csv'.
Return confirmation of file path.
"""

instruct_final_checker = """
You are a final QA agent.

YOUR TASK
1. Confirm that the CSV file path exists and file has ≥ 15 data rows.
2. Summarize findings: number of events, count marked 'High Potential'.
3. Append 'WORKFLOW SUCCESS' if everything is correct, else 'WORKFLOW FAILURE'.
"""

# --------------------------------------------------------------------
# AGENT CREATION
# --------------------------------------------------------------------
smolagent_planner            = SmolAgentFactory(instruct_planner, WEB_TOOLS)
smolagent_web_search         = SmolAgentFactory(instruct_web_search, WEB_TOOLS)
smolagent_event_details      = SmolAgentFactory(instruct_event_details, WEB_TOOLS)
smolagent_missing_data       = SmolAgentFactory(instruct_missing_data, WEB_TOOLS)
smolagent_attendance_est     = SmolAgentFactory(instruct_attendance_estimator, WEB_TOOLS)
smolagent_classifier         = SmolAgentFactory(instruct_classifier, [])
smolagent_csv_builder        = SmolAgentFactory(instruct_csv_builder, CSV_TOOLS)
smolagent_csv_saver          = SmolAgentFactory(instruct_csv_saver, CSV_TOOLS)
smolagent_final_checker      = SmolAgentFactory(instruct_final_checker, [])

# --------------------------------------------------------------------
# HELPER: Create router factory
# --------------------------------------------------------------------
def make_router(next_node: str, retry_node: str):
    def router(state):
        print("\n=========== ROUTER ===========")
        # ensure lists exist
        state.setdefault("step_name", [])
        state.setdefault("success", [])
        current = state["step_name"][-1] if state["step_name"] else "UNKNOWN"
        print(f"Current step   : {current}")
        last_success = state["success"][-1] if state["success"] else False
        print(f"Last success   : {last_success}")
        if last_success:
            print(f"Routing next   : {next_node}")
            state["step_name"].append(next_node)
            return next_node
        else:
            print(f"Retrying node  : {retry_node}")
            state["step_name"].append(retry_node)
            return retry_node
    return router

# --------------------------------------------------------------------
# CUSTOM VALIDATION NODE
# --------------------------------------------------------------------
def csv_validation_node(state):
    print("\n=========== CSV VALIDATION NODE ===========")
    state.setdefault("observations", [])
    state.setdefault("success", [])
    state.setdefault("step_name", [])
    csv_text = ""
    if state["observations"]:
        csv_text = state["observations"][-1].get("data", "")
    lines = [l for l in csv_text.split("\n") if l.strip()]
    valid = len(lines) - 1 >= 15  # minus header
    print(f"Rows detected (excluding header): {len(lines)-1}")
    state["success"].append(valid)
    state["step_name"].append("csv_validator")
    state.setdefault("answers", []).append(
        "CSV validation passed" if valid else "CSV validation failed"
    )
    return state

# --------------------------------------------------------------------
# BUILD WORKFLOW
# --------------------------------------------------------------------
workflow = StateGraph(WorkflowState)

# Add agent nodes
workflow.add_node("planner"          , WorkflowNodeFactory.create_agent_node(smolagent_planner))
workflow.add_node("web_search"       , WorkflowNodeFactory.create_agent_node(smolagent_web_search))
workflow.add_node("event_details"    , WorkflowNodeFactory.create_agent_node(smolagent_event_details))
workflow.add_node("missing_data"     , WorkflowNodeFactory.create_agent_node(smolagent_missing_data))
workflow.add_node("attendance_est"   , WorkflowNodeFactory.create_agent_node(smolagent_attendance_est))
workflow.add_node("classifier"       , WorkflowNodeFactory.create_agent_node(smolagent_classifier))
workflow.add_node("csv_builder"      , WorkflowNodeFactory.create_agent_node(smolagent_csv_builder))
workflow.add_node("csv_validator"    , csv_validation_node)
workflow.add_node("csv_saver"        , WorkflowNodeFactory.create_agent_node(smolagent_csv_saver))
workflow.add_node("final_checker"    , WorkflowNodeFactory.create_agent_node(smolagent_final_checker))

# --------------------------------------------------------------------
# EDGE DEFINITIONS
# --------------------------------------------------------------------
# START ➜ planner
workflow.add_edge(START, "planner")

# planner ➜ router
workflow.add_conditional_edges(
    "planner",
    make_router("web_search", "planner"),
    {"web_search": "web_search", "planner": "planner"}
)

# web_search ➜ router
workflow.add_conditional_edges(
    "web_search",
    make_router("event_details", "web_search"),
    {"event_details": "event_details", "web_search": "web_search"}
)

# event_details ➜ router
workflow.add_conditional_edges(
    "event_details",
    make_router("missing_data", "event_details"),
    {"missing_data": "missing_data", "event_details": "event_details"}
)

# missing_data ➜ router
workflow.add_conditional_edges(
    "missing_data",
    make_router("attendance_est", "missing_data"),
    {"attendance_est": "attendance_est", "missing_data": "missing_data"}
)

# attendance_est ➜ router
workflow.add_conditional_edges(
    "attendance_est",
    make_router("classifier", "attendance_est"),
    {"classifier": "classifier", "attendance_est": "attendance_est"}
)

# classifier ➜ router
workflow.add_conditional_edges(
    "classifier",
    make_router("csv_builder", "classifier"),
    {"csv_builder": "csv_builder", "classifier": "classifier"}
)

# csv_builder ➜ router
workflow.add_conditional_edges(
    "csv_builder",
    make_router("csv_validator", "csv_builder"),
    {"csv_validator": "csv_validator", "csv_builder": "csv_builder"}
)

# csv_validator ➜ router (if fail, back to missing_data)
def validator_router(state):
    print("\n=========== VALIDATOR ROUTER ===========")
    state.setdefault("success", [])
    state.setdefault("step_name", [])
    passed = state["success"][-1] if state["success"] else False
    if passed:
        print("Validation passed ➜ csv_saver")
        state["step_name"].append("csv_saver")
        return "csv_saver"
    else:
        print("Validation failed ➜ missing_data (retry cycle)")
        state["step_name"].append("missing_data")
        return "missing_data"

workflow.add_conditional_edges(
    "csv_validator",
    validator_router,
    {"csv_saver": "csv_saver", "missing_data": "missing_data"}
)

# csv_saver ➜ router
workflow.add_conditional_edges(
    "csv_saver",
    make_router("final_checker", "csv_saver"),
    {"final_checker": "final_checker", "csv_saver": "csv_saver"}
)

# final_checker ➜ END router
def final_router(state):
    print("\n=========== FINAL ROUTER ===========")
    state.setdefault("success", [])
    succeeded = "WORKFLOW SUCCESS" in (state.get("answers", [])[-1] if state.get("answers") else "")
    return END if succeeded else "final_checker"

workflow.add_conditional_edges(
    "final_checker",
    final_router,
    {END: END, "final_checker": "final_checker"}
)

# --------------------------------------------------------------------
# COMPILE APP
# --------------------------------------------------------------------
app = workflow.compile()

initial_state: WorkflowState = {
    "step_name": ["Initial Step"],
    "actions": [],
    "observations": [],
    "rewards": [],
    "answers": [],
    "success": []
}

try:
    png = app.get_graph().draw_mermaid_png()
    path_graph = os.path.join("./", "workflow_graph.png")
    with open(path_graph, "wb") as f:
        f.write(png)
except Exception as e:
    print(f"Could not save workflow graph.")

result_state = app.invoke(initial_state)
print(result_state)

path_json = os.path.join("workflows/ce16f4a74130408fab5874330684ee36/", "state_result.json")
try:
    with open(path_json, "w") as f:
        json.dump(result_state, f, indent=2)
except Exception as e:
    print(f"Could not save workflow data: {e}")

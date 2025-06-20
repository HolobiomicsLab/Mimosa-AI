
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
    Tool
)
import json

API_BASE_URL = 'http://localhost:5002'


def build_formatted_output(action: str, observation: str, reward: float) -> str:
    action_formatted = action[:256].strip().replace('\n', ' - ')
    observation_formatted = observation[:1024].strip().replace('\n', ' - ')
    return f"""
action: {action_formatted}
observation: {observation_formatted}
reward: {reward}
"""

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
        action = "search:" + query
        try:
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
    description = "Navigate to a specified URL and return the page content."
    inputs = {"url": {"type": "string", "description": "The URL to navigate to."}}
    output_type = "string"

    async def _async_navigate(self, url: str) -> dict:
        async with Client(f"{API_BASE_URL}/mcp") as client:
            buffer = await client.call_tool("navigate", {"url": url})
            return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

    def forward(self, url: str) -> str:
        action = "go_to_url_tool(" + url + ")"
        obs = ''
        reward = 0.0
        try:
            result = asyncio.run(self._async_navigate(url))
        except Exception as e:
            print(str(e))
            obs = f'failed to navigate to {url} due to error: {str(e)}'
            return build_formatted_output(action, obs, reward)
        
        if not result or 'error' in result.get('status', {}):
            return build_formatted_output(action, obs, reward)
        
        title = result.get('title', 'No title found')
        content = result.get('content', 'No content found')
        obs = f'''Tile: {title}
            Start of page:
            {content}
            End of page.
        '''
        reward = 1.0
        return build_formatted_output(action, obs, reward)

class GetNavigableLinksTool(Tool):
    name = "get_navigable_links_tool"
    description = "Retrieves a list of navigable links from the browser."
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
            self.local = True
            return MLXModel(
                model_id=self.model_id,
                max_tokens=self.max_tokens,
            )
        elif self.engine_name == "hf_api":
            return HfApiModel(
                model_id=self.model_id,
                provider=self.provider,
                token=self.token,
                max_tokens=self.max_tokens,
            )
        elif self.engine_name == "inference_client":
            return InferenceClientModel(
                model_id=self.model_id,
                provider=self.provider,
                token=self.token,
                max_tokens=self.max_tokens,
            )
        else:
            raise ValueError(f"Unknown engine name: {self.engine_name}. Supported engines are: mlx, hf_api, inference_client.")
        
    def build_worflow_step_prompt(self, state: WorkflowState) -> str:
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
        state_answers = state.get("answers", [])
        prev_infos = state_answers[-1] if state_answers else "No information yet"
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
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('action:'):
                action = line[7:].strip()
                actions.append(action)
            elif line.startswith('observation:'):
                obs_str = line[12:].strip()
                observations.append(obs_str)
            elif line.startswith('reward:'):
                reward_str = line[7:].strip()
                reward = float(reward_str)
                rewards.append(reward)
                success.append(reward > 0)
        return ('\n'.join(actions),
                '\n'.join(observations),
                (sum(rewards) / len(rewards)) if len(rewards) > 0 else sum(rewards),
                any(success) or len(success) == 0
        )

    def parse_memory_output(self):
        actions, observations, rewards, success = [], [], [], []
        for idx, step in enumerate(self.agent.memory.steps):
            if isinstance(step, ActionStep):
                error, feedback = step.error, step.observations
                step_output = error if error else feedback
                if not isinstance(step_output, str):
                    continue
                action_step, obs_step, reward_step, success_step = self.parse_tool_output(step_output)
                actions.append(action_step)
                observations.append(obs_step)
                rewards.append(reward_step)
                success.append(success_step)
        return actions, observations, rewards, success

    def run(self, state: WorkflowState) -> dict:
        instructions = self.build_worflow_step_prompt(state)
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
        reward = sum(rewards) if rewards else 0.0
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

# ---------- Workflow State Schema (already provided) ----------
from typing import List, TypedDict

class Action(TypedDict):
    tool: str
    inputs: dict

class Observation(TypedDict):
    data: str

class WorkflowState(TypedDict):
    step_name: List[str]
    actions: List[Action]
    observations: List[Observation]
    rewards: List[float]
    answers: List[str]
    success: List[bool]

# ---------- Router Factory ----------
def make_router(success_node: str, failure_node: str):
    def router(state: WorkflowState) -> str:
        print("======== ROUTING DECISION ========")
        print(f"📊 Current state keys: {list(state.keys())}")
        try:
            success_list = state.get("success", [])
            if success_list and success_list[-1]:
                print(f"🎉 Last step succeeded - routing to {success_node}")
                return success_node
            else:
                print(f"❌ Last step failed or no success recorded - routing to {failure_node}")
                return failure_node
        except Exception as e:
            print(f"🚨 Routing error: {e} - defaulting to {failure_node}")
            return failure_node
        finally:
            print("📍 ======== END ROUTING ========\n")
    return router

# ---------- Agent Instructions ----------
instruct_planner = """
You are an event research planning agent.

## TASK
- Produce a concise plan outlining how to gather information on community events in Austin, TX for July–August 2025.
- Identify keywords, municipal calendars, event aggregators and other sources to search.
- Output actionable next-step bullet points.

## CONSIDERATIONS
- Do not perform the searches yourself; only design the research roadmap.
"""

instruct_searcher = """
You are a web search agent.

## TASK
- Execute targeted web searches to find community events in Austin, TX occurring July–August 2025.
- Save each promising event page link as an action output for downstream agents.
- Limit yourself to 5 distinct pages per run to conserve context.
"""

instruct_collector = """
You are an event list extraction agent.

## TASK
- From the provided event page links, extract event names, dates and locations.
- Return one JSON object per event with the extracted fields filled.
- Do NOT speculate on attendance or vendor info yet.
"""

instruct_vendor_info = """
You are a vendor-information agent.

## TASK
- For each event passed to you, locate vendor application details:
  • application deadline
  • application fee
  • link to vendor application page
- Return enriched JSON objects.
"""

instruct_attendance = """
You are an attendance estimation agent.

## TASK
- Using reliable web sources (news articles, past event recaps, municipal data), estimate expected attendance for each event.
- Return the same JSON objects now including 'estimated_attendance'.
- Cite source links in a 'source' field for each estimate.
"""

instruct_csv_compiler = """
You are a CSV generation agent.

## TASK
- Receive a list of fully-populated event JSON objects.
- Produce a CSV file with columns:
  Event Name, Date, Location, Estimated Attendance, Vendor Deadline, Application Fee, Application Link, High Potential
- Use provided tools to write the CSV to disk and return the file path in the observation.
"""

# ---------- Tools ----------
WEB_TOOLS = BROWSER_TOOLS_TOOL
CSV_TOOLS = CSV_TOOLS_TOOL

# ---------- SmolAgent Creation ----------
smol_planner        = SmolAgentFactory(instruct_planner, WEB_TOOLS)
smol_searcher       = SmolAgentFactory(instruct_searcher, WEB_TOOLS)
smol_collector      = SmolAgentFactory(instruct_collector, WEB_TOOLS)
smol_vendor_info    = SmolAgentFactory(instruct_vendor_info, WEB_TOOLS)
smol_attendance     = SmolAgentFactory(instruct_attendance, WEB_TOOLS)
smol_csv_compiler   = SmolAgentFactory(instruct_csv_compiler, CSV_TOOLS)

# ---------- Function Nodes ----------
def data_validator_node(state: WorkflowState) -> dict:
    print("🔍 Running data_validator_node")
    observations = state.get("observations", [])
    success_flag = bool(observations)
    state["step_name"].append("data_validator")
    state.setdefault("success", []).append(success_flag)
    return state

def high_potential_node(state: WorkflowState) -> dict:
    print("⭐ Running high_potential_node")
    observations = state.get("observations", [])
    if observations:
        last_obs = observations[-1]
        # Very naive heuristic: mark High Potential if attendance > 5000 and fee <= 300
        try:
            import json, copy
            events = json.loads(last_obs["data"])
            enhanced_events = []
            for evt in events:
                evt_cp = copy.deepcopy(evt)
                att = evt_cp.get("estimated_attendance", 0) or 0
                fee = evt_cp.get("application_fee", 0) or 0
                evt_cp["High Potential"] = "Yes" if att >= 5000 and fee <= 300 else "No"
                enhanced_events.append(evt_cp)
            last_obs["data"] = json.dumps(enhanced_events)
        except Exception as e:
            print(f"⚠️  Enhancement failed: {e}")
    state["step_name"].append("high_potential")
    state.setdefault("success", []).append(True)
    return state

def quality_checker_node(state: WorkflowState) -> dict:
    print("✅ Running quality_checker_node")
    success_flag = True
    observations = state.get("observations", [])
    if not observations:
        success_flag = False
    state["step_name"].append("quality_checker")
    state.setdefault("success", []).append(success_flag)
    return state

def finalizer_node(state: WorkflowState) -> dict:
    print("🏁 Finalizer node reached – marking workflow complete.")
    state["step_name"].append("finalizer")
    state.setdefault("success", []).append(True)
    return state

# ---------- Workflow Initialization ----------
workflow = StateGraph(WorkflowState)

# ---------- Add Nodes ----------
workflow.add_node("planner",        WorkflowNodeFactory.create_agent_node(smol_planner))
workflow.add_node("searcher",       WorkflowNodeFactory.create_agent_node(smol_searcher))
workflow.add_node("collector",      WorkflowNodeFactory.create_agent_node(smol_collector))
workflow.add_node("vendor_info",    WorkflowNodeFactory.create_agent_node(smol_vendor_info))
workflow.add_node("attendance",     WorkflowNodeFactory.create_agent_node(smol_attendance))
workflow.add_node("validator",      data_validator_node)
workflow.add_node("high_potential", high_potential_node)
workflow.add_node("csv_compiler",   WorkflowNodeFactory.create_agent_node(smol_csv_compiler))
workflow.add_node("quality_check",  quality_checker_node)
workflow.add_node("finalizer",      finalizer_node)

# ---------- Edge Definitions ----------
workflow.add_edge(START, "planner")

workflow.add_conditional_edges("planner",
    make_router("searcher", "planner"),
    {"searcher": "searcher", "planner": "planner"}
)

workflow.add_conditional_edges("searcher",
    make_router("collector", "searcher"),
    {"collector": "collector", "searcher": "searcher"}
)

workflow.add_conditional_edges("collector",
    make_router("vendor_info", "searcher"),
    {"vendor_info": "vendor_info", "searcher": "searcher"}
)

workflow.add_conditional_edges("vendor_info",
    make_router("attendance", "vendor_info"),
    {"attendance": "attendance", "vendor_info": "vendor_info"}
)

workflow.add_conditional_edges("attendance",
    make_router("validator", "attendance"),
    {"validator": "validator", "attendance": "attendance"}
)

workflow.add_conditional_edges("validator",
    make_router("high_potential", "searcher"),
    {"high_potential": "high_potential", "searcher": "searcher"}
)

workflow.add_conditional_edges("high_potential",
    make_router("csv_compiler", "high_potential"),
    {"csv_compiler": "csv_compiler", "high_potential": "high_potential"}
)

workflow.add_conditional_edges("csv_compiler",
    make_router("quality_check", "csv_compiler"),
    {"quality_check": "quality_check", "csv_compiler": "csv_compiler"}
)

workflow.add_conditional_edges("quality_check",
    make_router("finalizer", "searcher"),
    {"finalizer": "finalizer", "searcher": "searcher"}
)

workflow.add_edge("finalizer", END)

# ---------- Compile Workflow ----------
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

path_json = os.path.join("workflows/341e9047a0534b07bc528e0c17f63fe4/", "state_result.json")
try:
    with open(path_json, "w") as f:
        json.dump(result_state, f, indent=2)
except Exception as e:
    print(f"Could not save workflow data: {e}")

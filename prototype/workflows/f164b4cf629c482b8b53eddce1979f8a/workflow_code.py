
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
                obs = f"Failed to create dataset: {result.get('message', 'Unknown error') if result else 'No response'}"
                
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
                error_msg = result.get('message', 'Unknown error') if result else 'No response'
                obs = f"Failed to load CSV from '{file_path}': {error_msg}"
                if name:
                    obs += f" (as dataset: '{name}')"
                    
        except Exception as e:
            obs = f"Error loading CSV from '{file_path}': {str(e)}"
            
        return build_formatted_output(action, obs, reward)

class GetCSVInfoTool(Tool):
    name = "get_csv_info_tool"
    description = "Get information about a dataset including shape, columns, and data types."
    inputs = {"name": {"type": "string", "description": "Name of the dataset."}}
    output_type = "string"

    def forward(self, name: str) -> str:
        action = f"get_csv_info_tool({name})"
        obs = ''
        reward = 0.0
        
        try:
            async def _get_csv_info():
                async with Client(f"{API_BASE_URL}/mcp") as client:
                    buffer = await client.call_tool("get_csv_info", {"name": name})
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

            result = asyncio.run(_get_csv_info())
            
            if result and result.get('status') == 'success':
                obs = f"Dataset '{name}': shape {result.get('shape')}, columns: {result.get('columns')}"
                reward = 1.0
            else:
                obs = f"Dataset not found: {name}"
                
        except Exception as e:
            obs = f"Error getting dataset info: {str(e)}"
            
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
                error_msg = result.get('message', 'Unknown error') if result else 'No response'
                obs = f"Failed to get data from '{name}': {error_msg}"
                if limit:
                    obs += f" (limit: {limit})"
                if columns:
                    obs += f" (columns: {columns})"
                    
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
                    buffer = await client.call_tool("add_csv_row", {"name": name, "row": row})
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

            result = asyncio.run(_add_csv_row())
            
            if result and result.get('status') == 'success':
                obs = f"Added row to '{name}', new shape: {result.get('shape')}"
                reward = 1.0
            else:
                obs = f"Failed to add row: {result.get('message', 'Unknown error') if result else 'No response'}"
                
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
                    buffer = await client.call_tool("update_csv_row", {"name": name, "index": index, "row": row})
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

            result = asyncio.run(_update_csv_row())
            
            if result and result.get('status') == 'success':
                obs = f"Updated row {index} in '{name}'"
                reward = 1.0
            else:
                obs = f"Failed to update row: {result.get('message', 'Unknown error') if result else 'No response'}"
                
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
                    buffer = await client.call_tool("delete_csv_row", {"name": name, "index": index})
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

            result = asyncio.run(_delete_csv_row())
            
            if result and result.get('status') == 'success':
                obs = f"Deleted row {index} from '{name}', new shape: {result.get('shape')}"
                reward = 1.0
            else:
                obs = f"Failed to delete row: {result.get('message', 'Unknown error') if result else 'No response'}"
                
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
                obs = f"Failed to add column: {result.get('message', 'Unknown error') if result else 'No response'}"
                
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
                    payload = {"name": name, "operation": operation}
                    buffer = await client.call_tool("query_csv", payload)
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

            result = asyncio.run(_query_csv())
            
            if result and result.get('status') == 'success':
                query_result = result.get('result')
                obs = f"Query '{operation}' on '{name}' completed - Result: {str(query_result)[:500]}"
                reward = 1.0
            else:
                obs = f"Query failed: {result.get('message', 'Unknown error') if result else 'No response'}"
                
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
                    buffer = await client.call_tool("list_csv_datasets", {})
                    return json.loads(buffer[0].text) if buffer else {"status": "error", "message": "No response from server"}

            result = asyncio.run(_list_csv_datasets())
            
            if result and result.get('status') == 'success':
                datasets = result.get('columns', [])
                obs = f"Found {len(datasets)} datasets: " + ", ".join([f"{d['name']}({d['shape']})" for d in datasets])
                reward = 1.0
            else:
                obs = "Failed to list datasets"
                
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
get_csv_info_tool = GetCSVInfoTool()
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
    get_csv_info_tool,
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
class SmolAgentFactory:

    def __init__(self, instruct_prompt, tools, model_id="Qwen/Qwen2.5-72B-Instruct", max_steps=5):
        self.model_id = model_id
        self.token = os.getenv("HF_TOKEN")
        self.tools = tools or []
        self.instruct_prompt = instruct_prompt
        self.local = False

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
        if self.local:
            return MLXModel(
                model_id=self.model_id,
                max_tokens=5000,
            )
        return HfApiModel(
            model_id=self.model_id,
            provider="nebius",
            token=self.token,
            max_tokens=5000,
        )
        
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
                any(success)
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
        result = self.agent.run(instructions)
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

# ===== Instruction Templates =====
instruct_web_research = """
You are a web research agent specialized in discovering state-of-the-art AI/ML techniques.

## YOUR TASK
1. Search reputable sources (arXiv, Google Scholar, journals, technical blogs) for the most recent and influential AI/ML techniques.
2. For each technique, collect:
   - Technique name
   - Authoritative paper link (original or highly cited)
   - Publication date
3. Read one page or paper per tool call to avoid context overload.
4. Provide the gathered data as a structured list for downstream agents.
"""

instruct_paper_analysis = """
You are a paper analysis agent.

## YOUR TASK
1. For each technique and paper link provided, read or skim the paper (one per tool call).
2. Produce a concise 2-3 sentence description explaining:
   - The core idea
   - Novelty and significance
3. Return a list of dictionaries with keys:
   - Technique
   - Description
   - Paper Link
   - Date
"""

instruct_csv_report = """
You are a CSV report generation agent.

## YOUR TASK
1. Take the structured list of techniques, descriptions, links, and dates.
2. Generate a CSV file (or CSV-formatted text) with exactly these columns:
   - Technique
   - Description
   - Paper Link
   - Date
3. Use the CSV tools provided to write or output the final CSV.
4. Ensure the CSV is clean, well-formatted, and ready for download.
"""

# ===== Agent Creation =====
smolagent_web_research = SmolAgentFactory(instruct_web_research, BROWSER_TOOLS_TOOL)
smolagent_paper_analysis = SmolAgentFactory(instruct_paper_analysis, BROWSER_TOOLS_TOOL)
smolagent_csv_report = SmolAgentFactory(instruct_csv_report, CSV_TOOLS_TOOL)

# ===== Workflow Graph =====
workflow = StateGraph(WorkflowState)

# ----- Nodes -----
workflow.add_node("web_research", WorkflowNodeFactory.create_agent_node(smolagent_web_research))
workflow.add_node("paper_analysis", WorkflowNodeFactory.create_agent_node(smolagent_paper_analysis))
workflow.add_node("csv_reporter", WorkflowNodeFactory.create_agent_node(smolagent_csv_report))

# ----- Routing Functions -----
def router_after_web(state: WorkflowState) -> str:
    state.setdefault("step_name", [])
    state.setdefault("success", [])
    last_success = state["success"][-1] if state["success"] else True
    if last_success:
        state["step_name"].append("paper_analysis")
        return "paper_analysis"
    attempts = state["step_name"].count("web_research")
    if attempts < 2:
        state["step_name"].append("web_research")
        return "web_research"
    state["step_name"].append("paper_analysis")
    return "paper_analysis"

def router_after_analysis(state: WorkflowState) -> str:
    state.setdefault("step_name", [])
    state.setdefault("success", [])
    last_success = state["success"][-1] if state["success"] else True
    if last_success:
        state["step_name"].append("csv_reporter")
        return "csv_reporter"
    attempts = state["step_name"].count("paper_analysis")
    if attempts < 2:
        state["step_name"].append("paper_analysis")
        return "paper_analysis"
    state["step_name"].append("csv_reporter")
    return "csv_reporter"

def router_after_csv(state: WorkflowState) -> str:
    state.setdefault("step_name", [])
    state.setdefault("success", [])
    last_success = state["success"][-1] if state["success"] else True
    if last_success:
        state["step_name"].append("end")
        return END
    attempts = state["step_name"].count("csv_reporter")
    if attempts < 2:
        state["step_name"].append("csv_reporter")
        return "csv_reporter"
    state["step_name"].append("end")
    return END

# ----- Edges -----
workflow.add_edge(START, "web_research")

workflow.add_conditional_edges(
    "web_research",
    router_after_web,
    {
        "web_research": "web_research",
        "paper_analysis": "paper_analysis",
        END: END,
    },
)

workflow.add_conditional_edges(
    "paper_analysis",
    router_after_analysis,
    {
        "paper_analysis": "paper_analysis",
        "csv_reporter": "csv_reporter",
        END: END,
    },
)

workflow.add_conditional_edges(
    "csv_reporter",
    router_after_csv,
    {
        "csv_reporter": "csv_reporter",
        END: END,
    },
)

# ----- Compile Workflow -----
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

path_json = os.path.join("workflows/f164b4cf629c482b8b53eddce1979f8a/", "state_result.json")
try:
    with open(path_json, "w") as f:
        json.dump(result_state, f, indent=2)
except Exception as e:
    print(f"Could not save workflow data: {e}")

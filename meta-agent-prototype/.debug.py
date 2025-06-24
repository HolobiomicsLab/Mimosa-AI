
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

    def __init__(self, instruct_prompt, tools,
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
        trajectories = zip(
            state_actions, 
            state_observations, 
            state_success
        )
        trajectory_str = ""
        for idx, (action, observation, success) in enumerate(trajectories):
            if not action or action == {}:
                continue
            trajectory_str += f"""
        ### Step {idx + 1}:
        Action: {action['tool']}
        Observation: {observation['data'][:256]}... (truncated for brevity)
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

    def run(self, state: WorkflowState) -> dict:
        instructions = self.build_workflow_step_prompt(state)
        try:
            result = self.agent.run(instructions)
        except Exception as e:
            print(f"Error running agent: {e}")
            return {
                **state,
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
            "actions": state.get("actions", []) + [action],
            "observations": state.get("observations", []) + [obs],
            "rewards": state.get("rewards", []) + [reward],
            "success": state.get("success", []) + [success_bool],
            "answers": state.get("answers", []) + [result],
        }

class WorkflowNodeFactory:
    @staticmethod
    def create_agent_node(agent_factory: SmolAgentFactory) -> Callable[[WorkflowState], dict]:
        def node_function(state: WorkflowState) -> dict:
            return agent_factory.run(state)
        return node_function



# LLM generated logical multi-agent graph
# ---------------------------------------------
# LANGGRAPH – SMOLAGENT WORKFLOW (GPAW install)
# ---------------------------------------------
#
# Prerequisites (already available in runtime):
# - SHELL_TOOLS
# - BROWSER_TOOLS
# - CSV_TOOLS
# - WorkflowState / Action / Observation  (schema)
# - SmolAgentFactory
# - WorkflowNodeFactory
# - langgraph.graph : StateGraph, START, END
#

from langgraph.graph import StateGraph, START, END

# ------------------------------------------------------------------
# 1. AGENT INSTRUCTIONS  (one atomic responsibility – no overlap)
# ------------------------------------------------------------------

instruct_search = """
You are a focused WEB SEARCH agent.

YOUR TASK
- Use browser-type tools to search the web for instructions on
  installing GPAW on macOS with an M1 (Apple-Silicon) chip.
- Save links, titles, and short snippets that look relevant.

UPON COMPLETION
- If you found at least one promising source say SEARCH_COMPLETE
- If you did not find anything useful say SEARCH_FAILURE
- If a tool error blocks you say GIVE_UP
"""

instruct_extract = """
You are an INFORMATION EXTRACTION agent.

YOUR TASK
- Read the latest browser observations (web pages) and extract ONLY
  the concrete step-by-step shell commands required to install GPAW
  on macOS (M1).  Ignore intel-only or linux-only commands.
- Output the ordered list of commands.

UPON COMPLETION
- If you have a clear list of commands say EXTRACT_COMPLETE
- If extraction failed say EXTRACT_FAILURE
- On unknown errors say GIVE_UP
"""

instruct_validate = """
You are a VALIDATION agent.

YOUR TASK
- Inspect the extracted installation commands.
- Check they explicitly target macOS on Apple Silicon (arm64/m1) and
  do not include incompatible binaries.
- If anything is missing, explain what and mark failure.

UPON COMPLETION
- If commands look correct say VALIDATION_PASS
- If they are wrong/incomplete say VALIDATION_FAIL
- On unhandled error say GIVE_UP
"""

instruct_install = """
You are an INSTALLATION EXECUTION agent.

YOUR TASK
- Run the validated shell commands *exactly* in order using the
  provided shell tool set.
- Capture stdout/stderr for each command.

UPON COMPLETION
- If every command succeeded say INSTALL_SUCCESS
- If a command failed say INSTALL_FAIL
- For unexpected tool errors say GIVE_UP
"""

instruct_verify = """
You are a VERIFICATION agent.

YOUR TASK
- Run `gpaw --version` (or an equivalent command) using the shell
  tool to confirm the installation.
  
UPON COMPLETION
- If GPAW returns a version string say VERIFY_SUCCESS
- If the command errors or GPAW missing say VERIFY_FAIL
- On unknown errors say GIVE_UP
"""

# ------------------------------------------------------------------
# 2. CREATE SMOLAGENTS (tool selection strictly matches responsibility)
# ------------------------------------------------------------------

search_agent_factory   = SmolAgentFactory(instruct_search,  BROWSER_TOOLS)
extract_agent_factory  = SmolAgentFactory(instruct_extract, BROWSER_TOOLS)
validate_agent_factory = SmolAgentFactory(instruct_validate, CSV_TOOLS)
install_agent_factory  = SmolAgentFactory(instruct_install, SHELL_TOOLS)
verify_agent_factory   = SmolAgentFactory(instruct_verify,  SHELL_TOOLS)

# ------------------------------------------------------------------
# 3. WORKFLOW GRAPH
# ------------------------------------------------------------------

workflow = StateGraph(WorkflowState)

# ---- 3.1  Agent Nodes ----
workflow.add_node("search_web",      WorkflowNodeFactory.create_agent_node(search_agent_factory))
workflow.add_node("extract_steps",   WorkflowNodeFactory.create_agent_node(extract_agent_factory))
workflow.add_node("validate_steps",  WorkflowNodeFactory.create_agent_node(validate_agent_factory))
workflow.add_node("install_software",WorkflowNodeFactory.create_agent_node(install_agent_factory))
workflow.add_node("verify_install",  WorkflowNodeFactory.create_agent_node(verify_agent_factory))

# ---- 3.2  Manual fallback node (function) ----
def manual_review(state: WorkflowState) -> dict:
    """
    Emergency graceful-degradation node.
    Simply marks the workflow as unsuccessful but returns all
    gathered information for human review.
    """
    state.setdefault("step_name", []).append("manual_review")
    state.setdefault("answers",   []).append("MANUAL_INTERVENTION_REQUIRED")
    state.setdefault("success",   []).append(False)
    return state

workflow.add_node("manual_review", manual_review)

# ------------------------------------------------------------------
# 4. ROUTING FUNCTIONS  (robust, multi-path, with retry counters)
# ------------------------------------------------------------------

MAX_RETRIES = 3        # universal retry cap per node

def _attempts(state: WorkflowState, node_name: str) -> int:
    """Helper: count how many times a node has already executed."""
    return sum(1 for n in state.get("step_name", []) if n.startswith(node_name))

# ---- search_router ----
def search_router(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "SEARCH_COMPLETE" in answer:
            return "extract_steps"
        elif _attempts(state, "search_web") < MAX_RETRIES:
            return "search_web"              # retry
        else:
            return "manual_review"           # hard fallback
    except Exception as e:
        print(f"Routing error (search): {e}")
        return "manual_review"

# ---- extract_router ----
def extract_router(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "EXTRACT_COMPLETE" in answer:
            return "validate_steps"
        elif _attempts(state, "extract_steps") < MAX_RETRIES:
            return "extract_steps"           # retry extraction
        else:
            return "search_web"              # re-search different sources
    except Exception as e:
        print(f"Routing error (extract): {e}")
        return "manual_review"

# ---- validate_router ----
def validate_router(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "VALIDATION_PASS" in answer:
            return "install_software"
        elif "VALIDATION_FAIL" in answer and _attempts(state, "search_web") < MAX_RETRIES:
            return "search_web"              # gather better resources
        elif _attempts(state, "validate_steps") < MAX_RETRIES:
            return "validate_steps"          # retry validation
        else:
            return "manual_review"
    except Exception as e:
        print(f"Routing error (validate): {e}")
        return "manual_review"

# ---- install_router ----
def install_router(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "INSTALL_SUCCESS" in answer:
            return "verify_install"
        elif _attempts(state, "install_software") < MAX_RETRIES:
            return "install_software"        # retry installation
        else:
            return "manual_review"
    except Exception as e:
        print(f"Routing error (install): {e}")
        return "manual_review"

# ---- verify_router ----
def verify_router(state: WorkflowState) -> str:
    try:
        answer = state.get("answers", [""])[-1]
        if "VERIFY_SUCCESS" in answer:
            return END
        elif _attempts(state, "install_software") < MAX_RETRIES:
            return "install_software"        # attempt re-install
        else:
            return "manual_review"
    except Exception as e:
        print(f"Routing error (verify): {e}")
        return "manual_review"

# ------------------------------------------------------------------
# 5. GRAPH EDGES & CONDITIONAL ROUTING
# ------------------------------------------------------------------

# start
workflow.add_edge(START, "search_web")

# search node edges
workflow.add_conditional_edges(
    "search_web",
    search_router,
    {
        "extract_steps":    "extract_steps",
        "search_web":       "search_web",      # retry
        "manual_review":    "manual_review"
    }
)

# extract node edges
workflow.add_conditional_edges(
    "extract_steps",
    extract_router,
    {
        "validate_steps":   "validate_steps",
        "extract_steps":    "extract_steps",   # retry
        "search_web":       "search_web",
        "manual_review":    "manual_review"
    }
)

# validate node edges
workflow.add_conditional_edges(
    "validate_steps",
    validate_router,
    {
        "install_software": "install_software",
        "validate_steps":   "validate_steps",  # retry
        "search_web":       "search_web",
        "manual_review":    "manual_review"
    }
)

# install node edges
workflow.add_conditional_edges(
    "install_software",
    install_router,
    {
        "verify_install":   "verify_install",
        "install_software": "install_software",# retry
        "manual_review":    "manual_review"
    }
)

# verify node edges
workflow.add_conditional_edges(
    "verify_install",
    verify_router,
    {
        END:                END,
        "install_software": "install_software",
        "manual_review":    "manual_review"
    }
)

# manual_review edges – always end
workflow.add_edge("manual_review", END)

# ------------------------------------------------------------------
# 6. COMPILE APP
# ------------------------------------------------------------------

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

path_json = os.path.join("", "state_result.json")
try:
    with open(path_json, "w") as f:
        json.dump(result_state, f, indent=2)
except Exception as e:
    print(f"Could not save workflow data: {e}")

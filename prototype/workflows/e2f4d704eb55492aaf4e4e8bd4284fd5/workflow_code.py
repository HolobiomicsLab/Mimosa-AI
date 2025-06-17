
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
import requests
import json

from smolagents import Tool

CSV_API_BASE_URL = 'http://localhost:5001'

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
        action = f"create_csv: {name}"
        try:
            route = CSV_API_BASE_URL + '/api/csv/create'
            payload = {"name": name}
            if columns:
                payload["columns"] = columns
            if rows:
                payload["rows"] = rows
            
            response = requests.post(route, json=payload)
            data = response.json()
            
            if data.get('status') == 'success':
                obs = f"Created dataset '{name}' with shape {data.get('shape')}"
                reward = 1.0
            else:
                obs = f"Failed to create dataset: {data.get('message', 'Unknown error')}"
                reward = 0.0
        except Exception as e:
            obs = f"Error creating dataset: {str(e)}"
            reward = 0.0
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
        action = f"load_csv: {file_path}"
        try:
            with open(file_path, 'rb') as f:
                files = {'file': f}
                data = {}
                if name:
                    data['name'] = name
                
                route = CSV_API_BASE_URL + '/api/csv/load'
                response = requests.post(route, files=files, data=data)
                result = response.json()
                
                if result.get('status') == 'success':
                    obs = f"Loaded dataset '{result.get('name')}' with shape {result.get('shape')}"
                    reward = 1.0
                else:
                    obs = f"Failed to load CSV: {result.get('message', 'Unknown error')}"
                    reward = 0.0
        except Exception as e:
            obs = f"Error loading CSV: {str(e)}"
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class GetCSVInfoTool(Tool):
    name = "get_csv_info_tool"
    description = "Get information about a dataset including shape, columns, and data types."
    inputs = {"name": {"type": "string", "description": "Name of the dataset."}}
    output_type = "string"

    def forward(self, name: str) -> str:
        action = f"get_csv_info: {name}"
        try:
            route = CSV_API_BASE_URL + f'/api/csv/info/{name}'
            response = requests.get(route)
            data = response.json()
            
            if data.get('status') == 'success':
                obs = f"Dataset '{name}': shape {data.get('shape')}, columns: {data.get('columns')}"
                reward = 1.0
            else:
                obs = f"Dataset not found: {name}"
                reward = 0.0
        except Exception as e:
            obs = f"Error getting dataset info: {str(e)}"
            reward = 0.0
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
        action = f"get_csv_data: {name}"
        try:
            route = CSV_API_BASE_URL + f'/api/csv/data/{name}'
            params = {}
            if limit:
                params['limit'] = limit
            if columns:
                params['columns'] = columns
            
            response = requests.get(route, params=params)
            data = response.json()
            
            if data.get('status') == 'success':
                rows = data.get('data', [])
                obs = f"Retrieved {len(rows)} rows from '{name}'"
                if rows:
                    obs += f" - Sample: {str(rows[0])[:200]}"
                reward = 1.0
            else:
                obs = f"Failed to get data: {data.get('message', 'Unknown error')}"
                reward = 0.0
        except Exception as e:
            obs = f"Error getting CSV data: {str(e)}"
            reward = 0.0
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
        action = f"add_csv_row: {name}"
        try:
            route = CSV_API_BASE_URL + f'/api/csv/add_row/{name}'
            response = requests.post(route, json={"row": row})
            data = response.json()
            
            if data.get('status') == 'success':
                obs = f"Added row to '{name}', new shape: {data.get('new_shape')}"
                reward = 1.0
            else:
                obs = f"Failed to add row: {data.get('message', 'Unknown error')}"
                reward = 0.0
        except Exception as e:
            obs = f"Error adding row: {str(e)}"
            reward = 0.0
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
        action = f"update_csv_row: {name}[{index}]"
        try:
            route = CSV_API_BASE_URL + f'/api/csv/update_row/{name}/{index}'
            response = requests.put(route, json={"row": row})
            data = response.json()
            
            if data.get('status') == 'success':
                obs = f"Updated row {index} in '{name}'"
                reward = 1.0
            else:
                obs = f"Failed to update row: {data.get('message', 'Unknown error')}"
                reward = 0.0
        except Exception as e:
            obs = f"Error updating row: {str(e)}"
            reward = 0.0
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
        action = f"delete_csv_row: {name}[{index}]"
        try:
            route = CSV_API_BASE_URL + f'/api/csv/delete_row/{name}/{index}'
            response = requests.delete(route)
            data = response.json()
            
            if data.get('status') == 'success':
                obs = f"Deleted row {index} from '{name}', new shape: {data.get('new_shape')}"
                reward = 1.0
            else:
                obs = f"Failed to delete row: {data.get('message', 'Unknown error')}"
                reward = 0.0
        except Exception as e:
            obs = f"Error deleting row: {str(e)}"
            reward = 0.0
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
        action = f"add_csv_column: {name}.{column_name}"
        try:
            route = CSV_API_BASE_URL + f'/api/csv/add_column/{name}'
            payload = {"name": column_name}
            if default_value is not None:
                payload["default_value"] = default_value
            
            response = requests.post(route, json=payload)
            data = response.json()
            
            if data.get('status') == 'success':
                obs = f"Added column '{column_name}' to '{name}', new shape: {data.get('new_shape')}"
                reward = 1.0
            else:
                obs = f"Failed to add column: {data.get('message', 'Unknown error')}"
                reward = 0.0
        except Exception as e:
            obs = f"Error adding column: {str(e)}"
            reward = 0.0
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
        action = f"query_csv: {name}.{operation}"
        try:
            route = CSV_API_BASE_URL + f'/api/csv/query/{name}'
            payload = {"operation": operation}
            if column:
                payload["column"] = column
            if conditions:
                payload["conditions"] = conditions
            
            response = requests.post(route, json=payload)
            data = response.json()
            
            if data.get('status') == 'success':
                result = data.get('result')
                obs = f"Query '{operation}' on '{name}' completed - Result: {str(result)[:500]}"
                reward = 1.0
            else:
                obs = f"Query failed: {data.get('message', 'Unknown error')}"
                reward = 0.0
        except Exception as e:
            obs = f"Error querying dataset: {str(e)}"
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class ListCSVDatasetsTool(Tool):
    name = "list_csv_datasets_tool"
    description = "List all available datasets with their basic information."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = "list_csv_datasets"
        try:
            route = CSV_API_BASE_URL + '/api/csv/list'
            response = requests.get(route)
            data = response.json()
            
            if data.get('status') == 'success':
                datasets = data.get('datasets', [])
                obs = f"Found {len(datasets)} datasets: " + ", ".join([f"{d['name']}({d['shape']})" for d in datasets])
                reward = 1.0
            else:
                obs = "Failed to list datasets"
                reward = 0.0
        except Exception as e:
            obs = f"Error listing datasets: {str(e)}"
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class ExportCSVTool(Tool):
    name = "export_csv_tool"
    description = "Export a dataset as a CSV file."
    inputs = {
        "name": {"type": "string", "description": "Name of the dataset to export."},
        "output_path": {"type": "string", "description": "Path where to save the exported CSV file."}
    }
    output_type = "string"

    def forward(self, name: str, output_path: str) -> str:
        action = f"export_csv: {name} -> {output_path}"
        try:
            route = CSV_API_BASE_URL + f'/api/csv/export/{name}'
            response = requests.get(route)
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                obs = f"Exported dataset '{name}' to '{output_path}'"
                reward = 1.0
            else:
                obs = f"Failed to export dataset '{name}'"
                reward = 0.0
        except Exception as e:
            obs = f"Error exporting dataset: {str(e)}"
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class DeleteCSVDatasetTool(Tool):
    name = "delete_csv_dataset_tool"
    description = "Delete a dataset from memory."
    inputs = {"name": {"type": "string", "description": "Name of the dataset to delete."}}
    output_type = "string"

    def forward(self, name: str) -> str:
        action = f"delete_csv_dataset: {name}"
        try:
            route = CSV_API_BASE_URL + f'/api/csv/delete/{name}'
            response = requests.delete(route)
            data = response.json()
            
            if data.get('status') == 'success':
                obs = f"Deleted dataset '{name}'"
                reward = 1.0
            else:
                obs = f"Failed to delete dataset '{name}'"
                reward = 0.0
        except Exception as e:
            obs = f"Error deleting dataset: {str(e)}"
            reward = 0.0
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
export_csv_tool = ExportCSVTool()
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
    export_csv_tool,
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

from smolagents import (
    Tool,
    DuckDuckGoSearchTool
)

API_BASE_URL = 'http://localhost:5000'

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

    def forward(self, query: str) -> str:
        obs = ''
        action = "search:" + query
        try:
            search_tool = DuckDuckGoSearchTool()  # Instantiate the tool
            obs = search_tool(query)              # Call the instance
            reward = 1.0 if obs else 0.0
        except Exception as e:
            obs = "Search failed for query: " + query + " due to error: " + str(e)
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class GoToUrlTool(Tool):
    name = "go_to_url_tool"
    description = "Navigate to a specified URL."
    inputs = {"url": {"type": "string", "description": "The URL to navigate to."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        action = "go_to_url: " + url
        obs = None
        try:
            route = API_BASE_URL + '/api/browser/navigate'
            response = requests.post(
                route,
                json={'url': url}
            )
            data = response.json()
            obs = f'navigated to ' + url if data.get('status') == 'success' else f'failed to navigate to ' + url
            reward = 1.0 if obs else 0.0
        except Exception as e:
            obs = obs if obs else 'failed to navigate to ' + url + ' due to error: ' + str(e)
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class GetPageTextTool(Tool):
    name = "get_page_text_tool"
    description = "Retrieves the text content from the current web page using the browser instance."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'get_page_text()'
        try:
            route = API_BASE_URL + '/api/browser/content'
            response = requests.get(route)
            data = response.json()
            obs = data.get('content', 'No text found on the page.')
            reward = 1.0 if obs != 'No text found on the page.' else 0.0
        except Exception as e:
            obs = 'Error getting page text due to error ' + str(e)
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class GetNavigableLinksTool(Tool):
    name = "get_navigable_links_tool"
    description = "Retrieves a list of navigable links from the browser."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'get_navigable_links()'
        try:
            route = API_BASE_URL + '/api/browser/links'
            response = requests.get(route)
            data = response.json()
            obs = data.get('links', [])
            reward = 1.0 if obs else 0.0
        except Exception as e:
            obs = 'Error getting navigable links due to error ' + str(e)
            reward = 0.0
        return build_formatted_output(action, obs, reward)

class IsLinkValidTool(Tool):
    name = "is_link_valid_tool"
    description = "Check if a link is valid for navigation."
    inputs = {"url": {"type": "string", "description": "The URL to check."}}
    output_type = "string"

    def forward(self, url: str) -> str:
        action = "IsLinkValidTool: " + url
        try:
            route = API_BASE_URL + '/api/browser/link_valid'
            response = requests.post(
                route,
                json={'url': url}
            )
            data = response.json()
            obs = data.get('valid', False)
            reward = 1.0 if obs else 0.0
        except Exception as e:
            obs = 'Error checking link validity due to error ' + str(e)
            reward = 0.0
        return build_formatted_output(action, str(obs), reward)

class ScreenshotTool(Tool):
    name = "screenshot_tool"
    description = "Take a screenshot of the current page."
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        action = 'screenshot()'
        try:
            route = API_BASE_URL + '/api/browser/screenshot'
            response = requests.get(route)
            data = response.json()
            obs = data.get('filename', '')
            reward = 1.0 if obs else 0.0
        except Exception as e:
            obs = 'Error taking screenshot due to error ' + str(e)
            reward = 0.0
        return build_formatted_output(action, obs, reward)

search_tool = SearchTool()
go_to_url_tool = GoToUrlTool()
get_page_text_tool = GetPageTextTool()
get_navigable_links_tool = GetNavigableLinksTool()
is_link_valid_tool = IsLinkValidTool()
screenshot_tool = ScreenshotTool()

tools = [
    search_tool,
    go_to_url_tool,
    get_page_text_tool,
    get_navigable_links_tool,
    is_link_valid_tool,
    screenshot_tool
]

tools_name = [tool.name for tool in tools]
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
    name: str
    inputs: dict

class Observation(TypedDict):
    data: str

class WorkflowState(TypedDict):
    goal: List[str]
    actions: List[Action]
    observations: List[Observation]
    rewards: List[float]
    success: List[bool]

class SmolAgentFactory:
    def __init__(self, instruct_prompt, tools, model_id="Qwen/Qwen2.5-Coder-32B-Instruct", max_steps=5):
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
        state_actions = state.get("actions", [])
        state_observations = state.get("observations", [])
        state_rewards = state.get("rewards", [])
        state_success = state.get("success", [])
        trajectories = zip(
            state_actions, 
            state_observations, 
            state_success
        )
        trajectories_prompt = "\n".join(
            f"\n\nStep {i}:\nAction: {action['tool']}\nObservation: {observation['data'][:256]}\nSuccess: {success}"
            for i, (action, observation, success) in enumerate(trajectories)
        )
        return f"""
        You previously performed the following actions:
        {trajectories_prompt}
        Your need to follow instructions:
        {self.instruct_prompt}
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
        action: Action = {
            "tool": actions[-1] if actions else "unknown"
            #"inputs": actions[-1]["inputs"] if actions else {},
        }
        obs: Observation = {
            "data": observations
        }
        reward = sum(rewards) if rewards else 0.0
        success = success[-1] if success else False
        return {
            **state,
            "goal": state["goal"],
            "actions": state["actions"] + [action],
            "observations": state["observations"] + [obs],
            "rewards": state["rewards"] + [reward],
            "success": state["success"] + [success],
        }

class WorkflowNodeFactory:
    @staticmethod
    def create_agent_node(agent_factory: SmolAgentFactory) -> Callable[[WorkflowState], dict]:
        def node_function(state: WorkflowState) -> dict:
            return agent_factory.run(state)
        return node_function



# LLM generated logical multi-agent graph
from langgraph.graph import StateGraph, START, END

# WorkflowState, Action, Observation classes, SmolAgentFactory, WorkflowNodeFactory,
# and the tool lists CSV_TOOLS_TOOL, BROWSER_TOOLS_TOOL are assumed to be pre-defined
# and available in the current scope as per system specification.

# ----------------------------------------------------------------------
# Agent instructions
# ----------------------------------------------------------------------
instruct_tokyo = """
You are a web research agent focused exclusively on Tokyo travel planning.

YOUR TASK
- Identify top attractions, cultural experiences, and food spots in Tokyo
- Determine the best seasons/months to visit based on weather and events
- Gather information on transportation options to reach Tokyo from abroad and move within the city
- Provide hotel recommendations in various budgets
- Return concise, well-sourced findings for each category

CONSIDERATIONS
- Use one web page per tool call
- Prioritize authoritative travel sources, official tourism sites, and recent blog posts
"""

instruct_kyoto = """
You are a web research agent focused exclusively on Kyoto travel planning.

YOUR TASK
- Identify top temples, gardens, cultural districts, and experiences in Kyoto
- Determine the best seasons/months to visit based on weather and festivals
- Gather information on transportation options from Tokyo to Kyoto and moving within Kyoto
- Provide hotel or ryokan recommendations in various budgets
- Return concise, well-sourced findings for each category

CONSIDERATIONS
- Use one web page per tool call
- Prioritize authoritative travel sources, official tourism sites, and recent blog posts
"""

instruct_osaka = """
You are a web research agent focused exclusively on Osaka travel planning.

YOUR TASK
- Identify major attractions, food districts, and nightlife spots in Osaka
- Determine the best seasons/months to visit based on weather and local events
- Gather information on transportation options from Kyoto to Osaka and moving within Osaka
- Provide hotel recommendations in various budgets
- Return concise, well-sourced findings for each category

CONSIDERATIONS
- Use one web page per tool call
- Prioritize authoritative travel sources, official tourism sites, and recent blog posts
"""

instruct_aggregator = """
You are a trip-planning aggregation agent.

YOUR TASK
- Combine all previous city research observations into a unified, structured itinerary
- Produce a table (CSV format) with columns: City, Recommended Activities, Best Time to Visit, Transportation Tips, Hotel Options
- Ensure information is clearly separated per city and easy to read
- Summarize any overarching tips for travelling between the cities
- Use CSV creation tools to output the structured data; include a brief narrative summary after the table

CONSIDERATIONS
- Do not perform additional web searches unless data appears missing
- The final output must contain the CSV first, then the narrative summary
"""

# ----------------------------------------------------------------------
# Agent creation
# ----------------------------------------------------------------------
tokyo_agent      = SmolAgentFactory(instruct_tokyo,  BROWSER_TOOLS_TOOL)
kyoto_agent      = SmolAgentFactory(instruct_kyoto,  BROWSER_TOOLS_TOOL)
osaka_agent      = SmolAgentFactory(instruct_osaka,  BROWSER_TOOLS_TOOL)
aggregator_tools = CSV_TOOLS_TOOL + BROWSER_TOOLS_TOOL
aggregator_agent = SmolAgentFactory(instruct_aggregator, aggregator_tools)

# ----------------------------------------------------------------------
# Routing functions with retry logic
# ----------------------------------------------------------------------
def route_tokyo(state):
    try:
        last_success = state.get("success", [])[-1]
        return "kyoto_researcher" if last_success else "tokyo_researcher"
    except Exception:
        return "tokyo_researcher"

def route_kyoto(state):
    try:
        last_success = state.get("success", [])[-1]
        return "osaka_researcher" if last_success else "kyoto_researcher"
    except Exception:
        return "kyoto_researcher"

def route_osaka(state):
    try:
        last_success = state.get("success", [])[-1]
        return "aggregator" if last_success else "osaka_researcher"
    except Exception:
        return "osaka_researcher"

def route_aggregator(state):
    try:
        last_success = state.get("success", [])[-1]
        return END if last_success else "aggregator"
    except Exception:
        return "aggregator"

# ----------------------------------------------------------------------
# Workflow construction
# ----------------------------------------------------------------------
workflow = StateGraph(WorkflowState)

workflow.add_node("tokyo_researcher",  WorkflowNodeFactory.create_agent_node(tokyo_agent))
workflow.add_node("kyoto_researcher",  WorkflowNodeFactory.create_agent_node(kyoto_agent))
workflow.add_node("osaka_researcher",  WorkflowNodeFactory.create_agent_node(osaka_agent))
workflow.add_node("aggregator",        WorkflowNodeFactory.create_agent_node(aggregator_agent))

workflow.add_edge(START, "tokyo_researcher")

workflow.add_conditional_edges(
    "tokyo_researcher",
    route_tokyo,
    {
        "kyoto_researcher": "kyoto_researcher",
        "tokyo_researcher": "tokyo_researcher"
    }
)

workflow.add_conditional_edges(
    "kyoto_researcher",
    route_kyoto,
    {
        "osaka_researcher": "osaka_researcher",
        "kyoto_researcher": "kyoto_researcher"
    }
)

workflow.add_conditional_edges(
    "osaka_researcher",
    route_osaka,
    {
        "aggregator": "aggregator",
        "osaka_researcher": "osaka_researcher"
    }
)

workflow.add_conditional_edges(
    "aggregator",
    route_aggregator,
    {
        END: END,
        "aggregator": "aggregator"
    }
)

app = workflow.compile()

initial_state: WorkflowState = {
    "goal": ["I want to do a trip to japan, can you help me with that? I want to visit Tokyo, Kyoto and Osaka. I want to know the best places to visit, the best time to go, and how to get there. I would like a strucutred result with activities, choice, hotel, and so on..."],
    "actions": [],
    "observations": [],
    "rewards": [],
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

path_json = os.path.join("workflows/e2f4d704eb55492aaf4e4e8bd4284fd5/", "state_result.json")
try:
    with open(path_json, "w") as f:
        json.dump(result_state, f, indent=2)
except Exception as e:
    print(f"Could not save workflow data: {e}")

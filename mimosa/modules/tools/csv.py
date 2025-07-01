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
            result = asyncio.run(_async_csv_tool_call("load_csv_from_path", {"source_path": file_path, "name": name}))
            
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
            result = asyncio.run(_async_csv_tool_call("get_csv_data", {"name": name, "limit": limit, "columns": columns}))
            
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
            result = asyncio.run(_async_csv_tool_call("add_csv_row", {"name": name, "row": row}))
            
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
            result = asyncio.run(_async_csv_tool_call("update_csv_row", {"name": name, "index": index, "row": row}))
            
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
            result = asyncio.run(_async_csv_tool_call("add_csv_column", {"name": name, "column_name": column_name, "default_value": default_value}))
            
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
            result = asyncio.run(_async_csv_tool_call("list_csv_datasets", {}))
            
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
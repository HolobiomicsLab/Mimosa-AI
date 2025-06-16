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
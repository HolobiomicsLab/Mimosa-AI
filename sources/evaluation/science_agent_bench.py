"""
Utility module for handling ScienceAgentBench dataset operations.
Provides methods for loading task data, getting dataset paths, and transferring files.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional


class ScienceAgentBenchLoader:
    """Handler for ScienceAgentBench dataset loading and path management."""
    
    def __init__(self, base_path: str = "datasets/ScienceAgentBench"):
        """
        Initialize ScienceAgentBench loader.
        
        Args:
            base_path: Base directory for ScienceAgentBench dataset
        """
        self.base_path = Path(base_path)
        self.datasets_path = self.base_path / "datasets"
        self.eval_programs_path = self.base_path / "eval_programs"
        self.gold_programs_path = self.base_path / "gold_programs"
        self.scoring_rubrics_path = self.base_path / "scoring_rubrics"
        
        # Cache for CSV data
        self._csv_data: list[dict[str, str]] = None
    
    def load_csv_data(self, csv_path: str = None) -> list[dict[str, str]]:
        """
        Load the ScienceAgentBench CSV file.
        
        Args:
            csv_path: Optional custom path to CSV file
            
        Returns:
            List of dictionaries, one per task row
        """
        if self._csv_data is not None:
            return self._csv_data
            
        if csv_path is None:
            csv_path = self.base_path / "ScienceAgentBench.csv"
        else:
            csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        with open(csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self._csv_data = list(reader)
        
        return self._csv_data
    
    def get_task_by_index(self, index: int, csv_path: str = None) -> dict[str, str]:
        """
        Get task data by row index (0-based).
        
        Args:
            index: Row index (0-101 for 102 tasks)
            csv_path: Optional custom path to CSV file
            
        Returns:
            Dictionary containing task data
        """
        data = self.load_csv_data(csv_path)
        if index < 0 or index >= len(data):
            raise IndexError(f"Task index {index} out of range (0-{len(data)-1})")
        return data[index]
    
    def get_task_by_id(self, instance_id: int, csv_path: str = None) -> dict[str, str]:
        """
        Get task data by instance_id (1-102).
        
        Args:
            instance_id: Task instance ID (1-102)
            csv_path: Optional custom path to CSV file
            
        Returns:
            Dictionary containing task data
        """
        data = self.load_csv_data(csv_path)
        for task in data:
            if int(task.get('instance_id', 0)) == instance_id:
                return task
        raise ValueError(f"Task with instance_id {instance_id} not found")
    
    def get_task_by_name(self, task_name: str, csv_path: str = None) -> dict[str, str]:
        """
        Get task data by gold program name or eval script name.
        
        Args:
            task_name: Name to search for (with or without .py extension)
            csv_path: Optional custom path to CSV file
            
        Returns:
            Dictionary containing task data
        """
        data = self.load_csv_data(csv_path)
        
        # Normalize task name
        task_name = task_name.replace('.py', '')
        
        for task in data:
            gold_name = task.get('gold_program_name', '').replace('.py', '')
            eval_name = task.get('eval_script_name', '').replace('.py', '')
            
            if task_name == gold_name or task_name == eval_name:
                return task
        
        raise ValueError(f"Task with name '{task_name}' not found")
    
    def get_dataset_path(self, task_data: dict[str, str]) -> Path:
        """
        Get the dataset directory path for a task.
        
        Args:
            task_data: Task dictionary from CSV
            
        Returns:
            Path to task's dataset directory
        """
        # Parse dataset folder from dataset_folder_tree
        folder_tree = task_data.get('dataset_folder_tree', '')
        
        # Extract first folder name from tree structure
        # Format is like: |-- folder_name/\n|---- subfolder/
        lines = folder_tree.strip().split('\n')
        if not lines:
            raise ValueError(f"No dataset folder found for task {task_data.get('instance_id')}")
        
        # First line should be like: |-- folder_name/
        first_line = lines[0].strip()
        folder_name = first_line.replace('|--', '').replace('/', '').strip()
        
        dataset_path = self.datasets_path / folder_name
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
        return dataset_path
    
    def get_eval_script_path(self, task_data: dict[str, str]) -> Path:
        """
        Get the evaluation script path for a task.
        
        Args:
            task_data: Task dictionary from CSV
            
        Returns:
            Path to evaluation script
        """
        eval_script_name = task_data.get('eval_script_name', '')
        if not eval_script_name:
            raise ValueError(f"No eval script name for task {task_data.get('instance_id')}")
        
        eval_path = self.eval_programs_path / eval_script_name
        
        if not eval_path.exists():
            raise FileNotFoundError(f"Evaluation script not found: {eval_path}")
        
        return eval_path
    
    def get_gold_program_path(self, task_data: dict[str, str]) -> Path:
        """
        Get the gold program path for a task.
        
        Args:
            task_data: Task dictionary from CSV
            
        Returns:
            Path to gold program
        """
        gold_program_name = task_data.get('gold_program_name', '')
        if not gold_program_name:
            raise ValueError(f"No gold program name for task {task_data.get('instance_id')}")
        
        gold_path = self.gold_programs_path / gold_program_name
        
        if not gold_path.exists():
            raise FileNotFoundError(f"Gold program not found: {gold_path}")
        
        return gold_path
    
    def get_output_path(self, task_data: dict[str, str]) -> str:
        """
        Get the expected output file path for a task.
        
        Args:
            task_data: Task dictionary from CSV
            
        Returns:
            Expected output file path (relative)
        """
        return task_data.get('output_fname', '')
    
    def get_task_summary(self, task_data: dict[str, str]) -> str:
        """
        Get a formatted summary of task information.
        
        Args:
            task_data: Task dictionary from CSV
            
        Returns:
            Formatted task summary string
        """
        return f"""Task ID: {task_data.get('instance_id')}
Domain: {task_data.get('domain')}
Subtasks: {task_data.get('subtask_categories')}
GitHub: {task_data.get('github_name')}
Dataset: {task_data.get('dataset_folder_tree', '').split('/')[0].replace('|--', '').strip()}
Output: {task_data.get('output_fname')}
Eval Script: {task_data.get('eval_script_name')}
Gold Program: {task_data.get('gold_program_name')}"""


# Convenience functions for quick access
def get_task_by_index(index: int, base_path: str = "datasets/ScienceAgentBench") -> dict[str, str]:
    """Quick access to get task by index."""
    loader = ScienceAgentBenchLoader(base_path)
    return loader.get_task_by_index(index)


def get_dataset_path_for_index(index: int, base_path: str = "datasets/ScienceAgentBench") -> Path:
    """Quick access to get dataset path by index."""
    loader = ScienceAgentBenchLoader(base_path)
    task = loader.get_task_by_index(index)
    return loader.get_dataset_path(task)


def get_task_by_id(instance_id: int, base_path: str = "datasets/ScienceAgentBench") -> dict[str, str]:
    """Quick access to get task by instance ID."""
    loader = ScienceAgentBenchLoader(base_path)
    return loader.get_task_by_id(instance_id)

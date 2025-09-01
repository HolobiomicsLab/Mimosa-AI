import json
import os
import pathlib
from typing import Any


def read_memory_file(uuid:str, file_name: str) -> dict[str, Any]:
    """
    Read a memory file and return its contents as a dictionary.
    
    Args:
        file_path: Path to the memory file
        
    Returns:
        The contents of the memory file as a dictionary
    """
    file_path = pathlib.Path('sources/memory') / uuid / file_name
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Memory file {file_path} does not exist.")
    
    with open(file_path) as f:
        return json.load(f)

if __name__ == "__main__":
    uuid = "20250820_121517_31385670"
    json_file = "scenario_judge_1.json"
    memory_data = read_memory_file(uuid,json_file)
    
    if memory_data:
        print(f"Found {len(memory_data)} memory files for UUID {uuid}")
        
        # Print the content of each memory file
        print(memory_data["message"][1]["content"])
    else:
        print(f"No memory files found for UUID {uuid}")
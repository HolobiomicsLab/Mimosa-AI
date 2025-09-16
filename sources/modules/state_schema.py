"""
State schema for Mimosa workflow assembly.
The code is not imported directly. It is loaded by the workflow factory and used in a Python environment as part of the crafted workflow.
"""

from typing import TypedDict, List, Union, Any
from pydantic import BaseModel
import json

class Answer(BaseModel):
    status: str
    message: str
    retry_advice: str = ""
    
    @classmethod
    def validate(cls, data: Union[str, dict, Any]) -> 'Answer':
        """
        Robust parsing that handles both JSON strings and dictionaries
        with comprehensive error handling and fallback strategies.
        """
        try:
            if isinstance(data, str):
                try:
                    return cls.model_validate_json(data)
                except (json.JSONDecodeError, ValueError):
                    raise ValueError(f"String input is not valid JSON: {data[:100]}...")
            elif isinstance(data, dict):
                return cls.model_validate(data)
            else:
                raise TypeError(f"Unsupported data type: {type(data)}. Expected str or dict.")
        except Exception as e:
            raise ValueError(f"Failed to parse Answer from {type(data)}: {str(e)}") from e

class Action(TypedDict):
    tool: str

class Observation(TypedDict):
    data: str

class WorkflowState(TypedDict):
    workflow_uuid: str
    model_id: str
    goal: str
    step_name: List[str]
    task_prompt: List[str]
    actions: List[Action]
    observations: List[Observation]
    answers: List[str]
    success: List[bool]

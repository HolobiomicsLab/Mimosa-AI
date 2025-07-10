"""
State schema for Mimosa workflow assembly.
The code is not imported directly. It is loaded by the workflow factory and used in a Python environment as part of the crafted workflow.
"""

from typing import TypedDict, List

class Action(TypedDict):
    tool: str

class Observation(TypedDict):
    data: str

class WorkflowState(TypedDict):
    workflow_uuid: str
    step_name: List[str]
    step_uuid: List[str]
    task_prompt: List[str]
    actions: List[Action]
    observations: List[Observation]
    answers: List[str]
    success: List[bool]
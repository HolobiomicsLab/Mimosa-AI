
from typing import TypedDict, List, Tuple, Any, Dict, Union, Optional, Callable

class Action(TypedDict):
    tool: str
    inputs: dict

class Observation(TypedDict):
    data: Any

class WorkflowState(TypedDict):
    goal: str
    actions: List[Action]
    observations: List[Observation]
    rewards: List[float]
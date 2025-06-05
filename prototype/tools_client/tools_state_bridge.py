from typing import List, Any, TypedDict, Dict, Optional, Tuple
import re

def print_action(args: List[Any]) -> None:
    """
    Print the action being performed.
    Args:
        name (str): The name of the action.
        args (List[Any]): The arguments for the action.
    """
    print(f"action:{args}")

def print_observation(observation: str) -> None:
    """
    Print the observation made.
    Args:
        observation (str): The observation to print.
    """
    print(f"observation:{observation}")

def print_reward(reward: float) -> None:
    """
    Print the reward received.
    Args:
        reward (float): The reward value to print.
    """
    print(f"reward:{reward}")
